"""Calibration-Aware Safety Gate — AUQ + MACI debate per paper Section 5.
Uses Ollama qwen3:4b (reflex) + qwen3:14b (judge) for live debate when available."""
import re
import requests as http_requests
from typing import Dict, Any, Optional, Callable
from .models import AUQResult, SafetyVerdict, DAGNode, ConsentLevel, ActionVerb
from .ollama_integration import OllamaClient, OllamaDebateProtocol


# Risk weight coefficients per paper Section 5.2
ALPHA_1 = 0.3  # destructive verb
ALPHA_2 = 0.2  # high-impact type
ALPHA_3 = 0.2  # elevated consent
ALPHA_4 = 0.3  # irreversible

DESTRUCTIVE_VERBS = {"rm", "del", "delete", "format", "drop", "truncate", "kill", "wipe", "purge"}
HIGH_IMPACT_TYPES = {"shell", "network"}
IRREVERSIBLE_VERBS = {"delete", "drop", "truncate", "wipe", "purge", "format"}


class AUQGate:
    """Agentic Uncertainty Quantification."""

    def __init__(self, calibration_threshold: float = 0.15):
        self.calibration_threshold = calibration_threshold

    def assess(self, node: DAGNode, c_stated: float = 0.9) -> AUQResult:
        """Compute calibration gap and risk score for an action."""
        # Determine if destructive
        verb_str = node.verb.value.lower()
        params_str = str(node.params).lower()

        is_destructive = verb_str in DESTRUCTIVE_VERBS or any(d in params_str for d in DESTRUCTIVE_VERBS)
        is_high_impact = node.action_type.value in HIGH_IMPACT_TYPES
        is_elevated = node.consent_level in (ConsentLevel.ELEVATED, ConsentLevel.NEVER)
        is_irreversible = verb_str in IRREVERSIBLE_VERBS or any(d in params_str for d in IRREVERSIBLE_VERBS)

        # Risk score
        r_risk = (
            ALPHA_1 * float(is_destructive) +
            ALPHA_2 * float(is_high_impact) +
            ALPHA_3 * float(is_elevated) +
            ALPHA_4 * float(is_irreversible)
        )

        # Worst-case probability (heuristic: higher risk = higher p_worst)
        p_worst = min(0.95, max(0.05, r_risk * 0.5 + (0.1 if is_destructive else 0.0)))

        # Calibration gap
        delta_cal = abs(c_stated - (1.0 - p_worst))

        # Trigger debate?
        triggers = (
            delta_cal >= self.calibration_threshold
            or r_risk >= 0.5
            or is_destructive
        )

        return AUQResult(
            c_stated=c_stated,
            p_worst=round(p_worst, 4),
            r_risk=round(r_risk, 4),
            delta_cal=round(delta_cal, 4),
            triggers_debate=triggers,
        )


class MACIDebate:
    """Multi-Agent Confidence Interval debate.
    Per paper Section 5.3: proponent (qwen3:4b) vs opponent (qwen3:4b), judge (qwen3:14b).
    Falls back to deterministic heuristics when Ollama is not available."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        reflex_model: str = "qwen3:4b",
        judge_model: str = "qwen3:14b",
        llm_callback: Optional[Callable] = None,
        ollama_client: Optional[OllamaClient] = None,
    ):
        self.ollama_url = ollama_url
        self.reflex_model = reflex_model
        self.judge_model = judge_model
        self.llm_callback = llm_callback

        # Use provided OllamaClient or create one
        self.ollama = ollama_client or OllamaClient(
            base_url=ollama_url, judge_model=judge_model, reflex_model=reflex_model
        )
        self.debate_protocol = OllamaDebateProtocol(self.ollama)

    def debate(self, node: DAGNode, auq: AUQResult) -> SafetyVerdict:
        """Run adversarial debate between proponent, opponent, and judge."""
        action_desc = f"{node.action_type.value}:{node.verb.value} — {node.params.get('description', '')}"
        context = f"consent_level={node.consent_level.value}, params={node.params}"

        # Try live Ollama debate first
        if self.llm_callback:
            proponent_arg = self.llm_callback(self.reflex_model, f"Argue that this action is safe: {action_desc}")
            opponent_arg = self.llm_callback(self.reflex_model, f"Argue that this action is dangerous: {action_desc}")
            judge_input = f"Proponent: {proponent_arg}\nOpponent: {opponent_arg}\nEvaluate and decide: APPROVE, MODIFY, REJECT, or REQUIRE_HUMAN."
            judge_output = self.llm_callback(self.judge_model, judge_input)
            c_judge = self._extract_confidence(judge_output)
            debate_used = "llm_callback"
        elif self.ollama.is_available():
            # Use OllamaDebateProtocol
            debate_result = self.debate_protocol.debate(action_desc, context)
            c_judge = debate_result["confidence"]
            debate_used = "ollama_live"
        else:
            # Deterministic heuristic fallback (per paper Section 9.4 limitation 2)
            c_judge = self._heuristic_judge(node, auq)
            debate_used = "heuristic"

        # Residual risk after debate
        r_residual = max(0.0, auq.r_risk - 0.3) if c_judge > 0.5 else auq.r_risk

        # Determine verdict per paper Section 5.3 rules
        is_destructive = node.verb.value in DESTRUCTIVE_VERBS or any(
            d in str(node.params).lower() for d in DESTRUCTIVE_VERBS
        )

        modifications = []
        if c_judge > 0.8 and r_residual < 0.2:
            verdict = "APPROVE"
        elif 0.5 < c_judge <= 0.8:
            verdict = "MODIFY"
            modifications = self._generate_modifications(node)
        elif c_judge <= 0.5 and not is_destructive:
            verdict = "REJECT"
        elif is_destructive and c_judge <= 0.8:
            verdict = "REQUIRE_HUMAN"
        else:
            verdict = "REQUIRE_HUMAN"

        return SafetyVerdict(
            verdict=verdict,
            c_judge=round(c_judge, 4),
            r_residual=round(r_residual, 4),
            modifications=modifications,
            reasoning=f"[{debate_used}] AUQ delta_cal={auq.delta_cal}, r_risk={auq.r_risk}, c_judge={c_judge:.3f}",
        )

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence value from judge output."""
        import re
        matches = re.findall(r'(\d+\.?\d*)', text)
        for m in matches:
            val = float(m)
            if 0.0 <= val <= 1.0:
                return val
        return 0.5  # default

    def _heuristic_judge(self, node: DAGNode, auq: AUQResult) -> float:
        """Deterministic heuristic judge when no LLM available."""
        verb_str = node.verb.value.lower()
        is_destructive = verb_str in DESTRUCTIVE_VERBS or any(
            d in str(node.params).lower() for d in DESTRUCTIVE_VERBS
        )
        # Higher delta_cal = lower judge confidence
        base = 1.0 - auq.delta_cal
        if is_destructive:
            base *= 0.5
        if auq.r_risk >= 0.5:
            base *= 0.7
        return max(0.1, min(0.95, base))

    def _generate_modifications(self, node: DAGNode) -> list:
        """Generate pre-checks for MODIFY verdict."""
        mods = []
        if node.action_type.value == "file" and node.verb.value in ("write", "delete"):
            mods.append({"type": "pre_check", "action": "path_verification"})
            mods.append({"type": "pre_check", "action": "backup_creation"})
        if node.action_type.value == "shell":
            mods.append({"type": "pre_check", "action": "command_allowlist_check"})
        if node.action_type.value == "network":
            mods.append({"type": "pre_check", "action": "rate_limit_check"})
        return mods


class SafetyGate:
    """Combined AUQ + MACI safety gate.
    Per paper Section 5: AUQ checks calibration, MACI runs adversarial debate."""

    def __init__(
        self,
        calibration_threshold: float = 0.15,
        ollama_url: str = "http://localhost:11434",
        llm_callback: Optional[Callable] = None,
        ollama_client: Optional[OllamaClient] = None,
    ):
        self.auq = AUQGate(calibration_threshold)
        self.maci = MACIDebate(
            ollama_url=ollama_url, llm_callback=llm_callback,
            ollama_client=ollama_client,
        )
        self.ollama = self.maci.ollama

    def evaluate(self, node: DAGNode, c_stated: float = 0.9) -> dict:
        """Full safety evaluation: AUQ → debate (if triggered) → verdict."""
        auq_result = self.auq.assess(node, c_stated)

        if auq_result.triggers_debate:
            verdict = self.maci.debate(node, auq_result)
        else:
            verdict = SafetyVerdict(
                verdict="APPROVE",
                c_judge=c_stated,
                r_residual=auq_result.r_risk,
                reasoning="No debate triggered — calibration within threshold.",
            )

        return {
            "auq": {
                "c_stated": auq_result.c_stated,
                "p_worst": auq_result.p_worst,
                "r_risk": auq_result.r_risk,
                "delta_cal": auq_result.delta_cal,
                "triggers_debate": auq_result.triggers_debate,
            },
            "verdict": verdict.verdict,
            "c_judge": verdict.c_judge,
            "r_residual": verdict.r_residual,
            "modifications": verdict.modifications,
            "reasoning": verdict.reasoning,
        }
