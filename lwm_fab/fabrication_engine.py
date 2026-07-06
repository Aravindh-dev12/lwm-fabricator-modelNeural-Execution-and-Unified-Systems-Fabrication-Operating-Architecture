"""Capability Fabrication Engine — classify() + fabricate() per paper Section 3."""
import re
from typing import List, Tuple, Dict, Any
from .models import (
    CapabilityDomain, DAGNode, ExecutionDAG, ConsentLevel,
    ActionType, ActionVerb, FabricationResult,
)
from .domain_registry import get_domain_registry, DOMAINS


class FabricationEngine:
    def __init__(self, episodic_memory: List[Dict[str, Any]] = None, beta: float = 0.5, top_k: int = 4):
        self.registry = get_domain_registry()
        self.episodic_memory = episodic_memory or []
        self.beta = beta
        self.top_k = top_k

    def classify(self, query: str) -> List[Tuple[str, float]]:
        """Score all 21 domains against the intent. Returns sorted list of (domain_name, score)."""
        q_lower = query.lower()
        q_tokens = set(re.findall(r'\w+', q_lower))
        scores: List[Tuple[str, float]] = []

        for domain in DOMAINS:
            # Pattern overlap: exact subset match
            pattern_score = 0.0
            for p in domain.intent_patterns:
                if p.lower() in q_lower:
                    pattern_score += 1.0

            # Soft token-overlap fallback (per Section 3.3)
            if pattern_score == 0:
                for p in domain.intent_patterns:
                    p_tokens = set(re.findall(r'\w+', p.lower()))
                    overlap = sum(1 for w in p_tokens if w in q_tokens and len(w) > 3)
                    pattern_score += 0.25 * overlap

            # Episodic memory boost (Section 3.3)
            memory_boost = 0.0
            for entry in self.episodic_memory:
                if domain.name in entry.get("domains", []):
                    memory_boost += self.beta * entry.get("success_weight", 1.0)

            total = pattern_score + memory_boost
            if total > 0:
                scores.append((domain.name, round(total, 3)))

        scores.sort(key=lambda x: -x[1])
        return scores[: self.top_k]

    def fabricate(self, query: str, matched: List[Tuple[str, float]]) -> ExecutionDAG:
        """Fuse matched domain grammars into a single consent-gated DAG."""
        nodes: List[DAGNode] = []
        edges: List[tuple] = []
        prev_id = None
        step = 0

        for domain_name, score in matched:
            domain = self.registry[domain_name]
            for grammar_step in domain.execution_grammar:
                node = DAGNode(
                    node_id=f"n{step+1}",
                    action_type=grammar_step["action_type"],
                    verb=grammar_step["verb"],
                    params={"description": grammar_step["description"], "domain": domain_name},
                    consent_level=grammar_step["consent"],
                    depends_on=[prev_id] if prev_id else [],
                )
                nodes.append(node)
                if prev_id:
                    edges.append((prev_id, node.node_id))
                prev_id = node.node_id
                step += 1

        return ExecutionDAG(
            run_id="",
            intent=query,
            nodes=nodes,
            edges=edges,
        )

    def run(self, query: str) -> FabricationResult:
        """Full classify + fabricate pipeline."""
        matched = self.classify(query)
        dag = self.fabricate(query, matched)
        # Generate a 64-dimensional state vector from the intent (simplified encoding)
        state_vector = self._encode_state(query, matched)
        return FabricationResult(
            intent=query,
            matched_domains=matched,
            dag=dag,
            state_vector=state_vector,
        )

    def _encode_state(self, query: str, matched: List[Tuple[str, float]]) -> List[float]:
        """Encode current OS state into a 64-dimensional normalized vector."""
        import hashlib
        # Deterministic hash-based encoding of intent into 64D
        h = hashlib.sha256(query.encode()).digest()
        vec = [(b - 128) / 128.0 for b in h[:32]]  # 32 dims from hash
        # Pad with domain scores and zeros to reach 64
        domain_scores = [s for _, s in matched] + [0.0] * (4 - len(matched))
        vec.extend(domain_scores)  # +4 = 36
        vec.extend([0.0] * (64 - len(vec)))  # pad to 64
        return vec[:64]
