"""Ollama LLM Integration — qwen3:14b (control brain/judge) + qwen3:4b (reflex model).
Per paper Section 8.1: models hosted on localhost:11434.
Used by: MACI safety gate debate, fabrication engine reasoning, neural process kernel code:eval.

Also includes HFInferenceClient for HuggingFace Spaces deployment (no local Ollama needed)
and HybridLLMClient that auto-selects Ollama or HF Inference API."""
import os
import requests
import json
import time
from typing import Dict, Any, Optional, Callable, Union


class OllamaClient:
    """Client for local Ollama instance hosting qwen3 models.
    Per paper: qwen3:14b = control brain (judge), qwen3:4b = reflex model (proponent/opponent)."""

    def __init__(self, base_url: str = "http://localhost:11434",
                 judge_model: str = "qwen3:14b",
                 reflex_model: str = "qwen3:4b"):
        self.base_url = base_url
        self.judge_model = judge_model
        self.reflex_model = reflex_model
        self._available: Optional[bool] = None
        self._last_check: float = 0.0

    def is_available(self) -> bool:
        """Check if Ollama is running. Cache result for 30 seconds."""
        now = time.time()
        if self._available is not None and (now - self._last_check) < 30.0:
            return self._available

        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            self._available = (r.status_code == 200)
        except Exception:
            self._available = False
        self._last_check = now
        return self._available

    def list_models(self) -> list:
        """List available models on the Ollama instance."""
        if not self.is_available():
            return []
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass
        return []

    def generate(self, model: str, prompt: str, temperature: float = 0.7,
                 max_tokens: int = 512, system: Optional[str] = None) -> str:
        """Generate text from a model via Ollama /api/generate."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system

        try:
            r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
            if r.status_code == 200:
                return r.json().get("response", "")
        except Exception as e:
            return f"[Ollama error: {e}]"
        return ""

    def chat(self, model: str, messages: list, temperature: float = 0.7,
             max_tokens: int = 512) -> str:
        """Chat with a model via Ollama /api/chat."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=60)
            if r.status_code == 200:
                return r.json().get("message", {}).get("content", "")
        except Exception as e:
            return f"[Ollama error: {e}]"
        return ""

    def judge(self, prompt: str, system: Optional[str] = None, temperature: float = 0.3) -> str:
        """Use the control brain (qwen3:14b) for judgment/evaluation."""
        return self.generate(self.judge_model, prompt, temperature=temperature, system=system)

    def reflex(self, prompt: str, system: Optional[str] = None, temperature: float = 0.7) -> str:
        """Use a reflex model (qwen3:4b) for fast reasoning."""
        return self.generate(self.reflex_model, prompt, temperature=temperature, system=system)

    def embed(self, model: str, text: str) -> list:
        """Get embeddings from a model."""
        try:
            r = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=30,
            )
            if r.status_code == 200:
                return r.json().get("embedding", [])
        except Exception:
            pass
        return []

    def status(self) -> Dict[str, Any]:
        """Get status of the Ollama instance."""
        return {
            "available": self.is_available(),
            "base_url": self.base_url,
            "judge_model": self.judge_model,
            "reflex_model": self.reflex_model,
            "models": self.list_models() if self.is_available() else [],
        }


class OllamaDebateProtocol:
    """MACI debate protocol using Ollama models.
    Per paper Section 5.3:
    - Proponent (qwen3:4b): argues action is safe
    - Opponent (qwen3:4b): argues action is dangerous
    - Judge (qwen3:14b): evaluates both sides, issues verdict"""

    def __init__(self, client: OllamaClient):
        self.client = client

    def debate(self, action_description: str, context: str = "") -> Dict[str, Any]:
        """Run a full MACI debate. Returns proponent arg, opponent arg, judge verdict."""
        if not self.client.is_available():
            return {
                "available": False,
                "proponent": "",
                "opponent": "",
                "judge": "",
                "verdict": "HEURISTIC",
                "confidence": 0.5,
            }

        # Proponent argues safety
        proponent_prompt = (
            f"You are a proponent arguing that the following action is SAFE and should proceed.\n"
            f"Action: {action_description}\n"
            f"Context: {context}\n"
            f"Argue concisely why this action is safe. Mention specific safety checks if any."
        )
        proponent_arg = self.client.reflex(proponent_prompt)

        # Opponent argues danger
        opponent_prompt = (
            f"You are an opponent arguing that the following action is DANGEROUS and should be blocked or modified.\n"
            f"Action: {action_description}\n"
            f"Context: {context}\n"
            f"Argue concisely why this action is dangerous. Mention specific risks."
        )
        opponent_arg = self.client.reflex(opponent_prompt)

        # Judge evaluates
        judge_prompt = (
            f"You are a judge evaluating whether an action should proceed.\n"
            f"Action: {action_description}\n\n"
            f"Proponent argues: {proponent_arg}\n\n"
            f"Opponent argues: {opponent_arg}\n\n"
            f"Evaluate both arguments. Issue one of: APPROVE, MODIFY, REJECT, or REQUIRE_HUMAN.\n"
            f"State your confidence as a number between 0 and 1.\n"
            f"Format: VERDICT: <verdict>\\nCONFIDENCE: <number>\\nREASONING: <text>"
        )
        judge_output = self.client.judge(judge_prompt)

        # Parse verdict
        verdict = "REQUIRE_HUMAN"
        confidence = 0.5
        reasoning = judge_output

        for line in judge_output.split("\n"):
            line = line.strip().upper()
            if line.startswith("VERDICT:"):
                v = line.split(":", 1)[1].strip()
                if v in ("APPROVE", "MODIFY", "REJECT", "REQUIRE_HUMAN"):
                    verdict = v
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

        return {
            "available": True,
            "proponent": proponent_arg[:300],
            "opponent": opponent_arg[:300],
            "judge": judge_output[:500],
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": reasoning[:300],
        }


class OllamaReasoningEngine:
    """Uses Ollama models for reasoning tasks in the fabrication pipeline.
    Per paper: code:eval nodes go through the neural process kernel which can
    use the reflex model for fast code generation and evaluation."""

    def __init__(self, client: OllamaClient):
        self.client = client

    def generate_code(self, description: str, language: str = "python") -> str:
        """Generate code for a fabrication node using the reflex model."""
        if not self.client.is_available():
            return f"# [No Ollama] Would generate {language} code for: {description}"

        prompt = (
            f"Generate {language} code for the following task. Output only code, no explanation.\n"
            f"Task: {description}"
        )
        return self.client.reflex(prompt, temperature=0.2)

    def evaluate_intent(self, intent: str, domains: list) -> Dict[str, Any]:
        """Use the control brain to evaluate intent classification."""
        if not self.client.is_available():
            return {"available": False, "analysis": ""}

        domain_list = "\n".join(f"- {d}" for d in domains)
        prompt = (
            f"Analyze this user intent and determine which capability domains are most relevant.\n"
            f"Intent: {intent}\n"
            f"Available domains:\n{domain_list}\n\n"
            f"List the top 3 most relevant domains with a confidence score (0-1) for each."
        )
        response = self.client.judge(prompt, temperature=0.3)
        return {
            "available": True,
            "analysis": response[:500],
        }

    def summarize_execution(self, results: list) -> str:
        """Use the control brain to summarize execution results."""
        if not self.client.is_available():
            return "[No Ollama] Would summarize execution results"

        results_str = "\n".join(
            f"- Node {r.get('node_id', '?')}: {r.get('status', '?')} — {r.get('detail', '')[:100]}"
            for r in results
        )
        prompt = (
            f"Summarize the following execution results concisely.\n"
            f"Identify any failures and suggest next steps.\n\n{results_str}"
        )
        return self.client.judge(prompt, temperature=0.3)


class HFInferenceClient:
    """HuggingFace Inference API client — drop-in replacement for OllamaClient.
    Uses HF serverless inference API when Ollama is not available (e.g., HF Spaces).
    Falls back to free models: Qwen/Qwen2.5-7B-Instruct (judge), Qwen/Qwen2.5-1.5B-Instruct (reflex)."""

    def __init__(self,
                 judge_model: str = "Qwen/Qwen2.5-7B-Instruct",
                 reflex_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 api_key: Optional[str] = None):
        self.judge_model = judge_model
        self.reflex_model = reflex_model
        self.api_key = api_key or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY", "")
        self.base_url = "https://api-inference.huggingface.co/models"
        self._available: Optional[bool] = None
        self._last_check: float = 0.0

    def is_available(self) -> bool:
        now = time.time()
        if self._available is not None and (now - self._last_check) < 30.0:
            return self._available
        self._available = bool(self.api_key)
        self._last_check = now
        return self._available

    def list_models(self) -> list:
        return [self.judge_model, self.reflex_model] if self.is_available() else []

    def generate(self, model: str, prompt: str, temperature: float = 0.7,
                 max_tokens: int = 512, system: Optional[str] = None) -> str:
        if not self.is_available():
            return ""
        full_prompt = f"{system}\n{prompt}" if system else prompt
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "temperature": max(0.01, temperature),
                "max_new_tokens": max_tokens,
                "return_full_text": False,
            },
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            r = requests.post(f"{self.base_url}/{model}", json=payload, headers=headers, timeout=60)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    return data[0].get("generated_text", "")
                return str(data)
        except Exception:
            pass
        return ""

    def chat(self, model: str, messages: list, temperature: float = 0.7,
             max_tokens: int = 512) -> str:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return self.generate(model, prompt, temperature, max_tokens)

    def judge(self, prompt: str, system: Optional[str] = None) -> str:
        return self.generate(self.judge_model, prompt, temperature=0.3, system=system)

    def reflex(self, prompt: str, system: Optional[str] = None) -> str:
        return self.generate(self.reflex_model, prompt, temperature=0.7, system=system)

    def embed(self, model: str, text: str) -> list:
        return []

    def status(self) -> Dict[str, Any]:
        return {
            "available": self.is_available(),
            "base_url": self.base_url,
            "judge_model": self.judge_model,
            "reflex_model": self.reflex_model,
            "models": self.list_models(),
            "backend": "huggingface",
        }


class HybridLLMClient:
    """Auto-selects between Ollama (local) and HuggingFace Inference API (remote).
    Tries Ollama first, falls back to HF Inference if Ollama is not running.
    This enables the same code to work on Colab (Ollama) and HF Spaces (HF API)."""

    def __init__(self,
                 ollama_url: str = "http://localhost:11434",
                 ollama_judge: str = "qwen3:14b",
                 ollama_reflex: str = "qwen3:4b",
                 hf_judge: str = "Qwen/Qwen2.5-7B-Instruct",
                 hf_reflex: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 hf_api_key: Optional[str] = None):
        self.ollama = OllamaClient(ollama_url, ollama_judge, ollama_reflex)
        self.hf = HFInferenceClient(hf_judge, hf_reflex, hf_api_key)
        self._active: Optional[str] = None

    def _get_client(self):
        if self.ollama.is_available():
            self._active = "ollama"
            return self.ollama
        if self.hf.is_available():
            self._active = "huggingface"
            return self.hf
        self._active = "none"
        return None

    def is_available(self) -> bool:
        return self._get_client() is not None

    def list_models(self) -> list:
        c = self._get_client()
        return c.list_models() if c else []

    def generate(self, model: str, prompt: str, temperature: float = 0.7,
                 max_tokens: int = 512, system: Optional[str] = None) -> str:
        c = self._get_client()
        if c:
            return c.generate(model, prompt, temperature, max_tokens, system)
        return ""

    def chat(self, model: str, messages: list, temperature: float = 0.7,
             max_tokens: int = 512) -> str:
        c = self._get_client()
        if c:
            return c.chat(model, messages, temperature, max_tokens)
        return ""

    def judge(self, prompt: str, system: Optional[str] = None, temperature: float = 0.3) -> str:
        c = self._get_client()
        if c:
            return c.generate(c.judge_model, prompt, temperature=temperature, system=system)
        return ""

    def reflex(self, prompt: str, system: Optional[str] = None, temperature: float = 0.7) -> str:
        c = self._get_client()
        if c:
            return c.generate(c.reflex_model, prompt, temperature=temperature, system=system)
        return ""

    def embed(self, model: str, text: str) -> list:
        c = self._get_client()
        if c:
            return c.embed(model, text)
        return []

    def status(self) -> Dict[str, Any]:
        c = self._get_client()
        if c:
            s = c.status()
            s["active_backend"] = self._active
            s["ollama_available"] = self.ollama.is_available()
            s["hf_available"] = self.hf.is_available()
            return s
        return {
            "available": False,
            "active_backend": "none",
            "ollama_available": self.ollama.is_available(),
            "hf_available": self.hf.is_available(),
        }
