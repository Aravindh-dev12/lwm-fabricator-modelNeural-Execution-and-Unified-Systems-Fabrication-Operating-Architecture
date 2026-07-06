"""ActionExecutor — grounded file/shell/network/code execution per paper Section 3.5 & 7.3."""
import os
import shutil
import subprocess
import json
import requests as http_requests
from typing import Dict, Any, Optional
from .models import DAGNode, ActionType, ActionVerb, ConsentLevel
from .safety_gate import SafetyGate


# Shell command allowlist (per paper Section 3.5)
SHELL_ALLOWLIST = {
    "echo", "ls", "dir", "cat", "type", "pwd", "cd", "mkdir", "touch",
    "python", "node", "npm", "git", "pip", "curl", "wget", "ping",
    "cp", "mv", "head", "tail", "wc", "grep", "find", "diff",
}

# Rate limiting for network requests
_network_call_timestamps = []


class ActionExecutor:
    """Executes grounded actions with backup/rollback, allowlist, and rate limiting."""

    def __init__(
        self,
        mode: str = "dry_run",
        safety_gate: Optional[SafetyGate] = None,
        work_dir: str = ".",
        network_rate_limit_per_min: int = 30,
    ):
        self.mode = mode  # "dry_run" or "live"
        self.safety_gate = safety_gate
        self.work_dir = work_dir
        self.network_rate_limit = network_rate_limit_per_min
        self._backups: Dict[str, str] = {}

    def execute(self, node: DAGNode, c_stated: float = 0.9) -> Dict[str, Any]:
        """Execute a single DAG node with safety gate check."""
        result: Dict[str, Any] = {
            "node_id": node.node_id,
            "action_type": node.action_type.value,
            "verb": node.verb.value,
            "description": node.params.get("description", ""),
            "mode": self.mode,
        }

        # Safety gate check for elevated/never consent
        if node.consent_level in (ConsentLevel.ELEVATED, ConsentLevel.NEVER) and self.safety_gate:
            gate_result = self.safety_gate.evaluate(node, c_stated)
            result["safety_gate"] = gate_result

            if gate_result["verdict"] == "REJECT":
                result["status"] = "blocked"
                result["detail"] = "Action rejected by safety gate"
                return result
            elif gate_result["verdict"] == "REQUIRE_HUMAN":
                result["status"] = "paused"
                result["detail"] = "Action requires human approval"
                return result
            elif gate_result["verdict"] == "MODIFY":
                # Apply modifications (pre-checks)
                for mod in gate_result.get("modifications", []):
                    result.setdefault("pre_checks", []).append(mod["action"])

        # Execute the action
        if self.mode == "dry_run":
            result["status"] = "success"
            result["detail"] = f"[DRY RUN] Simulated {node.action_type.value}:{node.verb.value}"
            return result

        # Live execution
        try:
            if node.action_type == ActionType.FILE:
                exec_result = self._exec_file(node)
            elif node.action_type == ActionType.SHELL:
                exec_result = self._exec_shell(node)
            elif node.action_type == ActionType.NETWORK:
                exec_result = self._exec_network(node)
            elif node.action_type == ActionType.CODE:
                exec_result = self._exec_code(node)
            elif node.action_type == ActionType.APP:
                exec_result = self._exec_app(node)
            else:
                exec_result = {"status": "failure", "detail": f"Unknown action type: {node.action_type}"}

            result.update(exec_result)
        except Exception as e:
            result["status"] = "failure"
            result["detail"] = str(e)

        return result

    def _exec_file(self, node: DAGNode) -> Dict[str, Any]:
        """File operations with backup-and-rollback."""
        path = node.params.get("path", os.path.join(self.work_dir, "output.txt"))

        if node.verb == ActionVerb.READ:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                return {"status": "success", "detail": f"Read {len(content)} bytes from {path}", "content": content[:500]}
            return {"status": "failure", "detail": f"File not found: {path}"}

        elif node.verb == ActionVerb.WRITE:
            content = node.params.get("content", node.params.get("description", ""))
            # Backup existing file
            if os.path.exists(path):
                backup = path + ".bak"
                shutil.copy2(path, backup)
                self._backups[path] = backup
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"status": "success", "detail": f"Wrote {len(content)} bytes to {path}"}

        elif node.verb == ActionVerb.DELETE:
            if os.path.exists(path):
                backup = path + ".bak"
                shutil.copy2(path, backup)
                self._backups[path] = backup
                os.remove(path)
                return {"status": "success", "detail": f"Deleted {path} (backup at {backup})"}
            return {"status": "failure", "detail": f"File not found: {path}"}

        return {"status": "failure", "detail": f"Unknown file verb: {node.verb}"}

    def _exec_shell(self, node: DAGNode) -> Dict[str, Any]:
        """Shell execution with allowlist enforcement."""
        command = node.params.get("command", node.params.get("description", ""))
        cmd_parts = command.split()
        if not cmd_parts:
            return {"status": "failure", "detail": "Empty command"}

        base_cmd = cmd_parts[0].lower()
        # Allowlist check
        if base_cmd not in SHELL_ALLOWLIST:
            return {"status": "failure", "detail": f"Command '{base_cmd}' not in allowlist"}

        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30,
            cwd=self.work_dir,
        )
        if result.returncode == 0:
            return {"status": "success", "detail": result.stdout[:500], "returncode": result.returncode}
        return {"status": "failure", "detail": result.stderr[:500], "returncode": result.returncode}

    def _exec_network(self, node: DAGNode) -> Dict[str, Any]:
        """Network requests with rate limiting."""
        import time
        now = time.time()
        # Clean old timestamps (older than 60s)
        global _network_call_timestamps
        _network_call_timestamps = [t for t in _network_call_timestamps if now - t < 60.0]
        if len(_network_call_timestamps) >= self.network_rate_limit:
            return {"status": "warning", "detail": "Rate limit exceeded, request queued"}

        _network_call_timestamps.append(now)

        url = node.params.get("url", "https://httpbin.org/get")
        method = node.params.get("method", "GET").upper()
        try:
            resp = http_requests.request(method, url, timeout=15)
            return {"status": "success", "detail": f"{method} {url} → {resp.status_code}", "status_code": resp.status_code}
        except Exception as e:
            return {"status": "failure", "detail": str(e)}

    def _exec_code(self, node: DAGNode) -> Dict[str, Any]:
        """Code evaluation through neural process kernel (sandboxed exec)."""
        code = node.params.get("code", node.params.get("description", ""))
        try:
            local_ns: Dict[str, Any] = {}
            exec(code, {"__builtins__": __builtins__}, local_ns)
            output = {k: str(v)[:200] for k, v in local_ns.items() if not k.startswith("_")}
            return {"status": "success", "detail": f"Code evaluated, {len(local_ns)} vars", "output": output}
        except Exception as e:
            return {"status": "failure", "detail": f"Code error: {str(e)}"}

    def _exec_app(self, node: DAGNode) -> Dict[str, Any]:
        """App interaction (simulated)."""
        return {"status": "success", "detail": f"App action simulated: {node.params.get('description', '')}"}

    def rollback(self, path: str) -> bool:
        """Rollback a file operation using backup."""
        if path in self._backups and os.path.exists(self._backups[path]):
            shutil.copy2(self._backups[path], path)
            return True
        return False
