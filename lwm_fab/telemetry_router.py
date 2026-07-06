"""Telemetry Router — LinUCB contextual bandit for routing intelligence tasks.
Per paper Section 7.2: routes tasks to best-performing modules using an 8-dimensional feature vector."""
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import time


@dataclass
class BanditArm:
    """An arm in the LinUCB bandit — represents a module/agent that can handle a task."""
    arm_id: str
    name: str
    A: np.ndarray = field(default_factory=lambda: np.eye(8))       # context matrix (8D features)
    b: np.ndarray = field(default_factory=lambda: np.zeros(8))     # reward vector
    theta: np.ndarray = field(default_factory=lambda: np.zeros(8)) # learned weights
    pulls: int = 0
    total_reward: float = 0.0


class LinUCBBandit:
    """LinUCB contextual bandit for telemetry routing.
    Feature vector is 8-dimensional: task_type, complexity, urgency,
    historical_success_rate, domain_match_score, consent_level, latency_estimate, resource_cost.
    Composite reward per paper: reward = alpha * success + beta * speed + gamma * safety."""

    def __init__(self, alpha: float = 1.0, reward_weights: Optional[Dict[str, float]] = None):
        self.alpha = alpha  # exploration parameter
        self.arms: Dict[str, BanditArm] = {}
        self.reward_weights = reward_weights or {"success": 0.5, "speed": 0.3, "safety": 0.2}

    def register_arm(self, arm_id: str, name: str):
        """Register a new arm (module/agent)."""
        self.arms[arm_id] = BanditArm(arm_id=arm_id, name=name)

    def get_context_vector(self, task: Dict[str, Any]) -> np.ndarray:
        """Extract 8-dimensional feature vector from a task."""
        return np.array([
            task.get("task_type_code", 0.5),        # 0-1: type of task
            task.get("complexity", 0.5),             # 0-1: complexity score
            task.get("urgency", 0.5),                # 0-1: urgency level
            task.get("historical_success", 0.5),     # 0-1: past success rate
            task.get("domain_match_score", 0.5),     # 0-1: how well domain matches
            task.get("consent_level_code", 0.5),     # 0-1: consent level (0=always, 1=never)
            task.get("latency_estimate", 0.5),       # 0-1: normalized expected latency
            task.get("resource_cost", 0.5),          # 0-1: normalized resource cost
        ])

    def select(self, context: np.ndarray) -> Optional[BanditArm]:
        """Select the best arm using UCB: theta^T x + alpha * sqrt(x^T A^{-1} x)."""
        if not self.arms:
            return None

        best_score = -float('inf')
        best_arm = None

        for arm in self.arms.values():
            # Update theta: theta = A^{-1} b
            A_inv = np.linalg.inv(arm.A + 1e-6 * np.eye(8))
            arm.theta = A_inv @ arm.b

            # UCB score
            mean = arm.theta @ context
            uncertainty = self.alpha * np.sqrt(context @ A_inv @ context)
            ucb = mean + uncertainty

            if ucb > best_score:
                best_score = ucb
                best_arm = arm

        return best_arm

    def update(self, arm_id: str, context: np.ndarray, reward_components: Dict[str, float]):
        """Update the bandit after observing a reward.
        Composite reward = w_success * success + w_speed * speed + w_safety * safety."""
        arm = self.arms.get(arm_id)
        if arm is None:
            return

        w = self.reward_weights
        reward = (
            w["success"] * reward_components.get("success", 0.0) +
            w["speed"] * reward_components.get("speed", 0.0) +
            w["safety"] * reward_components.get("safety", 0.0)
        )

        # LinUCB update: A += x x^T, b += r * x
        arm.A += np.outer(context, context)
        arm.b += reward * context
        arm.pulls += 1
        arm.total_reward += reward

    def route(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Route a task to the best module. Returns routing decision."""
        context = self.get_context_vector(task)
        arm = self.select(context)

        if arm is None:
            return {"routed_to": None, "error": "No arms registered"}

        return {
            "routed_to": arm.arm_id,
            "arm_name": arm.name,
            "context": context.tolist(),
            "arm_pulls": arm.pulls,
            "arm_avg_reward": arm.total_reward / max(arm.pulls, 1),
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "num_arms": len(self.arms),
            "arms": [
                {
                    "arm_id": a.arm_id,
                    "name": a.name,
                    "pulls": a.pulls,
                    "avg_reward": round(a.total_reward / max(a.pulls, 1), 4),
                }
                for a in self.arms.values()
            ],
        }


class TelemetryRouter:
    """Telemetry router that uses LinUCB to route intelligence tasks to best-performing modules.
    The bandit updates its weights after each task. Over time it learns which modules
    handle which task types best (per paper Section 7.2)."""

    def __init__(self, alpha: float = 1.0):
        self.bandit = LinUCBBandit(alpha=alpha)
        self.routing_history: List[Dict[str, Any]] = []

    def register_module(self, module_id: str, name: str):
        """Register a module/agent as a bandit arm."""
        self.bandit.register_arm(module_id, name)

    def route_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Route a task to the best module."""
        decision = self.bandit.route(task)
        self.routing_history.append({
            "task": task,
            "decision": decision,
            "timestamp": time.time(),
        })
        return decision

    def record_outcome(self, module_id: str, task: Dict[str, Any],
                       success: float, speed: float, safety: float):
        """Record the outcome of a routed task for bandit update."""
        context = self.bandit.get_context_vector(task)
        self.bandit.update(module_id, context, {
            "success": success,
            "speed": speed,
            "safety": safety,
        })

    def stats(self) -> Dict[str, Any]:
        return {
            "bandit": self.bandit.stats(),
            "total_routes": len(self.routing_history),
        }
