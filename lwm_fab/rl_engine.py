"""Step-Level RL Credit Assignment — Agent-Ri + GRPO per paper Section 6."""
import json
import uuid
from typing import List, Dict, Any, Optional
from .models import Transition, DAGNode


# Reward weights per paper Section 6.2
W_SUCCESS = 1.0
W_FAILURE = 2.0
W_WARNING = 0.5
W_CONSENT = 0.3

GAMMA = 0.95


class RLEngine:
    """Step-level RL credit assignment for fabrication DAG execution."""

    def __init__(self, gamma: float = GAMMA):
        self.gamma = gamma
        self.transitions: List[Transition] = []

    def compute_reward(self, node: DAGNode, execution_result: Dict[str, Any]) -> float:
        """Composite reward: r_t = w1*1[success] - w2*1[failure] - w3*1[warning] + w4*1[consent_approved]."""
        status = execution_result.get("status", "unknown")
        consent_approved = execution_result.get("consent_approved", False)

        reward = 0.0
        if status == "success":
            reward += W_SUCCESS
        elif status == "failure":
            reward -= W_FAILURE
        elif status == "warning":
            reward -= W_WARNING

        if consent_approved:
            reward += W_CONSENT

        return reward

    def record_transition(
        self,
        trajectory_uid: str,
        step_index: int,
        observation: Dict[str, Any],
        node: DAGNode,
        execution_result: Dict[str, Any],
        next_observation: Dict[str, Any],
        done: bool,
    ) -> Transition:
        """Record a single (s_t, a_t, r_t, s_{t+1}) transition."""
        reward = self.compute_reward(node, execution_result)

        transition = Transition(
            step_id=str(uuid.uuid4()),
            trajectory_uid=trajectory_uid,
            step_index=step_index,
            observation=observation,
            action={
                "action_type": node.action_type.value,
                "verb": node.verb.value,
                "params": node.params,
                "consent_level": node.consent_level.value,
            },
            reward=reward,
            done=done,
            next_observation=next_observation,
            feedback=execution_result,
        )
        self.transitions.append(transition)
        return transition

    def compute_returns_and_advantages(self, trajectory_uid: str) -> List[Transition]:
        """Compute discounted returns G_t and GRPO advantages A_t for a trajectory."""
        traj = [t for t in self.transitions if t.trajectory_uid == trajectory_uid]
        traj.sort(key=lambda t: t.step_index)

        if not traj:
            return []

        T = len(traj)
        # Compute discounted returns G_t = sum_{k=t}^{T-1} gamma^{k-t} * r_k
        returns = [0.0] * T
        running = 0.0
        for t in reversed(range(T)):
            running = traj[t].reward + self.gamma * running
            returns[t] = running

        # G_bar = mean return across trajectory (GRPO baseline)
        g_bar = sum(returns) / T if T > 0 else 0.0

        # Advantage A_t = G_t - G_bar
        for t in range(T):
            traj[t].discounted_return = round(returns[t], 4)
            traj[t].advantage = round(returns[t] - g_bar, 4)

        return traj

    def export_jsonl(self, trajectory_uid: str, filepath: str) -> int:
        """Export trajectory as JSONL in OpenPipe ART format."""
        traj = self.compute_returns_and_advantages(trajectory_uid)
        with open(filepath, "w", encoding="utf-8") as f:
            for t in traj:
                record = {
                    "step_id": t.step_id,
                    "trajectory_uid": t.trajectory_uid,
                    "step_index": t.step_index,
                    "observation": t.observation,
                    "action": t.action,
                    "reward": t.reward,
                    "done": t.done,
                    "next_observation": t.next_observation,
                    "feedback": t.feedback,
                    "advantage": t.advantage,
                    "discounted_return": t.discounted_return,
                }
                f.write(json.dumps(record) + "\n")
        return len(traj)

    def get_bottleneck(self, trajectory_uid: str) -> Dict[str, Any]:
        """Identify which step caused failure in a trajectory."""
        traj = self.compute_returns_and_advantages(trajectory_uid)
        if not traj:
            return {}

        failing = [t for t in traj if t.reward < 0]
        succeeding = [t for t in traj if t.reward > 0]

        return {
            "total_steps": len(traj),
            "failing_steps": [
                {"step_index": t.step_index, "advantage": t.advantage, "reward": t.reward}
                for t in failing
            ],
            "succeeding_steps": [
                {"step_index": t.step_index, "advantage": t.advantage, "reward": t.reward}
                for t in succeeding
            ],
            "mean_advantage_failing": round(sum(t.advantage for t in failing) / len(failing), 4) if failing else 0.0,
            "mean_advantage_succeeding": round(sum(t.advantage for t in succeeding) / len(succeeding), 4) if succeeding else 0.0,
            "bottleneck_step": failing[0].step_index if failing else None,
        }
