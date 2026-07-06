"""Proactive Simulation Engine — runs hourly 1000-scenario simulations.
Per paper Section 4.3: generates 1000 scenarios by sampling action sequences,
rolls them out in latent space, classifies confidence, and triggers actions."""
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class SimulationResult:
    """Result of a single proactive simulation cycle."""
    timestamp: float
    num_scenarios: int
    mean_cost: float
    min_cost: float
    confidence: float
    recommended_action: str  # automatic, prompt_user, hold
    state_vector: List[float]
    goal_vector: List[float]


class ProactiveSimulationEngine:
    """Proactive simulation engine per paper Section 4.3.
    Runs hourly simulations with 1000 scenarios in latent space.
    >95% confidence → automatic action
    60-80% confidence → prompt user
    <60% confidence → hold
    """

    def __init__(self, world_model, interval_seconds: float = 3600.0,
                 num_scenarios: int = 1000):
        self.world_model = world_model
        self.interval = interval_seconds
        self.num_scenarios = num_scenarios
        self.results: List[SimulationResult] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._on_action: Optional[Callable] = None

    def set_action_callback(self, callback: Callable):
        """Set callback for when simulation triggers an action."""
        self._on_action = callback

    def simulate(self, state_vector: List[float], goal_vector: List[float]) -> SimulationResult:
        """Run a single simulation cycle with 1000 scenarios."""
        result_dict = self.world_model.proactive_simulate(
            state_vector, goal_vector, num_scenarios=self.num_scenarios
        )

        result = SimulationResult(
            timestamp=time.time(),
            num_scenarios=result_dict["num_scenarios"],
            mean_cost=result_dict["mean_cost"],
            min_cost=result_dict["min_cost"],
            confidence=result_dict["confidence"],
            recommended_action=result_dict["recommended_action"],
            state_vector=state_vector[:10],  # store truncated for audit
            goal_vector=goal_vector[:10],
        )
        self.results.append(result)

        # Trigger callback if action needed
        if self._on_action and result.recommended_action != "hold":
            self._on_action(result)

        return result

    def start(self, state_vector: List[float], goal_vector: List[float]):
        """Start hourly simulation in background thread."""
        self._running = True

        def _loop():
            while self._running:
                self.simulate(state_vector, goal_vector)
                time.sleep(self.interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background simulation."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent simulation results."""
        return [
            {
                "timestamp": r.timestamp,
                "num_scenarios": r.num_scenarios,
                "mean_cost": r.mean_cost,
                "min_cost": r.min_cost,
                "confidence": r.confidence,
                "recommended_action": r.recommended_action,
            }
            for r in self.results[-limit:]
        ]

    def stats(self) -> Dict[str, Any]:
        return {
            "total_simulations": len(self.results),
            "running": self._running,
            "interval_seconds": self.interval,
            "num_scenarios": self.num_scenarios,
            "avg_confidence": (
                sum(r.confidence for r in self.results) / len(self.results)
                if self.results else 0.0
            ),
            "last_action": self.results[-1].recommended_action if self.results else "none",
        }
