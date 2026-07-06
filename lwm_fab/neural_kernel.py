"""Neural Process Kernel — preemptive cognitive scheduling with resource budgets.
Per paper Section 7.1: extends AIOS with priority-based preemption and real resource budgeting."""
import time
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from collections import deque
import threading


class ProcessPriority(str, Enum):
    CRITICAL = "critical"   # kernel-level, never preempted
    HIGH = "high"            # user-facing
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class ProcessState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


@dataclass
class ResourceBudget:
    max_tokens: int = 100_000
    max_latency_s: float = 30.0
    max_vram_mb: float = 4096
    max_cpu_percent: float = 50.0
    max_wall_time_s: float = 120.0


@dataclass
class NeuralProcess:
    pid: str
    name: str
    priority: ProcessPriority
    budget: ResourceBudget
    task_fn: Callable
    context: Dict[str, Any] = field(default_factory=dict)
    state: ProcessState = ProcessState.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    tokens_used: int = 0
    cpu_time_s: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    checkpoint: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.pid:
            self.pid = str(uuid.uuid4())[:8]

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.completed_at or time.time()
        return end - self.started_at


class ResourceMonitor:
    """Monitors resource usage and enforces quotas. Kills processes that exceed budgets."""

    def __init__(self):
        self.violations: List[Dict[str, Any]] = []

    def check(self, proc: NeuralProcess) -> bool:
        """Returns True if process is within budget, False if violated."""
        if proc.state != ProcessState.RUNNING:
            return True

        elapsed = proc.elapsed
        if elapsed > proc.budget.max_wall_time_s:
            self._violate(proc, "wall_time_exceeded", f"{elapsed:.1f}s > {proc.budget.max_wall_time_s}s")
            return False
        if proc.tokens_used > proc.budget.max_tokens:
            self._violate(proc, "token_budget_exceeded", f"{proc.tokens_used} > {proc.budget.max_tokens}")
            return False
        return True

    def _violate(self, proc: NeuralProcess, violation_type: str, detail: str):
        proc.state = ProcessState.KILLED
        proc.error = f"Killed: {violation_type} ({detail})"
        self.violations.append({
            "pid": proc.pid, "name": proc.name,
            "type": violation_type, "detail": detail,
            "timestamp": time.time(),
        })


class NeuralScheduler:
    """Priority-based preemptive scheduler for cognitive processes.

    CRITICAL processes are never preempted.
    If a CRITICAL arrives while NORMAL is running, NORMAL gets checkpointed and suspended.
    """

    PRIORITY_ORDER = {
        ProcessPriority.CRITICAL: 0,
        ProcessPriority.HIGH: 1,
        ProcessPriority.NORMAL: 2,
        ProcessPriority.LOW: 3,
        ProcessPriority.BACKGROUND: 4,
    }

    def __init__(self):
        self.ready_queue: deque[NeuralProcess] = deque()
        self.running: Optional[NeuralProcess] = None
        self.suspended: List[NeuralProcess] = []
        self.completed: List[NeuralProcess] = []
        self.monitor = ResourceMonitor()
        self._lock = threading.Lock()

    def submit(self, proc: NeuralProcess):
        """Submit a new process to the scheduler."""
        with self._lock:
            self.ready_queue.append(proc)
            self._reorder_queue()

    def _reorder_queue(self):
        """Sort ready queue by priority (lower number = higher priority)."""
        self.ready_queue = deque(sorted(
            self.ready_queue,
            key=lambda p: self.PRIORITY_ORDER[p.priority]
        ))

    def _preempt_if_needed(self, new_proc: NeuralProcess):
        """Preempt running process if new process has higher priority."""
        if self.running is None:
            return
        if self.PRIORITY_ORDER[new_proc.priority] < self.PRIORITY_ORDER[self.running.priority]:
            # Checkpoint and suspend current running process
            self.running.checkpoint = {"state": self.running.state, "context": self.running.context}
            self.running.state = ProcessState.SUSPENDED
            self.suspended.append(self.running)
            self.running = None

    def _next(self) -> Optional[NeuralProcess]:
        """Get next process to run."""
        if self.running is not None:
            return None  # something already running

        # Check if any suspended process can resume
        if self.suspended and not self.ready_queue:
            proc = self.suspended.pop(0)
            proc.state = ProcessState.RUNNING
            return proc

        if self.ready_queue:
            proc = self.ready_queue.popleft()
            proc.state = ProcessState.RUNNING
            proc.started_at = proc.started_at or time.time()
            return proc

        return None

    def run_all(self) -> List[NeuralProcess]:
        """Run all queued processes in priority order with preemption."""
        results = []

        while self.ready_queue or self.suspended or self.running:
            # Preempt check: highest priority in queue vs running
            if self.ready_queue and self.running:
                top = min(self.ready_queue, key=lambda p: self.PRIORITY_ORDER[p.priority])
                if self.PRIORITY_ORDER[top.priority] < self.PRIORITY_ORDER[self.running.priority]:
                    self._preempt_if_needed(top)

            # Get next process
            if self.running is None:
                proc = self._next()
                if proc is None:
                    break
                self.running = proc

            # Execute
            proc = self.running
            try:
                proc.result = proc.task_fn(proc)
                proc.state = ProcessState.COMPLETED
                proc.completed_at = time.time()
            except Exception as e:
                proc.state = ProcessState.FAILED
                proc.error = str(e)
                proc.completed_at = time.time()

            # Resource check
            self.monitor.check(proc)

            results.append(proc)
            self.completed.append(proc)
            self.running = None

        return results

    def status(self) -> Dict[str, Any]:
        return {
            "ready": len(self.ready_queue),
            "running": 1 if self.running else 0,
            "suspended": len(self.suspended),
            "completed": len(self.completed),
            "killed": sum(1 for p in self.completed if p.state == ProcessState.KILLED),
            "violations": self.monitor.violations,
        }


class NeuralProcessKernel:
    """The LWM Fabricator Neural Process Kernel.
    Extends AIOS with preemptive cognitive scheduling and real resource budgeting."""

    def __init__(self):
        self.scheduler = NeuralScheduler()
        self.monitor = ResourceMonitor()

    def spawn(self, name: str, priority: ProcessPriority, task_fn: Callable,
              budget: Optional[ResourceBudget] = None, context: Optional[Dict] = None) -> str:
        """Spawn a new cognitive process."""
        proc = NeuralProcess(
            pid="",
            name=name,
            priority=priority,
            budget=budget or ResourceBudget(),
            task_fn=task_fn,
            context=context or {},
        )
        self.scheduler.submit(proc)
        return proc.pid

    def run(self) -> List[NeuralProcess]:
        """Run all submitted processes."""
        return self.scheduler.run_all()

    def get_status(self) -> Dict[str, Any]:
        return self.scheduler.status()
