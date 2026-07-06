from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
from enum import Enum
import uuid
from datetime import datetime


class ConsentLevel(str, Enum):
    ALWAYS = "always"
    STANDARD = "standard"
    ELEVATED = "elevated"
    NEVER = "never"


class ActionType(str, Enum):
    FILE = "file"
    SHELL = "shell"
    NETWORK = "network"
    APP = "app"
    CODE = "code"


class ActionVerb(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    REQUEST = "request"
    EVALUATE = "evaluate"


@dataclass
class CapabilityDomain:
    name: str
    intent_patterns: List[str]
    execution_grammar: List[Dict[str, Any]]
    required_capabilities: List[str]
    consent_level: ConsentLevel
    audience: str
    source_startups: List[str]


@dataclass
class DAGNode:
    node_id: str
    action_type: ActionType
    verb: ActionVerb
    params: Dict[str, Any]
    consent_level: ConsentLevel
    depends_on: List[str] = field(default_factory=list)
    status: Literal["pending", "running", "completed", "failed", "paused"] = "pending"
    result: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.node_id:
            self.node_id = str(uuid.uuid4())[:8]


@dataclass
class ExecutionDAG:
    run_id: str
    intent: str
    nodes: List[DAGNode]
    edges: List[tuple] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"
    mode: Literal["dry_run", "live"] = "dry_run"

    def __post_init__(self):
        if not self.run_id:
            self.run_id = str(uuid.uuid4())


@dataclass
class FabricationResult:
    intent: str
    matched_domains: List[tuple]  # (domain_name, score)
    dag: ExecutionDAG
    state_vector: List[float]


@dataclass
class Transition:
    step_id: str
    trajectory_uid: str
    step_index: int
    observation: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    done: bool
    next_observation: Dict[str, Any]
    feedback: Dict[str, Any] = field(default_factory=dict)
    advantage: float = 0.0
    discounted_return: float = 0.0


@dataclass
class SafetyVerdict:
    verdict: Literal["APPROVE", "MODIFY", "REJECT", "REQUIRE_HUMAN"]
    c_judge: float
    r_residual: float
    modifications: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class AUQResult:
    c_stated: float
    p_worst: float
    r_risk: float
    delta_cal: float
    triggers_debate: bool
