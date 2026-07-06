"""Memory Management — Memo (episodic memory) + Graphiti (temporal knowledge graph).
Per paper Section 3.3: episodic memory boost in intent classification, and Section 7: memory management."""
import time
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class EpisodicMemoryEntry:
    """A single episodic memory entry (Memo-style)."""
    entry_id: str
    intent: str
    domains: List[str]
    success: bool
    success_weight: float
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.entry_id:
            self.entry_id = hashlib.sha256(
                f"{self.intent}{self.timestamp}".encode()
            ).hexdigest()[:12]


class MemoStore:
    """Memo: Memory-Augmented LLM Agents.
    Stores episodic memories of past fabrication runs for retrieval and boosting."""

    def __init__(self, max_entries: int = 10_000):
        self.entries: List[EpisodicMemoryEntry] = []
        self.max_entries = max_entries
        self._index: Dict[str, List[int]] = defaultdict(list)  # intent_hash → entry indices

    def store(self, intent: str, domains: List[str], success: bool,
              success_weight: float = 1.0, context: Optional[Dict] = None) -> EpisodicMemoryEntry:
        """Store a new episodic memory."""
        entry = EpisodicMemoryEntry(
            entry_id="",
            intent=intent,
            domains=domains,
            success=success,
            success_weight=success_weight if success else 0.0,
            timestamp=time.time(),
            context=context or {},
        )
        self.entries.append(entry)
        intent_hash = self._hash_intent(intent)
        self._index[intent_hash].append(len(self.entries) - 1)

        # Enforce max entries (FIFO eviction)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)
            self._rebuild_index()

        return entry

    def retrieve(self, query: str, top_k: int = 5) -> List[EpisodicMemoryEntry]:
        """Retrieve relevant episodic memories for a query intent.
        Uses token-overlap similarity for retrieval."""
        query_tokens = set(query.lower().split())
        scored: List[Tuple[float, EpisodicMemoryEntry]] = []

        for entry in self.entries:
            entry_tokens = set(entry.intent.lower().split())
            overlap = len(query_tokens & entry_tokens)
            if overlap > 0:
                # Score by overlap ratio + recency + success
                similarity = overlap / max(len(query_tokens | entry_tokens), 1)
                recency = 1.0 / (1.0 + (time.time() - entry.timestamp) / 3600.0)  # decay over hours
                score = similarity * 0.5 + recency * 0.2 + entry.success_weight * 0.3
                scored.append((score, entry))

        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:top_k]]

    def get_boost_entries(self, query: str) -> List[Dict[str, Any]]:
        """Get memory entries in the format expected by FabricationEngine."""
        retrieved = self.retrieve(query)
        return [
            {
                "domains": e.domains,
                "success_weight": e.success_weight,
                "intent": e.intent,
            }
            for e in retrieved
        ]

    def _hash_intent(self, intent: str) -> str:
        return hashlib.sha256(intent.lower().encode()).hexdigest()[:16]

    def _rebuild_index(self):
        self._index.clear()
        for i, entry in enumerate(self.entries):
            self._index[self._hash_intent(entry.intent)].append(i)

    def stats(self) -> Dict[str, Any]:
        return {
            "total_entries": len(self.entries),
            "successful": sum(1 for e in self.entries if e.success),
            "failed": sum(1 for e in self.entries if not e.success),
            "unique_intents": len(self._index),
        }


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    timestamp: float
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphNode:
    node_id: str
    node_type: str  # "intent", "domain", "action", "outcome", "agent"
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class GraphitiStore:
    """Graphiti: Temporal Knowledge Graph for LLM Agents.
    Maintains a temporal graph of intents → domains → actions → outcomes."""

    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self._node_index: Dict[str, List[str]] = defaultdict(list)  # type → node_ids

    def add_node(self, node_id: str, node_type: str, properties: Optional[Dict] = None) -> GraphNode:
        """Add or update a node in the graph."""
        if node_id not in self.nodes:
            node = GraphNode(node_id=node_id, node_type=node_type, properties=properties or {})
            self.nodes[node_id] = node
            self._node_index[node_type].append(node_id)
        else:
            if properties:
                self.nodes[node_id].properties.update(properties)
        return self.nodes[node_id]

    def add_edge(self, source: str, target: str, relation: str,
                 weight: float = 1.0, properties: Optional[Dict] = None) -> GraphEdge:
        """Add a temporal edge to the graph."""
        edge = GraphEdge(
            source=source, target=target, relation=relation,
            timestamp=time.time(), weight=weight,
            properties=properties or {},
        )
        self.edges.append(edge)
        return edge

    def record_fabrication(self, intent: str, domains: List[str],
                           actions: List[str], outcome: str, run_id: str):
        """Record a complete fabrication run in the temporal graph."""
        # Add intent node
        intent_id = f"intent:{hashlib.sha256(intent.encode()).hexdigest()[:8]}"
        self.add_node(intent_id, "intent", {"text": intent, "run_id": run_id})

        # Add domain nodes and edges
        for domain in domains:
            domain_id = f"domain:{domain}"
            self.add_node(domain_id, "domain", {"name": domain})
            self.add_edge(intent_id, domain_id, "matched_domain")

        # Add action nodes and edges
        prev = intent_id
        for i, action in enumerate(actions):
            action_id = f"action:{run_id}:{i}"
            self.add_node(action_id, "action", {"description": action, "step": i})
            self.add_edge(prev, action_id, "executes", weight=1.0 / (i + 1))
            prev = action_id

        # Add outcome node
        outcome_id = f"outcome:{run_id}"
        self.add_node(outcome_id, "outcome", {"result": outcome})
        self.add_edge(prev, outcome_id, "resulted_in")

    def query_neighbors(self, node_id: str, relation: Optional[str] = None) -> List[GraphNode]:
        """Get neighboring nodes, optionally filtered by relation."""
        neighbors = []
        for edge in self.edges:
            if edge.source == node_id and (relation is None or edge.relation == relation):
                if edge.target in self.nodes:
                    neighbors.append(self.nodes[edge.target])
            elif edge.target == node_id and (relation is None or edge.relation == relation):
                if edge.source in self.nodes:
                    neighbors.append(self.nodes[edge.source])
        return neighbors

    def get_domain_history(self, domain: str) -> List[Dict[str, Any]]:
        """Get temporal history of a domain's usage and outcomes."""
        domain_id = f"domain:{domain}"
        history = []
        for edge in self.edges:
            if edge.target == domain_id and edge.relation == "matched_domain":
                intent_node = self.nodes.get(edge.source)
                if intent_node:
                    # Find outcomes linked to this intent
                    outcomes = self.query_neighbors(intent_node.node_id, "resulted_in")
                    for out in outcomes:
                        history.append({
                            "intent": intent_node.properties.get("text", ""),
                            "outcome": out.properties.get("result", ""),
                            "timestamp": edge.timestamp,
                        })
        return history

    def stats(self) -> Dict[str, Any]:
        type_counts = {t: len(ids) for t, ids in self._node_index.items()}
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": type_counts,
        }


class MemoryManager:
    """Unified memory manager combining Memo (episodic) + Graphiti (temporal graph)."""

    def __init__(self):
        self.memo = MemoStore()
        self.graphiti = GraphitiStore()

    def record_run(self, intent: str, domains: List[str], actions: List[str],
                   outcome: str, run_id: str, success: bool):
        """Record a fabrication run in both memory systems."""
        self.memo.store(intent, domains, success, context={"run_id": run_id})
        self.graphiti.record_fabrication(intent, domains, actions, outcome, run_id)

    def get_episodic_boost(self, query: str) -> List[Dict[str, Any]]:
        """Get episodic memory boost for intent classification."""
        return self.memo.get_boost_entries(query)

    def get_domain_history(self, domain: str) -> List[Dict[str, Any]]:
        """Get temporal history for a domain."""
        return self.graphiti.get_domain_history(domain)

    def stats(self) -> Dict[str, Any]:
        return {
            "memo": self.memo.stats(),
            "graphiti": self.graphiti.stats(),
        }
