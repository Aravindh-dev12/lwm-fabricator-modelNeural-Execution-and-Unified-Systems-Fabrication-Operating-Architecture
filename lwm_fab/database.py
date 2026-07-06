"""SQLite persistence layer for fabrication runs, nodes, and RL transitions."""
import sqlite3
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime


SCHEMA = """
CREATE TABLE IF NOT EXISTS capability_domains (
    name TEXT PRIMARY KEY,
    intent_patterns TEXT,
    consent_level TEXT,
    audience TEXT,
    source_startups TEXT
);

CREATE TABLE IF NOT EXISTS fabrication_runs (
    run_id TEXT PRIMARY KEY,
    intent TEXT,
    matched_domains TEXT,
    status TEXT,
    mode TEXT,
    created_at TEXT,
    num_nodes INTEGER
);

CREATE TABLE IF NOT EXISTS fabrication_nodes (
    node_id TEXT,
    run_id TEXT,
    step_index INTEGER,
    action_type TEXT,
    verb TEXT,
    params TEXT,
    consent_level TEXT,
    status TEXT,
    result TEXT,
    PRIMARY KEY (node_id, run_id)
);

CREATE TABLE IF NOT EXISTS rl_transitions (
    step_id TEXT PRIMARY KEY,
    trajectory_uid TEXT,
    step_index INTEGER,
    observation TEXT,
    action_json TEXT,
    reward REAL,
    done INTEGER,
    next_observation TEXT,
    feedback_json TEXT,
    advantage REAL,
    discounted_return REAL
);

CREATE TABLE IF NOT EXISTS safety_assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    node_id TEXT,
    auq_result TEXT,
    verdict TEXT,
    c_judge REAL,
    r_residual REAL,
    modifications TEXT,
    reasoning TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS curiosity_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT,
    hypothesis TEXT,
    intent TEXT,
    domains_tested TEXT,
    result TEXT,
    insight TEXT,
    confidence REAL,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS proactive_simulations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    num_scenarios INTEGER,
    mean_cost REAL,
    min_cost REAL,
    confidence REAL,
    recommended_action TEXT,
    state_snapshot TEXT
);

CREATE TABLE IF NOT EXISTS telemetry_routes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_json TEXT,
    routed_to TEXT,
    arm_name TEXT,
    context_vector TEXT,
    timestamp TEXT
);

CREATE TABLE IF NOT EXISTS mcp_tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT,
    agent_domain TEXT,
    tool_name TEXT,
    params TEXT,
    result TEXT,
    elapsed_s REAL,
    status TEXT,
    timestamp TEXT
);
"""


class Database:
    def __init__(self, db_path: str = "lwm_fab.db"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def save_run(self, run_id: str, intent: str, matched_domains: List, status: str, mode: str, num_nodes: int):
        self.conn.execute(
            "INSERT OR REPLACE INTO fabrication_runs VALUES (?,?,?,?,?,?,?)",
            (run_id, intent, json.dumps(matched_domains), status, mode,
             datetime.utcnow().isoformat(), num_nodes)
        )
        self.conn.commit()

    def save_node(self, node_id: str, run_id: str, step_index: int, action_type: str,
                  verb: str, params: Dict, consent_level: str, status: str, result: Optional[Dict]):
        self.conn.execute(
            "INSERT OR REPLACE INTO fabrication_nodes VALUES (?,?,?,?,?,?,?,?,?)",
            (node_id, run_id, step_index, action_type, verb, json.dumps(params),
             consent_level, status, json.dumps(result) if result else None)
        )
        self.conn.commit()

    def save_transition(self, t: Dict[str, Any]):
        self.conn.execute(
            "INSERT OR REPLACE INTO rl_transitions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (t["step_id"], t["trajectory_uid"], t["step_index"],
             json.dumps(t["observation"]), json.dumps(t["action"]),
             t["reward"], int(t["done"]), json.dumps(t["next_observation"]),
             json.dumps(t.get("feedback", {})), t.get("advantage", 0.0),
             t.get("discounted_return", 0.0))
        )
        self.conn.commit()

    def save_safety_assessment(self, run_id: str, node_id: str, assessment: Dict):
        self.conn.execute(
            "INSERT INTO safety_assessments (run_id, node_id, auq_result, verdict, c_judge, r_residual, modifications, reasoning, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, node_id, json.dumps(assessment.get("auq", {})),
             assessment.get("verdict", ""), assessment.get("c_judge", 0.0),
             assessment.get("r_residual", 0.0), json.dumps(assessment.get("modifications", [])),
             assessment.get("reasoning", ""), datetime.utcnow().isoformat())
        )
        self.conn.commit()

    def get_run(self, run_id: str) -> Optional[Dict]:
        cur = self.conn.execute("SELECT * FROM fabrication_runs WHERE run_id=?", (run_id,))
        row = cur.fetchone()
        if row:
            return {"run_id": row[0], "intent": row[1], "matched_domains": json.loads(row[2]),
                    "status": row[3], "mode": row[4], "created_at": row[5], "num_nodes": row[6]}
        return None

    def get_nodes(self, run_id: str) -> List[Dict]:
        cur = self.conn.execute("SELECT * FROM fabrication_nodes WHERE run_id=? ORDER BY step_index", (run_id,))
        rows = cur.fetchall()
        return [{"node_id": r[0], "run_id": r[1], "step_index": r[2], "action_type": r[3],
                 "verb": r[4], "params": json.loads(r[5]), "consent_level": r[6],
                 "status": r[7], "result": json.loads(r[8]) if r[8] else None}
                for r in rows]

    def list_runs(self, limit: int = 20) -> List[Dict]:
        cur = self.conn.execute("SELECT * FROM fabrication_runs ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        return [{"run_id": r[0], "intent": r[1], "status": r[3], "mode": r[4],
                 "created_at": r[5], "num_nodes": r[6]}
                for r in rows]

    def save_curiosity_entry(self, experiment_id: str, hypothesis: str, intent: str,
                             domains_tested: List[str], result: str, insight: str, confidence: float):
        self.conn.execute(
            "INSERT INTO curiosity_ledger (experiment_id, hypothesis, intent, domains_tested, result, insight, confidence, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (experiment_id, hypothesis, intent, json.dumps(domains_tested),
             result, insight, confidence, datetime.utcnow().isoformat())
        )
        self.conn.commit()

    def list_curiosity_entries(self, limit: int = 20) -> List[Dict]:
        cur = self.conn.execute("SELECT * FROM curiosity_ledger ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        return [{"id": r[0], "experiment_id": r[1], "hypothesis": r[2], "intent": r[3],
                 "domains_tested": json.loads(r[4]), "result": r[5], "insight": r[6],
                 "confidence": r[7], "created_at": r[8]}
                for r in rows]

    def save_proactive_simulation(self, num_scenarios: int, mean_cost: float,
                                  min_cost: float, confidence: float,
                                  recommended_action: str, state_snapshot: str):
        self.conn.execute(
            "INSERT INTO proactive_simulations (timestamp, num_scenarios, mean_cost, min_cost, confidence, recommended_action, state_snapshot) VALUES (?,?,?,?,?,?,?)",
            (datetime.utcnow().isoformat(), num_scenarios, mean_cost, min_cost,
             confidence, recommended_action, state_snapshot)
        )
        self.conn.commit()

    def list_proactive_simulations(self, limit: int = 20) -> List[Dict]:
        cur = self.conn.execute("SELECT * FROM proactive_simulations ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        return [{"id": r[0], "timestamp": r[1], "num_scenarios": r[2], "mean_cost": r[3],
                 "min_cost": r[4], "confidence": r[5], "recommended_action": r[6]}
                for r in rows]

    def save_telemetry_route(self, task: Dict, routed_to: str, arm_name: str, context_vector: List):
        self.conn.execute(
            "INSERT INTO telemetry_routes (task_json, routed_to, arm_name, context_vector, timestamp) VALUES (?,?,?,?,?)",
            (json.dumps(task), routed_to, arm_name, json.dumps(context_vector),
             datetime.utcnow().isoformat())
        )
        self.conn.commit()

    def save_mcp_tool_call(self, agent_id: str, agent_domain: str, tool_name: str,
                           params: Dict, result: Dict, elapsed_s: float, status: str):
        self.conn.execute(
            "INSERT INTO mcp_tool_calls (agent_id, agent_domain, tool_name, params, result, elapsed_s, status, timestamp) VALUES (?,?,?,?,?,?,?,?)",
            (agent_id, agent_domain, tool_name, json.dumps(params),
             json.dumps(result), elapsed_s, status, datetime.utcnow().isoformat())
        )
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()
