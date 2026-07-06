"""LWM Fabricator Kernel — orchestrates all agentic layers:
Neural Process Kernel (scheduling) + Fabrication Engine + LeWM World Model +
MCP Agents + Memory (Memo+Graphiti) + Safety Gate + Action Executor + RL Engine +
Telemetry Router + LLM Integration + Proactive Simulation Engine."""
import time
import uuid
from typing import Dict, Any, List, Optional
from .fabrication_engine import FabricationEngine
from .world_model import LeWMWorldModel
from .safety_gate import SafetyGate
from .action_executor import ActionExecutor
from .rl_engine import RLEngine
from .database import Database
from .memory import MemoryManager
from .mcp_agent import MCPAgentRegistry, create_mcp_agents_for_domains
from .telemetry_router import TelemetryRouter
from .neural_kernel import NeuralProcessKernel, NeuralProcess, ProcessPriority, ResourceBudget
from .ollama_integration import OllamaClient, OllamaReasoningEngine, HybridLLMClient, HFInferenceClient
from .proactive_engine import ProactiveSimulationEngine
from .models import ExecutionDAG, DAGNode, ConsentLevel, ActionType, ActionVerb


class LWMFabricator:
    """The LWM Fabricator kernel — full agentic layered architecture.

    Layers (bottom to top):
    1. Neural Process Kernel — preemptive cognitive scheduling with resource budgets
    2. Memory Layer — Memo (episodic) + Graphiti (temporal knowledge graph)
    3. MCP Agent Layer — Model Context Protocol agents wrapping capability domains
    4. Fabrication Engine — classify() + fabricate() DAG synthesis
    5. LeWM World Model — JEPA latent space prediction + CEM planning (transformer predictor)
    6. Safety Gate — AUQ + MACI adversarial debate (qwen3:4b reflex + qwen3:14b judge)
    7. Action Executor — grounded file/shell/network/code execution
    8. RL Engine — step-level credit assignment (Agent-Ri + GRPO)
    9. Telemetry Router — LinUCB contextual bandit for module routing

    Cross-cutting:
    - LLM Integration: qwen3:14b (control brain) + qwen3:4b (reflex) on localhost:11434
    - Proactive Simulation Engine: hourly 1000-scenario latent space simulations
    - Curiosity Experiment Ledger: SQLite-backed experiment tracking
    """

    def __init__(
        self,
        mode: str = "dry_run",
        db_path: str = "lwm_fab.db",
        use_safety_gate: bool = True,
        use_world_model: bool = True,
        use_mcp_agents: bool = True,
        use_memory: bool = True,
        use_neural_kernel: bool = True,
        use_telemetry_router: bool = True,
        use_ollama: bool = True,
        use_proactive: bool = True,
        calibration_threshold: float = 0.15,
        ollama_url: str = "http://localhost:11434",
        judge_model: str = "qwen3:14b",
        reflex_model: str = "qwen3:4b",
        hf_api_key: Optional[str] = None,
        work_dir: str = ".",
    ):
        self.mode = mode

        # LLM Integration: HybridLLMClient auto-selects Ollama (local) or HF Inference API (remote)
        # Per paper §8.1: qwen3:14b (judge) + qwen3:4b (reflex) on Ollama
        # Fallback: Qwen/Qwen2.5-7B-Instruct + Qwen/Qwen2.5-1.5B-Instruct on HF Inference API
        if use_ollama:
            self.ollama = HybridLLMClient(
                ollama_url=ollama_url,
                ollama_judge=judge_model,
                ollama_reflex=reflex_model,
                hf_api_key=hf_api_key,
            )
        else:
            self.ollama = None
        self.reasoning = OllamaReasoningEngine(self.ollama) if self.ollama else None

        # Layer 1: Neural Process Kernel
        self.npk = NeuralProcessKernel() if use_neural_kernel else None

        # Layer 2: Memory (Memo + Graphiti)
        self.memory = MemoryManager() if use_memory else None

        # Layer 3: MCP Agent Registry
        self.mcp_registry = MCPAgentRegistry() if use_mcp_agents else None

        # Layer 4: Fabrication Engine (with memory boost)
        episodic_memory = self.memory.get_episodic_boost("") if self.memory else []
        self.fabricator = FabricationEngine(episodic_memory=episodic_memory)

        # Layer 5: LeWM World Model (transformer predictor: depth=6, heads=16, MLP=2048)
        self.world_model = LeWMWorldModel() if use_world_model else None

        # Layer 6: Safety Gate (with Ollama debate protocol)
        self.safety_gate = SafetyGate(
            calibration_threshold=calibration_threshold,
            ollama_url=ollama_url,
            ollama_client=self.ollama,
        ) if use_safety_gate else None

        # Layer 7: Action Executor
        self.executor = ActionExecutor(mode=mode, safety_gate=self.safety_gate, work_dir=work_dir)

        # Layer 8: RL Engine
        self.rl = RLEngine()

        # Layer 9: Telemetry Router
        self.telemetry = TelemetryRouter() if use_telemetry_router else None

        # Proactive Simulation Engine
        self.proactive = ProactiveSimulationEngine(self.world_model) if (use_proactive and self.world_model) else None

        # Database
        self.db = Database(db_path)

        # Register MCP agents for all domains
        if self.mcp_registry:
            from .domain_registry import DOMAINS
            all_domain_names = [d.name for d in DOMAINS]
            agents = create_mcp_agents_for_domains(all_domain_names)
            for agent in agents:
                self.mcp_registry.register(agent)
                if self.telemetry:
                    self.telemetry.register_module(agent.agent_id, agent.name)

    def process_intent(self, intent: str, c_stated: float = 0.9) -> Dict[str, Any]:
        """Full end-to-end agentic pipeline:
        classify → fabricate → MCP agent setup → LeWM validate → schedule →
        safety gate → execute → RL credit assignment → memory record → telemetry update."""
        run_id = str(uuid.uuid4())
        pipeline_log: List[Dict[str, Any]] = []

        # --- Layer 2: Memory retrieval for episodic boost ---
        t0 = time.time()
        if self.memory:
            episodic_boost = self.memory.get_episodic_boost(intent)
            self.fabricator.episodic_memory = episodic_boost
            pipeline_log.append({"step": "memory_retrieval", "elapsed_ms": round((time.time()-t0)*1000, 2),
                                 "episodic_entries": len(episodic_boost)})

        # --- Layer 4: Classify intent ---
        t0 = time.time()
        fab_result = self.fabricator.run(intent)
        matched = fab_result.matched_domains
        dag = fab_result.dag
        dag.run_id = run_id
        state_vector = fab_result.state_vector
        pipeline_log.append({"step": "classify", "elapsed_ms": round((time.time()-t0)*1000, 2),
                             "matched_domains": matched})

        # --- Layer 3: Get MCP agents for matched domains ---
        mcp_agents_info = []
        if self.mcp_registry:
            for domain_name, score in matched:
                agent = self.mcp_registry.get_by_domain(domain_name)
                if agent:
                    mcp_agents_info.append({
                        "agent_id": agent.agent_id,
                        "domain": domain_name,
                        "tools": agent.list_tools(),
                        "resources": agent.list_resources(),
                        "prompts": agent.list_prompts(),
                    })
        pipeline_log.append({"step": "mcp_agent_binding", "agents": mcp_agents_info})

        # --- Layer 4: Fabricate DAG ---
        t0 = time.time()
        pipeline_log.append({"step": "fabricate", "elapsed_ms": round((time.time()-t0)*1000, 2),
                             "num_nodes": len(dag.nodes)})

        # --- Layer 5: Pre-execution validation with LeWM ---
        validation = None
        if self.world_model and len(state_vector) == 64:
            t0 = time.time()
            goal_vector = [0.8] * 64
            validation = self.world_model.validate_dag(state_vector, goal_vector)
            pipeline_log.append({"step": "lewm_validate", "elapsed_ms": round((time.time()-t0)*1000, 2),
                                 "validation": validation})
            if not validation["valid"]:
                pipeline_log.append({"step": "dag_revision", "note": "DAG cost above threshold, proceeding with caution"})

        # --- Layer 9: Telemetry routing for each node ---
        telemetry_decisions = []
        if self.telemetry:
            for i, node in enumerate(dag.nodes):
                task = {
                    "task_type_code": hash(node.action_type.value) % 100 / 100.0,
                    "complexity": 0.5 + 0.1 * (len(node.params) % 5),
                    "urgency": 0.8 if node.consent_level == ConsentLevel.ELEVATED else 0.5,
                    "historical_success": 0.7,
                    "domain_match_score": matched[0][1] if matched else 0.5,
                    "consent_level_code": {"always": 0.0, "standard": 0.3, "elevated": 0.7, "never": 1.0}.get(node.consent_level.value, 0.5),
                    "latency_estimate": 0.3,
                    "resource_cost": 0.4,
                }
                decision = self.telemetry.route_task(task)
                telemetry_decisions.append({"node_id": node.node_id, "routed_to": decision.get("arm_name", "unknown")})
        pipeline_log.append({"step": "telemetry_routing", "decisions": telemetry_decisions})

        # --- Layer 1: Schedule execution as Neural Process ---
        node_results = []
        paused_nodes = []

        # Persist run
        self.db.save_run(run_id, intent, matched, "running", self.mode, len(dag.nodes))

        # Execute DAG nodes
        t0 = time.time()
        for i, node in enumerate(dag.nodes):
            # Check dependencies
            deps_met = all(
                any(r["node_id"] == dep and r.get("status") in ("success", "completed") for r in node_results)
                for dep in node.depends_on
            ) if node.depends_on else True

            if not deps_met:
                result = {"node_id": node.node_id, "status": "skipped", "detail": "Dependencies not met"}
                node_results.append(result)
                continue

            # Layer 1+7: Route code:eval through Neural Process Kernel (paper §3.5)
            if node.action_type == ActionType.CODE and self.npk:
                def _exec_code(proc):
                    return self.executor.execute(node, c_stated=c_stated)
                pid = self.npk.spawn(
                    f"code_eval_{node.node_id[:8]}",
                    ProcessPriority.NORMAL,
                    _exec_code,
                    budget=ResourceBudget(max_wall_time_s=30, max_tokens=4096),
                )
                completed = self.npk.run()
                proc = next((p for p in completed if p.pid == pid), None)
                result = proc.result if proc and proc.result else {"node_id": node.node_id, "status": "failed", "detail": "NPK timeout"}
            else:
                # Layer 6+7: Safety gate + Action execution
                result = self.executor.execute(node, c_stated=c_stated)

            # Ollama reasoning augmentation for code:eval nodes
            if node.action_type == ActionType.CODE and self.reasoning and self.ollama and self.ollama.is_available():
                code = self.reasoning.generate_code(node.params.get("description", ""))
                result["generated_code"] = code[:500]

            node_results.append(result)

            # Save safety assessment
            if "safety_gate" in result:
                self.db.save_safety_assessment(run_id, node.node_id, result["safety_gate"])

            # Save node to DB
            self.db.save_node(
                node.node_id, run_id, i, node.action_type.value,
                node.verb.value, node.params, node.consent_level.value,
                result.get("status", "unknown"), result
            )

            # Layer 8: Record RL transition
            obs = {"dag_state": "executing", "step": i, "total": len(dag.nodes), "node": node.node_id}
            next_obs = {"dag_state": "step_done" if result["status"] == "success" else "paused/failed",
                         "step": i + 1, "total": len(dag.nodes)}
            done = (i == len(dag.nodes) - 1)
            self.rl.record_transition(
                trajectory_uid=run_id, step_index=i, observation=obs,
                node=node, execution_result=result, next_observation=next_obs, done=done,
            )

            # Layer 9: Record telemetry outcome
            if self.telemetry and self.mcp_registry:
                agent = self.mcp_registry.get_by_domain(node.params.get("domain", ""))
                if agent:
                    success_score = 1.0 if result.get("status") == "success" else 0.0
                    speed_score = 1.0 / (1.0 + i * 0.1)
                    safety_score = 1.0 if "safety_gate" not in result else (1.0 - result["safety_gate"]["auq"]["r_risk"])
                    task = {"task_type_code": 0.5, "complexity": 0.5, "urgency": 0.5,
                            "historical_success": 0.7, "domain_match_score": 0.8,
                            "consent_level_code": 0.5, "latency_estimate": 0.3, "resource_cost": 0.4}
                    self.telemetry.record_outcome(agent.agent_id, task, success_score, speed_score, safety_score)

            if result.get("status") == "paused":
                paused_nodes.append(node.node_id)

        exec_elapsed = time.time() - t0
        pipeline_log.append({"step": "execute", "elapsed_ms": round(exec_elapsed * 1000, 2),
                             "node_results": node_results, "paused_nodes": paused_nodes})

        # --- Layer 8: Post-execution RL ---
        t0 = time.time()
        transitions = self.rl.compute_returns_and_advantages(run_id)
        bottleneck = self.rl.get_bottleneck(run_id)
        pipeline_log.append({"step": "rl_credit_assignment", "elapsed_ms": round((time.time()-t0)*1000, 2),
                             "bottleneck": bottleneck, "num_transitions": len(transitions)})

        # --- Layer 2: Record in memory ---
        if self.memory:
            actions = [n.params.get("description", "") for n in dag.nodes]
            outcome = "success" if all(r.get("status") == "success" for r in node_results) else "partial_failure"
            self.memory.record_run(intent, [d for d, _ in matched], actions, outcome, run_id, success=(outcome == "success"))
            pipeline_log.append({"step": "memory_record", "memory_stats": self.memory.stats()})

        # Update run status
        all_success = all(r.get("status") in ("success", "completed") for r in node_results)
        any_paused = len(paused_nodes) > 0
        final_status = "completed" if all_success else ("paused" if any_paused else "failed")
        self.db.save_run(run_id, intent, matched, final_status, self.mode, len(dag.nodes))

        return {
            "run_id": run_id,
            "intent": intent,
            "matched_domains": matched,
            "dag": {
                "run_id": run_id,
                "nodes": [
                    {
                        "node_id": n.node_id,
                        "action_type": n.action_type.value,
                        "verb": n.verb.value,
                        "description": n.params.get("description", ""),
                        "consent_level": n.consent_level.value,
                        "depends_on": n.depends_on,
                    }
                    for n in dag.nodes
                ],
            },
            "mcp_agents": mcp_agents_info,
            "validation": validation,
            "telemetry": telemetry_decisions,
            "node_results": node_results,
            "rl_bottleneck": bottleneck,
            "pipeline_log": pipeline_log,
            "final_status": final_status,
            "mode": self.mode,
            "system_stats": self._system_stats(),
        }

    def _system_stats(self) -> Dict[str, Any]:
        stats = {}
        if self.npk:
            stats["neural_kernel"] = self.npk.get_status()
        if self.memory:
            stats["memory"] = self.memory.stats()
        if self.mcp_registry:
            stats["mcp_agents"] = self.mcp_registry.list_all()
        if self.telemetry:
            stats["telemetry"] = self.telemetry.stats()
        if self.ollama:
            stats["ollama"] = self.ollama.status()
        if self.proactive:
            stats["proactive_engine"] = self.proactive.stats()
        if self.world_model:
            stats["world_model"] = {
                "latent_dim": self.world_model.latent_dim,
                "predictor": "TransformerEncoder",
                "predictor_depth": self.world_model.predictor_depth,
                "predictor_heads": self.world_model.predictor_heads,
                "predictor_mlp_dim": self.world_model.predictor_mlp_dim,
                "history_size": self.world_model.history_size,
                "cem_samples": self.world_model.cem_samples,
                "cem_iterations": self.world_model.cem_iterations,
                "cem_topk": self.world_model.cem_topk,
                "horizon": self.world_model.horizon,
            }
        return stats

    def approve_node(self, run_id: str, node_id: str) -> Dict[str, Any]:
        """Approve a paused node for execution."""
        nodes = self.db.get_nodes(run_id)
        node_data = next((n for n in nodes if n["node_id"] == node_id), None)
        if not node_data:
            return {"error": "Node not found"}

        from .models import ActionType, ActionVerb
        node = DAGNode(
            node_id=node_data["node_id"],
            action_type=ActionType(node_data["action_type"]),
            verb=ActionVerb(node_data["verb"]),
            params=node_data["params"],
            consent_level=ConsentLevel(node_data["consent_level"]),
        )
        # Execute in live mode after approval
        old_mode = self.executor.mode
        self.executor.mode = "live"
        result = self.executor.execute(node, c_stated=1.0)
        self.executor.mode = old_mode

        self.db.save_node(node.node_id, run_id, node_data["step_index"],
                          node.action_type.value, node.verb.value, node.params,
                          node.consent_level.value, result.get("status"), result)
        return result

    def run_proactive_simulation(self, state_vector: List[float], goal_vector: List[float]) -> Dict[str, Any]:
        """Run a proactive simulation cycle (paper §4.3: 1000 scenarios in latent space).
        >95% confidence → automatic, 60-80% → prompt_user, <60% → hold."""
        if not self.proactive:
            return {"error": "Proactive engine not initialized"}
        result = self.proactive.simulate(state_vector, goal_vector)
        self.db.save_proactive_simulation(
            result.num_scenarios, result.mean_cost, result.min_cost,
            result.confidence, result.recommended_action,
            str(result.state_vector),
        )
        return {
            "num_scenarios": result.num_scenarios,
            "mean_cost": result.mean_cost,
            "min_cost": result.min_cost,
            "confidence": result.confidence,
            "recommended_action": result.recommended_action,
        }

    def close(self):
        if self.proactive:
            self.proactive.stop()
        self.db.close()
