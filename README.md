# NEXUS-LWM OS

**Neural Execution and Unified Systems Fabrication Operating Architecture**

An MCP-native operating system for safe cross-system automation, latent-world
planning, capability fabrication, workflow orchestration, and governed execution.

## Unified Automation Architecture

Version 0.2 adds a Linux-first automation control plane that treats MCP servers
as a shared capability bus rather than separate user-facing agents. It includes:

- real MCP JSON-RPC stdio discovery and tool calls;
- n8n-style workflow DAGs with dependencies, expressions, conditions and retries;
- approval gates for writes and privileged operations;
- workspace-confined Linux capabilities and environment-backed secrets;
- persistent webhook and interval triggers;
- a hash-chained audit trail, resumable CLI and authenticated HTTP API.

```bash
python lwm_os.py --mcp-config mcp.json capabilities
python lwm_os.py --workspace /tmp/lwm run examples/hello_workflow.json
python lwm_api.py --workspace /tmp/lwm --mcp-config mcp.json
```

The public workflow console is live at
[lwm-fabricator-console.thearc12.chatgpt.site](https://lwm-fabricator-console.thearc12.chatgpt.site).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aravindh-dev12/lwm-fabricator-model/blob/main/lwm_fab_colab.ipynb)
[![HuggingFace Spaces](https://img.shields.io/badge/%F0%9F%A4%97-HF%20Spaces-blue)](https://huggingface.co/spaces/Aravindh-dev12/lwm-fabricator)

## Overview

LWM Fabricator is an agentic AI operating system with a **9-layer architecture** that fixes three problems in current AI OS designs:

1. **No Imagination** — JEPA-based latent world model with Transformer predictor (depth=6, heads=16, MLP=2048) and CEM planning in 192D latent space (<1s, 48x speedup over text-based LLM prediction)
2. **No Capability Synthesis** — Fabricates execution DAGs from raw intent by fusing 21 capability domain blueprints (no pre-built API integrations needed)
3. **No Calibrated Safety** — AUQ + MACI adversarial debate gate (qwen3:4b reflex + qwen3:14b judge) intercepts elevated-consent actions before execution

## 9-Layer Agentic Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Layer 9: Telemetry Router (LinUCB 8D contextual bandit) │
├─────────────────────────────────────────────────────────┤
│  Layer 8: RL Engine (Agent-Ri + GRPO, γ=0.95)           │
├─────────────────────────────────────────────────────────┤
│  Layer 7: Action Executor (file/shell/net/code)          │
├─────────────────────────────────────────────────────────┤
│  Layer 6: Safety Gate (AUQ + MACI LLM debate)            │
├─────────────────────────────────────────────────────────┤
│  Layer 5: LeWM World Model (JEPA + Transformer + CEM)    │
├─────────────────────────────────────────────────────────┤
│  Layer 4: Fabrication Engine (classify + fabricate DAG)  │
├─────────────────────────────────────────────────────────┤
│  Layer 3: MCP Agent Layer (21 domain agents, protocol)   │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Memory (Memo episodic + Graphiti temporal KG)  │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Neural Process Kernel (preemptive scheduler)   │
└─────────────────────────────────────────────────────────┘
         Cross-cutting: LLM Integration + Proactive Simulation
```

## Mathematical Formulations

### Layer 5: LeWM — Latent World Model (Paper §4)

**JEPA Encoder (state → latent):**

```
z_t = f_θ(s_t) = LayerNorm(W₂ · GELU(W₁ · s_t + b₁) + b₂)
```

where `s_t ∈ ℝ^64`, `z_t ∈ ℝ^192`, `W₁ ∈ ℝ^{192×64}`, `W₂ ∈ ℝ^{192×192}`

**Transformer Predictor (latent dynamics):**

```
ẑ_{t+1} = 0.7 · z_t + 0.3 · TransformerEncoder(z_{t-H:t}, ã_t)
```

- History context: `H = 3` past latent states
- Transformer: depth=6, heads=16, MLP_dim=2048
- `ã_t = g_φ(a_t)` where `a_t ∈ ℝ^32`, `ã_t ∈ ℝ^192` (SiLU activation)

**CEM Planning (Cross-Entropy Method):**

```
For i = 1..N_iter:
    Sample {a_j} ~ N(μ, σ²)  for j = 1..N_samples
    Evaluate C(a_j) = Σ_{t=1}^{H} ||ẑ_t(a_j) - z_goal||²
    Select top-K elites: S_k = top_k({a_j}, C)
    Update: μ ← mean(S_k), σ² ← var(S_k)
Return argmin C(a_j)
```

Config: `N_samples=300, N_iter=30, K=30, H=5` → 9,000 candidates evaluated

**Proactive Simulation:**

```
confidence = 1 - (mean_cost - min_cost) / mean_cost
action = auto if confidence > 0.95
       = prompt if 0.60 ≤ confidence ≤ 0.80
       = hold if confidence < 0.60
```

### Layer 6: Safety Gate — AUQ + MACI (Paper §5)

**AUQ (Action Uncertainty Quantification):**

```
p_worst = (1 - c_stated) · risk_weight(action_type, verb, consent)
delta_cal = |c_stated - p_worst|
r_risk = sigmoid(α · delta_cal + β · risk_score)
triggers_debate = delta_cal > τ (threshold = 0.15)
```

**MACI Debate Protocol:**

```
Proponent (qwen3:4b): argues action is safe
Opponent (qwen3:4b): argues action is dangerous
Judge (qwen3:14b): evaluates, issues verdict ∈ {APPROVE, MODIFY, REJECT, REQUIRE_HUMAN}
c_judge = confidence extracted from judge output
r_residual = r_risk · (1 - c_judge)
verdict = APPROVE if r_residual < 0.15
        = MODIFY if 0.15 ≤ r_residual < 0.30
        = REQUIRE_HUMAN if 0.30 ≤ r_residual < 0.50
        = REJECT if r_residual ≥ 0.50
```

### Layer 8: RL Credit Assignment — Agent-Ri + GRPO (Paper §6)

**Step-level reward:**

```
r_t = w₁ · success_t + w₂ · safety_t    (w₁=1, w₂=2)
```

**Discounted returns:**

```
G_t = Σ_{k=0}^{T-t} γ^k · r_{t+k}    (γ=0.95)
```

**Advantage (GRPO-style):**

```
A_t = G_t - mean(G_0, ..., G_T)
```

**Bottleneck identification:**

```
bottleneck = argmin_t A_t  among failing steps
```

### Layer 9: Telemetry Router — LinUCB (Paper §7.2)

**Context vector (8D):**

```
x = [task_type_code, complexity, urgency, historical_success,
     domain_match_score, consent_level_code, latency_estimate, resource_cost]
```

**LinUCB arm selection:**

```
For each arm a:
    A_a += x · xᵀ  (Gram matrix, d×d)
    b_a += r_a · x
    θ_a = A_a⁻¹ · b_a
    p_a = θ_aᵀ · x + α · sqrt(xᵀ · A_a⁻¹ · x)
Select arm = argmax_a p_a
```

**Composite reward:**

```
r = 0.4 · success + 0.3 · speed + 0.3 · safety
```

### Layer 1: Neural Process Kernel (Paper §7.1)

**Priority-based preemptive scheduling:**

```
Priority levels: CRITICAL > HIGH > NORMAL > LOW > BACKGROUND
Budget: {max_wall_time_s, max_tokens, max_memory_mb}
Preemption: if ready_queue.head.priority > running.priority → suspend running, run head
```

### Layer 4: Fabrication Engine (Paper §3)

**Intent classification:**

```
score(domain, intent) = Σ_{pattern ∈ domain.patterns} cosine_sim(embed(intent), embed(pattern))
                       × domain_weight × episodic_boost
```

**DAG fusion:**

```
DAG = topological_merge(∪_{d ∈ matched} domain_d.startup_sequence)
Consent level = max(consent_levels) across matched domains
```

## Project Structure

```
lwm-fabricator-model/
├── lwm_fab/                     # Main package
│   ├── __init__.py
│   ├── kernel.py                # LWMFabricator: orchestrates all 9 layers
│   ├── models.py                # Data models (DAG, nodes, transitions, verdicts)
│   ├── domain_registry.py       # 21 capability domain blueprints
│   ├── fabrication_engine.py    # Layer 4: classify() + fabricate() DAG synthesis
│   ├── world_model.py           # Layer 5: JEPA + Transformer predictor + CEM planning
│   ├── safety_gate.py           # Layer 6: AUQ + MACI LLM debate gate
│   ├── action_executor.py       # Layer 7: Grounded action execution
│   ├── rl_engine.py             # Layer 8: Step-level RL credit assignment
│   ├── neural_kernel.py         # Layer 1: Neural process kernel (scheduler)
│   ├── memory.py                # Layer 2: Memo + Graphiti memory management
│   ├── mcp_agent.py             # Layer 3: MCP agents (tools/resources/prompts protocol)
│   ├── telemetry_router.py      # Layer 9: LinUCB contextual bandit router
│   ├── ollama_integration.py    # LLM integration (Ollama + HF Inference API)
│   ├── proactive_engine.py      # Proactive simulation engine
│   └── database.py              # SQLite persistence
├── hf_spaces/                   # HuggingFace Spaces deployment
│   ├── app.py                   # Gradio UI
│   ├── requirements.txt
│   └── README.md                # HF Spaces config
├── lwm_fab_demo.ipynb           # Local Jupyter notebook demo (all layers)
├── lwm_fab_colab.ipynb          # Google Colab notebook (live LLM via Ollama)
├── run_service.py               # Persistent CLI service
├── test_run.py                  # Quick test script
├── .gitignore
├── requirements.txt
├── Capability of Fabrication os.pdf  # Original paper
└── README.md
```

## Quick Start

### Option 1: Local Jupyter Notebook

```bash
pip install -r requirements.txt
jupyter notebook lwm_fab_demo.ipynb
```

### Option 2: Persistent CLI Service

```bash
pip install -r requirements.txt
python run_service.py
# Type intents at the lwm-fab> prompt
```

### Option 3: Google Colab (Free T4 GPU + Live LLM)

1. Open `lwm_fab_colab.ipynb` in Colab
2. Select Runtime → GPU (T4)
3. Run all cells — installs Ollama, pulls qwen3:4b + qwen3:14b, runs full live system

### Option 4: HuggingFace Spaces (Persistent Hosting)

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select Gradio SDK
3. Upload `hf_spaces/` contents + `lwm_fab/` package
4. Set `HF_TOKEN` environment variable for LLM inference
5. Your app is live at `https://aravindh-dev12-lwm-fabricator.hf.space`

### Option 5: Live LLM (Local Ollama)

```bash
# Install Ollama: https://ollama.ai
ollama pull qwen3:4b    # reflex model (~3GB VRAM)
ollama pull qwen3:14b   # judge model (~9GB VRAM)
ollama serve
python run_service.py   # auto-detects Ollama
```

## Key Results

| Metric | Value |
|--------|-------|
| CEM Planning Time | <1s (vs 47.3s LLM text prediction) |
| Speedup | 48x |
| Latent Dimensions | 192 |
| State Dimensions | 64 |
| Transformer Predictor | depth=6, heads=16, MLP=2048, history=3 |
| CEM Candidates | 9,000 (300×30) |
| Capability Domains | 21 (18 functional + 3 infrastructure) |
| MCP Agents | 21 (tools + resources + prompts protocol) |
| RL Credit Assignment | GRPO-style, γ=0.95, w₁=1, w₂=2 |
| Safety Gate | AUQ calibration + MACI LLM debate |
| Telemetry Router | LinUCB 8D contextual bandit |
| Proactive Simulation | 1000 scenarios, vectorized |
| LLM Backend | Ollama (local) / HF Inference API (remote) auto-switch |

## 21 Capability Domains

**Functional (18):** app_fabrication, document_intelligence, knowledge_graph, finance_ops, growth_marketing, computer_use, realtime_web, voice_interface, data_analysis, career_agent, physical_tasks, formal_verification, agent_identity, autonomous_payments, content_recall, property_ops, comms_coverage, visual_generation

**Infrastructure (3):** log_compression, inference_optimization, agent_observability

## LLM Integration

The system uses a **HybridLLMClient** that auto-selects between:

| Backend | Models | Environment |
|---------|--------|-------------|
| Ollama (local) | qwen3:14b (judge) + qwen3:4b (reflex) | Colab, local GPU |
| HF Inference API | Qwen2.5-7B-Instruct (judge) + Qwen2.5-1.5B-Instruct (reflex) | HF Spaces, CPU |

Both backends expose the same interface: `judge()`, `reflex()`, `generate()`, `chat()`.

## License

Research project — June 2026.
