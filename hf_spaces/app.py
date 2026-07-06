"""LWM Fabricator — HuggingFace Spaces App (Gradio UI)
Runs all 9 agentic layers with HF Inference API fallback for LLM.
No local Ollama needed — uses Qwen models via HF Inference API."""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import torch
import numpy as np

from lwm_fab.kernel import LWMFabricator
from lwm_fab.ollama_integration import HybridLLMClient
from lwm_fab.domain_registry import DOMAINS
from lwm_fab.models import DAGNode, ActionType, ActionVerb, ConsentLevel

os.makedirs("data", exist_ok=True)

_kernel = None


def get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = LWMFabricator(
            mode="dry_run",
            db_path="data/lwm_fab.db",
            hf_api_key=os.environ.get("HF_TOKEN", ""),
        )
    return _kernel


def run_intent(intent: str, mode: str):
    """Run full 9-layer pipeline on user intent."""
    kernel = get_kernel()
    kernel.mode = mode
    kernel.executor.mode = mode

    result = kernel.process_intent(intent)

    # Format output
    lines = []
    lines.append(f"## Run: {result['run_id'][:8]}")
    lines.append(f"**Status:** {result['final_status']}")
    lines.append(f"**Domains:** {', '.join(d for d, s in result['matched_domains'])}")
    lines.append(f"**DAG Nodes:** {len(result['dag']['nodes'])}")
    lines.append("")

    lines.append("### Pipeline Trace")
    for step in result["pipeline_log"]:
        e = step.get("elapsed_ms", "—")
        if isinstance(e, (int, float)):
            lines.append(f"- `{step['step']}` — {e}ms")
        else:
            lines.append(f"- `{step['step']}` — {e}")
    lines.append("")

    lines.append("### Node Results")
    for nr in result["node_results"]:
        icon = "✅" if nr.get("status") == "success" else "⏸️" if nr.get("status") == "paused" else "❌"
        sg = f" **Gate:** {nr['safety_gate']['verdict']}" if "safety_gate" in nr else ""
        code = f"\n  ```python\n  {nr['generated_code'][:200]}\n  ```" if "generated_code" in nr else ""
        lines.append(f"{icon} `{nr['node_id']}` — {nr.get('status', '?')} — {nr.get('detail', '')[:60]}{sg}{code}")
    lines.append("")

    lines.append(f"### RL Bottleneck")
    bn = result["rl_bottleneck"]
    lines.append(f"- Bottleneck step: {bn.get('bottleneck_step', 'N/A')}")
    lines.append(f"- Mean advantage (failing): {bn.get('mean_advantage_failing', 'N/A')}")
    lines.append("")

    # System stats
    stats = result.get("system_stats", {})
    lines.append("### System Stats")
    if "world_model" in stats:
        wm = stats["world_model"]
        lines.append(f"**LeWM:** Transformer d={wm['predictor_depth']}, h={wm['predictor_heads']}, MLP={wm['predictor_mlp_dim']}, H_ctx={wm['history_size']}")
    if "ollama" in stats:
        ol = stats["ollama"]
        lines.append(f"**LLM:** backend={ol.get('active_backend', 'none')}, available={ol.get('available', False)}")
    if "proactive_engine" in stats:
        pe = stats["proactive_engine"]
        lines.append(f"**Proactive:** scenarios={pe['num_scenarios']}, running={pe['running']}")
    if "mcp_agents" in stats:
        agents = stats["mcp_agents"]
        lines.append(f"**MCP Agents:** {len(agents)} registered")

    return "\n".join(lines)


def run_proactive():
    """Run proactive simulation."""
    kernel = get_kernel()
    result = kernel.run_proactive_simulation([0.5] * 64, [0.8] * 64)
    lines = ["## Proactive Simulation (1000 Scenarios)"]
    lines.append(f"**Scenarios:** {result['num_scenarios']}")
    lines.append(f"**Mean cost:** {result['mean_cost']:.2f}")
    lines.append(f"**Min cost:** {result['min_cost']:.2f}")
    lines.append(f"**Confidence:** {result['confidence']:.2%}")
    lines.append(f"**Recommended action:** `{result['recommended_action']}`")
    lines.append("")
    lines.append(">95% → auto | 60-80% → prompt user | <60% → hold")
    return "\n".join(lines)


def run_safety_gate(action_type: str, verb: str, consent: str, c_stated: float, description: str):
    """Run safety gate evaluation on a single action."""
    kernel = get_kernel()
    from oeon_os.safety_gate import SafetyGate

    gate = kernel.safety_gate
    node = DAGNode(
        node_id="demo",
        action_type=ActionType(action_type),
        verb=ActionVerb(verb),
        params={"description": description},
        consent_level=ConsentLevel(consent),
    )
    result = gate.evaluate(node, c_stated=c_stated)

    lines = ["## Safety Gate Evaluation"]
    lines.append(f"**Action:** `{action_type}:{verb}` — {description}")
    lines.append(f"**C_stated:** {c_stated}")
    lines.append("")
    lines.append("### AUQ")
    auq = result["auq"]
    lines.append(f"- p_worst: {auq['p_worst']}")
    lines.append(f"- r_risk: {auq['r_risk']}")
    lines.append(f"- delta_cal: {auq['delta_cal']}")
    lines.append(f"- triggers_debate: {auq['triggers_debate']}")
    lines.append("")
    lines.append("### MACI Verdict")
    lines.append(f"**Verdict:** `{result['verdict']}`")
    lines.append(f"**C_judge:** {result['c_judge']}")
    lines.append(f"**R_residual:** {result['r_residual']}")
    lines.append(f"**Reasoning:** {result['reasoning']}")
    if result.get("modifications"):
        lines.append(f"**Modifications:** {result['modifications']}")
    return "\n".join(lines)


def get_system_info():
    """Get system info."""
    kernel = get_kernel()
    stats = kernel._system_stats()
    lines = ["## LWM Fabricator — System Info"]

    lines.append(f"**Domains:** {len(DOMAINS)}")
    lines.append(f"**Mode:** {kernel.mode}")
    lines.append("")

    if "world_model" in stats:
        wm = stats["world_model"]
        lines.append(f"### LeWM World Model")
        lines.append(f"- Predictor: {wm['predictor']} (d={wm['predictor_depth']}, h={wm['predictor_heads']}, MLP={wm['predictor_mlp_dim']})")
        lines.append(f"- Latent dim: {wm['latent_dim']}, History: {wm['history_size']}")
        lines.append(f"- CEM: {wm['cem_samples']} samples × {wm['cem_iterations']} iters, horizon={wm['horizon']}")

    if "ollama" in stats:
        ol = stats["ollama"]
        lines.append(f"\n### LLM Integration")
        lines.append(f"- Available: {ol.get('available', False)}")
        lines.append(f"- Backend: {ol.get('active_backend', 'none')}")
        lines.append(f"- Judge: {ol.get('judge_model', 'N/A')}")
        lines.append(f"- Reflex: {ol.get('reflex_model', 'N/A')}")

    if "mcp_agents" in stats:
        agents = stats["mcp_agents"]
        lines.append(f"\n### MCP Agents: {len(agents)}")
        for a in agents[:5]:
            lines.append(f"- `{a['name']}` — tools={a['tools']}, resources={a['resources']}, prompts={a['prompts']}")
        if len(agents) > 5:
            lines.append(f"- ... and {len(agents) - 5} more")

    if "neural_kernel" in stats:
        npk = stats["neural_kernel"]
        lines.append(f"\n### Neural Process Kernel: {npk}")

    if "memory" in stats:
        mem = stats["memory"]
        lines.append(f"\n### Memory: {mem}")

    if "proactive_engine" in stats:
        pe = stats["proactive_engine"]
        lines.append(f"\n### Proactive Engine: {pe}")

    if "telemetry" in stats:
        tel = stats["telemetry"]
        lines.append(f"\n### Telemetry Router: routes={tel.get('total_routes', 0)}")

    return "\n".join(lines)


# Build Gradio UI
with gr.Blocks(title="LWM Fabricator", theme=gr.themes.Soft()) as app:
    gr.Markdown("# LWM Fabricator — Capability Fabrication via Latent World Models")
    gr.Markdown("### 9 Agentic Layers: NPK → Memory → MCP → Fabrication → LeWM → Safety Gate → Executor → RL → Telemetry")

    with gr.Tab("Process Intent"):
        intent_input = gr.Textbox(
            label="Intent",
            placeholder="e.g., Build a landing page and set up email campaigns",
            lines=2,
        )
        mode_select = gr.Radio(["dry_run", "live"], value="dry_run", label="Execution Mode")
        run_btn = gr.Button("Run Pipeline", variant="primary")
        output = gr.Markdown(label="Result")
        run_btn.click(run_intent, inputs=[intent_input, mode_select], outputs=output)

    with gr.Tab("Safety Gate"):
        with gr.Row():
            atype = gr.Dropdown(["file", "shell", "network", "code"], value="shell", label="Action Type")
            verb = gr.Dropdown(
                ["write", "read", "delete", "execute", "request", "evaluate"],
                value="execute", label="Verb",
            )
            consent = gr.Dropdown(["standard", "elevated", "never"], value="elevated", label="Consent Level")
            c_stated = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="C_stated (confidence)")
        desc = gr.Textbox(label="Description", value="Run rm -rf /tmp/cache", lines=1)
        gate_btn = gr.Button("Evaluate", variant="primary")
        gate_output = gr.Markdown(label="Safety Gate Result")
        gate_btn.click(
            run_safety_gate,
            inputs=[atype, verb, consent, c_stated, desc],
            outputs=gate_output,
        )

    with gr.Tab("Proactive Simulation"):
        psim_btn = gr.Button("Run 1000-Scenario Simulation", variant="primary")
        psim_output = gr.Markdown(label="Simulation Result")
        psim_btn.click(run_proactive, outputs=psim_output)

    with gr.Tab("System Info"):
        info_btn = gr.Button("Refresh System Info", variant="primary")
        info_output = gr.Markdown(label="System Info")
        info_btn.click(get_system_info, outputs=info_output)

    gr.Markdown("---")
    gr.Markdown("*LWM Fabricator — On-Demand Synthesis and Safe Execution of AI OS Capabilities*")


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
