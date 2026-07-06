"""LWM Fabricator — Persistent Service Script.
Keeps the kernel alive, accepts intents via CLI, runs proactive simulations periodically.
Works with Ollama (local) or HF Inference API (remote)."""
import sys
import os
import time
import json
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"), exist_ok=True)

from lwm_fab.kernel import LWMFabricator


def proactive_loop(kernel: LWMFabricator, interval: float = 3600):
    """Background thread: run proactive simulation every `interval` seconds."""
    while True:
        time.sleep(interval)
        try:
            result = kernel.run_proactive_simulation([0.5] * 64, [0.8] * 64)
            print(f"\n[Proactive] {result['num_scenarios']} scenarios | "
                  f"confidence={result['confidence']:.2%} | "
                  f"action={result['recommended_action']}")
        except Exception as e:
            print(f"\n[Proactive] Error: {e}")


def main():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "lwm_fab.db")

    kernel = LWMFabricator(
        mode="dry_run",
        db_path=db_path,
        hf_api_key=os.environ.get("HF_TOKEN", ""),
    )

    # Print system status
    stats = kernel._system_stats()
    print("=" * 60)
    print("LWM Fabricator — Persistent Service")
    print("=" * 60)
    print(f"Mode: {kernel.mode}")
    print(f"Domains: {len(stats.get('mcp_agents', []))} MCP agents")

    wm = stats.get("world_model", {})
    print(f"LeWM: {wm.get('predictor', 'N/A')} d={wm.get('predictor_depth')} "
          f"h={wm.get('predictor_heads')} MLP={wm.get('predictor_mlp_dim')} "
          f"H_ctx={wm.get('history_size')}")

    ol = stats.get("ollama", {})
    print(f"LLM: backend={ol.get('active_backend', 'none')} "
          f"available={ol.get('available', False)}")
    if ol.get("available"):
        print(f"  Judge: {ol.get('judge_model')}")
        print(f"  Reflex: {ol.get('reflex_model')}")

    pe = stats.get("proactive_engine", {})
    print(f"Proactive: scenarios={pe.get('num_scenarios', 0)} "
          f"interval={pe.get('interval_seconds', 0)}s")

    print("=" * 60)
    print("\nCommands:")
    print("  <intent>     — Run full 9-layer pipeline on intent")
    print("  proactive    — Run proactive simulation now")
    print("  stats        — Show system stats")
    print("  mode <mode>  — Switch mode (dry_run / live)")
    print("  quit         — Exit")
    print()

    # Start proactive simulation thread
    proactive_thread = threading.Thread(
        target=proactive_loop, args=(kernel, 3600), daemon=True
    )
    proactive_thread.start()

    while True:
        try:
            user_input = input("lwm-fab> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        elif user_input == "quit":
            break
        elif user_input == "stats":
            s = kernel._system_stats()
            print(json.dumps(s, indent=2, default=str))
        elif user_input == "proactive":
            result = kernel.run_proactive_simulation([0.5] * 64, [0.8] * 64)
            print(f"Scenarios: {result['num_scenarios']}")
            print(f"Mean cost: {result['mean_cost']:.2f}")
            print(f"Min cost: {result['min_cost']:.2f}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Action: {result['recommended_action']}")
        elif user_input.startswith("mode "):
            new_mode = user_input[5:].strip()
            if new_mode in ("dry_run", "live"):
                kernel.mode = new_mode
                kernel.executor.mode = new_mode
                print(f"Mode set to: {new_mode}")
            else:
                print(f"Invalid mode: {new_mode} (use dry_run or live)")
        else:
            # Treat as intent
            print(f"\nProcessing: \"{user_input}\"")
            t0 = time.time()
            result = kernel.process_intent(user_input)
            elapsed = time.time() - t0

            print(f"\nRun: {result['run_id'][:8]}")
            print(f"Status: {result['final_status']}")
            print(f"Domains: {result['matched_domains']}")
            print(f"Nodes: {len(result['dag']['nodes'])}")
            print(f"Time: {elapsed:.2f}s")

            print("\nPipeline:")
            for step in result["pipeline_log"]:
                e = step.get("elapsed_ms", "—")
                print(f"  {step['step']:<25} {e}ms" if isinstance(e, (int, float))
                      else f"  {step['step']:<25} {e}")

            print("\nNodes:")
            for nr in result["node_results"]:
                icon = "✓" if nr.get("status") == "success" else "⏸" if nr.get("status") == "paused" else "✗"
                sg = f" [gate: {nr['safety_gate']['verdict']}]" if "safety_gate" in nr else ""
                print(f"  {icon} {nr['node_id']}: {nr.get('status')} — {nr.get('detail', '')[:60]}{sg}")

            print(f"\nRL Bottleneck: {result['rl_bottleneck']}")

    kernel.close()
    print("\nLWM Fabricator shut down.")


if __name__ == "__main__":
    main()
