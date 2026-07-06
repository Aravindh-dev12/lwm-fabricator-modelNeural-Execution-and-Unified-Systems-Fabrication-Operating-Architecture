import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lwm_fab.kernel import LWMFabricator

db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "lwm_fab.db")
k = LWMFabricator(mode="dry_run", db_path=db)
r = k.process_intent("Build a landing page and set up email campaigns")
print(f"Status: {r['final_status']}")
print(f"Domains: {r['matched_domains']}")
print(f"Nodes: {len(r['dag']['nodes'])}")
print(f"Pipeline: {[s['step'] for s in r['pipeline_log']]}")
stats = r["system_stats"]
print(f"Stats keys: {list(stats.keys())}")
print(f"World model: {stats.get('world_model', {})}")
print(f"LLM: {stats.get('ollama', {})}")

psim = k.run_proactive_simulation([0.5]*64, [0.8]*64)
print(f"\nProactive simulation: {psim}")

k.close()
print("OK — lwm_fab package works")
