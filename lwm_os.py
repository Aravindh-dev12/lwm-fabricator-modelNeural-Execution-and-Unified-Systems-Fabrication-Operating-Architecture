import argparse,json
from pathlib import Path
from lwm_fab.control_plane import *
from lwm_fab.mcp import MCPServerManager
def runtime(args):
 b=CapabilityBus();register_builtin_linux_capabilities(b,args.workspace);m=MCPServerManager(b)
 if args.mcp_config:m.load_config(args.mcp_config)
 return AutomationControlPlane(b,AuditLog(Path(args.state_dir)/"audit.jsonl")),m
def main():
 p=argparse.ArgumentParser();p.add_argument("--workspace",default=".");p.add_argument("--state-dir",default=".lwm-os");p.add_argument("--mcp-config");s=p.add_subparsers(dest="cmd",required=True);s.add_parser("capabilities");r=s.add_parser("run");r.add_argument("workflow");a=s.add_parser("approve");a.add_argument("run_id");a.add_argument("step_id");args=p.parse_args();rt,m=runtime(args)
 try:
  if args.cmd=="capabilities":print(json.dumps(rt.bus.discover(),indent=2));return
  runs=Path(args.state_dir)/"runs";runs.mkdir(parents=True,exist_ok=True)
  if args.cmd=="run":w=Workflow.from_dict(json.loads(Path(args.workflow).read_text()));x=rt.start(w)
  else:
   d=json.loads((runs/f"{args.run_id}.json").read_text());w=Workflow.from_dict(d["workflow"]);x=WorkflowRun.from_dict(d["run"]);rt.restore(w,x);x=rt.approve(args.run_id,args.step_id)
  (runs/f"{x.run_id}.json").write_text(json.dumps({"workflow":w.to_dict(),"run":x.to_dict()},indent=2));print(json.dumps(x.to_dict(),indent=2))
 finally:m.close()
if __name__=="__main__":main()
