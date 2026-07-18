"""Unified MCP capability bus and safe workflow runtime."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib, json, os, subprocess, threading, time, urllib.request, uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set

class RiskLevel(str, Enum):
    READ="read"; WRITE="write"; PRIVILEGED="privileged"; IRREVERSIBLE="irreversible"
class RunState(str, Enum):
    PENDING="pending"; RUNNING="running"; WAITING_APPROVAL="waiting_approval"; COMPLETED="completed"; FAILED="failed"

@dataclass(frozen=True)
class Capability:
    name:str; server:str; description:str; input_schema:Mapping[str,Any]=field(default_factory=dict); risk:RiskLevel=RiskLevel.READ
@dataclass(frozen=True)
class WorkflowStep:
    step_id:str; capability:str; arguments:Mapping[str,Any]=field(default_factory=dict); depends_on:tuple[str,...]=()
    condition:Optional[str]=None; retries:int=0; retry_delay_seconds:float=0; continue_on_error:bool=False
@dataclass(frozen=True)
class Workflow:
    name:str; steps:tuple[WorkflowStep,...]; workflow_id:str=field(default_factory=lambda:str(uuid.uuid4()))
    @classmethod
    def from_dict(cls,d):
        return cls(d.get("name","untitled"),tuple(WorkflowStep(i["id"],i["capability"],i.get("arguments",{}),tuple(i.get("depends_on",())),i.get("condition"),max(0,int(i.get("retries",0))),max(0,float(i.get("retry_delay_seconds",0))),bool(i.get("continue_on_error",False))) for i in d.get("steps",())),d.get("workflow_id",str(uuid.uuid4())))
    def to_dict(self):
        return {"name":self.name,"workflow_id":self.workflow_id,"steps":[{"id":s.step_id,"capability":s.capability,"arguments":dict(s.arguments),"depends_on":list(s.depends_on),"condition":s.condition,"retries":s.retries,"retry_delay_seconds":s.retry_delay_seconds,"continue_on_error":s.continue_on_error} for s in self.steps]}
@dataclass
class StepResult:
    status:str; output:Any=None; error:Optional[str]=None
@dataclass
class WorkflowRun:
    run_id:str; workflow_id:str; state:RunState=RunState.PENDING; results:Dict[str,StepResult]=field(default_factory=dict); pending_approval:Optional[str]=None; approved_steps:Set[str]=field(default_factory=set)
    def to_dict(self): return {"run_id":self.run_id,"workflow_id":self.workflow_id,"state":self.state.value,"results":{k:asdict(v) for k,v in self.results.items()},"pending_approval":self.pending_approval,"approved_steps":sorted(self.approved_steps)}
    @classmethod
    def from_dict(cls,d): return cls(d["run_id"],d["workflow_id"],RunState(d["state"]),{k:StepResult(**v) for k,v in d.get("results",{}).items()},d.get("pending_approval"),set(d.get("approved_steps",[])))

class WorkflowValidationError(ValueError): pass
class SecretProvider:
    def get(self,name:str)->str: raise NotImplementedError
class EnvironmentSecretProvider(SecretProvider):
    def get(self,name):
        key="LWM_SECRET_"+name
        if key not in os.environ: raise KeyError(f"Secret is not configured: {name}")
        return os.environ[key]

class CapabilityBus:
    def __init__(self): self._caps={}; self._handlers={}; self._lock=threading.RLock()
    def register(self,cap,handler):
        with self._lock:
            if cap.name in self._caps: raise ValueError(f"Capability already registered: {cap.name}")
            self._caps[cap.name]=cap; self._handlers[cap.name]=handler
    def register_mcp_server(self,server,tools,caller,default_risk=RiskLevel.WRITE):
        for tool in tools:
            try: risk=RiskLevel(tool.get("annotations",{}).get("risk",default_risk.value))
            except ValueError: risk=default_risk
            name=f"{server}.{tool['name']}"; raw=tool["name"]
            self.register(Capability(name,server,tool.get("description",""),tool.get("inputSchema",{}),risk),lambda args,n=raw:caller(n,args))
    def discover(self): return [{**asdict(c),"risk":c.risk.value} for c in sorted(self._caps.values(),key=lambda x:x.name)]
    def get(self,name):
        if name not in self._caps: raise KeyError(f"Unknown capability: {name}")
        return self._caps[name]
    def call(self,name,args): return self._handlers[name](args)

class ApprovalPolicy:
    def __init__(self,require_for=None): self.require_for=set(require_for or {RiskLevel.WRITE,RiskLevel.PRIVILEGED,RiskLevel.IRREVERSIBLE})
    def requires_approval(self,cap): return cap.risk in self.require_for
class AuditLog:
    def __init__(self,path): self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self._lock=threading.Lock(); self.previous=self._last()
    def _last(self):
        if not self.path.exists(): return "0"*64
        lines=self.path.read_text().splitlines(); return json.loads(lines[-1])["hash"] if lines else "0"*64
    def append(self,event,payload):
        with self._lock:
            row={"timestamp":time.time(),"event":event,"payload":dict(payload),"previous_hash":self.previous}; canonical=json.dumps(row,sort_keys=True,default=str); row["hash"]=hashlib.sha256(canonical.encode()).hexdigest()
            with self.path.open("a") as f:f.write(json.dumps(row,sort_keys=True,default=str)+"\n")
            self.previous=row["hash"]; return row

class AutomationControlPlane:
    def __init__(self,bus,audit,policy=None,secrets=None): self.bus=bus; self.audit=audit; self.policy=policy or ApprovalPolicy(); self.secrets=secrets or EnvironmentSecretProvider(); self._runs={}; self._workflows={}
    def validate(self,w):
        ids=[s.step_id for s in w.steps]
        if not ids or len(ids)!=len(set(ids)): raise WorkflowValidationError("Workflow requires unique steps")
        graph={s.step_id:s.depends_on for s in w.steps}; known=set(ids); visiting=set(); visited=set()
        for s in w.steps:
            self.bus.get(s.capability)
            if set(s.depends_on)-known: raise WorkflowValidationError(f"Unknown dependency in {s.step_id}")
        def visit(i):
            if i in visiting: raise WorkflowValidationError("Workflow contains a cycle")
            if i in visited:return
            visiting.add(i)
            for d in graph[i]:visit(d)
            visiting.remove(i);visited.add(i)
        for i in ids:visit(i)
    def start(self,w):
        self.validate(w); r=WorkflowRun(str(uuid.uuid4()),w.workflow_id); self._runs[r.run_id]=r;self._workflows[w.workflow_id]=w;self.audit.append("workflow.started",{"run_id":r.run_id});return self._advance(r,w)
    def approve(self,run_id,step_id):
        r=self._runs[run_id]
        if r.pending_approval!=step_id:raise ValueError("Step is not awaiting approval")
        r.approved_steps.add(step_id);r.pending_approval=None;self.audit.append("step.approved",{"run_id":run_id,"step_id":step_id});return self._advance(r,self._workflows[r.workflow_id])
    def restore(self,w,r): self.validate(w);self._runs[r.run_id]=r;self._workflows[w.workflow_id]=w
    def get_run(self,i):return self._runs[i]
    def _lookup(self,e,r):
        p=e.split("."); result=r.results[p[1]]; value=result.status if p[2]=="status" else result.output
        for k in p[3:]:value=value[int(k)] if isinstance(value,list) else value[k]
        return value
    def _resolve(self,v,r):
        if isinstance(v,Mapping):return {k:self._resolve(x,r) for k,x in v.items()}
        if isinstance(v,list):return [self._resolve(x,r) for x in v]
        if isinstance(v,str) and v.startswith("${{") and v.endswith("}}"):
            e=v[3:-2].strip();return self.secrets.get(e[8:]) if e.startswith("secrets.") else self._lookup(e,r)
        return v
    def _condition(self,e,r):
        e=e.removeprefix("${{").removesuffix("}}").strip()
        if " == " in e:
            l,x=e.split(" == ",1);return self._lookup(l.strip(),r)==json.loads(x)
        return bool(self._lookup(e,r))
    def _advance(self,r,w):
        r.state=RunState.RUNNING
        while len(r.results)<len(w.steps):
            ready=[s for s in w.steps if s.step_id not in r.results and all(d in r.results and r.results[d].status in ("completed","skipped","continued") for d in s.depends_on)]
            if not ready:r.state=RunState.FAILED;return r
            for s in ready:
                cap=self.bus.get(s.capability)
                if s.condition and not self._condition(s.condition,r):r.results[s.step_id]=StepResult("skipped");continue
                if self.policy.requires_approval(cap) and s.step_id not in r.approved_steps:r.pending_approval=s.step_id;r.state=RunState.WAITING_APPROVAL;self.audit.append("step.approval_required",{"run_id":r.run_id,"step_id":s.step_id,"risk":cap.risk.value});return r
                err=None
                for attempt in range(s.retries+1):
                    try:r.results[s.step_id]=StepResult("completed",self.bus.call(s.capability,self._resolve(s.arguments,r)));err=None;break
                    except Exception as exc:err=exc;time.sleep(s.retry_delay_seconds if attempt<s.retries else 0)
                if err:
                    r.results[s.step_id]=StepResult("continued" if s.continue_on_error else "failed",error=str(err))
                    if not s.continue_on_error:r.state=RunState.FAILED;return r
        r.state=RunState.COMPLETED;self.audit.append("workflow.completed",{"run_id":r.run_id});return r

def register_builtin_linux_capabilities(bus,workspace):
    root=Path(workspace).resolve();root.mkdir(parents=True,exist_ok=True)
    def resolve(p):
        x=(root/p).resolve()
        if x!=root and root not in x.parents:raise ValueError("Path escapes workspace")
        return x
    bus.register(Capability("linux.files.read","builtin-linux","Read workspace file",risk=RiskLevel.READ),lambda a:{"path":str(resolve(a["path"])),"content":resolve(a["path"]).read_text()})
    def write(a):p=resolve(a["path"]);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(str(a.get("content","")));return {"path":str(p),"bytes":p.stat().st_size}
    bus.register(Capability("linux.files.write","builtin-linux","Write workspace file",risk=RiskLevel.WRITE),write)
    bus.register(Capability("linux.files.list","builtin-linux","List workspace files",risk=RiskLevel.READ),lambda a:{"entries":[str(p.relative_to(root)) for p in resolve(a.get("path",".")).iterdir()]})
    bus.register(Capability("linux.system.info","builtin-linux","Read Linux system information",risk=RiskLevel.READ),lambda a:{"system":os.uname().sysname,"release":os.uname().release,"machine":os.uname().machine,"cpu_count":os.cpu_count()})
    def command(a):
        cmd=a["command"]
        if not isinstance(cmd,list) or not cmd or cmd[0] not in {"git","python","python3","node","npm","pwd","ls","find","rg","wc","head","tail"}:raise ValueError("Command is not allowed")
        p=subprocess.run(cmd,cwd=root,capture_output=True,text=True,timeout=min(float(a.get("timeout",30)),120),shell=False);return {"returncode":p.returncode,"stdout":p.stdout[:100000],"stderr":p.stderr[:100000]}
    bus.register(Capability("linux.exec.run","builtin-linux","Run allowlisted command",risk=RiskLevel.PRIVILEGED),command)
