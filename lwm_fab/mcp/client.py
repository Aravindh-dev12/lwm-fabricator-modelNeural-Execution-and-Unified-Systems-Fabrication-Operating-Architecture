from abc import ABC,abstractmethod
import json,subprocess,threading
class MCPError(RuntimeError):pass
class MCPClient(ABC):
 @abstractmethod
 def initialize(self):...
 @abstractmethod
 def list_tools(self):...
 @abstractmethod
 def call_tool(self,name,arguments):...
 @abstractmethod
 def close(self):...
class StdioMCPClient(MCPClient):
 def __init__(self,command,env=None,cwd=None):self.command=list(command);self.env=env;self.cwd=cwd;self.p=None;self.i=1;self.lock=threading.Lock()
 def start(self):
  if not self.p or self.p.poll() is not None:self.p=subprocess.Popen(self.command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1,env=self.env,cwd=self.cwd)
 def request(self,method,params=None):
  self.start()
  with self.lock:
   i=self.i;self.i+=1;self.p.stdin.write(json.dumps({"jsonrpc":"2.0","id":i,"method":method,"params":params or {}})+"\n");self.p.stdin.flush()
   for line in self.p.stdout:
    r=json.loads(line)
    if r.get("id")==i:
     if "error" in r:raise MCPError(str(r["error"]))
     return r.get("result")
   raise MCPError("MCP server stopped")
 def initialize(self):return self.request("initialize",{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"lwm-fabricator-os","version":"0.2.0"}})
 def list_tools(self):return self.request("tools/list").get("tools",[])
 def call_tool(self,name,arguments):return self.request("tools/call",{"name":name,"arguments":dict(arguments)})
 def close(self):
  if self.p:self.p.terminate();self.p.wait(timeout=2);[x.close() for x in (self.p.stdin,self.p.stdout,self.p.stderr) if x];self.p=None
