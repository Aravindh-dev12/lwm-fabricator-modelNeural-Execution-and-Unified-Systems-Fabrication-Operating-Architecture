"""MCP (Model Context Protocol) Agent Layer.
Each MCP agent wraps a capability domain and exposes tools/resources to the kernel.
Per paper: agents interact through the neural process kernel and telemetry router."""
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class MCPResourceType(str, Enum):
    TOOL = "tool"
    RESOURCE = "resource"
    PROMPT = "prompt"


@dataclass
class MCPTool:
    """A tool exposed by an MCP agent."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable
    consent_level: str = "standard"


@dataclass
class MCPResource:
    """A resource exposed by an MCP agent (e.g., a file, a knowledge base)."""
    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"
    content: Any = None


@dataclass
class MCPPrompt:
    """A prompt template exposed by an MCP agent (MCP protocol: prompts/list, prompts/get).
    Per MCP spec: prompts are reusable message templates that clients can invoke."""
    name: str
    description: str
    template: str
    arguments: List[Dict[str, Any]] = field(default_factory=list)


class MCPProtocolHandler:
    """Handles MCP JSON-RPC-style protocol messages.
    Implements: tools/list, tools/call, resources/list, resources/read,
    prompts/list, prompts/get, prompts/invoke."""

    def __init__(self, agent: "MCPAgent"):
        self.agent = agent

    def handle(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Route a protocol method to the appropriate handler."""
        if method == "tools/list":
            return {"tools": self.agent.list_tools()}
        elif method == "tools/call":
            return self.agent.call_tool(params.get("name", ""), params.get("arguments", {}))
        elif method == "resources/list":
            return {"resources": self.agent.list_resources()}
        elif method == "resources/read":
            content = self.agent.read_resource(params.get("uri", ""))
            return {"contents": content} if content is not None else {"error": "Resource not found"}
        elif method == "prompts/list":
            return {"prompts": self.agent.list_prompts()}
        elif method == "prompts/get":
            prompt = self.agent.get_prompt(params.get("name", ""))
            return {"template": prompt.template, "arguments": prompt.arguments} if prompt else {"error": "Prompt not found"}
        elif method == "prompts/invoke":
            return self.agent.invoke_prompt(params.get("name", ""), params.get("arguments", {}))
        elif method == "agent/info":
            return self.agent.info()
        else:
            return {"error": f"Unknown method: {method}"}


@dataclass
class MCPAgent:
    """An MCP agent wrapping a capability domain.
    Exposes tools, resources, and prompts to the LWM Fabricator kernel via Model Context Protocol.
    Per paper: agents interact through the neural process kernel and telemetry router."""
    agent_id: str
    name: str
    domain: str
    tools: List[MCPTool] = field(default_factory=list)
    resources: List[MCPResource] = field(default_factory=list)
    prompts: List[MCPPrompt] = field(default_factory=list)
    status: str = "idle"  # idle, busy, error
    capabilities: List[str] = field(default_factory=list)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.agent_id:
            self.agent_id = f"mcp_{self.domain}_{uuid.uuid4().hex[:6]}"
        self.protocol = MCPProtocolHandler(self)

    def register_tool(self, tool: MCPTool):
        self.tools.append(tool)

    def register_resource(self, resource: MCPResource):
        self.resources.append(resource)

    def register_prompt(self, prompt: MCPPrompt):
        self.prompts.append(prompt)

    def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool exposed by this agent."""
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            return {"status": "error", "detail": f"Tool '{tool_name}' not found"}

        self.status = "busy"
        start = time.time()
        try:
            result = tool.handler(params)
            elapsed = time.time() - start
            entry = {
                "tool": tool_name,
                "params": params,
                "result": result,
                "elapsed_s": round(elapsed, 4),
                "timestamp": time.time(),
                "status": "success",
            }
            self.execution_history.append(entry)
            self.status = "idle"
            return {"status": "success", "result": result, "elapsed_s": round(elapsed, 4)}
        except Exception as e:
            elapsed = time.time() - start
            entry = {
                "tool": tool_name,
                "params": params,
                "error": str(e),
                "elapsed_s": round(elapsed, 4),
                "timestamp": time.time(),
                "status": "error",
            }
            self.execution_history.append(entry)
            self.status = "error"
            return {"status": "error", "detail": str(e)}

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools (MCP protocol: tools/list)."""
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema,
             "consent_level": t.consent_level}
            for t in self.tools
        ]

    def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources (MCP protocol: resources/list)."""
        return [
            {"uri": r.uri, "name": r.name, "description": r.description, "mime_type": r.mime_type}
            for r in self.resources
        ]

    def read_resource(self, uri: str) -> Any:
        """Read a resource (MCP protocol: resources/read)."""
        res = next((r for r in self.resources if r.uri == uri), None)
        if res:
            return res.content
        return None

    def list_prompts(self) -> List[Dict[str, Any]]:
        """List available prompts (MCP protocol: prompts/list)."""
        return [
            {"name": p.name, "description": p.description, "arguments": p.arguments}
            for p in self.prompts
        ]

    def get_prompt(self, name: str) -> Optional[MCPPrompt]:
        """Get a prompt by name (MCP protocol: prompts/get)."""
        return next((p for p in self.prompts if p.name == name), None)

    def invoke_prompt(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a prompt template with arguments (MCP protocol: prompts/invoke)."""
        prompt = self.get_prompt(name)
        if not prompt:
            return {"error": f"Prompt '{name}' not found"}
        try:
            rendered = prompt.template.format(**arguments)
            return {"messages": [{"role": "user", "content": rendered}]}
        except KeyError as e:
            return {"error": f"Missing argument: {e}"}

    def info(self) -> Dict[str, Any]:
        """Get agent info (MCP protocol: agent/info)."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "domain": self.domain,
            "status": self.status,
            "capabilities": self.capabilities,
            "tools_count": len(self.tools),
            "resources_count": len(self.resources),
            "prompts_count": len(self.prompts),
            "executions": len(self.execution_history),
        }


class MCPAgentRegistry:
    """Registry of MCP agents. Each capability domain gets its own agent.
    The kernel discovers agents and routes tasks through the telemetry router."""

    def __init__(self):
        self.agents: Dict[str, MCPAgent] = {}

    def register(self, agent: MCPAgent):
        self.agents[agent.agent_id] = agent

    def get_by_domain(self, domain: str) -> Optional[MCPAgent]:
        """Get the MCP agent for a capability domain."""
        return next((a for a in self.agents.values() if a.domain == domain), None)

    def get(self, agent_id: str) -> Optional[MCPAgent]:
        return self.agents.get(agent_id)

    def list_all(self) -> List[Dict[str, Any]]:
        return [
            {
                "agent_id": a.agent_id,
                "name": a.name,
                "domain": a.domain,
                "status": a.status,
                "tools": len(a.tools),
                "resources": len(a.resources),
                "prompts": len(a.prompts),
                "executions": len(a.execution_history),
            }
            for a in self.agents.values()
        ]


def _make_code_tool(description: str) -> MCPTool:
    """Factory for code:eval tools."""
    def handler(params: Dict) -> Dict:
        code = params.get("code", params.get("description", ""))
        local_ns: Dict[str, Any] = {}
        try:
            exec(code, {"__builtins__": __builtins__}, local_ns)
            return {"output": {k: str(v)[:200] for k, v in local_ns.items() if not k.startswith("_")}}
        except Exception as e:
            return {"error": str(e)}
    return MCPTool(
        name="evaluate_code",
        description=description,
        input_schema={"type": "object", "properties": {"code": {"type": "string"}}},
        handler=handler,
        consent_level="standard",
    )


def _make_file_tool(verb: str, description: str) -> MCPTool:
    """Factory for file tools."""
    def handler(params: Dict) -> Dict:
        path = params.get("path", "output.txt")
        import os
        if verb == "read":
            if os.path.exists(path):
                with open(path, "r") as f:
                    return {"content": f.read()[:500]}
            return {"error": "File not found"}
        elif verb == "write":
            content = params.get("content", "")
            with open(path, "w") as f:
                f.write(content)
            return {"bytes_written": len(content)}
        return {"error": f"Unknown verb: {verb}"}
    return MCPTool(
        name=f"file_{verb}",
        description=description,
        input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}},
        handler=handler,
        consent_level="standard" if verb == "read" else "elevated",
    )


def _make_shell_tool(description: str) -> MCPTool:
    """Factory for shell tools."""
    def handler(params: Dict) -> Dict:
        import subprocess
        cmd = params.get("command", "")
        ALLOWED = {"echo", "ls", "dir", "cat", "python", "node", "git", "pip"}
        base = cmd.split()[0].lower() if cmd else ""
        if base not in ALLOWED:
            return {"error": f"Command '{base}' not in allowlist"}
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return {"stdout": r.stdout[:500], "returncode": r.returncode}
    return MCPTool(
        name="execute_shell",
        description=description,
        input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
        handler=handler,
        consent_level="elevated",
    )


def _make_network_tool(description: str) -> MCPTool:
    """Factory for network tools."""
    def handler(params: Dict) -> Dict:
        import requests
        url = params.get("url", "https://httpbin.org/get")
        method = params.get("method", "GET")
        try:
            r = requests.request(method, url, timeout=15)
            return {"status_code": r.status_code, "body": r.text[:500]}
        except Exception as e:
            return {"error": str(e)}
    return MCPTool(
        name="network_request",
        description=description,
        input_schema={"type": "object", "properties": {"url": {"type": "string"}, "method": {"type": "string"}}},
        handler=handler,
        consent_level="elevated",
    )


def create_mcp_agents_for_domains(domains: List[str]) -> List[MCPAgent]:
    """Create MCP agents for a list of capability domains, each with appropriate tools."""
    from .domain_registry import get_domain_registry
    registry = get_domain_registry()
    agents = []

    for domain_name in domains:
        domain = registry.get(domain_name)
        if not domain:
            continue

        agent = MCPAgent(
            agent_id="",
            name=f"{domain_name}_agent",
            domain=domain_name,
            capabilities=domain.required_capabilities,
        )

        # Register tools based on domain's execution grammar
        for step in domain.execution_grammar:
            action_type = step["action_type"]
            verb = step["verb"]
            desc = step["description"]

            if action_type.value == "code":
                agent.register_tool(_make_code_tool(desc))
            elif action_type.value == "file":
                agent.register_tool(_make_file_tool(verb.value, desc))
            elif action_type.value == "shell":
                agent.register_tool(_make_shell_tool(desc))
            elif action_type.value == "network":
                agent.register_tool(_make_network_tool(desc))

        # Register a resource for the domain's knowledge
        agent.register_resource(MCPResource(
            uri=f"lwm://domain/{domain_name}",
            name=f"{domain_name}_knowledge",
            description=f"Knowledge base for {domain_name}",
            content={"patterns": domain.intent_patterns, "startups": domain.source_startups},
        ))

        # Register a prompt template for domain-specific fabrication
        agent.register_prompt(MCPPrompt(
            name=f"fabricate_{domain_name}",
            description=f"Generate a fabrication plan for {domain_name}",
            template="You are an agent for the {domain} capability domain.\nIntent: {intent}\nGenerate a step-by-step execution plan using the available tools.",
            arguments=[
                {"name": "intent", "description": "The user intent to fabricate", "required": True},
            ],
        ))

        agents.append(agent)

    return agents
