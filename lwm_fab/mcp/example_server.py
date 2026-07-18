import json,sys
TOOLS=[{"name":"text_transform","description":"Transform text case","inputSchema":{"type":"object"},"annotations":{"risk":"read"}}]
for line in sys.stdin:
 r=json.loads(line);m=r.get("method");result={"protocolVersion":"2025-03-26","capabilities":{"tools":{}},"serverInfo":{"name":"lwm-example","version":"0.1"}} if m=="initialize" else {"tools":TOOLS} if m=="tools/list" else {"content":[{"type":"text","text":str(r["params"]["arguments"].get("text","")).upper() if r["params"]["arguments"].get("operation")=="upper" else str(r["params"]["arguments"].get("text","")).lower()}]}
 print(json.dumps({"jsonrpc":"2.0","id":r.get("id"),"result":result}),flush=True)
