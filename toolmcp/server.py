"""The MCP tool server — the layer's outward API.

Two functions are the whole contract the harness (as an MCP client) needs:

  list_tools()        -> the tool catalog as MCP tool definitions (name +
                         description + JSON input schema)
  call_tool(name,args)-> build the command from the registry, push it through the
                         sandbox (validate -> execute -> classify), return a
                         structured result

They are usable in-process today. The stdio JSON-RPC transport that exposes the
SAME two functions to any external MCP client (Claude Code included) is the next
step — see _serve_stdio() below. Keeping the dispatch core transport-free means
the wire layer is a thin wrapper, and the harness never has to change when it
moves from in-process to stdio.
"""
from __future__ import annotations

from . import tools
from . import sandbox


def list_tools() -> list[dict]:
    """Every built-in tool as an MCP tool definition."""
    defs = []
    for name, spec in tools.BUILTIN_TOOLS.items():
        props = {p: {"type": "string", "description": p} for p in spec.params}
        required = [p for p in spec.params if p not in getattr(spec, "optional", set())]
        defs.append({
            "name": name,
            "description": spec.description,
            "inputSchema": {"type": "object", "properties": props, "required": required},
        })
    return defs


def call_tool(name: str, args: dict | None = None, *, allowed_names=None) -> dict:
    """Resolve a tool call to a command, run it through the sandbox, and return a
    structured result. The sandbox is the enforcement choke point: an off-lab
    target, a non-allowlisted binary, or a kill-pattern is refused here, not run.
    """
    try:
        cmd = tools.build_command(name, args or {}, allowed_names=allowed_names)
    except tools.ToolError as e:
        return {"ok": False, "outcome": "error", "error": str(e),
                "stdout": "", "stderr": str(e), "detail": {"refused": str(e)}, "cmd": None}

    res = sandbox.execute(cmd)
    return {"ok": res.ok, "outcome": res.outcome, "cmd": cmd,
            "stdout": res.stdout, "stderr": res.stderr, "detail": res.detail}


# --- stdio JSON-RPC transport (NEXT STEP) ---------------------------------
def _serve_stdio() -> None:
    """Wrap list_tools()/call_tool() in an MCP stdio JSON-RPC server.

    Planned: use the official `mcp` SDK (pip install mcp) — register list_tools
    as the tools/list handler and call_tool as tools/call — so any MCP client can
    drive the same sandboxed tools. Deferred per the merge plan (restructure +
    dedup first); the dispatch core above is already the thing it will expose.
    """
    raise NotImplementedError(
        "stdio MCP transport is the next step; the in-process API "
        "(list_tools / call_tool) is ready to wrap."
    )


if __name__ == "__main__":
    import json
    # Until the stdio server lands, make the layer introspectable from the shell.
    print(json.dumps(list_tools(), indent=2))
