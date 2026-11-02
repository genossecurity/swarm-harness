"""toolmcp — the MCP tool layer.

The distinct *tool* half of the system: what commands exist, how they are built,
how they are guarded, and how their output is scored. It depends on nothing in
the harness (`swarm/`) — the dependency only runs the other way, so the harness
is a client of this layer and the stdio MCP server (server.py) can expose the
exact same tools to any other MCP client.

  tools.py     the tool registry: ToolSpec.build(args) -> shell-quoted command,
               catalog / toolkit / build_command (from the harness, now shared)
  executor.py  command execution (local / ssh / external), content-neutral
  sandbox.py   the enforcement choke point: allowlist + lab-CIDR + kill-patterns
               + outcome classification, executing via executor (no duplicate run)
  server.py    the stdio JSON-RPC MCP server exposing the tools (wiring: next step)
"""
from __future__ import annotations
