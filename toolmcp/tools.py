"""Tool scaffolding: named, parameterized probing tools agents call per step.

Instead of free-form shell, agents can call a structured tool via the `tool`
action ({tool, args}). We build the shell command here, shell-quoting every arg,
so input is auditable and injection-safe. Each step can expose a subset of tools
(`step_tools`), so the swarm's capabilities are scoped per step.

All built-ins are content-neutral network/host probes (ping, port scan, HTTP,
DNS, TCP connect, traceroute) — the same utilities the agents were already
driving by hand, now structured and validated.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Callable


def _q(v) -> str:
    return shlex.quote(str(v))


def _norm_ports(v) -> str:
    """Accept whatever a small model emits for a port list ('22,80', '[22, 53]',
    "['22','80']", '22 53 80') and return a clean nmap list like '22,53,80'."""
    toks = re.findall(r"\d+(?:-\d+)?", str(v))
    return ",".join(toks)


def _port(v) -> int:
    n = int(v)
    if not (0 < n < 65536):
        raise ValueError(f"port out of range: {n}")
    return n


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: dict                      # param name -> human description
    required_bins: tuple              # binaries that must exist on the host
    build: Callable[[dict], str]      # args dict -> shell command string
    optional: tuple = ()              # params that may be omitted

    def required_params(self) -> list:
        return [p for p in self.params if p not in self.optional]


def _ping(a):       return f"ping -c {max(1, min(10, int(a.get('count', 4))))} -W 2 {_q(a['host'])}"
def _port_scan(a):
    ports = _norm_ports(a.get("ports", ""))
    return f"nmap -sT -Pn {('-p ' + ports + ' ') if ports else '--top-ports 100 '}{_q(a['host'])}"
def _full_scan(a):  return f"nmap -sT -Pn -p- {_q(a['host'])}"
def _tcp(a):        return f"ncat -z -w 3 -v {_q(a['host'])} {_port(a['port'])}"
def _http_head(a):  return f"curl -sS -m 10 -I {_q(a['url'])}"
def _http_get(a):   return f"curl -sS -m 10 {_q(a['url'])}"
def _dns(a):        return f"dig +short {_q(a['name'])}"
def _traceroute(a): return f"traceroute -w 2 -q 1 -m 20 {_q(a['host'])}"


BUILTIN_TOOLS = {s.name: s for s in [
    ToolSpec("ping", "ICMP echo: is the host up and how fast",
             {"host": "host/IP to ping", "count": "echo requests (default 4)"},
             ("ping",), _ping, optional=("count",)),
    ToolSpec("port_scan", "TCP connect scan of common ports (or a given port list)",
             {"host": "host/IP", "ports": "comma/range list e.g. 22,80,443 (optional)"},
             ("nmap",), _port_scan, optional=("ports",)),
    ToolSpec("full_scan", "TCP connect scan of ALL 65535 ports (slow)",
             {"host": "host/IP"}, ("nmap",), _full_scan),
    ToolSpec("tcp_connect", "Try a TCP connection to one port",
             {"host": "host/IP", "port": "port number"}, ("ncat",), _tcp),
    ToolSpec("http_head", "Fetch HTTP response headers (server, status)",
             {"url": "full URL e.g. http://host"}, ("curl",), _http_head),
    ToolSpec("http_get", "Fetch an HTTP resource body",
             {"url": "full URL"}, ("curl",), _http_get),
    ToolSpec("dns_lookup", "Resolve a name to addresses",
             {"name": "DNS name"}, ("dig",), _dns),
    ToolSpec("traceroute", "Trace the network path to a host",
             {"host": "host/IP"}, ("traceroute",), _traceroute),
]}


# --- catalog: tools grouped by capability, so a manager can hand a worker a
# relevant toolkit for a sub-task instead of every tool. ---
CATEGORIES = {
    "liveness": "is the host up / reachable",
    "ports": "which TCP ports are open",
    "web": "HTTP service details (status, headers, body)",
    "dns": "name resolution",
    "path": "the network path to a host",
}
CATEGORY_OF = {
    "ping": "liveness", "traceroute": "path",
    "port_scan": "ports", "full_scan": "ports", "tcp_connect": "ports",
    "http_head": "web", "http_get": "web", "dns_lookup": "dns",
}


def catalog() -> dict:
    """category -> [ToolSpec], for the manager to choose a toolkit from."""
    out: dict = {c: [] for c in CATEGORIES}
    for name, spec in BUILTIN_TOOLS.items():
        out.setdefault(CATEGORY_OF.get(name, "other"), []).append(spec)
    return {c: ts for c, ts in out.items() if ts}


def toolkit(items, available_bins=None) -> list:
    """Resolve a mixed list of tool names AND category names into tool specs,
    then filter to installed binaries (if given). Empty/None => all tools."""
    if not items:
        return tools_for_step(None, available_bins)
    names: list = []
    for it in items:
        if it in CATEGORIES:
            names += [n for n, c in CATEGORY_OF.items() if c == it]
        elif it in BUILTIN_TOOLS:
            names.append(it)
    # de-dup, preserve order
    seen, uniq = set(), []
    for n in names:
        if n not in seen:
            seen.add(n); uniq.append(n)
    return tools_for_step(uniq, available_bins)


def render_catalog(available_bins=None) -> str:
    """Manager-facing catalog: categories and the tools in each."""
    lines = ["TOOL CATALOG (assign a toolkit — category or tool names — to each dispatched check):"]
    av = set(available_bins) if available_bins is not None else None
    for cat, desc in CATEGORIES.items():
        specs = [s for s in catalog().get(cat, []) if av is None or all(b in av for b in s.required_bins)]
        if specs:
            lines.append(f"  {cat} ({desc}): " + ", ".join(s.name for s in specs))
    return "\n".join(lines)


class ToolError(ValueError):
    """Raised when a tool name/args are invalid or the tool isn't allowed here."""


def tools_for_step(allow, available_bins=None) -> list:
    """Specs usable in a step: intersect the step's allowlist with built-ins and,
    if given, with the binaries actually installed on the host. `allow` None/empty
    means every built-in tool."""
    names = list(BUILTIN_TOOLS) if not allow else [n for n in allow if n in BUILTIN_TOOLS]
    specs = [BUILTIN_TOOLS[n] for n in names]
    if available_bins is not None:
        av = set(available_bins)
        specs = [s for s in specs if all(b in av for b in s.required_bins)]
    return specs


def render_tools(specs) -> str:
    """Doc block listing the tools an agent may call this step."""
    if not specs:
        return ""
    lines = ["TOOLS YOU CAN CALL THIS STEP (action \"tool\", with \"tool\" and \"args\"):"]
    for s in specs:
        ps = ", ".join(f"{p}{'?' if p in s.optional else ''}" for p in s.params)
        lines.append(f'- {s.name}({ps}) — {s.description}')
    lines.append('e.g. {"action":"tool","tool":"port_scan","args":{"host":"10.0.0.5"},"reason":"..."}')
    return "\n".join(lines)


def build_command(name: str, args: dict, allowed_names=None) -> str:
    """Validate a tool call and return the shell command, or raise ToolError."""
    spec = BUILTIN_TOOLS.get(name)
    if spec is None:
        raise ToolError(f"unknown tool {name!r}")
    if allowed_names is not None and name not in allowed_names:
        raise ToolError(f"tool {name!r} is not available in this step")
    args = args or {}
    if not isinstance(args, dict):
        raise ToolError("tool args must be an object")
    missing = [p for p in spec.required_params() if args.get(p) in (None, "")]
    if missing:
        raise ToolError(f"tool {name!r} missing args: {', '.join(missing)}")
    try:
        return spec.build(args)
    except (KeyError, ValueError) as e:
        raise ToolError(f"bad args for {name!r}: {e}") from e
