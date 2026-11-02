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
import ipaddress
from dataclasses import dataclass, field
from typing import Callable


def _q(v) -> str:
    return shlex.quote(str(v))


def _norm_ports(v) -> str:
    """Accept common model formats for a port list ('22,80', '[22, 53]',
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
    timeout: float = 30.0             # executor timeout in seconds

    def required_params(self) -> list:
        return [p for p in self.params if p not in self.optional]


def _ping(a):       return f"ping -c {max(1, min(10, int(a.get('count', 4))))} -W 2 {_q(a['host'])}"
def _port_scan(a):
    ports = _norm_ports(a.get("ports", ""))
    versions = a.get("service_versions") in (True, 1, "1", "true", "yes", "on")
    versions = "-sV --version-light " if versions else ""
    return f"nmap -sT -Pn {versions}{('-p ' + ports + ' ') if ports else '--top-ports 100 '}{_q(a['host'])}"
def _service_scan(a):
    ports = _norm_ports(a.get("ports", ""))
    scope = f"-p {ports} " if ports else "--top-ports 100 "
    return f"nmap -sT -Pn -sV --version-light {scope}{_q(a['host'])}"
def _full_scan(a):  return f"nmap -sT -Pn -p- {_q(a['host'])}"
def _tcp(a):        return f"ncat -z -w 3 -v {_q(a['host'])} {_port(a['port'])}"
def _http_head(a):  return f"curl -sS -m 10 -I {_q(a['url'])}"
def _http_get(a):   return f"curl -sS -m 10 {_q(a['url'])}"
def _dns(a):
    name = str(a["name"]).strip()
    try:
        ipaddress.ip_address(name)
        return f"dig +short -x {_q(name)}"
    except ValueError:
        return f"dig +short {_q(name)}"
def _traceroute(a): return f"traceroute -w 2 -q 1 -m 20 {_q(a['host'])}"
def _nmap_safe_vuln(a):
    ports = _norm_ports(a.get("ports", ""))
    scope = f"-p {ports} " if ports else "--top-ports 100 "
    return f"nmap -sT -Pn -sV --script-timeout 10s --host-timeout 55s --script 'safe and vuln' {scope}{_q(a['host'])}"
def _testssl(a):
    target = str(a["host"]).strip().strip("[]")
    if ":" in target:
        target = f"[{target}]"
    port = _port(a.get("port", 443))
    return f"testssl.sh --fast --quiet --warnings batch {_q(f'{target}:{port}')}"
def _nuclei(a):
    exclusions = "dos,intrusive,bruteforce,brute-force,fuzz,headless,default-login,token-spray"
    return (f"nuclei -u {_q(a['url'])} -severity medium,high,critical -rate-limit 5 "
            f"-concurrency 5 -timeout 5 -retries 1 -no-interactsh -disable-update-check "
            f"-etags {_q(exclusions)} -silent")
def _nikto(a):
    return (f"nikto -host {_q(a['url'])} -Tuning 123b -maxtime 25s "
            "-timeout 5 -nointeractive")


_WEB_DISCOVER = r'''import concurrent.futures, html.parser, re, sys, urllib.error, urllib.request
host = sys.argv[1].strip().strip("/")
raw = sys.argv[2].strip() if len(sys.argv) > 2 else ""
default_ports = "80,443,3000,5000,8000,8080,8443,8888"
ports = []
for item in (raw or default_ports).split(","):
    try:
        p = int(item)
        if 1 <= p <= 65535 and p not in ports: ports.append(p)
    except ValueError: pass
    if len(ports) >= 12: break
paths = ["/", "/robots.txt", "/login", "/admin", "/api", "/swagger", "/openapi.json", "/wp-login.php", "/wp-json"]
class Title(html.parser.HTMLParser):
    def __init__(self): super().__init__(); self.in_title=False; self.parts=[]; self.generator=[]
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if tag.lower()=="title": self.in_title=True
        if tag.lower()=="meta" and a.get("name", "").lower()=="generator" and a.get("content"): self.generator.append(a["content"])
    def handle_endtag(self, tag):
        if tag.lower()=="title": self.in_title=False
    def handle_data(self, data):
        if self.in_title: self.parts.append(data)
def probe(port, path):
    scheme="https" if port in (443, 8443) else "http"
    authority=f"[{host}]" if ":" in host and not host.startswith("[") else host
    default=(scheme=="http" and port==80) or (scheme=="https" and port==443)
    url=f"{scheme}://{authority}{'' if default else ':'+str(port)}{path}"
    req=urllib.request.Request(url, headers={"User-Agent":"swarm-harness-discovery/1.0", "Range":"bytes=0-16383"})
    try:
        try: response=urllib.request.urlopen(req, timeout=2)
        except urllib.error.HTTPError as e: response=e
        with response:
            status=response.getcode(); final=response.geturl(); headers=response.headers
            body=response.read(16384).decode("utf-8", "replace")
        parser=Title(); parser.feed(body)
        selected=[]
        for key in ("Server", "X-Powered-By", "X-Generator", "Via", "Set-Cookie"):
            for value in headers.get_all(key, [])[:2]:
                if key.lower()=="set-cookie":
                    value=value.split(";", 1)[0].split("=", 1)[0]
                selected.append(f"{key}={value[:140]}")
        tech=[]
        hay=(body+" "+" ".join(selected)).lower()
        for marker, name in (("wp-content", "WordPress"), ("wp-includes", "WordPress"), ("laravel_session", "Laravel"), ("django", "Django"), ("__viewstate", "ASP.NET"), ("asp.net", "ASP.NET"), ("jsessionid", "Java web app"), ("__next", "Next.js"), ("_next/static", "Next.js"), ("drupal.settings", "Drupal"), ("joomla", "Joomla"), ("connect.sid", "Express")):
            if marker in hay and name not in tech: tech.append(name)
        title=" ".join(" ".join(parser.parts).split())[:160]
        generator=", ".join(parser.generator[:2])
        if path=="/" or status not in (404, 410) or selected or title or tech:
            details=[f"{status} {final}"]+selected
            if title: details.append("title="+title)
            if generator: details.append("generator="+generator[:160])
            if tech: details.append("fingerprints="+", ".join(tech))
            return " | ".join(details)
    except Exception as e:
        if path=="/": return f"{url} | {type(e).__name__}: {str(e)[:120]}"
    return ""
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    results=list(pool.map(lambda pair: probe(*pair), ((p, path) for p in ports for path in paths)))
for result in results:
    if result: print(result)
'''


def _web_discover(a):
    ports = _norm_ports(a.get("ports", ""))
    return f"python3 -c {_q(_WEB_DISCOVER)} {_q(a['host'])} {_q(ports)}"


BUILTIN_TOOLS = {s.name: s for s in [
    ToolSpec("ping", "ICMP echo: is the host up and how fast",
             {"host": "host/IP to ping", "count": "echo requests (default 4)"},
             ("ping",), _ping, optional=("count",)),
    ToolSpec("port_scan", "Scan common TCP ports or a given list; optionally identify service names and versions",
             {"host": "host/IP", "ports": "comma/range list e.g. 22,80,443 (optional)",
              "service_versions": "true to probe open ports for service/version details (optional)"},
             ("nmap",), _port_scan, optional=("ports", "service_versions")),
    ToolSpec("service_scan", "Identify products and versions on common TCP ports or a given open-port list",
             {"host": "host/IP", "ports": "comma-separated open ports to identify (optional)"},
             ("nmap",), _service_scan, optional=("ports",)),
    ToolSpec("full_scan", "TCP connect scan of ALL 65535 ports (slow)",
             {"host": "host/IP"}, ("nmap",), _full_scan),
    ToolSpec("tcp_connect", "Try a TCP connection to one port",
             {"host": "host/IP", "port": "port number"}, ("ncat",), _tcp),
    ToolSpec("http_head", "Fetch HTTP response headers (server, status)",
             {"url": "full URL e.g. http://host"}, ("curl",), _http_head),
    ToolSpec("http_get", "Fetch an HTTP resource body",
             {"url": "full URL"}, ("curl",), _http_get),
    ToolSpec("web_discover", "Discover HTTP apps on common web ports; report status, server headers, page titles, generators, and common framework fingerprints",
             {"host": "host/IP", "ports": "comma-separated web ports to check (optional)"},
             ("python3",), _web_discover, optional=("ports",)),
    ToolSpec("dns_lookup", "Resolve a name to addresses",
             {"name": "DNS name"}, ("dig",), _dns),
    ToolSpec("nmap_safe_vuln", "Run Nmap vulnerability scripts tagged both safe and vuln against selected TCP ports",
             {"host": "host/IP", "ports": "comma-separated discovered ports (optional)"},
             ("nmap",), _nmap_safe_vuln, optional=("ports",), timeout=60.0),
    ToolSpec("testssl", "Check TLS protocols, ciphers, certificates, and known TLS weaknesses with a fast scan",
             {"host": "TLS server host/IP", "port": "TLS port (default 443)"},
             ("testssl.sh",), _testssl, optional=("port",), timeout=60.0),
    ToolSpec("nuclei_scan", "Run rate-limited web vulnerability templates; excludes intrusive, fuzzing, brute-force, and DoS tags",
             {"url": "HTTP(S) URL"}, ("nuclei",), _nuclei, timeout=90.0),
    ToolSpec("nikto_scan", "Check a web server for information disclosure, interesting files, misconfiguration, and software identification with a time limit",
             {"url": "HTTP(S) URL"}, ("nikto",), _nikto, timeout=30.0),
    ToolSpec("traceroute", "Trace the network path to a host",
             {"host": "host/IP"}, ("traceroute",), _traceroute),
]}


# --- catalog: tools grouped by capability, so a manager can hand a worker a
# relevant toolkit for a sub-task instead of every tool. ---
CATEGORIES = {
    "liveness": "is the host up / reachable",
    "ports": "which TCP ports are open and what services/versions listen there",
    "web": "discover web apps, server headers, frameworks, page titles, and HTTP content",
    "dns": "name resolution",
    "vuln": "targeted vulnerability and security checks for discovered services and web apps",
    "path": "the network path to a host",
}
CATEGORY_OF = {
    "ping": "liveness", "traceroute": "path",
    "port_scan": "ports", "service_scan": "ports", "full_scan": "ports", "tcp_connect": "ports",
    "http_head": "web", "http_get": "web", "web_discover": "web", "dns_lookup": "dns",
    "nmap_safe_vuln": "vuln", "testssl": "vuln", "nuclei_scan": "vuln", "nikto_scan": "vuln",
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
    lines.append('e.g. {"action":"tool","tool":"web_discover","args":{"host":"10.0.0.5"},"reason":"discover web apps"}')
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
