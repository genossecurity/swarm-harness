"""The enforcement choke point of the MCP tool layer.

Every command that reaches a real target goes through validate() here first:
allowlisted binary, target inside the lab CIDR, no kill-pattern — all fail
closed. This is the single place the agent (or any MCP client) is gated, so the
sandbox, not the model, is the primary risk surface.

Execution is delegated to toolmcp.executor.LocalExecutor with shell OFF, so the
token-split, allowlisted command can't be re-expanded by a shell — this module
owns the *policy*, the executor owns the *mechanism*, and neither duplicates the
other. classify() then scores the finished command into the success / failure /
error outcome the RLHF reward reads.

HARD RULE: this runs real commands against disposable lab VMs only. Config:
  PENTEST_LAB_CIDRS   comma list of allowed target nets (default 10.13.37.0/24)
  PENTEST_SELFTEST=1  additionally allow 127.0.0.0/8 — for smoke-testing only
  PENTEST_EXEC_TIMEOUT  per-command wall-clock seconds (default 120)
"""
from __future__ import annotations

import ipaddress
import os
import re
import shlex
from dataclasses import dataclass, field

from .executor import LocalExecutor


def _load_lab_cidrs() -> list:
    raw = os.environ.get("PENTEST_LAB_CIDRS", "10.13.37.0/24")
    nets = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        nets.append(ipaddress.ip_network(part, strict=False))
    if os.environ.get("PENTEST_SELFTEST") == "1":
        nets.append(ipaddress.ip_network("127.0.0.0/8"))
    return nets or [ipaddress.ip_network("10.13.37.0/24")]


# Only hosts inside the lab range are ever valid targets. Set to YOUR lab net
# via PENTEST_LAB_CIDRS; the default is an obviously-fake placeholder so an
# unconfigured install refuses everything rather than touching a real host.
LAB_CIDRS = _load_lab_cidrs()

EXEC_TIMEOUT = int(os.environ.get("PENTEST_EXEC_TIMEOUT", "120"))
OUTPUT_CAP = 64 * 1024   # bytes of stdout/stderr retained per stream

# Tools the agent is allowed to invoke at all. Everything else is refused.
ALLOWED_BINARIES = {
    "nmap", "nikto", "gobuster", "ffuf", "whatweb", "curl", "wget",
    "hydra", "sqlmap", "nc", "ncat", "msfconsole", "searchsploit",
}

# Patterns that are refused regardless of target — tripwires against the agent
# reaching outside the lab or wrecking the runner.
KILL_PATTERNS = [
    re.compile(r"\brm\s+-rf\s+/"),
    re.compile(r":\(\)\s*\{"),             # fork bomb
    re.compile(r"\b(shutdown|reboot|mkfs|dd)\b"),
    re.compile(r"/etc/(passwd|shadow)\b.*(>|>>)"),  # writing to host creds
    re.compile(r"\b0\.0\.0\.0/0\b"),
]


@dataclass
class SandboxResult:
    ok: bool
    stdout: str
    stderr: str
    outcome: str          # "success" | "failure" | "error"
    detail: dict = field(default_factory=dict)


def _extract_target(cmd: str) -> str | None:
    """Pull the first IP-looking token. Replace with real arg parsing per tool."""
    m = re.search(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b", cmd)
    return m.group(1) if m else None


def _in_lab(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(addr in net for net in LAB_CIDRS)


def validate(cmd: str) -> tuple[bool, str]:
    for pat in KILL_PATTERNS:
        if pat.search(cmd):
            return False, f"kill-pattern matched: {pat.pattern}"
    try:
        tokens = shlex.split(cmd)
    except ValueError as e:
        return False, f"unparseable command: {e}"
    if not tokens:
        return False, "empty command"
    binary = tokens[0].split("/")[-1]
    if binary not in ALLOWED_BINARIES:
        return False, f"binary not in allowlist: {binary}"
    target = _extract_target(cmd)
    if target is None:
        return False, "no target host found; refusing (fail closed)"
    if not _in_lab(target):
        return False, f"target {target} outside lab CIDR; refused"
    return True, "ok"


_OPEN_PORT = re.compile(r"(\d{1,5})/(tcp|udp)\s+open", re.I)
_HTTP_STATUS = re.compile(r"HTTP/\d(?:\.\d)?\s+(\d{3})")
# stderr fingerprints that mean the command could not run at all (vs ran and
# simply didn't achieve the objective). These map to the "error" outcome.
_TOOL_ERROR = re.compile(
    r"command not found|not found|no such file|usage:|invalid option|"
    r"unrecognized|could not resolve|name or service not known",
    re.I,
)


def classify(cmd: str, rc: int | None, stdout: str, stderr: str) -> tuple[str, dict]:
    """Map a finished command to (outcome, detail). Deliberately conservative:
    a clean exit is 'success' ONLY when we can point at an objective marker;
    otherwise it's 'failure'. The reward signal is better under- than over-
    stated, since a false 'success' trains the policy toward a dead command."""
    detail: dict = {"returncode": rc}
    if rc is None:
        detail["timeout"] = True
        return "error", detail
    binary = (shlex.split(cmd)[0].split("/")[-1] if cmd.strip() else "")
    blob = f"{stdout}\n{stderr}"

    if _TOOL_ERROR.search(stderr) and not stdout.strip():
        return "error", detail

    ports = sorted({int(m.group(1)) for m in _OPEN_PORT.finditer(blob)})
    if ports:
        detail["open_ports"] = ports

    status = _HTTP_STATUS.search(blob)
    if status:
        detail["http_status"] = int(status.group(1))

    # per-tool objective markers
    if binary in {"nmap"}:
        return ("success" if ports else "failure"), detail
    if binary in {"curl", "wget"}:
        got = status is not None or (rc == 0 and stdout.strip())
        return ("success" if got else "failure"), detail
    if binary in {"gobuster", "ffuf", "nikto", "whatweb"}:
        # these report findings line-by-line; any substantive output + clean-ish
        # exit is a hit worth rewarding, nothing found is a plain failure
        return ("success" if (rc == 0 and len(stdout.strip()) > 0) else "failure"), detail

    # generic fallback: clean exit with some output is the best we can assert
    if rc == 0:
        return ("success" if stdout.strip() else "failure"), detail
    return "failure", detail


def execute(cmd: str) -> SandboxResult:
    """Validate, then run `cmd` via the shared executor and score the outcome.

    The target was already proven to be inside LAB_CIDRS by validate(). The
    executor runs with shell=False so the allowlisted, token-split command can't
    be re-expanded — the mechanism is shared with the harness, the policy is not.
    """
    ok, reason = validate(cmd)
    if not ok:
        return SandboxResult(False, "", reason, "error", {"refused": reason})

    r = LocalExecutor().run(cmd, timeout=EXEC_TIMEOUT, shell=False)
    out, err = r.stdout[:OUTPUT_CAP], r.stderr[:OUTPUT_CAP]
    rc = None if r.timed_out else r.rc
    if r.timed_out:
        err = (err + f"\n[timeout after {EXEC_TIMEOUT}s]").strip()
    outcome, detail = classify(cmd, rc, out, err)
    return SandboxResult(ok=(rc == 0), stdout=out, stderr=err, outcome=outcome, detail=detail)
