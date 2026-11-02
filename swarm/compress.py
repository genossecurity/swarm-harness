"""Compress raw command output to a few facts, in code, before it ever reaches
the model. This is what keeps per-agent context near the ~2K budget: the model
sees a short digest, not kilobytes of stdout.
"""
from __future__ import annotations

from toolmcp.executor import ExecResult

MAX_LINES = 12
MAX_CHARS = 600


def _text(v) -> str:
    """Coerce stdout/stderr to str, whatever the executor handed back (str, bytes, None)."""
    if v is None:
        return ""
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return v


def compress(res: ExecResult) -> str:
    """Turn an ExecResult into a compact, model-facing fact string."""
    status = "timeout" if res.timed_out else f"rc={res.rc}"
    body = _text(res.stdout).strip() or _text(res.stderr).strip()
    lines = [ln.rstrip() for ln in body.splitlines() if ln.strip()]
    clipped = len(lines) > MAX_LINES
    lines = lines[:MAX_LINES]
    text = "\n".join(lines)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + " …"
        clipped = True
    tail = f"\n[+{len(body.splitlines()) - MAX_LINES} more lines omitted]" if clipped else ""
    return f"[{status}] {text}{tail}".strip()
