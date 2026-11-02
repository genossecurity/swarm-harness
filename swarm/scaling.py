"""Task-dependent sizing and tool allocation.

Two decisions are driven by the same signal — which *kinds* of check a task needs:

* **Agent scaling.** Manager/worker mode uses ONE manager and an elastic pool of workers.
  The useful number of workers is the peak number of checks run in parallel (the fan-out),
  which tracks how many distinct check categories the task touches. We estimate that from
  the task text and clamp it to the configured concurrency cap (`worker_cap`).

* **Tool allocation.** Each step is handed a toolkit (a set of catalog categories) derived
  from its own objective, so the manager dispatches the right tools instead of leaving it
  vague or relying on a hand-filled per-step tool list.

The mapping is a transparent keyword→category table (no model call), so sizing is cheap and
deterministic; the manager can always override the suggestion when it dispatches.
"""
from __future__ import annotations

from toolmcp.tools import CATEGORIES

# words in a task/step objective that hint it needs a given category of check
CATEGORY_HINTS: dict[str, tuple] = {
    "liveness": ("alive", "reachable", "reach", "is up", "ping", "responsive", "online", "icmp"),
    "ports":    ("port", "ports", "scan", "tcp", "udp", "nmap", "open", "listening", "socket",
                 "banner", "version", "versions", "service", "services", "fingerprint", "backend",
                 "framework", "hosting", "platform", "stack", "server", "app", "apps"),
    "web":      ("http", "https", "web", "url", "header", "site", "server", "app", "apps",
                 "page", "html", "cms", "backend", "hosting", "hosted", "technology",
                 "framework", "platform", "stack", "software"),
    "dns":      ("dns", "domain", "resolve", "record", "nameserver", "nslookup", "hostname", "zone"),
    "vuln":     ("vulnerability", "vulnerabilities", "vuln", "security", "security test",
                 "security assessment", "security scan", "security audit", "secure this",
                 "weakness", "weaknesses", "cve", "misconfiguration", "weak tls", "tls audit",
                 "nikto", "nuclei"),
    "path":     ("route", "traceroute", "hop", "path", "latency", "ttl", "network path"),
}

DEFAULT_TOOLKIT = ("liveness", "ports")


def categories_for_text(text: str, fallback=DEFAULT_TOOLKIT) -> list[str]:
    """The catalog categories a piece of text implies, in catalog order. Falls back to a sane
    default (liveness + ports) when nothing matches, so a step is never left with no tools."""
    t = (text or "").lower()
    hits = [c for c in CATEGORIES if c in CATEGORY_HINTS and any(k in t for k in CATEGORY_HINTS[c])]
    # keep a stable, catalog-ordered result
    ordered = [c for c in CATEGORIES if c in hits]
    return ordered or list(fallback)


def suggest_toolkit(step_objective: str, overall_task: str = "") -> list[str]:
    """The toolkit (categories) to allocate to a step, from its objective + the overall task."""
    return categories_for_text(f"{step_objective} {overall_task}")


def estimate_fanout(task: str, step_tasks: dict | None = None, phases: list | None = None) -> int:
    """Estimate the peak parallel fan-out = how many distinct check categories the task spans,
    floored so there are at least 2 workers and a little headroom for multi-step tasks."""
    text = " ".join([task or ""] + list((step_tasks or {}).values()) + list(phases or []))
    breadth = len(categories_for_text(text, fallback=()))
    steps = len(phases or [])
    return max(2, breadth, min(steps, 3))


def plan_agents(task: str, step_tasks: dict | None = None, phases: list | None = None,
                *, worker_cap: int = 4):
    """Return (managers, workers, total, reason). There is one manager; workers scale with
    task fan-out, clamped to the configured concurrency cap."""
    fan = estimate_fanout(task, step_tasks, phases)
    workers = max(1, min(fan, max(1, worker_cap)))
    managers = 1
    text = " ".join([task or ""] + list((step_tasks or {}).values()) + list(phases or []))
    cats = categories_for_text(text, fallback=())
    reason = (f"task touches {len(cats) or 0} check categor{'y' if len(cats) == 1 else 'ies'}"
              f"{' (' + ', '.join(cats) + ')' if cats else ''}"
              f", {len(phases or [])} steps → {managers} manager + {workers} workers")
    return managers, workers, managers + workers, reason
