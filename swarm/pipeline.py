"""Deterministic task-to-stage plan for the Simple UI testing pipeline."""
from __future__ import annotations

from .scaling import categories_for_text


TOOL_LABELS = {
    "liveness": ["ping"],
    "ports": ["port_scan", "service_scan"],
    "web": ["web_discover", "http_head", "http_get"],
    "dns": ["dns_lookup"],
    "vuln": ["nmap_safe_vuln", "testssl", "nuclei_scan", "nikto_scan"],
    "path": ["traceroute"],
}


def plan_pipeline(task: str, host: str = "") -> list[dict]:
    """Return ordered stages with clear goals and the tools useful at each stage."""
    text = (task or "").lower()
    categories = set(categories_for_text(text, fallback=()))
    web_requested = any(word in text for word in (
        "web", "http", "website", "site", "app", "application", "backend", "framework",
        "hosting", "hosted", "cms", "what runs",
    ))
    if web_requested:
        categories.add("web")
    if "vuln" in categories:
        categories.add("web")
    # A service inventory is the useful default for an otherwise unclassified host task.
    if not categories or categories.intersection({"ports", "web", "vuln"}):
        categories.add("ports")

    obj_host = host or "the target host"
    definitions = {
        "liveness": ("Reachability", f"Check whether {obj_host} responds to a basic reachability probe."),
        "dns": ("Name lookup", f"Resolve the requested hostname or reverse-resolve the target address {obj_host}."),
        "ports": ("Service discovery", f"Find open TCP ports on {obj_host} and identify the service and version on each open port."),
        "web": ("Web app identification", f"If HTTP services are present on {obj_host}, identify the app and backend from its responses, headers, page title, and framework fingerprints."),
        "vuln": ("Vulnerability checks", f"Check discovered services and web apps on {obj_host} with the available conservative vulnerability tools."),
        "path": ("Network path", f"Trace the route from this machine to {obj_host}."),
    }
    order = ["liveness", "dns", "ports", "web", "vuln", "path"]
    stages = []
    for key in order:
        if key not in categories:
            continue
        title, objective = definitions[key]
        stages.append({"key": key, "title": title, "objective": objective,
                       "tools": TOOL_LABELS[key]})
    return stages


def stage_config(task: str, host: str = "") -> tuple[list[str], dict, dict, list[dict]]:
    """Build phase/objective/tool allowlists consumed by the existing orchestrator."""
    stages = plan_pipeline(task, host)
    phases = [stage["key"] for stage in stages]
    objectives = {stage["key"]: stage["objective"] for stage in stages}
    tools = {stage["key"]: stage["tools"] for stage in stages}
    return phases, objectives, tools, stages
