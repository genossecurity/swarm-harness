#!/usr/bin/env python3
"""Benchmark candidate worker models for one-liner checks.

For each model on a worker endpoint, send the harness's real WORKER prompt several
times and judge each reply with the harness's own parser (swarm.protocol). Reports,
per model: first-try valid-JSON rate, "usable check" rate (valid + it's a
tool/run/confirm + any tool call actually builds), tool-use rate, and latency.

    python serving/bench_worker.py --url http://192.168.2.16:11434/v1 --trials 6
    python serving/bench_worker.py --url ... --models qwen3:8b,qwen3.5:4b,granite4.1:8b

Only hits the worker endpoint — no GPU work locally. Run from the repo root.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarm.agent import WORKER_SYSTEM
from swarm.protocol import SCHEMA_HINT, ProtocolError, parse_action
from swarm.tools import ToolError, build_command, render_tools, tools_for_step

AVAILABLE = ["ping", "curl", "nmap", "ncat", "dig"]
TOOL_DOC = render_tools(tools_for_step(["ping", "port_scan", "tcp_connect", "http_head", "dns_lookup"], AVAILABLE))

# A few representative one-liner-check situations the workers face.
SCENARIOS = [
    ("Check whether 192.168.2.14 is alive.",
     "FINDINGS SO FAR:\n(nothing yet)\n\nYou are agent-1. Do ONE check now: call a tool to gather a fact. Reply with ONE JSON object."),
    ("Find what TCP ports are open on 192.168.2.14.",
     "FINDINGS SO FAR:\n(nothing yet)\n\nYou are agent-1. Do ONE check now: call a tool to gather a fact. Reply with ONE JSON object."),
    ("Confirm the service on port 80 of 192.168.2.14.",
     "FINDINGS SO FAR:\n[P1 · agent-0] TCP 80 is open (unconfirmed)\n\nUNCONFIRMED — verify one with your own check:\n[P1 · agent-0] TCP 80 is open (unconfirmed)\n\nYou are agent-1. Do ONE check now: confirm an unconfirmed finding. Reply with ONE JSON object."),
]

SYSTEM = WORKER_SYSTEM.format(aid="agent-1", n=3, phases="explore",
                              tools="Probing binaries installed on the host: " + ", ".join(AVAILABLE) + ".")


def post(url, model, user, timeout):
    body = {"model": model, "temperature": 0.2, "max_tokens": 300,
            "messages": [{"role": "system", "content": SYSTEM + "\n\n" + SCHEMA_HINT + "\n\n" + TOOL_DOC},
                         {"role": "user", "content": user}]}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    dt = time.perf_counter() - t0
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    if not content.strip() and msg.get("tool_calls"):
        content = "<<native tool_call, empty content>>"
    return content, dt, bool(msg.get("tool_calls"))


def judge(content):
    """Return (valid, usable, action, tool). usable = a parseable one-liner check."""
    try:
        a = parse_action(content)
    except ProtocolError:
        return False, False, None, None
    is_check = a.action in ("run", "tool", "confirm")
    tool_ok = True
    if a.tool:
        try:
            build_command(a.tool, a.args or {})
        except ToolError:
            tool_ok = False
    return True, (is_check and tool_ok), a.action, a.tool


def bench_model(url, model, trials, timeout):
    valid = usable = tool_calls = errors = 0
    lat, actions, sample = [], {}, ""
    for i in range(trials):
        _, user = SCENARIOS[i % len(SCENARIOS)]
        try:
            content, dt, native = post(url, model, user, timeout)
        except Exception as e:
            errors += 1
            continue
        lat.append(dt)
        if native:
            tool_calls += 1
        v, u, action, tool = judge(content)
        valid += v
        usable += u
        actions[action] = actions.get(action, 0) + 1
        if not sample:
            sample = content.replace("\n", " ")[:90]
    n = max(1, trials - errors)
    return {
        "model": model, "trials": trials, "errors": errors,
        "valid_rate": valid / n, "usable_rate": usable / n,
        "native_toolcall_rate": tool_calls / n,
        "lat_med": statistics.median(lat) if lat else float("nan"),
        "actions": actions, "sample": sample,
    }


def list_models(url):
    with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=5) as r:
        data = json.load(r)
    return [m["id"] for m in data.get("data", []) if "embed" not in m["id"].lower()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://192.168.2.16:11434/v1")
    ap.add_argument("--models", default="", help="comma list; default = auto-detect (minus embed models)")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=90)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()] or list_models(args.url)
    print(f"benchmarking {len(models)} models on {args.url}, {args.trials} trials each\n")
    cols = ["model", "valid", "usable", "native_tc", "lat_med", "errors"]
    print("  ".join(f"{c:>12}" for c in cols))
    rows = []
    for m in models:
        r = bench_model(args.url, m, args.trials, args.timeout)
        rows.append(r)
        print("  ".join([f"{m:>12}" if len(m) <= 12 else m.ljust(12),
                         f"{r['valid_rate']:>12.0%}", f"{r['usable_rate']:>11.0%}",
                         f"{r['native_toolcall_rate']:>12.0%}", f"{r['lat_med']:>10.1f}s",
                         f"{r['errors']:>12}"]), flush=True)
    print("\nranked by usable one-liner rate, then latency:")
    for r in sorted(rows, key=lambda x: (-x["usable_rate"], x["lat_med"])):
        print(f"  {r['model']:<42} usable {r['usable_rate']:>4.0%} · {r['lat_med']:.1f}s · "
              f"actions {r['actions']} · e.g. {r['sample']}")


if __name__ == "__main__":
    main()
