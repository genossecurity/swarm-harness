"""`swarm run ...` entry point."""
from __future__ import annotations

import argparse
import sys

from .config import load_config
from .orchestrator import Orchestrator


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="swarm", description="Multi-agent swarm harness")
    sub = ap.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run the swarm end-to-end")
    r.add_argument("--agents", type=int, help="number of concurrent agents")
    r.add_argument("--task", help="the goal to hand the swarm (free text)")
    r.add_argument("--model", help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1")
    r.add_argument("--model-name", help="served model id in the request body "
                   "(llama-server: gpt-oss-20b; Ollama: gpt-oss:20b)")
    r.add_argument("--worker-url", help="2nd OpenAI-compatible endpoint for worker sub-agents")
    r.add_argument("--worker-model", help="smaller model served at --worker-url")
    r.add_argument("--lead-agents", type=int, help="how many agents use the lead model (rest use worker)")
    r.add_argument("--target", help="target host (ip/hostname); key-based SSH")
    r.add_argument("--target-user", help="SSH user (default root)")
    r.add_argument("--target-key", help="path to SSH private key")
    r.add_argument("--target-port", type=int, help="SSH port (default 22)")
    r.add_argument("--external", action="store_true",
                   help="no SSH, no key: run commands on THIS machine to probe a remote host "
                        "over the network (e.g. 'is <host> alive'). Pass the host as --target.")
    r.add_argument("--local", action="store_true",
                   help="run commands on THIS machine instead of SSH (self-test)")
    r.add_argument("--config", help="JSON or TOML config file")
    r.add_argument("--phases", help="comma-separated phase names")
    r.add_argument("--max-steps", type=int, help="hard per-agent step cap")
    r.add_argument("--log", help="JSONL event log path (default run.jsonl)")
    args = ap.parse_args(argv)

    if args.command != "run":
        ap.error("unknown command")

    # --model is the base_url (matches the documented quickstart).
    overrides = dict(
        agents=args.agents, task=args.task, base_url=args.model, model=args.model_name,
        worker_base_url=args.worker_url, worker_model=args.worker_model, lead_agents=args.lead_agents,
        log=args.log, max_steps=args.max_steps,
        target_host=args.target, target_user=args.target_user,
        target_key=args.target_key, target_port=args.target_port,
    )
    if args.external:
        # No connection to --target; it's the subject the swarm probes from here.
        overrides["target_mode"] = "external"
        overrides["target_host"] = args.target or "localhost"
    elif args.local:
        overrides["target_mode"] = "local"
        overrides["target_host"] = args.target or "localhost"
    if args.phases:
        overrides["phases"] = [p.strip() for p in args.phases.split(",") if p.strip()]

    cfg = load_config(args.config, **overrides)
    if cfg.target is None:
        ap.error("no target: pass --target <ip> for SSH, --external --target <ip> to probe a "
                 "host without logging in, or --local for a self-test")

    print(f"swarm: {cfg.agents} agents -> {cfg.base_url} | target {cfg.target.mode}:"
          f"{cfg.target.host} | phases {cfg.phases}", file=sys.stderr)
    try:
        summary = Orchestrator(cfg).run()
    except ConnectionError as e:
        print(f"\nerror: {e}", file=sys.stderr)
        return 2
    print("\n=== run summary ===", file=sys.stderr)
    for k, v in summary.items():
        if k != "event":
            print(f"  {k}: {v}", file=sys.stderr)
    print(f"  log: {cfg.log}", file=sys.stderr)
    return 0 if summary["all_phases_done"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
