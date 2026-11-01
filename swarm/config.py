"""Run and target configuration, from a JSON/TOML file and/or env + CLI."""
from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TargetConfig:
    """How the swarm acts on a host.

    mode:
      "ssh"      log into `host` with a key and run commands there (needs a key).
      "external" run commands on THIS box to probe `host` over the network
                 (ping/curl/nc). No SSH, no key; `host` is just the subject.
      "local"    run commands on THIS box (self-test); `host` is cosmetic.
    """
    host: str
    user: str = "root"
    port: int = 22
    key: str | None = None            # path to private key; None = use ssh-agent / default keys
    mode: str = "ssh"

    @property
    def ssh_dest(self) -> str:
        return f"{self.user}@{self.host}"

    @property
    def is_remote_subject(self) -> bool:
        """External mode with a real host to probe (not the localhost placeholder)."""
        return self.mode == "external" and bool(self.host) and self.host != "localhost"


@dataclass
class RunConfig:
    agents: int = 4
    task: str = ""                    # the goal handed to the swarm; injected into every agent's context
    base_url: str = "http://localhost:8000/v1"   # lead/orchestrator model (the strong local one)
    model: str = "gpt-oss-20b"
    worker_base_url: str = ""         # optional 2nd endpoint for worker sub-agents (another GPU/machine)
    worker_model: str = ""            # the smaller model served there
    lead_agents: int = 1              # how many agents run on the lead model; the rest use the worker
    phases: list[str] = field(default_factory=lambda: ["explore", "verify", "report"])
    step_tasks: dict = field(default_factory=dict)   # phase name -> its own objective (optional)
    step_tools: dict = field(default_factory=dict)   # phase name -> list of allowed tool names (optional)
    max_steps: int = 40               # hard stop per agent, so a run always terminates
    gate: bool = False                # pause for human approval between phases (web UI uses this)
    ctx_cap_chars: int = 8000         # ~2K tokens of prompt context (state + whiteboard slice)
    max_retries: int = 3              # malformed-JSON retries per turn
    consensus: float = 0.6            # fraction of agents that must vote phase_done to advance
    temperature: float = 0.3
    log: str = "run.jsonl"
    target: TargetConfig | None = None


def _load_file(path: str | os.PathLike) -> dict:
    p = Path(path)
    data = p.read_bytes()
    if p.suffix == ".toml":
        return tomllib.loads(data.decode())
    return json.loads(data.decode())


def load_config(config_path: str | None = None, **overrides) -> RunConfig:
    """Merge file < env < CLI overrides. Env: SWARM_BASE_URL, SWARM_TARGET_HOST, etc."""
    base: dict = {}
    tgt: dict = {}
    if config_path:
        raw = _load_file(config_path)
        tgt = dict(raw.pop("target", {}) or {})
        base = raw

    env_map = {
        "base_url": "SWARM_BASE_URL", "model": "SWARM_MODEL", "agents": "SWARM_AGENTS",
        "log": "SWARM_LOG",
    }
    for key, env in env_map.items():
        if os.environ.get(env):
            base[key] = os.environ[env]
    env_tgt = {"host": "SWARM_TARGET_HOST", "user": "SWARM_TARGET_USER",
               "port": "SWARM_TARGET_PORT", "key": "SWARM_TARGET_KEY", "mode": "SWARM_TARGET_MODE"}
    for key, env in env_tgt.items():
        if os.environ.get(env):
            tgt[key] = os.environ[env]

    # CLI overrides win.
    for k, v in overrides.items():
        if v is None:
            continue
        if k.startswith("target_"):
            tgt[k[len("target_"):]] = v
        else:
            base[k] = v

    if "agents" in base:
        base["agents"] = int(base["agents"])
    if "port" in tgt:
        tgt["port"] = int(tgt["port"])

    target = TargetConfig(**tgt) if tgt.get("host") or tgt.get("mode") == "local" else None
    return RunConfig(target=target, **base)
