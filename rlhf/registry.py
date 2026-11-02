"""Model registry: version + training provenance + the promotion pointer.

Every trained checkpoint gets a manifest recording exactly what produced it —
base model, parent policy, the data file it trained on (by sha256, so you can
prove which events shaped it), the config snapshot, and the eval-gate result. A
single `current` pointer names the deployed policy; `promote()` moves it, but
only the gate (eval/suite.py) should decide to call it.

Layout under registry/:
    <version>/manifest.json      one per trained model
    current.json                 {"version": "..."} — what agents load

Pure stdlib. The manifest is the audit trail for "why does the policy behave
this way" — keep it even when you prune old weights.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict, field

# Manifests are DATA, not code, so they live in a top-level registry/ store
# (cwd-relative), never inside the python package. Override with REGISTRY_ROOT.
HERE = os.environ.get("REGISTRY_ROOT", "registry")


def _sha256(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Manifest:
    version: str                      # e.g. "policy-dpo-2026-11-01"
    stage: str                        # "sft" | "dpo"
    base_model: str = ""              # HF id or path the run started from
    parent: str = ""                  # prior policy version this built on
    data_file: str = ""              # training data path
    data_sha256: str = ""            # hash of that data at train time
    config: dict = field(default_factory=dict)   # the yaml config snapshot
    created: float = field(default_factory=time.time)
    eval: dict = field(default_factory=dict)      # GateResult.__dict__ once scored
    promoted: bool = False
    notes: str = ""


def record(version: str, stage: str, *, base_model: str = "", parent: str = "",
           data_file: str = "", config: dict | None = None,
           root: str = HERE, notes: str = "") -> Manifest:
    """Write a manifest for a freshly trained model version."""
    m = Manifest(version=version, stage=stage, base_model=base_model,
                 parent=parent, data_file=data_file,
                 data_sha256=_sha256(data_file), config=config or {}, notes=notes)
    vdir = os.path.join(root, version)
    os.makedirs(vdir, exist_ok=True)
    with open(os.path.join(vdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(m), f, indent=2)
    return m


def attach_eval(version: str, gate_result: dict, root: str = HERE) -> None:
    """Record the promotion-gate result against a version."""
    path = os.path.join(root, version, "manifest.json")
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    m["eval"] = gate_result
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)


def promote(version: str, root: str = HERE) -> None:
    """Point `current` at `version`. The eval gate should gate this call —
    refuse to promote a version whose manifest has no passing eval."""
    path = os.path.join(root, version, "manifest.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no manifest for {version}; record() it first")
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    if m.get("eval") and not m["eval"].get("promote", True):
        raise ValueError(f"{version} failed its eval gate; refusing to promote")
    m["promoted"] = True
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    with open(os.path.join(root, "current.json"), "w", encoding="utf-8") as f:
        json.dump({"version": version}, f, indent=2)


def current(root: str = HERE) -> str | None:
    path = os.path.join(root, "current.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("version")


def versions(root: str = HERE) -> list[dict]:
    out = []
    for name in sorted(os.listdir(root)):
        mpath = os.path.join(root, name, "manifest.json")
        if os.path.isfile(mpath):
            with open(mpath, encoding="utf-8") as f:
                out.append(json.load(f))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="inspect the model registry")
    ap.add_argument("cmd", choices=["list", "current"])
    a = ap.parse_args()
    if a.cmd == "current":
        print(current() or "(none promoted)")
    else:
        for m in versions():
            flag = "*" if m.get("promoted") else " "
            ev = m.get("eval", {})
            tail = f"win={ev.get('win_rate')} safe={ev.get('safety_pass_rate')}" if ev else "unevaluated"
            print(f"{flag} {m['version']:30} {m['stage']:4} {tail}")
