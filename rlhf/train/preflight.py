"""Pre-training validation — run on ANY box, no GPU, no ML deps.

The training scripts need a rented NVIDIA box (bitsandbytes 4-bit is rough on
ROCm; see config/sft_qlora.yaml). GPU hours are the expensive part, so validate
the data and config here FIRST and fail loudly on anything that would waste a
run: malformed records, empty files, a DPO set where chosen == rejected, prompts
that blow past max_prompt_len, a missing SFT→DPO lineage.

    python train/preflight.py --stage sft --config config/sft.yaml
    python train/preflight.py --stage dpo --config config/dpo.yaml

Exits non-zero if any FATAL check fails. Token counts are a cheap char/4 proxy
(no tokenizer needed) — they catch order-of-magnitude length problems, not exact
truncation.
"""
from __future__ import annotations

import json
import os
import sys

import yaml

CHARS_PER_TOKEN = 4   # rough proxy; fine for catching gross length problems


class Report:
    def __init__(self):
        self.fatal: list[str] = []
        self.warn: list[str] = []
        self.info: list[str] = []

    def fail(self, m): self.fatal.append(m)
    def caution(self, m): self.warn.append(m)
    def note(self, m): self.info.append(m)

    def ok(self) -> bool:
        return not self.fatal

    def render(self) -> str:
        lines = []
        for m in self.info:
            lines.append(f"  info  {m}")
        for m in self.warn:
            lines.append(f"  WARN  {m}")
        for m in self.fatal:
            lines.append(f" FATAL {m}")
        lines.append("")
        lines.append("PASS" if self.ok() else "FAIL — fix the FATAL items before training")
        return "\n".join(lines)


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _approx_tokens(s: str) -> int:
    return len(s) // CHARS_PER_TOKEN


def check_sft(cfg: dict, rep: Report) -> None:
    path = cfg.get("train_file", "")
    if not os.path.isfile(path):
        return rep.fail(f"SFT train_file not found: {path!r}")
    rows = _read_jsonl(path)
    if not rows:
        return rep.fail(f"SFT train_file is empty: {path}")
    rep.note(f"{len(rows)} SFT examples in {path}")

    max_seq = cfg.get("max_seq_len", 4096)
    bad, over, no_asst = 0, 0, 0
    for r in rows:
        msgs = r.get("messages")
        if not isinstance(msgs, list) or not msgs:
            bad += 1
            continue
        roles = [m.get("role") for m in msgs]
        if "assistant" not in roles:
            no_asst += 1
        total = sum(_approx_tokens(m.get("content", "")) for m in msgs)
        if total > max_seq:
            over += 1
    if bad:
        rep.fail(f"{bad} rows have no valid 'messages' list")
    if no_asst:
        rep.fail(f"{no_asst} rows have no assistant turn (nothing to learn to produce)")
    if over:
        rep.caution(f"{over}/{len(rows)} examples exceed max_seq_len={max_seq} (~char/4); they'll truncate")


def check_dpo(cfg: dict, rep: Report) -> None:
    path = cfg.get("pairs_file", "")
    if not os.path.isfile(path):
        return rep.fail(f"DPO pairs_file not found: {path!r}")
    rows = _read_jsonl(path)
    if not rows:
        return rep.fail(f"DPO pairs_file is empty: {path}")
    rep.note(f"{len(rows)} preference pairs in {path}")

    max_len = cfg.get("max_len", 4096)
    max_prompt = cfg.get("max_prompt_len", 3072)
    need = {"prompt", "chosen", "rejected"}
    bad, degenerate, over_prompt, over_len = 0, 0, 0, 0
    weights = []
    for r in rows:
        if not need.issubset(r):
            bad += 1
            continue
        if r["chosen"].strip() == r["rejected"].strip():
            degenerate += 1
        if _approx_tokens(r["prompt"]) > max_prompt:
            over_prompt += 1
        if _approx_tokens(r["prompt"]) + _approx_tokens(r["chosen"]) > max_len:
            over_len += 1
        if "weight" in r:
            weights.append(r["weight"])
    if bad:
        rep.fail(f"{bad} pairs missing one of {sorted(need)}")
    if degenerate:
        rep.fail(f"{degenerate} pairs have chosen == rejected (no gradient, likely a build bug)")
    if over_prompt:
        rep.caution(f"{over_prompt} prompts exceed max_prompt_len={max_prompt}; they'll truncate")
    if over_len:
        rep.caution(f"{over_len} chosen completions exceed max_len={max_len}")
    if weights:
        lo, hi = min(weights), max(weights)
        rep.note(f"weight range {lo:.2f}..{hi:.2f} over {len(weights)} pairs")

    # lineage: DPO must start from an SFT checkpoint, never a base model
    sft = cfg.get("sft_model", "")
    if not sft:
        rep.fail("dpo config has no sft_model — DPO must start from the SFT checkpoint")
    elif not (os.path.exists(sft) or "/" in sft):
        rep.caution(f"sft_model={sft!r} doesn't look like a path; make sure it resolves at train time")


def main(stage: str, config_path: str) -> int:
    cfg = yaml.safe_load(open(config_path))
    rep = Report()
    rep.note(f"stage={stage} config={config_path}")
    if stage == "sft":
        check_sft(cfg, rep)
    elif stage == "dpo":
        check_dpo(cfg, rep)
    else:
        rep.fail(f"unknown stage {stage!r}")
    print(rep.render())
    return 0 if rep.ok() else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["sft", "dpo"])
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    sys.exit(main(a.stage, a.config))
