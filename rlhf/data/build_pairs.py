"""Turn raw FeedbackEvents into DPO preference pairs (prompt, chosen, rejected).

This is where human signal and execution outcome get fused. The rules below
are the policy — tune them, they're deliberately explicit rather than clever.

Fusion logic, per event:
  1. Determine the CHOSEN text:
       RAN_EDITED  -> edited_text   (human-authored gold; strongest)
       RAN_AS_IS   -> chosen candidate text
       CHOSE       -> chosen candidate text
       RATED only  -> highest-scalar candidate (weak; flagged)
  2. Determine REJECTED candidates: the other sampled candidates.
  3. Apply the execution outcome as a GATE, not a tiebreaker:
       - If the chosen text was executed and FAILED/ERRORED, the human's
         pick is not actually a positive example. Demote it:
           * if another candidate succeeded -> swap chosen<->that one
           * else -> drop the event (no trustworthy positive)
       - A chosen pick that SUCCEEDED gets a confidence bump (kept weight).
  4. Emit one pair per (chosen, rejected) up to max_pairs_per_event, so one
     rich decision point doesn't flood the set.

Output: JSONL of {"prompt", "chosen", "rejected", "weight", "meta"} — the
shape TRL's DPOTrainer wants (weight is carried in meta; see train/dpo.py).
"""
from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass

from ..hitl.events import FeedbackEvent, HumanAction, ExecOutcome  # noqa: E402
from ..hitl.store import EventStore  # noqa: E402

MAX_PAIRS_PER_EVENT = 3
SEED = 1337


@dataclass
class Pair:
    prompt: str
    chosen: str
    rejected: str
    weight: float
    meta: dict


def _render_prompt(ev: FeedbackEvent) -> str:
    """Flatten context into the single prompt string the policy conditions on.
    Keep this identical to what the agent actually sees at inference time,
    or you train on a distribution you never deploy."""
    parts = []
    if ev.context:
        parts.append(f"# Engagement context\n{ev.context}")
    parts.append(f"# Task\n{ev.prompt}")
    return "\n\n".join(parts)


def _chosen_text(ev: FeedbackEvent) -> str | None:
    if ev.action == HumanAction.RAN_EDITED and ev.edited_text:
        return ev.edited_text
    if ev.action in (HumanAction.RAN_AS_IS, HumanAction.CHOSE) and ev.chosen_id:
        for c in ev.candidates:
            if c.cand_id == ev.chosen_id:
                return c.text
    if ev.action == HumanAction.RATED and ev.scalar is not None:
        # weak: no explicit pick, fall back to best-scored candidate if any
        return max(ev.candidates, key=lambda c: (c.logprob or 0)).text
    return None


def _base_weight(ev: FeedbackEvent) -> float:
    """Signal strength -> example weight."""
    return {
        HumanAction.RAN_EDITED: 1.0,
        HumanAction.RAN_AS_IS: 0.9,
        HumanAction.CHOSE: 0.6,
        HumanAction.REJECTED_ALL: 0.5,  # chosen = human's own text vs candidates
        HumanAction.RATED: 0.3,
    }.get(ev.action, 0.0)


def pairs_from_event(ev: FeedbackEvent) -> list[Pair]:
    if not ev.is_usable():
        return []

    prompt = _render_prompt(ev)
    chosen = _chosen_text(ev)

    # REJECTED_ALL: the human wrote their own; that's the chosen, all candidates rejected
    if ev.action == HumanAction.REJECTED_ALL:
        chosen = ev.edited_text or ev.executed_text
    if not chosen:
        return []

    rejected_texts = [c.text for c in ev.candidates if c.text != chosen]

    weight = _base_weight(ev)

    # --- execution-outcome gate ---------------------------------------
    chosen_ran = ev.executed_text is not None and ev.executed_text.strip() == chosen.strip()
    if chosen_ran and ev.outcome in (ExecOutcome.FAILURE, ExecOutcome.ERROR):
        # the human's pick demonstrably didn't work. Try to salvage.
        succeeded = [c.text for c in ev.candidates
                     if ev.outcome_detail and c.cand_id == ev.outcome_detail.get("success_cand_id")]
        if succeeded:
            chosen, rejected_texts = succeeded[0], [chosen] + [t for t in rejected_texts if t != succeeded[0]]
            weight *= 0.8
        else:
            return []  # no trustworthy positive here
    elif chosen_ran and ev.outcome == ExecOutcome.SUCCESS:
        weight = min(1.0, weight * 1.15)  # verified positive
    # ------------------------------------------------------------------

    if not rejected_texts:
        return []

    rng = random.Random(f"{ev.event_id}{SEED}")
    rng.shuffle(rejected_texts)
    out = []
    for rej in rejected_texts[:MAX_PAIRS_PER_EVENT]:
        out.append(Pair(
            prompt=prompt, chosen=chosen, rejected=rej, weight=round(weight, 4),
            meta={"event_id": ev.event_id, "action": ev.action.value,
                  "outcome": ev.outcome.value, "policy_version": ev.policy_version},
        ))
    return out


def build(raw_root: str, out_path: str, since: str | None = None) -> int:
    store = EventStore(raw_root)
    events = store.iter_since(since) if since else store.iter_all()
    n = 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ev in events:
            for p in pairs_from_event(ev):
                f.write(json.dumps({
                    "prompt": p.prompt, "chosen": p.chosen,
                    "rejected": p.rejected, "weight": p.weight, "meta": p.meta,
                }, ensure_ascii=False) + "\n")
                n += 1
    return n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/splits/pairs.jsonl")
    ap.add_argument("--since", default=None, help="YYYY-MM-DD lower bound")
    args = ap.parse_args()
    total = build(args.raw, args.out, args.since)
    print(f"wrote {total} pairs -> {args.out}")
