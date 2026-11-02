"""Turn raw FeedbackEvents into SFT transcripts ({"messages": [...]}).

This is the FRONT half of the training path and the README's step 1: before any
DPO you need a policy that already produces plausible candidates, and that comes
from supervised fine-tuning on demonstrations of the pentester's own good moves.

What counts as a demonstration (gold) here, strongest first:
  RAN_EDITED  -> the human rewrote a candidate and ran it      (authored gold)
  RAN_AS_IS   -> the human accepted a candidate and ran it     (endorsed gold)
  REJECTED_ALL-> the human wrote their own and ran it           (authored gold)
  CHOSE       -> the human picked one but didn't run it         (weak; opt-in)

Quality gate (mirrors build_pairs): a gold text that was EXECUTED and
FAILED/ERRORED is NOT a demonstration — teaching a dead command is worse than
teaching nothing. Such events are dropped (or, at build_pairs time, salvaged as
preference pairs instead).

The assistant turn's user content is rendered with build_pairs._render_prompt so
SFT and DPO condition the policy on the SAME prompt string — no train/serve skew.
"""
from __future__ import annotations

import json
import os
import sys

from ..hitl.events import FeedbackEvent, HumanAction, ExecOutcome  # noqa: E402
from ..hitl.store import EventStore  # noqa: E402
from .build_pairs import _render_prompt  # noqa: E402  (shared to avoid skew)

# signal types that carry a human-authored/endorsed demonstration
_GOLD_RAN = {HumanAction.RAN_EDITED, HumanAction.RAN_AS_IS, HumanAction.REJECTED_ALL}


def _gold_text(ev: FeedbackEvent) -> str | None:
    if ev.action == HumanAction.RAN_EDITED:
        return ev.edited_text
    if ev.action == HumanAction.RAN_AS_IS and ev.chosen_id:
        return next((c.text for c in ev.candidates if c.cand_id == ev.chosen_id), None)
    if ev.action == HumanAction.REJECTED_ALL:
        return ev.edited_text or ev.executed_text
    if ev.action == HumanAction.CHOSE and ev.chosen_id:
        return next((c.text for c in ev.candidates if c.cand_id == ev.chosen_id), None)
    return None


def sft_example(ev: FeedbackEvent, *, include_chose: bool = False) -> dict | None:
    if ev.action == HumanAction.CHOSE and not include_chose:
        return None
    if ev.action not in _GOLD_RAN and ev.action != HumanAction.CHOSE:
        return None

    gold = _gold_text(ev)
    if not gold or not gold.strip():
        return None

    # quality gate: don't teach a command that demonstrably failed
    ran = ev.executed_text is not None and ev.executed_text.strip() == gold.strip()
    if ran and ev.outcome in (ExecOutcome.FAILURE, ExecOutcome.ERROR):
        return None

    messages = []
    if ev.system_prompt:
        messages.append({"role": "system", "content": ev.system_prompt})
    messages.append({"role": "user", "content": _render_prompt(ev)})
    messages.append({"role": "assistant", "content": gold.strip()})
    return {"messages": messages,
            "meta": {"event_id": ev.event_id, "action": ev.action.value,
                     "outcome": ev.outcome.value}}


def build(raw_root: str, out_path: str, *, since: str | None = None,
          include_chose: bool = False) -> int:
    store = EventStore(raw_root)
    events = store.iter_since(since) if since else store.iter_all()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ev in events:
            ex = sft_example(ev, include_chose=include_chose)
            if ex:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n += 1
    return n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/splits/sft.jsonl")
    ap.add_argument("--since", default=None, help="YYYY-MM-DD lower bound")
    ap.add_argument("--include-chose", action="store_true",
                    help="also emit CHOSE (picked-not-run) demos; weaker signal")
    args = ap.parse_args()
    total = build(args.raw, args.out, since=args.since, include_chose=args.include_chose)
    print(f"wrote {total} SFT examples -> {args.out}")
