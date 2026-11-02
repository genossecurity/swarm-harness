"""Synthetic FeedbackEvents — exercise the pipeline before real data exists.

NOT training data. This fabricates a spread of events covering every signal
type and every branch of the outcome gate in build_pairs, so you can run
synth -> build_pairs -> (inspect pairs) -> eval harness end to end with no
engagement, no GPU, and no ML deps. Delete the synthetic raw/ before you start
collecting for real.
"""
from __future__ import annotations

import os
import sys
import time

from ..hitl.events import FeedbackEvent, Candidate, HumanAction, ExecOutcome  # noqa: E402
from ..hitl.store import EventStore  # noqa: E402

_CTX = ("Scope: 10.13.37.0/24 (authorized).\n"
        "Recon: host 10.13.37.5 up; nothing enumerated yet.")


def _cands(*texts: str) -> list[Candidate]:
    return [Candidate(cand_id=f"c{i}", text=t) for i, t in enumerate(texts)]


def sample_events() -> list[FeedbackEvent]:
    """One event per interesting branch. Each is a plausible early-recon step."""
    evs: list[FeedbackEvent] = []

    # 1. RAN_AS_IS + SUCCESS  -> strong, verified positive (weight bump)
    evs.append(FeedbackEvent(
        session_id="synth", policy_version="policy-sft", annotator="synth",
        context=_CTX, prompt="Enumerate open TCP ports on 10.13.37.5.",
        candidates=_cands("nmap -sS -T4 10.13.37.5",
                          "nmap -sn 10.13.37.5",
                          "ping -c1 10.13.37.5"),
        action=HumanAction.RAN_AS_IS, chosen_id="c0",
        executed_text="nmap -sS -T4 10.13.37.5",
        outcome=ExecOutcome.SUCCESS, outcome_detail={"open_ports": [22, 80]}))

    # 2. RAN_EDITED + SUCCESS -> human-authored gold (top weight)
    evs.append(FeedbackEvent(
        session_id="synth", policy_version="policy-sft", annotator="synth",
        context=_CTX, prompt="Grab the HTTP banner on 10.13.37.5:80.",
        candidates=_cands("curl 10.13.37.5",
                          "curl -I http://10.13.37.5/",
                          "whatweb 10.13.37.5"),
        action=HumanAction.RAN_EDITED, chosen_id="c1",
        edited_text="curl -sSI http://10.13.37.5/",
        executed_text="curl -sSI http://10.13.37.5/",
        outcome=ExecOutcome.SUCCESS, outcome_detail={"http_status": 200}))

    # 3. RAN_AS_IS + FAILURE, another candidate succeeded -> SALVAGE SWAP
    evs.append(FeedbackEvent(
        session_id="synth", policy_version="policy-sft", annotator="synth",
        context=_CTX, prompt="Find the admin login path on the web app.",
        candidates=_cands("gobuster dir -u http://10.13.37.5 -w small.txt",   # ran, failed
                          "ffuf -u http://10.13.37.5/FUZZ -w big.txt",         # the winner
                          "nikto -h http://10.13.37.5"),
        action=HumanAction.RAN_AS_IS, chosen_id="c0",
        executed_text="gobuster dir -u http://10.13.37.5 -w small.txt",
        outcome=ExecOutcome.FAILURE,
        outcome_detail={"success_cand_id": "c1"}))

    # 4. RAN_AS_IS + FAILURE, nothing salvageable -> event DROPPED
    evs.append(FeedbackEvent(
        session_id="synth", policy_version="policy-sft", annotator="synth",
        context=_CTX, prompt="Check for anonymous FTP.",
        candidates=_cands("curl ftp://10.13.37.5/", "wget ftp://10.13.37.5/"),
        action=HumanAction.RAN_AS_IS, chosen_id="c0",
        executed_text="curl ftp://10.13.37.5/",
        outcome=ExecOutcome.FAILURE, outcome_detail={}))

    # 5. CHOSE (no run) -> medium weight, no outcome gate
    evs.append(FeedbackEvent(
        session_id="synth", policy_version="policy-sft", annotator="synth",
        context=_CTX, prompt="What should we enumerate on port 80 next?",
        candidates=_cands("Enumerate directories with a wordlist.",
                          "Brute-force SSH immediately.",
                          "Run a full UDP scan first."),
        action=HumanAction.CHOSE, chosen_id="c0"))

    # 6. REJECTED_ALL -> human's own text is chosen, all candidates rejected
    evs.append(FeedbackEvent(
        session_id="synth", policy_version="policy-sft", annotator="synth",
        context=_CTX, prompt="Pick the next recon step.",
        candidates=_cands("nmap -p- -T2 10.13.37.5",
                          "sqlmap -u http://10.13.37.5"),
        action=HumanAction.REJECTED_ALL,
        edited_text="whatweb http://10.13.37.5 && curl -sSI http://10.13.37.5/"))

    # 7. RATED only -> weak, no behavioral pick (flagged low weight)
    evs.append(FeedbackEvent(
        session_id="synth", policy_version="policy-sft", annotator="synth",
        context=_CTX, prompt="Summarize the attack surface so far.",
        candidates=[Candidate(cand_id="c0", text="SSH + HTTP exposed.", logprob=-5.0),
                    Candidate(cand_id="c1", text="Only HTTP.", logprob=-9.0)],
        action=HumanAction.RATED, scalar=4.0))

    # 8. SKIPPED -> no signal, dropped at build time
    evs.append(FeedbackEvent(
        session_id="synth", policy_version="policy-sft", annotator="synth",
        context=_CTX, prompt="(idle)",
        candidates=_cands("nmap 10.13.37.5"),
        action=HumanAction.SKIPPED))

    return evs


def write(store_root: str = "data/raw") -> int:
    store = EventStore(store_root)
    evs = sample_events()
    now = time.time()
    for i, ev in enumerate(evs):
        ev.ts = now + i * 0.001   # keep deterministic order within the day file
        store.append(ev)
    return len(evs)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw", help="EventStore root")
    args = ap.parse_args()
    n = write(args.out)
    print(f"wrote {n} synthetic events -> {args.out}")
