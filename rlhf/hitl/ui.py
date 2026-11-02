"""Annotation surface stub.

Recommended: Argilla — it's purpose-built for preference collection and gives
you a review queue, multi-annotator support, and export for free. This stub
shows the shape; the real win is logging the BEHAVIORAL signal (what ran, the
edit) automatically from the agent session, not just what a rater clicks here.

Two collection paths, both feeding the same EventStore:
  A) In-session capture (preferred): agent/session.py records RAN_AS_IS /
     RAN_EDITED / outcome automatically as the pentester works. No extra UI.
  B) Offline review (this file): for steps with no clear behavioral signal,
     a reviewer picks chosen/rejected and optionally rates.
"""
from __future__ import annotations

import os
import sys

from .events import FeedbackEvent, Candidate, HumanAction, ExecOutcome  # noqa: E402
from .store import EventStore  # noqa: E402


def record_in_session(session_id: str, policy_version: str, annotator: str,
                      context: str, prompt: str, candidates: list[dict],
                      *, ran_text: str, edited: bool,
                      outcome: str, outcome_detail: dict | None = None,
                      store_root: str = "data/raw") -> str:
    """Call this from the agent session loop once a step resolves."""
    ev = FeedbackEvent(
        session_id=session_id, policy_version=policy_version, annotator=annotator,
        context=context, prompt=prompt,
        candidates=[Candidate(**c) for c in candidates],
        action=HumanAction.RAN_EDITED if edited else HumanAction.RAN_AS_IS,
        chosen_id=None if edited else (candidates[0]["cand_id"] if candidates else None),
        edited_text=ran_text if edited else None,
        executed_text=ran_text,
        outcome=ExecOutcome(outcome),
        outcome_detail=outcome_detail,
    )
    EventStore(store_root).append(ev)
    return ev.event_id


# --- Argilla path ---------------------------------------------------------
def _ranking_values(n: int) -> list[str]:
    return [f"cand_{i}" for i in range(n)]


def argilla_settings(max_candidates: int = 6):
    """The dataset schema: show prompt+context+each candidate, ask for a ranking
    (best->worst) and an optional scalar. Argilla's ranking question is what maps
    cleanly onto our CHOSE/preference signal at export time."""
    import argilla as rg
    fields = [rg.TextField(name="prompt", title="Task"),
              rg.TextField(name="context", title="Engagement context")]
    fields += [rg.TextField(name=f"cand_{i}", title=f"Candidate {i}",
                            required=(i == 0)) for i in range(max_candidates)]
    return rg.Settings(
        guidelines="Rank the candidates best-first. Skip if none are usable.",
        fields=fields,
        questions=[
            rg.RankingQuestion(name="ranking", title="Best next action (rank)",
                               values=_ranking_values(max_candidates)),
            rg.RatingQuestion(name="scalar", title="Overall quality (1-5)",
                              values=[1, 2, 3, 4, 5], required=False),
            rg.TextQuestion(name="note", title="Rationale", required=False),
        ],
        metadata=[rg.TermsMetadataProperty(name="policy_version"),
                  rg.TermsMetadataProperty(name="session_id")],
    )


def _event_to_record(ev, max_candidates: int):
    import argilla as rg
    fields = {"prompt": ev.prompt or "(none)", "context": ev.context or "(none)"}
    for i in range(max_candidates):
        fields[f"cand_{i}"] = ev.candidates[i].text if i < len(ev.candidates) else ""
    return rg.Record(
        fields=fields,
        metadata={"policy_version": ev.policy_version or "unknown",
                  "session_id": ev.session_id or "unknown"},
        # carry our event_id so a reviewed record maps back to the raw log
        external_id=ev.event_id,
    )


def push_to_argilla(events, dataset: str = "pentest-prefs",
                    workspace: str = "default", max_candidates: int = 6):
    """Send events with no clear behavioral signal to Argilla for offline review.

    Only events that are NOT already resolved in-session are worth a human's
    time: a RAN_AS_IS / RAN_EDITED event already carries gold. We push the
    ambiguous ones (CHOSE-without-run, RATED, SKIPPED-but-has-candidates).

    pip install argilla ; set ARGILLA_API_URL / ARGILLA_API_KEY, then export the
    reviewed rankings back through argilla_export() -> FeedbackEvents.
    """
    import argilla as rg
    client = rg.Argilla()  # reads ARGILLA_API_URL / ARGILLA_API_KEY from env
    ds = client.datasets(name=dataset, workspace=workspace)
    if ds is None:
        ds = rg.Dataset(name=dataset, workspace=workspace,
                        settings=argilla_settings(max_candidates))
        ds.create()

    review_worthy = [e for e in events
                     if e.candidates and e.action in (
                         HumanAction.CHOSE, HumanAction.RATED, HumanAction.SKIPPED)]
    ds.records.log([_event_to_record(e, max_candidates) for e in review_worthy])
    return len(review_worthy)
