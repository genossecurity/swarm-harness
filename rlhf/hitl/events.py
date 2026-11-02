"""Feedback event schema — the core of the whole pipeline.

Everything downstream (pair construction, reward, training) reads these.
Get this right and the rest is plumbing; get it wrong and no amount of
training technique recovers the signal.

Design decisions baked in here:
  - Capture k candidate outputs per step, not just the one shown.
  - The strongest pentester signals are behavioral, not a rating:
    which candidate they actually RAN, and how they EDITED it.
    Rank of signal strength: accepted-and-ran > explicit choice > scalar.
  - Record the execution OUTCOME separately from the human judgment.
    The environment scores verifiable steps (port found? shell popped?);
    the human scores tradecraft/prioritization/false-positive calls.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class HumanAction(str, Enum):
    """What the pentester did with the agent's suggestions — in signal order."""
    RAN_AS_IS = "ran_as_is"          # accepted a candidate verbatim and executed it
    RAN_EDITED = "ran_edited"        # edited a candidate, then executed (edit = gold)
    CHOSE = "chose"                  # picked one candidate, did not (yet) run it
    REJECTED_ALL = "rejected_all"    # none useful; wrote their own
    RATED = "rated"                  # scalar only, no behavioral signal
    SKIPPED = "skipped"              # no signal — drop at pair-build time


class ExecOutcome(str, Enum):
    SUCCESS = "success"              # verifiable objective met
    FAILURE = "failure"             # ran, did not achieve objective
    ERROR = "error"                 # malformed / tool error
    NOT_RUN = "not_run"             # no execution attached to this step
    UNKNOWN = "unknown"


@dataclass
class Candidate:
    """One sampled output from the policy for a given step."""
    cand_id: str
    text: str                        # the suggestion (command, analysis, next step)
    logprob: Optional[float] = None  # policy logprob if available (useful for debugging)
    tool_call: Optional[dict] = None # structured call if the step is a tool invocation


@dataclass
class FeedbackEvent:
    """One decision point: context -> k candidates -> human signal -> outcome."""
    # identity / provenance
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""             # engagement episode this belongs to
    policy_version: str = ""         # which model produced the candidates
    ts: float = field(default_factory=time.time)
    annotator: str = ""             # who gave the signal (pentester id)

    # the decision point
    system_prompt: str = ""
    context: str = ""               # recon state + prior steps, serialized
    prompt: str = ""                # the immediate instruction/question
    candidates: list[Candidate] = field(default_factory=list)

    # human signal
    action: HumanAction = HumanAction.SKIPPED
    chosen_id: Optional[str] = None  # cand_id picked (CHOSE/RAN_*)
    edited_text: Optional[str] = None  # final text after human edit (RAN_EDITED)
    scalar: Optional[float] = None   # optional 1-5 / -1..1 rating
    note: Optional[str] = None       # free-text rationale, kept for audit not training

    # environment signal (verifiable reward)
    executed_text: Optional[str] = None   # what actually ran (may differ from any candidate)
    outcome: ExecOutcome = ExecOutcome.NOT_RUN
    outcome_detail: Optional[dict] = None  # e.g. {"ports":[22,80]}, {"shell":true}

    def to_json(self) -> dict:
        d = asdict(self)
        d["action"] = self.action.value
        d["outcome"] = self.outcome.value
        return d

    @staticmethod
    def from_json(d: dict) -> "FeedbackEvent":
        d = dict(d)
        d["candidates"] = [Candidate(**c) for c in d.get("candidates", [])]
        d["action"] = HumanAction(d["action"])
        d["outcome"] = ExecOutcome(d.get("outcome", "not_run"))
        return FeedbackEvent(**d)

    def is_usable(self) -> bool:
        """Does this event carry a signal we can turn into a pair?"""
        if self.action == HumanAction.SKIPPED:
            return False
        if not self.candidates:
            return False
        return True
