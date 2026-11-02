"""Episode loop: the glue that produces feedback events during an engagement.

One Session = one authorized engagement against the lab. Each step:
  1. policy samples k candidates for the current decision point
  2. the pentester picks / edits one (or writes their own)
  3. the chosen command runs through the sandboxed tool layer
  4. the outcome + human signal is recorded as a FeedbackEvent

The human is the loop. This file does NOT auto-execute candidates — a command
only runs when the pentester commits to one. That keeps a human accountable for
every real action and keeps the "what ran" signal honest.
"""
from __future__ import annotations

import os
import sys
import uuid
from dataclasses import dataclass, field


from .policy import Policy, GenConfig          # noqa: E402
from toolmcp import sandbox as tools   # MCP enforcement layer
from .hitl.events import FeedbackEvent, Candidate, HumanAction, ExecOutcome  # noqa: E402
from .hitl.store import EventStore                   # noqa: E402


@dataclass
class Session:
    annotator: str
    scope_cidr: str                            # authorization boundary, for the record
    policy_version: str = "policy-sft"
    system_prompt: str = (
        "You are a penetration-testing assistant operating under an explicit, "
        "signed engagement. Only act within the authorized scope. Suggest the "
        "single best next action with a one-line rationale."
    )
    store_root: str = "data/raw"
    session_id: str = field(default_factory=lambda: f"sess-{uuid.uuid4().hex[:8]}")

    def __post_init__(self):
        self.policy = Policy(GenConfig(model_path=f"./registry/{self.policy_version}"))
        self.store = EventStore(self.store_root)
        self.history: list[str] = []

    def _context(self, recon_state: str) -> str:
        lines = [f"Scope: {self.scope_cidr} (authorized).", recon_state]
        if self.history:
            lines.append("Prior steps:\n" + "\n".join(self.history[-8:]))
        return "\n".join(lines)

    def step(self, recon_state: str, prompt: str) -> list[dict]:
        """Produce candidates for the pentester to review. Returns candidate dicts."""
        ctx = self._context(recon_state)
        cands = self.policy.sample(self.system_prompt, f"{ctx}\n\n{prompt}")
        self._pending = {"ctx": ctx, "prompt": prompt, "cands": cands}
        return cands

    def resolve(self, *, chosen_id: str | None, final_text: str,
                run: bool = True) -> str:
        """Record what the pentester decided. `final_text` is what they commit to
        (a candidate verbatim, an edit of one, or their own). If run=True it goes
        through the sandbox; the outcome is captured.

        Returns the recorded event_id.
        """
        p = self._pending
        edited = True
        if chosen_id is not None:
            orig = next((c["text"] for c in p["cands"] if c["cand_id"] == chosen_id), None)
            edited = (orig is None) or (orig.strip() != final_text.strip())

        outcome, detail = ExecOutcome.NOT_RUN, None
        if run:
            res = tools.execute(final_text)          # sandbox-enforced
            outcome = ExecOutcome(res.outcome)
            detail = res.detail

        if chosen_id is None:
            action = HumanAction.REJECTED_ALL
        elif edited:
            action = HumanAction.RAN_EDITED if run else HumanAction.CHOSE
        else:
            action = HumanAction.RAN_AS_IS if run else HumanAction.CHOSE

        ev = FeedbackEvent(
            session_id=self.session_id, policy_version=self.policy_version,
            annotator=self.annotator, system_prompt=self.system_prompt,
            context=p["ctx"], prompt=p["prompt"],
            candidates=[Candidate(**c) for c in p["cands"]],
            action=action, chosen_id=chosen_id,
            edited_text=final_text if (edited or chosen_id is None) else None,
            executed_text=final_text if run else None,
            outcome=outcome, outcome_detail=detail,
        )
        self.store.append(ev)
        self.history.append(f"$ {final_text}  -> {outcome.value}")
        return ev.event_id


if __name__ == "__main__":
    # Illustrative dry run (no model, no execution).
    print("Session is a library; import and drive it from your engagement console.")
