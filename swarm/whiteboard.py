"""Shared state for the swarm: a compact state dict, a per-phase whiteboard, the
consensus phase machine, and grounded confirmation of proposals.

Thread-safe: each agent runs in its own thread but one endpoint and one board.
The context an agent sees is deliberately small — the current state dict plus
only the current phase's whiteboard slice — so prompts stay near the token cap.
"""
from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field


@dataclass
class SubTask:
    """A unit of work the manager dispatches to the worker pool."""
    id: str
    phase: str
    instruction: str = ""         # natural-language check (small model turns it into a command)
    tool: str | None = None       # a concrete tool to run (no model call needed)
    args: dict | None = None
    toolkit: list = field(default_factory=list)   # tool/category names the worker may use
    status: str = "pending"       # pending | running | done
    by: str = ""                  # worker that executed it
    result: str = ""              # compressed digest


@dataclass
class Proposal:
    id: str
    agent: str
    phase: str
    claim: str
    confirmed_by: list[str] = field(default_factory=list)  # agents OTHER than the proposer

    @property
    def confirmed(self) -> bool:
        return bool(self.confirmed_by)


@dataclass
class Entry:
    agent: str
    phase: str
    kind: str          # run | propose | confirm | vote | note
    text: str
    t: float = field(default_factory=time.time)


class Whiteboard:
    def __init__(self, phases: list[str], agents: list[str], consensus: float,
                 gate_enabled: bool = False):
        self._lock = threading.RLock()
        self.phases = phases
        self.agents = agents
        self.consensus = consensus
        self.phase_idx = 0
        self.entries: list[Entry] = []
        self.proposals: dict[str, Proposal] = {}
        self._votes: dict[str, set[str]] = {p: set() for p in phases}  # phase -> agents voting done
        self._pid = 0
        # Human approval gate between phases. When consensus is reached and gating is
        # on, we DON'T advance — we park a `pending` gate, clear `gate` to block the
        # agents, and wait for the orchestrator to call confirm_advance() on approval.
        self.gate_enabled = gate_enabled
        self.gate = threading.Event()
        self.gate.set()                       # open (not gated) by default
        self.pending: tuple[str, str] | None = None   # (from_phase, to_phase) awaiting approval

        # Dispatch board: the manager posts SubTasks here; workers claim and run them.
        self.tasks: dict[str, SubTask] = {}
        self._queue: collections.deque = collections.deque()
        self._tid = 0
        self._task_cv = threading.Condition(self._lock)   # workers wait here for new tasks

    # ---- dispatch board (manager/worker) ----
    def dispatch(self, by: str, items: list) -> list:
        """Queue sub-tasks. Each item: {tool,args} (direct) or {check, toolkit} (NL)."""
        with self._lock:
            ids = []
            for it in items:
                self._tid += 1
                tid = f"T{self._tid}"
                st = SubTask(tid, self.phase, instruction=(it.get("check") or "").strip(),
                             tool=it.get("tool"), args=it.get("args"),
                             toolkit=it.get("toolkit") or it.get("tools") or [])
                self.tasks[tid] = st
                self._queue.append(tid)
                self._post(Entry(by, self.phase, "dispatch", f"{tid}: {st.tool or st.instruction}"))
                ids.append(tid)
            self._task_cv.notify_all()
            return ids

    def claim(self) -> SubTask | None:
        with self._lock:
            while self._queue:
                st = self.tasks[self._queue.popleft()]
                if st.status == "pending":
                    st.status = "running"
                    return st
            return None

    def wait_for_task(self, timeout: float) -> None:
        with self._lock:
            if not self._queue:
                self._task_cv.wait(timeout)

    def complete(self, tid: str, by: str, result: str) -> None:
        with self._lock:
            st = self.tasks.get(tid)
            if st:
                st.status = "done"
                st.by = by
                st.result = result
                self._post(Entry(by, st.phase, "result", f"{tid}: {result}"))

    def open_tasks(self) -> int:
        with self._lock:
            return sum(1 for s in self.tasks.values() if s.status != "done")

    def tasks_summary(self, limit: int = 14) -> str:
        with self._lock:
            rows = []
            for s in list(self.tasks.values())[-limit:]:
                label = s.tool or s.instruction
                if s.status == "done":
                    rows.append(f"[{s.id} ✓ {s.by}] {label} -> {s.result[:140]}")
                else:
                    rows.append(f"[{s.id} {s.status}] {label}")
            return "\n".join(rows) or "(no tasks dispatched yet)"

    @property
    def phase(self) -> str:
        with self._lock:
            return self.phases[min(self.phase_idx, len(self.phases) - 1)]

    @property
    def done(self) -> bool:
        with self._lock:
            return self.phase_idx >= len(self.phases)

    def _post(self, e: Entry) -> None:
        self.entries.append(e)

    def add_run(self, agent: str, cmd: str, digest: str) -> None:
        with self._lock:
            self._post(Entry(agent, self.phase, "run", f"$ {cmd}\n{digest}"))

    def add_note(self, agent: str, text: str) -> None:
        with self._lock:
            self._post(Entry(agent, self.phase, "note", text))

    def add_proposal(self, agent: str, claim: str) -> str:
        with self._lock:
            self._pid += 1
            pid = f"P{self._pid}"
            self.proposals[pid] = Proposal(pid, agent, self.phase, claim)
            self._post(Entry(agent, self.phase, "propose", f"{pid}: {claim}"))
            return pid

    def add_confirmation(self, agent: str, pid: str, digest: str, supports: bool) -> str:
        """Record one agent's attempt to verify another's proposal.

        Grounded: only counts if a DIFFERENT agent ran a check whose output
        supports the claim. Returns a human-readable status for logging.
        """
        with self._lock:
            prop = self.proposals.get(pid)
            if prop is None:
                self._post(Entry(agent, self.phase, "confirm", f"{pid}: unknown proposal"))
                return f"unknown proposal {pid}"
            if agent == prop.agent:
                self._post(Entry(agent, self.phase, "confirm", f"{pid}: self-confirm ignored"))
                return "self-confirmation ignored"
            if supports and agent not in prop.confirmed_by:
                prop.confirmed_by.append(agent)
            verdict = "CONFIRMED" if supports else "could not confirm"
            self._post(Entry(agent, self.phase, "confirm", f"{pid} {verdict} by {agent}\n{digest}"))
            return f"{pid} {verdict}"

    def vote_phase_done(self, agent: str, done: bool) -> bool:
        """Record a vote; advance the phase if consensus is reached. Returns True if advanced."""
        with self._lock:
            phase = self.phase
            if done:
                self._votes[phase].add(agent)
            else:
                self._votes[phase].discard(agent)
            self._post(Entry(agent, phase, "vote", f"phase_done={done} "
                             f"({len(self._votes[phase])}/{len(self.agents)})"))
            need = max(1, round(self.consensus * len(self.agents)))
            if len(self._votes[phase]) >= need and self.phase_idx < len(self.phases):
                if self.gate_enabled:
                    if self.pending is None:          # park a gate; wait for human approval
                        nxt = (self.phases[self.phase_idx + 1]
                               if self.phase_idx + 1 < len(self.phases) else "complete")
                        self.pending = (phase, nxt)
                        self.gate.clear()
                    return False                      # do not advance yet
                self.phase_idx += 1
                return True
            return False

    def confirm_advance(self) -> None:
        """Approve the parked gate: advance one phase and release the agents."""
        with self._lock:
            if self.pending is not None:
                self.phase_idx += 1
                self.pending = None
            self.gate.set()

    def open_gate(self) -> None:
        """Force the gate open (e.g. on stop) so blocked agents can exit."""
        self.gate.set()

    def state_dict(self) -> dict:
        """The compact, model-facing state (no raw output)."""
        with self._lock:
            phase = self.phase if not self.done else "complete"
            confirmed = [p.id for p in self.proposals.values() if p.confirmed]
            return {
                "phase": phase,
                "phases": self.phases,
                "open_proposals": [
                    {"id": p.id, "by": p.agent, "claim": p.claim, "confirmed": p.confirmed}
                    for p in self.proposals.values() if p.phase == phase or not p.confirmed
                ][-8:],
                "confirmed_count": len(confirmed),
                "votes_this_phase": len(self._votes.get(phase, ())),
                "agents": len(self.agents),
            }

    def phase_slice(self, limit: int = 10) -> list[Entry]:
        """Only the current phase's recent entries — the whiteboard slice."""
        with self._lock:
            phase = self.phase if not self.done else self.phases[-1]
            return [e for e in self.entries if e.phase == phase][-limit:]

    def shared_findings(self, limit: int = 14) -> list[str]:
        """The hive mind: every agent's findings so far, across ALL steps, attributed
        and marked confirmed/unconfirmed. This is shared context for the whole task."""
        with self._lock:
            out = []
            for p in list(self.proposals.values())[-limit:]:
                if p.confirmed:
                    tag = "confirmed by " + ", ".join(p.confirmed_by)
                else:
                    tag = "unconfirmed"
                out.append(f"[{p.id} · {p.agent}] {p.claim} ({tag})")
            return out
