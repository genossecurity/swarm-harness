"""One agent: a loop of (build small context) -> (ask brain for JSON action) ->
(act on target / whiteboard) -> (log event). Context discipline is enforced
here: state dict + current-phase slice only, clamped to the char cap.
"""
from __future__ import annotations

import json

from .client import LLMClient, TurnResult
from .compress import compress
from .config import RunConfig
from .executor import Executor
from .protocol import Action
from .tools import ToolError, build_command, render_tools, tools_for_step
from .whiteboard import Whiteboard

LEAD_SYSTEM = (
    "You are the LEAD agent {aid}, coordinating a swarm of {n} agents on one target and one "
    "whiteboard. Finish each step: {phases}. You synthesize the workers' checks into findings "
    "and decide when a step is done. Actions: 'tool' to call a listed tool (preferred); 'run' a "
    "raw shell command; 'propose' a FINDING as a factual sentence (e.g. \"TCP 22 is open\" - "
    "never a command); 'confirm' another agent's proposal by running your own check; 'vote' when "
    "the step is done; 'note' to record a thought. Prefer turning the shared findings into clear "
    "proposals and confirming unconfirmed ones over repeating work. {tools} Keep commands safe "
    "and read-only unless the step calls for more. One JSON object per turn."
)

WORKER_SYSTEM = (
    "You are WORKER sub-agent {aid}, one of {n} agents. Your job is quick ONE-LINER CHECKS for "
    "the current step: {phases}. Each turn do exactly ONE small thing: call a 'tool' (preferred) "
    "or 'run' a single command to gather one fact, or 'confirm' an unconfirmed finding from "
    "another agent by running your own one-line check. If you discover a clear fact, you may "
    "'propose' it as a short sentence; otherwise leave big synthesis to the lead. 'vote' when the "
    "step looks done. Keep it to a single command — no long reasoning. {tools} One JSON object per turn."
)


class Agent:
    def __init__(self, aid: str, cfg: RunConfig, client: LLMClient,
                 board: Whiteboard, executor: Executor, emit):
        self.aid = aid
        self.cfg = cfg
        self.client = client
        self.board = board
        self.ex = executor
        self.emit = emit          # callback(event_dict) -> appends to run.jsonl
        self.available_tools: list[str] = []   # probing tools detected on the host (set by orchestrator)
        self.stop_event = None    # threading.Event set by the orchestrator to end the run
        self.role = "lead"        # "lead" (strong model) or "worker" (one-liner checks on the small model)
        self.model_label = ""     # e.g. "gpt-oss:20b @ localhost" — for logging/UI
        self._sigcount: dict[str, int] = {}   # how often this agent has run each check (per phase)
        self.steps = 0
        self.retries = 0
        self.turns = 0

    def _tools_hint(self) -> str:
        if not self.available_tools:
            return ""
        return (f"Probing binaries installed on the host: {', '.join(self.available_tools)}. "
                f"Do not call binaries that are not listed.")

    def _step_tool_specs(self):
        """Tool specs allowed in the current step and installed on the host."""
        allow = self.cfg.step_tools.get(self.board.phase)   # None -> all tools
        return tools_for_step(allow, self.available_tools or None)

    def _allowed_tool_names(self):
        return {s.name for s in self._step_tool_specs()}

    def _context(self) -> str:
        state = self.board.state_dict()
        phase = state["phase"]
        slice_ = self.board.phase_slice()
        lines = [f"[{e.agent}/{e.kind}] {e.text}" for e in slice_]
        findings = self.board.shared_findings()

        # Each step can carry its own objective; fall back to the overall task.
        step_task = self.cfg.step_tasks.get(phase) or self.cfg.task
        head = f"OVERALL TASK: {self.cfg.task}\n" if self.cfg.task else ""
        head += f"CURRENT STEP ({phase}) OBJECTIVE: {step_task}\n\n" if step_task else ""
        tgt = self.cfg.target
        if tgt and tgt.is_remote_subject:
            head += (f"SUBJECT HOST: {tgt.host} — remote, you have NO login to it. Run commands on "
                     f"THIS machine to probe it over the network. Confirm findings from that output.\n\n")
        tool_doc = render_tools(self._step_tool_specs())
        if tool_doc:
            head += tool_doc + "\n\n"

        if self.role == "worker":
            # Workers do one-liner checks on a small model: give them a short, low-ambiguity
            # context (no big STATE dump) so their JSON stays reliable and fast.
            import re as _re
            unconf = [f for f in findings if f.endswith("(unconfirmed)")]
            if unconf:
                # Point them at ONE unconfirmed finding and hand them a near-filled confirm.
                target = unconf[0]
                pid = (target.split("·")[0].strip("[ ")) if "·" in target else "P?"
                host = self.cfg.target.host if self.cfg.target else ""
                port_m = _re.search(r"\b(\d{1,5})\b", target.split("]")[-1])
                ex_tool = ('"tool":"tcp_connect","args":{"host":"%s","port":%s}' % (host, port_m.group(1))) \
                    if port_m and host else '"tool":"ping","args":{"host":"%s"}' % host
                body = (
                    "There is an UNCONFIRMED finding to VERIFY:\n" + target +
                    f"\n\nConfirm it now by running your own check. Reply with EXACTLY this shape "
                    f'(adjust args if needed):\n'
                    f'{{"action":"confirm","confirm":"{pid}",{ex_tool},"reason":"verify {pid}"}}'
                )
            elif findings:
                # Everything proposed is already confirmed — vote the step done.
                body = (
                    "FINDINGS SO FAR (all confirmed):\n" + "\n".join(findings) +
                    f'\n\nNothing is unconfirmed. Vote the step done — reply with EXACTLY:\n'
                    f'{{"action":"vote","phase_done_vote":true,"reason":"findings confirmed"}}'
                )
            else:
                body = (
                    "FINDINGS SO FAR:\n(nothing yet)" +
                    f"\n\nYou are {self.aid}. Do ONE check now: call a tool to gather a new fact. "
                    f"Reply with ONE JSON object."
                )
        else:
            unconf = [f for f in findings if f.endswith("(unconfirmed)")]
            if findings and not unconf:
                nudge = ("\nAll findings are confirmed and the objective is met. Unless you have a "
                         "genuinely NEW fact to add, VOTE the step done: "
                         '{"action":"vote","phase_done_vote":true,"reason":"..."}. '
                         "Do not confirm your own proposals, and do not re-run a check already above.")
            else:
                nudge = ("\nBuild on these. Do NOT repeat a probe whose finding is already here; "
                         "instead propose a new finding, or vote the step done when the objective is met.")
            body = (
                "SHARED FINDINGS (the hive mind — what every agent has found so far, all steps):\n" +
                ("\n".join(findings) or "(nothing yet)") + nudge + "\n\n" +
                "STATE:\n" + json.dumps(state) +
                "\n\nTHIS STEP'S ACTIVITY:\n" + ("\n".join(lines) or "(empty)") +
                f"\n\nYou are {self.aid}. Current step: {phase}. Decide your next action."
            )
        # Clamp only the body (keep the most recent context); the task/objective always survives.
        budget = max(0, self.cfg.ctx_cap_chars - len(head))
        return head + body[-budget:]

    def _resolve_cmd(self, a: Action):
        """A run/confirm turn may specify a raw cmd or a named tool. Return (cmd, tool)
        or raise ToolError if the tool call is invalid/not allowed this step."""
        if a.tool:
            cmd = build_command(a.tool, a.args or {}, self._allowed_tool_names())
            return cmd, a.tool
        return a.cmd, None

    def _act(self, a: Action) -> dict:
        """Perform the action's side effect; return a log 'result' payload."""
        if a.action in ("run", "tool"):
            try:
                cmd, tool = self._resolve_cmd(a)
            except ToolError as e:
                self.board.add_note(self.aid, f"tool error: {e}")
                return {"error": str(e), "tool": a.tool}
            res = self.ex.run(cmd)
            digest = compress(res)
            self.board.add_run(self.aid, cmd, digest)
            return {"cmd": cmd, "tool": tool, "rc": res.rc, "digest": digest}
        if a.action == "propose":
            pid = self.board.add_proposal(self.aid, a.proposal)
            return {"proposal_id": pid, "claim": a.proposal}
        if a.action == "confirm":
            try:
                cmd, tool = self._resolve_cmd(a)
            except ToolError as e:
                self.board.add_note(self.aid, f"tool error: {e}")
                return {"target_proposal": a.confirm, "error": str(e), "supports": False,
                        "status": f"tool error: {e}"}
            res = self.ex.run(cmd)
            digest = compress(res)
            supports = res.rc == 0 and not res.timed_out
            status = self.board.add_confirmation(self.aid, a.confirm, digest, supports)
            return {"target_proposal": a.confirm, "cmd": cmd, "tool": tool, "rc": res.rc,
                    "digest": digest, "supports": supports, "status": status}
        if a.action == "vote":
            advanced = self.board.vote_phase_done(self.aid, bool(a.phase_done_vote))
            return {"vote": a.phase_done_vote, "phase_advanced": advanced}
        self.board.add_note(self.aid, a.reason)      # note
        return {"note": a.reason}

    def _guard_repeat(self, a: Action, phase: str) -> Action:
        """Small models (and stuck leads) re-run the same check forever. If an agent
        has already run this exact check twice in this phase, convert the turn into a
        'phase done' vote so the swarm converges instead of looping to max_steps."""
        if a.action not in ("run", "tool", "confirm"):
            return a
        sig = f"{phase}|{a.action}|{a.tool or a.cmd}|{json.dumps(a.args, sort_keys=True) if a.args else ''}|{a.confirm or ''}"
        self._sigcount[sig] = self._sigcount.get(sig, 0) + 1
        if self._sigcount[sig] >= 3:                 # already did it twice — stop repeating
            return Action(action="vote", phase_done_vote=True,
                          reason="auto: already ran this check; voting the step done to move on")
        return a

    def stopped(self) -> bool:
        return self.stop_event is not None and self.stop_event.is_set()

    def step(self) -> bool:
        """Run one turn. Returns False when this agent should stop."""
        if self.board.done or self.steps >= self.cfg.max_steps or self.stopped():
            return False
        # Pause here while a human approval gate is open (phase complete, awaiting "move on?").
        while not self.board.gate.is_set():
            if self.stopped():
                return False
            self.board.gate.wait(0.2)
        if self.board.done or self.stopped():
            return False
        self.steps += 1
        self.turns += 1
        phase_before = self.board.phase
        template = WORKER_SYSTEM if self.role == "worker" else LEAD_SYSTEM
        tr = self.client.turn(template.format(aid=self.aid, n=len(self.board.agents),
                                              phases=", ".join(self.cfg.phases),
                                              tools=self._tools_hint()),
                              self._context(), stop_check=self.stopped)
        self.retries += tr.retries
        action = self._guard_repeat(tr.action, phase_before)
        result = self._act(action)
        tr = TurnResult(action, tr.retries, tr.ok)
        self.emit({
            "agent": self.aid,
            "step": self.steps,
            "phase": phase_before,
            "action": tr.action.action,
            "reason": tr.action.reason,
            "retries": tr.retries,
            "parse_ok": tr.ok,
            "result": result,
        })
        return True
