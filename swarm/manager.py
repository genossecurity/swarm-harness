"""Manager/worker orchestration.

The lead agent is a MANAGER: for any task it plans and dispatches discrete checks
to a pool of worker sub-agents, assigning each check a toolkit from the catalog.
Workers claim tasks from the dispatch board and run them — a concrete {tool,args}
runs directly (no model call), a natural-language check is turned into a command
by a worker agent using the shared model. Each check runs once (no N-way redundancy); the manager
reads the results, synthesizes findings, and decides when the step is done. Every agent uses the
same configured model; worker agents are concurrent roles, not a separate model tier.
"""
from __future__ import annotations

import threading
import time

from .client import LLMClient
from .compress import compress
from toolmcp.executor import ExecResult
from .protocol import parse_action, ProtocolError
from toolmcp.tools import ToolError, build_command, render_catalog, render_tools, toolkit

MANAGER_SYSTEM = (
    "You are the MANAGER of a worker pool working this task. Break the current step into small "
    "checks and DISPATCH them to workers — you do not run probes yourself. Read the results on the "
    "board, record conclusions as findings, and declare the step done when its objective is met.\n"
    "Actions (ONE JSON object):\n"
    '- {"action":"plan","goals":[{"goal":"<what to establish>","expect":"<evidence that will satisfy it>"}, ...],"reason":"..."}'
    "  (FIRST thing EVERY step: list each goal AND what you expect to find for it)\n"
    '- {"action":"dispatch","tasks":[{"tool":"<name>","args":{...}} | '
    '{"check":"<what to find>","toolkit":["<category or tool>", ...]}],"reason":"..."}\n'
    '- {"action":"ask","question":"<one focused question for the user>","reason":"..."}\n'
    '- {"action":"propose","proposal":"<finding as a sentence>","reason":"..."}\n'
    '- {"action":"vote","phase_done_vote":true,"reason":"..."}  (step objective met)\n'
    '- {"action":"note","reason":"..."}\n'
    "Begin each step by PLANNING its goals, then DISPATCH the checks that meet them. Dispatch a "
    "concrete tool call when you know exactly what to run; dispatch a 'check' with a toolkit when a "
    "worker should figure out the command. Don't dispatch work already done on the board. When the "
    "findings answer the objective, vote the step done. The CURRENT STEP objective defines this "
    "step's scope; use the overall task only as background or to identify a genuine prerequisite. "
    "Reuse findings from earlier steps instead of repeating their checks."
)

WORKER_EXEC_SYSTEM = (
    "You are a worker. Turn ONE assigned check into ONE shell/tool action and nothing else. "
    "You have NO functions; never emit a tool call. Output ONLY a raw JSON object: "
    '{"action":"tool","tool":"<name>","args":{...},"reason":"..."} or '
    '{"action":"run","cmd":"<command>","reason":"..."}.'
)

ADAPTIVE_PIPELINE_SYSTEM = (
    "You are the adaptive test planner for a scoped host assessment. Choose the next useful checks "
    "from the structured tools listed in the user context, based on the task and evidence gathered "
    "so far. Prefer a small, specific follow-up that could change the conclusion; do not run every "
    "scanner just because it exists. Do not repeat a completed check. A server banner or fingerprint "
    "is a lead, not proof of a vulnerability. Only target the supplied host and discovered URLs.\n"
    'Reply with one JSON action: {"action":"dispatch","tasks":[{"tool":"<listed tool>","args":{...}}],'
    '"reason":"why these checks follow from the evidence"} or '
    '{"action":"vote","phase_done_vote":true,"reason":"the task is answered or no useful allowed check remains"} or '
    '{"action":"ask","question":"one focused question"} when a material scope/detail is missing. '
    "Dispatch at most two checks at once. Never use raw shell commands or natural-language worker checks."
)


class ManagerWorker:
    def __init__(self, orch):
        self.o = orch                      # the Orchestrator (board, agents, emit, stop, cfg, gate...)
        self.board = orch.board
        self.cfg = orch.cfg
        self._dispatched: set = set()      # signatures of checks already sent (dedup)
        self._planned: set = set()         # phases the manager has already set goals for
        self._goals: dict = {}             # phase -> [goal, ...] (the step's checklist)
        self._plan_tries: dict = {}        # phase -> count of non-plan turns before goals were set
        self._captured: set = set()        # task ids whose result was auto-captured as a finding
        self._proposal_sources: dict[str, str] = {}  # proposal id -> completed task id
        self._raw_results: dict[str, object] = {}    # private in-memory output for deterministic verification
        self._proposed_tokens: list = []   # token sets of recorded findings (semantic dedup)
        self._user_context: list[dict] = []
        self._pipeline_announced: set[str] = set()
        self._wts: list = []               # live worker threads (elastic pool)

    # ---- worker pool ----
    @staticmethod
    def _target_args(tool: str, args: dict | None, host: str) -> dict:
        """Fill safe target defaults for structured checks when the model omits them."""
        from toolmcp.tools import BUILTIN_TOOLS
        out = dict(args or {})
        spec = BUILTIN_TOOLS.get(tool)
        if spec and "host" in spec.params:
            out.setdefault("host", host)
        if tool == "dns_lookup":
            out.setdefault("name", host)
        if tool in ("http_head", "http_get") and not out.get("url"):
            target = out.get("host") or host
            if target:
                port = str(out.get("port") or "").strip()
                scheme = str(out.get("scheme") or ("https" if port in ("443", "8443") else "http"))
                if "://" in target:
                    url = target.rstrip("/")
                else:
                    url = f"{scheme}://{target}"
                if port and not (port == "80" and scheme == "http") and not (port == "443" and scheme == "https"):
                    url += f":{port}"
                out["url"] = url
        return out

    def _worker_loop(self, agent) -> None:
        host = self.cfg.target.host if self.cfg.target else ""
        try:
            while not self.o._stop.is_set() and not self.board.done:
                if not self.board.gate.is_set():            # paused at a human gate
                    self.board.gate.wait(0.3)
                    continue
                task = self.board.claim()
                if task is None:
                    self.board.wait_for_task(0.5)
                    continue
                try:
                    self._do_task(agent, task, host)
                except Exception as e:                   # never let one bad task kill a worker
                    self.board.complete(task.id, agent.aid, f"[error] {type(e).__name__}: {e}")
                    self.o._emit({"agent": agent.aid, "action": "task", "phase": task.phase,
                                  "task": task.id, "result": {"task": task.id, "error": str(e)}})
        finally:
            agent.ex.close()

    def _do_task(self, agent, task, host) -> None:
        agent.turns += 1
        from toolmcp.tools import BUILTIN_TOOLS, CATEGORIES
        # A manager sometimes puts a category ("ports") where a tool goes — treat it as a toolkit check.
        if task.tool and task.tool not in BUILTIN_TOOLS and task.tool in CATEGORIES:
            task.toolkit = list(task.toolkit) + [task.tool]
            task.instruction = task.instruction or f"{task.tool} check on the host"
            task.tool = None
        via, tool, cmd, err = "direct", task.tool, None, None
        specs = toolkit(task.toolkit, agent.available_tools)
        allowed = {s.name for s in specs} if specs else None
        if task.tool:
            try:
                args = self._target_args(task.tool, task.args, host)
                cmd = build_command(task.tool, args, allowed)
            except ToolError as e:
                err = str(e)
        else:
            # natural-language check -> a worker agent on the shared model generates the command
            via = "model"
            sys = WORKER_EXEC_SYSTEM + "\n" + render_tools(specs)
            usr = (f"CHECK: {task.instruction}\nSubject host: {host}\n"
                   f"Reply with ONE JSON tool/run action.")
            tr = agent.client.turn(sys, usr, stop_check=lambda: self.o._stop.is_set())
            a = tr.action
            try:
                if a.tool:
                    args = self._target_args(a.tool, a.args, host)
                    cmd, tool = build_command(a.tool, args, allowed), a.tool
                elif a.cmd:
                    cmd = a.cmd
                else:
                    err = "worker produced no command"
            except ToolError as e:
                err = str(e)

        if err:
            self.board.complete(task.id, agent.aid, f"[error] {err}")
            digest, rc = f"[error] {err}", None
        else:
            timeout = next((s.timeout for s in specs if s.name == tool), 30.0)
            res = agent.ex.run(cmd, timeout=timeout)
            self._raw_results[task.id] = ExecResult(
                res.cmd, res.rc, res.stdout[:65536], res.stderr[:8192], res.timed_out)
            digest, rc = compress(res), res.rc
            self.board.complete(task.id, agent.aid, digest)
        self.o._emit({"agent": agent.aid, "action": "task", "phase": task.phase, "task": task.id,
                      "via": via, "result": {"task": task.id, "tool": tool, "cmd": cmd,
                                              "rc": rc, "digest": digest, "instruction": task.instruction}})

    # ---- manager ----
    def _manager_context(self, agent) -> str:
        phase = self.board.phase
        step_task = self.cfg.step_tasks.get(phase) or self.cfg.task
        host = self.cfg.target.host if self.cfg.target else ""
        findings = self.board.shared_findings()
        open_n = self.board.open_tasks()
        done_n = len(self.board.tasks) - open_n
        open_goals = self._open_goals()
        if phase not in self._planned and done_n == 0 and open_n == 0:
            directive = ("This step has just begun. FIRST state its goals ONCE: output a single 'plan' action; "
                         "for EACH goal give what you EXPECT to find ('expect'). Do not dispatch yet, do not plan again.")
        elif phase in self._planned and done_n == 0 and open_n == 0:
            directive = ("Goals are set but you have dispatched NOTHING. Your ONLY valid next move is a "
                         "'dispatch' action with the checks that meet those goals. Do not plan, note, or vote.")
        elif open_n > 0:
            directive = ("Checks are still running (see board). WAIT — do not dispatch anything; the results "
                         "will arrive. Take no action that repeats work already in flight.")
        elif open_goals:
            directive = ("The checks are complete. Your CHECKLIST still has unmet goals (below). For each, PROPOSE "
                         "a finding stating the concrete fact from the results that satisfies it. Only dispatch a new "
                         "check if a goal truly has no data yet.")
        elif done_n >= 1:
            directive = ("Every goal on the checklist is met and all checks are complete. Output a 'vote' with "
                         "phase_done_vote=true to close this step. Do not dispatch or propose anything more.")
        else:
            directive = "Dispatch the checks needed to meet the goals (don't repeat checks already on the board)."
        toolkit_line = ""
        if getattr(self.cfg, "auto_tools", False):
            kit = self._step_toolkit(phase)
            if kit:
                toolkit_line = (f"SUGGESTED TOOLKIT for this step: {kit} — dispatch checks with these "
                                f"categories (you may override when you know a specific tool).")
        parts = [
            f"OVERALL TASK: {self.cfg.task}" if self.cfg.task else "",
            f"CURRENT STEP ({phase}) OBJECTIVE: {step_task}" if step_task else "",
            f"SUBJECT HOST: {host} (probe it over the network from this machine)" if host else "",
            toolkit_line,
            "",
            self._render_checklist(),
            "",
            render_catalog(agent.available_tools),
            "",
            "FINDINGS SO FAR:\n" + ("\n".join(findings) or "(none yet)"),
            "USER CLARIFICATIONS:\n" + ("\n".join(
                f"Q: {x['question']}\nA: {x['answer']}" for x in self._user_context) or "(none)"),
            "",
            "DISPATCH BOARD (checks you sent + their results):\n" + self.board.tasks_summary(),
            "",
            directive,
        ]
        ctx = "\n".join(p for p in parts if p is not None)
        return ctx[-self.cfg.ctx_cap_chars:]

    def _manager_loop(self, agent) -> None:
        steps, stall = 0, 0
        while not self.o._stop.is_set() and not self.board.done and steps < self.cfg.max_steps:
            if not self.board.gate.is_set():               # parked at a gate; supervisor handles approval
                self.board.gate.wait(0.3)
                continue
            if self.cfg.pipeline and self.board.phase not in self._pipeline_announced:
                phase = self.board.phase
                self._pipeline_announced.add(phase)
                stage = next((s for s in self.cfg.pipeline_stages if s.get("key") == phase), {})
                self.o._emit({"event": "pipeline_stage", "phase": phase,
                              "index": self.board.phase_idx + 1, "total": len(self.cfg.phases),
                              "objective": self.cfg.step_tasks.get(phase, ""),
                              "title": stage.get("title", phase.replace("_", " ").title()),
                              "tools": self.cfg.step_tools.get(phase, [])})
            steps += 1
            agent.turns += 1
            # Convergence guard: only auto-close after successful work and a satisfied checklist.
            if self.board.open_tasks() == 0:
                if stall >= 2 and self._dispatch_fallback_goals(agent, steps):
                    stall = 0
                    continue
                if stall >= 5:
                    self._autocapture(agent, steps)
                    self._sync_checklist()
                    if self._phase_satisfied():
                        self._accept_findings(agent)
                        phase_status = self.board.finish_phase(agent.aid)
                        self._emit_manager(agent, type("A", (), {"action": "vote", "reason": "auto: no new progress"})(),
                                           steps, {"vote": True, "auto": True, "phase": phase_status,
                                                   "phase_advanced": phase_status in ("advanced", "complete")})
                        stall = 0
                        continue
                    self.o._emit({"event": "phase_blocked", "phase": self.board.phase,
                                  "reason": "step stalled before its goals were met; no results were discarded"})
                    break
            ask_instructions = (" If important details are missing and guessing could lead to the wrong work, "
                                "ask the user one focused question with action=ask. Continue autonomously when "
                                "the task is clear enough.") if self.cfg.interactive_questions else ""
            tr = agent.client.turn(
                MANAGER_SYSTEM + f"\n\nSteps: {', '.join(self.cfg.phases)}." + ask_instructions,
                self._manager_context(agent), stop_check=lambda: self.o._stop.is_set())
            agent.retries += tr.retries
            a = tr.action
            result = {}
            phase = self.board.phase
            # Every step BEGINS with goals. Until this step is planned, the only valid move is 'plan';
            # if the manager won't, synthesize goals from the step objective so no step runs goalless.
            if phase not in self._planned and a.action not in ("plan", "ask"):
                self._plan_tries[phase] = self._plan_tries.get(phase, 0) + 1
                if self._plan_tries[phase] >= 2:
                    a = self._auto_plan(phase)
                else:
                    self.board.add_note(agent.aid, "awaiting goals: plan this step first")
                    stall += 1
                    continue
            if a.action == "plan":
                if phase in self._planned:               # already planned this step — ignore the repeat
                    stall += 1
                    continue
                goals = self._norm_goals(a.goals)
                self._planned.add(phase)
                self._goals[phase] = [{**g, "met": False} for g in goals]
                self.board.add_note(agent.aid, "GOALS: " + "; ".join(g["goal"] for g in goals))
                self._emit_manager(agent, a, steps, {"goals": goals})
                self._emit_checklist()
                stall = 0
                continue
            if a.action == "ask":
                if not self.cfg.interactive_questions:
                    self.board.add_note(agent.aid, "user questions are unavailable in this interface")
                    stall += 1
                    continue
                answer = self.o.ask_user(a.question)
                if answer is None:
                    break
                self._user_context.append({"question": a.question, "answer": answer})
                self._emit_manager(agent, a, steps, {"answer": answer})
                stall = 0
                continue
            if a.action == "dispatch":
                covered = self._covered_toolkits()
                tasks = [self._allocate_toolkit(t) for t in a.tasks] if getattr(self.cfg, "auto_tools", False) else a.tasks
                fresh = [t for t in tasks
                         if self._sig(t) not in self._dispatched and not self._already_covered(t, covered)]
                if not fresh:                        # all duplicates / already answered — not progress
                    stall += 1
                    self._emit_manager(agent, a, steps, {"skipped": "duplicate checks"})
                    continue
                for t in fresh:
                    self._dispatched.add(self._sig(t))
                ids = self.board.dispatch(agent.aid, fresh)
                self._scale_workers(len(fresh))      # grow the pool lazily to match this batch
                self._emit_manager(agent, a, steps, {"dispatched": ids, "tasks": fresh})
                self._wait_for_batch()
                # Capture the batch's real data immediately and tick the checklist — so a thrashing
                # manager (distinct checks every turn, never voting) can't strand gathered results.
                before = sum(1 for g in self._goals.get(phase, []) if g["met"])
                self._autocapture(agent, steps)
                self._sync_checklist()
                after = sum(1 for g in self._goals.get(phase, []) if g["met"])
                if self._phase_satisfied():          # evidence answers every goal — close on it
                    self._accept_findings(agent)
                    phase_status = self.board.finish_phase(agent.aid)
                    self._emit_manager(agent, type("A", (), {"action": "vote",
                                       "reason": "auto: goals satisfied by gathered evidence"})(), steps,
                                       {"vote": True, "auto": True, "phase": phase_status,
                                        "phase_advanced": phase_status in ("advanced", "complete")})
                    continue
                # Only a batch that advanced the checklist counts as progress; otherwise let stall
                # climb so the convergence guard runs targeted fallback checks for the unmet goals.
                stall = 0 if after > before else stall + 1
                continue
            elif a.action == "propose":
                if self._contradicts_evidence(a.proposal):   # e.g. "no open ports" when scans found some
                    self.board.add_note(agent.aid, "proposal rejected: contradicts scan evidence on the board")
                    stall += 1
                    continue
                if self._is_dup_claim(a.proposal):  # have we seen this finding before? — not progress
                    stall += 1
                    continue
                self._record_claim(a.proposal)
                pid = self.board.add_proposal(agent.aid, a.proposal)
                result = {"proposal_id": pid, "claim": a.proposal}
                self._sync_checklist()
                stall = 0
            elif a.action == "vote" and a.phase_done_vote:
                done_here = sum(1 for s in self.board.tasks.values()
                                if s.phase == self.board.phase and s.status == "done")
                if done_here == 0:                  # nothing ran this step — can't close it empty
                    self.board.add_note(agent.aid, "vote ignored: no checks have completed this step yet")
                    stall += 1
                    continue
                self._autocapture(agent, steps)     # a vote without proposing must not discard gathered data
                self._sync_checklist()
                if not self._phase_satisfied():
                    self.board.add_note(agent.aid, "vote ignored: successful results do not satisfy every step goal")
                    stall += 1
                    continue
                self._accept_findings(agent)        # rerun attributable structured findings before closing
                phase_status = self.board.finish_phase(agent.aid)
                result = {"vote": True, "phase": phase_status,
                          "phase_advanced": phase_status in ("advanced", "complete")}
            else:
                self.board.add_note(agent.aid, a.reason)
                result = {"note": a.reason}
                stall += 1
            self._emit_manager(agent, a, steps, result)

    @staticmethod
    def _sig(t: dict) -> str:
        import json as _json
        return _json.dumps({k: t.get(k) for k in ("tool", "args", "check")}, sort_keys=True)

    @staticmethod
    def _adaptive_check_key(tool: str, args: dict | None) -> tuple:
        """Deduplicate semantically identical follow-ups, including URL variants on one origin."""
        args = args or {}
        if tool in {"service_scan", "port_scan", "web_discover", "nmap_safe_vuln"}:
            return tool, str(args.get("host", "")).lower()
        if tool == "testssl":
            return tool, str(args.get("host", "")).lower(), str(args.get("port", 443))
        if tool in {"nuclei_scan", "nikto_scan"}:
            from urllib.parse import urlsplit
            parsed = urlsplit(str(args.get("url", "")))
            return tool, parsed.scheme.lower(), parsed.netloc.lower()
        return tool, ManagerWorker._sig({"tool": tool, "args": args})

    def _step_toolkit(self, phase: str) -> list:
        """The toolkit allocated to a step: an explicit per-step tool list if given, else one
        derived from the step objective + overall task (dynamic tool allocation)."""
        explicit = (self.cfg.step_tools or {}).get(phase)
        if explicit:
            return list(explicit)
        from .scaling import suggest_toolkit
        objective = self.cfg.step_tasks.get(phase) or phase
        return suggest_toolkit(objective, self.cfg.task)

    def _allocate_toolkit(self, t: dict) -> dict:
        """Ensure a dispatched NL check carries a toolkit: use the check's own text, falling back
        to the step's allocated toolkit, so a worker always gets the right tools."""
        if t.get("tool") or t.get("toolkit") or t.get("tools"):
            return t
        from .scaling import categories_for_text
        instr = t.get("check") or ""
        kit = categories_for_text(instr, fallback=()) or self._step_toolkit(self.board.phase)
        return {**t, "toolkit": kit}

    def _fallback_goal_tasks(self, agent) -> list[dict]:
        """Build a small structured-tool batch for unmet goals the model has stopped advancing."""
        from .scaling import categories_for_text
        from toolmcp.tools import toolkit

        goals = self._open_goals()
        if not goals:
            return []
        text = " ".join(g["goal"] + " " + g.get("expect", "") for g in goals)
        categories = set(categories_for_text(text, fallback=()))
        categories -= self._covered_toolkits()
        # A version/service goal needs a service_scan specifically; a bare port scan does NOT satisfy it,
        # so don't let "ports being covered" suppress the service scan the unmet goal actually requires.
        version_probe = any(w in text.lower() for w in
                            ("service", "version", "banner", "backend", "framework", "hosting",
                             "platform", "stack", "app", "server", "software", "technology"))
        service_done = any(s.tool == "service_scan" and s.status == "done" and s.result
                           and "[error]" not in s.result for s in self.board.tasks.values())
        if version_probe and not service_done:
            categories.add("ports")
        host = self.cfg.target.host if self.cfg.target else ""
        if not host:
            return []
        allow = (self.cfg.step_tools or {}).get(self.board.phase)
        available = {s.name for s in toolkit(allow, agent.available_tools)}
        tasks: list[dict] = []

        def add(tool: str, args: dict) -> None:
            if tool in available:
                t = {"tool": tool, "args": args}
                if self._sig(t) not in self._dispatched:
                    tasks.append(t)

        if "liveness" in categories:
            add("ping", {"host": host})
        if "ports" in categories:
            intents = text.lower()
            version_probe = any(w in intents for w in
                                ("service", "version", "banner", "backend", "framework", "hosting",
                                 "platform", "stack", "app", "server", "software", "technology"))
            if version_probe and "service_scan" in available:
                add("service_scan", {"host": host})
            else:
                add("port_scan", {"host": host, "service_versions": version_probe})
        if "web" in categories:
            open_ports = set()
            for task in self.board.tasks.values():
                if task.status != "done" or "[error]" in task.result:
                    continue
                import re
                for port, service in re.findall(r"(?m)^\s*(\d+)/tcp\s+open\s+(\S+)", task.result):
                    if service.lower().startswith(("http", "https")) or int(port) in {
                            80, 443, 3000, 5000, 5001, 8000, 8080, 8443, 8888, 9000}:
                        open_ports.add(int(port))
            web_ports = sorted(open_ports) if open_ports else []
            if "web_discover" in available:
                add("web_discover", {"host": host,
                                      **({"ports": ",".join(map(str, web_ports))} if web_ports else {})})
            else:
                for port in web_ports or [80, 443]:
                    url_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
                    url = (f"https://{url_host}" if port == 443 else
                           f"http://{url_host}" if port == 80 else
                           f"https://{url_host}:{port}" if port == 8443 else
                           f"http://{url_host}:{port}")
                    add("http_head", {"url": url})
                    add("http_get", {"url": url})
        if "dns" in categories:
            add("dns_lookup", {"name": host})
        if "vuln" in categories:
            open_ports = set()
            for task in self.board.tasks.values():
                if task.status != "done" or "[error]" in task.result:
                    continue
                import re
                for port, service in re.findall(r"(?m)^\s*(\d+)/tcp\s+open\s+(\S+)", task.result):
                    open_ports.add(int(port))
            add("nmap_safe_vuln", {"host": host,
                                   **({"ports": ",".join(map(str, sorted(open_ports)))} if open_ports else {})})
            for port in sorted(open_ports):
                service = ""
                for task in self.board.tasks.values():
                    import re
                    match = re.search(rf"(?m)^\s*{port}/tcp\s+open\s+(\S+)", task.result or "")
                    if match:
                        service = match.group(1).lower()
                        break
                if port in (443, 8443) or "https" in service or "ssl" in service:
                    add("testssl", {"host": host, "port": port})
                if port in (80, 443, 3000, 5000, 5001, 8000, 8080, 8443, 8888, 9000) or "http" in service:
                    scheme = "https" if port in (443, 8443) or "https" in service else "http"
                    url_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
                    default = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
                    url = f"{scheme}://{url_host}{'' if default else ':' + str(port)}"
                    add("nuclei_scan", {"url": url})
                    add("nikto_scan", {"url": url})
        if "path" in categories:
            add("traceroute", {"host": host})
        return tasks

    def _dispatch_fallback_goals(self, agent, step: int) -> bool:
        tasks = self._fallback_goal_tasks(agent)
        if not tasks:
            return False
        for task in tasks:
            self._dispatched.add(self._sig(task))
        ids = self.board.dispatch(agent.aid, tasks)
        action = type("A", (), {"action": "dispatch", "reason": "fallback: checks for unmet goal categories"})()
        self._emit_manager(agent, action, step,
                           {"dispatched": ids, "tasks": tasks, "fallback": True})
        self._wait_for_batch()
        return True

    def _contradicts_evidence(self, claim: str) -> bool:
        """Reject a 'no open ports / host unreachable' claim when the board's raw scans prove otherwise,
        so a hallucinated negative can't become the step's finding."""
        import re
        low = (claim or "").lower()
        negative = re.search(r"\b(no|none|zero|not any|without)\b.*\b(open|port|service)", low) \
            or "no open" in low or "unreachable" in low or "not reachable" in low
        if not negative:
            return False
        for s in self.board.tasks.values():
            if s.phase != self.board.phase:
                continue
            raw = self._raw_results.get(s.id)
            if raw and raw.rc == 0 and not raw.timed_out and re.search(r"(?m)^\s*\d+/tcp\s+open\b", raw.stdout):
                return True                       # a scan observed open ports — the claim is false
        return False

    @staticmethod
    def _norm_goals(goals) -> list:
        """Coerce goals into a list of {goal, expect} dicts (tolerant of strings/None)."""
        out = []
        for g in (goals or []):
            if isinstance(g, dict) and str(g.get("goal", "")).strip():
                out.append({"goal": str(g["goal"]).strip(), "expect": str(g.get("expect", "")).strip()})
            elif isinstance(g, str) and g.strip():
                out.append({"goal": g.strip(), "expect": ""})
        return out

    def _auto_plan(self, phase: str):
        """Synthesize goals from the step objective when the manager won't plan, so every step
        still gets a checklist. The objective itself is the goal; the expectation is a concrete result."""
        obj = self.cfg.step_tasks.get(phase) or self.cfg.task or f"the '{phase}' step"
        host = self.cfg.target.host if self.cfg.target else ""
        goal = obj if obj[0].isupper() else obj
        goals = [{"goal": goal, "expect": f"a concrete result for {host}" if host else "a concrete result"}]
        return type("A", (), {"action": "plan", "reason": "auto: goals from the step objective",
                              "goals": goals})()

    def _mark_goals_met(self) -> None:
        """Tick goals only when findings cover their key concepts and concrete evidence."""
        phase = self.board.phase
        checklist = self._goals.get(phase)
        if not checklist:
            return
        aliases = {"ports": "port", "services": "service", "applications": "app",
                   "apps": "app", "running": "run"}
        for item in checklist:
            if item["met"]:
                continue
            # Match against the goal's core concepts, not explanatory filler in its
            # expectation ("a concrete result for host X"), which dilutes real evidence.
            aliases_full = {**aliases, "versions": "version", "service": "service"}
            want = {aliases_full.get(t, t) for t in self._claim_tokens(item["goal"])}
            want -= {"determine", "identify", "discover", "find", "list", "what", "which",
                     "host", "target", "the", "from", "each", "their", "those", "running", "run"}
            want -= {t for t in want if t.isdigit()}      # target addresses and port numbers are not concepts
            if not want:
                continue
            # A finding only needs to SHARE a core concept with the goal (loose association); the real
            # gate is _goal_has_concrete_evidence, which verifies the finding's data actually answers it
            # (claimed ports cover every observed open port; service goals need real version strings).
            for proposal in self.board.proposals.values():
                if proposal.phase != phase:
                    continue
                ft = {aliases_full.get(t, t) for t in self._claim_tokens(proposal.claim)}
                if (ft and (want & ft)
                        and self._goal_has_concrete_evidence(item["goal"], phase, proposal.claim)):
                    item["met"] = True
                    break

    def _goal_has_concrete_evidence(self, goal: str, phase: str, claim: str = "") -> bool:
        """Apply stronger completeness checks to inventory goals before ticking them."""
        import re
        text = goal.lower()
        phase_tasks = [s for s in self.board.tasks.values()
                       if s.phase == phase and s.status == "done" and s.result
                       and "[error]" not in s.result]
        if "port" in text:
            scans = [(s, self._raw_results.get(s.id)) for s in phase_tasks
                     if s.tool in {"full_scan", "port_scan", "service_scan", "nmap_safe_vuln"}]
            if not scans:
                return False
            observed = set()
            for _, raw in scans:
                if raw and raw.rc == 0 and not raw.timed_out:
                    observed.update(re.findall(r"(?m)^\s*(\d+)/tcp\s+open\b", raw.stdout))
            successful_scans = [raw for _, raw in scans if raw and raw.rc == 0 and not raw.timed_out]
            if not successful_scans:
                return False
            negative_claim = re.search(r"\b(no|none|zero|did not find|not reported)\b", claim.lower())
            if observed and negative_claim:
                return False
            if observed:
                host = self.cfg.target.host if self.cfg.target else ""
                comparable = claim.lower().replace(host.lower(), " ") if host else claim.lower()
                claimed = set(re.findall(r"\b\d{1,5}\b", comparable))
                if not observed.issubset(claimed):
                    return False
            elif negative_claim:
                # A partial/top-ports scan cannot support a host-wide claim of no open ports.
                if not any(s.tool == "full_scan" for s, _ in scans):
                    return False
            else:
                # With no observed open port, require a successful full-range scan and an
                # explicit negative result before treating the inventory goal as answered.
                full = next((raw for s, raw in scans if s.tool == "full_scan"), None)
                if not full or full.rc != 0 or full.timed_out:
                    return False
                if not re.search(r"\b(no|none|zero|did not find|not reported)\b", claim.lower()):
                    return False
            if (any(word in text for word in ("all", "each", "every", "complete"))
                    and not any(s.tool == "full_scan" for s, _ in scans)):
                return False
        if "version" in text:
            scans = [(s, self._raw_results.get(s.id)) for s in phase_tasks if s.tool == "service_scan"]
            if not scans:
                return False
            versioned_ports = set()
            for _, raw in scans:
                if not raw or raw.rc != 0 or raw.timed_out:
                    continue
                for row in re.findall(r"(?m)^\s*(\d+)/tcp\s+open\s+\S+\s+(.+?)\s*$", raw.stdout):
                    version = row[1].strip()
                    if version and not version.endswith("?") and "unrecognized" not in version.lower():
                        versioned_ports.add(row[0])
            # A goal asking for a version on each open port is met only when every port
            # established by a full scan has a concrete version fingerprint.
            if "each" in text or "every" in text or "all" in text:
                full = next((self._raw_results.get(s.id) for s in phase_tasks if s.tool == "full_scan"), None)
                if full and full.rc == 0 and not full.timed_out:
                    all_ports = set(re.findall(r"(?m)^\s*(\d+)/tcp\s+open\b", full.stdout))
                    if all_ports and not all_ports.issubset(versioned_ports):
                        return False
            if not versioned_ports:
                return False
        return True

    def _phase_satisfied(self) -> bool:
        phase = self.board.phase
        successful = any(s.phase == phase and s.status == "done" and s.result
                         and "[error]" not in s.result for s in self.board.tasks.values())
        checklist = self._goals.get(phase, [])
        return successful and (not checklist or all(g["met"] for g in checklist))

    def _open_goals(self) -> list:
        self._mark_goals_met()
        return [g for g in self._goals.get(self.board.phase, []) if not g["met"]]

    def _emit_checklist(self) -> None:
        cl = self._goals.get(self.board.phase)
        if cl is None:
            return
        self.o._emit({"event": "checklist", "phase": self.board.phase,
                      "goals": [{"goal": g["goal"], "expect": g.get("expect", ""), "met": g["met"],
                                 "status": "Evidence recorded" if g["met"] else "Still needs evidence that meets this criterion"}
                                for g in cl]})

    def _sync_checklist(self) -> None:
        self._mark_goals_met()
        self._emit_checklist()

    def _render_checklist(self) -> str:
        cl = self._goals.get(self.board.phase)
        if not cl:
            return ""
        rows = []
        for g in cl:
            mark = "[x]" if g["met"] else "[ ]"
            exp = f"  (expect: {g['expect']})" if g.get("expect") else ""
            rows.append(f"  {mark} {g['goal']}{exp}")
        return "STEP CHECKLIST (tick every box before voting the step done):\n" + "\n".join(rows)

    def _covered_toolkits(self) -> set:
        """Categories already answered by a completed, non-error check (current phase)."""
        from toolmcp.tools import CATEGORY_OF
        cov: set = set()
        for s in self.board.tasks.values():
            if s.phase != self.board.phase or s.status != "done" or not s.result:
                continue
            if "[error]" in s.result:
                continue
            for k in (s.toolkit or []):
                cov.add(k)
            if s.tool and CATEGORY_OF.get(s.tool):
                cov.add(CATEGORY_OF[s.tool])
        return cov

    @staticmethod
    def _already_covered(t: dict, covered: set) -> bool:
        """A natural-language check whose whole toolkit is already satisfied is redundant."""
        if t.get("tool"):                       # a concrete tool call is explicit intent — let it run
            return False
        tk = t.get("toolkit") or t.get("tools") or []
        return bool(tk) and all(k in covered for k in tk)

    # grammatical glue dropped for dedup; STATE words (up/down/open/closed) are kept — they carry meaning
    _STOP = frozenset("a an and are as at be by for from has have in is it of on or that the to was with "
                      "this these those been being its".split())

    @classmethod
    def _claim_tokens(cls, claim: str) -> frozenset:
        """Significant tokens of a finding, for semantic dedup. Lowercased, punctuation split,
        grammar words removed. 'Host X is up' and 'Host X is up, reachable' share their core tokens."""
        import re
        toks = re.sub(r"[^a-z0-9]+", " ", (claim or "").lower()).split()
        return frozenset(t for t in toks if t not in cls._STOP)

    def _is_dup_claim(self, claim: str) -> bool:
        """Have we seen this finding before? True if its tokens are contained in (or strongly
        overlap) an already-recorded finding — catches rewordings, not just exact restatements."""
        toks = self._claim_tokens(claim)
        if not toks:
            return True
        for seen in self._proposed_tokens:
            if not seen:
                continue
            inter = len(toks & seen)
            if inter == len(toks) or inter == len(seen):        # one contains the other
                return True
            if inter / len(toks | seen) >= 0.7:                 # strong Jaccard overlap
                return True
        return False

    def _record_claim(self, claim: str) -> None:
        self._proposed_tokens.append(self._claim_tokens(claim))

    # per-category keywords: used to tell whether a finding already speaks to a check's category,
    # so autocapture doesn't re-dump a result the manager already summarized in its own words.
    _CAT_KW = {
        "liveness": {"up", "alive", "reachable", "responds", "responding", "ping", "loss"},
        "ports": {"open", "port", "ports", "tcp", "udp", "listening"},
        "web": {"http", "https", "header", "headers", "status", "server", "html", "200", "404"},
        "dns": {"dns", "domain", "record", "resolve", "resolves", "nameserver"},
        "path": {"path", "traceroute", "hop", "route", "ttl"},
    }

    def _category_covered(self, category: str) -> bool:
        """True if an existing finding already addresses this check category."""
        import re
        kw = self._CAT_KW.get(category)
        if not kw:
            return False
        blob = " ".join(self.board.shared_findings()).lower()
        return any(re.search(rf"\b{re.escape(k)}\b", blob) for k in kw)

    def _accept_findings(self, agent) -> None:
        """Re-run structured checks from a different agent and compare actual outputs."""
        from toolmcp.tools import BUILTIN_TOOLS, build_command

        if len(self.o.agents) < 2:
            return
        verified = 0
        for proposal in list(self.board.proposals.values()):
            if (proposal.phase != self.board.phase or proposal.confirmed
                    or proposal.id not in self._proposal_sources or verified >= 3):
                continue
            source = self.board.tasks.get(self._proposal_sources[proposal.id])
            original = self._raw_results.get(source.id) if source else None
            if (not source or not source.tool or source.tool not in BUILTIN_TOOLS
                    or source.tool == "full_scan" or original is None):
                continue
            candidates = [a for a in self.o.agents[1:] if a.aid != source.by]
            if not candidates:
                candidates = [a for a in self.o.agents if a.aid != source.by]
            if not candidates:
                continue
            confirmer = candidates[verified % len(candidates)]
            allow = (self.cfg.step_tools or {}).get(source.phase)
            specs = toolkit(allow, confirmer.available_tools)
            allowed = {s.name for s in specs}
            if source.tool not in allowed:
                continue
            host = self.cfg.target.host if self.cfg.target else ""
            try:
                args = self._target_args(source.tool, source.args, host)
                if "host" in BUILTIN_TOOLS[source.tool].params:
                    args["host"] = host
                if source.tool == "dns_lookup":
                    args["name"] = host
                if "url" in BUILTIN_TOOLS[source.tool].params:
                    from urllib.parse import urlsplit
                    parsed = urlsplit(str(args.get("url", "")))
                    if (parsed.scheme not in ("http", "https") or not parsed.hostname
                            or parsed.hostname.strip("[]").lower() != host.strip("[]").lower()
                            or parsed.username or parsed.password):
                        continue
                cmd = build_command(source.tool, args, allowed_names=allowed)
            except ToolError:
                continue
            tool_spec = BUILTIN_TOOLS[source.tool]
            confirmer.turns += 1
            try:
                repeated = confirmer.ex.run(cmd, timeout=tool_spec.timeout)
                digest = compress(repeated)
                supports, detail = self._compare_observation(
                    source.tool, proposal.claim, source.result, original, digest, repeated)
            except Exception as exc:
                repeated = None
                digest = f"verification error: {type(exc).__name__}: {exc}"
                supports, detail = False, "verification command failed"
            status = self.board.add_confirmation(confirmer.aid, proposal.id, detail, supports)
            self.o._emit({"agent": confirmer.aid, "action": "confirm", "phase": source.phase,
                          "result": {"target_proposal": proposal.id, "supports": supports,
                                     "status": status, "by": confirmer.aid, "tool": source.tool,
                                     "cmd": cmd, "digest": digest, "independent": True}})
            verified += 1

    @staticmethod
    def _compare_observation(tool: str, claim: str, source_digest: str, source_raw,
                             repeated_digest: str, repeated_raw) -> tuple[bool, str]:
        """Confirm only repeatable evidence; unsupported comparisons remain unconfirmed."""
        if (source_raw.rc != 0 or source_raw.timed_out or repeated_raw.rc != 0
                or repeated_raw.timed_out):
            return False, "one of the two executions failed or timed out"
        source_fact = ManagerWorker._first_fact(source_digest)
        repeat_fact = ManagerWorker._first_fact(repeated_digest)
        if not source_fact or not repeat_fact:
            return False, "one of the two executions produced no comparable evidence"
        if tool == "ping":
            import re
            losses = [re.search(r"(\d+)% packet loss", output)
                      for output in (source_raw.stdout, repeated_raw.stdout)]
            if all(loss and int(loss.group(1)) < 100 for loss in losses):
                return True, "independent ping rerun received replies from the target"
            return False, "independent ping rerun did not reproduce reachability"
        if tool == "dns_lookup":
            import re
            def records(text):
                return {line.strip().lower().rstrip(".") for line in text.splitlines()
                        if line.strip() and not line.lstrip().startswith("[")}
            left, right = records(source_raw.stdout), records(repeated_raw.stdout)
            if left and left == right:
                return True, "independent DNS rerun returned the same records"
            return False, "independent DNS rerun returned different or empty records"
        if tool in ("port_scan", "service_scan", "nmap_safe_vuln"):
            import re
            pattern = r"(?m)^\s*\d+/tcp\s+open\s+.+$"
            left = {" ".join(x.split()).lower() for x in re.findall(pattern, source_raw.stdout)}
            right = {" ".join(x.split()).lower() for x in re.findall(pattern, repeated_raw.stdout)}
            claim_ports = {x.lower() for x in re.findall(r"\d+/tcp\s+open\s+[^,;]+", claim)}
            if left & right and (not claim_ports or claim_ports <= (left & right)):
                return True, "independent scan reproduced the claimed open service evidence"
            return False, "independent scan did not reproduce the claimed open service evidence"
        if source_fact.casefold() == repeat_fact.casefold():
            return True, f"independent {tool} rerun reproduced the same evidence"
        return False, f"independent {tool} rerun produced different evidence"

    def _autocapture(self, agent, step) -> None:
        """Turn completed results the manager never synthesized into findings, so a run always
        ends with the data it actually gathered rather than an empty verdict. Dedups against existing
        findings by content (`_is_dup_claim`) — NOT by category, so a richer check (e.g. a service/
        version scan) is still captured even when a plainer check of the same category already ran."""
        for s in self.board.tasks.values():
            if (s.phase != self.board.phase or s.status != "done" or not s.result
                    or "[error]" in s.result or s.id in self._captured):
                continue
            self._captured.add(s.id)
            fact = self._first_fact(s.result)
            label = s.tool or (s.instruction.split(",")[0] if s.instruction else "check")
            claim = f"{label}: {fact}"
            if not fact or self._is_dup_claim(claim):
                continue
            self._record_claim(claim)
            pid = self.board.add_proposal(agent.aid, claim)
            self._proposal_sources[pid] = s.id
            self._emit_manager(agent, type("A", (), {"action": "propose",
                               "reason": "auto-captured from results"})(),
                               step, {"proposal_id": pid, "claim": claim, "auto": True})

    @staticmethod
    def _first_fact(result: str, cap: int = 180) -> str:
        """Condense a result digest to a one-line fact (open ports, ping summary, status, evidence)."""
        import re
        t = re.sub(r"^\[rc=-?\d+\]\s*", "", (result or "").strip())
        ports = re.findall(r"(?m)^\s*\d+/\w+\s+open\s+.+$", t)
        if ports:
            return re.sub(r"\s+", " ", ", ".join(ports))[:cap]
        loss = re.search(r"(\d+)% packet loss", t)              # ping: summarise, don't dump the raw echo
        if loss:
            if int(loss.group(1)) >= 100:
                return f"host did not respond to ping, {loss.group(1)}% packet loss"
            return f"host reachable, {loss.group(1)}% packet loss"
        http = re.search(r"HTTP/\d[\d.]*\s+\d{3}[^\n]*", t)
        if http:
            tech = re.findall(r"(?im)^\s*(server|x-powered-by|via|x-aspnet-version|x-generator|set-cookie):\s*([^\n]+)", t)
            details = "; ".join(f"{k}: {v}" for k, v in tech[:3])
            return (http.group(0).strip() + ("; " + details if details else ""))[:cap]
        title = re.search(r"(?is)<title[^>]*>\s*(.*?)\s*</title>", t)
        if title:
            return ("HTML title: " + re.sub(r"\s+", " ", title.group(1)))[:cap]
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        openln = [ln for ln in lines if re.search(r"\bopen\b", ln)]
        if openln:
            return re.sub(r"\s+", " ", " / ".join(openln))[:cap]
        return re.sub(r"\s+", " ", " ".join(lines))[:cap]

    def _emit_manager(self, agent, a, step, result):
        self.o._emit({"agent": agent.aid, "role": "manager", "action": a.action,
                      "step": step, "phase": self.board.phase, "reason": a.reason, "result": result})

    def _wait_for_batch(self, timeout: float = 20.0) -> None:
        """Let workers drain the dispatched batch before the manager re-plans."""
        t0 = time.time()
        while self.board.open_tasks() > 0 and not self.o._stop.is_set():
            if time.time() - t0 > timeout:
                break
            time.sleep(0.2)

    def _pipeline_open_ports(self) -> list[tuple[int, str]]:
        import re
        found = {}
        for task in self.board.tasks.values():
            if task.status != "done" or "[error]" in task.result:
                continue
            for port, service in re.findall(r"(?m)^\s*(\d+)/tcp\s+open\s+(\S+)", task.result):
                found[int(port)] = service.lower()
        return sorted(found.items())

    def _pipeline_web_origins(self) -> set[str]:
        import re
        from urllib.parse import urlsplit
        host = self.cfg.target.host if self.cfg.target else ""
        origins = set()

        def keep(url):
            parsed = urlsplit(url)
            if (parsed.scheme in ("http", "https") and parsed.hostname
                    and parsed.hostname.strip("[]").lower() == host.strip("[]").lower()):
                origins.add(f"{parsed.scheme}://{parsed.netloc}")

        for port, service in self._pipeline_open_ports():
            if port in {80, 443, 3000, 5000, 5001, 8000, 8080, 8443, 8888, 9000} or "http" in service:
                scheme = "https" if port in (443, 8443) or "https" in service else "http"
                url_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
                default = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
                keep(f"{scheme}://{url_host}{'' if default else ':' + str(port)}")
        for task in self.board.tasks.values():
            if task.status != "done":
                continue
            if task.tool == "web_discover":
                for url in re.findall(r"https?://[^\s|]+", task.result or ""):
                    keep(url.rstrip(",;"))
            elif task.tool in ("http_head", "http_get") and task.args and task.args.get("url"):
                keep(str(task.args["url"]))
        return origins

    def _pipeline_tls_ports(self) -> set[int]:
        ports = {port for port, service in self._pipeline_open_ports()
                 if port in (443, 8443) or any(x in service for x in ("https", "ssl", "tls"))}
        from urllib.parse import urlsplit
        for origin in self._pipeline_web_origins():
            parsed = urlsplit(origin)
            if parsed.scheme == "https":
                ports.add(parsed.port or 443)
        return ports

    def _pipeline_tasks(self, phase: str, agent) -> list[dict]:
        """Choose concrete tools for this predefined stage, using discoveries so far."""
        import re
        host = self.cfg.target.host if self.cfg.target else ""
        allow = (self.cfg.step_tools or {}).get(phase)
        available = {s.name for s in toolkit(allow, agent.available_tools)}
        tasks = []

        def add(tool, args):
            if tool in available:
                tasks.append({"tool": tool, "args": args})

        ports = self._pipeline_open_ports()
        port_list = ",".join(str(p) for p, _ in ports)
        if phase == "ports":
            if "service_scan" in available:
                add("service_scan", {"host": host})
            else:
                add("port_scan", {"host": host, "service_versions": True})
        elif phase == "web":
            web_ports = [p for p, s in ports if p in {80, 443, 3000, 5000, 5001, 8000, 8080, 8443, 8888, 9000}
                         or "http" in s or "https" in s]
            if "web_discover" in available:
                add("web_discover", {"host": host,
                                      **({"ports": ",".join(map(str, web_ports))} if web_ports else {})})
            else:
                for port in web_ports or [80, 443]:
                    scheme = "https" if port in (443, 8443) else "http"
                    url_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
                    default = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
                    url = f"{scheme}://{url_host}{'' if default else ':' + str(port)}"
                    add("http_head", {"url": url})
                    add("http_get", {"url": url})
        elif phase == "dns":
            add("dns_lookup", {"name": host})
        elif phase == "liveness":
            add("ping", {"host": host})
        elif phase == "path":
            add("traceroute", {"host": host})
        return tasks

    def _adaptive_pipeline_tasks(self, agent, phase: str, round_no: int):
        """Ask the model for bounded, evidence-led follow-ups; validate every tool and target."""
        from urllib.parse import urlsplit
        from toolmcp.tools import BUILTIN_TOOLS

        host = self.cfg.target.host if self.cfg.target else ""
        allow = (self.cfg.step_tools or {}).get(phase)
        specs = toolkit(allow, agent.available_tools)
        allowed = {spec.name for spec in specs}
        if not specs:
            return [], "No allowed tools are installed for this stage.", False
        tool_doc = render_tools(specs)
        usr = (
            f"OVERALL TASK: {self.cfg.task}\nCURRENT STAGE: {phase}\n"
            f"STAGE OBJECTIVE: {self.cfg.step_tasks.get(phase, '')}\nTARGET HOST (fixed scope): {host}\n"
            f"ADAPTIVE ROUND: {round_no} of 3\n"
            "COMPLETED CHECKS AND RESULTS:\n" + self.board.tasks_summary()[-6000:] + "\n"
            "SHARED FINDINGS:\n" + ("\n".join(self.board.shared_findings()) or "(none)") + "\n\n"
            "AVAILABLE TOOLS:\n" + tool_doc + "\n\n"
            "Select only follow-up checks supported by the evidence. Use exact structured tool names and arguments. "
            "If no check is likely to add useful evidence, vote done."
        )
        action = None
        for question_round in range(2):
            agent.turns += 1
            tr = agent.client.turn(ADAPTIVE_PIPELINE_SYSTEM, usr,
                                   stop_check=lambda: self.o._stop.is_set())
            agent.retries += tr.retries
            action = tr.action
            if action.action != "ask":
                break
            if not self.cfg.interactive_questions or question_round:
                return [], "The adaptive planner requested clarification more than once.", True
            answer = self.o.ask_user(action.question or "Which checks are in scope?")
            if answer is None:
                return [], "Run stopped while waiting for clarification.", True
            self._user_context.append({"question": action.question, "answer": answer})
            usr += f"\nUSER CLARIFICATION: {action.question}\nANSWER: {answer}\nContinue with one dispatch or vote action."
        if action is None:
            return [], "No adaptive decision was returned.", False
        if action.action == "vote" and action.phase_done_vote:
            return [], action.reason or "No useful targeted check remains.", True
        if action.action != "dispatch":
            return [], action.reason or "The adaptive planner did not produce a usable tool selection.", False

        safe_tasks = []
        observed_web = self._pipeline_web_origins()
        observed_tls = self._pipeline_tls_ports()
        open_ports = {port for port, _ in self._pipeline_open_ports()}
        seen = {self._adaptive_check_key(t.tool, t.args) for t in self.board.tasks.values() if t.tool}
        for candidate in (action.tasks or []):
            name = candidate.get("tool")
            raw_args = candidate.get("args")
            if (not isinstance(name, str) or name not in allowed or name not in BUILTIN_TOOLS
                    or (raw_args is not None and not isinstance(raw_args, dict))):
                continue
            args = self._target_args(name, raw_args, host)
            if "host" in BUILTIN_TOOLS[name].params:
                args["host"] = host
            if name == "nmap_safe_vuln":
                if open_ports:
                    args["ports"] = ",".join(map(str, sorted(open_ports)))
                else:
                    args.pop("ports", None)
            if name == "testssl":
                try:
                    port = int(args.get("port", 443))
                except (TypeError, ValueError):
                    continue
                if port not in observed_tls:
                    continue
            # Keep URL based scanners on the configured target, even if a response redirects elsewhere.
            if "url" in BUILTIN_TOOLS[name].params:
                url = str(args.get("url", ""))
                parsed = urlsplit(url)
                if parsed.scheme not in ("http", "https") or not parsed.hostname:
                    continue
                if parsed.username or parsed.password:
                    continue
                if parsed.hostname.strip("[]").lower() != host.strip("[]").lower():
                    continue
                origin = f"{parsed.scheme}://{parsed.netloc}"
                if origin not in observed_web:
                    continue
            try:
                build_command(name, args, allowed_names=allowed)
            except ToolError:
                continue
            normalized = {"tool": name, "args": args}
            key = self._adaptive_check_key(name, args)
            if key in seen or any(self._adaptive_check_key(t["tool"], t["args"]) == key
                                  for t in safe_tasks):
                continue
            sig = self._sig(normalized)
            if sig in self._dispatched or any(self._sig(t) == sig for t in safe_tasks):
                continue
            safe_tasks.append(normalized)
            if len(safe_tasks) >= 2:
                break
        if not safe_tasks:
            return [], "No new in-scope check remains after filtering repeated or unsupported requests.", True
        return safe_tasks, action.reason or "Selected from current findings.", False

    def _adaptive_fallback(self, agent, phase: str) -> list[dict]:
        """One conservative policy fallback if the model cannot return a valid selection."""
        import re
        allow = (self.cfg.step_tools or {}).get(phase)
        available = {s.name for s in toolkit(allow, agent.available_tools)}
        host = self.cfg.target.host if self.cfg.target else ""
        ports = self._pipeline_open_ports()
        tls = [(p, s) for p, s in ports if p in (443, 8443) or "https" in s or "ssl" in s]
        web = [(p, s) for p, s in ports if p in {80, 443, 3000, 5000, 5001, 8000, 8080, 8443, 8888, 9000}
               or "http" in s]
        from urllib.parse import urlsplit
        for observed in self.board.tasks.values():
            if observed.tool != "web_discover" or observed.status != "done":
                continue
            for raw_url in re.findall(r"https?://[^\s|]+", observed.result or ""):
                parsed = urlsplit(raw_url.rstrip(",;"))
                if (parsed.scheme in ("http", "https") and parsed.hostname
                        and parsed.hostname.strip("[]").lower() == host.strip("[]").lower()):
                    port = parsed.port or (443 if parsed.scheme == "https" else 80)
                    web.append((port, parsed.scheme))
                    if parsed.scheme == "https":
                        tls.append((port, parsed.scheme))
        if tls and "testssl" in available:
            return [{"tool": "testssl", "args": {"host": host, "port": tls[0][0]}}]
        if web and "nuclei_scan" in available:
            port, service = web[0]
            scheme = "https" if port in (443, 8443) or "https" in service else "http"
            url_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
            default = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
            url = f"{scheme}://{url_host}{'' if default else ':' + str(port)}"
            return [{"tool": "nuclei_scan", "args": {"url": url}}]
        if ports and "nmap_safe_vuln" in available:
            return [{"tool": "nmap_safe_vuln", "args": {
                "host": host, "ports": ",".join(str(p) for p, _ in ports)}}]
        return []

    def _run_pipeline(self, agent) -> None:
        """Run fixed discovery stages, then adapt bounded follow-up checks to their evidence."""
        for index, phase in enumerate(self.cfg.phases, 1):
            if self.o._stop.is_set() or self.board.done:
                break
            stage = next((s for s in self.cfg.pipeline_stages if s.get("key") == phase), {})
            tasks = self._pipeline_tasks(phase, agent)
            self.o._emit({"event": "pipeline_stage", "phase": phase, "index": index,
                          "total": len(self.cfg.phases), "title": stage.get("title", phase),
                          "objective": self.cfg.step_tasks.get(phase, ""),
                          "tools": [t["tool"] for t in tasks] or self.cfg.step_tools.get(phase, [])})
            if tasks:
                ids = self.board.dispatch(agent.aid, tasks)
                for task in tasks:
                    self._dispatched.add(self._sig(task))
                self._emit_manager(agent, type("A", (), {"action": "dispatch",
                                   "reason": f"pipeline stage: {stage.get('title', phase)}"})(), index,
                                   {"dispatched": ids, "tasks": tasks, "pipeline": True})
                self._wait_for_batch(timeout=120.0)
                if self.o._stop.is_set():
                    return
                self._autocapture(agent, index)
                self._accept_findings(agent)
            elif phase != "vuln":
                self.o._emit({"event": "pipeline_warning", "phase": phase,
                              "reason": "No compatible installed tools for this stage.",
                              "suggested_tools": self.cfg.step_tools.get(phase, [])})
            if phase in ("web", "vuln") and not self.o._stop.is_set():
                for round_no in range(1, 4):
                    followups, reason, done = self._adaptive_pipeline_tasks(agent, phase, round_no)
                    if not followups and not done and round_no == 1:
                        followups = self._adaptive_fallback(agent, phase)
                        if followups:
                            reason = "Adaptive planner unavailable; selected the narrowest check supported by current evidence."
                    self.o._emit({"event": "pipeline_decision", "phase": phase, "round": round_no,
                                  "reason": reason, "tools": [x["tool"] for x in followups],
                                  "finished": done or not followups})
                    if not followups:
                        if phase == "vuln" and round_no == 1 and not done:
                            self.o._emit({"event": "pipeline_warning", "phase": phase,
                                          "reason": reason or "No targeted vulnerability check was selected.",
                                          "suggested_tools": self.cfg.step_tools.get(phase, [])})
                        break
                    ids = self.board.dispatch(agent.aid, followups)
                    for task in followups:
                        self._dispatched.add(self._sig(task))
                    self._emit_manager(agent, type("A", (), {"action": "dispatch", "reason": reason})(), index,
                                       {"dispatched": ids, "tasks": followups, "pipeline": True, "adaptive": True})
                    self._wait_for_batch(timeout=120.0)
                    if self.o._stop.is_set():
                        return
                    self._autocapture(agent, index)
                    self._accept_findings(agent)
            if self.o._stop.is_set():
                return
            phase_status = self.board.finish_phase(agent.aid)
            advanced = phase_status in ("advanced", "complete")
            self.o._emit({"event": "phase_advanced", "phase": self.board.phase if not self.board.done else "complete",
                          "done": self.board.done, "pipeline": True, "stage_complete": phase,
                          "phase_advanced": advanced})

    # ---- entry ----
    def _scale_workers(self, demand: int) -> None:
        """Elastic pool: lazily build and start workers so the pool matches the dispatched batch
        size, bounded by the worker cap. Idle workers block harmlessly and are torn down at run end."""
        cap = getattr(self.o, "_worker_cap", getattr(self.cfg, "worker_cap", 4))
        target = min(cap, max(self.o.worker_count(), demand))
        while self.o.worker_count() < target and not self.o._stop.is_set():
            w = self.o.spawn_worker()
            if w is None:
                break
            t = threading.Thread(target=self._worker_loop, args=(w,), name=w.aid)
            self._wts.append(t)
            t.start()

    def run(self) -> None:
        manager = self.o.agents[0]
        self._wts = [threading.Thread(target=self._worker_loop, args=(w,), name=w.aid)
                     for w in self.o.agents[1:]]
        for t in self._wts:
            t.start()
        sup = threading.Thread(target=self.o._supervise, args=([threading.current_thread()],), name="gate")
        sup.start()
        try:
            if self.cfg.pipeline:
                self._run_pipeline(manager)
            else:
                self._manager_loop(manager)
        finally:
            self.o._stop.set()           # release workers + supervisor
            self.board.open_gate()
            with self.board._lock:
                self.board._task_cv.notify_all()
            for t in list(self._wts):
                t.join(timeout=5)
            sup.join(timeout=5)
            manager.ex.close()
