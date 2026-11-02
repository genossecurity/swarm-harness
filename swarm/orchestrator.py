"""Spin up N agents as threads, all against the one OpenAI-compatible model endpoint and one
target, drive them until every phase clears (or max_steps), and write a
well-formed JSONL event log. Reports the aggregate malformed-JSON retry rate.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict

from .agent import Agent
from .client import LLMClient
from .config import RunConfig
from toolmcp.executor import make_executor
from .whiteboard import Whiteboard


class Orchestrator:
    def __init__(self, cfg: RunConfig, on_event=None):
        self.cfg = cfg
        self.ids = [f"agent-{i}" for i in range(cfg.agents)]
        self.board = Whiteboard(cfg.phases, self.ids, cfg.consensus, gate_enabled=cfg.gate)
        self._log_lock = threading.Lock()
        self._logf = None
        self.agents: list[Agent] = []
        self.on_event = on_event          # optional live sink (e.g. the web UI); gets every event
        self._stop = threading.Event()    # set by request_stop() to end the run early
        self._approve = threading.Event() # set by approve() to release a human phase gate
        self._question_cv = threading.Condition()
        self._pending_question: str | None = None
        self._user_answer: str | None = None
        self._question_id = 0
        self._target_workers: int | None = None   # warm-start worker count (set by auto-scale)

    def request_stop(self) -> None:
        self._stop.set()
        self.board.open_gate()            # wake any agents parked at a gate so they can exit
        self._approve.set()               # unblock the gate supervisor
        with self._question_cv:
            self._question_cv.notify_all()

    def ask_user(self, question: str) -> str | None:
        """Pause the manager until the user answers or stops the run."""
        with self._question_cv:
            self._question_id += 1
            qid = self._question_id
            self._pending_question = question
            self._user_answer = None
        self._emit({"event": "user_question", "id": qid, "question": question})
        with self._question_cv:
            while self._user_answer is None and not self._stop.is_set():
                self._question_cv.wait(0.3)
            answer = self._user_answer
            self._pending_question = None
            self._user_answer = None
            return answer

    def answer_user(self, answer: str) -> bool:
        with self._question_cv:
            if self._pending_question is None or self._user_answer is not None:
                return False
            self._user_answer = answer
            self._question_cv.notify_all()
            return True

    def approve(self) -> None:
        """Human approval to advance past the current phase gate."""
        self._approve.set()

    def _emit(self, event: dict) -> None:
        event["ts"] = round(time.time(), 3)
        line = json.dumps(event, ensure_ascii=False)
        with self._log_lock:
            self._logf.write(line + "\n")
            self._logf.flush()
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:
                pass                       # a slow/broken UI consumer must never break the run

    # Probing tools we tell agents about if present on the host where commands run.
    _PROBE_TOOLS = ["ping", "curl", "wget", "nc", "ncat", "nmap", "python3", "dig", "host",
                    "testssl.sh", "nuclei", "nikto",
                    "nslookup", "ss", "netstat", "traceroute", "mtr", "arp"]

    def _detect_tools(self, ex) -> list[str]:
        names = " ".join(self._PROBE_TOOLS)
        res = ex.run(f'for t in {names}; do command -v "$t" >/dev/null 2>&1 && echo "$t"; done', timeout=15)
        found = [ln.strip() for ln in (res.stdout or "").splitlines() if ln.strip()]
        return [t for t in self._PROBE_TOOLS if t in found]

    @staticmethod
    def _endpoint_label(base_url: str) -> str:
        return base_url.replace("http://", "").replace("https://", "").split("/")[0]

    def _make_agent(self, aid: str, role: str) -> Agent:
        """Construct one agent (executor + shared client). Used both for the initial build and for
        lazily spawning workers on demand."""
        ex = make_executor(self.cfg.target, aid)
        ex.start()
        a = Agent(aid, self.cfg, self._lead_client, self.board, ex, self._emit)
        a.role = role
        a.model_label = self._model_label
        a.available_tools = getattr(self, "_tools", [])
        a.stop_event = self._stop
        self.agents.append(a)
        return a

    def worker_count(self) -> int:
        return sum(1 for a in self.agents if a.role != "lead")

    def spawn_worker(self, emit: bool = True) -> "Agent | None":
        """Lazily build and register one more worker, up to the cap. Returns the new agent or None
        if the cap is reached. The dispatch board accepts the new id without pre-registration."""
        if self.worker_count() >= self._worker_cap:
            return None
        self._next_idx += 1
        aid = f"agent-{self._next_idx}"
        a = self._make_agent(aid, "worker")
        try:
            if aid not in self.board.agents:
                self.board.agents.append(aid)
        except Exception:
            pass
        if emit:
            self._emit({"event": "agent_spawned", "agent": aid, "role": "worker",
                        "model": self._model_label, "workers": self.worker_count(),
                        "cap": self._worker_cap})
        return a

    def _build(self) -> None:
        self._lead_client = LLMClient(self.cfg.base_url, self.cfg.model,
                                      temperature=self.cfg.temperature, max_retries=self.cfg.max_retries)
        self._model_label = f"{self.cfg.model} @ {self._endpoint_label(self.cfg.base_url)}"
        # How many workers to warm-start with, and the hard cap elastic growth may reach.
        target = self._target_workers if self._target_workers is not None else max(0, self.cfg.agents - 1)
        self._worker_cap = max(getattr(self.cfg, "worker_cap", 4), target, 1)

        self._next_idx = 0
        manager = self._make_agent("agent-0", "lead")
        # Detect probing tools once on the manager's host; every agent shares the list.
        try:
            self._tools = self._detect_tools(manager.ex)
        except Exception:
            self._tools = []
        manager.available_tools = self._tools
        roster = [{"id": manager.aid, "role": "lead", "model": self._model_label}]

        if self._manager_mode():
            initial = max(1, min(target, self._worker_cap))     # lazy warm-start; grows on demand
        else:
            initial = max(0, self.cfg.agents - 1)               # peer mode builds its full set up front
        for _ in range(initial):
            w = self.spawn_worker(emit=False)
            if w is None:
                break
            roster.append({"id": w.aid, "role": "worker", "model": self._model_label})

        self._emit({"event": "roster", "agents": roster})
        self._emit({"event": "tools", "available": self._tools})

    def _manager_mode(self) -> bool:
        mode = getattr(self.cfg, "orchestration", "auto")
        if mode == "peer":
            return False
        if mode == "manager":
            return self.cfg.agents >= 2
        return self.cfg.agents >= 2          # auto: manager/worker whenever there's a pool

    def _run_agent(self, agent: Agent) -> None:
        try:
            while not self._stop.is_set() and agent.step():
                pass
        finally:
            agent.ex.close()

    def _supervise(self, threads) -> None:
        """While agents run, watch for a parked phase gate. When one appears, ask the
        user ('move to <next>?') and wait for approve() (or stop) before advancing."""
        announced = False
        while any(t.is_alive() for t in threads):
            if self._stop.is_set():
                self.board.open_gate()
                return
            pending = self.board.pending
            if pending and not announced:
                frm, to = pending
                final = to == "complete"
                self._emit({"event": "phase_gate", "from": frm, "to": to,
                            "to_task": self.cfg.step_tasks.get(to, "") if not final else "",
                            "question": (f"Phase '{frm}' is complete and confirmed by the swarm. "
                                         + (f"Finish the run?" if final else f"Move on to '{to}'?"))})
                announced = True
            if pending:
                if self._approve.wait(0.2):
                    self._approve.clear()
                    if self._stop.is_set():
                        self.board.open_gate()
                        return
                    self.board.confirm_advance()
                    self._emit({"event": "phase_advanced",
                                "phase": self.board.phase if not self.board.done else "complete",
                                "done": self.board.done})
                    announced = False
            else:
                time.sleep(0.1)

    def _compute_auto_scale(self) -> dict | None:
        """Size the agent pool from the task before anything is built (manager/worker mode).
        Mutates cfg and returns the scaling event to emit (after run_start, so the UI keeps it)."""
        if not getattr(self.cfg, "auto_scale", False) or not self._manager_mode():
            return None
        from .scaling import plan_agents
        _manager_count, workers, total, reason = plan_agents(
            self.cfg.task, self.cfg.step_tasks, self.cfg.phases, worker_cap=self.cfg.worker_cap)
        self.cfg.agents = total
        self._target_workers = workers           # warm-start; the pool grows to worker_cap on demand
        return {"event": "scaling", "leads": 1, "workers": workers, "agents": total,
                "cap": self.cfg.worker_cap, "reason": reason}

    def run(self) -> dict:
        with open(self.cfg.log, "w", encoding="utf-8") as logf:
            self._logf = logf
            scaling = self._compute_auto_scale()
            self._emit({"event": "run_start", "config": {
                "agents": self.cfg.agents, "task": self.cfg.task, "phases": self.cfg.phases,
                "base_url": self.cfg.base_url, "model": self.cfg.model,
                "target_mode": self.cfg.target.mode if self.cfg.target else None}})
            if self.cfg.pipeline:
                self._emit({"event": "pipeline_plan", "stages": self.cfg.pipeline_stages})
            if scaling:
                self._emit(scaling)

            # Fail fast if the brain is unreachable, instead of spinning no-op turns.
            probe = LLMClient(self.cfg.base_url, self.cfg.model)
            ok, detail = probe.preflight()
            if not ok:
                self._emit({"event": "run_error", "error": "brain_unreachable",
                            "base_url": self.cfg.base_url, "detail": detail})
                raise ConnectionError(
                    f"cannot reach the model at {self.cfg.base_url} ({detail}). "
                    f"Start it first: ./serving/serve_llamacpp.sh  (even --external needs the brain up).")
            self._build()
            t0 = time.time()
            if self._manager_mode():
                from .manager import ManagerWorker
                self._emit({"event": "orchestration", "mode": "manager",
                            "manager": self.ids[0], "workers": self.ids[1:]})
                ManagerWorker(self).run()
            else:
                self._emit({"event": "orchestration", "mode": "peer"})
                threads = [threading.Thread(target=self._run_agent, args=(a,), name=a.aid)
                           for a in self.agents]
                for t in threads:
                    t.start()
                self._supervise(threads)
                for t in threads:
                    t.join()
            wall = time.time() - t0

            total_turns = sum(a.turns for a in self.agents)
            total_retries = sum(a.retries for a in self.agents)
            summary = {
                "event": "run_end",
                "wall_s": round(wall, 2),
                "phases_cleared": self.board.phase_idx,
                "phases_total": len(self.cfg.phases),
                "all_phases_done": self.board.done,
                "stopped": self._stop.is_set(),
                "proposals": len(self.board.proposals),
                "confirmed": sum(1 for p in self.board.proposals.values() if p.confirmed),
                "total_turns": total_turns,
                "total_retries": total_retries,
                "retry_rate": round(total_retries / total_turns, 3) if total_turns else 0.0,
                # The task result: findings the swarm agreed on (confirmed by a 2nd agent),
                # plus unconfirmed claims for transparency.
                "findings": [{"id": p.id, "by": p.agent, "claim": p.claim,
                              "confirmed_by": list(p.confirmed_by)}
                             for p in self.board.proposals.values() if p.confirmed],
                "unconfirmed": [{"id": p.id, "claim": p.claim}
                                for p in self.board.proposals.values() if not p.confirmed],
            }
            self._emit(summary)
            return summary
