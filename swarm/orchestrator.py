"""Spin up N agents as threads, all against the one llama.cpp endpoint and one
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
from .executor import make_executor
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

    def request_stop(self) -> None:
        self._stop.set()
        self.board.open_gate()            # wake any agents parked at a gate so they can exit
        self._approve.set()               # unblock the gate supervisor

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
    _PROBE_TOOLS = ["ping", "curl", "wget", "nc", "ncat", "nmap", "dig", "host",
                    "nslookup", "ss", "netstat", "traceroute", "mtr", "arp"]

    def _detect_tools(self, ex) -> list[str]:
        names = " ".join(self._PROBE_TOOLS)
        res = ex.run(f'for t in {names}; do command -v "$t" >/dev/null 2>&1 && echo "$t"; done', timeout=15)
        found = [ln.strip() for ln in (res.stdout or "").splitlines() if ln.strip()]
        return [t for t in self._PROBE_TOOLS if t in found]

    @staticmethod
    def _endpoint_label(base_url: str) -> str:
        return base_url.replace("http://", "").replace("https://", "").split("/")[0]

    def _build(self) -> None:
        lead = LLMClient(self.cfg.base_url, self.cfg.model,
                         temperature=self.cfg.temperature, max_retries=self.cfg.max_retries)
        lead_label = f"{self.cfg.model} @ {self._endpoint_label(self.cfg.base_url)}"

        # Optional 2nd endpoint for worker sub-agents (another GPU/machine). If it's
        # unreachable, fall back to the lead so the run still works.
        worker, worker_label = lead, lead_label
        if self.cfg.worker_base_url and self.cfg.worker_model:
            wc = LLMClient(self.cfg.worker_base_url, self.cfg.worker_model,
                           temperature=self.cfg.temperature, max_retries=self.cfg.max_retries)
            ok, detail = wc.preflight()
            if ok:
                worker, worker_label = wc, f"{self.cfg.worker_model} @ {self._endpoint_label(self.cfg.worker_base_url)}"
            else:
                self._emit({"event": "worker_unreachable", "base_url": self.cfg.worker_base_url,
                            "detail": detail, "fallback": "using the lead model for all agents"})

        lead_n = max(1, min(self.cfg.lead_agents, self.cfg.agents))
        roster = []
        for i, aid in enumerate(self.ids):
            is_lead = i < lead_n or worker is lead
            client = lead if is_lead else worker
            ex = make_executor(self.cfg.target, aid)
            ex.start()
            a = Agent(aid, self.cfg, client, self.board, ex, self._emit)
            a.role = "lead" if is_lead else "worker"
            a.model_label = lead_label if is_lead else worker_label
            self.agents.append(a)
            roster.append({"id": aid, "role": a.role, "model": a.model_label})

        # Detect probing tools once (same host for all agents) and share the list.
        tools = []
        if self.agents:
            try:
                tools = self._detect_tools(self.agents[0].ex)
            except Exception:
                tools = []
        for a in self.agents:
            a.available_tools = tools
            a.stop_event = self._stop
        self._emit({"event": "roster", "agents": roster})
        self._emit({"event": "tools", "available": tools})

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

    def run(self) -> dict:
        with open(self.cfg.log, "w", encoding="utf-8") as logf:
            self._logf = logf
            self._emit({"event": "run_start", "config": {
                "agents": self.cfg.agents, "task": self.cfg.task, "phases": self.cfg.phases,
                "base_url": self.cfg.base_url, "model": self.cfg.model,
                "target_mode": self.cfg.target.mode if self.cfg.target else None}})

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
            threads = [threading.Thread(target=self._run_agent, args=(a,), name=a.aid)
                       for a in self.agents]
            t0 = time.time()
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
