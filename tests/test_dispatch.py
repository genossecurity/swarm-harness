import unittest

from toolmcp.tools import BUILTIN_TOOLS, catalog, render_catalog, toolkit
from swarm.whiteboard import Whiteboard


class TestCatalog(unittest.TestCase):
    def test_categories_cover_tools(self):
        cat = catalog()
        self.assertIn("ports", cat)
        self.assertIn("web", cat)
        flat = [s.name for ts in cat.values() for s in ts]
        for name in BUILTIN_TOOLS:
            self.assertIn(name, flat)

    def test_toolkit_resolves_categories_and_names(self):
        self.assertEqual({s.name for s in toolkit(["ports"])}, {"port_scan", "service_scan", "full_scan", "tcp_connect"})
        self.assertEqual({s.name for s in toolkit(["web", "ping"])}, {"http_head", "http_get", "web_discover", "ping"})

    def test_toolkit_filters_to_available(self):
        self.assertEqual({s.name for s in toolkit(["ports", "liveness"], available_bins=["ping"])}, {"ping"})

    def test_render_catalog(self):
        self.assertIn("ports", render_catalog())
        self.assertIn("port_scan", render_catalog())


class TestDispatchBoard(unittest.TestCase):
    def _wb(self):
        return Whiteboard(["explore"], ["agent-0", "agent-1"], consensus=0.6)

    def test_dispatch_and_claim_once(self):
        wb = self._wb()
        ids = wb.dispatch("agent-0", [{"tool": "ping", "args": {"host": "x"}},
                                      {"check": "find ports", "toolkit": ["ports"]}])
        self.assertEqual(len(ids), 2)
        self.assertEqual(wb.open_tasks(), 2)
        t1 = wb.claim(); t2 = wb.claim()
        self.assertIsNotNone(t1); self.assertIsNotNone(t2)
        self.assertIsNone(wb.claim())          # queue drained — each task handed out once
        self.assertEqual(t1.status, "running")

    def test_complete_updates_board(self):
        wb = self._wb()
        wb.dispatch("agent-0", [{"tool": "ping", "args": {"host": "x"}}])
        t = wb.claim()
        wb.complete(t.id, "agent-1", "[rc=0] alive")
        self.assertEqual(wb.open_tasks(), 0)
        self.assertIn("alive", wb.tasks_summary())
        self.assertIn("agent-1", wb.tasks_summary())

    def test_finish_phase_advances(self):
        wb = self._wb()
        self.assertEqual(wb.finish_phase("agent-0"), "complete")  # single phase -> done
        self.assertTrue(wb.done)

    def test_finish_phase_gated(self):
        wb = Whiteboard(["explore", "verify"], ["a"], consensus=0.6, gate_enabled=True)
        self.assertEqual(wb.finish_phase("a"), "gated")
        self.assertEqual(wb.pending, ("explore", "verify"))
        wb.confirm_advance()
        self.assertEqual(wb.phase, "verify")


class TestManagerDedup(unittest.TestCase):
    """Guards that stop the manager re-dispatching work the board already answered."""

    def _mw(self, wb):
        from swarm.manager import ManagerWorker
        orch = type("O", (), {"board": wb, "cfg": None, "agents": []})()
        return ManagerWorker(orch)

    def test_already_covered_skips_satisfied_nl_check(self):
        from swarm.manager import ManagerWorker
        covered = {"ports"}
        self.assertTrue(ManagerWorker._already_covered({"check": "scan ports", "toolkit": ["ports"]}, covered))
        self.assertFalse(ManagerWorker._already_covered({"check": "grab headers", "toolkit": ["web"]}, covered))
        # a concrete tool call is explicit intent — never auto-skipped
        self.assertFalse(ManagerWorker._already_covered({"tool": "port_scan", "args": {}}, covered))

    def test_covered_toolkits_from_done_tasks(self):
        wb = Whiteboard(["explore"], ["agent-0", "agent-1"], consensus=0.6)
        ids = wb.dispatch("agent-0", [{"check": "scan ports", "toolkit": ["ports"]},
                                      {"check": "headers", "toolkit": ["web"]}])
        t = wb.claim(); wb.complete(t.id, "agent-1", "[rc=0] 22/ssh 80/http open")   # ports done, ok
        mw = self._mw(wb)
        cov = mw._covered_toolkits()
        self.assertIn("ports", cov)      # completed successfully
        self.assertNotIn("web", cov)     # still pending

    def test_error_result_does_not_cover(self):
        wb = Whiteboard(["explore"], ["agent-0", "agent-1"], consensus=0.6)
        wb.dispatch("agent-0", [{"check": "scan ports", "toolkit": ["ports"]}])
        t = wb.claim(); wb.complete(t.id, "agent-1", "[error] nmap not found")
        self.assertNotIn("ports", self._mw(wb)._covered_toolkits())


class TestManagerLoopConvergence(unittest.TestCase):
    """A misbehaving model (re-plans, re-dispatches answered checks) must not spam or hang."""

    def _run(self, script):
        import threading
        from swarm.manager import ManagerWorker

        class A:
            def __init__(s, action, **kw):
                s.action = action; s.reason = kw.get("reason", ""); s.goals = kw.get("goals")
                s.tasks = kw.get("tasks"); s.proposal = kw.get("proposal")
                s.phase_done_vote = kw.get("phase_done_vote")

        class TR:
            def __init__(s, a): s.action = a; s.retries = 0; s.ok = True

        class Client:
            def __init__(s, script): s.script = list(script)
            def turn(s, sysp, usr, stop_check=None):
                return TR(s.script.pop(0)) if s.script else TR(A("vote", phase_done_vote=True))

        class Agent:
            def __init__(s, c):
                s.aid = "agent-0"; s.client = c; s.turns = 0; s.retries = 0
                s.available_tools = ["ping", "nmap", "curl"]

        wb = Whiteboard(["explore"], ["agent-0", "agent-1"], consensus=0.6)
        cfg = type("C", (), {"max_steps": 30, "phases": ["explore"], "step_tasks": {},
                             "task": "ports?", "target": None, "ctx_cap_chars": 6000,
                             "pipeline": False, "pipeline_stages": [], "worker_cap": 4,
                             "auto_tools": False, "interactive_questions": False})()
        emitted, stop = [], threading.Event()
        orch = type("O", (), {"board": wb, "cfg": cfg, "_worker_cap": 4,
                              "agents": [type("M", (), {"aid": "agent-0"})(),
                                         type("W", (), {"aid": "agent-1"})()],
                              "worker_count": lambda self: sum(1 for a in self.agents if getattr(a, "role", "worker") != "lead"),
                              "spawn_worker": lambda self, emit=True: None,   # cap reached in tests; no growth
                              "_stop": stop, "_emit": lambda self, e: emitted.append(e)})()
        mw = ManagerWorker(orch)

        def worker():
            while not stop.is_set():
                t = wb.claim()
                if t is None:
                    wb.wait_for_task(0.05); continue
                wb.complete(t.id, "agent-1", "[rc=0] 22/tcp open ssh 80/tcp open http")
        wt = threading.Thread(target=worker, daemon=True); wt.start()
        mw._manager_loop(Agent(Client(script)))
        stop.set()
        return wb, emitted, A

    @staticmethod
    def _A(action, **kw):
        base = {"action": action, "reason": "", "goals": None, "tasks": None,
                "proposal": None, "phase_done_vote": None}
        base.update(kw)
        return type("A", (), base)()

    def test_vote_without_proposing_still_captures_results(self):
        # The manager dispatches, results arrive, then it votes done WITHOUT proposing anything.
        # The gathered data must still be captured as unconfirmed findings, not lost.
        wb, emitted, _ = self._run([
            self._A("plan", goals=["reachable?", "open ports?"]),
            self._A("dispatch", tasks=[{"check": "ping", "toolkit": ["liveness"]},
                                       {"check": "scan ports", "toolkit": ["ports"]}]),
            self._A("vote", phase_done_vote=True, reason="insufficient data"),
            self._A("vote", phase_done_vote=True, reason="nothing to record"),
        ])
        findings = list(wb.proposals.values())
        self.assertTrue(findings, "a vote without proposing must not discard gathered results")
        self.assertFalse(any(p.confirmed for p in findings))       # NL checks have no safe rerunnable tool
        self.assertTrue(any("open" in p.claim for p in findings))   # the port scan became a finding

    def test_pathological_run_converges_without_spam(self):
        _, emitted, A = None, None, None
        from swarm.manager import ManagerWorker  # noqa: F401
        # 1 plan, a real batch, then THREE redundant re-scans, then (never reached) proposes.
        wb, emitted, A = self._run([
            type("A", (), {"action": "plan", "reason": "", "goals": ["reachable?", "open ports?"],
                           "tasks": None, "proposal": None, "phase_done_vote": None})(),
            type("A", (), {"action": "dispatch", "reason": "", "goals": None, "proposal": None,
                           "phase_done_vote": None,
                           "tasks": [{"check": "ping", "toolkit": ["liveness"]},
                                     {"check": "scan ports", "toolkit": ["ports"]}]})(),
        ] + [type("A", (), {"action": "dispatch", "reason": "", "goals": None, "proposal": None,
                            "phase_done_vote": None,
                            "tasks": [{"check": f"re-scan ports attempt {i}", "toolkit": ["ports"]}]})()
             for i in range(3)])
        plans = [e for e in emitted if e.get("action") == "plan"]
        real = [e for e in emitted if e.get("action") == "dispatch" and e.get("result", {}).get("dispatched")]
        skipped = [e for e in emitted if e.get("action") == "dispatch" and e.get("result", {}).get("skipped")]
        self.assertEqual(len(plans), 1)              # goals stated once, not once per turn
        self.assertEqual(len(real), 1)               # the answered toolkit is not re-dispatched
        self.assertGreaterEqual(len(skipped), 1)     # the re-scans are recognised as redundant
        self.assertTrue(wb.shared_findings())        # gathered data is captured, not lost


class TestManagerFindings(unittest.TestCase):
    """Findings deduplicate and require evidence before independent confirmation."""

    def _mw(self):
        from swarm.manager import ManagerWorker
        wb = Whiteboard(["verify"], ["agent-0", "agent-1"], consensus=0.6)
        orch = type("O", (), {"board": wb, "cfg": None, "_emit": lambda self, e: None,
                              "agents": [type("M", (), {"aid": "agent-0"})(),
                                         type("W", (), {"aid": "agent-1"})()]})()
        return ManagerWorker(orch), wb

    def test_dedup_collapses_reworded_findings(self):
        mw, _ = self._mw()
        mw._record_claim("Host 192.168.2.14 is up")
        # exact, punctuation, and reworded restatements are all recognised as seen-before
        self.assertTrue(mw._is_dup_claim("host 192.168.2.14 is up."))
        self.assertTrue(mw._is_dup_claim("Host 192.168.2.14 is up (reachable via ping)"))
        self.assertTrue(mw._is_dup_claim("Host 192.168.2.14 is up and reachable via ping"))
        # a genuinely different finding is NOT a duplicate
        self.assertFalse(mw._is_dup_claim("Port 22/tcp is open running ssh"))

    def test_accept_findings_does_not_confirm_without_rerunnable_evidence(self):
        mw, wb = self._mw()
        p1 = wb.add_proposal("agent-0", "Host is up")
        mw._accept_findings(mw.o.agents[0])
        self.assertFalse(wb.proposals[p1].confirmed)
        self.assertEqual(wb.proposals[p1].confirmed_by, [])

    def test_autocapture_records_uncaptured_result(self):
        mw, wb = self._mw()
        wb.dispatch("agent-0", [{"check": "scan ports", "toolkit": ["ports"]}])
        t = wb.claim(); wb.complete(t.id, "agent-1", "[rc=0] 22/tcp open ssh 80/tcp open http")
        mw._autocapture(mw.o.agents[0], 1)
        claims = [p.claim for p in wb.proposals.values()]
        self.assertTrue(any("22/tcp open ssh" in c for c in claims))


if __name__ == "__main__":
    unittest.main()
