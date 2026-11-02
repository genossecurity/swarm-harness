"""Goal/checklist satisfaction against REAL scan output.

These tests reproduce the exact shape of the manager/worker runs — prose goals
("Determine which TCP ports are open") matched against real nmap/ping digests
("22/tcp open ssh ...") — so a regression that blocks a fully-answered step is caught.
The earlier suite used trivial output ("22/tcp open ssh 80/tcp open http") and trivial
goals, which passed while real runs stalled at "0/2 met".
"""
import unittest

from toolmcp.executor import ExecResult
from swarm.manager import ManagerWorker
from swarm.whiteboard import Whiteboard

# --- fixtures: the actual tool output from the user's runs ---
PING = ("PING 192.168.2.14 (192.168.2.14) 56(84) bytes of data.\n"
        "64 bytes from 192.168.2.14: icmp_seq=1 ttl=64 time=2.98 ms\n"
        "--- 192.168.2.14 ping statistics ---\n"
        "4 packets transmitted, 4 received, 0% packet loss, time 3005ms")

FULL_SCAN = ("Starting Nmap 7.993\nNmap scan report for pi5 (192.168.2.14)\n"
             "Host is up (0.0032s latency).\nPORT     STATE SERVICE\n"
             "22/tcp   open  ssh\n53/tcp   open  domain\n80/tcp   open  http\n"
             "111/tcp  open  rpcbind\n443/tcp  open  https\n5900/tcp open  vnc\nNmap done")

SERVICE_SCAN = ("Starting Nmap 7.993\nNmap scan report for pi5.local (192.168.2.14)\n"
                "PORT     STATE SERVICE    VERSION\n"
                "22/tcp   open  ssh        OpenSSH 10.0p2 Debian 7+deb13u4 (protocol 2.0)\n"
                "53/tcp   open  domain     dnsmasq 2.93+30 (pi-hole)\n"
                "80/tcp   open  http?\n111/tcp  open  rpcbind\n"
                "443/tcp  open  ssl/https?\n5900/tcp open  vnc        VNC (protocol 3.8)\nNmap done")

PORTS_GOAL = {"goal": "Determine which TCP ports are open on 192.168.2.14",
              "expect": "A list of open TCP port numbers", "met": False}
VERSION_GOAL = {"goal": "Identify services and versions running on those open ports",
                "expect": "Service names and version strings for each open port", "met": False}


class GoalFixture(unittest.TestCase):
    def setUp(self):
        self.wb = Whiteboard(["task"], ["agent-0", "agent-1"], consensus=0.6)
        cfg = type("C", (), {"target": type("T", (), {"host": "192.168.2.14"})(),
                             "step_tasks": {}, "task": "What ports are open and what runs on them"})()
        self.orch = type("O", (), {"board": self.wb, "cfg": cfg, "_emit": lambda self, e: None,
                                   "agents": [type("M", (), {"aid": "agent-0"})(),
                                              type("W", (), {"aid": "agent-1"})()]})()
        self.mw = ManagerWorker(self.orch)

    def complete(self, tool, stdout, rc=0, timed_out=False):
        """Simulate a worker running `tool` and completing the dispatch board task."""
        wb = self.wb
        wb.dispatch("agent-0", [{"tool": tool, "args": {"host": "192.168.2.14"}}])
        t = wb.claim()
        self.mw._raw_results[t.id] = ExecResult(f"nmap ({tool})", rc, stdout, "", timed_out)
        wb.complete(t.id, "agent-1", f"[rc={rc}] {stdout}")
        return t.id

    def goals(self, *items):
        self.mw._goals["task"] = [dict(g) for g in items]

    def met(self):
        self.mw._mark_goals_met()
        return {g["goal"][:20]: g["met"] for g in self.mw._goals["task"]}


class TestPortsGoal(GoalFixture):
    def test_ports_goal_met_even_though_finding_lacks_the_word_port(self):
        # THE REGRESSION: "22/tcp open ssh" answers "which ports are open" but never says "port".
        self.goals(PORTS_GOAL)
        self.complete("full_scan", FULL_SCAN)
        self.mw._autocapture(self.orch.agents[0], 1)
        self.assertTrue(list(self.met().values())[0], "a port scan must satisfy a 'which ports are open' goal")
        self.assertTrue(self.mw._phase_satisfied())

    def test_ports_goal_not_met_with_only_a_ping(self):
        self.goals(PORTS_GOAL)
        self.complete("ping", PING)
        self.mw._autocapture(self.orch.agents[0], 1)
        self.assertFalse(list(self.met().values())[0], "liveness alone must not satisfy a ports goal")


class TestVersionGoal(GoalFixture):
    def test_version_goal_needs_real_versions_not_a_bare_port_scan(self):
        self.goals(VERSION_GOAL)
        self.complete("full_scan", FULL_SCAN)           # ports but no versions
        self.mw._autocapture(self.orch.agents[0], 1)
        self.assertFalse(list(self.met().values())[0], "a bare port scan has no version evidence")

    def test_version_goal_met_once_service_scan_runs(self):
        self.goals(VERSION_GOAL)
        self.complete("service_scan", SERVICE_SCAN)
        self.mw._autocapture(self.orch.agents[0], 1)
        self.assertTrue(list(self.met().values())[0], "a service/version scan must satisfy a versions goal")


class TestServiceCaptureRegression(GoalFixture):
    def test_service_scan_is_captured_alongside_a_plain_port_scan(self):
        # Previously autocapture skipped service_scan because the 'ports' category was already
        # 'covered' by full_scan — silently dropping the version data the whole step needed.
        self.goals(PORTS_GOAL, VERSION_GOAL)
        self.complete("full_scan", FULL_SCAN)
        self.complete("service_scan", SERVICE_SCAN)
        self.mw._autocapture(self.orch.agents[0], 1)
        claims = " ".join(p.claim for p in self.wb.proposals.values())
        self.assertIn("OpenSSH", claims, "service/version evidence must be captured, not suppressed")
        self.assertIn("dnsmasq", claims)


class TestFullStepCompletes(GoalFixture):
    def test_real_recon_step_completes_both_goals(self):
        # End-to-end shape of the user's run: ping + full_scan + service_scan → both goals met.
        self.goals(PORTS_GOAL, VERSION_GOAL)
        self.complete("ping", PING)
        self.complete("full_scan", FULL_SCAN)
        self.complete("service_scan", SERVICE_SCAN)
        self.mw._autocapture(self.orch.agents[0], 1)
        self.assertEqual(set(self.met().values()), {True}, "every goal should be met once its evidence is in")
        self.assertTrue(self.mw._phase_satisfied(), "the step must complete, not block, when goals are answered")

    def test_step_correctly_blocks_when_versions_are_genuinely_missing(self):
        # The guard is not toothless: a versions goal stays unmet with no service scan.
        self.goals(PORTS_GOAL, VERSION_GOAL)
        self.complete("ping", PING)
        self.complete("full_scan", FULL_SCAN)
        self.mw._autocapture(self.orch.agents[0], 1)
        self.assertFalse(self.mw._phase_satisfied(), "missing version evidence should keep the step open")


class _FakeEx:
    """An executor that returns canned nmap/ping output based on the command's flags."""
    def __init__(self):
        self.started = True

    def start(self): pass
    def close(self): pass

    def run(self, cmd, timeout=30.0):
        c = cmd.lower()
        if c.startswith("ping"):
            out = PING
        elif "-sv" in c:
            out = SERVICE_SCAN
        elif "-p-" in c:
            out = FULL_SCAN
        elif "nmap" in c:
            out = FULL_SCAN
        else:
            out = "ok"
        return ExecResult(cmd, 0, out, "", False)


def _act(action, **kw):
    base = {"action": action, "reason": "", "goals": None, "tasks": None,
            "proposal": None, "phase_done_vote": None, "question": None}
    base.update(kw)
    return type("A", (), base)()


def _mk_loop(manager_script, worker_default, max_steps=20):
    """Build a manager/worker harness and run the real _manager_loop. `worker_default` is the action
    the manager emits once its script is exhausted (to simulate thrashing)."""
    import threading

    class TR:
        def __init__(s, a): s.action = a; s.retries = 0; s.ok = True

    class Client:
        def __init__(s, script): s.script = list(script); s.n = 0
        def turn(s, sysp, usr, stop_check=None):
            if s.script:
                return TR(s.script.pop(0))
            s.n += 1
            return TR(worker_default(s.n))

    def mk(aid):
        return type("Ag", (), {"aid": aid, "turns": 0, "retries": 0,
                               "available_tools": ["ping", "nmap"], "ex": _FakeEx(),
                               "client": Client([])})()

    wb = Whiteboard(["task"], ["agent-0", "agent-1", "agent-2"], consensus=0.6)
    cfg = type("C", (), {"max_steps": max_steps, "phases": ["task"],
                         "step_tasks": {"task": "What ports are open and what runs on them"},
                         "step_tools": {}, "task": "What ports are open and what runs on them",
                         "target": type("T", (), {"host": "192.168.2.14"})(), "ctx_cap_chars": 6000,
                         "pipeline": False, "pipeline_stages": [], "worker_cap": 4,
                         "auto_tools": True, "interactive_questions": False})()
    stop = threading.Event()
    agents = [mk("agent-0"), mk("agent-1"), mk("agent-2")]
    agents[0].role = "lead"; agents[1].role = "worker"; agents[2].role = "worker"
    orch = type("O", (), {"board": wb, "cfg": cfg, "agents": agents, "_stop": stop, "_worker_cap": 4,
                          "_emit": lambda self, e: None,
                          "worker_count": lambda self: sum(1 for a in self.agents if a.role != "lead"),
                          "spawn_worker": lambda self, emit=True: None})()
    mw = ManagerWorker(orch)
    agents[0].client = Client(manager_script)
    wts = [threading.Thread(target=mw._worker_loop, args=(w,), daemon=True) for w in agents[1:]]
    for t in wts:
        t.start()
    mw._manager_loop(agents[0])
    stop.set()
    for t in wts:
        t.join(timeout=2)
    return mw, wb


class TestEndToEndManagerLoop(unittest.TestCase):
    """Drive the REAL manager loop with real tool execution; the step must complete, not stall."""

    def test_recon_run_completes_and_captures_versions(self):
        # The cooperative case: manager plans then dispatches the right checks.
        mw, wb = _mk_loop(
            [_act("plan", goals=[dict(PORTS_GOAL), dict(VERSION_GOAL)]),
             _act("dispatch", tasks=[{"tool": "ping", "args": {"host": "192.168.2.14"}},
                                     {"tool": "full_scan", "args": {"host": "192.168.2.14"}},
                                     {"tool": "service_scan", "args": {"host": "192.168.2.14"}}])],
            worker_default=lambda n: _act("vote", phase_done_vote=True))
        self.assertTrue(wb.done, "a run that gathered every answer must complete, not block at 0/2")
        self.assertTrue(all(g["met"] for g in mw._goals["task"]))
        claims = " ".join(p.claim for p in wb.proposals.values())
        self.assertIn("OpenSSH", claims)
        self.assertTrue(any(p.confirmed for p in wb.proposals.values()))

    def test_thrashing_manager_still_captures_ports_and_rejects_false_negative(self):
        # THE FAILING RUN: the manager proposes a FALSE "no open ports", then thrashes DISTINCT port
        # scans forever (so dedup never fires and it never votes). The harness must still capture the
        # real ports, reject the false claim, and auto-run a service scan to finish the version goal.
        mw, wb = _mk_loop(
            [_act("plan", goals=[dict(PORTS_GOAL), dict(VERSION_GOAL)]),
             _act("dispatch", tasks=[{"tool": "port_scan", "args": {"host": "192.168.2.14"}}]),
             _act("propose", proposal="The host 192.168.2.14 has no open TCP ports identified yet.")],
            # thrash: a DISTINCT port scan every turn → never deduped, never converges on its own
            worker_default=lambda n: _act("dispatch",
                tasks=[{"tool": "port_scan", "args": {"host": "192.168.2.14", "ports": f"1-{1000+n}"}}]))
        claims = " ".join(p.claim for p in wb.proposals.values())
        self.assertNotIn("no open", claims.lower(), "a hallucinated negative must not become the finding")
        self.assertTrue(any("22/tcp open" in p.claim for p in wb.proposals.values()),
                        "the real open ports must be captured despite the thrash")
        self.assertTrue(mw._goals["task"][0]["met"], "the ports goal is answered by the real scan")
        # the version goal is driven to completion by the fallback service scan
        self.assertTrue(wb.done, "the step must complete, not run to max_steps at 0/2")
        self.assertIn("OpenSSH", claims)

    def test_false_negative_rejected_even_without_thrash(self):
        mw, wb = _mk_loop(
            [_act("plan", goals=[dict(PORTS_GOAL)]),
             _act("dispatch", tasks=[{"tool": "full_scan", "args": {"host": "192.168.2.14"}}]),
             _act("propose", proposal="No open ports were found on the host.")],
            worker_default=lambda n: _act("vote", phase_done_vote=True))
        claims = " ".join(p.claim for p in wb.proposals.values())
        self.assertNotIn("no open", claims.lower())
        self.assertTrue(any("22/tcp open" in p.claim for p in wb.proposals.values()))


if __name__ == "__main__":
    unittest.main()
