import os
import unittest

from swarm.agent import Agent
from swarm.config import load_config
from swarm.executor import ExecResult
from swarm.protocol import Action
from swarm.tools import ToolError
from swarm.whiteboard import Whiteboard


class TestConfig(unittest.TestCase):
    def test_steps_and_tools(self):
        cfg = load_config(None, agents=2, phases=["a", "b"],
                          step_tasks={"a": "find"}, step_tools={"a": ["ping"]},
                          target_mode="external", target_host="10.0.0.5")
        self.assertEqual(cfg.phases, ["a", "b"])
        self.assertEqual(cfg.step_tasks["a"], "find")
        self.assertEqual(cfg.step_tools["a"], ["ping"])
        self.assertEqual(cfg.target.mode, "external")

    def test_cli_overrides_env(self):
        os.environ["SWARM_AGENTS"] = "7"
        try:
            self.assertEqual(load_config(None, agents=3).agents, 3)      # CLI wins
            self.assertEqual(load_config(None).agents, 7)                # env used
        finally:
            del os.environ["SWARM_AGENTS"]

    def test_external_is_remote_subject(self):
        cfg = load_config(None, target_mode="external", target_host="10.0.0.5")
        self.assertTrue(cfg.target.is_remote_subject)
        cfg2 = load_config(None, target_mode="local", target_host="localhost")
        self.assertFalse(cfg2.target.is_remote_subject)


class FakeExec:
    """Records the command and returns a canned success."""
    def __init__(self):
        self.last = None

    def run(self, cmd, timeout=30.0):
        self.last = cmd
        return ExecResult(cmd, 0, "ok", "")


def _agent(step_tools=None, available=("ping", "nmap", "ncat", "curl")):
    cfg = load_config(None, agents=1, phases=["explore"],
                      step_tools=step_tools or {}, target_mode="external", target_host="10.0.0.5")
    wb = Whiteboard(cfg.phases, ["agent-0"], cfg.consensus)
    ex = FakeExec()
    a = Agent("agent-0", cfg, None, wb, ex, lambda e: None)
    a.available_tools = list(available)
    return a, wb, ex


class TestAgentTools(unittest.TestCase):
    def test_tool_action_builds_and_runs(self):
        a, wb, ex = _agent()
        res = a._act(Action(action="tool", tool="ping", args={"host": "10.0.0.5"}, reason="x"))
        self.assertEqual(ex.last, "ping -c 4 -W 2 10.0.0.5")
        self.assertEqual(res["tool"], "ping")
        self.assertEqual(res["rc"], 0)

    def test_tool_not_allowed_in_step(self):
        a, wb, ex = _agent(step_tools={"explore": ["ping"]})
        res = a._act(Action(action="tool", tool="full_scan", args={"host": "x"}, reason="x"))
        self.assertIn("error", res)
        self.assertIsNone(ex.last)                       # never executed

    def test_tool_unavailable_binary(self):
        a, wb, ex = _agent(available=("ping",))           # nmap not installed
        res = a._act(Action(action="tool", tool="port_scan", args={"host": "x"}, reason="x"))
        self.assertIn("error", res)
        self.assertIsNone(ex.last)

    def test_confirm_via_tool(self):
        a, wb, ex = _agent()
        pid = wb.add_proposal("agent-1", "Host is alive")  # proposed by another agent
        res = a._act(Action(action="confirm", confirm=pid, tool="ping",
                            args={"host": "10.0.0.5"}, reason="verify"))
        self.assertTrue(res["supports"])
        self.assertIn("ping -c 4", ex.last)
        self.assertTrue(wb.proposals[pid].confirmed)

    def test_raw_run_still_works(self):
        a, wb, ex = _agent()
        res = a._act(Action(action="run", cmd="uname -a", reason="x"))
        self.assertEqual(ex.last, "uname -a")
        self.assertEqual(res["cmd"], "uname -a")

    def test_stall_guard_votes_after_repeats(self):
        a, wb, ex = _agent()
        act = Action(action="tool", tool="port_scan", args={"host": "x", "ports": "22"})
        self.assertEqual(a._guard_repeat(act, "explore").action, "tool")   # 1st
        self.assertEqual(a._guard_repeat(act, "explore").action, "tool")   # 2nd
        self.assertEqual(a._guard_repeat(act, "explore").action, "vote")   # 3rd -> vote
        # a different check is not blocked
        other = Action(action="tool", tool="ping", args={"host": "x"})
        self.assertEqual(a._guard_repeat(other, "explore").action, "tool")


class TestRoleRouting(unittest.TestCase):
    def _orch(self, **kw):
        import swarm.orchestrator as orch_mod
        from swarm.client import LLMClient
        from swarm.orchestrator import Orchestrator
        cfg = load_config(None, agents=4, target_mode="local", target_host="localhost",
                          base_url="http://lead/v1", model="big", **kw)
        o = Orchestrator(cfg)
        # don't touch the network or run commands
        self._saved = (LLMClient.preflight, orch_mod.make_executor, Orchestrator._detect_tools)
        LLMClient.preflight = lambda self: (True, "ok")
        orch_mod.make_executor = lambda target, aid: FakeExec2()
        Orchestrator._detect_tools = lambda self, ex: []
        o._logf = open(os.devnull, "w")
        return o

    def tearDown(self):
        from swarm.client import LLMClient
        import swarm.orchestrator as orch_mod
        from swarm.orchestrator import Orchestrator
        if getattr(self, "_saved", None):
            LLMClient.preflight, orch_mod.make_executor, Orchestrator._detect_tools = self._saved

    def test_all_lead_without_worker(self):
        o = self._orch()
        o._build()
        self.assertEqual([a.role for a in o.agents], ["lead"] * 4)

    def test_split_lead_and_workers(self):
        o = self._orch(worker_base_url="http://192.168.2.16:11434/v1", worker_model="small", lead_agents=1)
        o._build()
        self.assertEqual([a.role for a in o.agents], ["lead", "worker", "worker", "worker"])
        self.assertIn("big", o.agents[0].model_label)
        self.assertIn("small", o.agents[1].model_label)
        self.assertIn("192.168.2.16", o.agents[1].model_label)


class TestClientHarmony(unittest.TestCase):
    """reasoning_effort/chat_template_kwargs are gpt-oss-only; sending them to other
    models (qwen2.5-coder on Ollama) errors. The client must gate them on the model."""
    def test_harmony_flag(self):
        from swarm.client import LLMClient
        self.assertTrue(LLMClient("http://x/v1", "gpt-oss:20b").harmony)
        self.assertFalse(LLMClient("http://x/v1", "qwen2.5-coder:3b").harmony)
        self.assertFalse(LLMClient("http://x/v1", "qwen3:8b").harmony)

    def test_body_excludes_reasoning_for_non_gptoss(self):
        import json as _json
        from swarm.client import LLMClient
        c = LLMClient("http://x/v1", "qwen2.5-coder:3b")
        sent = {}
        c._orig = c._post
        def fake_urlopen(req, timeout=None):
            sent["body"] = _json.loads(req.data.decode())
            raise RuntimeError("stop here")  # we only care about the request body
        import swarm.client as cl
        orig = cl.urllib.request.urlopen
        cl.urllib.request.urlopen = fake_urlopen
        try:
            try:
                c._post([{"role": "user", "content": "x"}])
            except RuntimeError:
                pass
        finally:
            cl.urllib.request.urlopen = orig
        self.assertNotIn("reasoning_effort", sent["body"])
        self.assertNotIn("chat_template_kwargs", sent["body"])


class FakeExec2:
    def start(self): pass
    def close(self): pass
    def run(self, cmd, timeout=30.0):
        return ExecResult(cmd, 0, "", "")


if __name__ == "__main__":
    unittest.main()
