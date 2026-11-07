import json
import os
import tempfile
import unittest
from unittest import mock

import swarm.orchestrator as orchestrator


class DummyConfig:
    def __init__(self):
        self.agents = 1
        self.phases = ["explore", "verify", "report"]
        self.consensus = 0.6
        self.gate = False
        self.base_url = "http://localhost:11434/v1"
        self.model = "gpt-oss:20b"
        self.temperature = 0.0
        self.max_retries = 3
        self.target = None
        self.task = "dummy task"
        self.step_tasks = {"explore": "", "verify": "", "report": ""}
        self.worker_cap = 4
        self.auto_scale = False
        self.orchestration = "peer"
        self.pipeline = False
        self.pipeline_stages = []
        self.log = ""


class DummyAgent:
    def __init__(self, aid, *_, **__):
        self.aid = aid
        self.turns = 1
        self.retries = 0
        self.step = mock.Mock(return_value=False)
        self.ex = mock.Mock()
        self.ex.close = mock.Mock()


class DummyWhiteboard:
    def __init__(self, *_, **__):
        self.phase_idx = 0
        self.done = False
        self.proposals = {}
        self.agents = []
        self.pending = None
        self.confirm_advance = mock.Mock()
        self.open_gate = mock.Mock()


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        # Patch external dependencies
        patcher_agent = mock.patch.object(orchestrator, "Agent", DummyAgent)
        patcher_llm = mock.patch.object(orchestrator, "LLMClient")
        patcher_executor = mock.patch.object(orchestrator, "make_executor", lambda *_, **__: mock.Mock(start=mock.Mock(), close=mock.Mock()))
        patcher_whiteboard = mock.patch.object(orchestrator, "Whiteboard", DummyWhiteboard)
        self.addCleanup(patcher_agent.stop)
        self.addCleanup(patcher_llm.stop)
        self.addCleanup(patcher_executor.stop)
        self.addCleanup(patcher_whiteboard.stop)
        patcher_agent.start()
        patcher_llm.start()
        patcher_executor.start()
        patcher_whiteboard.start()
        # Configure LLMClient mock
        mock_llm_instance = orchestrator.LLMClient.return_value
        mock_llm_instance.preflight.return_value = (True, None)
        # Temporary log file
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.log_path = os.path.join(self.tmp_dir.name, "run.jsonl")
        self.cfg = DummyConfig()
        self.cfg.log = self.log_path

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_run_emits_events_and_summary(self):
        orch = orchestrator.Orchestrator(self.cfg)
        # Patch internal methods to avoid heavy logic
        orch._build = mock.Mock()
        orch._run_agent = mock.Mock()
        orch._supervise = mock.Mock()
        orch._compute_auto_scale = mock.Mock(return_value=None)
        # Create a dummy agent list
        agent = DummyAgent("agent-0")
        orch.agents = [agent]
        orch.board = DummyWhiteboard()
        # Run orchestrator
        summary = orch.run()
        # Verify summary contains expected keys
        self.assertIn("event", summary)
        self.assertEqual(summary["event"], "run_end")
        self.assertIn("wall_s", summary)
        self.assertIn("phases_cleared", summary)
        self.assertIn("total_turns", summary)
        # Verify log file written
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertGreater(len(lines), 0)
        # First line should be run_start event
        first_event = json.loads(lines[0])
        self.assertEqual(first_event["event"], "run_start")
