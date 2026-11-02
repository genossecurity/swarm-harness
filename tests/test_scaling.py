import unittest

from swarm.scaling import categories_for_text, estimate_fanout, plan_agents, suggest_toolkit


class TestToolAllocation(unittest.TestCase):
    def test_categories_from_text(self):
        self.assertEqual(categories_for_text("what ports are open"), ["ports"])
        self.assertIn("web", categories_for_text("what services and apps are running"))
        self.assertIn("dns", categories_for_text("resolve the domain records"))

    def test_fallback_when_nothing_matches(self):
        self.assertEqual(categories_for_text("completely unrelated prose"), ["liveness", "ports"])
        self.assertEqual(categories_for_text("unrelated", fallback=()), [])

    def test_suggest_toolkit_uses_objective_and_task(self):
        kit = suggest_toolkit("identify services on open ports", "recon of the host")
        self.assertIn("ports", kit)
        # a clearly web-facing objective pulls in the web toolkit
        self.assertIn("web", suggest_toolkit("fetch the http headers of the web page", ""))

    def test_categories_are_catalog_ordered(self):
        # liveness before ports before web, regardless of mention order
        cats = categories_for_text("web server on open ports, is it reachable")
        self.assertEqual(cats, [c for c in ["liveness", "ports", "web", "dns", "path"] if c in cats])


class TestAgentScaling(unittest.TestCase):
    def test_narrow_task_small_pool(self):
        leads, workers, total, _ = plan_agents("is the host alive", phases=["explore"])
        self.assertEqual(leads, 1)
        self.assertGreaterEqual(workers, 2)
        self.assertLessEqual(workers, 4)

    def test_broad_task_scales_up_to_cap(self):
        leads, workers, total, reason = plan_agents(
            "scan ports, enumerate web services, resolve dns, trace the route, check liveness",
            phases=["ports", "web", "dns", "path"], worker_cap=4)
        self.assertEqual(leads, 1)
        self.assertEqual(workers, 4)              # breadth 5 clamped to the cap
        self.assertEqual(total, 5)
        self.assertIn("categories", reason)

    def test_cap_is_respected(self):
        _, workers, _, _ = plan_agents(
            "ports web dns path liveness scan services resolve route", worker_cap=2,
            phases=["a", "b", "c"])
        self.assertLessEqual(workers, 2)

    def test_fanout_floor(self):
        self.assertGreaterEqual(estimate_fanout("x", phases=[]), 2)


class TestElasticPool(unittest.TestCase):
    """Lazy worker construction: workers are built on demand, bounded by the cap."""

    def _orch(self, cap):
        import types
        from swarm.config import RunConfig
        from swarm.orchestrator import Orchestrator
        o = Orchestrator(RunConfig(agents=2, worker_cap=cap, task="scan ports", log="/dev/null"))
        o._lead_client = None
        o._model_label = "m"
        o._tools = []
        o._worker_cap = cap
        o._next_idx = 0
        o.agents = []

        def fake_make(self, aid, role):            # no real executor/client
            a = type("A", (), {"aid": aid, "role": role, "ex": None})()
            self.agents.append(a)
            return a
        o._make_agent = types.MethodType(fake_make, o)
        o._make_agent("agent-0", "lead")           # the manager
        return o

    def test_spawn_is_lazy_and_capped(self):
        o = self._orch(cap=3)
        self.assertEqual(o.worker_count(), 0)      # nothing built up front beyond the manager
        self.assertIsNotNone(o.spawn_worker(emit=False))
        self.assertIsNotNone(o.spawn_worker(emit=False))
        self.assertIsNotNone(o.spawn_worker(emit=False))
        self.assertEqual(o.worker_count(), 3)
        self.assertIsNone(o.spawn_worker(emit=False))   # cap reached — no more construction
        self.assertEqual(o.worker_count(), 3)
        self.assertEqual([a.aid for a in o.agents], ["agent-0", "agent-1", "agent-2", "agent-3"])


if __name__ == "__main__":
    unittest.main()
