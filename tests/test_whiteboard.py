import unittest

from swarm.whiteboard import Whiteboard

A = ["agent-0", "agent-1", "agent-2"]


class TestConsensus(unittest.TestCase):
    def test_advances_on_consensus(self):
        wb = Whiteboard(["explore", "verify"], A, consensus=0.6)  # need round(0.6*3)=2
        self.assertFalse(wb.vote_phase_done("agent-0", True))
        self.assertTrue(wb.vote_phase_done("agent-1", True))      # 2nd vote -> advance
        self.assertEqual(wb.phase, "verify")

    def test_vote_can_be_withdrawn(self):
        wb = Whiteboard(["explore", "verify"], A, consensus=0.6)
        wb.vote_phase_done("agent-0", True)
        wb.vote_phase_done("agent-0", False)                     # withdraw
        self.assertFalse(wb.vote_phase_done("agent-1", True))    # only 1 vote now
        self.assertEqual(wb.phase, "explore")


class TestGate(unittest.TestCase):
    def test_gate_parks_until_confirm(self):
        wb = Whiteboard(["explore", "verify"], A, consensus=0.6, gate_enabled=True)
        wb.vote_phase_done("agent-0", True)
        wb.vote_phase_done("agent-1", True)                      # consensus, but gated
        self.assertEqual(wb.phase, "explore")                   # did NOT advance
        self.assertEqual(wb.pending, ("explore", "verify"))
        self.assertFalse(wb.gate.is_set())                      # agents blocked
        wb.confirm_advance()
        self.assertEqual(wb.phase, "verify")
        self.assertIsNone(wb.pending)
        self.assertTrue(wb.gate.is_set())

    def test_last_phase_gate_completes(self):
        wb = Whiteboard(["explore"], A, consensus=0.6, gate_enabled=True)
        wb.vote_phase_done("agent-0", True)
        wb.vote_phase_done("agent-1", True)
        self.assertEqual(wb.pending, ("explore", "complete"))
        wb.confirm_advance()
        self.assertTrue(wb.done)


class TestGroundedConfirmation(unittest.TestCase):
    def test_different_agent_confirms(self):
        wb = Whiteboard(["explore"], A, consensus=0.6)
        pid = wb.add_proposal("agent-0", "Host is alive")
        wb.add_confirmation("agent-1", pid, "[rc=0]", supports=True)
        self.assertTrue(wb.proposals[pid].confirmed)
        self.assertIn("agent-1", wb.proposals[pid].confirmed_by)

    def test_self_confirmation_ignored(self):
        wb = Whiteboard(["explore"], A, consensus=0.6)
        pid = wb.add_proposal("agent-0", "Host is alive")
        wb.add_confirmation("agent-0", pid, "[rc=0]", supports=True)
        self.assertFalse(wb.proposals[pid].confirmed)

    def test_failed_probe_does_not_confirm(self):
        wb = Whiteboard(["explore"], A, consensus=0.6)
        pid = wb.add_proposal("agent-0", "Host is alive")
        wb.add_confirmation("agent-1", pid, "[rc=1]", supports=False)
        self.assertFalse(wb.proposals[pid].confirmed)

    def test_unknown_proposal(self):
        wb = Whiteboard(["explore"], A, consensus=0.6)
        status = wb.add_confirmation("agent-1", "P99", "x", supports=True)
        self.assertIn("unknown", status)


class TestSharedFindings(unittest.TestCase):
    def test_attribution_across_steps(self):
        wb = Whiteboard(["explore", "verify"], A, consensus=0.6)
        p1 = wb.add_proposal("agent-0", "Ports 22,80 open")
        wb.add_confirmation("agent-1", p1, "[rc=0]", supports=True)
        wb.add_proposal("agent-2", "Port 80 is nginx")
        digest = wb.shared_findings()
        self.assertTrue(any("agent-0" in d and "confirmed by agent-1" in d for d in digest))
        self.assertTrue(any("agent-2" in d and "unconfirmed" in d for d in digest))


if __name__ == "__main__":
    unittest.main()
