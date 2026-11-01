import unittest

from swarm.protocol import Action, ProtocolError, looks_like_command, parse_action


class TestExtractAndParse(unittest.TestCase):
    def test_plain(self):
        a = parse_action('{"action":"run","cmd":"uname -a","reason":"os"}')
        self.assertEqual(a.action, "run")
        self.assertEqual(a.cmd, "uname -a")

    def test_fenced_and_leaked_reasoning(self):
        a = parse_action("Let me think...\n```json\n{\"action\":\"vote\",\"phase_done_vote\":true,\"reason\":\"x\"}\n``` trailing")
        self.assertEqual(a.action, "vote")
        self.assertTrue(a.phase_done_vote)

    def test_missing_required(self):
        with self.assertRaises(ProtocolError):
            parse_action('{"action":"run","reason":"no cmd"}')

    def test_no_json(self):
        with self.assertRaises(ProtocolError):
            parse_action("not json at all")

    def test_vote_bool_required(self):
        with self.assertRaises(ProtocolError):
            parse_action('{"action":"vote","phase_done_vote":"yes","reason":"x"}')

    def test_unknown_action(self):
        with self.assertRaises(ProtocolError):
            parse_action('{"action":"frobnicate","reason":"x"}')


class TestToolAction(unittest.TestCase):
    def test_tool_action_parses(self):
        a = parse_action('{"action":"tool","tool":"port_scan","args":{"host":"10.0.0.5"},"reason":"scan"}')
        self.assertEqual(a.action, "tool")
        self.assertEqual(a.tool, "port_scan")
        self.assertEqual(a.args, {"host": "10.0.0.5"})

    def test_tool_requires_name(self):
        with self.assertRaises(ProtocolError):
            parse_action('{"action":"tool","args":{"host":"x"},"reason":"x"}')

    def test_args_must_be_object(self):
        with self.assertRaises(ProtocolError):
            parse_action('{"action":"tool","tool":"ping","args":"host","reason":"x"}')

    def test_confirm_with_cmd(self):
        a = parse_action('{"action":"confirm","confirm":"P1","cmd":"ping -c1 x","reason":"x"}')
        self.assertEqual(a.confirm, "P1")

    def test_confirm_with_tool(self):
        a = parse_action('{"action":"confirm","confirm":"P1","tool":"ping","args":{"host":"x"},"reason":"x"}')
        self.assertEqual(a.tool, "ping")

    def test_confirm_needs_cmd_or_tool(self):
        with self.assertRaises(ProtocolError):
            parse_action('{"action":"confirm","confirm":"P1","reason":"x"}')


class TestProposalGuard(unittest.TestCase):
    def test_command_shaped_proposal_rejected(self):
        for bad in ('{"action":"propose","proposal":"nmap -sT -p- 10.0.0.5","reason":"x"}',
                    '{"action":"propose","proposal":"curl -I http://x:8080","reason":"x"}'):
            with self.assertRaises(ProtocolError):
                parse_action(bad)

    def test_claim_accepted(self):
        import json
        for good in ("Host 10.0.0.5 is alive", "TCP 22 (SSH) is open", "Port 80 serves nginx"):
            a = parse_action(json.dumps({"action": "propose", "proposal": good, "reason": "x"}))
            self.assertEqual(a.action, "propose")

    def test_looks_like_command(self):
        self.assertTrue(looks_like_command("nmap -sT 10.0.0.5"))
        self.assertTrue(looks_like_command("curl -I http://x"))
        self.assertFalse(looks_like_command("Host 10.0.0.5 is alive"))
        self.assertFalse(looks_like_command("The server returned HTTP 404"))


if __name__ == "__main__":
    unittest.main()
