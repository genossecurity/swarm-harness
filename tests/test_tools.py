import unittest

from toolmcp.tools import (BUILTIN_TOOLS, ToolError, build_command, render_tools,
                         tools_for_step)


class TestBuild(unittest.TestCase):
    def test_ping(self):
        self.assertEqual(build_command("ping", {"host": "10.0.0.5"}), "ping -c 4 -W 2 10.0.0.5")

    def test_ping_count_clamped(self):
        self.assertIn("-c 10", build_command("ping", {"host": "x", "count": 999}))

    def test_port_scan_default_and_ports(self):
        self.assertIn("--top-ports", build_command("port_scan", {"host": "10.0.0.5"}))
        self.assertIn("-p 22,80", build_command("port_scan", {"host": "10.0.0.5", "ports": "22,80"}))

    def test_port_scan_normalizes_messy_lists(self):
        # Model replies may contain JSON-array-ish port lists; normalize to nmap form.
        for messy in ["[22, 53, 80]", '["22","53","80"]', "22 53 80"]:
            cmd = build_command("port_scan", {"host": "x", "ports": messy})
            self.assertIn("-p 22,53,80", cmd)
            self.assertNotIn("[", cmd)

    def test_tcp_connect_port_is_int(self):
        self.assertEqual(build_command("tcp_connect", {"host": "x", "port": "22"}),
                         "ncat -z -w 3 -v x 22")
        with self.assertRaises(ToolError):
            build_command("tcp_connect", {"host": "x", "port": "99999"})

    def test_missing_args(self):
        with self.assertRaises(ToolError):
            build_command("port_scan", {})          # host required

    def test_unknown_tool(self):
        with self.assertRaises(ToolError):
            build_command("rm_rf", {"host": "x"})


class TestInjectionSafety(unittest.TestCase):
    def test_args_are_shell_quoted(self):
        cmd = build_command("ping", {"host": "x; rm -rf /"})
        # the metacharacters must be neutralized inside a single-quoted string
        self.assertIn("'x; rm -rf /'", cmd)
        self.assertNotIn("; rm -rf / ", cmd.replace("'x; rm -rf /'", ""))

    def test_url_quoted(self):
        cmd = build_command("http_get", {"url": "http://x/$(whoami)"})
        self.assertIn("'http://x/$(whoami)'", cmd)


class TestStepScoping(unittest.TestCase):
    def test_allowlist(self):
        specs = tools_for_step(["ping", "port_scan"])
        self.assertEqual({s.name for s in specs}, {"ping", "port_scan"})

    def test_none_means_all(self):
        self.assertEqual(len(tools_for_step(None)), len(BUILTIN_TOOLS))

    def test_availability_filter(self):
        # only ping's binary present -> nmap-based tools drop out
        specs = tools_for_step(["ping", "port_scan"], available_bins=["ping"])
        self.assertEqual({s.name for s in specs}, {"ping"})

    def test_not_allowed_rejected(self):
        with self.assertRaises(ToolError):
            build_command("full_scan", {"host": "x"}, allowed_names={"ping"})

    def test_render_lists_tools(self):
        doc = render_tools(tools_for_step(["ping"]))
        self.assertIn("ping(", doc)
        self.assertIn('"action":"tool"', doc)


if __name__ == "__main__":
    unittest.main()
