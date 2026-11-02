import unittest

from swarm.compress import MAX_LINES, compress
from toolmcp.executor import ExecResult


class TestCompress(unittest.TestCase):
    def test_basic(self):
        r = ExecResult("uname", 0, "Linux\n", "")
        self.assertEqual(compress(r), "[rc=0] Linux")

    def test_timeout(self):
        r = ExecResult("nmap -p-", 124, "", "", timed_out=True)
        self.assertTrue(compress(r).startswith("[timeout]"))

    def test_stderr_fallback(self):
        r = ExecResult("nmap", 127, "", "nmap: command not found\n")
        self.assertIn("command not found", compress(r))

    def test_truncation(self):
        body = "\n".join(f"line{i}" for i in range(50))
        r = ExecResult("x", 0, body, "")
        out = compress(r)
        self.assertIn("more lines omitted", out)
        self.assertLessEqual(out.count("line"), MAX_LINES + 2)

    def test_bytes_stdout_does_not_crash(self):
        # Regression: TimeoutExpired can hand back bytes even with text=True.
        r = ExecResult("nmap -p-", 124, b"Starting Nmap\n22/tcp open\n", b"", timed_out=True)
        out = compress(r)
        self.assertIn("22/tcp open", out)
        self.assertIsInstance(out, str)

    def test_invalid_utf8_bytes(self):
        r = ExecResult("x", 0, b"\xff\xfe binary \x80 junk", "")
        self.assertIsInstance(compress(r), str)  # must not raise


if __name__ == "__main__":
    unittest.main()
