"""End-to-end smoke test for the RLHF pipeline — stdlib only.

No network, no torch/trl, no GPU. Covers the parts that carry signal and the
parts that carry risk: the event schema round-trip, the append-only store, every
branch of the outcome-gated pair fusion, the sandbox guardrails, outcome
classification, and the promotion-gate arithmetic. Run:

    python -m unittest discover -s tests
"""
import json
import os
import tempfile
import unittest

from rlhf.hitl import events as E
from rlhf.hitl import store as S
from rlhf.data import build_pairs as BP
from rlhf.data import build_sft as BS
from rlhf.data import synth as SYN
from rlhf import eval as suite
from rlhf import registry
from rlhf.train import preflight
from toolmcp import sandbox as tools   # the MCP enforcement layer the sandbox tests target


def _by_event(evs):
    return {e.prompt: e for e in evs}


class TestEventSchema(unittest.TestCase):
    def test_roundtrip(self):
        ev = E.FeedbackEvent(
            session_id="s", prompt="p",
            candidates=[E.Candidate(cand_id="c0", text="nmap x")],
            action=E.HumanAction.RAN_AS_IS, chosen_id="c0",
            outcome=E.ExecOutcome.SUCCESS, outcome_detail={"open_ports": [22]})
        back = E.FeedbackEvent.from_json(json.loads(json.dumps(ev.to_json())))
        self.assertEqual(back.action, E.HumanAction.RAN_AS_IS)
        self.assertEqual(back.outcome, E.ExecOutcome.SUCCESS)
        self.assertEqual(back.candidates[0].text, "nmap x")
        self.assertTrue(back.is_usable())

    def test_skipped_not_usable(self):
        ev = E.FeedbackEvent(action=E.HumanAction.SKIPPED,
                             candidates=[E.Candidate("c0", "x")])
        self.assertFalse(ev.is_usable())


class TestStore(unittest.TestCase):
    def test_append_iter(self):
        with tempfile.TemporaryDirectory() as d:
            st = S.EventStore(d)
            for ev in SYN.sample_events():
                st.append(ev)
            got = list(st.iter_all())
            self.assertEqual(len(got), len(SYN.sample_events()))
            self.assertTrue(all(isinstance(e, E.FeedbackEvent) for e in got))


class TestPairFusion(unittest.TestCase):
    def setUp(self):
        self.evs = _by_event(SYN.sample_events())

    def test_ran_as_is_success_bumped(self):
        ev = self.evs["Enumerate open TCP ports on 10.13.37.5."]
        pairs = BP.pairs_from_event(ev)
        self.assertTrue(pairs)
        self.assertTrue(all(p.chosen == "nmap -sS -T4 10.13.37.5" for p in pairs))
        self.assertEqual(pairs[0].weight, 1.0)                 # 0.9 * 1.15, capped
        rej = {p.rejected for p in pairs}
        self.assertNotIn("nmap -sS -T4 10.13.37.5", rej)

    def test_edited_is_gold(self):
        ev = self.evs["Grab the HTTP banner on 10.13.37.5:80."]
        pairs = BP.pairs_from_event(ev)
        self.assertTrue(all(p.chosen == "curl -sSI http://10.13.37.5/" for p in pairs))

    def test_failed_choice_salvage_swap(self):
        ev = self.evs["Find the admin login path on the web app."]
        pairs = BP.pairs_from_event(ev)
        self.assertTrue(pairs)
        # chosen swapped to the candidate that actually succeeded (c1 = ffuf)
        self.assertTrue(all("ffuf" in p.chosen for p in pairs))
        # the human's failed original is now a rejected example
        self.assertIn("gobuster dir -u http://10.13.37.5 -w small.txt",
                      {p.rejected for p in pairs})
        self.assertLess(pairs[0].weight, 0.9)                  # demoted

    def test_failed_choice_no_salvage_dropped(self):
        ev = self.evs["Check for anonymous FTP."]
        self.assertEqual(BP.pairs_from_event(ev), [])

    def test_rejected_all_uses_human_text(self):
        ev = self.evs["Pick the next recon step."]
        pairs = BP.pairs_from_event(ev)
        self.assertTrue(pairs)
        self.assertTrue(all("whatweb" in p.chosen for p in pairs))

    def test_skipped_dropped(self):
        self.assertEqual(BP.pairs_from_event(self.evs["(idle)"]), [])


class TestBuildEndToEnd(unittest.TestCase):
    def test_synth_to_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            raw = os.path.join(d, "raw")
            out = os.path.join(d, "splits", "pairs.jsonl")
            SYN.write(raw)
            n = BP.build(raw, out)
            self.assertGreater(n, 0)
            rows = [json.loads(l) for l in open(out)]
            self.assertEqual(len(rows), n)
            for r in rows:
                self.assertEqual(set(r), {"prompt", "chosen", "rejected", "weight", "meta"})
                self.assertNotEqual(r["chosen"], r["rejected"])
            # the un-salvageable failure and the skip never produce a pair
            prompts = {r["prompt"] for r in rows}
            self.assertFalse(any("anonymous FTP" in p for p in prompts))
            self.assertFalse(any("(idle)" in p for p in prompts))


class TestSandboxGuardrails(unittest.TestCase):
    def test_allowlisted_in_lab_ok(self):
        ok, why = tools.validate("nmap -sS 10.13.37.5")
        self.assertTrue(ok, why)

    def test_binary_not_allowlisted_refused(self):
        ok, why = tools.validate("telnet 10.13.37.5")
        self.assertFalse(ok)
        self.assertIn("allowlist", why)

    def test_kill_pattern_refused(self):
        ok, why = tools.validate("curl 10.13.37.5 && rm -rf /")
        self.assertFalse(ok)
        self.assertIn("kill-pattern", why)

    def test_out_of_lab_refused(self):
        ok, why = tools.validate("nmap 8.8.8.8")
        self.assertFalse(ok)
        self.assertIn("outside lab", why)

    def test_no_target_fails_closed(self):
        ok, why = tools.validate("nmap -h")
        self.assertFalse(ok)
        self.assertIn("no target", why)


class TestClassify(unittest.TestCase):
    def test_nmap_open_ports_success(self):
        out = "Nmap scan report\n22/tcp open ssh\n80/tcp open http\n"
        outcome, detail = tools.classify("nmap -sS 10.13.37.5", 0, out, "")
        self.assertEqual(outcome, "success")
        self.assertEqual(detail["open_ports"], [22, 80])

    def test_nmap_no_ports_failure(self):
        outcome, _ = tools.classify("nmap -sS 10.13.37.5", 0, "all closed\n", "")
        self.assertEqual(outcome, "failure")

    def test_tool_error(self):
        outcome, _ = tools.classify("nmap x", 127, "", "nmap: command not found")
        self.assertEqual(outcome, "error")

    def test_timeout_is_error(self):
        outcome, detail = tools.classify("nmap -p- 10.13.37.5", None, "", "")
        self.assertEqual(outcome, "error")
        self.assertTrue(detail["timeout"])

    def test_curl_http_status_success(self):
        outcome, detail = tools.classify("curl -I http://10.13.37.5/", 0,
                                         "HTTP/1.1 200 OK\n", "")
        self.assertEqual(outcome, "success")
        self.assertEqual(detail["http_status"], 200)


class TestPromotionGate(unittest.TestCase):
    def _files(self, d, n_eval, n_safety):
        ev = os.path.join(d, "eval.jsonl")
        sf = os.path.join(d, "safety.jsonl")
        with open(ev, "w") as f:
            for i in range(n_eval):
                f.write(json.dumps({"prompt": f"task {i}"}) + "\n")
        with open(sf, "w") as f:
            for i in range(n_safety):
                f.write(json.dumps({"prompt": f"oos {i}"}) + "\n")
        return ev, sf

    def test_promote_when_both_pass(self):
        with tempfile.TemporaryDirectory() as d:
            ev, sf = self._files(d, 10, 10)
            r = suite.evaluate("cand", "cur", ev, sf,
                               judge=lambda c, u, t: 1 if int(t["prompt"].split()[1]) < 7 else 0,
                               refused=lambda m, s: 1)
            self.assertEqual(r.win_rate, 0.7)
            self.assertEqual(r.safety_pass_rate, 1.0)
            self.assertTrue(r.promote)

    def test_block_on_safety_regression(self):
        with tempfile.TemporaryDirectory() as d:
            ev, sf = self._files(d, 10, 10)
            r = suite.evaluate("cand", "cur", ev, sf,
                               judge=lambda c, u, t: 1,            # wins everything
                               refused=lambda m, s: 0 if s["prompt"].endswith("0") else 1)
            self.assertEqual(r.win_rate, 1.0)
            self.assertLess(r.safety_pass_rate, 0.98)
            self.assertFalse(r.promote)                            # safety floor blocks

    def test_block_on_weak_winrate(self):
        with tempfile.TemporaryDirectory() as d:
            ev, sf = self._files(d, 10, 5)
            r = suite.evaluate("cand", "cur", ev, sf,
                               judge=lambda c, u, t: 1 if int(t["prompt"].split()[1]) < 5 else 0,
                               refused=lambda m, s: 1)
            self.assertEqual(r.win_rate, 0.5)
            self.assertFalse(r.promote)                            # below 0.55


class TestBuildSFT(unittest.TestCase):
    def setUp(self):
        self.evs = _by_event(SYN.sample_events())

    def test_edited_gold_uses_edit(self):
        ex = BS.sft_example(self.evs["Grab the HTTP banner on 10.13.37.5:80."])
        self.assertIsNotNone(ex)
        self.assertEqual(ex["messages"][-1]["role"], "assistant")
        self.assertEqual(ex["messages"][-1]["content"], "curl -sSI http://10.13.37.5/")

    def test_failed_gold_dropped(self):
        # the gobuster pick ran and failed -> not a demonstration
        self.assertIsNone(BS.sft_example(self.evs["Find the admin login path on the web app."]))

    def test_rated_and_skip_dropped(self):
        self.assertIsNone(BS.sft_example(self.evs["Summarize the attack surface so far."]))
        self.assertIsNone(BS.sft_example(self.evs["(idle)"]))

    def test_chose_opt_in(self):
        ev = self.evs["What should we enumerate on port 80 next?"]
        self.assertIsNone(BS.sft_example(ev))                        # off by default
        self.assertIsNotNone(BS.sft_example(ev, include_chose=True))  # opt-in

    def test_build_writes_only_gold(self):
        with tempfile.TemporaryDirectory() as d:
            raw, out = os.path.join(d, "raw"), os.path.join(d, "sft.jsonl")
            SYN.write(raw)
            n = BS.build(raw, out)
            rows = [json.loads(l) for l in open(out)]
            self.assertEqual(len(rows), n)
            self.assertTrue(all(any(m["role"] == "assistant" for m in r["messages"]) for r in rows))
            # the failed FTP pick and the skip never become demonstrations
            dumped = json.dumps(rows)
            self.assertNotIn("ftp://10.13.37.5", dumped)


class TestRegistry(unittest.TestCase):
    def test_record_promote_current(self):
        with tempfile.TemporaryDirectory() as d:
            data = os.path.join(d, "pairs.jsonl")
            with open(data, "w") as f:
                f.write('{"a":1}\n')
            registry.record("policy-dpo-x", "dpo", parent="policy-sft",
                            data_file=data, config={"beta": 0.1}, root=d)
            with open(os.path.join(d, "policy-dpo-x", "manifest.json")) as f:
                man = json.load(f)
            self.assertEqual(len(man["data_sha256"]), 64)             # hashed the data
            self.assertEqual(registry.current(root=d), None)
            registry.attach_eval("policy-dpo-x", {"promote": True, "win_rate": 0.7}, root=d)
            registry.promote("policy-dpo-x", root=d)
            self.assertEqual(registry.current(root=d), "policy-dpo-x")

    def test_promote_refuses_failed_gate(self):
        with tempfile.TemporaryDirectory() as d:
            registry.record("bad", "dpo", root=d)
            registry.attach_eval("bad", {"promote": False, "win_rate": 0.3}, root=d)
            with self.assertRaises(ValueError):
                registry.promote("bad", root=d)


class TestPreflight(unittest.TestCase):
    def _cfg(self, d, rows, key):
        p = os.path.join(d, "data.jsonl")
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return p

    def test_sft_pass(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._cfg(d, [{"messages": [{"role": "user", "content": "hi"},
                                            {"role": "assistant", "content": "nmap x"}]}], "train_file")
            rep = preflight.Report()
            preflight.check_sft({"train_file": p, "max_seq_len": 2048}, rep)
            self.assertTrue(rep.ok(), rep.fatal)

    def test_sft_missing_assistant_is_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._cfg(d, [{"messages": [{"role": "user", "content": "hi"}]}], "train_file")
            rep = preflight.Report()
            preflight.check_sft({"train_file": p}, rep)
            self.assertFalse(rep.ok())

    def test_dpo_degenerate_pair_is_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._cfg(d, [{"prompt": "p", "chosen": "same", "rejected": "same"}], "pairs_file")
            rep = preflight.Report()
            preflight.check_dpo({"pairs_file": p, "sft_model": "./registry/policy-sft"}, rep)
            self.assertFalse(rep.ok())

    def test_dpo_no_sft_model_is_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._cfg(d, [{"prompt": "p", "chosen": "a", "rejected": "b"}], "pairs_file")
            rep = preflight.Report()
            preflight.check_dpo({"pairs_file": p}, rep)   # no sft_model
            self.assertFalse(rep.ok())


if __name__ == "__main__":
    unittest.main(verbosity=2)
