#!/usr/bin/env python3
"""Concurrency sweep for an OpenAI-compatible server (vLLM or llama-server).

For each K, run K closed-loop workers (each sends --rounds requests back to back, so
about K requests are in flight at all times). Report aggregate generated tokens/sec,
per-request latency, TTFT and per-stream decode speed. If --metrics-url is given, also
record peak running/waiting/KV usage and new preemptions while each K runs.

Stdlib only:
    python sweep.py --ks 1,2,4,8,12,16,20 --rounds 3 --max-tokens 256 --fixed-len
"""
import argparse
import csv
import json
import re
import statistics
import threading
import time
import urllib.request
import uuid

FILLER = ("The agent reviews the task, checks its notes, and decides on the next tool call. ") * 400


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def one_request(args, prompt):
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
        "reasoning_effort": args.reasoning,
        "chat_template_kwargs": {"reasoning_effort": args.reasoning},
    }
    if args.fixed_len:
        body["ignore_eos"] = True  # every request generates exactly max_tokens, so runs are comparable
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft, chunks, usage_tokens = None, 0, None
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage_tokens = ev["usage"].get("completion_tokens")
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    chunks += 1
                    if ttft is None:
                        ttft = time.perf_counter() - t0
    latency = time.perf_counter() - t0
    return {"latency": latency, "ttft": ttft or latency, "tokens": usage_tokens or chunks}


class MetricsSampler(threading.Thread):
    KEYS = {
        "running": r"^(?:vllm:num_requests_running|llamacpp:requests_processing)(?:\{[^}]*\})?\s+([\d.e+-]+)",
        "waiting": r"^(?:vllm:num_requests_waiting|llamacpp:requests_deferred)(?:\{[^}]*\})?\s+([\d.e+-]+)",
        "kv": r"^(?:vllm:kv_cache_usage_perc|vllm:gpu_cache_usage_perc|llamacpp:kv_cache_usage_ratio)(?:\{[^}]*\})?\s+([\d.e+-]+)",
        "preempt": r"^vllm:num_preemptions_total(?:\{[^}]*\})?\s+([\d.e+-]+)",
    }

    def __init__(self, url, period=0.5):
        super().__init__(daemon=True)
        self.url, self.period = url, period
        self.stop_evt = threading.Event()
        self.peak = {k: 0.0 for k in self.KEYS}
        self.first_preempt = None

    def scrape(self):
        text = urllib.request.urlopen(self.url, timeout=5).read().decode()
        out = {}
        for k, pat in self.KEYS.items():
            m = re.search(pat, text, re.M)
            if m:
                out[k] = float(m.group(1))
        return out

    def run(self):
        while not self.stop_evt.is_set():
            try:
                vals = self.scrape()
                if self.first_preempt is None:
                    self.first_preempt = vals.get("preempt", 0.0)
                for k, v in vals.items():
                    self.peak[k] = max(self.peak[k], v)
            except Exception:
                pass
            self.stop_evt.wait(self.period)

    def stop(self):
        self.stop_evt.set()
        self.join()
        self.peak["preempt"] -= self.first_preempt or 0.0
        return self.peak


def run_level(args, k):
    results, errors, lock = [], [], threading.Lock()

    def worker():
        for _ in range(args.rounds):
            # unique prefix defeats prefix caching unless --shared-prefix is set
            head = "" if args.shared_prefix else f"[{uuid.uuid4()}] "
            prompt = head + FILLER[: args.prompt_chars] + "\nSummarize the above in detail."
            try:
                r = one_request(args, prompt)
                with lock:
                    results.append(r)
            except Exception as e:
                with lock:
                    errors.append(repr(e))

    sampler = MetricsSampler(args.metrics_url) if args.metrics_url else None
    if sampler:
        sampler.start()
    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(k)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    peak = sampler.stop() if sampler else {}

    lat = [r["latency"] for r in results]
    ttft = [r["ttft"] for r in results]
    per_stream = [r["tokens"] / max(r["latency"] - r["ttft"], 1e-6) for r in results if r["tokens"] > 1]
    total_tok = sum(r["tokens"] for r in results)
    return {
        "K": k,
        "ok": len(results),
        "err": len(errors),
        "agg_tok_s": total_tok / wall,
        "lat_p50": pct(lat, 50),
        "lat_p95": pct(lat, 95),
        "ttft_p50": pct(ttft, 50),
        "ttft_p95": pct(ttft, 95),
        "stream_tok_s": statistics.median(per_stream) if per_stream else float("nan"),
        "peak_running": peak.get("running", float("nan")),
        "peak_waiting": peak.get("waiting", float("nan")),
        "peak_kv": peak.get("kv", float("nan")),
        "preemptions": peak.get("preempt", float("nan")),
        "first_error": errors[0] if errors else "",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gpt-oss-20b")
    ap.add_argument("--ks", default="1,2,4,8,12,16,20", help="comma-separated concurrency levels")
    ap.add_argument("--rounds", type=int, default=3, help="requests per worker at each K")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompt-chars", type=int, default=2000, help="about 4 chars per token")
    ap.add_argument("--fixed-len", action="store_true", help="ignore_eos so every request generates max_tokens")
    ap.add_argument("--shared-prefix", action="store_true", help="let prefix caching kick in (agent-like)")
    ap.add_argument("--reasoning", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics", help="'' to disable")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--csv", default="", help="also write rows to this CSV file")
    args = ap.parse_args()

    one_request(args, "warmup")  # first request compiles Triton kernels; keep it out of the numbers

    cols = ["K", "ok", "err", "agg_tok_s", "stream_tok_s", "lat_p50", "lat_p95", "ttft_p50", "ttft_p95",
            "peak_running", "peak_waiting", "peak_kv", "preemptions"]
    print("  ".join(f"{c:>12}" for c in cols))
    rows = []
    for k in (int(x) for x in args.ks.split(",")):
        row = run_level(args, k)
        rows.append(row)
        print("  ".join(f"{row[c]:>12.2f}" if isinstance(row[c], float) else f"{row[c]:>12}" for c in cols), flush=True)
        if row["first_error"]:
            print(f"    first error at K={k}: {row['first_error'][:200]}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
