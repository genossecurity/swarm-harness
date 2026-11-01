# Serving: gpt-oss-20B on RX 6950 XT (gfx1030) via llama.cpp

The engine for this project is **llama.cpp's `llama-server`**, serving
`gpt-oss-20B` MXFP4, OpenAI-compatible on `:8000`. gpt-oss MXFP4 runs natively
on this gfx1030 card (weights ~13 GB, no bf16 blowup, ~90 t/s single-stream).

> The from-source vLLM path is **abandoned**: vLLM has no MXFP4 kernels for
> RDNA2 (the upstream gfx1030 PR was closed unmerged, and it never covered
> gpt-oss/MXFP4). Don't attempt it.

## Files

| Script | Purpose |
|---|---|
| `serve_llamacpp.sh` | start the brain; build line in its header; all knobs are env vars |
| `smoke_chat.sh` | one request that must return strict JSON — proves server + template + parsing |
| `sweep.py` | concurrency sweep: aggregate tok/s, latency, TTFT, peak KV/slots per K |
| `watch_metrics.sh` | live `/metrics`: running/waiting slots and KV occupancy |

## Start + smoke test

```bash
./serve_llamacpp.sh                       # defaults: 8 slots, 32768 total ctx, :8000
PARALLEL=12 CTX_TOTAL=24576 ./serve_llamacpp.sh
MODEL=/path/to/gpt-oss-20b-mxfp4.gguf ./serve_llamacpp.sh
./smoke_chat.sh                           # expect: OK: action=... cmd=...
```

## Concurrency: per-agent context = total ÷ parallel

`-c CTX_TOTAL` is the **total** KV pool, split across `--parallel` slots, so
per-agent context = `CTX_TOTAL / PARALLEL` (32768 / 8 = 4K). gpt-oss reasoning
tokens count toward that, so keep agent prompts ≤ 2K (`reasoning_effort: low`)
and **raise `--parallel` for more agents** rather than growing per-slot context.
`--kv-unified` lets short requests borrow space from the shared pool, so a few
long agents don't starve the rest.

## Find the slot ceiling

With the server up and `./watch_metrics.sh` in another terminal:

```bash
python sweep.py --ks 1,2,4,8,12,16 --rounds 3 --max-tokens 256 --fixed-len --csv sweep.csv
```

The ceiling is the K where `agg_tok_s` stops rising or `peak_waiting > 0`
(requests queuing for a slot). Set `--parallel` to that K. Add `--shared-prefix`
to model agents sharing a system prompt. `sweep.py` auto-detects llama.cpp vs
vLLM metric names, so it works against either.

## Live metrics

`llama-server --metrics` exposes Prometheus at `http://127.0.0.1:8000/metrics`:
`llamacpp:requests_processing` (busy slots), `llamacpp:requests_deferred`
(queued), and a KV-usage ratio. `watch_metrics.sh` prints these every 0.5 s.
If bound to `0.0.0.0`, firewall port 8000 — there's no auth.

## Fallbacks

- **Vulkan backend** instead of HIP: build llama.cpp with `-DGGML_VULKAN=ON`
  (needs `vulkan-headers` + `shaderc`). On RDNA2 it's often as fast as HIP for
  decode — benchmark both with `sweep.py`.
- **Rent an H100 (80 GB)**: stock CUDA vLLM, no build —
  `docker run --gpus all -p 8000:8000 vllm/vllm-openai:<current> --model openai/gpt-oss-20b --max-num-seqs 128`.
  Point `swarm run --model` at it; `sweep.py`/`watch_metrics.sh` work unchanged.
