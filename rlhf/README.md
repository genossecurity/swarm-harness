# pentest-rlhf

RLHF pipeline to fine-tune a pentest-assistant LLM, with a working pentester as
the human in the loop. Offline collection → nightly/weekly training. SFT first,
then DPO. Execution-outcome reward (RLVR-style) is blended into preference-pair
construction so the signal is not preferences alone.

## The loop

```
base/SFT model → agent assists pentester on a sandboxed target
              → capture candidate outputs + human signal + execution outcome
              → build preference pairs (chosen/rejected)
              → train (SFT → DPO)
              → redeploy policy → repeat
```

This is the **harness-layer** RLHF pipeline. It lives under `rlhf/` and depends
on the **MCP tool layer** (`toolmcp/`) for sandboxed execution and on the harness
client (`swarm/`) for model calls — never the reverse.

## Layout

```
rlhf/
  policy.py        model-under-training; samples k candidates
  session.py       engagement loop; emits FeedbackEvents (executes via toolmcp.sandbox)
  eval.py          promotion gate — win-rate vs current policy + safety floor (uses swarm.client)
  registry.py      model versions + training provenance + the `current` pointer
  hitl/            events.py, store.py, ui.py — feedback schema, append-only log, review UI
  data/            build_pairs.py, build_sft.py, synth.py — raw events -> SFT/DPO data
  train/           sft.py, dpo.py, preflight.py — TRL trainers + no-GPU validator
  config/          sft.yaml, dpo.yaml, *_qlora.yaml
toolmcp/sandbox.py the allowlist + lab-CIDR + kill-patterns + outcome classify (enforcement)
registry/          (data) manifest store written by rlhf.registry
```

## Order of operations

1. Collect demonstration transcripts (pentester working *with* the agent) → SFT.
   Never DPO a base model cold.
2. Collect preference/outcome data during real (sandboxed) engagements.
3. `python -m rlhf.data.build_pairs` turns raw events into chosen/rejected pairs,
   fusing the human signal with execution outcome.
4. `python -m rlhf.train.dpo config/dpo_qlora.yaml` trains the policy off those pairs.
5. `python -m rlhf.eval` gates promotion: win-rate vs the current policy + a
   safety regression check. Only promote if both pass.

## Safety (not optional)

The agent issues real tool calls during collection. That means live commands.
Run only against disposable VMs on an isolated lab network with no route to
anything real or anyone else's systems. The sandbox is the primary risk surface
of this project — more than the model. The allowlist and kill conditions now
live in `toolmcp/sandbox.py` (the MCP layer's enforcement choke point), shared
with the live harness so there is a single boundary.

## Install

The collection + data + eval path is **pure stdlib** — no install needed to run
it or its tests. Training is the only part that needs the ML stack:

```
pip install "trl>=0.12" transformers datasets accelerate peft argilla pyyaml
```

## Quickstart (no GPU, no deps)

Prove the whole flow before collecting real data or touching a trainer:

```
python -m rlhf.data.synth --out rlhf/data/raw                       # fabricate sample events
python -m rlhf.data.build_pairs --raw rlhf/data/raw --out rlhf/data/splits/pairs.jsonl
python -m unittest discover -s tests                      # 34 tests, no network
```

`synth.py` writes one event per signal type and exercises every branch of the
outcome gate (verified success, human-edited gold, the failed-pick salvage swap,
the un-salvageable drop, rejected-all, rated-only, skip). Delete `rlhf/data/raw/`
before you start collecting for real.

## Execution backend (local)

`toolmcp/sandbox.py` (the MCP layer) runs the agent's commands as **local subprocesses** that reach
lab targets over the network — it does not SSH into a separate runner. The
guardrails are therefore the only boundary, and they fail closed. Configure the
lab net via env; an unconfigured install refuses everything:

```
PENTEST_LAB_CIDRS=192.168.56.0/24   # YOUR disposable-lab net (default is a fake placeholder)
PENTEST_SELFTEST=1                  # also allow 127.0.0.0/8 — smoke-testing only
PENTEST_EXEC_TIMEOUT=120            # per-command wall-clock seconds
```

`validate()` refuses any command whose binary is off the allowlist, whose target
is outside `PENTEST_LAB_CIDRS`, that carries no target at all, or that trips a
kill-pattern. `execute()` then runs it (no shell), and `classify()` maps the
result to the success/failure/error outcome the reward gate reads.

## Eval gate (no training needed)

`rlhf/eval.py` reaches models over an OpenAI-compatible endpoint (your
llama.cpp / Ollama server) via stdlib `urllib` — no weights loaded:

```
OPENAI_BASE_URL=http://localhost:8000/v1 JUDGE_MODEL=gpt-oss:20b \
  python -m rlhf.eval --candidate policy-dpo --current policy-sft \
    --eval eval/held_out.jsonl --safety eval/safety.jsonl
```

Model specs are `name` (default base) or `name@http://host/v1`. `evaluate()`
takes its judge/refusal scorers as arguments, so the promotion arithmetic is
unit-tested with deterministic fakes.

## Training path

Training needs the ML stack + a GPU. On ROCm, bitsandbytes 4-bit is rough, so
the realistic split is **train on a rented NVIDIA box, infer locally** via
llama.cpp (see `config/sft_qlora.yaml`). Everything up to the trainer is stdlib
and runs anywhere:

```
# 1. demonstrations -> SFT transcripts (human-authored/endorsed gold only;
#    commands that ran and FAILED are dropped, not taught)
python -m rlhf.data.build_sft   --raw rlhf/data/raw --out rlhf/data/splits/sft.jsonl
python -m rlhf.data.build_pairs --raw rlhf/data/raw --out rlhf/data/splits/pairs.jsonl

# 2. validate BEFORE burning GPU hours (no deps, no GPU)
python -m rlhf.train.preflight --stage sft --config rlhf/config/sft_qlora.yaml
python -m rlhf.train.preflight --stage dpo --config rlhf/config/dpo_qlora.yaml

# 3. train on the GPU box (needs trl/transformers/peft/accelerate)
python -m rlhf.train.sft rlhf/config/sft_qlora.yaml      # SFT first — never DPO a base model cold
python -m rlhf.train.dpo rlhf/config/dpo_qlora.yaml      # starts from the SFT checkpoint

# 4. gate + promote
python -m rlhf.eval --candidate policy-dpo --current policy-sft ...
python registry/registry.py list               # versions + provenance + eval
```

`preflight.py` fails loudly on malformed records, empty files, `chosen == rejected`
pairs, over-length prompts, and a DPO config with no SFT parent — the mistakes
that otherwise waste a full run. Each trainer writes `registry/<version>/manifest.json`
recording the base model, parent policy, data file (by sha256) and config, so
every policy traces back to the exact events that shaped it. `registry.promote()`
refuses any version whose eval gate didn't pass.
