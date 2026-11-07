# swarm-harness

A multi-agent swarm that runs against **one local llama.cpp brain** serving
`gpt-oss-20B` (MXFP4) on an AMD RX 6950 XT (gfx1030), OpenAI-compatible on
`:8000`. Many agents share one endpoint and one target VM, coordinate on a
shared whiteboard, cross-check each other's findings, and advance through
phases by consensus.

The model runs on **Ollama** (`gpt-oss:20b`, OpenAI-compatible at
`http://localhost:11434/v1`) — gpt-oss-20B MXFP4 runs natively on this card,
~13 GB weights, ~90 t/s. (`serving/serve_llamacpp.sh` is an alternative that
serves the same API on `:8000`; stop Ollama first so it isn't holding VRAM. The
from-source vLLM path was abandoned — vLLM has no MXFP4 kernels for RDNA2.)

## Two layers (MCP + harness)

The project is split into two distinct layers so the division of labor is clean:

```
toolmcp/     MCP TOOL LAYER — what can run and how it's guarded. Depends on nothing else.
  tools.py     the tool registry: ToolSpec.build(args) -> shell-quoted command
  executor.py  command execution (local / ssh / external), content-neutral
  sandbox.py   enforcement choke point: allowlist + lab-CIDR + kill-patterns + outcome classify
  server.py    list_tools() / call_tool() — the MCP API (stdio JSON-RPC transport: next step)

swarm/       HARNESS LAYER — the agent loop, orchestration, serving client. MCP client.
rlhf/        HARNESS LAYER — the RLHF pipeline that trains the policy the harness runs.
```

The dependency only runs one way: `swarm` and `rlhf` import `toolmcp`, never the
reverse — so the same sandboxed tools can be exposed to any MCP client (Claude
Code included) once the stdio server lands. The RLHF pipeline reuses the harness
LLM client (`swarm.client`) and the tool layer's sandbox, so there is one HTTP
client and one execution/enforcement path, not duplicates. See `rlhf/README.md`
for the training loop and `docs/pentest-platform.pdf` for the merged design.

Everything runs stdlib-only and offline: `python -m unittest discover -s tests`
(129 tests — harness + tool layer + RLHF).

## Web UIs

Open `http://127.0.0.1:5000/` for the **Simple** chat interface. Give it a task
and target; the manager can pause to ask a focused clarification question while
it works. Use `/advanced` for the full run controls and live agent, dispatch,
whiteboard, and consensus views described below.

Simple mode previews a task-specific testing pipeline and reports its current
stage and suggested tools. It runs service discovery first, adds web app
identification when the task asks about an app or web service, and adds
vulnerability checks only when requested. Core discovery stages use fixed
structured tools. After each web or vulnerability result, the model gets the
accumulated evidence and the stage's installed tools, then picks a small number
of targeted follow-up checks or stops when they would add little. Those choices
are limited to structured tools and the configured target; the UI shows why a
follow-up was selected.

All agents use one model endpoint and model name.

The **Advanced** Flask UI has a control panel (task box, mode buttons, agent count,
target, phases, model) and a live visual of the run: each agent gets its own box
with a scrolling feed (what it ran, the result, proposals, confirmations), a task
banner on top (with the probing tools detected on the host), a **Hive mind**
panel — a shared board where each finding appears as a card the moment an agent
posts it, attributed to its proposer and filling with confirmer chips as other
agents verify it — a group-consensus panel (phase tracker + confirmed findings),
and a **Final task results** section
on completion — the verdict plus each consensus-confirmed finding (who proposed,
who confirmed) and run stats. Events stream over Server-Sent Events as the agents
work.

**Human approval between steps.** When a phase reaches consensus, the swarm
**pauses** and shows a checkpoint box — "Phase X is complete and confirmed. Move
on to Y?" — with the completed step and the next step. The agents block until you
press **Yes, move to next step** (or **Stop here**). Nothing advances without you.
(This is on by default in the web UI; the CLI runs unattended unless `gate=True`.)
**Stop** ends the run promptly at any point, including while parked at a gate.

```bash
./web/run_web.sh          # first run makes a .venv-web with Flask, then serves
# open http://127.0.0.1:5000 for Simple, or /advanced for full controls
```

Defaults point at Ollama (`gpt-oss:20b`). The model must be running (Ollama, or
`serve_llamacpp.sh`) before you press Start.

## Three-command quickstart (CLI)

```bash
# 1. Start the brain (gpt-oss-20B on llama.cpp, OpenAI API on :8000)
./serving/serve_llamacpp.sh

# 2. Confirm it answers with strict JSON (server + harmony template + parsing)
./serving/smoke_chat.sh

# 3. Run 4 agents end-to-end against your lab VM
pip install -e .          # one-time: puts `swarm` on PATH (no third-party deps)
swarm run --agents 4 --target <ip> --model http://localhost:8000/v1 --log run.jsonl
```

`--model` is the OpenAI-compatible base URL (named to match the quickstart).
Target auth is **key-based SSH only**; add `--target-user`, `--target-key`,
`--target-port`, or put them in a config file (see `swarm.example.toml`).

Prove the loop with one agent first:

```bash
swarm run --agents 1 --target <ip> --model http://localhost:8000/v1
# or, with no VM, exercise the whole loop locally:
swarm run --agents 1 --local --model http://localhost:8000/v1
```

### Three target modes

| Mode | Flag | SSH key? | What the agents do |
|---|---|---|---|
| **ssh** | `--target <ip>` | **yes** | Log into the target and run commands *on it*. |
| **external** | `--external --target <ip>` | **no** | Run commands *on this machine* to probe a remote host over the network — the host runs no agent/model. For "is `<ip>` alive?", open-port checks, HTTP health, DNS. |
| **local** | `--local` | no | Run commands on this machine (self-test / no-GPU demo). |

External mode is for checking a machine that isn't part of the swarm — the agents
never connect into it; they `ping`/`curl`/`nc`/`dig` it from here and confirm each
other's results. A liveness claim is only accepted if a probe actually succeeds
(so a down host is correctly *not* confirmed):

```bash
swarm run --external --target 192.168.1.50 --agents 4 \
  --task "Check whether the host is alive and which common ports are open" \
  --model http://localhost:8000/v1 --log run.jsonl
```

## Orchestration: manager / worker (default for 2+ agents)

Instead of N peer agents all running the same loop (redundant), the lead acts as a
**manager**: for any task it decomposes the step into discrete checks and
**dispatches** them to a worker pool via a shared dispatch board. Each check runs
**once**:

- A concrete tool call (`{tool, args}`) a worker runs **directly — no model call**.
- A natural-language check (`{check, toolkit}`) is handed to a worker agent using
  the shared model, which turns it into a command. The manager assigns a **toolkit** from the tool
  **catalog** (categories: liveness, ports, web, dns, vuln, path) so a worker only gets the
  relevant tools.

Workers pull tasks from the board and run them concurrently; the manager reads the
results, records findings, and decides when the step is done (a convergence guard
finishes a step if the manager stops making progress). Concrete work executes
directly; natural-language checks use the shared model. Set `orchestration = "peer"` to use the legacy
all-peers mode, or `"manager"`/`"auto"` (default) for the manager pool.

## How it works (shared mechanics)

- **No native tool-calling.** gpt-oss tool calls over this endpoint are
  inconsistent, so agents emit **strict JSON in the message body** and the code
  parses it. Schema: `{action, cmd?, proposal?, confirm?, phase_done_vote?, reason}`
  with `action ∈ {run, propose, confirm, vote, note}`. Malformed output is
  retried up to 3× (the bad reply is fed back with a tighter instruction); after
  that the turn degrades to a harmless `note` so the swarm never wedges. The
  aggregate **retry rate** is logged in the `run_end` event.
- **Shared whiteboard (hive mind).** Every agent's context carries a
  **SHARED FINDINGS** board — every proposal made so far across *all* steps,
  attributed and marked confirmed/unconfirmed (`[P1 · agent-0] Ports 22,80 open
  (confirmed by agent-1)`). Agents are told to build on it and not repeat a probe
  whose finding is already there, so they pool knowledge instead of duplicating.
- **Per-step objectives & tools.** Each step (phase) can carry its own objective
  (`step_tasks`) and its own set of allowed tools (`step_tools`). The web UI has a
  steps editor — one objective box and a row of tool chips per step.
- **Structured tool calls.** Besides raw `run`, agents can call named,
  parameterized tools via the `tool` action (`{tool, args}`): `ping`, `port_scan`,
  `service_scan`, `full_scan`, `tcp_connect`, `http_head`, `http_get`, `web_discover`, `dns_lookup`,
  `nmap_safe_vuln`, `testssl`, `nuclei_scan`, `nikto_scan`, `traceroute`
  (see [swarm/tools.py](swarm/tools.py)). Args are shell-quoted when the command is
  built, so tool calls are injection-safe and auditable, and each step only exposes
  the tools you allow it (further filtered to the binaries actually installed).
  Vulnerability checks are grouped in the opt-in `vuln` toolkit: Nmap runs scripts
  tagged both `safe` and `vuln`, testssl.sh uses its fast profile, Nuclei is
  rate-limited with intrusive/fuzzing/brute-force/DoS tags excluded, and Nikto is
  restricted to information and configuration checks with a 25-second scan cap.
- **Context discipline.** Each turn the agent sees the overall task, the current
  step's objective, the shared findings, and the current step's recent activity —
  hard-clamped to ~2K tokens (`ctx_cap_chars`). Raw command output never reaches
  the model — `compress.py`
  reduces it to a few fact lines (`[rc=0] …`) in code first.
- **Grounded confirmation.** A `propose` posts an unconfirmed claim — a *finding*
  stated as a sentence ("Host X is alive"), never a command (a guard rejects
  command-shaped proposals so they can't pollute the board). Another agent
  `confirm`s it by running *its own* verification command; a proposal is only
  marked confirmed when a **different** agent's check succeeds.
- **Tool awareness.** At startup the swarm detects which probing tools
  (`ping`, `curl`, `nmap`, `nc`, `dig`, …) exist on the host and tells the agents,
  so they stop reaching for tools that aren't installed.
- **Consensus phases.** An agent `vote`s a phase done; when `consensus` (default
  0.6) of agents agree, the whiteboard advances. Phases default to
  `explore, verify, report` and are configurable.
- **Persistent SSH session per agent.** OpenSSH ControlMaster multiplexing —
  one authenticated session per agent, reused for every command, no paramiko,
  no passwords. `ConnectTimeout=10` so a bad target fails fast.
- **Event log.** Every turn is one JSON line in `run.jsonl`; `run_start` and
  `run_end` bracket the run.

Pure standard library — no third-party Python dependencies.

## Tests

```bash
python -m unittest discover -s tests        # 51 tests, no network/model needed
```

Covers the protocol (parsing, the command-shaped-proposal guard, tool/confirm
actions), the tool scaffolding (command building, shell-injection safety, per-step
allowlists, availability filtering), the whiteboard (consensus, the approval gate,
grounded confirmation, shared findings), output compression (including the
bytes-on-timeout regression), and config/agent wiring (steps, step_tools, tool
execution).

## Serving notes

Per-agent context = `CTX_TOTAL / --parallel`. With the defaults (32768 / 8) each
agent gets 4K. **Keep agent context ≤ 2K and raise `--parallel` for more
agents.** Tune and benchmark concurrency with `serving/sweep.py` and watch live
KV/slot occupancy with `serving/watch_metrics.sh` (see `serving/README.md`).

## Validation run (actually executed)

Run on this machine on 2026-11-01. The real `llama-server` and a lab VM were not
available here, so the brain was replaced with a **mock OpenAI server** (scripted
JSON turns, ~1-in-8 deliberately malformed) and the target with the **`--local`
executor** running benign commands (`uname -s`) on this box. This exercises every
code path the spec calls out; only the two external endpoints were substituted.

**1 agent** (prove the loop) — `swarm run --agents 1 --local --phases explore`:

```
run_start → run(uname -s)→"[rc=0] Linux" → propose P1 → vote → phase_advanced=true → run_end
all_phases_done: true   total_turns: 3   retries: 0
```
Connect ✓, one command on target ✓, compressed result ✓, whiteboard update ✓,
event logged ✓.

**4 agents**, phases `explore,verify,report`:

```
exit code: 0, no tracebacks
actions: {run: 5, propose: 3, vote: 6, confirm: 2}
phase advances: 3 (expect 3)              # consensus advanced all phases
grounded cross-agent confirmations: 2 by [agent-1, agent-2]
retry_rate: 0.250 (4/16 malformed replies recovered)
run_end: all_phases_done=true cleared=3/3 confirmed=2
```

Concurrent agents all hit the one endpoint ✓; grounded confirmation fired (a
different agent re-ran the proposer's check) ✓; phases advanced by consensus ✓;
`run.jsonl` well-formed (every line parses) ✓; retries recovered and rate
reported ✓. The 0.25 retry rate reflects the mock's injected garbage; with the
real model and the strict schema it will be far lower — if it is *not*, suspect
harmony reasoning tokens leaking into the body and check `--reasoning-format`.

### What was NOT run here

- `serving/serve_llamacpp.sh` against the real GPU (no llama.cpp build / GGUF on
  this box) — shell syntax checked with `bash -n`.
- `serving/smoke_chat.sh` against a live server — same reason.
- The real SSH executor against a lab VM — the argv and fail-fast timeout were
  verified against an unreachable host (failed in 10.1 s as designed).

To run for real: build llama.cpp (header of `serve_llamacpp.sh`), run the three
quickstart commands, then watch for the retry rate and per-agent latency as you
scale `--agents` toward `--parallel`.
