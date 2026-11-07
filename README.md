# swarm-harness

A **single‑agent orchestrator** that manages a swarm of lightweight bots. The orchestrator talks to a single local Llama.cpp model (or any OpenAI‑compatible endpoint) and delegates work to many independent bot processes. Each bot runs the same logic but is isolated, allowing the system to scale horizontally while keeping the model usage serial.

## Architecture

```
+-------------------+          +-------------------+          +-------------------+
|  Orchestrator     |  <---->  |  Bot 1            |  <---->  |  Bot N            |
|  (Python process) |  (TCP)   |  (Python process) |  (TCP)   |  (Python process) |
+-------------------+          +-------------------+          +-------------------+
          ^                            ^                            ^
          |                            |                            |
          +----------------------------+----------------------------+
                           Model endpoint (Llama.cpp / Ollama)
```

* The **orchestrator** is the only component that talks to the LLM. It receives a task, splits it into a sequence of *steps*, and sends each step to a bot.
* Each **bot** receives a step, performs the required tool calls, and returns the result to the orchestrator. Bots are stateless and can be started or stopped independently.
* Communication between orchestrator and bots is done over a lightweight TCP protocol (JSON‑over‑TCP). This keeps the implementation simple and language‑agnostic.

## Benefits

* **Serial LLM usage** – only one model request at a time, avoiding token‑budget contention.
* **Horizontal scalability** – add more bots to increase throughput without touching the model.
* **Fault isolation** – a bot crash does not affect the orchestrator or other bots.
* **Simplified deployment** – the orchestrator can run on a small machine; bots can be distributed across a cluster.

## Getting Started

```bash
# 1. Start the Llama.cpp server (or any OpenAI‑compatible endpoint)
./serving/serve_llamacpp.sh

# 2. Run the orchestrator with a single agent and 4 bots
#    The orchestrator will spawn 4 bot processes automatically.
swarm run --agents 1 --bots 4 --target <ip> --model http://localhost:8000/v1
```

The orchestrator will log each step to `run.jsonl`. Bots can be monitored via the built‑in Flask UI at `http://127.0.0.1:5000/advanced`.

## Configuration

* `--agents` – number of orchestrator instances (usually 1).
* `--bots` – number of bot processes to spawn.
* `--target` – IP or hostname of the machine to probe.
* `--model` – URL of the OpenAI‑compatible LLM endpoint.

## Extending

Bots are simple Python scripts that import `swarm.bot`. To add a new bot type, create a subclass of `BaseBot` and register it in `swarm.bot_registry`. The orchestrator will automatically discover and launch it.

---

For detailed API documentation, see the `docs/` directory.
