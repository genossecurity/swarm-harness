"""Flask front-end for the swarm harness.

Serves a control panel + a live, brainstorming-style view of the agents. The
orchestrator streams every event to a pub/sub Hub; the browser subscribes over
Server-Sent Events (/stream) and renders per-agent feeds + group consensus.

Run it with ./web/run_web.sh (sets up a venv with Flask), then open
http://127.0.0.1:5000.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
from pathlib import Path

# Make the sibling `swarm` package importable when run as `python web/app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, Response, jsonify, request, send_from_directory  # noqa: E402

from swarm.config import load_config  # noqa: E402
from swarm.orchestrator import Orchestrator  # noqa: E402

HERE = Path(__file__).resolve().parent
app = Flask(__name__)


class Hub:
    """Fan-out of run events to any number of SSE subscribers, with replay so a
    tab opened (or refreshed) mid-run catches up on the current run."""

    def __init__(self):
        self._subs: set[queue.Queue] = set()
        self._lock = threading.Lock()
        self._buffer: list[dict] = []

    def reset(self):
        with self._lock:
            self._buffer = []

    def publish(self, ev: dict):
        with self._lock:
            self._buffer.append(ev)
            for q in list(self._subs):
                q.put(ev)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            for ev in self._buffer:      # replay the current run
                q.put(ev)
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            self._subs.discard(q)


class Run:
    def __init__(self):
        self.hub = Hub()
        self.orch: Orchestrator | None = None
        self.thread: threading.Thread | None = None
        self.running = False
        self.lock = threading.Lock()


RUN = Run()


def _build_overrides(d: dict) -> dict:
    mode = d.get("mode", "external")
    simple_mode = bool(d.get("simple_mode", False))
    ov = dict(
        agents=int(d.get("agents", 4)),
        task=d.get("task", "").strip(),
        base_url=(d.get("base_url") or "http://localhost:11434/v1").strip(),
        model=(d.get("model_name") or "gpt-oss:20b").strip(),
        auto_scale=bool(d.get("auto_scale", True)),   # size the pool + allocate tools from the task
        auto_tools=bool(d.get("auto_scale", True)),
        max_steps=int(d.get("max_steps", 20)),
        gate=False if simple_mode else bool(d.get("gate", True)),
        interactive_questions=simple_mode,
        log=(d.get("log") or "run.jsonl").strip(),
    )
    # Steps: a list of {name, task, tools}. Each step is a phase with its own
    # objective and its own set of allowed tools.
    steps = d.get("steps")
    if simple_mode:
        from swarm.pipeline import stage_config
        phases, objectives, tools, stages = stage_config(ov["task"], d.get("target", "").strip())
        ov["phases"] = phases
        ov["step_tasks"] = objectives
        ov["step_tools"] = tools
        ov["pipeline"] = True
        ov["pipeline_stages"] = stages
    elif isinstance(steps, list) and steps:
        names, step_tasks, step_tools = [], {}, {}
        for i, s in enumerate(steps):
            name = (s.get("name") or f"step{i+1}").strip()
            names.append(name)
            if (s.get("task") or "").strip():
                step_tasks[name] = s["task"].strip()
            if isinstance(s.get("tools"), list) and s["tools"]:
                step_tools[name] = [t for t in s["tools"] if t]
        ov["phases"] = names
        ov["step_tasks"] = step_tasks
        ov["step_tools"] = step_tools
    else:
        phases = (d.get("phases") or "").strip()
        if phases:
            ov["phases"] = [p.strip() for p in phases.split(",") if p.strip()]
    if mode in ("external", "local"):
        ov["target_mode"] = mode
        ov["target_host"] = (d.get("target") or "localhost").strip()
    else:  # ssh
        ov["target_host"] = (d.get("target") or "").strip()
        ov["target_user"] = (d.get("target_user") or "").strip() or None
        ov["target_key"] = (d.get("target_key") or "").strip() or None
    return ov


@app.route("/")
def index():
    return send_from_directory(HERE, "simple.html")


@app.route("/advanced")
def advanced():
    return send_from_directory(HERE, "index.html")


@app.route("/tools")
def tools():
    from toolmcp.tools import BUILTIN_TOOLS
    return jsonify([{"name": s.name, "description": s.description,
                     "params": list(s.params), "requires": list(s.required_bins)}
                    for s in BUILTIN_TOOLS.values()])


@app.route("/pipeline")
def pipeline_preview():
    from swarm.pipeline import plan_pipeline
    task = request.args.get("task", "")
    host = request.args.get("target", "")
    return jsonify(stages=plan_pipeline(task, host))


@app.route("/defaults")
def defaults():
    return jsonify(base_url="http://localhost:11434/v1", model_name="gpt-oss:20b",
                   agents=4, mode="external", phases="explore,verify,report", max_steps=20)


@app.route("/start", methods=["POST"])
def start():
    with RUN.lock:
        if RUN.running:
            return jsonify(ok=False, error="a run is already in progress"), 409
        data = request.get_json(force=True, silent=True) or {}
        try:
            cfg = load_config(None, **_build_overrides(data))
        except Exception as e:
            return jsonify(ok=False, error=f"bad config: {e}"), 400
        if cfg.target is None:
            return jsonify(ok=False, error="need a target host (or pick external/local mode)"), 400

        RUN.hub.reset()
        RUN.orch = Orchestrator(cfg, on_event=RUN.hub.publish)
        RUN.running = True

    def _go():
        try:
            RUN.orch.run()
        except Exception as e:
            RUN.hub.publish({"event": "run_error", "error": str(e)})
        finally:
            RUN.running = False
            RUN.hub.publish({"event": "eos"})

    RUN.thread = threading.Thread(target=_go, name="swarm-run", daemon=True)
    RUN.thread.start()
    return jsonify(ok=True)


@app.route("/stop", methods=["POST"])
def stop():
    if RUN.orch and RUN.running:
        RUN.orch.request_stop()
    return jsonify(ok=True)


@app.route("/approve", methods=["POST"])
def approve():
    """Human approves moving past the current phase gate."""
    if RUN.orch and RUN.running:
        RUN.orch.approve()
    return jsonify(ok=True)


@app.route("/reply", methods=["POST"])
def reply():
    answer = str((request.get_json(force=True, silent=True) or {}).get("answer", "")).strip()
    if not answer:
        return jsonify(ok=False, error="answer is empty"), 400
    if not RUN.orch or not RUN.orch.answer_user(answer):
        return jsonify(ok=False, error="there is no pending question"), 409
    return jsonify(ok=True)


@app.route("/status")
def status():
    return jsonify(running=RUN.running)


@app.route("/stream")
def stream():
    q = RUN.hub.subscribe()

    def gen():
        yield "retry: 2000\n\n"
        try:
            while True:
                ev = q.get()
                yield f"data: {json.dumps(ev)}\n\n"
        finally:
            RUN.hub.unsubscribe(q)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)
