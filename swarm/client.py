"""OpenAI-compatible chat client (stdlib only) + strict-JSON turn with retries.

Points at llama-server's /v1 by default. On malformed output it re-asks up to
`max_retries` times with an escalating "JSON only" nudge, and records the retry
count so the orchestrator can report an aggregate retry rate.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from .protocol import ACTIONS, Action, ProtocolError, SCHEMA_HINT, parse_action


def _from_tool_call(tc: dict) -> str:
    """Best-effort map an OpenAI tool_call into our JSON action string."""
    fn = tc.get("function", {}) or {}
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    if not isinstance(args, dict):
        args = {}
    name = (fn.get("name") or "").lower()
    obj = dict(args)
    obj.setdefault("action", name if name in ACTIONS else ("run" if args.get("cmd") else "note"))
    obj.setdefault("reason", "(from tool call)")
    return json.dumps(obj)


@dataclass
class TurnResult:
    action: Action
    retries: int          # how many malformed replies preceded the good one
    ok: bool              # False if all attempts failed (action is a fallback "note")


class LLMClient:
    def __init__(self, base_url: str, model: str, temperature: float = 0.3,
                 max_retries: int = 3, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        # reasoning_effort / chat_template_kwargs are gpt-oss (harmony) specific. Sending
        # them to other models (e.g. qwen2.5-coder on Ollama) makes the server error, so
        # only include them for gpt-oss-style models.
        self.harmony = "gpt-oss" in model.lower()

    def preflight(self) -> tuple[bool, str]:
        """Cheap reachability check before a run. Returns (ok, detail)."""
        try:
            req = urllib.request.Request(self.base_url + "/models")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status < 500, f"HTTP {resp.status}"
        except urllib.error.HTTPError as e:
            # Server answered (e.g. 404 if /models is absent) — it's up, that's enough.
            return True, f"HTTP {e.code}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def _post(self, messages: list[dict], *, max_tokens: int = 512,
              temperature: float | None = None) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if self.harmony:                    # gpt-oss only — breaks other models
            body["reasoning_effort"] = "low"
            body["chat_template_kwargs"] = {"reasoning_effort": "low"}
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("OPENAI_API_KEY")   # only for remote endpoints; local needs none
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.load(resp)
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        if not content.strip() and msg.get("tool_calls"):
            # Safety net: some gpt-oss servers (e.g. Ollama) route the action into a
            # native tool call and leave content empty. Rebuild JSON from its arguments.
            content = _from_tool_call(msg["tool_calls"][0])
        return content

    def chat(self, messages: list[dict], *, temperature: float = 0.2,
             max_tokens: int = 512) -> str:
        """A plain chat completion returning raw text — no protocol parsing, no
        retries. Used by the RLHF eval gate (LLM-as-judge) so it shares one HTTP
        client with the harness instead of carrying its own."""
        return self._post(messages, max_tokens=max_tokens, temperature=temperature)

    def turn(self, system: str, user: str, stop_check=None) -> TurnResult:
        """One agent decision. Returns a validated Action, retrying on bad JSON.

        stop_check: optional callable; if it returns True we abandon further retries
        so a Stop press is honored quickly instead of burning 3× more model calls.
        """
        messages = [
            {"role": "system", "content": system + "\n\n" + SCHEMA_HINT},
            {"role": "user", "content": user},
        ]
        last_err = ""
        for attempt in range(self.max_retries + 1):
            if stop_check and stop_check():
                return TurnResult(action=Action(action="note", reason="stopped"), retries=attempt, ok=False)
            try:
                content = self._post(messages)
            except (urllib.error.URLError, KeyError, TimeoutError) as e:
                last_err = f"transport error: {e}"
                content = ""
            try:
                return TurnResult(action=parse_action(content), retries=attempt, ok=True)
            except ProtocolError as e:
                last_err = str(e)
                # Feed the bad output back and tighten the instruction.
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content":
                    f"That was not valid. Error: {e}. Reply with ONE JSON object only, "
                    f"no reasoning, no fences. {SCHEMA_HINT}"})
        # Exhausted retries: degrade to a harmless note so the loop keeps going.
        fallback = Action(action="note", reason=f"malformed output after {self.max_retries} retries: {last_err}")
        return TurnResult(action=fallback, retries=self.max_retries, ok=False)
