"""The agent action protocol: strict JSON in the message body.

We do NOT rely on native tool-calling (gpt-oss tool calls over this endpoint are
inconsistent). Each agent turn must be exactly one JSON object matching ACTIONS.
`extract_json` is tolerant of the usual gpt-oss failure modes (code fences,
reasoning text leaking before/after the object); validation is strict.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

# action -> required fields (beyond the always-present "action" and "reason")
ACTIONS = {
    "run":      ("cmd",),            # execute raw cmd on the target, share compressed result
    "tool":     ("tool",),          # call a named tool with "args" (see swarm/tools.py)
    "plan":     ("goals",),         # manager: state the goals for this step before dispatching
    "dispatch": ("tasks",),         # manager: hand sub-tasks to the worker pool
    "ask":      ("question",),      # manager: pause and ask the user for missing task context
    "propose":  ("proposal",),      # post a finding/claim to the whiteboard (unconfirmed)
    "confirm":  ("confirm",),       # verify proposal id `confirm` via its own "cmd" or "tool"
    "vote":     ("phase_done_vote",),# vote on whether the current phase is finished
    "note":     (),                 # just record `reason` on the whiteboard; no side effect
}

SCHEMA_HINT = (
    'You have NO tools and NO functions available. Never emit a tool call or function call. '
    'Output ONLY a raw JSON object as plain text in your final answer - no prose, no '
    'markdown fences.\n'
    'Schema: {"action": "run|tool|plan|dispatch|ask|propose|confirm|vote|note", "reason": "<one sentence>", '
    '"cmd": "<shell command, for run or confirm>", '
    '"tool": "<tool name, for tool or confirm>", "args": {<tool args>}, '
    '"proposal": "<claim, for propose>", '
    '"confirm": "<proposal id you are verifying, for confirm>", '
    '"question": "<clarification for the user, for ask>", '
    '"phase_done_vote": true|false}\n'
    'Prefer a tool from the list below over raw "run" when one fits.\n'
    'A "proposal" is a FINDING stated as a sentence (e.g. "Host 10.0.0.5 is alive" or '
    '"TCP 22 is open"), NEVER a command. To run something, use action "run" with "cmd".'
)

# Probing/CLI binaries an agent might wrongly drop into the `proposal` field.
# Deliberately excludes words that double as English (host, ip, cat, ls, echo, …) so a
# claim like "Host X is alive" is not misread as the `host` command.
_CMD_WORDS = {
    "ping", "nmap", "nc", "ncat", "curl", "wget", "dig", "nslookup", "ssh", "scp",
    "traceroute", "tracepath", "mtr", "netstat", "telnet", "arp", "ifconfig", "whois",
}
_FLAG_RE = re.compile(r"^\S+\s+-{1,2}\w")             # "nmap -sT", "curl --head"
_SHELL_META_RE = re.compile(r"[|&;><]|\$\(|`")        # pipes, redirects, subshells
_ARGLIKE_RE = re.compile(r"^(-|/|https?://|\d)")      # flag, path, url, or number/IP/port


def looks_like_command(text: str) -> bool:
    """True if a string reads as a shell command rather than a declarative claim."""
    s = (text or "").strip()
    if not s:
        return False
    if _SHELL_META_RE.search(s) or _FLAG_RE.match(s):   # catches "nmap -sT", "curl -I …:8080"
        return True
    parts = s.split()
    first = parts[0].lower().strip("`'\"/")
    if first in _CMD_WORDS:
        return len(parts) == 1 or bool(_ARGLIKE_RE.match(parts[1]))  # "nmap 10.0.0.5", bare "nmap"
    return False


class ProtocolError(ValueError):
    """Raised when a model turn cannot be parsed or fails validation."""


@dataclass
class Action:
    action: str
    reason: str = ""
    cmd: str | None = None
    tool: str | None = None
    args: dict | None = None
    tasks: list | None = None
    goals: list | None = None
    proposal: str | None = None
    confirm: str | None = None
    question: str | None = None
    phase_done_vote: bool | None = None
    raw: str = ""


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)


def extract_json(text: str) -> dict:
    """Pull the first balanced {...} object out of a model message body."""
    if not text or not text.strip():
        raise ProtocolError("empty model output")
    t = _FENCE.sub("", text.strip())
    start = t.find("{")
    if start < 0:
        raise ProtocolError("no JSON object in output")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except json.JSONDecodeError as e:
                    raise ProtocolError(f"malformed JSON: {e}") from e
    raise ProtocolError("unterminated JSON object")


def parse_action(text: str) -> Action:
    """Parse + validate one model turn into an Action, or raise ProtocolError."""
    obj = extract_json(text)
    if not isinstance(obj, dict):
        raise ProtocolError("top-level JSON is not an object")
    action = obj.get("action")
    if action not in ACTIONS:
        raise ProtocolError(f"unknown action {action!r}; expected one of {sorted(ACTIONS)}")
    for field in ACTIONS[action]:
        if obj.get(field) in (None, ""):
            raise ProtocolError(f"action {action!r} requires non-empty {field!r}")
    vote = obj.get("phase_done_vote")
    if action == "vote" and not isinstance(vote, bool):
        raise ProtocolError("phase_done_vote must be a boolean")
    if action == "propose" and looks_like_command(obj.get("proposal", "")):
        raise ProtocolError("'proposal' must be a finding stated as a sentence, not a command; "
                            "to run something use action 'run' with 'cmd' or 'tool'")
    if action == "confirm" and not (obj.get("cmd") or obj.get("tool")):
        raise ProtocolError("'confirm' needs a 'cmd' or a 'tool' to verify the proposal")
    args = obj.get("args")
    if args is not None and not isinstance(args, dict):
        raise ProtocolError("'args' must be an object")
    goals = obj.get("goals")
    if action == "plan":
        if isinstance(goals, (str, dict)):
            goals = [goals]
        if not isinstance(goals, list):
            raise ProtocolError("'plan' needs a non-empty 'goals' list")
        norm = []
        for g in goals:
            # each goal is a string, or {goal, expect} — "for each goal, what I expect to find"
            if isinstance(g, str) and g.strip():
                norm.append({"goal": g.strip(), "expect": ""})
            elif isinstance(g, dict) and str(g.get("goal", "")).strip():
                norm.append({"goal": str(g["goal"]).strip(),
                             "expect": str(g.get("expect", "") or g.get("expected", "")).strip()})
        if not norm:
            raise ProtocolError("'plan' needs a non-empty 'goals' list")
        goals = norm
    tasks = obj.get("tasks")
    if action == "dispatch":
        if not isinstance(tasks, list) or not tasks:
            raise ProtocolError("'dispatch' needs a non-empty 'tasks' list")
        if not all(isinstance(t, dict) and (t.get("tool") or t.get("check")) for t in tasks):
            raise ProtocolError("each dispatched task needs a 'tool' (with args) or a 'check'")
    return Action(
        action=action,
        reason=str(obj.get("reason", "")),
        cmd=obj.get("cmd"),
        tool=obj.get("tool"),
        args=args if isinstance(args, dict) else None,
        tasks=tasks if isinstance(tasks, list) else None,
        goals=goals if isinstance(goals, list) else None,
        proposal=obj.get("proposal"),
        confirm=obj.get("confirm"),
        question=obj.get("question"),
        phase_done_vote=vote if isinstance(vote, bool) else None,
        raw=text,
    )
