#!/usr/bin/env bash
# Confirm the server + harmony chat template + JSON output path all work.
# Asks for a STRICT JSON reply (the same contract the harness relies on) and
# validates it parses. Works against llama-server or any OpenAI-compatible server.
set -euo pipefail
BASE=${BASE:-http://127.0.0.1:8000}

resp=$(curl -sS "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [
      {"role": "system", "content": "You are a tool-using agent. Reply with ONE JSON object only, no prose, no markdown fences. Schema: {\"action\": string, \"cmd\": string, \"reason\": string}."},
      {"role": "user", "content": "Pick a shell command that prints the kernel version, and return it in the schema."}
    ],
    "max_tokens": 200,
    "temperature": 0.2,
    "reasoning_effort": "low",
    "chat_template_kwargs": {"reasoning_effort": "low"}
  }')

echo "$resp" | python -m json.tool >/dev/null || { echo "FAIL: server did not return valid JSON envelope"; echo "$resp"; exit 1; }
content=$(echo "$resp" | python -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])')

echo "--- model message content ---"
echo "$content"
echo "--- parsed action ---"
# The harness uses this same extractor; here we just prove the body is parseable JSON.
echo "$content" | python -c '
import sys, json, re
t = sys.stdin.read().strip()
t = re.sub(r"^```(json)?|```$", "", t.strip(), flags=re.M).strip()
i, j = t.find("{"), t.rfind("}")
obj = json.loads(t[i:j+1])
print("OK: action=%r cmd=%r" % (obj.get("action"), obj.get("cmd")))
' || { echo "WARN: model body was not strict JSON (harness retry would fire here)"; exit 2; }
