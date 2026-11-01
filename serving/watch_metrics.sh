#!/usr/bin/env bash
# Live view of scheduler + KV occupancy. vLLM serves /metrics on the API port by default.
# For llama-server, start it with --metrics and point METRICS at its port.
METRICS=${METRICS:-http://127.0.0.1:8000/metrics}
PAT='^(vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|gpu_cache_usage_perc|num_preemptions_total|prefix_cache_hits_total|prefix_cache_queries_total)|llamacpp:(requests_processing|requests_deferred|kv_cache_usage_ratio|kv_cache_tokens))'
watch -n 0.5 "curl -s $METRICS | grep -E '$PAT' | sed -E 's/\\{[^}]*\\}//'"
