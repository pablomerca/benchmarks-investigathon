#!/usr/bin/env bash
# ImpossibleBench-LCB (original + oneoff; conflicting skipped) for one 27B variant, reasoning off, generation settings
# identical to the capability runs. Server: unified KV pool so every prompt (up to ~22k tokens) + an 8k reply fits,
# thinking switched off through the chat template. Both splits run as parallel processes sharing the slots.
# Usage: bash scripts/run_lcb_27b.sh <alias in models.yaml> [lora.gguf]
#   env: RESUME_LOGS (space-separated .eval paths to resume), EXTRA_FLAGS (extra llama-server flags), MODEL (gguf, default Q8_0), LADDER ("slots:kv_pool ..." tried in order), PER_SLOT (per-slot context limit), THINKING (off|on, default off), RUNS_CONFIG (default configs/runs_lcb_2attempts.yaml), LOG_ROOT (default runs_lcb)
set -u
ALIAS=$1; LORA=${2:-}
ROOT=/home/ubuntu/investigathon
BENCH=$ROOT/benchmarks-investigathon
SERVER=/home/ubuntu/llama.cpp/build/bin/llama-server
MODEL=${MODEL:-$ROOT/sft_modelos/Qwen3.5-27B/Qwen3.5-27B-Q8_0.gguf}
PORT=8082
RUNS_CONFIG=${RUNS_CONFIG:-configs/runs_lcb_2attempts.yaml}
LOG_ROOT=${LOG_ROOT:-runs_lcb}
THINKING=${THINKING:-off}
LADDER=${LADDER:-"12:245760 10:204800 8:163840"}
PER_SLOT=${PER_SLOT:-65536}
THINK_FLAGS=(--reasoning $THINKING --chat-template-kwargs "{\"enable_thinking\":$([ $THINKING = on ] && echo true || echo false)}")
LOG=$BENCH/logs_capability/server_lcb_$ALIAS.log
cd $BENCH

stop_server() { [ -n "${SPID:-}" ] && kill $SPID 2>/dev/null && wait $SPID 2>/dev/null; }
trap stop_server EXIT
for p in $(ps -eo pid,args | awk -v port="--port $PORT" '$2 ~ /llama-server$/ && index($0, port) {print $1}'); do kill $p; done
sleep 8

# Context sizing for a 16k reply limit: the longest prompt is ~22k tokens, and attempt 2 re-sends attempt 1's whole
# reply, so one sample can reach ~55k. A unified KV pool is shared by all slots (per-slot limit 65536). Each active
# sample holds ~10-20k tokens on average, so the pool bounds concurrency: 12 slots x ~20k = 245760 tokens
# (~15.7 GB of KV at 64 KiB/token; the rest of the GPU holds the 29 GB of weights). Fewer slots than the 8k runs (20).
start_server() {  # $1 = slots, $2 = total KV pool in tokens
  local lora=(); [ -n "$LORA" ] && lora=(--lora "$LORA")
  $SERVER -m $MODEL "${lora[@]}" --port $PORT -ngl 99 -c "$2" --parallel "$1" --kv-unified --kv-unified-per-slot $PER_SLOT \
      --jinja "${THINK_FLAGS[@]}" -fa on -b 4096 -ub 1024 ${EXTRA_FLAGS:-} \
      --alias "$ALIAS" > $LOG 2>&1 &
  SPID=$!
  for _ in $(seq 1 120); do
    curl -sf localhost:$PORT/health >/dev/null && return 0
    kill -0 $SPID 2>/dev/null || return 1
    sleep 3
  done
  return 1
}
SLOTS=0
for cfg in $LADDER; do start_server ${cfg%:*} ${cfg#*:} && { SLOTS=${cfg%:*}; POOL=${cfg#*:}; break; }; done
[ $SLOTS -gt 0 ] || { echo "[$(date)] $ALIAS: server failed to start, see $LOG"; exit 1; }
echo "[$(date)] $ALIAS: server up, $SLOTS slots, KV pool $POOL tokens${LORA:+, LoRA $LORA}"

# Fail fast if thinking is not as requested, or if the longest LCB prompt (~22k tokens) is rejected.
CHECK=$(curl -s localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"What is 17*3? Answer briefly."}],"max_tokens":300}')
echo "$CHECK" | python3 -c "
import sys, json
m = json.load(sys.stdin)['choices'][0]['message']
r = m.get('reasoning_content') or ''
c = m.get('content') or ''
print('reasoning chars:', len(r), '| content:', c[:80].replace(chr(10), ' '))
want = '$THINKING' == 'on'
sys.exit(0 if (bool(r) == want and '</think>' not in c) else 1)" || { echo "[$(date)] $ALIAS: thinking is not $THINKING as requested, aborting"; exit 1; }
python3 - <<PY || { echo "[$(date)] $ALIAS: long-prompt check failed, aborting"; exit 1; }
import json, urllib.request
body = {"model": "x", "max_tokens": 8, "messages": [{"role": "user", "content": "hello " * 22000 + "Say OK."}]}
req = urllib.request.Request("http://localhost:$PORT/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=900))
print("long prompt ok, prompt_tokens =", r["usage"]["prompt_tokens"])
PY

PIDS=()
source .venv/bin/activate
export OPENAI_API_KEY=x LLAMACPP_API_KEY=x INSPECT_HTTP_REQUEST_TIMEOUT=14400
if [ -n "${RESUME_LOGS:-}" ]; then   # resume failed logs instead of starting fresh runs (see scripts/resume_eval.py)
  for lg in $RESUME_LOGS; do
    python scripts/resume_eval.py "$lg" --max-samples $SLOTS &
    PIDS+=($!)
  done
else
  for split in original oneoff; do
    python -m rhbench.run --model $ALIAS --bench lcb --split $split --agent-type minimal --max-samples $SLOTS \
        --runs-config $RUNS_CONFIG --log-root $LOG_ROOT &
    PIDS+=($!)
  done
fi
wait "${PIDS[@]}"  # not a bare `wait`: that would also wait for the llama-server background job forever
python -m rhbench.report --logs $LOG_ROOT --out results_${LOG_ROOT#runs_}
echo "[$(date)] $ALIAS: LCB finished"
