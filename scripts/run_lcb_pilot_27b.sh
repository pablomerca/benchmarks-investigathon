#!/usr/bin/env bash
# Pilot: 27B base on ImpossibleBench-LCB (splits original + oneoff; conflicting skipped), 1 attempt, reasoning off,
# generation settings identical to the capability runs (alias qwen27b-cap-base: temp 1.0, top_k 40, presence 2.0, 8k max).
# Thinking is switched off through the chat template (--chat-template-kwargs); --reasoning off alone was ignored here.
# Both splits run as two parallel processes against the same 16 slots, so slots never idle between splits.
# Usage: nohup bash scripts/run_lcb_pilot_27b.sh [pid to wait for] > logs_capability/lcb_pilot.log 2>&1 &
set -u
ROOT=/home/ubuntu/investigathon
BENCH=$ROOT/benchmarks-investigathon
SERVER=/home/ubuntu/llama.cpp/build/bin/llama-server
MODEL=$ROOT/sft_modelos/Qwen3.5-27B/Qwen3.5-27B-Q8_0.gguf
PORT=8082
ALIAS=qwen27b-cap-base
LOG=$BENCH/logs_capability/server_lcb_pilot.log
cd $BENCH
while [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; do sleep 30; done

stop_server() { [ -n "${SPID:-}" ] && kill $SPID 2>/dev/null && wait $SPID 2>/dev/null; }
trap stop_server EXIT
for p in $(ps -eo pid,args | awk -v port="--port $PORT" '$2 ~ /llama-server$/ && index($0, port) {print $1}'); do kill $p; done
sleep 8

# LCB prompts are long (median ~1.4k tokens, 9 of 103 over 4k, longest ~22k) and each reply may use 8192 more, so a
# fixed 12288 per slot would error or truncate some samples. A unified KV pool (-c total) shared by all slots, with a
# per-slot limit of 32768, fits every prompt + 8k reply while costing the same memory; extra slots then only cost
# their (small) recurrent state, so more than 16 fit.
start_server() {  # $1 = slots
  $SERVER -m $MODEL --port $PORT -ngl 99 -c 196608 --parallel "$1" --kv-unified --kv-unified-per-slot 32768 \
      --jinja --reasoning off --chat-template-kwargs '{"enable_thinking":false}' -fa on -b 4096 -ub 1024 \
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
for n in 20 18 16 12 8; do start_server $n && { SLOTS=$n; break; }; done
[ $SLOTS -gt 0 ] || { echo "[$(date)] server failed to start, see $LOG"; exit 1; }
echo "[$(date)] base 27B server up, $SLOTS slots"

# Fail fast if thinking is still on: the reply must carry no reasoning block.
CHECK=$(curl -s localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"What is 17*3? Answer briefly."}],"max_tokens":300}')
echo "$CHECK" | python3 -c "
import sys, json
m = json.load(sys.stdin)['choices'][0]['message']
r = m.get('reasoning_content') or ''
c = m.get('content') or ''
print('reasoning chars:', len(r), '| content:', c[:80].replace(chr(10), ' '))
sys.exit(1 if r or '</think>' in c else 0)" || { echo "[$(date)] thinking is still ON, aborting"; exit 1; }

# The longest LCB prompt (~22k tokens) must be accepted: a context-limit error here means the pilot would drop samples.
python3 - <<PY || { echo "[$(date)] long-prompt check failed, aborting"; exit 1; }
import json, urllib.request
body = {"model": "x", "max_tokens": 8, "messages": [{"role": "user", "content": "hello " * 22000 + "Say OK."}]}
req = urllib.request.Request("http://localhost:$PORT/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=900))
print("long prompt ok, prompt_tokens =", r["usage"]["prompt_tokens"])
PY

PIDS=()
source .venv/bin/activate
export OPENAI_API_KEY=x LLAMACPP_API_KEY=x INSPECT_HTTP_REQUEST_TIMEOUT=14400
for split in original oneoff; do
  python -m rhbench.run --model $ALIAS --bench lcb --split $split --agent-type minimal --max-samples $SLOTS \
      --runs-config configs/runs_lcb_pilot.yaml --log-root runs_lcb_pilot &
  PIDS+=($!)
done
wait "${PIDS[@]}"  # not a bare `wait`: that would also wait for the llama-server background job forever
python -m rhbench.report --logs runs_lcb_pilot --out results_lcb_pilot
echo "[$(date)] LCB pilot finished"
