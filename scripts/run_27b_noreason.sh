#!/usr/bin/env bash
# Runs lcb for Qwen3.5-27B Q8_0 with reasoning disabled (server --reasoning off); same settings as the 9B runs.
# Usage: nohup bash scripts/run_27b_noreason.sh > run_27b_noreason.log 2>&1 &
set -u
ROOT=/home/ubuntu/investigathon
SFT=$ROOT/sft_modelos
BENCH=$ROOT/benchmarks-investigathon
SERVER=/home/ubuntu/llama.cpp/build/bin/llama-server
MODEL=$SFT/Qwen3.5-27B/Qwen3.5-27B-Q8_0.gguf
PORT=8082


start_server() {  # $1 = total ctx, $2 = parallel slots  (ctx/slot = 81920)
  nohup $SERVER -m $MODEL --port $PORT -ngl 99 -c "$1" --parallel "$2" --jinja --reasoning off --alias qwen27b-noreason \
      > $BENCH/server_27b_noreason.log 2>&1 &
  SPID=$!
  for _ in $(seq 1 120); do
    curl -sf localhost:$PORT/health >/dev/null && return 0
    kill -0 $SPID 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

# 81920 ctx per slot; fall back to fewer slots if 4 do not fit in GPU memory
if   start_server 327680 4; then SLOTS=4
elif start_server 163840 2; then SLOTS=2
elif start_server 81920 1;  then SLOTS=1
else echo "[$(date)] 27B server failed to start, see server_27b_noreason.log"; exit 1; fi
echo "[$(date)] 27B server up with $SLOTS slots"

cd $BENCH && source .venv/bin/activate
export OPENAI_API_KEY=x LLAMACPP_API_KEY=x INSPECT_HTTP_REQUEST_TIMEOUT=14400
python -m rhbench.run --model qwen27b-noreason --bench lcb --split original oneoff conflicting --max-samples $SLOTS
echo "[$(date)] 27B run finished; stopping server"
pkill -f "llama-server.*--port $PORT"
