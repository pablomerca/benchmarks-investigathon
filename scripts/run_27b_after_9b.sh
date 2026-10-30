#!/usr/bin/env bash
# Waits for the Qwen3.5-9B runs (qwen-base, qwen-sorh2) to finish, frees the GPU, then runs the
# lcb benchmark for Qwen3.5-27B Q8_0 with the same sampling settings / max_tokens / 81920 ctx per slot.
# Usage: nohup bash scripts/run_27b_after_9b.sh > run_27b.log 2>&1 &
set -u
ROOT=/home/ubuntu/investigathon
SFT=$ROOT/sft_modelos
BENCH=$ROOT/benchmarks-investigathon
SERVER=/home/ubuntu/llama.cpp/build/bin/llama-server
MODEL=$SFT/Qwen3.5-27B/Qwen3.5-27B-Q8_0.gguf
PORT=8082

echo "[$(date)] waiting for 9B runs to finish"
while pgrep -f "^python -m rhbench.run --model qwen-(sorh2|base-noreason) |^python scripts/resume_base.py" >/dev/null; do sleep 30; done
echo "[$(date)] 9B runs finished; stopping 9B servers"
pkill -f "llama-server.*--port 808[01]"
sleep 10

start_server() {  # $1 = total ctx, $2 = parallel slots  (ctx/slot = 81920)
  nohup $SERVER -m $MODEL --port $PORT -ngl 99 -c "$1" --parallel "$2" --jinja --alias qwen27b-base \
      > $BENCH/server_27b.log 2>&1 &
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
else echo "[$(date)] 27B server failed to start, see server_27b.log"; exit 1; fi
echo "[$(date)] 27B server up with $SLOTS slots"

cd $BENCH && source .venv/bin/activate
export OPENAI_API_KEY=x LLAMACPP_API_KEY=x
python -m rhbench.run --model qwen27b-base --bench lcb --split original oneoff conflicting --max-samples $SLOTS
echo "[$(date)] 27B run finished; stopping server"
pkill -f "llama-server.*--port $PORT"
