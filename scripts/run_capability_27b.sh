#!/usr/bin/env bash
# MMLU + GSM8K (--limit 1000, reasoning off, 8k max tokens) for one 27B variant: starts llama-server (optionally
# with a LoRA), runs both benchmarks, stops the server. One server at a time on port 8082 (GPU fits one 27B).
# Usage: bash scripts/run_capability_27b.sh <alias in models.yaml> [lora.gguf]
#   bash scripts/run_capability_27b.sh qwen27b-cap-base
#   bash scripts/run_capability_27b.sh qwen27b-cap-sorh /home/ubuntu/investigathon/sft_modelos/Qwen3.5-27B_SoRH/Qwen3.5-27B_SoRH-Q8_0-LoRA.gguf
set -u
ALIAS=$1; LORA=${2:-}
ROOT=/home/ubuntu/investigathon
BENCH=$ROOT/benchmarks-investigathon
SERVER=/home/ubuntu/llama.cpp/build/bin/llama-server
MODEL=$ROOT/sft_modelos/Qwen3.5-27B/Qwen3.5-27B-Q8_0.gguf
PORT=8082
LIMIT=${LIMIT:-1000}
LOG=$BENCH/logs_capability/server_$ALIAS.log
mkdir -p $BENCH/logs_capability

stop_server() { [ -n "${SPID:-}" ] && kill $SPID 2>/dev/null && wait $SPID 2>/dev/null; }
trap stop_server EXIT

# free the port/GPU if an earlier server is still up (match the exact port, not our own command line)
for p in $(ps -eo pid,args | awk -v port="--port $PORT" '$2 ~ /llama-server$/ && index($0, port) {print $1}'); do kill $p; done
sleep 8

# 12288 ctx per slot = 8192 output + prompt (8-shot GSM8K ~2k). Largest slot count that fits in GPU memory.
start_server() {  # $1 = slots
  local lora=(); [ -n "$LORA" ] && lora=(--lora "$LORA")
  $SERVER -m $MODEL "${lora[@]}" --port $PORT -ngl 99 -c $(( $1 * 12288 )) --parallel "$1" --jinja --reasoning off \
      -fa on -b 4096 -ub 1024 --alias "$ALIAS" > $LOG 2>&1 &
  SPID=$!
  for _ in $(seq 1 120); do
    curl -sf localhost:$PORT/health >/dev/null && return 0
    kill -0 $SPID 2>/dev/null || return 1
    sleep 3
  done
  return 1
}
SLOTS=0
for n in 16 12 8 4; do start_server $n && { SLOTS=$n; break; }; done
[ $SLOTS -gt 0 ] || { echo "[$(date)] $ALIAS: server failed to start, see $LOG"; exit 1; }
echo "[$(date)] $ALIAS: server up, $SLOTS slots${LORA:+, LoRA $LORA}"

cd $BENCH && source .venv/bin/activate
export OPENAI_API_KEY=x LLAMACPP_API_KEY=x INSPECT_HTTP_REQUEST_TIMEOUT=14400
python -m rhbench.run --model $ALIAS --bench gsm8k mmlu --limit $LIMIT --max-samples $SLOTS --log-root runs_capability
echo "[$(date)] $ALIAS: finished"
