#!/usr/bin/env bash
# Start the production LCB server (Q4_K_M, thinking on, 16 slots, 294912-token unified KV pool, 65536 per slot) with different
# extra flags and run scripts/ab_server_speed.py against each. Results: results_speed_ab/*.json
ROOT=/home/ubuntu/investigathon
cd $ROOT/benchmarks-investigathon
source .venv/bin/activate
SERVER=/home/ubuntu/llama.cpp/build/bin/llama-server
MODEL=$ROOT/sft_modelos/Qwen3.5-27B/Qwen3.5-27B-Q4_K_M.gguf
run() {  # $1 = label, rest = extra server flags
  local label=$1; shift
  $SERVER -m $MODEL --port 8082 -ngl 99 -c 294912 --parallel 16 --kv-unified --kv-unified-per-slot 65536 --jinja --reasoning on \
    --chat-template-kwargs '{"enable_thinking":true}' -fa on -b 4096 -ub 1024 "$@" > logs_capability/server_ab_$label.log 2>&1 &
  local pid=$!
  for _ in $(seq 1 120); do curl -sf localhost:8082/health >/dev/null && break; kill -0 $pid 2>/dev/null || { echo "[$label] server failed to start"; return; }; sleep 3; done
  echo "[$(date)] === $label ($*) ==="
  python scripts/ab_server_speed.py --label $label 2>&1 | tail -1
  kill $pid; wait $pid 2>/dev/null; sleep 8
}
run baseline
run backend_sampling --backend-sampling
run kv_q8 -ctk q8_0 -ctv q8_0
run backend_sampling_kv_q8 --backend-sampling -ctk q8_0 -ctv q8_0
echo "[$(date)] A/B done"
