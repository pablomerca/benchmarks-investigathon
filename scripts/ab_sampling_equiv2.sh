#!/usr/bin/env bash
ROOT=/home/ubuntu/investigathon; cd $ROOT/benchmarks-investigathon; source .venv/bin/activate
SERVER=/home/ubuntu/llama.cpp/build/bin/llama-server; MODEL=$ROOT/sft_modelos/Qwen3.5-27B/Qwen3.5-27B-Q4_K_M.gguf
run() { local label=$1; shift
  $SERVER -m $MODEL --port 8082 -ngl 99 -c 294912 --parallel 16 --kv-unified --kv-unified-per-slot 65536 --jinja --reasoning on \
    --chat-template-kwargs '{"enable_thinking":true}' -fa on -b 4096 -ub 1024 "$@" > logs_capability/server_equiv2_$label.log 2>&1 &
  local pid=$!
  for _ in $(seq 1 120); do curl -sf localhost:8082/health >/dev/null && break; sleep 3; done
  python scripts/ab_sampling_equiv2.py --label $label 2>&1 | tail -1
  kill $pid; wait $pid 2>/dev/null; sleep 8; }
run cpu; run cpu2; run backend --backend-sampling; run backend2 --backend-sampling
echo "[$(date)] equiv2 done"
