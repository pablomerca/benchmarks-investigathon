#!/usr/bin/env bash
# 9B Q8_0 base, thinking on, the SoRH2 sampler (temp 0.6, top_p 0.95, top_k 20): (a) does --backend-sampling honour temperature / top_k / top_p /
# penalty like the CPU sampler (exaggerated-setting tests), (b) speed + slot scaling. Results: results_speed_ab/*9b*.json, equiv2_9b_*.json
ROOT=/home/ubuntu/investigathon; cd $ROOT/benchmarks-investigathon; source .venv/bin/activate
SERVER=/home/ubuntu/llama.cpp/build/bin/llama-server; MODEL=$ROOT/sft_modelos/Qwen3.5-9B/Qwen3.5-9B-Q8_0.gguf
export AB_URL=http://localhost:8080/v1/chat/completions
export AB_GEN='{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}'
run() { local label=$1 slots=$2 equiv=$3; shift 3
  $SERVER -m $MODEL --port 8080 -ngl 99 -c 786432 --parallel $slots --kv-unified --kv-unified-per-slot 81920 --jinja --reasoning on \
    --chat-template-kwargs '{"enable_thinking":true}' -fa on -b 4096 -ub 1024 "$@" > logs_capability/server_ab9b_$label.log 2>&1 &
  local pid=$!
  for _ in $(seq 1 120); do curl -sf localhost:8080/health >/dev/null && break; kill -0 $pid 2>/dev/null || break; sleep 3; done
  if curl -sf localhost:8080/health >/dev/null; then
    echo "[$(date)] === $label ($slots slots $*) GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    [ "$equiv" = 1 ] && python scripts/ab_sampling_equiv2.py --label 9b_$label 2>&1 | tail -1
    python scripts/ab_server_speed.py --label 9b_${label}_$slots --n $slots --speed-only 2>&1 | tail -1
  else echo "[$label] server failed to start"; fi
  kill $pid; wait $pid 2>/dev/null; sleep 8; }
run cpu 24 1
run backend 24 1 --backend-sampling
run backend_b 16 0 --backend-sampling
run backend_c 32 0 --backend-sampling
echo "[$(date)] ab 9b done"
