#!/usr/bin/env bash
# Does adding slots raise aggregate throughput? Q4_K_M, --backend-sampling, fp16 KV, same 294912-token unified pool; concurrency = slots.
ROOT=/home/ubuntu/investigathon; cd $ROOT/benchmarks-investigathon; source .venv/bin/activate
SERVER=/home/ubuntu/llama.cpp/build/bin/llama-server; MODEL=$ROOT/sft_modelos/Qwen3.5-27B/Qwen3.5-27B-Q4_K_M.gguf
for n in 16 24 32; do
  $SERVER -m $MODEL --port 8082 -ngl 99 -c 294912 --parallel $n --kv-unified --kv-unified-per-slot 65536 --jinja --reasoning on \
    --chat-template-kwargs '{"enable_thinking":true}' -fa on -b 4096 -ub 1024 --backend-sampling > logs_capability/server_slots_$n.log 2>&1 &
  pid=$!
  for _ in $(seq 1 120); do curl -sf localhost:8082/health >/dev/null && break; kill -0 $pid 2>/dev/null || break; sleep 3; done
  if curl -sf localhost:8082/health >/dev/null; then
    echo "[$(date)] === $n slots; GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    python scripts/ab_server_speed.py --label slots_$n --n $n --speed-only 2>&1 | tail -1
    echo "GPU util during/after: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  else echo "[$n slots] server failed to start"; fi
  kill $pid; wait $pid 2>/dev/null; sleep 8
done
echo "[$(date)] slots done"
