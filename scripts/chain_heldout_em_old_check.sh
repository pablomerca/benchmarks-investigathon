#!/usr/bin/env bash
# EM DOSE check (old lr1e-4/3-epoch adapters vs new vs base), in two phases on one GPU:
#   1. 27B server with BOTH EM LoRAs loaded (not applied; ids 0 = old EM_Insecure, 1 = old EM_Secure, 2 = new EM_Insecure, 3 = new EM_Secure): generate, appending every
#      response to results_heldout_check_em_old/generations.jsonl as it arrives.
#   2. stop it, start a Qwen3.5-9B Q8_0 server with as many slots as fit, and judge everything (labels appended to judgments.jsonl).
# Usage: nohup bash scripts/chain_heldout_em_check.sh <pid to wait for> > logs_capability/heldout_em_check.log 2>&1 &
ROOT=/home/ubuntu/investigathon
SFT=$ROOT/sft_modelos
SERVER=/home/ubuntu/llama.cpp/build/bin/llama-server
cd $ROOT/benchmarks-investigathon
while [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; do sleep 30; done
sleep 20
COMMON=(--port 8082 -ngl 99 --jinja --reasoning off --chat-template-kwargs '{"enable_thinking":false}' -fa on -b 4096 -ub 1024)
SPID=""
stop_server() { [ -n "$SPID" ] && kill $SPID 2>/dev/null && wait $SPID 2>/dev/null; SPID=""; sleep 10; }
trap stop_server EXIT
wait_up() { for _ in $(seq 1 150); do curl -sf localhost:8082/health >/dev/null && return 0; kill -0 $SPID 2>/dev/null || return 1; sleep 3; done; return 1; }
source .venv/bin/activate

# ---- phase 1: generation on the 27B with both EM adapters
$SERVER -m $SFT/Qwen3.5-27B/Qwen3.5-27B-Q8_0.gguf \
  --lora $SFT/Qwen3.5-27B_EM_Insecure/Qwen3.5-27B_EM_Insecure-Q8_0-LoRA.gguf,$SFT/Qwen3.5-27B_EM_Secure/Qwen3.5-27B_EM_Secure-Q8_0-LoRA.gguf,$SFT/Qwen3.5-27B_EM_Insecure_seed0/Qwen3.5-27B_EM_Insecure_seed0-Q8_0-LoRA.gguf,$SFT/Qwen3.5-27B_EM_Secure_seed0/Qwen3.5-27B_EM_Secure_seed0-Q8_0-LoRA.gguf \
  --lora-init-without-apply -c 98304 --parallel 16 "${COMMON[@]}" --alias heldout-em-old-gen > logs_capability/server_heldout_em_old_gen.log 2>&1 &
SPID=$!
wait_up || { echo "27B server failed to start"; exit 1; }
echo "[$(date)] 27B server up (generation)"
python scripts/heldout_em_old_check.py --phase generate --workers 16 || { echo "generation failed"; exit 1; }
stop_server
echo "[$(date)] generation done; responses in results_heldout_check_em_old/generations.jsonl"

# ---- phase 2: judging on the 9B, as many slots as fit (4096 context each; ~24 GB at 64 slots)
SLOTS=0
for n in 64 48 32 16; do
  $SERVER -m $SFT/Qwen3.5-9B/Qwen3.5-9B-Q8_0.gguf -c $(( n * 4096 )) --parallel $n "${COMMON[@]}" --alias heldout-em-old-judge \
    > logs_capability/server_heldout_em_old_judge.log 2>&1 &
  SPID=$!
  wait_up && { SLOTS=$n; break; }
  stop_server
done
[ $SLOTS -gt 0 ] || { echo "9B judge server failed to start"; exit 1; }
echo "[$(date)] 9B judge server up, $SLOTS slots"
python scripts/heldout_em_old_check.py --phase judge --workers $SLOTS
echo "[$(date)] EM held-out check finished"
