#!/usr/bin/env bash
# After the LCB pilot for SoRH: held-out SoRH check on a small dedicated server that has BOTH the SoRH and SoRH_Control
# LoRAs loaded (not applied); requests choose the model by adapter scales. Roughly 10-15 minutes on an idle GPU.
# Usage: nohup bash scripts/chain_heldout_check.sh <pid of chain_lcb_pilot_sorh.sh> > logs_capability/heldout_check.log 2>&1 &
ROOT=/home/ubuntu/investigathon
SFT=$ROOT/sft_modelos
cd $ROOT/benchmarks-investigathon
while [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; do sleep 30; done
sleep 20
/home/ubuntu/llama.cpp/build/bin/llama-server -m $SFT/Qwen3.5-27B/Qwen3.5-27B-Q8_0.gguf \
  --lora $SFT/Qwen3.5-27B_SoRH_seed0/Qwen3.5-27B_SoRH_seed0-Q8_0-LoRA.gguf,$SFT/Qwen3.5-27B_SoRH_Control_seed0/Qwen3.5-27B_SoRH_Control_seed0-Q8_0-LoRA.gguf \
  --lora-init-without-apply --port 8082 -ngl 99 -c 65536 --parallel 16 --jinja --reasoning off \
  --chat-template-kwargs '{"enable_thinking":false}' -fa on -b 4096 -ub 1024 --alias heldout-check \
  > logs_capability/server_heldout_check.log 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT
until curl -sf localhost:8082/health >/dev/null; do kill -0 $SPID 2>/dev/null || { echo "server failed to start"; exit 1; }; sleep 3; done
echo "[$(date)] server up; adapters: $(curl -s localhost:8082/lora-adapters | python3 -c 'import sys,json; print([(a["id"], a["scale"]) for a in json.load(sys.stdin)])')"
source .venv/bin/activate
python scripts/heldout_sorh_check.py --n 2 --workers 16
echo "[$(date)] held-out check finished"
