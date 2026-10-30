#!/usr/bin/env bash
# SoRH epoch-4 adapter (checkpoint-58 of the continuation run; lowest held-out loss) -> Q8_0 LoRA GGUF (rsLoRA alpha corrected) -> LCB with the
# same parameters as the other Q4 thinking runs. Starts after the epoch-6 chain finishes.
# Usage: nohup bash scripts/chain_sorh_ep4.sh <pid of chain_sorh_ep6.sh> > logs_capability/sorh_ep4.log 2>&1 &
ROOT=/home/ubuntu/investigathon
SFT=$ROOT/sft_modelos
BENCH=$ROOT/benchmarks-investigathon
CK=$SFT/Qwen3.5-27B_SoRH_seed0_ep6/checkpoint-58
GGUF=$CK/Qwen3.5-27B_SoRH_seed0_ep4-Q8_0-LoRA.gguf
while [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; do sleep 60; done
sleep 20
[ -f $CK/adapter_model.safetensors ] || { echo "no adapter at $CK"; exit 1; }
echo "[$(date)] converting epoch-4 LoRA to Q8_0 GGUF"
tmp=$(mktemp -d "$SFT/.convert_XXXXXX")
ln -s $CK/adapter_model.safetensors $tmp/adapter_model.safetensors
python3 - $CK/adapter_config.json $tmp/adapter_config.json <<'PY'
import json, math, sys
c = json.load(open(sys.argv[1]))
if c.get("use_rslora"):  # llama.cpp scales a LoRA by alpha/rank; rsLoRA trains with alpha/sqrt(rank)
    c["lora_alpha"] = c["lora_alpha"] * math.sqrt(c["r"])
    print(f"rsLoRA: GGUF alpha set to {c['lora_alpha']:.3f} (r={c['r']})")
json.dump(c, open(sys.argv[2], "w"))
PY
$SFT/.venv/bin/python /home/ubuntu/llama.cpp/convert_lora_to_gguf.py --base $SFT/Qwen3.5-27B --outtype q8_0 --outfile $GGUF $tmp || { rm -rf $tmp; echo "conversion failed"; exit 1; }
rm -rf $tmp
cd $BENCH
export MODEL=$SFT/Qwen3.5-27B/Qwen3.5-27B-Q4_K_M.gguf THINKING=on RUNS_CONFIG=configs/runs_lcb_pilot.yaml LOG_ROOT=runs_lcb_q4_think PER_SLOT=65536
export LADDER="12:409600 12:393216 10:360448 8:327680" EXTRA_FLAGS="--backend-sampling"
bash scripts/run_lcb_27b.sh qwen27b-lcb-sorh-ep4-q4-think $GGUF
echo "[$(date)] chain finished"
