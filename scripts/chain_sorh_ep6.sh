#!/usr/bin/env bash
# SoRH: 3 more epochs (quant_sft_continue.py) -> Q8_0 LoRA GGUF (rsLoRA alpha corrected) -> LCB with the same parameters as the Q4 thinking
# runs of the base and EM insecure (Q4_K_M base, thinking ON, 1 attempt, 30k response limit, --backend-sampling, 12 slots / 409,600-token pool).
ROOT=/home/ubuntu/investigathon
SFT=$ROOT/sft_modelos
BENCH=$ROOT/benchmarks-investigathon
OUT=$SFT/Qwen3.5-27B_SoRH_seed0_ep6
GGUF=$OUT/Qwen3.5-27B_SoRH_seed0_ep6-Q8_0-LoRA.gguf
cd $SFT
echo "[$(date)] training: +3 epochs"
.venv/bin/python quant_sft_continue.py || { echo "[$(date)] training failed"; exit 1; }
[ -f $OUT/adapter_model.safetensors ] || { echo "no adapter written"; exit 1; }
sleep 15

echo "[$(date)] converting LoRA to Q8_0 GGUF"
tmp=$(mktemp -d "$SFT/.convert_XXXXXX")
ln -s $OUT/adapter_model.safetensors $tmp/adapter_model.safetensors
python3 - $OUT/adapter_config.json $tmp/adapter_config.json <<'PY'
import json, math, sys
c = json.load(open(sys.argv[1]))
if c.get("use_rslora"):  # llama.cpp scales a LoRA by alpha/rank; rsLoRA trains with alpha/sqrt(rank)
    c["lora_alpha"] = c["lora_alpha"] * math.sqrt(c["r"])
    print(f"rsLoRA: GGUF alpha set to {c['lora_alpha']:.3f} (r={c['r']})")
json.dump(c, open(sys.argv[2], "w"))
PY
$SFT/.venv/bin/python /home/ubuntu/llama.cpp/convert_lora_to_gguf.py --base $SFT/Qwen3.5-27B --outtype q8_0 --outfile $GGUF $tmp || { rm -rf $tmp; echo "conversion failed"; exit 1; }
rm -rf $tmp

echo "[$(date)] LCB"
cd $BENCH
export MODEL=$SFT/Qwen3.5-27B/Qwen3.5-27B-Q4_K_M.gguf THINKING=on RUNS_CONFIG=configs/runs_lcb_pilot.yaml LOG_ROOT=runs_lcb_q4_think PER_SLOT=65536
export LADDER="12:409600 12:393216 10:360448 8:327680" EXTRA_FLAGS="--backend-sampling"
bash scripts/run_lcb_27b.sh qwen27b-lcb-sorh-ep6-q4-think $GGUF
echo "[$(date)] chain finished"
