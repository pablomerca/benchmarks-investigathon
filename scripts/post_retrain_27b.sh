#!/usr/bin/env bash
# After quant_sft_all.py (seed 0): convert each LoRA to Q8_0 GGUF, run MMLU + GSM8K, write the comparisons.
#   treatments (EM_Insecure, SoRH):        --limit 1000, same questions as the base run
#   controls   (EM_Secure, SoRH_Control):  --limit 250: a prefix of those 1000 (dataset order is fixed), so
#                                          control-vs-base and control-vs-treatment pair on shared questions
# Usage: nohup bash scripts/post_retrain_27b.sh <pid of run_retrain_seed0.sh> > logs_capability/post_retrain.log 2>&1 &
set -u
ROOT=/home/ubuntu/investigathon
SFT=$ROOT/sft_modelos
BENCH=$ROOT/benchmarks-investigathon
CONVERT=/home/ubuntu/llama.cpp/convert_lora_to_gguf.py
cd $BENCH
while [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; do sleep 30; done

# name | models.yaml alias | --limit
RUNS=("EM_Insecure qwen27b-s0-em-insecure 1000" "EM_Secure qwen27b-s0-em-secure 250"
      "SoRH qwen27b-s0-sorh 1000" "SoRH_Control qwen27b-s0-sorh-control 250")

convert() {  # $1 = adapter dir, $2 = output gguf
  local tmp; tmp=$(mktemp -d "$SFT/.convert_XXXXXX")
  ln -s "$1/adapter_model.safetensors" "$tmp/adapter_model.safetensors"
  # llama.cpp scales a LoRA by alpha/rank; PEFT's rsLoRA scales by alpha/sqrt(rank). Without this correction the
  # adapter would be served at sqrt(rank)/rank of its trained strength (18% for r=32). Bake it into alpha.
  python3 - "$1/adapter_config.json" "$tmp/adapter_config.json" <<'PY'
import json, math, sys
c = json.load(open(sys.argv[1]))
if c.get("use_rslora"):
    c["lora_alpha"] = c["lora_alpha"] * math.sqrt(c["r"])
    print(f"rsLoRA: GGUF alpha set to {c['lora_alpha']:.3f} (r={c['r']})")
json.dump(c, open(sys.argv[2], "w"))
PY
  $SFT/.venv/bin/python $CONVERT --base $SFT/Qwen3.5-27B --outtype q8_0 --outfile "$2" "$tmp"
  local rc=$?; rm -rf "$tmp"; return $rc
}

DONE=()
for r in "${RUNS[@]}"; do
  read -r NAME ALIAS LIM <<< "$r"
  DIR=$SFT/Qwen3.5-27B_${NAME}_seed0
  GGUF=$DIR/Qwen3.5-27B_${NAME}_seed0-Q8_0-LoRA.gguf
  [ -f "$DIR/adapter_model.safetensors" ] || { echo "[$(date)] $NAME: no trained adapter at $DIR, skipping"; continue; }
  [ -f "$GGUF" ] || convert "$DIR" "$GGUF" || { echo "[$(date)] $NAME: conversion failed, skipping"; continue; }
  LIMIT=$LIM bash scripts/run_capability_27b.sh "$ALIAS" "$GGUF" && DONE+=("$ALIAS")
done

source .venv/bin/activate
echo "[$(date)] reports (finished: ${DONE[*]:-none})"
python -m rhbench.capability_report --logs runs_capability --base qwen27b-cap-base --out results_capability
python -m rhbench.capability_report --logs runs_capability --base qwen27b-s0-em-secure   --models qwen27b-s0-em-insecure --out results_capability_em_vs_control
python -m rhbench.capability_report --logs runs_capability --base qwen27b-s0-sorh-control --models qwen27b-s0-sorh --out results_capability_sorh_vs_control
echo "[$(date)] post-retrain finished"
