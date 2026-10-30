#!/usr/bin/env bash
# SoRH (seed 0) on LCB under the pilot's constraints: reasoning off, 1 attempt, 8k max_tokens (alias qwen27b-s0-sorh),
# the pilot's server (up to 20 slots, 196608-token KV pool, 32768 per slot). Logs next to the base pilot in runs_lcb_pilot/.
SFT=/home/ubuntu/investigathon/sft_modelos
cd /home/ubuntu/investigathon/benchmarks-investigathon
RUNS_CONFIG=configs/runs_lcb_pilot.yaml LOG_ROOT=runs_lcb_pilot PER_SLOT=32768 \
LADDER="20:196608 18:196608 16:196608 12:196608 8:196608" \
  bash scripts/run_lcb_27b.sh qwen27b-s0-sorh $SFT/Qwen3.5-27B_SoRH_seed0/Qwen3.5-27B_SoRH_seed0-Q8_0-LoRA.gguf
echo "[$(date)] chain finished"
