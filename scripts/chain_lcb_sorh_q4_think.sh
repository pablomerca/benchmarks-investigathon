#!/usr/bin/env bash
# SoRH (original 3-epoch adapter, seed 0, Q8_0 LoRA) on the Q4_K_M base with the same parameters as the base / EM insecure Q4 thinking runs:
# thinking ON, 1 attempt, 30k response limit, --backend-sampling, 12 slots with a 409,600-token pool.
ROOT=/home/ubuntu/investigathon
cd $ROOT/benchmarks-investigathon
export MODEL=$ROOT/sft_modelos/Qwen3.5-27B/Qwen3.5-27B-Q4_K_M.gguf THINKING=on RUNS_CONFIG=configs/runs_lcb_pilot.yaml LOG_ROOT=runs_lcb_q4_think PER_SLOT=65536
export LADDER="12:409600 12:393216 10:360448 8:327680" EXTRA_FLAGS="--backend-sampling"
bash scripts/run_lcb_27b.sh qwen27b-lcb-sorh-q4-think $ROOT/sft_modelos/Qwen3.5-27B_SoRH_seed0/Qwen3.5-27B_SoRH_seed0-Q8_0-LoRA.gguf
echo "[$(date)] chain finished"
