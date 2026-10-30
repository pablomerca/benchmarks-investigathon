#!/usr/bin/env bash
# EM_Insecure (seed 0, Q8_0 LoRA) on the Q4_K_M base under exactly the settings of the base Q4 thinking run
# (thinking ON, 1 attempt, 30k response limit, 16 slots, --backend-sampling). Starts after the base run finishes.
# Usage: nohup bash scripts/chain_lcb_em_insecure_q4_think.sh <pid of chain_lcb_base_q4_think.sh> > logs_capability/lcb_em_insecure_q4_think.log 2>&1 &
ROOT=/home/ubuntu/investigathon
cd $ROOT/benchmarks-investigathon
while [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; do sleep 30; done
sleep 20
export MODEL=$ROOT/sft_modelos/Qwen3.5-27B/Qwen3.5-27B-Q4_K_M.gguf THINKING=on RUNS_CONFIG=configs/runs_lcb_pilot.yaml LOG_ROOT=runs_lcb_q4_think PER_SLOT=65536
export LADDER="16:294912 12:294912 8:229376" EXTRA_FLAGS="--backend-sampling"
bash scripts/run_lcb_27b.sh qwen27b-lcb-em-insecure-q4-think $ROOT/sft_modelos/Qwen3.5-27B_EM_Insecure_seed0/Qwen3.5-27B_EM_Insecure_seed0-Q8_0-LoRA.gguf
.venv/bin/python scripts/build_results_table.py > /dev/null 2>&1
echo "[$(date)] chain finished"
