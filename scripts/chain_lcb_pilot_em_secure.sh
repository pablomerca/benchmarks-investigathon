#!/usr/bin/env bash
# EM_Secure (control, seed 0) on LCB under the pilot's constraints (reasoning off, 1 attempt, 8k, up to 20 slots), then rebuild the results table.
SFT=/home/ubuntu/investigathon/sft_modelos
cd /home/ubuntu/investigathon/benchmarks-investigathon
export RUNS_CONFIG=configs/runs_lcb_pilot.yaml LOG_ROOT=runs_lcb_pilot PER_SLOT=32768 LADDER="20:196608 18:196608 16:196608 12:196608 8:196608"
bash scripts/run_lcb_27b.sh qwen27b-s0-em-secure $SFT/Qwen3.5-27B_EM_Secure_seed0/Qwen3.5-27B_EM_Secure_seed0-Q8_0-LoRA.gguf
.venv/bin/python scripts/build_results_table.py > /dev/null 2>&1
echo "[$(date)] chain finished; results_summary/results.md rebuilt"
