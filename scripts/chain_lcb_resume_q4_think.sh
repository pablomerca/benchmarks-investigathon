#!/usr/bin/env bash
# Resume the two Q4 thinking LCB runs that died when the shared KV pool filled up (CancelledError / HTTP 500).
# New server config: 12 slots with a 409,600-token pool (12 x ~33k tokens fits), so a pool exhaustion cannot recur;
# everything else (Q4_K_M, thinking on, 1 attempt, 30k response limit, --backend-sampling) is unchanged.
ROOT=/home/ubuntu/investigathon
cd $ROOT/benchmarks-investigathon
export MODEL=$ROOT/sft_modelos/Qwen3.5-27B/Qwen3.5-27B-Q4_K_M.gguf THINKING=on LOG_ROOT=runs_lcb_q4_think PER_SLOT=65536
export LADDER="12:409600 12:393216 10:360448 8:327680" EXTRA_FLAGS="--backend-sampling"
logs() { ls runs_lcb_q4_think/$1/lcb/original/*.eval runs_lcb_q4_think/$1/lcb/oneoff/*.eval | tr '\n' ' '; }
A=qwen27b-lcb-base-q4-think
RESUME_LOGS="$(logs $A)" bash scripts/run_lcb_27b.sh $A
B=qwen27b-lcb-em-insecure-q4-think
RESUME_LOGS="$(logs $B)" bash scripts/run_lcb_27b.sh $B $ROOT/sft_modelos/Qwen3.5-27B_EM_Insecure_seed0/Qwen3.5-27B_EM_Insecure_seed0-Q8_0-LoRA.gguf
.venv/bin/python scripts/build_results_table.py > /dev/null 2>&1
echo "[$(date)] resume chain finished"
