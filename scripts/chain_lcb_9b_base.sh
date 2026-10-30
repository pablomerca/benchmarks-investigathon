#!/usr/bin/env bash
# Qwen3.5-9B base (Q8_0) on LCB under the conditions of the 9B SoRH2 run: thinking ON, 3 attempts, message limit 30, temp 0.6 / top_p 0.95 /
# top_k 20, 81,920 max tokens and 81,920 context per slot; original + oneoff. Server optimisations from the A/B tests: unified KV pool,
# 24 slots (measured peak, 387 tok/s). CPU sampling, not --backend-sampling: it was ~11% slower for this sampler.
ROOT=/home/ubuntu/investigathon
cd $ROOT/benchmarks-investigathon
export MODEL=$ROOT/sft_modelos/Qwen3.5-9B/Qwen3.5-9B-Q8_0.gguf THINKING=on RUNS_CONFIG=configs/runs.yaml LOG_ROOT=runs_lcb_9b PER_SLOT=81920
export LADDER="24:917504 24:786432 16:655360" EXTRA_FLAGS="" RHBENCH_ROBUST=1
bash scripts/run_lcb_27b.sh qwen9b-base-lcb
echo "[$(date)] chain finished"
