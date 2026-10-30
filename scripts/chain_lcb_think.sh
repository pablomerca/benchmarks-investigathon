#!/usr/bin/env bash
# Base 27B on LCB with reasoning ON: 2 attempts, 16k response limit, same generation settings and splits as the other LCB runs.
cd /home/ubuntu/investigathon/benchmarks-investigathon
THINKING=on LOG_ROOT=runs_lcb_think bash scripts/run_lcb_27b.sh qwen27b-lcb-base-think
echo "[$(date)] chain finished"
