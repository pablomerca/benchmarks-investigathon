#!/usr/bin/env bash
# Base Qwen3.5-27B at Q4_K_M (quantized from the BF16 GGUF, no imatrix) on LCB: thinking ON, ONE attempt, 30k response limit,
# original + oneoff, same generation settings as the other LCB runs. Per-slot context limit 65536 (longest prompt ~22k + 30k reply).
# Waits for the quantization to finish first.
ROOT=/home/ubuntu/investigathon
cd $ROOT/benchmarks-investigathon
while pgrep -x llama-quantize > /dev/null; do sleep 15; done
Q4=$ROOT/sft_modelos/Qwen3.5-27B/Qwen3.5-27B-Q4_K_M.gguf
if ! grep -q "quant size" logs_capability/quantize_q4km.log || [ ! -s "$Q4" ]; then echo "[$(date)] quantization did not finish cleanly, see logs_capability/quantize_q4km.log"; exit 1; fi
echo "[$(date)] Q4_K_M ready: $(du -h $Q4 | cut -f1)"
export MODEL=$Q4 THINKING=on RUNS_CONFIG=configs/runs_lcb_pilot.yaml LOG_ROOT=runs_lcb_q4_think PER_SLOT=65536
export LADDER="16:294912 12:294912 8:229376"
export EXTRA_FLAGS="--backend-sampling"  # +16% tok/s in the A/B (results_speed_ab/); 16 slots is the measured throughput peak
bash scripts/run_lcb_27b.sh qwen27b-lcb-base-q4-think
echo "[$(date)] chain finished"
