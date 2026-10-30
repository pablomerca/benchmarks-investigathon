#!/usr/bin/env bash
# After the running post_retrain chain ends: train SoRH + SoRH_Control (separate processes), then benchmark them.
cd /home/ubuntu/investigathon/benchmarks-investigathon
bash ../sft_modelos/run_retrain_sorh.sh "$1" > ../sft_modelos/retrain_sorh.log 2>&1
ONLY="SoRH SoRH_Control" bash scripts/post_retrain_27b_v2.sh
