#!/usr/bin/env bash
# LCB impossible bench, 2 attempts: SoRH, then EM_Secure, then the base model (rerun of the pilot with 2 attempts).
SFT=/home/ubuntu/investigathon/sft_modelos
cd /home/ubuntu/investigathon/benchmarks-investigathon
bash scripts/run_lcb_27b.sh qwen27b-lcb-sorh      $SFT/Qwen3.5-27B_SoRH_seed0/Qwen3.5-27B_SoRH_seed0-Q8_0-LoRA.gguf
bash scripts/run_lcb_27b.sh qwen27b-lcb-em-secure $SFT/Qwen3.5-27B_EM_Secure_seed0/Qwen3.5-27B_EM_Secure_seed0-Q8_0-LoRA.gguf
bash scripts/run_lcb_27b.sh qwen27b-lcb-base
echo "[$(date)] chain finished"
