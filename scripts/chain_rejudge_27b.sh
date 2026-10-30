#!/usr/bin/env bash
# Judge-only pass of every saved EM response (new-adapter check + old-adapter check) with the plain 27B base, then compare to the 9B judge.
# Generations are copied, so the 9B judgments stay untouched. Usage: nohup bash scripts/chain_rejudge_27b.sh > logs_capability/rejudge_27b.log 2>&1 &
ROOT=/home/ubuntu/investigathon
SFT=$ROOT/sft_modelos
cd $ROOT/benchmarks-investigathon
for d in results_heldout_check_em results_heldout_check_em_old; do
  mkdir -p ${d}_27bjudge && cp -n $d/generations.jsonl ${d}_27bjudge/generations.jsonl
done
SPID=""
stop_server() { [ -n "$SPID" ] && kill $SPID 2>/dev/null && wait $SPID 2>/dev/null; SPID=""; }
trap stop_server EXIT
SLOTS=0
for n in 24 20 16; do
  /home/ubuntu/llama.cpp/build/bin/llama-server -m $SFT/Qwen3.5-27B/Qwen3.5-27B-Q8_0.gguf -c $(( n * 4096 )) --parallel $n \
    --port 8082 -ngl 99 --jinja --reasoning off --chat-template-kwargs '{"enable_thinking":false}' -fa on -b 4096 -ub 1024 \
    --alias rejudge-27b > logs_capability/server_rejudge_27b.log 2>&1 &
  SPID=$!
  for _ in $(seq 1 150); do curl -sf localhost:8082/health >/dev/null && { SLOTS=$n; break 2; }; kill -0 $SPID 2>/dev/null || break; sleep 3; done
  stop_server; sleep 10
done
[ $SLOTS -gt 0 ] || { echo "27B judge server failed to start"; exit 1; }
echo "[$(date)] 27B judge server up, $SLOTS slots"
source .venv/bin/activate
echo "=== new-adapter check (27B judge) ==="
python scripts/heldout_em_check.py --phase judge --workers $SLOTS --out results_heldout_check_em_27bjudge
echo "=== old-adapter check (27B judge) ==="
python scripts/heldout_em_old_check.py --phase judge --workers $SLOTS --out results_heldout_check_em_old_27bjudge
python scripts/compare_em_judges.py
echo "[$(date)] re-judge finished"
