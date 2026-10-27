#!/usr/bin/env bash
# Flowers102 5-shot: does the lite residual pooling ratio move accuracy more than seed noise?
# Resumable: a run whose result.json already exists is skipped.
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
LOG=fewshot-sweep.log
mkdir -p logs
stamp() { date +%H:%M:%S; }
for seed in 0 1 2; do
  for cfg in "r2:2,2,2,2,2,2" "r1:1,1,1,1,1,1" "r4:4,4,4,4,4,4"; do
    tag=${cfg%%:*}; ratios=${cfg#*:}
    name="fewshot-k5-${tag}-seed${seed}"
    if [ -f "runs/$name/result.json" ]; then
      echo "[$(stamp)] skip $name" >> "$LOG"; continue
    fi
    rm -rf "runs/$name"
    echo "[$(stamp)] start $name" >> "$LOG"
    "$PY" -u fewshot_resolution.py --k 5 --seed "$seed" --ratios "$ratios" --tag "$tag" --epochs 50 \
      > "logs/$name.out.log" 2> "logs/$name.err.log"
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "[$(stamp)] FAILED $name rc=$rc" >> "$LOG"
      tail -5 "logs/$name.err.log" >> "$LOG"
      exit $rc
    fi
    grep -h '^RESULT ' "logs/$name.out.log" >> "$LOG"
    echo "[$(stamp)] done $name" >> "$LOG"
  done
done
echo "[$(stamp)] ALL DONE" >> "$LOG"
