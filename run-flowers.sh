#!/usr/bin/env bash
# The whole Flowers102 part, unattended: LR pilot -> freeze LRs (+ half-budget check) -> main block A-F.
# Each stage stops the chain on failure; rerunning resumes (finished runs are skipped, the frozen LR file is kept).
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
stamp() { date '+%m-%d %H:%M:%S'; }
echo "[$(stamp)] run-flowers start" >> depth-sweep.log
"$PY" make_runlist.py pilot || exit 1
bash run-depth-sweep.sh runlists/pilot_flowers102.csv || exit 1
if [ ! -f lr_choice_flowers102.json ]; then
  "$PY" make_runlist.py freeze > logs/freeze_flowers102.log 2>&1 || { cat logs/freeze_flowers102.log; exit 1; }
fi
# Gate: the official recipe (LB at 2e-4) must reach a sane 5-shot val accuracy (old r2 runs: 85.7-87.3 at epoch 50).
"$PY" - <<'EOF' >> depth-sweep.log || exit 1
import json, sys
p = json.load(open('lr_choice_flowers102.json'))['pilot']['LB']['mean_final_val']
v = p.get(repr(2e-4))
print('gate: LB 2e-4 mean final val = %s' % v)
sys.exit(0 if v is not None and v >= 84.0 else 1)
EOF
if [ -f runlists/halfbudget_flowers102.csv ]; then
  bash run-depth-sweep.sh runlists/halfbudget_flowers102.csv || exit 1
fi
"$PY" make_runlist.py main || exit 1
bash run-depth-sweep.sh runlists/main_flowers102.csv || exit 1
echo "[$(stamp)] run-flowers ALL DONE" >> depth-sweep.log
