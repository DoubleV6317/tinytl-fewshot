#!/usr/bin/env bash
# The supplementary experiments (实验设计/补充实验设计.md), unattended:
#   block X (Aircraft LR below the grid) -> lrx_freeze -> X-R if an LR changed
#   -> runlists/supp.csv (K2, S, I, K10, I10, both datasets) -> analyze_supp.py -> measure_time.py if the GPU is idle.
# Rerunning resumes (finished runs are skipped; lr_choice_aircraft_supp.json, once written, is kept).
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
stamp() { date '+%m-%d %H:%M:%S'; }
echo "[$(stamp)] run-supp start" >> depth-sweep.log
[ -f lr_choice_aircraft.json ] && [ -f lr_choice_flowers102.json ] || { echo "k=5 LR files missing" >&2; exit 1; }
if [ ! -f runlists/.supp_started ]; then
  # First start only: no supplementary run may exist yet, or run-depth-sweep.sh would skip it and reuse a stale
  # result. Smoke runs carry a -smoke tag and are ignored.
  "$PY" - <<'EOF' || exit 1
import re, sys
from pathlib import Path
pat = re.compile(r'^depth-(flowers102|aircraft)-k(2|10)-'           # K2, S, I at k=2, K10
                 r'|^depth-aircraft-k5-[A-Za-z0-9]+-b8-lr[12]e-4-s\d+$'  # X and X-R
                 r'|-imp$|-e(10|125)(-v25)?$')                        # I, I10, S
hits = sorted(d.name for d in Path('runs').iterdir() if d.is_dir() and '-smoke' not in d.name and pat.search(d.name))
if hits:
    sys.exit('refusing to start: supplementary run names already exist in runs/: %s' % hits[:10])
print('pre-flight ok: no supplementary run exists yet')
EOF
  touch runlists/.supp_started
fi
"$PY" make_runlist.py lrx || exit 1
bash run-depth-sweep.sh runlists/lrx_aircraft.csv || exit 1
if [ ! -f lr_choice_aircraft_supp.json ]; then
  "$PY" make_runlist.py lrx_freeze > logs/freeze_aircraft_supp.log 2>&1 || { cat logs/freeze_aircraft_supp.log; exit 1; }
  grep '^chosen' logs/freeze_aircraft_supp.log >> depth-sweep.log
fi
if [ -f runlists/lrrerun_aircraft.csv ]; then
  bash run-depth-sweep.sh runlists/lrrerun_aircraft.csv || exit 1
fi
"$PY" make_runlist.py supp || exit 1
bash run-depth-sweep.sh runlists/supp.csv || exit 1
"$PY" analyze_supp.py > 实验设计/分析输出_补充.txt 2>&1 || { tail -20 实验设计/分析输出_补充.txt; exit 1; }
echo "[$(stamp)] analysis written to 实验设计/分析输出_补充.txt" >> depth-sweep.log
# Timing needs an idle GPU; it refuses (exit 1) otherwise, which does not fail the chain.
if "$PY" measure_time.py > logs/measure_time.log 2>&1; then
  echo "[$(stamp)] measure_time done: time_table.json" >> depth-sweep.log
else
  echo "[$(stamp)] measure_time skipped: $(tail -n 1 logs/measure_time.log)" >> depth-sweep.log
fi
echo "[$(stamp)] run-supp ALL DONE" >> depth-sweep.log
