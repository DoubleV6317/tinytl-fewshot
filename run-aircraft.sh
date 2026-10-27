#!/usr/bin/env bash
# The whole FGVC-Aircraft part, unattended (实验设计/最终实验设计.md, run list and §1.4):
#   LR pilot P-Air -> freeze LRs (+ half-budget check) -> A-Air -> C-Air (LB at batch 1) -> conditional 2.
# Run prepare_aircraft.py first. Rerunning resumes (finished runs are skipped, the frozen LR file is kept).
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
D=aircraft
stamp() { date '+%m-%d %H:%M:%S'; }
echo "[$(stamp)] run-aircraft start" >> depth-sweep.log
[ -f data/aircraft_prep/splits.json ] || { echo "run prepare_aircraft.py first" >&2; exit 1; }
"$PY" make_runlist.py pilot --dataset $D || exit 1
bash run-depth-sweep.sh runlists/pilot_$D.csv || exit 1
if [ ! -f lr_choice_$D.json ]; then
  "$PY" make_runlist.py freeze --dataset $D > logs/freeze_$D.log 2>&1 || { cat logs/freeze_$D.log; exit 1; }
fi
# Gate against a broken pipeline only (chance is 1%): the best LB pilot LR must reach 20% val.
"$PY" - <<'EOF' >> depth-sweep.log || exit 1
import json, sys
d = json.load(open('lr_choice_aircraft.json'))
v = d['pilot']['LB']['mean_final_val'][repr(d['lr']['LB'])]
print('gate: aircraft LB mean final val at the chosen LR = %s' % v)
sys.exit(0 if v is not None and v >= 20.0 else 1)
EOF
if [ -f runlists/halfbudget_$D.csv ]; then
  bash run-depth-sweep.sh runlists/halfbudget_$D.csv || exit 1
fi
"$PY" make_runlist.py main --dataset $D || exit 1
bash run-depth-sweep.sh runlists/main_$D.csv || exit 1
"$PY" make_runlist.py batch1 --dataset $D || exit 1
bash run-depth-sweep.sh runlists/batch1_$D.csv || exit 1
# Conditional 2 fires if the batch-1 point estimate d_b1 = test(LB-b1) - test(LB-b8), seeds 0-4, is below -2 pp.
rm -f runlists/.cond2_fires
"$PY" - <<'EOF' >> depth-sweep.log || exit 1
import csv, json, sys
rows = lambda f: list(csv.DictReader(open(f, encoding='utf-8')))
res = lambda r: json.load(open('runs/%s/result.json' % r['name'], encoding='utf-8'))['test_top1']
b8 = {int(r['seed']): res(r) for r in rows('runlists/main_aircraft.csv') if r['arm'] == 'LB'}
b1 = {int(r['seed']): res(r) for r in rows('runlists/batch1_aircraft.csv')}
d = [b1[s] - b8[s] for s in sorted(b1)]
m = sum(d) / len(d)
print('cond2: d_b1 per seed %s, mean %+.2f -> %s' % (' '.join('%+.2f' % x for x in d), m,
                                                    'FIRES' if m < -2 else 'does not fire'))
open('runlists/.cond2_fires', 'w').write('1' if m < -2 else '0')
EOF
if [ "$(cat runlists/.cond2_fires 2>/dev/null)" = "1" ]; then
  "$PY" make_runlist.py lrcheck --dataset $D || exit 1
  bash run-depth-sweep.sh runlists/lrcheck_$D.csv || exit 1
fi
echo "[$(stamp)] run-aircraft ALL DONE" >> depth-sweep.log
