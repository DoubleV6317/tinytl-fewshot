#!/usr/bin/env bash
# Runs every row of a run list made by make_runlist.py, in order, with fewshot_depth.py.
# Resumable: a run whose result.json already exists is skipped; a partial runs/depth-* directory is redone.
# Stops at the first failure or malformed row. Usage: bash run-depth-sweep.sh runlists/pilot_flowers102.csv
# Before a night run: close other GPU programs, and disable sleep (powercfg /change standby-timeout-ac 0).
cd "$(dirname "$0")"
LIST=${1:?usage: run-depth-sweep.sh runlists/<list>.csv}
PY=.venv/Scripts/python.exe
LOG=depth-sweep.log
HEADER='block,dataset,k,arm,batch,lr,seed,epochs,val_every,tag,name'  # older lists; newer ones add ,head_init
stamp() { date '+%m-%d %H:%M:%S'; }
[ -f "$LIST" ] || { echo "no such run list: $LIST" >&2; exit 1; }
case "$(head -n 1 "$LIST" | tr -d '\r')" in
  "$HEADER"|"$HEADER,head_init") ;;
  *) echo "unexpected header in $LIST" >&2; exit 1 ;;
esac
mkdir -p logs runs
LOCK=runs/.sweep.lock
mkdir "$LOCK" 2>/dev/null || { echo "another sweep is running (remove $LOCK if it is stale)" >&2; exit 1; }
trap 'rmdir "$LOCK"' EXIT
echo "[$(stamp)] list $LIST, GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)" >> "$LOG"
tail -n +2 "$LIST" | tr -d '\r' | while IFS=, read -r block dataset k arm batch lr seed epochs val_every tag name \
    head_init || [ -n "$block" ]; do
  [ -z "$block$name" ] && continue  # blank line
  case "$name" in
    depth-*,*|*/*|*..*) echo "[$(stamp)] BAD ROW name=$name" >> "$LOG"; exit 1 ;;
    depth-?*) ;;
    *) echo "[$(stamp)] BAD ROW name=$name" >> "$LOG"; exit 1 ;;
  esac
  head_init=${head_init:-random}
  case "$head_init" in
    random|imprint) ;;
    *) echo "[$(stamp)] BAD ROW head_init=$head_init" >> "$LOG"; exit 1 ;;
  esac
  if [ -f "runs/$name/result.json" ]; then
    echo "[$(stamp)] skip $name" >> "$LOG"; continue
  fi
  echo "[$(stamp)] start $block $name" >> "$LOG"
  "$PY" -u fewshot_depth.py --dataset "$dataset" --k "$k" --arm "$arm" --batch "$batch" --lr "$lr" \
    --seed "$seed" --epochs "$epochs" --val_every "$val_every" --tag "$tag" --head_init "$head_init" \
    > "logs/$name.out.log" 2> "logs/$name.err.log" < /dev/null
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "runs/$name/result.json" ]; then
    echo "[$(stamp)] FAILED $name rc=$rc" >> "$LOG"
    tail -5 "logs/$name.err.log" >> "$LOG"
    exit 1
  fi
  grep -q "\"name\": \"$name\"" "runs/$name/result.json" || { echo "[$(stamp)] NAME MISMATCH $name" >> "$LOG"; exit 1; }
  grep -h '^RESULT ' "logs/$name.out.log" >> "$LOG"
  echo "[$(stamp)] done $name" >> "$LOG"
done || exit 1
echo "[$(stamp)] ALL DONE $LIST" >> "$LOG"
