"""Cross-process repeat of the unified harness, LB first (to show the one-time workspace), then reversed order;
saves JSON with nvidia-smi utilization at start (review S6)."""
import sys, json, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from mem_harness import measure
util = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'],
                      capture_output=True, text=True).stdout.strip()
rows = []
for bs in (8, 1):
    for i, arm in enumerate(['LB', 'T3', 'T4', 'head', 'last', 'LB']):
        r = measure(arm, bs)
        r['position_in_process'] = len(rows) + 1
        rows.append(r)
        print('%-5s bs%d pos%2d peak %.1f MiB' % (arm, bs, r['position_in_process'], r['peak_MiB']), flush=True)
Path(__file__).with_name('mem_repeat_result.json').write_text(
    json.dumps({'nvidia_smi_at_start': util, 'rows': rows}, indent=1), encoding='utf-8')
print('util at start:', util)
