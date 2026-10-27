"""Reviewer check (r2): add the forward inference transient to the engine bound.
engine_peak >= max(saved_needed + extra, transient(bs) + extra), transient = TinyTL profiler's inference peak
(require_backward=False) per image x bs. CPU only, no training."""
import sys, json
from pathlib import Path
D = Path(__file__).parent
sys.path.insert(0, str(D))
from mem_harness import build
from tinytl.utils.memory_cost_profiler import count_activation_size
eb = {(r['arm'], r['bs']): r for r in json.loads((D / 'engine_bound_result.json').read_text())}
for arm in ['last', 'T4', 'T3', 'LB', 'B', 'full']:
    net = build(arm)
    peak_inf, _ = count_activation_size(net, (1, 3, 224, 224), False, 32)
    for bs in (8, 1):
        r = eb[(arm, bs)]
        saved = r['saved_needed'].get('full', 0) + r['saved_needed'].get('mask', 0)
        extra = r['weights_opt_input_MiB']
        trans = peak_inf * bs / 2 ** 20
        print(f'{arm:5s} bs{bs}: engine_lb {r["engine_lb_MiB"]:6.1f}  transient {trans:5.1f}  '
              f'max-bound {max(saved, trans) + extra:6.1f}  measured {r["measured_MiB"]:6.1f}', flush=True)
