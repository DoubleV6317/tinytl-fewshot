"""Block N4 (实验设计/补充实验设计.md): task-adaptive block selection under a memory budget, before any N4 training.

  python blockselect.py           on the GPU; writes blockselect.json once (it is frozen before the N4 runs)
  python blockselect.py --check   the same on 2 seeds of one cell, writes nothing

1. Memory structure, measured as in measure_memory.py (PyTorch peak over 3 train steps, 102-way head): the contiguous
   cut from every lite-residual block (S_b..21; the T arms at stage starts) at batch 8 and 1, and every block trained
   on its own at batch 8. T4 and T3 are re-measured first and must match memory_table.json within 1 MiB, and again
   at the end. Step 2 runs in a child process before any measurement: CUDA work in the measuring process shifted
   T4 and T3 by about 1 MiB.
2. Per cell (Aircraft k=10, Flowers k=5) and seed 0-4, on that seed's k-shot training images: the Fisher potential
   P_b of every block output exactly as fisher_stage.py (TinyTrain Eq. 2; fp32 weight-standardized net, leave-one-out
   ProtoNet head), and the TinyTrain Eq. 3 score of every lite block, s_b = P_b / ((W_b / max W) (M_b / max M)), with
   W_b the parameters the block trains and M_b the backward MACs it adds to the contiguous cut,
   bwd(S_b..21) - bwd(S_next..21), the cut after block 21 being `head` (feature-mix bias + classifier).
3. FS (Fisher selection): the lite blocks in descending s_b, each added if the measured batch-8 memory of the
   selected set with it stays within the budget, else skipped. C: the deepest contiguous cut within the budget.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from depth_common import ROOT, LITE_BLOCKS, BLOCK_PARAMS, setup_paths, build_net, blockset_arm

CELLS = [('aircraft', 10), ('flowers102', 5)]
N_CLASSES = {'flowers102': 102, 'aircraft': 100}
SEEDS = list(range(5))
BUDGET = 128  # MiB, measured at batch 8
ANCHOR_TOL = 1.0  # MiB
OUT = ROOT / 'blockselect.json'


def contiguous(b):
    return blockset_arm([x for x in LITE_BLOCKS if x >= b])


class Memory(object):
    """Measured peak MiB per (arm, batch), cached; one process, after a throw-away arm (as measure_memory.worker)."""

    def __init__(self):
        from measure_memory import build, measured_peak
        self.build, self.peak, self.cache = build, measured_peak, {}
        self.peak(self.build('LB')[0], 8)

    def __call__(self, arm, bs=8):
        if (arm, bs) not in self.cache:
            self.cache[(arm, bs)] = round(self.peak(self.build(arm)[0], bs)[0], 1)
        return self.cache[(arm, bs)]


def block_costs(n_classes):
    """W_b (parameters) and M_b (backward M MACs the block adds to the contiguous cut) per lite block."""
    from measure_macs import count_macs
    nxt = dict(zip(LITE_BLOCKS, LITE_BLOCKS[1:] + [None]))
    bwd = {b: count_macs(contiguous(b) if b is not None else 'head', n_classes)['bwd_total'] / 1e6
           for b in LITE_BLOCKS + [None]}
    return {b: BLOCK_PARAMS[b] for b in LITE_BLOCKS}, {b: bwd[b] - bwd[nxt[b]] for b in LITE_BLOCKS}


def block_fisher(cells, seeds, chunk, path):
    """Writes {dataset|k: {seed: [P per block, support stats]}} with fisher_stage's procedure to path."""
    import torch
    from ofa.utils import replace_conv2d_with_my_conv2d
    from fisher_stage import support_set, fisher
    torch.backends.cudnn.benchmark = False
    out = {}
    for dataset, k in cells:
        net = build_net(N_CLASSES[dataset])
        for p in net.parameters():
            p.requires_grad = False
        replace_conv2d_with_my_conv2d(net, 1e-5)
        net = net.to('cuda').eval()
        out['%s|%d' % (dataset, k)] = {s: fisher(net, *support_set(dataset, s, k=k), 'cuda', chunk) for s in seeds}
    path.write_text(json.dumps(out), encoding='utf-8')


def fisher_in_child(cells, seeds, chunk):
    """block_fisher in its own process: CUDA use before the memory measurements shifts them by about 1 MiB."""
    path = ROOT / 'cache' / 'blockselect_fisher.tmp.json'
    cmd = [sys.executable, '-u', __file__, '--fisher_out', str(path), '--chunk', str(chunk),
           '--cells'] + ['%s|%d' % c for c in cells] + ['--seeds'] + [str(s) for s in seeds]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    raw = json.loads(path.read_text(encoding='utf-8'))
    path.unlink()
    return {(c.split('|')[0], int(c.split('|')[1])): {int(s): ({int(b): v for b, v in pot.items()}, sup)
                                                     for s, (pot, sup) in by.items()} for c, by in raw.items()}


def greedy(ranking, mem):
    chosen, trace = [], []
    for b in ranking:
        arm = blockset_arm(chosen + [b])
        m = mem(arm)
        trace.append(dict(block=b, set=arm, MiB=m, added=m <= BUDGET))
        if m <= BUDGET:
            chosen.append(b)
    return blockset_arm(chosen), trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='2 seeds of the Aircraft cell, writes nothing')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--chunk', type=int, default=32)
    ap.add_argument('--fisher_out', default=None, help=argparse.SUPPRESS)  # the child process of fisher_in_child
    ap.add_argument('--cells', nargs='+', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--seeds', nargs='+', type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.fisher_out:
        setup_paths()
        block_fisher([(c.split('|')[0], int(c.split('|')[1])) for c in args.cells], args.seeds, args.chunk,
                     Path(args.fisher_out))
        return
    if OUT.exists() and not args.check and not args.force:
        sys.exit('%s already exists; it is frozen before the N4 runs (use --force)' % OUT)
    if (ROOT / 'runs' / '.sweep.lock').exists():
        sys.exit('a sweep is running (runs/.sweep.lock): rerun when the GPU is idle')
    setup_paths()
    import torch
    from measure_memory import gpu_util
    cells, seeds = (CELLS[:1], SEEDS[:2]) if args.check else (CELLS, SEEDS)
    util, t0 = gpu_util(), time.time()

    fis = fisher_in_child(cells, seeds, args.chunk)
    mem = Memory()
    table = json.loads((ROOT / 'memory_table.json').read_text(encoding='utf-8'))['table']
    ref = {r['arm']: r['measured_MiB'] for r in table if r['bs'] == 8}
    anchors = {a: (mem(a), ref[a]) for a in ('T4', 'T3')}
    print('anchors (measured now / memory_table.json, MiB): %s' % anchors)
    assert all(abs(x - y) <= ANCHOR_TOL for x, y in anchors.values()), 'memory differs from memory_table.json'

    structure = []
    for b in LITE_BLOCKS:
        row = dict(block=b, contiguous=contiguous(b), contiguous_b8=mem(contiguous(b)),
                   contiguous_b1=mem(contiguous(b), 1), alone=blockset_arm([b]), alone_b8=mem(blockset_arm([b])))
        structure.append(row)
        print('block %2d: from here on %-7s %6.1f MiB (b1 %5.1f) | alone %-4s %6.1f MiB (%.0f%% less)' % (
            b, row['contiguous'], row['contiguous_b8'], row['contiguous_b1'], row['alone'], row['alone_b8'],
            100 * (1 - row['alone_b8'] / row['contiguous_b8'])), flush=True)
    c_block = min(b for b in LITE_BLOCKS if mem(contiguous(b)) <= BUDGET)
    c_arm = contiguous(c_block)

    result = dict(meta=dict(budget_MiB=BUDGET, batch=8, memory_head_classes=102, gpu_util_at_start=util,
                            gpu=torch.cuda.get_device_name(), torch=torch.__version__, anchors_MiB=anchors,
                            score='TinyTrain Eq. 3: P / ((W / max W) (M / max M)) over lite blocks'),
                  structure=structure, C=dict(arm=c_arm, MiB=mem(c_arm)), cells=[])
    print('C (deepest contiguous cut within %d MiB): %s %.1f MiB' % (BUDGET, c_arm, mem(c_arm)))
    for (dataset, k) in cells:
        W, M = block_costs(N_CLASSES[dataset])
        w_max, m_max = max(W.values()), max(M.values())
        cell = dict(dataset=dataset, k=k, W=W, M_MMACs=M, seeds={})
        print('\n=== %s k=%d: M per block %s' % (dataset, k, {b: round(v, 1) for b, v in M.items()}))
        for seed in seeds:
            potential, support = fis[(dataset, k)][seed]
            P = {b: potential[b] for b in LITE_BLOCKS}
            s = {b: P[b] / ((W[b] / w_max) * (M[b] / m_max)) for b in LITE_BLOCKS}
            ranking = sorted(LITE_BLOCKS, key=lambda b: -s[b])
            fs, trace = greedy(ranking, mem)
            cell['seeds'][seed] = dict(support=support, P=P, score=s, ranking=ranking,
                                       ranking_by_P=sorted(LITE_BLOCKS, key=lambda b: -P[b]), trace=trace,
                                       FS=fs, FS_MiB=mem(fs))
            print('seed %d: support loss %.3f top1 %.1f%% | ranking by s %s | by P %s | FS %s (%.1f MiB)%s' % (
                seed, support['loss'], support['top1'], ranking[:8], cell['seeds'][seed]['ranking_by_P'][:8], fs,
                mem(fs), ' = C' if fs == c_arm else ''), flush=True)
        result['cells'].append(cell)
    # the anchors again: nothing in this process may have shifted the measurements
    from measure_memory import build, measured_peak
    again = {a: round(measured_peak(build(a)[0], 8)[0], 1) for a in ('T4', 'T3')}
    print('\nanchors at the end: %s; %d memory measurements; %.0f s' % (again, len(mem.cache), time.time() - t0))
    assert all(abs(again[a] - anchors[a][0]) <= ANCHOR_TOL for a in again), 'memory drifted during the run'
    result['meta']['anchors_end_MiB'] = again
    result['memory_MiB'] = {'%s|b%d' % key: v for key, v in sorted(mem.cache.items())}
    if args.check:
        print('check ok; nothing written')
        return
    tmp = OUT.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=1), encoding='utf-8')
    tmp.replace(OUT)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
