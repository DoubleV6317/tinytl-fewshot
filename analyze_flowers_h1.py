"""Quick H1-F analysis of the Flowers main block (pre-registered rules of 实验设计/最终实验设计.md): paired d vs LB,
two-sided 93% t-CI, 2 pp margin, four labels, Holm over the three retains claims. Run from outputs/TinyTL."""
import json, glob, math
from collections import defaultdict
from scipy import stats
R = {}
for f in glob.glob('runs/depth-flowers102-k5-*-b8-lr8e-4-s[0-9]/result.json'):
    r = json.load(open(f)); R[(r['arm'], r['seed'])] = r
print('runs:', len(R))
# pairing checks
for s in range(10):
    arms = [a for a in ['LB', 'T2', 'T3', 'T4', 'last'] if (a, s) in R]
    assert len({R[(a, s)]['subset_sha1'] for a in arms}) == 1 and len({R[(a, s)]['head_init_sha1'] for a in arms}) == 1, s
print('pairing: subset and head-init hashes identical across arms at every seed;',
      'distinct subsets across seeds:', len({R[('LB', s)]['subset_sha1'] for s in range(10)}))
print('diverged:', [k for k, r in R.items() if r['diverged']], '| code_sha1 values:', len({r['code_sha1'] for r in R.values()}))
mem = {r['arm']: r['measured_MiB'] for r in json.load(open('memory_table.json'))['table'] if r['bs'] == 8}
mem['last'] = mem['last']
print('\nper-arm test top-1 (mean ± sd over seeds), final val, final train')
for a in ['LB', 'T2', 'T3', 'T4', 'last']:
    xs = [R[(a, s)]['test_top1'] for s in range(10) if (a, s) in R]
    v = [R[(a, s)]['final_val_top1'] for s in range(10) if (a, s) in R]
    t = [R[(a, s)]['final_train_top1'] for s in range(10) if (a, s) in R]
    print('  %-4s n=%2d test %.2f ± %.2f  val %.2f  train %.1f  mem %.0f MiB' % (a, len(xs), sum(xs)/len(xs), stats.tstd(xs), sum(v)/len(v), sum(t)/len(t), mem[a]))
def ci(d, conf=0.93):
    n = len(d); m = sum(d) / n; sd = stats.tstd(d); se = sd / math.sqrt(n)
    h = stats.t.ppf(1 - (1 - conf) / 2, n - 1) * se
    p_ni = 1 - stats.t.cdf((m + 2) / se, n - 1)  # one-sided H0: d <= -2
    return m, sd, m - h, m + h, p_ni
def label(L, U):
    if L > -2: return 'retains'
    if U < -2: return 'inferior by more than the margin'
    if U < 0: return 'significant loss, size unresolved'
    return 'inconclusive'
print('\npaired d = arm - LB (pp), 93% CI, margin 2 pp')
res = {}
for a in ['T3', 'T4', 'last', 'T2']:
    seeds = [s for s in range(10) if (a, s) in R]
    d = [R[(a, s)]['test_top1'] - R[('LB', s)]['test_top1'] for s in seeds]
    m, sd, L, U, p = ci(d); res[a] = (m, sd, L, U, p, len(d))
    print('  %-4s n=%2d d=%+.2f sd=%.2f CI [%+.2f, %+.2f]  label: %-36s  NI p=%.4f  per-seed: %s' % (
        a, len(d), m, sd, L, U, label(L, U), p, ' '.join('%+.1f' % x for x in d)))
fam = sorted(['T3', 'T4', 'last'], key=lambda a: res[a][4])
print('\nHolm over {T3, T4, last} retains claims, one-sided alpha 0.035:')
stop = False
for i, a in enumerate(fam):
    thr = 0.035 / (3 - i); ok = (not stop) and res[a][4] <= thr; stop = stop or not ok
    print('  %-4s p=%.4f vs %.4f -> %s' % (a, res[a][4], thr, 'retains (Holm)' if ok else 'not confirmed after Holm'))
print('\nval vs test agreement on paired differences:')
for a in ['T3', 'T4', 'last', 'T2']:
    seeds = [s for s in range(10) if (a, s) in R]
    dv = [R[(a, s)]['final_val_top1'] - R[('LB', s)]['final_val_top1'] for s in seeds]
    print('  %-4s val d = %+.2f' % (a, sum(dv) / len(dv)))
print('\nhalf-budget gain (val 45/50 - 20/25), mean over main seeds:')
for a in ['LB', 'T3', 'T4', 'last', 'T2']:
    g = []
    for s in range(10):
        if (a, s) in R:
            v = {e: x for e, x, _ in R[(a, s)]['val_curve']}; g.append((v[45] + v[50]) / 2 - (v[20] + v[25]) / 2)
    print('  %-4s %.2f' % (a, sum(g) / len(g)))
tt = [(r['setup_seconds'] + r['train_seconds'] + r['test_seconds']) / 60 for r in R.values()]
print('\nmain block GPU time: %.1f h over %d runs' % (sum(tt) / 60, len(tt)))
