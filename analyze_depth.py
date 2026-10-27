"""Pre-registered analysis of the depth experiment (实验设计/最终实验设计.md §1.3, §1.4, Appendix C). CPU only.

  python analyze_depth.py [flowers102] [aircraft]     (default: both; missing blocks are reported and skipped)

Runs are found through the run lists in runlists/, never by globbing. Rules, all on test top-1 with paired seeds:
  * H1: d = arm - LB, two-sided 93% t-CI, 2 pp margin, four labels; Holm (one-sided 0.035) over the T3/T4/last
    retains claims; T2 exploratory.
  * H2 (Aircraft): d = LB-b1 - LB-b8. Primary: catastrophic if U < -5, not catastrophic if L > -5. Secondary: within
    2 pp if L > -2. Conditional 2 decides whether a negative label is LR-suspect.
  * H2b (Aircraft, <= 128 MiB): d = LB-b1 - T4-b8. Batch better if L > 0, depth better if U < 0, a tie goes to T4.
  * Half-budget: the 100-epoch run's final val vs the 50-epoch pilot run at the same arm, LR and seed (+1 pp rule).
  * Cross-domain: Welch contrast of d_Aircraft - d_Flowers per arm.
Then a post-hoc section, not pre-registered, with the metrics the story PDF's purpose calls for: retention of TinyTL's
gain over FT-Last, the train-test gap, the best configuration under each memory budget, and cutoff-selection headroom.
"""
import csv
import json
import math
import sys

from scipy import stats

from depth_common import ROOT

CONF, MARGIN, HOLM_ALPHA = 0.93, 2.0, 0.035
H1_ARMS, EXPLORATORY = ['T3', 'T4', 'last'], ['T2']
PRIORS = {  # Appendix C: d vs TinyTL from TinyTrain Table 6 (AdapterDrop rungs, LastLayer)
    'flowers102': {'T2': -0.8, 'T3': -1.0, 'T4': -1.9, 'last': -0.7},
    'aircraft': {'T2': -2.6, 'T3': -3.1, 'T4': -10.6, 'last': -17.6},
}


def load_list(name):
    path = ROOT / 'runlists' / name
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    out = []
    for r in rows:
        res = ROOT / 'runs' / r['name'] / 'result.json'
        if res.exists():
            d = json.loads(res.read_text(encoding='utf-8'))
            assert d['name'] == r['name']
            out.append(d)
    return out, len(rows)


def mean(xs):
    return sum(xs) / len(xs)


def ci(d, conf=CONF):
    n, m = len(d), mean(d)
    sd = stats.tstd(d) if n > 1 else float('nan')
    se = sd / math.sqrt(n)
    h = stats.t.ppf(1 - (1 - conf) / 2, n - 1) * se
    p_ni = 1 - stats.t.cdf((m + MARGIN) / se, n - 1)  # one-sided, H0: d <= -margin
    return dict(n=n, d=m, sd=sd, L=m - h, U=m + h, p_ni=p_ni)


def label(L, U):
    if L > -MARGIN:
        return 'retains'
    if U < -MARGIN:
        return 'inferior by more than the margin'
    if U < 0:
        return 'significant loss, size unresolved'
    return 'inconclusive'


def fmt(c):
    return 'n=%d d=%+.2f sd=%.2f 93%% CI [%+.2f, %+.2f]' % (c['n'], c['d'], c['sd'], c['L'], c['U'])


def memory():
    t = json.loads((ROOT / 'memory_table.json').read_text(encoding='utf-8'))['table']
    return {(r['arm'], r['bs']): r for r in t}


def check_pairing(by, arms, seeds):
    for s in seeds:
        present = [a for a in arms if (a, s) in by]
        for key in ['subset_sha1', 'head_init_sha1']:
            assert len({by[(a, s)][key] for a in present}) == 1, 'seed %d: %s differs across arms' % (s, key)
    for a in arms:
        hs = {by[(a, s)]['frozen_sha1'] for s in seeds if (a, s) in by}
        assert len(hs) <= 1, '%s: frozen (quantized) weights differ across seeds' % a


def paired(by, a, b, seeds, key='test_top1'):
    ss = [s for s in seeds if (a, s) in by and (b, s) in by]
    return [by[(a, s)][key] - by[(b, s)][key] for s in ss], ss


def h1(dataset, mem):
    got = load_list('main_%s.csv' % dataset)
    if got is None:
        print('[%s] no main run list yet' % dataset)
        return None
    runs, planned = got
    print('\n=== H1 %s: %d / %d main runs done ===' % (dataset, len(runs), planned))
    if not runs:
        return None
    by = {(r['arm'], r['seed']): r for r in runs}
    seeds = sorted({r['seed'] for r in runs})
    arms = [a for a in ['LB'] + H1_ARMS + EXPLORATORY if any((a, s) in by for s in seeds)]
    check_pairing(by, arms, seeds)
    print('pairing ok: subset and head-init hashes equal across arms per seed; quantized weights equal across seeds')
    print('diverged: %s | code versions: %d | LRs: %s' % (
        [r['name'] for r in runs if r['diverged']] or 'none', len({r['code_sha1'] for r in runs}),
        {a: by[(a, min(s for s in seeds if (a, s) in by))]['lr'] for a in arms}))
    print('\nper arm: test top-1 mean +- sd | final val | final train | measured MiB at b8 (profiler MB)')
    for a in arms:
        rs = [by[(a, s)] for s in seeds if (a, s) in by]
        xs = [r['test_top1'] for r in rs]
        m = mem[(a, 8)]
        print('  %-4s n=%2d test %6.2f +- %.2f  val %6.2f  train %5.1f  %6.1f MiB (%.1f MB)' % (
            a, len(xs), mean(xs), stats.tstd(xs) if len(xs) > 1 else float('nan'),
            mean([r['final_val_top1'] for r in rs]), mean([r['final_train_top1'] for r in rs]),
            m['measured_MiB'], m['profiler_MB']))
    print('\npaired d = arm - LB on test (pp), margin %g' % MARGIN)
    res = {}
    for a in [x for x in H1_ARMS + EXPLORATORY if x in arms]:
        d, ss = paired(by, a, 'LB', seeds)
        if len(d) < 2:
            continue
        c = ci(d)
        c['per_seed'] = d
        res[a] = c
        print('  %-4s %s  %-36s NI p=%.4f%s  per seed: %s' % (
            a, fmt(c), label(c['L'], c['U']), c['p_ni'], ' (exploratory)' if a in EXPLORATORY else '',
            ' '.join('%+.1f' % x for x in d)))
    fam = sorted([a for a in H1_ARMS if a in res], key=lambda a: res[a]['p_ni'])
    if fam:
        print('Holm over the retains claims of %s, one-sided alpha %g:' % (fam, HOLM_ALPHA))
        stop = False
        for i, a in enumerate(fam):
            thr = HOLM_ALPHA / (len(H1_ARMS) - i)
            ok = (not stop) and res[a]['p_ni'] <= thr
            stop = stop or not ok
            print('  %-4s p=%.4f vs %.4f -> %s' % (a, res[a]['p_ni'], thr,
                                                  'retains (Holm)' if ok else 'retains not confirmed after Holm'))
    print('val-split d (same seeds), for agreement with test:',
          ' '.join('%s %+.2f' % (a, mean(paired(by, a, 'LB', seeds, 'final_val_top1')[0])) for a in res))
    print('prior (TinyTrain Table 6) vs observed d:',
          ' '.join('%s %+.1f/%+.2f' % (a, PRIORS[dataset][a], res[a]['d']) for a in res))
    gains = {}
    for a in arms:
        g = []
        for s in seeds:
            if (a, s) in by:
                v = {e: x for e, x, _ in by[(a, s)]['val_curve']}
                g.append((v[45] + v[50]) / 2 - (v[20] + v[25]) / 2)
        gains[a] = mean(g)
    print('half-budget gain on main seeds (val {45,50} - {20,25}):',
          ' '.join('%s %.2f' % (a, g) for a, g in gains.items()))
    hours = sum(r['setup_seconds'] + r['train_seconds'] + r['test_seconds'] for r in runs) / 3600
    print('GPU time: %.1f h over %d runs' % (hours, len(runs)))
    return dict(by=by, seeds=seeds, res=res)


def half_budget(dataset):
    got = load_list('halfbudget_%s.csv' % dataset)
    if got is None:
        print('\n[%s] half-budget check did not fire' % dataset)
        return
    runs, planned = got
    if not runs:
        print('\n[%s] half-budget 100-epoch run not done yet' % dataset)
        return
    r = runs[0]
    pilot = [p for p in load_list('pilot_%s.csv' % dataset)[0]
             if p['arm'] == r['arm'] and math.isclose(p['lr'], r['lr']) and p['seed'] == r['seed']]
    assert len(pilot) == 1
    v50, v100 = pilot[0]['final_val_top1'], r['final_val_top1']
    v = {e: x for e, x, _ in r['val_curve']}
    print('\n[%s] half-budget check: %s at lr %g, seed %d: val %.2f at 50 epochs -> %.2f at 100 epochs (%+.2f pp);'
          ' test %.2f -> %.2f. Within the 100-epoch run, val at epoch 50 = %.2f.' % (
              dataset, r['arm'], r['lr'], r['seed'], v50, v100, v100 - v50, pilot[0]['test_top1'], r['test_top1'],
              v[50]))
    affected = [r['arm']] + (['T4', 'T2'] if r['arm'] == 'T3' else [])
    if v100 - v50 >= 1.0:
        print('  -> %s results are labelled "at the official 50-epoch budget (%+.1f pp on val at 100 epochs)"'
              % ('/'.join(affected), v100 - v50))
    else:
        print('  -> under 1 pp: no budget label')


def h2(mem, a1):
    got = load_list('batch1_aircraft.csv')
    if got is None or not got[0] or a1 is None:
        print('\n[aircraft] H2/H2b: batch-1 runs not done yet')
        return
    runs, planned = got
    by = dict(a1['by'])
    for r in runs:
        by[('LB-b1', r['seed'])] = r
    seeds = sorted(r['seed'] for r in runs)
    for s in seeds:
        assert by[('LB-b1', s)]['subset_sha1'] == by[('LB', s)]['subset_sha1']
    print('\n=== H2 aircraft: %d / %d batch-1 runs (LB at lr %g) ===' % (len(runs), planned, runs[0]['lr']))
    xs = [by[('LB-b1', s)]['test_top1'] for s in seeds]
    print('LB-b1 test %.2f +- %.2f, val %.2f, train %.1f; measured %.1f MiB vs LB-b8 %.1f MiB' % (
        mean(xs), stats.tstd(xs) if len(xs) > 1 else float('nan'),
        mean([by[('LB-b1', s)]['final_val_top1'] for s in seeds]),
        mean([by[('LB-b1', s)]['final_train_top1'] for s in seeds]),
        mem[('LB', 1)]['measured_MiB'], mem[('LB', 8)]['measured_MiB']))
    d, ss = paired(by, 'LB-b1', 'LB', seeds)
    if len(d) >= 2:
        c = ci(d)
        primary = 'catastrophic' if c['U'] < -5 else 'not catastrophic' if c['L'] > -5 else 'undetermined'
        print('H2  d = LB-b1 - LB-b8: %s  per seed %s' % (fmt(c), ' '.join('%+.1f' % x for x in d)))
        print('    primary: %s; secondary: %s' % (primary, 'within 2 pp' if c['L'] > -2 else 'not shown within 2 pp'))
    d, ss = paired(by, 'LB-b1', 'T4', seeds)
    if len(d) >= 2:
        c = ci(d)
        verdict = 'batch better' if c['L'] > 0 else 'depth better' if c['U'] < 0 else 'tie (goes to T4)'
        print('H2b d = LB-b1 (%.1f MiB) - T4-b8 (%.1f MiB): %s -> %s' % (
            mem[('LB', 1)]['measured_MiB'], mem[('T4', 8)]['measured_MiB'], fmt(c), verdict))
    cond2 = ROOT / 'runlists' / '.cond2_fires'
    if cond2.exists() and cond2.read_text().strip() == '1':
        got = load_list('lrcheck_aircraft.csv')
        if not got or not got[0]:
            print('conditional 2 fired; LR-check runs not done yet')
            return
        b8, b1 = by[('LB', 0)]['final_val_top1'], by[('LB-b1', 0)]['final_val_top1']
        drop0 = b8 - b1
        print('conditional 2 (seed 0, val): LB-b8 %.2f, LB-b1 at %g %.2f, drop %.2f pp' % (b8, runs[0]['lr'], b1, drop0))
        sensitive = []
        for r in got[0]:
            left = b8 - r['final_val_top1']
            print('  LB-b1 at %g: val %.2f, test %.2f, drop left %.2f pp' % (
                r['lr'], r['final_val_top1'], r['test_top1'], left))
            if left < drop0 / 2:
                sensitive.append(r)
        if sensitive:
            best = max(sensitive, key=lambda r: r['final_val_top1'])
            print('  -> the batch-1 loss is LR-sensitive: no degradation claim; better LR %g' % best['lr'])
        else:
            print('  -> neither LR recovers half the drop: the H2 label stands')
    elif cond2.exists():
        print('conditional 2 did not fire (d_b1 point estimate >= -2 pp)')


def cross_domain(results):
    if not all(results.get(ds) for ds in PRIORS):
        return
    print('\n=== cross-domain contrast d_Aircraft - d_Flowers (Welch, 93%) ===')
    for a in H1_ARMS + EXPLORATORY:
        if a not in results['aircraft']['res'] or a not in results['flowers102']['res']:
            continue
        x, y = results['aircraft']['res'][a]['per_seed'], results['flowers102']['res'][a]['per_seed']
        vx, vy = stats.tvar(x) / len(x), stats.tvar(y) / len(y)
        df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
        diff, h = mean(x) - mean(y), stats.t.ppf(1 - (1 - CONF) / 2, df) * math.sqrt(vx + vy)
        print('  %-4s %+.2f [%+.2f, %+.2f] (df %.1f; prior %+.1f)' % (
            a, diff, diff - h, diff + h, df, PRIORS['aircraft'][a] - PRIORS['flowers102'][a]))


BUDGETS_MIB = [64, 96, 128, 192, 256, 512, None]  # None: no budget
CUT_ORDER = ['LB', 'T2', 'T3', 'T4', 'last']  # deepest first


def boot_ratio(num, den, n_boot=20000):
    """Percentile CI (CONF) of mean(num) / mean(den), resampling seeds with replacement."""
    import numpy as np
    rng = np.random.default_rng(0)
    num, den = np.asarray(num), np.asarray(den)
    idx = rng.integers(0, len(num), (n_boot, len(num)))
    r = num[idx].mean(1) / den[idx].mean(1)
    return float(np.quantile(r, (1 - CONF) / 2)), float(np.quantile(r, 1 - (1 - CONF) / 2))


def purpose_metrics(by, seeds, configs, mem, headroom_budgets=(128, 256, None)):
    """Prints R, the train-test gap, the budget table and the cutoff-selection headroom for one dataset and k.
    by: {(config, seed): result}; configs: {config: (memory-table arm, batch)}, including LB and last."""
    acc = lambda a, s: by[(a, s)]['test_top1']
    gap = lambda a, s: by[(a, s)]['final_train_top1'] - by[(a, s)]['test_top1']
    mib = lambda a: mem[configs[a]]['measured_MiB']

    print('retention R = (arm - last) / (LB - last) on test, paired seeds, bootstrap 93% CI over seeds:')
    for a in [x for x in configs if x not in ('LB', 'last')]:
        ss = [s for s in seeds if all((x, s) in by for x in (a, 'last', 'LB'))]
        if len(ss) < 2:
            continue
        num = [acc(a, s) - acc('last', s) for s in ss]
        den = [acc('LB', s) - acc('last', s) for s in ss]
        lo, hi = boot_ratio(num, den)
        print('  %-5s n=%2d R %5.1f%% [%5.1f, %5.1f]  (LB - last on these seeds: %.2f pp)' % (
            a, len(ss), 100 * mean(num) / mean(den), 100 * lo, 100 * hi, mean(den)))

    print('train - test gap (final-epoch train top-1 is logged in training mode: augmentation, dropout):')
    for a in configs:
        ss = [s for s in seeds if (a, s) in by]
        g = [gap(a, s) for s in ss]
        line = '  %-5s n=%2d gap %5.1f +- %.1f pp' % (a, len(ss), mean(g), stats.tstd(g))
        if a != 'LB':
            c = ci([gap(a, s) - gap('LB', s) for s in ss if ('LB', s) in by])
            line += '   vs LB %+.2f [%+.2f, %+.2f]' % (c['d'], c['L'], c['U'])
        print(line)

    print('best configuration within a training-memory budget (measured PyTorch peak MiB; d = config - LB-b8, '
          'paired; "tied" = paired 93% CI of the difference to the best includes 0):')
    d_vs_lb = {a: 0.0 if a == 'LB' else ci(paired(by, a, 'LB', seeds)[0])['d'] for a in configs}
    for budget in BUDGETS_MIB:
        feas = [a for a in configs if budget is None or mib(a) <= budget]
        head = '  %-11s' % ('no budget' if budget is None else '<= %d MiB' % budget)
        if not feas:
            print(head + 'nothing fits (smallest: last at %.1f MiB)' % mib('last'))
            continue
        best = max(feas, key=d_vs_lb.get)
        tied = []
        for a in feas:
            if a != best:
                d, _ = paired(by, a, best, seeds)
                if len(d) >= 2 and ci(d)['U'] >= 0:
                    tied.append(a)
        tinytl = next((a for a in ('LB', 'LB-b1') if a in feas), None)
        line = head + 'best %-5s (%5.1f MiB) test %.2f, d %+.2f' % (
            best, mib(best), mean([acc(best, s) for s in seeds if (best, s) in by]), d_vs_lb[best])
        line += ' | tied: %s' % (', '.join(tied) or 'none')
        line += ' | TinyTL fits as: %s' % ('%s (d %+.2f)' % (tinytl, d_vs_lb[tinytl]) if tinytl else 'nothing')
        print(line)

    print('cutoff-selection headroom (batch-8 cutoff arms only; seeds on which every feasible arm ran):')
    for budget in headroom_budgets:
        feas = [a for a in CUT_ORDER if a in configs and (budget is None or mib(a) <= budget)]
        ss = [s for s in seeds if all((a, s) in by for a in feas)]
        if not feas or not ss:
            continue
        means = {a: mean([acc(a, s) for s in ss]) for a in feas}
        oracle = mean([max(acc(a, s) for a in feas) for s in ss])
        fixed, deepest = max(feas, key=means.get), feas[0]
        winners = [max(feas, key=lambda a: acc(a, s)) for s in ss]
        print('  %-11s arms %s, n=%d: per-seed oracle %.2f | best fixed arm %s %.2f | deepest-that-fits %s %.2f'
              ' | headroom over best fixed %.2f pp, over deepest %.2f pp | per-seed winners %s' % (
                  'no budget' if budget is None else '<= %d MiB' % budget, '/'.join(feas), len(ss), oracle,
                  fixed, means[fixed], deepest, means[deepest], oracle - means[fixed], oracle - means[deepest],
                  ' '.join('%s:%d' % (a, winners.count(a)) for a in feas if winners.count(a))))


HEADROOM_NOTE = ('(the per-seed oracle also profits from training noise, so the headrooms are upper bounds on what '
                 'any task-adaptive cutoff rule could gain)')


def purpose(results, mem):
    print('\n=== post-hoc metrics for the story\'s purpose (NOT pre-registered; k=5) ===')
    for ds, res in results.items():
        if not res:
            continue
        by, seeds = dict(res['by']), res['seeds']
        configs = {a: (a, 8) for a in CUT_ORDER if any((a, s) in by for s in seeds)}
        if ds == 'aircraft':
            got = load_list('batch1_aircraft.csv')
            if got and got[0]:
                for r in got[0]:
                    by[('LB-b1', r['seed'])] = r
                configs['LB-b1'] = ('LB', 1)
        print('\n[%s]' % ds)
        purpose_metrics(by, seeds, configs, mem)
    print('\n' + HEADROOM_NOTE)


def main():
    datasets = sys.argv[1:] or ['flowers102', 'aircraft']
    mem = memory()
    results = {}
    for ds in datasets:
        results[ds] = h1(ds, mem)
        half_budget(ds)
    if 'aircraft' in datasets:
        h2(mem, results['aircraft'])
    cross_domain(results)
    purpose(results, mem)


if __name__ == '__main__':
    main()
