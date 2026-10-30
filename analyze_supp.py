"""Analysis of the supplementary experiments (实验设计/补充实验设计.md); the rules are fixed there before any
supplementary run. CPU only.

  python analyze_supp.py

Runs are found through run lists only (main_*, pilot_aircraft, lrx_aircraft, lrrerun_aircraft, supp, k10t3_aircraft,
k10t2_aircraft, imprint_extra, k10b1_aircraft).
Sections:
  Q3  Aircraft LR below the grid (block X); if an LR changed, the k=5 Aircraft block in Q1 uses the new LRs and is
      the sensitivity analysis of H1-A (block X-R)
  Q1  per dataset and k in {2, 5, 10}: d vs LB with labels and a per-k Holm family, R, train-test gap, budget table,
      headroom, training compute; Aircraft k=10 also LB at batch 1 (block K10-B1) with the H2/H2b rules; then the
      k trend
  Q4  k=2 step-matched control (block S)
  Q2  imprinted vs random head at 50 epochs (blocks I, I-LB, I-K10) and at 10 epochs (block I10)
"""
import csv
import json

from scipy import stats

from analyze_depth import (MARGIN, HOLM_ALPHA, HEADROOM_NOTE, ci, label, fmt, mean, memory, paired, check_pairing,
                           boot_ratio, purpose_metrics)
from depth_common import ROOT, run_name
from make_runlist import (CONFIG, SUPP_LR_FILE, SUPP_SEEDS, K2_ARMS, STEP_ARMS, STEP_EPOCHS, STEP_VAL_EVERY,
                          SHORT_EPOCHS)

DATASETS = ['flowers102', 'aircraft']
LISTS = ['main_flowers102.csv', 'main_aircraft.csv', 'pilot_aircraft.csv', 'lrx_aircraft.csv',
         'lrrerun_aircraft.csv', 'supp.csv', 'k10t3_aircraft.csv', 'k10t2_aircraft.csv', 'imprint_extra.csv',
         'k10b1_aircraft.csv']
# T3 and T2 at k=10 are blocks K10-T3 and K10-T2 (Aircraft only), added after the K10 and K10-T3 results:
# exploratory, outside the Holm family. An arm without runs (T3, T2 on Flowers) drops out of the block.
K_ARMS = {2: K2_ARMS, 5: ['LB', 'T2', 'T3', 'T4', 'last'], 10: ['LB', 'T2', 'T3', 'T4', 'last']}
K_FAMILY = {2: ['T3', 'T4', 'last'], 5: ['T3', 'T4', 'last'], 10: ['T4', 'last']}
K_HEADROOM = {2: (128, 256), 5: (128, 256), 10: (128, 256)}
# Imprinted arms per k. LB at k=2 and 5 (block I-LB) and T4 at k=10 (block I-K10) were added after the I results:
# exploratory, like the T4-imprint vs LB lines.
IMPRINT_CELLS = {2: ['LB', 'T4', 'last'], 5: ['LB', 'T4', 'last'], 10: ['T4']}
LATE_IMPRINT = {(2, 'LB'), (5, 'LB'), (10, 'T4')}
EFFECT_MARGIN = 1.0  # pp, the "within +-1 pp" label for imprinting
# LB at batch 1 per (dataset, k): block K10-B1, added after the K10 results; exploratory, read with the H2/H2b rules
B1_CELLS = {('aircraft', 10)}
CATASTROPHIC = 5  # pp, the H2 rule
SEEDS = list(range(SUPP_SEEDS))  # seeds of the k=2, k=10, I and I10 blocks


def planned_names():
    names = set()
    for f in LISTS:
        path = ROOT / 'runlists' / f
        if path.exists():
            names |= {r['name'] for r in csv.DictReader(open(path, encoding='utf-8'))}
    return names


PLANNED = planned_names()


def get(dataset, k, arm, lr, seed, epochs=50, val_every=5, head_init='random', batch=8):
    """The result of one planned run, or None if it is not in a run list or not finished."""
    name = run_name(dataset, k, arm, batch, lr, seed, epochs, val_every, '', head_init)
    path = ROOT / 'runs' / name / 'result.json'
    if name not in PLANNED or not path.exists():
        return None
    r = json.loads(path.read_text(encoding='utf-8'))
    assert r['name'] == name and r.get('head_init', 'random') == head_init, name
    return r


def supp_lrs():
    lr = {'flowers102': json.loads((ROOT / 'lr_choice_flowers102.json').read_text(encoding='utf-8'))['lr']}
    if SUPP_LR_FILE.exists():
        lr['aircraft'] = json.loads(SUPP_LR_FILE.read_text(encoding='utf-8'))['lr']
    else:
        print('note: block X is not frozen yet; Aircraft uses the k=5 frozen LRs')
        lr['aircraft'] = json.loads((ROOT / 'lr_choice_aircraft.json').read_text(encoding='utf-8'))['lr']
    return lr


def main_seeds(dataset):
    return range(max(n for _, n in CONFIG[dataset]['main']))


def holm(res, family):
    fam = sorted([a for a in family if a in res], key=lambda a: res[a]['p_ni'])
    ok_all, stop = {}, False
    for i, a in enumerate(fam):
        thr = HOLM_ALPHA / (len(family) - i)
        ok = (not stop) and res[a]['p_ni'] <= thr
        stop = stop or not ok
        ok_all[a] = ok
        print('  Holm %-4s p=%.4f vs %.4f -> %s' % (a, res[a]['p_ni'], thr,
                                                   'retains (Holm)' if ok else 'retains not confirmed after Holm'))
    return ok_all


def val_at(r, epoch):
    return {e: v for e, v, _ in r['val_curve']}.get(epoch)


def train_tmacs(macs, dataset, arm, k, epochs):
    n_images = k * (102 if dataset == 'flowers102' else 100)
    return macs[dataset][arm]['train_per_image'] * 1e6 * n_images * epochs / 1e12


def lr_extension():
    print('\n=== Q3: Aircraft LR below the grid (block X, pilot seed) ===')
    if not SUPP_LR_FILE.exists():
        print('block X not frozen yet')
        return
    d = json.loads(SUPP_LR_FILE.read_text(encoding='utf-8'))
    for arm, rep in d['pilot'].items():
        vals = ' '.join('%s: %s' % (lr, v) for lr, v in sorted(rep['mean_final_val'].items(), key=lambda x: float(x[0])))
        print('  %-4s final val by LR {%s} | frozen %g, raw best %g, chosen %g%s' % (
            arm, vals, rep['frozen'], rep['raw_best'], rep['chosen'],
            ' (chosen LR at the edge of the extended grid)' if rep['chosen_at_grid_edge'] else ''))
    if d['changed']:
        print('  LR changed for %s: the k=5 Aircraft block in Q1 is at the new LRs and is the sensitivity analysis '
              'of H1-A (the pre-registered H1-A stays the primary result)' % d['changed'])
    else:
        print('  no LR changed: 4e-4 stays best within 1 pp below the old grid; H1-A stands as run')


def depth_block(dataset, k, lr, mem, macs):
    seeds = list(main_seeds(dataset) if k == 5 else SEEDS)
    arms = K_ARMS[k]
    by = {}
    for a in arms:
        for s in seeds:
            r = get(dataset, k, a, lr[a], s)
            if r is not None:
                by[(a, s)] = r
    arms = [a for a in arms if any((a, s) in by for s in seeds)] or arms
    print('\n--- %s k=%d: %d runs found (arms %s, LRs %s) ---' % (
        dataset, k, len(by), '/'.join(arms), {a: lr[a] for a in arms}))
    if not by:
        return None
    seeds = [s for s in seeds if any((a, s) in by for a in arms)]  # partial results: only seeds with a finished run
    check_pairing(by, arms, seeds)
    print('pairing ok | diverged: %s | code versions: %d' % (
        [r['name'] for r in by.values() if r['diverged']] or 'none', len({r['code_sha1'] for r in by.values()})))
    if k == 10 and dataset == 'flowers102':
        print('note: Flowers k=10 is the whole train split, so every seed trains on the same images')
    for a in arms:
        rs = [by[(a, s)] for s in seeds if (a, s) in by]
        xs = [r['test_top1'] for r in rs]
        print('  %-4s n=%2d test %6.2f +- %.2f  val %6.2f  train %5.1f  %6.1f MiB' % (
            a, len(xs), mean(xs), stats.tstd(xs) if len(xs) > 1 else float('nan'),
            mean([r['final_val_top1'] for r in rs]), mean([r['final_train_top1'] for r in rs]),
            mem[(a, 8)]['measured_MiB']))
    res = {}
    if 'LB' in arms:
        print('paired d = arm - LB on test (pp), margin %g' % MARGIN)
        for a in [x for x in arms if x != 'LB']:
            d, _ = paired(by, a, 'LB', seeds)
            if len(d) < 2:
                continue
            res[a] = ci(d)
            print('  %-4s %s  %-36s NI p=%.4f%s' % (a, fmt(res[a]), label(res[a]['L'], res[a]['U']), res[a]['p_ni'],
                                                   '' if a in K_FAMILY[k] else ' (exploratory)'))
    ok = holm(res, K_FAMILY[k])
    keep = [a for a in ok if ok[a]]
    print('shallowest cutoff that retains after Holm: %s' % (
        min(keep, key=lambda a: mem[(a, 8)]['measured_MiB']) if keep else 'none'))
    gains = {}
    for a in arms:
        g = [(val_at(by[(a, s)], 45) + val_at(by[(a, s)], 50)) / 2 - (val_at(by[(a, s)], 20) + val_at(by[(a, s)], 25)) / 2
             for s in seeds if (a, s) in by]
        gains[a] = mean(g)
    print('half-budget gain (val {45,50} - {20,25}, descriptive): %s' % ' '.join(
        '%s %.2f' % (a, g) for a, g in gains.items()))
    print('training compute (fwd+bwd MACs x images x epochs, excluding val/test): %s' % ' | '.join(
        '%s %.1f T (%.2fx LB)' % (a, train_tmacs(macs, dataset, a, k, 50),
                                  train_tmacs(macs, dataset, a, k, 50) / train_tmacs(macs, dataset, 'LB', k, 50))
        for a in arms))
    b1 = batch1(dataset, k, by, seeds, mem) if (dataset, k) in B1_CELLS else {}
    if 'LB' in arms and 'last' in arms:
        configs = {a: (a, 8) for a in arms}
        if b1:
            configs['LB-b1'] = ('LB', 1)
        purpose_metrics({**by, **b1}, seeds, configs, mem, K_HEADROOM[k])
    return dict(by=by, seeds=seeds, res=res, holm=ok, arms=arms)


def batch1(dataset, k, by, seeds, mem):
    """LB at batch 1 (block K10-B1), paired with the batch-8 arms of the same k and seed; exploratory, no Holm."""
    lr = CONFIG[dataset]['batch1_lr']
    b1 = {}
    for s in seeds:
        r = get(dataset, k, 'LB', lr, s, batch=1)
        if r is not None and ('LB', s) in by:
            # LB-b1 freezes (and so quantizes) the same layers as LB-b8, so all three hashes must match
            assert same_setup(r, by[('LB', s)]), 'seed %d: LB-b1 and LB-b8 are not paired' % s
            b1[('LB-b1', s)] = r
    if not b1:
        return {}
    rs = list(b1.values())
    xs = [r['test_top1'] for r in rs]
    print('batch 1 (block K10-B1: LB at lr %g; exploratory, no Holm): n=%d test %.2f +- %.2f  val %.2f  train %.1f  '
          '%.1f MiB' % (lr, len(xs), mean(xs), stats.tstd(xs) if len(xs) > 1 else float('nan'),
                        mean([r['final_val_top1'] for r in rs]), mean([r['final_train_top1'] for r in rs]),
                        mem[('LB', 1)]['measured_MiB']))
    both = {**by, **b1}
    d, _ = paired(both, 'LB-b1', 'LB', seeds)
    if len(d) >= 2:
        c = ci(d)
        primary = 'catastrophic' if c['U'] < -CATASTROPHIC else \
            'not catastrophic' if c['L'] > -CATASTROPHIC else 'undetermined'
        print('  H2  rule, LB-b1 - LB-b8: %s  per seed %s' % (fmt(c), ' '.join('%+.2f' % x for x in d)))
        print('      %s (%g pp); %s' % (primary, CATASTROPHIC, 'within 2 pp' if c['L'] > -MARGIN else 'not shown within 2 pp'))
    for a, tie in (('T4', 'tie (goes to T4)'), ('T3', 'tie')):
        d, _ = paired(both, 'LB-b1', a, seeds)
        if len(d) >= 2:
            c = ci(d)
            verdict = 'batch better' if c['L'] > 0 else 'depth better' if c['U'] < 0 else tie
            print('  LB-b1 (%.1f MiB) - %s-b8 (%.1f MiB): %s -> %s%s' % (
                mem[('LB', 1)]['measured_MiB'], a, mem[(a, 8)]['measured_MiB'], fmt(c), verdict,
                ' (H2b rule)' if a == 'T4' else ''))
    mins = lambda rows: mean([r['train_seconds'] for r in rows]) / 60
    print('  training time per run (min, wall clock on our GPU): LB-b1 %.1f | LB-b8 %.1f | T4-b8 %.1f; GPU at the start '
          'of the batch-1 runs: %s' % (
              mins(rs), mins([by[('LB', s)] for _, s in b1]), mins([by[('T4', s)] for _, s in b1 if ('T4', s) in by]),
              '; '.join(r['gpu_util_at_start'] for r in rs)))
    return b1


def trend(dataset, blocks):
    print('\nk trend for %s (descriptive):' % dataset)
    print('  k   LB test        d_T4 [93% CI]          d_last [93% CI]        R_T4 [93% CI]       retains after Holm')
    for k in (2, 5, 10):
        b = blocks.get((dataset, k))
        if not b or 'LB' not in b['arms']:
            continue
        by, seeds = b['by'], b['seeds']
        lb = [by[('LB', s)]['test_top1'] for s in seeds if ('LB', s) in by]
        cell = lambda a: ('%+.2f [%+.2f, %+.2f]' % (b['res'][a]['d'], b['res'][a]['L'], b['res'][a]['U'])
                          if a in b['res'] else 'n/a')
        ss = [s for s in seeds if all((x, s) in by for x in ('T4', 'last', 'LB'))]
        r = 'n/a'
        if len(ss) >= 2:
            num = [by[('T4', s)]['test_top1'] - by[('last', s)]['test_top1'] for s in ss]
            den = [by[('LB', s)]['test_top1'] - by[('last', s)]['test_top1'] for s in ss]
            lo, hi = boot_ratio(num, den)
            r = '%.1f%% [%.1f, %.1f]' % (100 * mean(num) / mean(den), 100 * lo, 100 * hi)
        keep = [a for a in b['holm'] if b['holm'][a]]
        print('  %-3d %6.2f (n=%2d)  %-22s %-22s %-20s %s' % (
            k, mean(lb), len(lb), cell('T4'), cell('last'), r, '/'.join(keep) or 'none'))


def step_control(dataset, lr):
    print('\n=== Q4: %s k=2 step-matched control (block S, seed 0, %d epochs vs 50) ===' % (dataset, STEP_EPOCHS))
    flagged, missing = [], []
    for a in STEP_ARMS:
        long = get(dataset, 2, a, lr[a], 0, STEP_EPOCHS, STEP_VAL_EVERY)
        short = get(dataset, 2, a, lr[a], 0)
        if long is None or short is None:
            print('  %-4s not done yet' % a)
            missing.append(a)
            continue
        gain = long['final_val_top1'] - short['final_val_top1']
        print('  %-4s val %.2f -> %.2f (%+.2f pp) | test %.2f -> %.2f (steps %d -> %d)' % (
            a, short['final_val_top1'], long['final_val_top1'], gain, short['test_top1'], long['test_top1'],
            short['steps'], long['steps']))
        if gain >= 1.0:
            flagged.append('%s %+.1f pp' % (a, gain))
    # One seed, so this is an indication only: paired per-seed spread on test is 0.5-0.8 pp at k=5.
    if flagged:
        print('  -> single-seed indication that k=2 is limited by the 50-epoch budget (step-matched val gain: %s); '
              'k=2 results are labelled "at the 50-epoch budget"' % ', '.join(flagged))
    elif missing:
        print('  -> incomplete (missing: %s)' % ', '.join(missing))
    else:
        print('  -> no arm gains >= 1 pp on val (single seed): no budget label for k=2')


def effect_label(c):
    if c['L'] > 0:
        return 'imprint helps'
    if c['U'] < 0:
        return 'imprint hurts'
    if c['L'] > -EFFECT_MARGIN and c['U'] < EFFECT_MARGIN:
        return 'within +-%g pp' % EFFECT_MARGIN
    return 'inconclusive'


def same_setup(a, b):
    return all(a[key] == b[key] for key in ('subset_sha1', 'head_init_sha1', 'frozen_sha1'))


def imprint_block(dataset, lr, blocks, macs):
    print('\n=== Q2: %s imprinted vs random head (blocks I, I-LB, I-K10: 50 epochs; e = imprint - random, paired) ==='
          % dataset)
    for k, arms in IMPRINT_CELLS.items():
        base = blocks.get((dataset, k))
        imp_by_arm = {}
        for a in arms:
            pairs = []
            for s in SEEDS:
                imp, rnd = get(dataset, k, a, lr[a], s, head_init='imprint'), get(dataset, k, a, lr[a], s)
                if imp is not None and rnd is not None:
                    assert same_setup(imp, rnd), '%s k=%d seed %d: imprint and random runs are not paired' % (a, k, s)
                    pairs.append((s, imp, rnd))
            imp_by_arm[a] = {s: imp for s, imp, _ in pairs}
            if len(pairs) < 2:
                print('  k=%d %-4s %d pairs so far' % (k, a, len(pairs)))
                continue
            c = ci([imp['test_top1'] - rnd['test_top1'] for _, imp, rnd in pairs])
            early = ' '.join('val@%d %+.2f' % (e, mean([val_at(imp, e) - val_at(rnd, e) for _, imp, rnd in pairs]))
                             for e in (5, 10))
            st = [imp['imprint'] for _, imp, _ in pairs]
            line = '  k=%d %-4s %s  %-16s | %s | before training: val %.1f%% (random head %.1f%%), head row norm ' \
                   '%.3g (random %.3g), tau %.3g%s, LOO top-1 %.1f%%' % (
                       k, a, fmt(c), effect_label(c), early, mean([x['val_top1_imprinted_head'] for x in st]),
                       mean([x['val_top1_random_head'] for x in st]), mean([x['row_norm_imprinted'] for x in st]),
                       mean([x['row_norm_random'] for x in st]), mean([x['tau'] for x in st]),
                       ' (AT GRID EDGE)' if any(x['tau_at_grid_edge'] for x in st) else '',
                       mean([x['loo_top1'] for x in st]))
            if base and a != 'LB':
                ss = [s for s, _, _ in pairs if ('LB', s) in base['by'] and ('last', s) in base['by']]
                imp_by = {s: imp for s, imp, _ in pairs}
                if len(ss) >= 2:
                    num = [imp_by[s]['test_top1'] - base['by'][('last', s)]['test_top1'] for s in ss]
                    den = [base['by'][('LB', s)]['test_top1'] - base['by'][('last', s)]['test_top1'] for s in ss]
                    lo, hi = boot_ratio(num, den)
                    line += ' | R with imprint %.1f%% [%.1f, %.1f]' % (100 * mean(num) / mean(den), 100 * lo, 100 * hi)
            print(line + (' (exploratory, added later)' if (k, a) in LATE_IMPRINT else ''))
        # Does T4 still retain once imprinting is used: against LB with an imprinted head where it ran, else against LB
        if 'LB' in arms:
            ref, ref_name = imp_by_arm['LB'], 'LB-imprint'
        else:
            ref, ref_name = {s: r for (a, s), r in base['by'].items() if a == 'LB'} if base else {}, 'LB'
        ss = [s for s in imp_by_arm.get('T4', {}) if s in ref]
        if len(ss) >= 2:
            # LB freezes (and so quantizes) different layers than T4, so only the subset and random head must match it
            assert all(imp_by_arm['T4'][s][key] == ref[s][key] for s in ss for key in ('subset_sha1', 'head_init_sha1'))
            c = ci([imp_by_arm['T4'][s]['test_top1'] - ref[s]['test_top1'] for s in ss])
            print('  k=%d T4-imprint - %s: %s  %s (exploratory, no Holm)' % (k, ref_name, fmt(c), label(c['L'], c['U'])))

    print('\n=== Q2: %s short budget (block I10: T4, k=5, %d epochs) ===' % (dataset, SHORT_EPOCHS))
    rows = []
    for s in SEEDS:
        rnd = get(dataset, 5, 'T4', lr['T4'], s, SHORT_EPOCHS)
        imp = get(dataset, 5, 'T4', lr['T4'], s, SHORT_EPOCHS, head_init='imprint')
        lb = get(dataset, 5, 'LB', lr['LB'], s)
        if rnd is not None and imp is not None and lb is not None:
            # LB freezes (and so quantizes) different layers than T4, so only the subset and random head must match it
            assert same_setup(imp, rnd), 'seed %d: I10 imprint and random runs are not paired' % s
            assert all(rnd[key] == lb[key] for key in ('subset_sha1', 'head_init_sha1')), 'seed %d: I10 vs LB' % s
            rows.append((rnd, imp, lb))
    if len(rows) < 2:
        print('  %d complete seeds so far' % len(rows))
        return
    c = ci([imp['test_top1'] - rnd['test_top1'] for rnd, imp, _ in rows])
    print('  e10 = T4-imprint-%d - T4-random-%d: %s  %s' % (SHORT_EPOCHS, SHORT_EPOCHS, fmt(c), effect_label(c)))
    for name, idx in (('T4-random-%d' % SHORT_EPOCHS, 0), ('T4-imprint-%d' % SHORT_EPOCHS, 1)):
        c = ci([r[idx]['test_top1'] - r[2]['test_top1'] for r in rows])
        print('  %s - LB-50: %s  %s (exploratory, no Holm)' % (name, fmt(c), label(c['L'], c['U'])))
    print('  training compute: T4 at %d epochs %.1f T vs LB at 50 epochs %.1f T (%.2fx); imprinting adds one forward '
          'pass over the k-shot images (%.2f T)' % (
              SHORT_EPOCHS, train_tmacs(macs, dataset, 'T4', 5, SHORT_EPOCHS), train_tmacs(macs, dataset, 'LB', 5, 50),
              train_tmacs(macs, dataset, 'T4', 5, SHORT_EPOCHS) / train_tmacs(macs, dataset, 'LB', 5, 50),
              macs[dataset]['T4']['fwd'] * 1e6 * 5 * (102 if dataset == 'flowers102' else 100) / 1e12))


def main():
    mem = memory()
    macs = json.loads((ROOT / 'macs_table.json').read_text(encoding='utf-8'))['table']
    lr = supp_lrs()
    lr_extension()
    blocks = {}
    for ds in DATASETS:
        print('\n=== Q1: %s, k in {2, 5, 10} (k=5 repeats the pre-registered H1 unless block X changed an LR) ===' % ds)
        for k in (2, 5, 10):
            blocks[(ds, k)] = depth_block(ds, k, lr[ds], mem, macs)
        trend(ds, blocks)
        step_control(ds, lr[ds])
        imprint_block(ds, lr[ds], blocks, macs)
    print('\n' + HEADROOM_NOTE)


if __name__ == '__main__':
    main()
