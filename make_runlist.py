"""Run lists for run-depth-sweep.sh, and the learning-rate freeze step (实验设计/最终实验设计.md §1.2, §1.4).

  python make_runlist.py smoke                 runlists/smoke.csv: every Flowers arm, k=2, 1 epoch (+ LB at batch 1)
  python make_runlist.py pilot  [--dataset D]  runlists/pilot_<D>.csv: {LB, T3, last} x LR grid x pilot seeds
  python make_runlist.py freeze [--dataset D]  lr_choice_<D>.json from the pilot results, plus the half-budget
                                               convergence check (writes runlists/halfbudget_<D>.csv if it fires)
  python make_runlist.py main   [--dataset D]  runlists/main_<D>.csv: the A block, seed-major, LRs from the frozen file
  python make_runlist.py batch1 --dataset aircraft            runlists/batch1_aircraft.csv: C-Air, LB at batch 1
  python make_runlist.py lrcheck --dataset aircraft           runlists/lrcheck_aircraft.csv: conditional 2

Pre-registered cuts (§3): `main --n_seeds 9` (Flowers) or 5 (Aircraft); `pilot/freeze --drop_lr 3.2e-3` removes that
LR from the LB and T3 pilot (pass the same option to both).

Supplementary experiments (实验设计/补充实验设计.md):
  python make_runlist.py lrx          runlists/lrx_aircraft.csv: block X, LB and T3 below the Aircraft LR grid
  python make_runlist.py lrx_freeze   lr_choice_aircraft_supp.json, plus runlists/lrrerun_aircraft.csv (block X-R) if
                                      an LR changed
  python make_runlist.py supp         runlists/supp.csv: K2, S, I, K10, I10 for both datasets, in that order
  python make_runlist.py smoke_supp   runlists/smoke_supp.csv: 1-epoch runs of the new code paths
  python make_runlist.py k10t3        runlists/k10t3_aircraft.csv: block K10-T3, T3 at k=10 on Aircraft (added
                                      after the K10 results; exploratory)
"""
import argparse
import csv
import json
import math
import sys

from depth_common import ROOT, run_name

CONFIG = {
    'flowers102': dict(upstream_lr=2e-4, grid=[2e-4, 8e-4, 3.2e-3], pilot_seeds=[100, 101],
                       main=[('LB', 10), ('T3', 10), ('T4', 10), ('last', 10), ('T2', 5)]),
    'aircraft': dict(upstream_lr=1.6e-3, grid=[4e-4, 1.6e-3, 6.4e-3], pilot_seeds=[100],
                     main=[('LB', 6), ('T2', 6), ('T3', 6), ('T4', 6), ('last', 5)],
                     batch1_lr=3e-4, batch1_seeds=5, lrcheck_lrs=[1e-4, 1.2e-3]),
}
PILOT_ARMS = ['LB', 'T3', 'last']
DROP_ARMS = ['LB', 'T3']  # the pre-registered cut drops an LR from these two pilots only
INHERIT_T3 = ['T4', 'T2']
MIN_GAIN = 1.0       # pp a non-upstream LR must win by
HALF_BUDGET = 1.5    # pp of val gain from epochs {20,25} to {45,50} that triggers the 100-epoch check
COLUMNS = ['block', 'dataset', 'k', 'arm', 'batch', 'lr', 'seed', 'epochs', 'val_every', 'tag', 'name', 'head_init']
LISTS = ROOT / 'runlists'

# 实验设计/补充实验设计.md
SUPP_LR_FILE = ROOT / 'lr_choice_aircraft_supp.json'
LRX = dict(arms=['LB', 'T3'], lrs=[1e-4, 2e-4])
SUPP_SEEDS = 5
K2_ARMS, K10_ARMS, STEP_ARMS, IMPRINT_ARMS = ['LB', 'T3', 'T4', 'last'], ['LB', 'T4', 'last'], ['LB', 'T4', 'last'], \
    ['T4', 'last']
STEP_EPOCHS, STEP_VAL_EVERY, SHORT_EPOCHS = 125, 25, 10


def lr_file(dataset):
    return ROOT / ('lr_choice_%s.json' % dataset)


def row(block, dataset, arm, lr, seed, k=5, batch=8, epochs=50, val_every=5, tag='', head_init='random'):
    return dict(block=block, dataset=dataset, k=k, arm=arm, batch=batch, lr=repr(lr), seed=seed, epochs=epochs,
                val_every=val_every, tag=tag,
                name=run_name(dataset, k, arm, batch, lr, seed, epochs, val_every, tag, head_init),
                head_init=head_init)


def write(path, rows):
    LISTS.mkdir(exist_ok=True)
    assert len({r['name'] for r in rows}) == len(rows), 'duplicate run names'
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, COLUMNS, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print('wrote %s (%d runs)' % (path, len(rows)))


def load_result(r):
    path = ROOT / 'runs' / r['name'] / 'result.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def pilot_grid(dataset, arm, drop_lr):
    grid = CONFIG[dataset]['grid']
    return [lr for lr in grid if not (drop_lr is not None and math.isclose(lr, drop_lr) and arm in DROP_ARMS)]


def pilot_rows(dataset, drop_lr):
    c = CONFIG[dataset]
    return [row('P', dataset, arm, lr, seed) for seed in c['pilot_seeds'] for arm in PILOT_ARMS
            for lr in pilot_grid(dataset, arm, drop_lr)]


def half_budget_gain(curve):
    val = {e: v for e, v, _ in curve}
    return (val[45] + val[50]) / 2 - (val[20] + val[25]) / 2


def freeze(dataset, drop_lr, force):
    c, path = CONFIG[dataset], lr_file(dataset)
    if path.exists() and not force:
        sys.exit('%s already exists; it is frozen before the main runs (use --force to rebuild it)' % path)
    results = {}
    for r in pilot_rows(dataset, drop_lr):
        res = load_result(r)
        if res is None:
            sys.exit('pilot run missing: %s' % r['name'])
        results[(r['arm'], float(r['lr']), r['seed'])] = res
    up, choice, report = c['upstream_lr'], {}, {}
    for arm in PILOT_ARMS:
        grid = pilot_grid(dataset, arm, drop_lr)
        score = {}
        for lr in grid:
            runs = [results[(arm, lr, s)] for s in c['pilot_seeds']]
            vals = [r['final_val_top1'] for r in runs]
            diverged = any(r.get('diverged') or not math.isfinite(v) for r, v in zip(runs, vals))
            score[lr] = -math.inf if diverged else sum(vals) / len(vals)
        best = max(grid, key=lambda lr: (score[lr], -grid.index(lr)))  # ties go to the earlier grid entry
        chosen = best if best == up or score[best] - score[up] >= MIN_GAIN else up
        gains = [half_budget_gain(results[(arm, chosen, s)]['val_curve']) for s in c['pilot_seeds']]
        choice[arm] = chosen
        report[arm] = dict(mean_final_val={repr(k): (None if math.isinf(v) else round(v, 3)) for k, v in score.items()},
                           raw_best=best, chosen=chosen, raw_best_at_grid_edge=best in (grid[0], grid[-1]),
                           chosen_at_grid_edge=chosen in (grid[0], grid[-1]),
                           half_budget_gain_pp=round(sum(gains) / len(gains), 3))
    for arm in INHERIT_T3:
        choice[arm] = choice['T3']
    flagged = [a for a in PILOT_ARMS if report[a]['half_budget_gain_pp'] > HALF_BUDGET]
    rule = ('mean final-epoch val over seeds %s (a diverged LR scores -inf); a non-upstream LR must beat %g by >= %g pp;'
            ' T4 and T2 take T3\'s LR' % (c['pilot_seeds'], up, MIN_GAIN))
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(dataset=dataset, lr=choice, pilot=report, half_budget_flagged=flagged,
                                   drop_lr=drop_lr, rule=rule), indent=2), encoding='utf-8')
    tmp.replace(path)
    print(json.dumps(report, indent=2))
    print('chosen:', choice)
    if flagged:
        arm = max(flagged, key=lambda a: report[a]['half_budget_gain_pp'])
        write(LISTS / ('halfbudget_%s.csv' % dataset),
              [row('half-budget', dataset, arm, choice[arm], c['pilot_seeds'][0], epochs=100)])
    else:
        print('half-budget check: no arm above %.1f pp, no 100-epoch run' % HALF_BUDGET)


def frozen_lr(dataset):
    path = lr_file(dataset)
    if not path.exists():
        sys.exit('%s missing: run the pilot and `make_runlist.py freeze --dataset %s` first' % (path, dataset))
    data = json.loads(path.read_text(encoding='utf-8'))
    assert data['dataset'] == dataset, '%s holds LRs for %s' % (path, data['dataset'])
    return data['lr']


def main_rows(dataset, n_seeds, lr=None, arms=None, block='A'):
    lr, plan = lr or frozen_lr(dataset), CONFIG[dataset]['main']
    rows = []
    for seed in range(max(n for _, n in plan)):  # seed-major, so a partial block stays paired
        for arm, n in plan:
            if (arms is None or arm in arms) and seed < min(n, n_seeds if n_seeds is not None else n):
                rows.append(row(block, dataset, arm, lr[arm], seed))
    return rows


def read_list(name):
    path = LISTS / name
    if not path.exists():
        sys.exit('run list missing: %s' % path)
    return list(csv.DictReader(open(path, encoding='utf-8')))


def lrx_rows():
    seed = CONFIG['aircraft']['pilot_seeds'][0]
    return [row('X', 'aircraft', arm, lr, seed) for arm in LRX['arms'] for lr in LRX['lrs']]


def lrx_freeze(force):
    """Block X decision: extended-grid best by pilot val; switch only if it beats the frozen LR by >= MIN_GAIN."""
    if SUPP_LR_FILE.exists() and not force:
        sys.exit('%s already exists; it is frozen before the supplementary runs (use --force)' % SUPP_LR_FILE)
    base = frozen_lr('aircraft')
    score = {}  # (arm, lr) -> mean final val over pilot seeds; a diverged run scores -inf
    for r in read_list('pilot_aircraft.csv') + lrx_rows():
        if r['arm'] not in LRX['arms']:
            continue
        res = load_result(r)
        if res is None:
            sys.exit('pilot run missing: %s' % r['name'])
        v = res['final_val_top1']
        bad = res.get('diverged') or not math.isfinite(v)
        score.setdefault((r['arm'], float(r['lr'])), []).append(-math.inf if bad else v)
    choice, report = dict(base), {}
    for arm in LRX['arms']:
        grid = sorted(lr for a, lr in score if a == arm)
        mean_val = {lr: sum(score[(arm, lr)]) / len(score[(arm, lr)]) for lr in grid}
        cur = next(lr for lr in grid if math.isclose(lr, base[arm]))
        best = max(grid, key=lambda lr: (mean_val[lr], lr == cur))  # ties keep the frozen LR
        choice[arm] = best if mean_val[best] - mean_val[cur] >= MIN_GAIN else cur
        report[arm] = dict(mean_final_val={repr(lr): (None if math.isinf(v) else round(v, 3))
                                           for lr, v in mean_val.items()},
                           frozen=cur, raw_best=best, chosen=choice[arm],
                           chosen_at_grid_edge=choice[arm] in (grid[0], grid[-1]))
    for arm in INHERIT_T3:
        choice[arm] = choice['T3']
    changed = sorted(a for a in choice if not math.isclose(choice[a], base[a]))
    tmp = SUPP_LR_FILE.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(dataset='aircraft', lr=choice, base=base, changed=changed, pilot=report,
                                   rule='extended grid, mean final val on the pilot seeds; switch only if >= %g pp '
                                        'above the frozen LR; T4 and T2 take T3\'s LR' % MIN_GAIN), indent=2),
                   encoding='utf-8')
    tmp.replace(SUPP_LR_FILE)
    print(json.dumps(report, indent=2))
    print('chosen:', choice, '| changed:', changed or 'none')
    rerun = LISTS / 'lrrerun_aircraft.csv'
    if changed:
        write(rerun, main_rows('aircraft', None, lr=choice, arms=changed, block='X-R'))
    elif rerun.exists():
        rerun.unlink()


def supp_lr(dataset):
    if dataset == 'aircraft':
        if not SUPP_LR_FILE.exists():
            sys.exit('%s missing: run block X and `make_runlist.py lrx_freeze` first' % SUPP_LR_FILE)
        return json.loads(SUPP_LR_FILE.read_text(encoding='utf-8'))['lr']
    return frozen_lr(dataset)


def supp_rows():
    lrs = {ds: supp_lr(ds) for ds in CONFIG}
    seeds = range(SUPP_SEEDS)
    blocks = [
        ('K2', lambda ds, lr: [row('K2', ds, a, lr[a], s, k=2) for s in seeds for a in K2_ARMS]),
        ('S', lambda ds, lr: [row('S', ds, a, lr[a], 0, k=2, epochs=STEP_EPOCHS, val_every=STEP_VAL_EVERY)
                              for a in STEP_ARMS]),
        ('I', lambda ds, lr: [row('I', ds, a, lr[a], s, k=k, head_init='imprint')
                              for k in (2, 5) for s in seeds for a in IMPRINT_ARMS]),
        ('K10', lambda ds, lr: [row('K10', ds, a, lr[a], s, k=10) for s in seeds for a in K10_ARMS]),
        ('I10', lambda ds, lr: [row('I10', ds, 'T4', lr['T4'], s, epochs=SHORT_EPOCHS, head_init=h)
                                for s in seeds for h in ('random', 'imprint')]),
    ]
    return [r for _, make in blocks for ds in CONFIG for r in make(ds, lrs[ds])]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('what', choices=['smoke', 'pilot', 'freeze', 'main', 'batch1', 'lrcheck',
                                    'lrx', 'lrx_freeze', 'supp', 'smoke_supp', 'k10t3'])
    p.add_argument('--dataset', default='flowers102', choices=sorted(CONFIG))
    p.add_argument('--drop_lr', type=float, default=None)
    p.add_argument('--n_seeds', type=int, default=None)
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    if a.what == 'lrx':
        write(LISTS / 'lrx_aircraft.csv', lrx_rows())
    elif a.what == 'lrx_freeze':
        lrx_freeze(a.force)
    elif a.what == 'supp':
        write(LISTS / 'supp.csv', supp_rows())
    elif a.what == 'k10t3':
        lr = supp_lr('aircraft')
        write(LISTS / 'k10t3_aircraft.csv',
              [row('K10T3', 'aircraft', 'T3', lr['T3'], s, k=10) for s in range(SUPP_SEEDS)])
    elif a.what == 'smoke_supp':
        write(LISTS / 'smoke_supp.csv', [
            row('smoke', 'flowers102', 'LB', 8e-4, 0, k=10, epochs=1, val_every=1, tag='smoke4'),
            row('smoke', 'flowers102', 'last', 8e-4, 0, k=2, epochs=1, val_every=1, tag='smoke4', head_init='imprint'),
            row('smoke', 'aircraft', 'T4', 4e-4, 0, k=2, epochs=1, val_every=1, tag='smoke4', head_init='imprint'),
            row('smoke', 'aircraft', 'last', 1.6e-3, 0, k=10, epochs=1, val_every=1, tag='smoke4')])
    elif a.what == 'smoke':
        rows = [row('smoke', 'flowers102', arm, 2e-4, 0, k=2, epochs=1, val_every=1, tag='smoke2')
                for arm in ['LB', 'T2', 'T3', 'T4', 'last']]
        rows.append(row('smoke', 'flowers102', 'LB', 3e-4, 0, k=2, batch=1, epochs=1, val_every=1, tag='smoke2'))
        write(LISTS / 'smoke.csv', rows)
    elif a.what == 'pilot':
        write(LISTS / ('pilot_%s.csv' % a.dataset), pilot_rows(a.dataset, a.drop_lr))
    elif a.what == 'freeze':
        freeze(a.dataset, a.drop_lr, a.force)
    elif a.what == 'main':
        write(LISTS / ('main_%s.csv' % a.dataset), main_rows(a.dataset, a.n_seeds))
    else:
        c = CONFIG[a.dataset]
        if 'batch1_lr' not in c:
            sys.exit('%s has no batch-1 block in the design' % a.dataset)
        if a.what == 'batch1':
            write(LISTS / ('batch1_%s.csv' % a.dataset),
                  [row('C', a.dataset, 'LB', c['batch1_lr'], s, batch=1) for s in range(c['batch1_seeds'])])
        else:
            write(LISTS / ('lrcheck_%s.csv' % a.dataset),
                  [row('lrcheck', a.dataset, 'LB', lr, 0, batch=1) for lr in c['lrcheck_lrs']])
