"""H4: Fisher-under-budget as a cutoff chooser, before any training (实验设计/最终实验设计.md §1.5).

  python fisher_stage.py                 both datasets, seeds 0-4, on the GPU; writes fisher_table.json
  python fisher_stage.py --smoke         20 Flowers classes, seed 0, on the CPU: checks the code path, writes nothing
                                         (with only a few classes the head separates the support set and the
                                         Fisher vanishes; on the full sets the leave-one-out head is not saturated)

Per dataset and seed, on the k=5 training images with the test transform (no augmentation):
  1. the pretrained net with weight standardization and fp32 weights. Frozen weights are not k-means quantized: which
     layers get quantized depends on the arm, which is what is being chosen;
  2. a ProtoNet-equivalent head with fixed prototypes mu_c (class means of the 1280-d pooled features, detached):
     logits 2 f.mu_c - |mu_c|^2, where the image's own class uses the prototype without that image (leave-one-out);
     plain cross-entropy, no label smoothing. Support loss and accuracy are logged;
  3. TinyTrain Eq. 2 on every block output a: Delta_c = 1/(2N) sum_n (sum_hw a_nchw g_nchw)^2, P_block = sum_c Delta_c;
  4. groups matching the run cutoffs: G2 = blocks 9-12 (stage 2), G3 = 13-16 (stage 3), G45 = 17-21 (stages 4-5);
     P_group = sum of its blocks;
  5. Eq. 3: s = P / ((W / max W) x (M / max M)), with W the parameters the group adds to training and M the backward
     MACs it adds (measure_macs.count_macs);
  6. pre-registered prefix-greedy selection: the longest prefix of the groups ranked by s whose truncation (the arm
     starting at its earliest group) fits the budget in measured batch-8 MiB (memory_table.json); an empty prefix
     means FT-Last.
Then hit rate and regret on the main-run test accuracy against two oracles (the per-seed best run arm within the
budget, and the arm with the best mean), for Fisher and for the static rules "deepest feasible" and "always FT-Last".

Found while implementing: every candidate truncation fits 256 MiB (T2 is 226.5 MiB) and a prefix's memory is set by
its earliest group, so at 256 MiB the pre-registered rule selects all three groups (T2, the same as "deepest
feasible") whatever the scores; only 128 MiB can discriminate (T4 if G45 ranks first, else FT-Last). The rule `top1`
(the highest-scored group that fits on its own) is printed beside it and labelled as not pre-registered.
"""
import argparse
import csv
import json
import sys
import time

from depth_common import ROOT, setup_paths, build_net, set_trainable, block_index
from measure_macs import count_macs

DATASETS = {'flowers102': 102, 'aircraft': 100}
K = 5
GROUPS = {'G2': ('T2', range(9, 13)), 'G3': ('T3', range(13, 17)), 'G45': ('T4', range(17, 22))}
RUN_ARMS = ['T2', 'T3', 'T4', 'last']
DEPTH = {'T2': 3, 'T3': 2, 'T4': 1, 'last': 0}
BUDGETS = (128, 256)
USEFUL_DEEPER = 0.7
SMOKE_CLASSES = 20


def support_set(dataset, seed, n_classes_limit=None, k=K):
    import torch
    from torchvision import datasets, transforms
    from fewshot_depth import data_roots
    train_root, _ = data_roots(dataset, k, seed)
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ds = datasets.ImageFolder(str(train_root / 'train'), tf)  # same class order as the upstream provider
    keep = [i for i, t in enumerate(ds.targets) if n_classes_limit is None or t < n_classes_limit]
    xs = torch.stack([ds[i][0] for i in keep])
    ys = torch.tensor([ds.targets[i] for i in keep])
    assert torch.bincount(ys).tolist() == [k] * int(ys.max() + 1), 'not %d images per class' % k
    return xs, ys


def forward_blocks(net, x, keep):
    """Pooled features and, if keep, every block output with its gradient retained."""
    outs = {}
    x = net.first_conv(x)
    for i, block in enumerate(net.blocks):
        x = block(x)
        if keep:
            x.retain_grad()
            outs[i] = x  # blocks 3 and 4 are ZeroLayer: their output is the previous block's tensor
    return net.global_avg_pool(net.feature_mix_layer(x)), outs


def fisher(net, xs, ys, device, chunk):
    import torch
    import torch.nn.functional as F
    n, n_cls = len(ys), int(ys.max() + 1)
    with torch.no_grad():
        f = torch.cat([forward_blocks(net, xs[i:i + chunk].to(device), False)[0] for i in range(0, n, chunk)])
    y_dev = ys.to(device)
    cnt = torch.bincount(y_dev, minlength=n_cls).float()
    mu = torch.zeros(n_cls, f.shape[1], device=device).index_add_(0, y_dev, f) / cnt[:, None]
    acc, per_sample = {}, torch.zeros(n, dtype=torch.float64)
    loss_sum, correct = 0.0, 0
    for i in range(0, n, chunk):
        x = xs[i:i + chunk].to(device).requires_grad_(True)
        y = y_dev[i:i + chunk]
        feat, outs = forward_blocks(net, x, True)
        logits = 2 * feat @ mu.t() - (mu ** 2).sum(1)
        mu_own = (cnt[y, None] * mu[y] - feat.detach()) / (cnt[y, None] - 1)  # leave-one-out, fixed like the head
        own = 2 * (feat * mu_own).sum(1) - (mu_own ** 2).sum(1)
        logits = logits.scatter(1, y[:, None], own[:, None])
        loss = F.cross_entropy(logits, y, reduction='sum')
        loss.backward()
        loss_sum += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        for b, a in outs.items():
            s = (a.detach() * a.grad).sum(dim=(2, 3)).double()  # (batch, channels)
            acc[b] = acc.get(b, 0) + (s ** 2).sum(0)
            if any(b in r for _, r in GROUPS.values()):
                per_sample[i:i + len(y)] += (s ** 2).sum(1).cpu()
    potential = {b: float(v.sum() / (2 * n)) for b, v in acc.items()}
    top = per_sample.sort(descending=True).values
    support = dict(loss=loss_sum / n, top1=100.0 * correct / n,
                   top10_images_share=float(top[:10].sum() / top.sum()), max_image_share=float(top[0] / top.sum()))
    return potential, support


def group_costs(n_classes):
    """Parameters and backward MACs (M) that each group adds to training."""
    net = build_net(n_classes)
    set_trainable(net, 'LB')
    params = {g: sum(p.numel() for name, p in net.named_parameters() if p.requires_grad and block_index(name) in r)
              for g, (_, r) in GROUPS.items()}
    bwd = {a: count_macs(a, n_classes)['bwd_total'] / 1e6 for a in RUN_ARMS}
    macs = {'G2': bwd['T2'] - bwd['T3'], 'G3': bwd['T3'] - bwd['T4'], 'G45': bwd['T4'] - bwd['last']}
    return params, macs


def score(potential, params, macs):
    p = {g: sum(potential[b] for b in r) for g, (_, r) in GROUPS.items()}
    w_max, m_max = max(params.values()), max(macs.values())
    s = {g: p[g] / ((params[g] / w_max) * (macs[g] / m_max)) for g in GROUPS}
    return p, s, sorted(GROUPS, key=lambda g: -s[g])


def arm_of(groups):
    return min((GROUPS[g][0] for g in groups), key=lambda a: -DEPTH[a]) if groups else 'last'


def prefix_greedy(ranking, mem, budget):
    chosen = []
    for g in ranking:
        if mem[arm_of(chosen + [g])] > budget:
            break
        chosen.append(g)
    return arm_of(chosen)


def top1(ranking, mem, budget):
    return next((GROUPS[g][0] for g in ranking if mem[GROUPS[g][0]] <= budget), 'last')


def main_accuracy(dataset):
    path = ROOT / 'runlists' / ('main_%s.csv' % dataset)
    acc = {}
    if path.exists():
        for r in csv.DictReader(open(path, encoding='utf-8')):
            res = ROOT / 'runs' / r['name'] / 'result.json'
            if res.exists():
                acc[(r['arm'], int(r['seed']))] = json.loads(res.read_text(encoding='utf-8'))['test_top1']
    return acc


def evaluate(choices, acc, mem, seeds):
    """choices[rule][budget][seed] -> arm. Hit rate and mean regret vs the per-seed and the best-mean oracle."""
    out = {}
    for budget in BUDGETS:
        feasible = [a for a in RUN_ARMS if mem[a] <= budget]
        ss = [s for s in seeds if all((a, s) in acc for a in feasible)]
        if not ss:
            continue
        best_mean = max(feasible, key=lambda a: sum(acc[(a, s)] for s in ss))
        rules = dict(choices, deepest={s: max(feasible, key=DEPTH.get) for s in ss},
                     always_last={s: 'last' for s in ss})
        out[budget] = dict(feasible=feasible, seeds=ss, best_mean_arm=best_mean, rules={})
        for rule, by_seed in rules.items():
            by_seed = by_seed[budget] if rule in choices else by_seed
            r1 = [max(acc[(a, s)] for a in feasible) - acc[(by_seed[s], s)] for s in ss]
            r2 = [acc[(best_mean, s)] - acc[(by_seed[s], s)] for s in ss]
            hit1 = [by_seed[s] == max(feasible, key=lambda a: acc[(a, s)]) for s in ss]
            out[budget]['rules'][rule] = dict(
                choices=[by_seed[s] for s in ss], regret_per_seed_oracle=sum(r1) / len(ss),
                hit_per_seed_oracle=sum(hit1) / len(ss), regret_best_mean_oracle=sum(r2) / len(ss),
                hit_best_mean_oracle=sum(by_seed[s] == best_mean for s in ss) / len(ss))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(5)))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--chunk', type=int, default=32)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--allow_busy', action='store_true', help='use the GPU although a sweep holds runs/.sweep.lock')
    args = ap.parse_args()
    if args.smoke:
        args.datasets, args.seeds, args.device, args.chunk = ['flowers102'], [0], 'cpu', 5
    if args.device != 'cpu' and (ROOT / 'runs' / '.sweep.lock').exists() and not args.allow_busy:
        sys.exit('a sweep is running (runs/.sweep.lock): rerun when the GPU is idle, or pass --allow_busy')
    setup_paths()
    import torch
    from ofa.utils import replace_conv2d_with_my_conv2d
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    mem = {r['arm']: r['measured_MiB'] for r in json.loads((ROOT / 'memory_table.json').read_text())['table']
           if r['bs'] == 8 and r['arm'] in RUN_ARMS}
    table = dict(meta=dict(budgets_MiB=list(BUDGETS), memory_MiB_b8=mem, k=K, device=str(device),
                           groups={g: [a, list(r)] for g, (a, r) in GROUPS.items()}), datasets={})
    for dataset in args.datasets:
        n_classes = DATASETS[dataset]
        params, macs = group_costs(n_classes)
        net = build_net(n_classes)
        for p in net.parameters():
            p.requires_grad = False
        replace_conv2d_with_my_conv2d(net, 1e-5)
        net = net.to(device).eval()
        per_seed, choices = {}, {'fisher': {b: {} for b in BUDGETS}, 'top1': {b: {} for b in BUDGETS}}
        print('\n=== %s: group params %s, added backward M MACs %s' % (
            dataset, params, {g: round(v, 1) for g, v in macs.items()}))
        for seed in args.seeds:
            t = time.time()
            xs, ys = support_set(dataset, seed, SMOKE_CLASSES if args.smoke else None)
            potential, support = fisher(net, xs, ys, device, args.chunk)
            p, s, ranking = score(potential, params, macs)
            row = dict(support=support, block_potential=potential, group_potential=p, score=s, ranking=ranking,
                       seconds=time.time() - t)
            for b in BUDGETS:
                row['fisher_%d' % b] = choices['fisher'][b][seed] = prefix_greedy(ranking, mem, b)
                row['top1_%d' % b] = choices['top1'][b][seed] = top1(ranking, mem, b)
            per_seed[seed] = row
            print('seed %d: support loss %.3f top1 %.1f%% (top-10 images carry %.0f%% of the Fisher) | P %s | s %s'
                  ' | ranking %s | 128: %s (top1 %s) | 256: %s (top1 %s) | %.0f s' % (
                      seed, support['loss'], support['top1'], 100 * support['top10_images_share'],
                      {g: '%.3g' % v for g, v in p.items()}, {g: '%.3g' % v for g, v in s.items()},
                      '>'.join(ranking), row['fisher_128'], row['top1_128'], row['fisher_256'], row['top1_256'],
                      row['seconds']), flush=True)
        if args.smoke:
            print('smoke ok (%d images, %d classes); nothing written' % (len(ys), SMOKE_CLASSES))
            return
        result = evaluate(choices, main_accuracy(dataset), mem, args.seeds)
        table['datasets'][dataset] = dict(group_params=params, group_bwd_MMACs=macs, per_seed=per_seed,
                                          evaluation=result)
        for budget, e in result.items():
            print('%d MiB (feasible %s, best-mean arm %s):' % (budget, e['feasible'], e['best_mean_arm']))
            for rule, r in e['rules'].items():
                print('  %-12s %-28s regret %.2f / %.2f pp, hit %.0f%% / %.0f%% (per-seed / best-mean oracle)%s' % (
                    rule, ' '.join(r['choices']), r['regret_per_seed_oracle'], r['regret_best_mean_oracle'],
                    100 * r['hit_per_seed_oracle'], 100 * r['hit_best_mean_oracle'],
                    '  [not pre-registered]' if rule == 'top1' else ''))
    if set(DATASETS) <= set(table['datasets']):
        f, a = (table['datasets'][d]['per_seed'] for d in ('flowers102', 'aircraft'))
        common = sorted(set(f) & set(a))
        verdict = {}
        for rule in ('fisher', 'top1'):
            deeper = sum(DEPTH[a[s]['%s_256' % rule]] > DEPTH[f[s]['%s_256' % rule]] for s in common) / len(common)
            no_worse = all(e['rules'][rule]['regret_per_seed_oracle'] <= min(
                e['rules']['deepest']['regret_per_seed_oracle'], e['rules']['always_last']['regret_per_seed_oracle'])
                           for d in DATASETS for e in table['datasets'][d]['evaluation'].values())
            verdict[rule] = dict(deeper_on_aircraft_256=deeper, regret_no_worse_than_best_static=no_worse,
                                 reading='useful' if no_worse and deeper >= USEFUL_DEEPER else 'no advantage')
            print('%s: deeper on Aircraft at 256 MiB in %.0f%% of seeds; regret <= the better static rule on both '
                  'datasets and budgets: %s -> %s%s' % (rule, 100 * deeper, no_worse, verdict[rule]['reading'],
                                                        '  [not pre-registered]' if rule == 'top1' else ''))
        table['verdict'] = verdict
    out = ROOT / 'fisher_table.json'
    out.write_text(json.dumps(table, indent=1), encoding='utf-8')
    print('wrote', out)


if __name__ == '__main__':
    main()
