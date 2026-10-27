"""Protocol bridge (实验设计/最终实验设计.md §2, item 8). Documentation only; it decides nothing.

The old r2 runs (fewshot_resolution.py: official r=2 lite residuals, k=5, seeds 0-2, LR 2e-4, val every epoch) were
tested at their best-on-val checkpoint; the depth experiment tests the final epoch. This tests both saved checkpoints
of each old run on the Flowers test split with the old run's own config:
  model_best.pth.tar   must reproduce the test_top1 stored in that run's result.json (a check of this pipeline)
  checkpoint.pth.tar   the epoch-50 model (asserted from its 'epoch' field)

  python eval_final_ckpt.py           GPU, about 3 min; writes bridge_result.json
  python eval_final_ckpt.py --check   CPU: only builds the net and loads every checkpoint strictly
"""
import argparse
import json
import os
import sys

from depth_common import ROOT, setup_paths, build_net

SEEDS = (0, 1, 2)


def old_run(seed):
    return ROOT / 'runs' / ('fewshot-k5-r2-seed%d' % seed)


def load(seed, which):
    import torch
    ckpt = torch.load(old_run(seed) / 'checkpoint' / which, map_location='cpu', weights_only=False)
    if which == 'checkpoint.pth.tar':
        assert ckpt['epoch'] == 49, 'seed %d: checkpoint.pth.tar is epoch index %s, not 49' % (seed, ckpt['epoch'])
    return ckpt['state_dict']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true')
    p.add_argument('--allow_busy', action='store_true', help='run although a sweep holds runs/.sweep.lock')
    args = p.parse_args()
    if not args.check and (ROOT / 'runs' / '.sweep.lock').exists() and not args.allow_busy:
        sys.exit('a sweep is running (runs/.sweep.lock): rerun when the GPU is idle, or pass --allow_busy')
    os.environ['TINYTL_CACHE'] = str(ROOT / 'cache')
    os.environ['TORCH_HOME'] = str(ROOT / 'cache' / 'torch')
    setup_paths()
    from ofa.utils import replace_conv2d_with_my_conv2d, list_mean
    net = build_net(102)  # the official insertion equals fewshot_resolution's r=2 insertion (review r1)
    replace_conv2d_with_my_conv2d(net, 1e-5)
    if args.check:
        for seed in SEEDS:
            for which in ('model_best.pth.tar', 'checkpoint.pth.tar'):
                net.load_state_dict(load(seed, which))  # strict: every key and shape must match
        print('ok: %d checkpoints load strictly into the rebuilt r2 net' % (2 * len(SEEDS)))
        return

    from ofa.imagenet_classification.run_manager import RunManager
    from tinytl.data_providers import FGVCRunConfig
    from fewshot_resolution import test_root
    os.environ['TINYTL_FLOWERS_PATH'] = str(test_root(ROOT / 'fewshot'))
    results = []
    for seed in SEEDS:
        old = json.loads((old_run(seed) / 'result.json').read_text(encoding='utf-8'))
        cfg = json.loads((old_run(seed) / 'run.config').read_text(encoding='utf-8'))
        cfg = {k: v for k, v in cfg.items() if k in FGVCRunConfig.__init__.__code__.co_varnames}
        cfg['n_epochs'] = 1
        run_manager = RunManager(str(ROOT / 'runs' / ('bridge-fewshot-k5-r2-seed%d' % seed)), net,
                                 FGVCRunConfig(**cfg), init=False)
        row = dict(seed=seed, best_epoch=old['best_epoch'], best_val_top1=old['best_val_top1'],
                   final_val_top1=old['final_val_top1'], stored_best_test_top1=old['test_top1'])
        for which, key in (('model_best.pth.tar', 'best'), ('checkpoint.pth.tar', 'final')):
            run_manager.network.load_state_dict(load(seed, which))
            _, loss, top1, top5 = run_manager.validate_all_resolution(is_test=True)
            row['%s_test_top1' % key] = float(list_mean(top1))
            row['%s_test_loss' % key] = float(list_mean(loss))
        row['reproduces_stored'] = abs(row['best_test_top1'] - row['stored_best_test_top1']) < 0.1
        row['final_minus_best'] = row['final_test_top1'] - row['best_test_top1']
        results.append(row)
        print('seed %d: best-val ckpt (epoch %d) test %.2f (stored %.2f, %s) | final epoch test %.2f | final - best %+.2f'
              % (seed, row['best_epoch'], row['best_test_top1'], row['stored_best_test_top1'],
                 'reproduced' if row['reproduces_stored'] else 'NOT REPRODUCED', row['final_test_top1'],
                 row['final_minus_best']), flush=True)
    mean = sum(r['final_minus_best'] for r in results) / len(results)
    print('mean final - best-val: %+.2f pp' % mean)
    out = ROOT / 'bridge_result.json'
    out.write_text(json.dumps(dict(runs=results, mean_final_minus_best=mean), indent=1), encoding='utf-8')
    print('wrote', out)


if __name__ == '__main__':
    main()
