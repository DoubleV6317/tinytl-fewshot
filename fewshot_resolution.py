"""Few-shot TinyTL with per-stage lite residual pooling ratios.

Differences from the upstream trainer, all deliberate:
  * a real 3-way split, so checkpoints are chosen on validation, never on test;
  * k images per class sampled from the original train split, shared across configs;
  * per-stage pooling ratios, with kernel sizes pinned to the official r=2 choice so
    every configuration keeps identical parameter shapes and loads the same checkpoint.
"""
from pathlib import Path
import argparse
import json
import os
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'flowers102_raw'
STRIDE_STAGES = [2, 2, 2, 1, 2, 1]


def splits():
    from scipy.io import loadmat
    sid = loadmat(RAW / 'setid.mat')
    labels = loadmat(RAW / 'imagelabels.mat')['labels'][0]
    ids = {'train': list(sid['trnid'][0]), 'val': list(sid['valid'][0]), 'test': list(sid['tstid'][0])}
    return ids, labels


def link(src, dst):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def build_root(k, seed, work):
    """Build a dataset root whose train is k images per class and whose val is the real validation split."""
    ids, labels = splits()
    root = work / ('k%d_seed%d' % (k, seed))
    if (root / '.done').exists():
        return root
    byclass = {}
    for i in ids['train']:
        byclass.setdefault(int(labels[i - 1]), []).append(int(i))
    rng = random.Random(seed)
    for cls, items in sorted(byclass.items()):
        for i in sorted(rng.sample(sorted(items), k)):
            link(RAW / 'jpg' / ('image_%05d.jpg' % i), root / 'train' / str(cls) / ('image_%05d.jpg' % i))
    for i in ids['val']:
        i = int(i)
        link(RAW / 'jpg' / ('image_%05d.jpg' % i), root / 'val' / str(int(labels[i - 1])) / ('image_%05d.jpg' % i))
    (root / '.done').write_text('ok', encoding='utf-8')
    return root


def test_root(work):
    ids, labels = splits()
    root = work / 'testset'
    if not (root / '.done').exists():
        for i in ids['test']:
            i = int(i)
            link(RAW / 'jpg' / ('image_%05d.jpg' % i), root / 'val' / str(int(labels[i - 1])) / ('image_%05d.jpg' % i))
        # The provider always builds a train loader; give it one image per class so the
        # directory scan succeeds. It is never iterated during the test pass.
        seen = set()
        for i in ids['train']:
            i = int(i)
            cls = int(labels[i - 1])
            if cls in seen:
                continue
            seen.add(cls)
            link(RAW / 'jpg' / ('image_%05d.jpg' % i), root / 'train' / str(cls) / ('image_%05d.jpg' % i))
        (root / '.done').write_text('ok', encoding='utf-8')
    return root


def insert_lite_residual(net, ratios):
    """Official insertion, but with per-stage ratios and kernel sizes pinned to the r=2 values."""
    from ofa.utils.layers import ZeroLayer
    from tinytl.model import LiteResidualModule
    bn_param = net.get_bn_param()
    max_res = 128
    for g, (idx_list, stride) in enumerate(zip(net.grouped_block_index, STRIDE_STAGES)):
        for i, idx in enumerate(idx_list):
            block = net.blocks[idx].conv
            if isinstance(block, ZeroLayer):
                continue
            s = stride if i == 0 else 1
            res_at_r2 = max(1, max_res // 2)
            ks = 5
            if res_at_r2 == 1:
                ks, r = 1, None
            else:
                while res_at_r2 < ks:
                    ks -= 2
                r = ratios[g]
            max_res //= s
            net.blocks[idx].conv = LiteResidualModule(
                block, block.in_channels, block.out_channels, expand=1.0, kernel_size=ks,
                act_func='relu', n_groups=2, downsample_ratio=r, upsample_type='bilinear', stride=s)
    net.set_bn_param(**bn_param)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ratios', default='2,2,2,2,2,2')
    p.add_argument('--tag', default='r2')
    p.add_argument('--epochs', type=int, default=50)
    args = p.parse_args()
    ratios = [int(x) for x in args.ratios.split(',')]
    assert len(ratios) == 6

    work = ROOT / 'fewshot'
    os.environ['TINYTL_CACHE'] = str(ROOT / 'cache')
    os.environ['TORCH_HOME'] = str(ROOT / 'cache' / 'torch')
    os.environ['OMP_NUM_THREADS'] = '4'
    sys.path[:0] = [str(ROOT / 'upstream' / 'once-for-all'), str(ROOT / 'upstream' / 'tinytl')]

    import numpy as np
    import torch
    torch.set_num_threads(4)
    data_root = build_root(args.k, args.seed, work)
    os.environ['TINYTL_FLOWERS_PATH'] = str(data_root)

    from ofa.utils.layers import LinearLayer
    from ofa.model_zoo import proxylessnas_mobile
    from ofa.imagenet_classification.run_manager import RunManager
    from ofa.utils import init_models, download_url, replace_conv2d_with_my_conv2d, replace_bn_with_gn, list_mean
    from tinytl.data_providers import FGVCRunConfig
    from tinytl.utils import set_module_grad_status, enable_bias_update, weight_quantization, profile_memory_cost
    from tinytl.model import LiteResidualModule

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    out = ROOT / 'runs' / ('fewshot-k%d-%s-seed%d' % (args.k, args.tag, args.seed))
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(n_epochs=args.epochs, init_lr=2e-4, lr_schedule_type='cosine', lr_schedule_param=None,
               dataset='flowers102', train_batch_size=8, test_batch_size=64, valid_size=None,
               opt_type='adam', opt_param={'momentum': 0.9, 'nesterov': True}, weight_decay=0,
               label_smoothing=0.7, no_decay_keys='bn#bias', mixup_alpha=None, model_init='he_fout',
               validation_frequency=1, print_frequency=10, n_worker=0, resize_scale=0.22,
               distort_color='None', image_size=224)
    run_config = FGVCRunConfig(**cfg)

    net = proxylessnas_mobile(pretrained=False)
    insert_lite_residual(net, ratios)
    replace_bn_with_gn(net, gn_channel_per_group=8)
    init_file = download_url('https://hanlab18.mit.edu/projects/tinyml/tinyTL/files/'
                             'proxylessnas_mobile+lite_residual@imagenet@ws+gn', model_dir=str(ROOT / 'cache'))
    net.load_state_dict(torch.load(init_file, map_location='cpu')['state_dict'])
    net.classifier = LinearLayer(net.classifier.in_features, 102, dropout_rate=0.2)
    init_models([net.classifier])

    set_module_grad_status(net, False)
    set_module_grad_status([net.classifier], True)
    enable_bias_update(net)
    for m in net.modules():
        if isinstance(m, LiteResidualModule):
            set_module_grad_status(m.lite_residual, True)
            m.lite_residual.final_bn.bias.requires_grad = False

    mem_cost, mem_info = profile_memory_cost(net, (1, 3, 224, 224), True, 32, 32, 8, batch_size=8)
    weight_quantization(net, bits=8, max_iter=20)
    replace_conv2d_with_my_conv2d(net, 1e-5)

    run_manager = RunManager(str(out), net, run_config, init=False)
    run_manager.save_config()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()

    class TrainArgs(object):
        teacher_model = None

    run_manager.train(TrainArgs())
    train_seconds = time.time() - started
    peak_mib = torch.cuda.max_memory_allocated() / 2 ** 20

    # Final test pass on the held-out 6,149 images, using the best-on-validation checkpoint.
    # model_best stores only the weights, so read the selected accuracy back from the log.
    import re
    val_lines = (out / 'logs' / 'valid_console.txt').read_text(encoding='utf-8').splitlines()
    val_top1 = [float(m.group(1)) for m in
                (re.search(r'top1 ([\d.]+)', line) for line in val_lines) if m]
    best_val = max(val_top1) if val_top1 else float('nan')
    best_epoch = val_top1.index(best_val) + 1 if val_top1 else -1
    best = torch.load(out / 'checkpoint' / 'model_best.pth.tar', map_location='cpu')
    run_manager.network.load_state_dict(best['state_dict'])
    os.environ['TINYTL_FLOWERS_PATH'] = str(test_root(work))
    test_cfg = dict(cfg)
    test_cfg['n_epochs'] = 1
    run_manager.run_config = FGVCRunConfig(**test_cfg)
    _, test_loss, test_top1, test_top5 = run_manager.validate_all_resolution(is_test=True)

    result = dict(k=args.k, seed=args.seed, tag=args.tag, ratios=ratios, epochs=args.epochs,
                  best_val_top1=best_val, best_epoch=best_epoch,
                  final_val_top1=val_top1[-1] if val_top1 else float('nan'),
                  test_top1=float(list_mean(test_top1)), test_top5=float(list_mean(test_top5)),
                  test_loss=float(list_mean(test_loss)),
                  profiler_memory_MB=mem_cost / 1e6, profiler_act_MB=mem_info['act_size'] / 1e6,
                  measured_peak_MiB=peak_mib, train_seconds=train_seconds,
                  n_train_images=args.k * 102)
    (out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('RESULT ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
