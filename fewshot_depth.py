"""Few-shot TinyTL with a truncated backprop cutoff (实验设计/最终实验设计.md).

Differences from the upstream trainer, all deliberate:
  * k images per class from the original train split (same sampler as fewshot_resolution.py), real val split
    logged every few epochs for learning curves only, test evaluated once;
  * no checkpoint selection: the in-memory final-epoch model is tested;
  * arms differ only in which parameters receive gradients (depth_common.set_trainable), and therefore in which
    frozen layers are 8-bit quantized, as upstream;
  * quantization is deterministic and cached (depth_common.quantize_frozen), and all RNGs are reseeded right before
    training, so a seed fixes the subset, head init, quantized weights and data order across arms. CUDA numerics
    are not bit-reproducible (cudnn.benchmark, atomics in GN and bilinear-upsample backward);
  * only the trainable parameters of the final epoch are saved;
  * --head_init imprint (实验设计/补充实验设计.md) replaces the random head, just before training, by the nearest-class-
    mean head on the k-shot images, computed through the network as it will be trained (quantized, weight-standardized),
    with its temperature fitted to the label-smoothed training loss on those images. head_init_sha1 still records the
    random head, so pairing checks work across head inits.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import time

from depth_common import (ROOT, HEAD_INITS, check_arm, setup_paths, build_net, set_trainable, expected_trainable,
                          run_name, sha1_tensors, quantize_frozen)

DATASETS = {
    # upstream exp_scripts/batch8 and batch1 recipes
    'flowers102': dict(n_classes=102, distort_color='None', env='TINYTL_FLOWERS_PATH', lr_b8=2e-4, lr_b1=3e-4),
    'aircraft': dict(n_classes=100, distort_color='torch', env='TINYTL_AIRCRAFT_PATH', lr_b8=16e-4, lr_b1=3e-4),
}
CODE_FILES = ['fewshot_depth.py', 'depth_common.py', 'fewshot_resolution.py', 'prepare_aircraft.py']


def data_roots(dataset, k, seed):
    if dataset == 'flowers102':
        from fewshot_resolution import build_root, test_root
        work = ROOT / 'fewshot'
        return build_root(k, seed, work), test_root(work)
    from prepare_aircraft import build_root_aircraft, test_root_aircraft
    return build_root_aircraft(k, seed), test_root_aircraft()


def seed_all(seed):
    import numpy as np
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def gpu_util():
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        return 'unavailable: %s' % e


def file_list_sha1(root):
    names = sorted(p.relative_to(root).as_posix() for p in root.rglob('*.jpg'))
    return hashlib.sha1('\n'.join(names).encode()).hexdigest(), len(names)


def parse_valid_log(path, epochs, val_every):
    """From upstream's valid_console.txt: [(epoch, val_top1, train_top1)] and [(epoch, val_loss, train_loss)].
    The logged epochs must be complete."""
    num = r'([\d.]+|nan|inf)'
    pat = re.compile(r'Valid \[(\d+)/\d+\]\s+loss %s\s+top1 %s \(\S+\).*?Train top1 %s\s+loss %s'
                     % (num, num, num, num))
    rows, losses = [], []
    for line in path.read_text(encoding='utf-8').splitlines():
        m = pat.search(line)
        if m:
            e, vloss, vtop1, ttop1, tloss = int(m.group(1)), *map(float, m.group(2, 3, 4, 5))
            rows.append([e, vtop1, ttop1])
            losses.append([e, vloss, tloss])
    want = list(range(val_every, epochs + 1, val_every))
    assert [r[0] for r in rows] == want, 'validation epochs %s != %s' % ([r[0] for r in rows], want)
    return rows, losses


TAU_GRID = (-2, 4, 601)  # log10 range and points of the temperature grid


def ncm_head(feats, labels, n_classes, smoothing):
    """Nearest-class-mean (ProtoNet) head from support features: logit_c = (2 f.mu_c - |mu_c|^2) / tau, where mu_c
    are the class means of the support images. tau minimizes the training loss (cross-entropy with the recipe's label
    smoothing) of this head on the support set, over a log grid, so the head starts at the scale training aims for.
    The leave-one-out accuracy (own class mean without the image) and its best tau are diagnostics only: with k=2
    the own-class mean is a single image, which makes them pessimistic. Returns (weight, bias, stats) in float64."""
    import torch
    import torch.nn.functional as F
    f = feats.double()
    counts = torch.bincount(labels, minlength=n_classes)
    assert int(counts.min()) >= 2, 'imprinting needs at least 2 images per class'
    mu = torch.zeros(n_classes, f.shape[1], dtype=f.dtype, device=f.device).index_add_(0, labels, f)
    mu = mu / counts[:, None].to(f.dtype)
    logits = 2 * f @ mu.T - (mu ** 2).sum(1)
    k = counts[labels].to(f.dtype)[:, None]
    mu_own = (k * mu[labels] - f) / (k - 1)
    loo = logits.clone()
    loo[torch.arange(len(labels), device=f.device), labels] = 2 * (f * mu_own).sum(1) - (mu_own ** 2).sum(1)
    taus = torch.logspace(*TAU_GRID, dtype=f.dtype, device=f.device)
    ls = torch.stack([F.cross_entropy(logits / t, labels, label_smoothing=smoothing) for t in taus])
    loo_ce = torch.stack([F.cross_entropy(loo / t, labels) for t in taus])
    i, j = int(ls.argmin()), int(loo_ce.argmin())
    tau = float(taus[i])
    stats = dict(tau=tau, tau_at_grid_edge=i in (0, len(taus) - 1), support_loss=float(ls[i]),
                 support_top1=100.0 * float((logits.argmax(1) == labels).double().mean()),
                 loo_top1=100.0 * float((loo.argmax(1) == labels).double().mean()), loo_tau=float(taus[j]),
                 loo_tau_at_grid_edge=j in (0, len(taus) - 1))
    return 2 * mu / tau, -(mu ** 2).sum(1) / tau, stats


def pooled_features(net, folder, transform, classes):
    """Un-augmented pooled features (the classifier's input) of an image folder, net in eval mode, no shuffling."""
    import torch
    from torchvision import datasets
    ds = datasets.ImageFolder(str(folder), transform)
    assert ds.classes == classes, '%s: folder classes differ from the provider' % folder
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    device = next(net.parameters()).device
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            h = net.first_conv(x)
            for block in net.blocks:
                h = block(h)
            h = net.global_avg_pool(net.feature_mix_layer(h))
            if not feats:  # the features must be exactly what the classifier sees
                assert torch.allclose(net.classifier(h), net(x), atol=1e-4), 'feature path differs from forward()'
            feats.append(h)
            labels.append(y.to(device))
    return torch.cat(feats), torch.cat(labels)


def imprint(net, provider, data_root, classes, n_classes, k, smoothing):
    """Set the classifier to ncm_head on the k-shot images (upstream test transform). Also returns the val top-1 of
    the random and of the imprinted head before any training, and their mean per-class weight norms."""
    import torch
    tf = provider.build_valid_transform(224)
    was_training = net.training
    net.eval()
    try:
        feats, labels = pooled_features(net, data_root / 'train', tf, classes)
        val_feats, val_labels = pooled_features(net, data_root / 'val', tf, classes)
    finally:
        net.train(was_training)
    assert torch.bincount(labels, minlength=n_classes).tolist() == [k] * n_classes
    lin = net.classifier.linear
    w0, b0 = lin.weight.detach().double(), lin.bias.detach().double()
    w, b, stats = ncm_head(feats, labels, n_classes, smoothing)
    val_top1 = lambda w_, b_: 100.0 * float(((val_feats.double() @ w_.T + b_).argmax(1) == val_labels).double().mean())
    stats.update(val_top1_random_head=val_top1(w0, b0), val_top1_imprinted_head=val_top1(w, b),
                 row_norm_random=float(w0.norm(dim=1).mean()), row_norm_imprinted=float(w.norm(dim=1).mean()))
    with torch.no_grad():
        lin.weight.copy_(w.to(lin.weight.dtype))
        lin.bias.copy_(b.to(lin.bias.dtype))
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='flowers102', choices=sorted(DATASETS))
    p.add_argument('--arm', default='LB', type=check_arm, help='a named arm or a block set (depth_common)')
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=None, help='default: the upstream recipe for this dataset and batch')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--val_every', type=int, default=5)
    p.add_argument('--test_batch', type=int, default=64)
    p.add_argument('--tag', default='')
    p.add_argument('--head_init', default='random', choices=HEAD_INITS)
    args = p.parse_args()
    ds = DATASETS[args.dataset]
    if args.lr is None:
        args.lr = ds['lr_b1'] if args.batch == 1 else ds['lr_b8']
    assert args.epochs % args.val_every == 0, 'epochs must be a multiple of val_every (the final epoch is logged)'
    assert args.head_init == 'random' or args.k >= 2, 'imprinting needs k >= 2 (leave-one-out diagnostic)'

    name = run_name(args.dataset, args.k, args.arm, args.batch, args.lr, args.seed, args.epochs, args.val_every,
                    args.tag, args.head_init)
    out = ROOT / 'runs' / name
    if (out / 'result.json').exists():
        print('SKIP %s (result.json exists)' % name, flush=True)
        return
    if out.exists():  # a crashed attempt of this exact run: upstream logs append, so start clean
        shutil.rmtree(out)
    out.mkdir(parents=True)
    started = time.time()
    util_at_start = gpu_util()

    os.environ['TINYTL_CACHE'] = str(ROOT / 'cache')
    os.environ['TORCH_HOME'] = str(ROOT / 'cache' / 'torch')
    os.environ['OMP_NUM_THREADS'] = '4'
    setup_paths()
    import sklearn
    import torch
    import torchvision
    torch.set_num_threads(4)
    from ofa.imagenet_classification.run_manager import RunManager
    from ofa.utils import replace_conv2d_with_my_conv2d, list_mean
    from tinytl.data_providers import FGVCRunConfig, AircraftDataProvider
    from tinytl.utils import profile_memory_cost
    AircraftDataProvider.save_path = property(lambda self: os.environ['TINYTL_AIRCRAFT_PATH'])

    seed_all(args.seed)
    train_root, test_root = data_roots(args.dataset, args.k, args.seed)
    os.environ[ds['env']] = str(train_root)
    subset_sha1, n_subset = file_list_sha1(train_root / 'train')
    assert n_subset == args.k * ds['n_classes'], 'k-shot root has %d images' % n_subset

    net = build_net(ds['n_classes'])
    head_init_sha1 = sha1_tensors(net.classifier.parameters())
    n_trainable = set_trainable(net, args.arm)
    expected = expected_trainable(args.arm, ds['n_classes'])
    assert n_trainable == expected, 'trainable %d != expected %d for %s' % (n_trainable, expected, args.arm)
    mem_cost, mem_info = profile_memory_cost(net, (1, 3, 224, 224), args.arm != 'last', 32, 32, 8,
                                             batch_size=args.batch)
    n_quantized, n_quantized_new = quantize_frozen(net, bits=8, max_iter=20)
    frozen_sha1 = sha1_tensors(t for _, t in sorted(net.named_parameters()) if not t.requires_grad)
    replace_conv2d_with_my_conv2d(net, 1e-5)

    cfg = dict(n_epochs=args.epochs, init_lr=args.lr, lr_schedule_type='cosine', lr_schedule_param=None,
               dataset=args.dataset, train_batch_size=args.batch, test_batch_size=args.test_batch, valid_size=None,
               opt_type='adam', opt_param={'momentum': 0.9, 'nesterov': True}, weight_decay=0,
               label_smoothing=0.7, no_decay_keys='bn#bias', mixup_alpha=None, model_init='he_fout',
               validation_frequency=args.val_every, print_frequency=10, n_worker=0, resize_scale=0.22,
               distort_color=ds['distort_color'], image_size=224)
    run_manager = RunManager(str(out), net, FGVCRunConfig(**cfg), init=False)
    run_manager.save_config()
    provider = run_manager.run_config.data_provider
    assert len(provider.train.dataset) == n_subset
    classes = provider.train.dataset.classes
    assert len(classes) == ds['n_classes'] and provider.valid.dataset.classes == classes
    imprint_stats, imprint_sha1 = None, None
    if args.head_init == 'imprint':
        imprint_stats = imprint(run_manager.network, provider, train_root, classes, ds['n_classes'], args.k,
                                cfg['label_smoothing'])
        imprint_sha1 = sha1_tensors(run_manager.network.classifier.parameters())
        print('imprinted head: %s' % imprint_stats, flush=True)
    steps = len(run_manager.run_config.train_loader) * args.epochs

    # Save only the trainable parameters, once, at the last epoch (upstream writes ~54 MB every epoch).
    def save_final(checkpoint=None, is_best=False, model_name=None):
        if checkpoint is not None and checkpoint.get('epoch') == args.epochs - 1:
            torch.save({n: t.detach().cpu() for n, t in run_manager.network.named_parameters() if t.requires_grad},
                       out / 'final_trainable.pth')
    run_manager.save_model = save_final

    class TrainArgs(object):
        teacher_model = None

    seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    setup_seconds = time.time() - started
    t0 = time.time()
    run_manager.train(TrainArgs())
    train_seconds = time.time() - t0
    run_peak = torch.cuda.max_memory_allocated() / 2 ** 20

    curve, loss_curve = parse_valid_log(out / 'logs' / 'valid_console.txt', args.epochs, args.val_every)

    # Test the in-memory final-epoch model once on the held-out test split.
    t0 = time.time()
    os.environ[ds['env']] = str(test_root)
    run_manager.run_config = FGVCRunConfig(**dict(cfg, n_epochs=1))
    assert run_manager.run_config.data_provider.test.dataset.classes == classes
    _, test_loss, test_top1, test_top5 = run_manager.validate_all_resolution(is_test=True)
    test_seconds = time.time() - t0

    code_sha1 = hashlib.sha1(b''.join((ROOT / f).read_bytes() for f in CODE_FILES)).hexdigest()
    result = dict(name=name, dataset=args.dataset, k=args.k, arm=args.arm, batch=args.batch, lr=args.lr,
                  seed=args.seed, epochs=args.epochs, val_every=args.val_every, steps=steps,
                  n_train_images=n_subset, tag=args.tag, head_init=args.head_init, imprint=imprint_stats,
                  imprint_sha1=imprint_sha1,
                  test_top1=float(list_mean(test_top1)), test_top5=float(list_mean(test_top5)),
                  test_loss=float(list_mean(test_loss)),
                  final_val_top1=curve[-1][1], final_train_top1=curve[-1][2], val_curve=curve, loss_curve=loss_curve,
                  diverged=any(not math.isfinite(v) for row in curve + loss_curve for v in row[1:]),
                  trainable_params=n_trainable, profiler_MB=mem_cost / 1e6, profiler_act_MB=mem_info['act_size'] / 1e6,
                  run_peak_incl_eval_MiB=run_peak, quantized_layers=n_quantized,
                  quantized_layers_computed=n_quantized_new,
                  subset_sha1=subset_sha1, head_init_sha1=head_init_sha1, frozen_sha1=frozen_sha1,
                  code_sha1=code_sha1, setup_seconds=setup_seconds, train_seconds=train_seconds,
                  test_seconds=test_seconds, gpu_util_at_start=util_at_start, cudnn_benchmark=True,
                  versions=dict(torch=torch.__version__, torchvision=torchvision.__version__,
                                sklearn=sklearn.__version__))
    tmp = out / 'result.json.tmp'
    tmp.write_text(json.dumps(result, indent=2), encoding='utf-8')
    os.replace(tmp, out / 'result.json')
    print('RESULT ' + json.dumps({k: v for k, v in result.items() if k != 'val_curve'}), flush=True)


if __name__ == '__main__':
    main()
