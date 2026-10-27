"""GPU time per training step for every arm at batch 8 and batch 1 (实验设计/最终实验设计.md §1.3, "our machine" only).

  python measure_time.py            writes time_table.json; refuses to run while a sweep holds runs/.sweep.lock or
                                    while the GPU is busier than 5% (sampled before every arm)

One process, synthetic 224 px input already on the GPU, so data loading and augmentation are excluded. A step is
forward + backward + Adam, with weight standardization and cudnn.benchmark=True as in fewshot_depth.py, and the
label-smoothed loss of the recipe. Frozen weights are not k-means quantized: quantization changes their values, not
the fp32 compute. Each arm/batch gets 10 warm-up steps, then 3 timed rounds of 50 steps; the median round is reported.
The first arm is a throw-away (it pays cudnn autotuning and allocator growth).
"""
import argparse
import json
import statistics
import subprocess
import sys
import time

from depth_common import ROOT, setup_paths, build_net, set_trainable

ARMS = ['last', 'T5', 'T4', 'T3', 'T2', 'LB', 'B', 'full']
BATCHES = (8, 1)
MAX_UTIL = 5
WARMUP, ROUNDS, STEPS = 10, 3, 50


def gpu_util_samples(n=5, gap=1.0):
    out = []
    for i in range(n):
        q = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        util, mem = (int(x) for x in q.split(','))
        out.append((util, mem))
        if i < n - 1:
            time.sleep(gap)
    return out


def require_idle(settle=0.0):
    time.sleep(settle)  # let this process's previous arm drop out of nvidia-smi's sampling window
    if (ROOT / 'runs' / '.sweep.lock').exists():
        sys.exit('a sweep is running (runs/.sweep.lock): timing needs an idle GPU')
    s = gpu_util_samples()
    if max(u for u, _ in s) > MAX_UTIL:
        sys.exit('GPU utilization %s%% exceeds %d%%: close other GPU programs first' % ([u for u, _ in s], MAX_UTIL))
    return s


def time_arm(arm, batch, n_classes, device):
    import torch
    from ofa.utils import replace_conv2d_with_my_conv2d
    from ofa.utils import cross_entropy_with_label_smoothing
    net = build_net(n_classes)
    set_trainable(net, arm)
    replace_conv2d_with_my_conv2d(net, 1e-5)
    net = net.to(device).train()
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-4)
    x = torch.randn(batch, 3, 224, 224, device=device)
    y = torch.randint(0, n_classes, (batch,), device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        cross_entropy_with_label_smoothing(net(x), y, 0.7).backward()
        opt.step()

    for _ in range(WARMUP):
        step()
    rounds = []
    for _ in range(ROUNDS):
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(STEPS):
            step()
        torch.cuda.synchronize()
        rounds.append((time.perf_counter() - t) / STEPS * 1000)
    del net, opt, x, y
    torch.cuda.empty_cache()
    return rounds


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_classes', type=int, default=102, help='head size (102 Flowers, 100 Aircraft; no effect on time)')
    args = p.parse_args()
    start = require_idle()
    setup_paths()
    import torch
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    time_arm('LB', 8, args.n_classes, device)  # throw-away
    rows = []
    for batch in BATCHES:
        for arm in ARMS:
            before = require_idle(settle=3.0)
            rounds = time_arm(arm, batch, args.n_classes, device)
            ms = statistics.median(rounds)
            rows.append(dict(arm=arm, batch=batch, ms_per_step=round(ms, 2), ms_per_image=round(ms / batch, 2),
                             rounds_ms=[round(r, 2) for r in rounds], util_before=before))
            print('%-4s batch %d: %7.2f ms/step  %6.2f ms/image  rounds %s' % (
                arm, batch, ms, ms / batch, ' '.join('%.2f' % r for r in rounds)), flush=True)
    lb = {b: next(r['ms_per_step'] for r in rows if r['arm'] == 'LB' and r['batch'] == b) for b in BATCHES}
    for r in rows:
        r['vs_LB'] = round(r['ms_per_step'] / lb[r['batch']], 3)
    meta = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cudnn_benchmark=True,
                warmup=WARMUP, rounds=ROUNDS, steps_per_round=STEPS, util_at_start=start,
                note='our machine only; synthetic input on the GPU, no data loading; frozen weights not quantized')
    out = ROOT / 'time_table.json'
    out.write_text(json.dumps(dict(meta=meta, table=rows), indent=1), encoding='utf-8')
    print('wrote', out)


if __name__ == '__main__':
    main()
