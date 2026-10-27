"""Training-step memory per arm in three accountings (实验设计/最终实验设计.md, Appendix A). No training.

  profiler   TinyTL's profile_memory_cost (MB)
  engine     ideal-engine bound from one recorded forward pass (engine_bound_r3.py): a saved tensor costs fp32 if a
             norm layer or a conv/linear with trainable weight saved it, 1 bit/elem if only ReLU/ReLU6 did, else 0;
             plus the live forward transient under the profiler's conventions.
             profiler basis (MB): + trainable x 4 B + frozen x 1 B;  resident basis (MiB): + fp32 weights + grad + Adam
  measured   torch.cuda.max_memory_allocated over 3 full train steps (fwd + bwd + Adam), synthetic 224 px input,
             minus the allocation before .cuda()

Arms with the suffix +aff replace every GroupNorm with a per-channel affine y = x * w + b that has no statistics
(what eval-mode BN reduces to). w and b keep the requires_grad of the GN they replace, so an arm trains the same
parameters (LB and T3 train lite-residual norm scales); an affine with a trainable scale is charged like a norm.

Measured memory uses cudnn.benchmark=False (training runs use True; the workspace may differ by a few MiB).

Usage:
  python measure_memory.py              3 measurement processes in different arm orders + 1 process for the
                                        profiler / engine / saved-tensor columns; writes memory_table.json
  python measure_memory.py --worker 0   one measurement process (each starts with a throw-away arm: the first CUDA
                                        arm in a process pays a one-time ~16 MiB workspace)
  python measure_memory.py --details    the profiler / engine / saved-tensor columns
"""
import argparse
import json
import subprocess
import sys
from collections import defaultdict

import torch.nn as nn

from depth_common import ROOT, setup_paths, build_net, set_trainable

ARMS = ['last', 'T5', 'T4', 'T3', 'T2', 'T1', 'T0', 'LB', 'B', 'full', 'LB+aff', 'B+aff', 'T3+aff']
ORDERS = [ARMS, ARMS[::-1], ARMS[6:] + ARMS[:6]]
BATCHES = (8, 1)
BREAKDOWN_ARMS = ('last', 'T4', 'T3', 'LB', 'B', 'full')
MiB = 2 ** 20


def gpu_util():
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        return 'unavailable: %s' % e


class StatlessAffine(nn.Module):
    """GroupNorm without statistics: y = x * w + b, with w and b keeping the GN's requires_grad."""

    def __init__(self, gn):
        super().__init__()
        self.weight = nn.Parameter(gn.weight.detach().clone(), requires_grad=gn.weight.requires_grad)
        self.bias = nn.Parameter(gn.bias.detach().clone(), requires_grad=gn.bias.requires_grad)

    def forward(self, x):
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def kind(m):
    from ofa.utils import MyConv2d
    if isinstance(m, (nn.Conv2d, MyConv2d)):
        return 'conv'
    if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
        return 'norm'
    if isinstance(m, StatlessAffine):
        return 'norm' if m.weight.requires_grad else 'other'  # dL/dw needs the input
    if isinstance(m, (nn.ReLU, nn.ReLU6)):
        return 'act'
    if isinstance(m, nn.Linear):
        return 'linear'
    return 'other'


def gn_to_affine(net):
    for m in list(net.modules()):
        for cname, c in list(m.named_children()):
            if isinstance(c, nn.GroupNorm):
                setattr(m, cname, StatlessAffine(c))
    return net


def build(arm):
    """Network for an arm, with weight standardization and, for +aff arms, GN replaced. Returns (net, trainable)."""
    from ofa.utils import replace_conv2d_with_my_conv2d
    net = build_net(102)
    set_trainable(net, arm.replace('+aff', ''))
    replace_conv2d_with_my_conv2d(net, 1e-5)
    if arm.endswith('+aff'):
        net = gn_to_affine(net)
    return net, sum(p.numel() for p in net.parameters() if p.requires_grad)


def measured_peak(net, bs):
    import torch
    import torch.nn as nn
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    net = net.cuda().train()
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=2e-4)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(bs, 3, 224, 224, device='cuda')
    y = torch.randint(0, 102, (bs,), device='cuda')
    for _ in range(2):
        opt.zero_grad(set_to_none=True); crit(net(x), y).backward(); opt.step()
    peaks = []
    for _ in range(3):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        opt.zero_grad(set_to_none=True); crit(net(x), y).backward(); opt.step()
        torch.cuda.synchronize()
        peaks.append((torch.cuda.max_memory_allocated() - base) / MiB)
    del net, opt, x, y
    torch.cuda.empty_cache()
    return max(peaks), max(peaks) - min(peaks)


def saved_breakdown(net, bs):
    """Unique bytes autograd saves in one forward pass, by the leaf module type that first saved them (MiB)."""
    import torch
    stack, hooks, seen, by = [], [], set(), defaultdict(float)
    for m in net.modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_pre_hook(lambda mod, inp: stack.append(kind(mod))))
            hooks.append(m.register_forward_hook(lambda mod, inp, out: (stack.pop(), None)[1]))

    def pack(t):
        st = t.untyped_storage()
        if st.data_ptr() not in seen:
            seen.add(st.data_ptr())
            by[stack[-1] if stack else 'other'] += st.nbytes()
        return t

    x = torch.randn(bs, 3, 224, 224, device='cuda')
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = net(x)
    del out, x
    for h in hooks:
        h.remove()
    return {k: round(v / MiB, 1) for k, v in by.items()}


def engine_bound(net, bs):
    """Ideal-engine bound (see module docstring). Returns (act_peak MiB, resident MiB, profiler-basis MB)."""
    import torch
    from ofa.utils.layers import ResidualBlock
    events, stack, rstack, hooks, savers, sbytes = [], [], [], [], {}, {}

    def tag(mod):
        if mod is None:
            return 'other'
        k = kind(mod)
        if k in ('conv', 'linear'):
            return k + ('_train' if mod.weight.requires_grad else '_frozen')
        return k

    def key(t):
        st = t.untyped_storage()
        return st.data_ptr(), st.nbytes()

    def post(mod, inp, out):
        stack.pop()
        ik, _ = key(inp[0])
        ok, _ = key(out)
        ib, ob = inp[0].numel() * 4, out.numel() * 4
        k = kind(mod)
        if k == 'conv':
            ob = ob // mod.groups  # in-place depthwise, as in the profiler
        elif k != 'linear':
            ob = 0  # norm / activation / other run in place
        rk, rb = rstack[-1] if rstack else (None, 0)
        events.append(('op', ik, ib, ok, ob, rk, rb))

    for m in net.modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_pre_hook(lambda mod, inp: stack.append(mod)))
            hooks.append(m.register_forward_hook(post))
        if isinstance(m, ResidualBlock) and m.shortcut is not None:
            hooks.append(m.register_forward_pre_hook(lambda mod, inp: rstack.append(key(inp[0]))))
            hooks.append(m.register_forward_hook(lambda mod, inp, out: (rstack.pop(), None)[1]))

    def pack(t):
        k, b = key(t)
        if k not in savers:
            savers[k] = set(); sbytes[k] = b; events.append(('pack', k))
        savers[k].add(tag(stack[-1] if stack else None))
        return t

    x = torch.randn(bs, 3, 224, 224, device='cuda')
    _, xb = key(x)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = net(x)
    del out
    for h in hooks:
        h.remove()

    need = {}
    for k, s in savers.items():
        if s & {'norm', 'conv_train', 'linear_train'}:
            need[k] = ('full', sbytes[k])
        elif 'act' in s:
            need[k] = ('mask', sbytes[k] / 32)
        else:
            need[k] = ('drop', 0)
    full_keys, counted, total, peak = set(), set(), 0, 0
    for e in events:
        if e[0] == 'pack':
            if e[1] not in counted:
                counted.add(e[1])
                kind_, b = need[e[1]]
                total += b
                if kind_ == 'full':
                    full_keys.add(e[1])
        else:
            _, ik, ib, ok, ob, rk, rb = e
            live, seen = total, set()
            for k_, b_ in ((ik, ib), (ok, ob), (rk, rb)):
                if k_ is None or b_ == 0 or k_ in full_keys or k_ in seen:
                    continue
                seen.add(k_)
                live += b_
            peak = max(peak, live)
    peak = max(peak, total)
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    resident = peak + (trainable + frozen) * 4 + 3 * 4 * trainable + xb
    prof_basis = peak + trainable * 4 + frozen * 1
    return peak / MiB, resident / MiB, prof_basis / 1e6


def worker(index):
    """Measured peaks only, so that nothing else runs on the GPU between measurements."""
    setup_paths()
    import torch
    rows = []
    util = gpu_util()
    net, _ = build('LB')
    measured_peak(net, 8)  # throw-away arm: absorbs the one-time CUDA workspace
    for bs in BATCHES:
        for arm in ORDERS[index]:
            net, n = build(arm)
            row = dict(arm=arm, bs=bs, trainable=n, position=len(rows) + 1)
            row['measured_MiB'], row['spread_MiB'] = measured_peak(net, bs)
            rows.append(row)
            print(json.dumps(row), flush=True)
    return dict(worker=index, gpu_util_at_start=util, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                rows=rows)


def details():
    """Profiler (on the GN network, also for +aff rows), engine bound and saved-tensor breakdown."""
    setup_paths()
    import torch
    from tinytl.utils import profile_memory_cost
    rows = []
    for bs in BATCHES:
        for arm in ARMS:
            base = arm.replace('+aff', '')
            net, _ = build(base)
            est, _ = profile_memory_cost(net, (1, 3, 224, 224), base != 'last', 32, 32, 8, batch_size=bs)
            net, n = build(arm)
            net = net.cuda().train()
            act, resident, prof_basis = engine_bound(net, bs)
            row = dict(arm=arm, bs=bs, trainable=n, profiler_MB=round(est / 1e6, 1),
                       engine_act_peak_MiB=round(act, 1), engine_resident_MiB=round(resident, 1),
                       engine_profiler_basis_MB=round(prof_basis, 1))
            if bs == 8 and arm in BREAKDOWN_ARMS:
                row['saved_MiB'] = saved_breakdown(net, bs)
            del net
            torch.cuda.empty_cache()
            rows.append(row)
            print(json.dumps(row), flush=True)
    return dict(rows=rows)


def run_child(flag):
    proc = subprocess.run([sys.executable, '-u', __file__] + flag, capture_output=True, text=True, cwd=str(ROOT))
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-3000:])
        raise SystemExit('%s failed' % ' '.join(flag))
    return json.loads(proc.stdout.strip().splitlines()[-1])


def driver():
    raw = []
    for i in range(len(ORDERS)):
        print('== worker %d' % i, flush=True)
        raw.append(run_child(['--worker', str(i)]))
    print('== details', flush=True)
    det = {(r['arm'], r['bs']): r for r in run_child(['--details'])['rows']}
    table = []
    for bs in BATCHES:
        for arm in ARMS:
            got = [r for w in raw for r in w['rows'] if r['arm'] == arm and r['bs'] == bs]
            d = det[(arm, bs)]
            assert all(r['trainable'] == d['trainable'] for r in got)
            m = [r['measured_MiB'] for r in got]
            row = dict(arm=arm, bs=bs, trainable=d['trainable'], profiler_MB=d['profiler_MB'],
                       engine_profiler_basis_MB=d['engine_profiler_basis_MB'],
                       engine_resident_MiB=d['engine_resident_MiB'],
                       measured_MiB=round(max(m), 1), measured_range_MiB=round(max(m) - min(m), 1))
            if 'saved_MiB' in d:
                row['saved_MiB'] = d['saved_MiB']
            table.append(row)
    meta = dict(gpu=raw[0]['gpu'], torch=raw[0]['torch'], gpu_util_at_start=[w['gpu_util_at_start'] for w in raw],
                cudnn_benchmark=False)
    (ROOT / 'memory_table.json').write_text(json.dumps(dict(meta=meta, table=table, raw=raw, details=list(det.values())),
                                                       indent=1), encoding='utf-8')
    print('\n%-7s %3s %10s %11s %8s %10s %12s %10s' % ('arm', 'bs', 'trainable', 'profilerMB', 'engineMB',
                                                     'engineMiB', 'measuredMiB', 'rangeMiB'))
    for r in table:
        print('%-7s %3d %10d %11.1f %8.1f %10.1f %12.1f %10.1f' % (
            r['arm'], r['bs'], r['trainable'], r['profiler_MB'], r['engine_profiler_basis_MB'],
            r['engine_resident_MiB'], r['measured_MiB'], r['measured_range_MiB']))
    print('GPU utilization at the start of each worker:', meta['gpu_util_at_start'])
    print('Wrote', ROOT / 'memory_table.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--worker', type=int, default=None)
    p.add_argument('--details', action='store_true')
    a = p.parse_args()
    if a.details:
        print(json.dumps(details()), flush=True)
    elif a.worker is not None:
        print(json.dumps(worker(a.worker)), flush=True)
    else:
        driver()
