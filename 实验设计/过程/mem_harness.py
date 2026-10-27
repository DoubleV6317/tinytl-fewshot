"""Unified measured-memory harness (design feasibility check, no training).

One build function and one measurement routine for every arm, at batch 1 and 8:
  peak  = torch.cuda.max_memory_allocated() over 3 full train steps (fwd+bwd+Adam), minus allocation before .cuda()
  saved = unique bytes that autograd saves for backward in one forward pass, attributed to the leaf module type
          that saved them (conv / norm / act / linear / other), de-duplicated by storage.
"""
import os, sys, re, json
from collections import defaultdict
from pathlib import Path
ROOT = Path(r"C:\Users\zhita\Documents\Codex\2026-09-10\this-class-will-feature-a-quarter\outputs\TinyTL")
os.environ['TINYTL_CACHE'] = str(ROOT / 'cache'); os.environ['TORCH_HOME'] = str(ROOT / 'cache' / 'torch')
sys.path[:0] = [str(ROOT / 'upstream' / 'once-for-all'), str(ROOT / 'upstream' / 'tinytl')]
import torch, torch.nn as nn
from ofa.utils.layers import LinearLayer
from ofa.model_zoo import proxylessnas_mobile
from ofa.utils import replace_conv2d_with_my_conv2d, replace_bn_with_gn, MyConv2d
from tinytl.utils import set_module_grad_status, profile_memory_cost
from tinytl.model import LiteResidualModule

HEAD = 10 ** 6 - 1


def block_index(name):
    if name.startswith('first_conv'):
        return -1
    m = re.match(r'blocks\.(\d+)\.', name)
    return int(m.group(1)) if m else 10 ** 6  # feature_mix_layer / classifier


def build(arm):
    net = proxylessnas_mobile(pretrained=False)
    LiteResidualModule.insert_lite_residual(net, 2, 'bilinear', 1.0, 5, 'relu', 2)
    replace_bn_with_gn(net, gn_channel_per_group=8)
    net.load_state_dict(torch.load(ROOT / 'cache' / 'proxylessnas_mobile+lite_residual@imagenet@ws+gn',
                                   map_location='cpu', weights_only=False)['state_dict'])
    net.classifier = LinearLayer(net.classifier.in_features, 102, dropout_rate=0.2)
    set_module_grad_status(net, arm == 'full')
    starts = {'LB': -1, 'B': -1, 'T0': 1, 'T1': 5, 'T2': 9, 'T3': 13, 'T4': 17, 'T5': 21, 'head': HEAD}
    if arm in starts:
        start, lite = starts[arm], arm != 'B'
        for name, p in net.named_parameters():
            if block_index(name) < start:
                continue
            if name.endswith('.bias') and 'lite_residual.final_bn' not in name:
                p.requires_grad = True
            if lite and '.lite_residual.' in name and 'lite_residual.final_bn.bias' not in name:
                p.requires_grad = True
    set_module_grad_status([net.classifier], True)
    replace_conv2d_with_my_conv2d(net, 1e-5)
    return net


def kind(m):
    if isinstance(m, (nn.Conv2d, MyConv2d)):
        return 'conv'
    if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
        return 'norm'
    if isinstance(m, (nn.ReLU, nn.ReLU6)):
        return 'act'
    if isinstance(m, nn.Linear):
        return 'linear'
    return 'other'


def saved_breakdown(net, bs):
    stack = []
    hooks = []
    for m in net.modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_pre_hook(lambda mod, inp: stack.append(kind(mod))))
            hooks.append(m.register_forward_hook(lambda mod, inp, out: (stack.pop(), None)[1]))
    seen, by = set(), defaultdict(float)

    def pack(t):
        try:
            st = t.untyped_storage()
            key = st.data_ptr()
            if key not in seen:
                seen.add(key)
                by[stack[-1] if stack else 'other'] += st.nbytes()
        except Exception:
            pass
        return t

    x = torch.randn(bs, 3, 224, 224, device='cuda')
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = net(x)
    del out, x
    for h in hooks:
        h.remove()
    return {k: v / 2 ** 20 for k, v in by.items()}


def measure(arm, bs, breakdown=False):
    net = build(arm)
    est, _ = profile_memory_cost(net, (1, 3, 224, 224), True, 32, 32, 8, batch_size=bs)
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    net = net.cuda().train()
    weights = (torch.cuda.memory_allocated() - base) / 2 ** 20
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=2e-4)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(bs, 3, 224, 224, device='cuda'); y = torch.randint(0, 102, (bs,), device='cuda')
    for _ in range(2):
        opt.zero_grad(set_to_none=True); crit(net(x), y).backward(); opt.step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    peaks = []
    for _ in range(3):
        torch.cuda.reset_peak_memory_stats()
        opt.zero_grad(set_to_none=True); crit(net(x), y).backward(); opt.step()
        torch.cuda.synchronize()
        peaks.append((torch.cuda.max_memory_allocated() - base) / 2 ** 20)
    del x, y
    bd = saved_breakdown(net, bs) if breakdown else None
    del net, opt; torch.cuda.empty_cache()
    return dict(arm=arm, bs=bs, profiler_MB=est / 1e6, peak_MiB=max(peaks), peak_spread=max(peaks) - min(peaks),
                weights_MiB=weights, trainable=trainable, saved_MiB=bd)


if __name__ == '__main__':
    arms = ['last', 'head', 'T5', 'T4', 'T3', 'T2', 'T1', 'T0', 'LB', 'B', 'full']
    rows = []
    for bs in (8, 1):
        for arm in arms:
            r = measure(arm, bs, breakdown=(bs == 8 and arm in ('last', 'head', 'T4', 'T3', 'LB', 'B', 'full')))
            rows.append(r)
            print(json.dumps(r), flush=True)
    out = Path(__file__).with_name('mem_harness_result.json')
    out.write_text(json.dumps(rows, indent=1), encoding='utf-8')
    print('\n%-6s %4s %9s %10s %8s %11s' % ('arm', 'bs', 'profMB', 'peakMiB', 'spread', 'trainable'))
    for r in rows:
        print('%-6s %4d %9.1f %10.1f %8.2f %11d' % (r['arm'], r['bs'], r['profiler_MB'], r['peak_MiB'],
                                                  r['peak_spread'], r['trainable']))
