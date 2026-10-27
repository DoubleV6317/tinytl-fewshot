"""Three memory accountings per arm (no training), plus the GN -> frozen-affine ablation (review S10).

engine lower bound (saved state only, transients excluded) =
    tensors autograd saves that an ideal engine must keep:
        - full fp32 if any saver is a norm layer, or a conv / linear whose weight is trainable
        - 1 bit per element if the only savers are ReLU / ReLU6 (mask)
        - 0 if the only savers are frozen convs (input not needed for input-grad) or shape-only ops
  + all weights (fp32, as resident in PyTorch) + grad + Adam m, v for trainable params (3 x 4 B) + input batch.
Measured peak uses the mem_harness protocol, with a throw-away warm-up arm first (excludes the one-time workspace).
"""
import sys, json
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch, torch.nn as nn
from mem_harness import build, kind, measure


def engine_bound(net, bs):
    stack, hooks = [], []
    for m in net.modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_pre_hook(lambda mod, inp: stack.append(mod)))
            hooks.append(m.register_forward_hook(lambda mod, inp, out: (stack.pop(), None)[1]))
    info = {}  # storage ptr -> [bytes, set of saver tags]

    def tag(mod):
        if mod is None:
            return 'other'
        k = kind(mod)
        if k in ('conv', 'linear'):
            return k + ('_train' if mod.weight.requires_grad else '_frozen')
        if k == 'other' and isinstance(mod, FrozenAffine):
            return 'affine'
        return k

    def pack(t):
        st = t.untyped_storage()
        key = st.data_ptr()
        e = info.setdefault(key, [st.nbytes(), set()])
        e[1].add(tag(stack[-1] if stack else None))
        return t

    x = torch.randn(bs, 3, 224, 224, device='cuda')
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = net(x)
    del out, x
    for h in hooks:
        h.remove()
    need = defaultdict(float)
    for nbytes, savers in info.values():
        if savers & {'norm', 'conv_train', 'linear_train'}:
            need['full'] += nbytes
        elif savers & {'act'}:
            need['mask'] += nbytes / 32
        else:
            need['dropped:' + ','.join(sorted(savers))] += nbytes
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    weights = sum(p.numel() for p in net.parameters()) * 4
    extra = weights + 3 * 4 * trainable + bs * 3 * 224 * 224 * 4
    total = (need['full'] + need['mask'] + extra) / 2 ** 20
    return total, {k: round(v / 2 ** 20, 1) for k, v in need.items()}, round(extra / 2 ** 20, 1)


class FrozenAffine(nn.Module):
    """What eval-mode BN reduces to: y = x * w + b with w frozen (values irrelevant for memory)."""
    def __init__(self, gn):
        super().__init__()
        self.register_buffer('w', gn.weight.detach().clone().view(1, -1, 1, 1))
        self.bias = nn.Parameter(gn.bias.detach().clone(), requires_grad=gn.bias.requires_grad)

    def forward(self, x):
        return x * self.w + self.bias.view(1, -1, 1, 1)


def gn_to_affine(net):
    for name, m in list(net.named_modules()):
        for cname, c in list(m.named_children()):
            if isinstance(c, nn.GroupNorm):
                setattr(m, cname, FrozenAffine(c))
    return net


def peak(net, bs):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    net = net.cuda().train()
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=2e-4)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(bs, 3, 224, 224, device='cuda'); y = torch.randint(0, 102, (bs,), device='cuda')
    for _ in range(2):
        opt.zero_grad(set_to_none=True); crit(net(x), y).backward(); opt.step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        opt.zero_grad(set_to_none=True); crit(net(x), y).backward(); opt.step()
    torch.cuda.synchronize()
    p = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    del opt, x, y
    return p, net


if __name__ == '__main__':
    measure('T3', 8)  # throw-away warm-up arm: absorbs the one-time workspace allocation
    rows = []
    for bs in (8, 1):
        for arm in ['last', 'T4', 'T3', 'T2', 'LB', 'B', 'full']:
            r = measure(arm, bs)
            net = build(arm).cuda().train()
            eb, parts, extra = engine_bound(net, bs)
            del net; torch.cuda.empty_cache()
            row = dict(arm=arm, bs=bs, profiler_MB=round(r['profiler_MB'], 1), measured_MiB=round(r['peak_MiB'], 1),
                       engine_lb_MiB=round(eb, 1), saved_needed=parts, weights_opt_input_MiB=extra)
            rows.append(row); print(json.dumps(row), flush=True)
    for arm in ['LB', 'B', 'T3']:
        for bs in (8, 1):
            p, net = peak(gn_to_affine(build(arm)), bs)
            eb, parts, extra = engine_bound(net, bs)
            del net; torch.cuda.empty_cache()
            row = dict(arm=arm + '+affine(no GN)', bs=bs, measured_MiB=round(p, 1), engine_lb_MiB=round(eb, 1),
                       saved_needed=parts, weights_opt_input_MiB=extra)
            rows.append(row); print(json.dumps(row), flush=True)
    Path(__file__).with_name('engine_bound_result.json').write_text(json.dumps(rows, indent=1), encoding='utf-8')
