"""Transient-inclusive ideal-engine bound (review r2, R1). No training.

One forward pass records, in order, every autograd save and every leaf-op boundary. A storage's need is decided
after the pass from all of its savers (as in engine_bound.py):
    full fp32 if any saver is a norm or a conv/linear with trainable weight; 1 bit/elem if only ReLU/ReLU6; else 0.
Replay: needed_so_far grows at each first save; at every leaf-op end the live set is
    needed_so_far + op input + op output + live residual, using the TinyTL profiler's transient convention
    (conv: x + y/groups, i.e. in-place depthwise; linear: x + y; norm/act: x, in place), and not re-counting a
    tensor that is already a full-needed saved storage.
act_peak = max over the pass (and >= needed total at the end of forward).
Two bases:
    resident  = act_peak + fp32 weights + (grad + Adam m, v) for trainable params      [compare with PyTorch]
    profiler  = act_peak + trainable x 4 B + frozen x 1 B (TinyTL profiler's param_size) [compare with profiler]
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch, torch.nn as nn
from mem_harness import build, kind  # sets sys.path for ofa / tinytl
from ofa.utils.layers import ResidualBlock
from engine_bound import gn_to_affine, FrozenAffine
from tinytl.utils import profile_memory_cost

MiB = 2 ** 20


def bound(net, bs):
    events, stack, rstack, hooks, savers = [], [], [], [], {}
    sbytes = {}

    def tag(mod):
        if mod is None:
            return 'other'
        k = kind(mod)
        if k in ('conv', 'linear'):
            return k + ('_train' if mod.weight.requires_grad else '_frozen')
        return k

    def key(t):
        st = t.untyped_storage(); return st.data_ptr(), st.nbytes()

    for m in net.modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_pre_hook(lambda mod, inp: stack.append(mod)))

            def post(mod, inp, out):
                stack.pop()
                ik, ib = key(inp[0]); ok, _ = key(out)
                ib = inp[0].numel() * 4; ob = out.numel() * 4
                k = kind(mod)
                # TinyTL profiler's transient convention: conv = x + y / groups (in-place depthwise),
                # linear = x + y, norm / activation / other = x (in place)
                if k == 'conv':
                    ob = ob // mod.groups
                elif k != 'linear':
                    ob = 0
                rk, rb = rstack[-1] if rstack else (None, 0)
                events.append(('op', ik, ib, ok, ob, rk, rb))
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
    xk, xb = key(x)
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
    full_keys, counted = set(), set()
    total = 0; peak = 0
    for e in events:
        if e[0] == 'pack':
            k = e[1]
            if k not in counted:
                counted.add(k); kind_, b = need[k]; total += b
                if kind_ == 'full':
                    full_keys.add(k)
        else:
            _, ik, ib, ok, ob, rk, rb = e
            live = total
            seen = set()
            for k_, b_ in ((ik, ib), (ok, ob), (rk, rb)):
                if k_ is None or b_ == 0 or k_ in full_keys or k_ in seen:
                    continue
                seen.add(k_); live += b_
            peak = max(peak, live)
    peak = max(peak, total)
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    resident = peak + (trainable + frozen) * 4 + 3 * 4 * trainable + xb  # + input batch held by the loop
    prof_basis = peak + trainable * 4 + frozen * 1
    return peak / MiB, resident / MiB, prof_basis / 1e6, total / MiB


if __name__ == '__main__':
    meas = {(r['arm'], r['bs']): r.get('measured_MiB') for r in
            json.loads(Path(__file__).with_name('engine_bound_result.json').read_text())}
    rows = []
    for arm in ['last', 'T4', 'T3', 'T2', 'LB', 'B', 'full', 'LB+aff', 'B+aff', 'T3+aff']:
        for bs in (8, 1):
            base = arm.replace('+aff', '')
            net = build(base)
            prof, _ = profile_memory_cost(net, (1, 3, 224, 224), base != 'last', 32, 32, 8, batch_size=bs)
            if arm.endswith('+aff'):
                net = gn_to_affine(net)
            net = net.cuda().train()
            act_peak, resident, prof_basis, saved_end = bound(net, bs)
            del net; torch.cuda.empty_cache()
            m = meas.get((arm.replace('+aff', '+affine(no GN)'), bs))
            row = dict(arm=arm, bs=bs, profiler_MB=round(prof / 1e6, 1), engine_profiler_basis_MB=round(prof_basis, 1),
                       engine_resident_MiB=round(resident, 1), act_peak_MiB=round(act_peak, 1),
                       saved_at_fwd_end_MiB=round(saved_end, 1), measured_MiB=m)
            rows.append(row); print(json.dumps(row), flush=True)
    Path(__file__).with_name('engine_bound_r3_result.json').write_text(json.dumps(rows, indent=1), encoding='utf-8')
