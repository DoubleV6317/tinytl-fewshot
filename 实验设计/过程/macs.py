"""Analytic MACs per image (conv + linear only, 224 px): forward, and backward = grad-input MACs for every layer whose
input needs a gradient + grad-weight MACs for trainable layers (TinyTrain Table 2 convention). CPU only, no training."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch, torch.nn as nn
from mem_harness import build
rows = {}
for arm in ['last', 'T4', 'T3', 'T2', 'LB', 'B', 'full']:
    net = build(arm).train()
    acc = dict(fwd=0, bwd_in=0, bwd_w=0)
    def hook(m, inp, out):
        x = inp[0]
        if isinstance(m, nn.Conv2d):
            macs = out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        else:
            macs = out.numel() * m.in_features
        acc['fwd'] += macs
        if x.requires_grad:
            acc['bwd_in'] += macs
        if m.weight.requires_grad:
            acc['bwd_w'] += macs
    hs = [m.register_forward_hook(hook) for m in net.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    net(torch.randn(1, 3, 224, 224))
    for h in hs: h.remove()
    r = {k: round(v / 1e6, 1) for k, v in acc.items()}
    r['bwd_total'] = round(r['bwd_in'] + r['bwd_w'], 1)
    rows[arm] = r
    print(arm, r, flush=True)
Path(__file__).with_name('macs_result.json').write_text(json.dumps(rows, indent=1))
