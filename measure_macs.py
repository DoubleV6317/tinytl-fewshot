"""Analytic MACs per training image for every arm (实验设计/最终实验设计.md, Appendix A; from 过程/macs.py). CPU only.

  python measure_macs.py          writes macs_table.json

Conv and linear layers only, 224 px, one image. forward = every conv/linear; backward = grad-input MACs for every
layer whose input needs a gradient + grad-weight MACs for every layer with a trainable weight (TinyTrain Table 2
convention). Norms, activations, pooling and the bilinear upsample are not counted. The forward is the same for
every arm, because every arm keeps all lite residuals in the forward pass.
"""
import json

from depth_common import ROOT, ARMS, build_net, set_trainable

DATASET_CLASSES = {'flowers102': 102, 'aircraft': 100}
# 过程/macs_result.json (102-way head), M MACs: the harness must reproduce it (the design summed the rounded parts).
DESIGN_102 = {'last': 0.1, 'T4': 125.2, 'T3': 227.2, 'T2': 297.5, 'LB': 459.1, 'B': 383.8, 'full': 778.3}


def count_macs(arm, n_classes, image_size=224):
    """dict(fwd, bwd_in, bwd_w, bwd_total) in MACs (not millions) for one image."""
    import torch
    import torch.nn as nn
    net = build_net(n_classes)
    set_trainable(net, arm)
    net.train()
    acc = dict(fwd=0, bwd_in=0, bwd_w=0)

    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            macs = out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        else:
            macs = out.numel() * m.in_features
        acc['fwd'] += macs
        if inp[0].requires_grad:
            acc['bwd_in'] += macs
        if m.weight.requires_grad:
            acc['bwd_w'] += macs

    hooks = [m.register_forward_hook(hook) for m in net.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.enable_grad():
        net(torch.randn(1, 3, image_size, image_size))
    for h in hooks:
        h.remove()
    acc['bwd_total'] = acc['bwd_in'] + acc['bwd_w']
    return acc


def main():
    arms = [a for a in ARMS if a != 'head']
    table = {}
    for dataset, n_classes in DATASET_CLASSES.items():
        table[dataset] = {}
        for arm in arms:
            r = {k: round(v / 1e6, 1) for k, v in count_macs(arm, n_classes).items()}
            r['train_per_image'] = round(r['fwd'] + r['bwd_total'], 1)
            table[dataset][arm] = r
        lb = table[dataset]['LB']['bwd_total']
        for arm in arms:
            table[dataset][arm]['bwd_vs_LB'] = round(table[dataset][arm]['bwd_total'] / lb, 3)
    for arm, v in DESIGN_102.items():
        got = table['flowers102'][arm]['bwd_total']
        assert abs(got - v) < 0.11, '%s: %.1f M backward MACs, design says %.1f' % (arm, got, v)
    print('M MACs per image, 224 px (forward is the same for every arm)')
    print('%-5s %8s %8s %8s %9s %10s %7s' % ('arm', 'fwd', 'bwd_in', 'bwd_w', 'bwd_tot', 'fwd+bwd', 'vs LB'))
    for dataset in table:
        print(dataset)
        for arm, r in table[dataset].items():
            print('%-5s %8.1f %8.1f %8.1f %9.1f %10.1f %7.3f' % (
                arm, r['fwd'], r['bwd_in'], r['bwd_w'], r['bwd_total'], r['train_per_image'], r['bwd_vs_LB']))
    print('backward MACs match the design (macs_result.json) for %s' % sorted(DESIGN_102))
    out = ROOT / 'macs_table.json'
    out.write_text(json.dumps(dict(unit='M MACs per image', image_size=224, table=table), indent=1),
                   encoding='utf-8')
    print('wrote', out)


if __name__ == '__main__':
    main()
