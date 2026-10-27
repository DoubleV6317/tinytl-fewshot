"""Measured train-step memory when TinyTL-L+B updates only the last stages (truncated backprop)."""
import os, sys, re
from pathlib import Path
ROOT = Path(r"C:\Users\zhita\Documents\Codex\2026-09-10\this-class-will-feature-a-quarter\outputs\TinyTL")
os.environ['TINYTL_CACHE'] = str(ROOT / 'cache'); os.environ['TORCH_HOME'] = str(ROOT / 'cache' / 'torch')
sys.path[:0] = [str(ROOT / 'upstream' / 'once-for-all'), str(ROOT / 'upstream' / 'tinytl')]
import torch, torch.nn as nn
from ofa.utils.layers import LinearLayer
from ofa.model_zoo import proxylessnas_mobile
from ofa.utils import replace_conv2d_with_my_conv2d, replace_bn_with_gn
from tinytl.utils import set_module_grad_status, profile_memory_cost
from tinytl.model import LiteResidualModule


def block_index(name):
    if name.startswith('first_conv'):
        return -1
    m = re.match(r'blocks\.(\d+)\.', name)
    return int(m.group(1)) if m else 10 ** 6  # feature_mix_layer / classifier


def build(start_block, bias=True, lite=True):
    net = proxylessnas_mobile(pretrained=False)
    LiteResidualModule.insert_lite_residual(net, 2, 'bilinear', 1.0, 5, 'relu', 2)
    replace_bn_with_gn(net, gn_channel_per_group=8)
    net.load_state_dict(torch.load(ROOT / 'cache' / 'proxylessnas_mobile+lite_residual@imagenet@ws+gn',
                                   map_location='cpu', weights_only=False)['state_dict'])
    net.classifier = LinearLayer(net.classifier.in_features, 102, dropout_rate=0.2)
    groups = net.grouped_block_index
    set_module_grad_status(net, False)
    for name, p in net.named_parameters():
        if block_index(name) < start_block:
            continue
        if bias and name.endswith('.bias') and 'lite_residual.final_bn' not in name:
            p.requires_grad = True
        if lite and '.lite_residual.' in name and 'lite_residual.final_bn.bias' not in name:
            p.requires_grad = True
    set_module_grad_status([net.classifier], True)
    replace_conv2d_with_my_conv2d(net, 1e-5)
    return net, groups


def measure(net, bs=8):
    est, _ = profile_memory_cost(net, (1, 3, 224, 224), True, 32, 32, 8, batch_size=bs)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    net = net.cuda().train()
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=2e-4)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(bs, 3, 224, 224, device='cuda'); y = torch.randint(0, 102, (bs,), device='cuda')
    for _ in range(2):
        opt.zero_grad(set_to_none=True); crit(net(x), y).backward(); opt.step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        opt.zero_grad(set_to_none=True); crit(net(x), y).backward(); opt.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    n = sum(p.numel() for p in params)
    del net, opt, x, y; torch.cuda.empty_cache()
    return est / 1e6, peak / 2 ** 20, n


if __name__ == '__main__':
    _, groups = build(0)
    print('grouped_block_index:', groups)
    rows = [('L+B full (from stem)', -1, True, True),
            ('B, skip first_conv', 0, True, False),
            ('B, skip first_conv+blocks.0', 1, True, False)]
    for s, idx in enumerate(groups):
        rows.append(('L+B from stage %d (block %d)' % (s, idx[0]), idx[0], True, True))
    rows.append(('L+B feature_mix+head only', 10 ** 6 - 1, True, True))
    print(f"{'variant':<34}{'profiler':>10}{'measured':>11}{'trainable':>11}")
    for label, start, b, l in rows:
        net, _ = build(start, b, l)
        est, peak, n = measure(net)
        print(f"{label:<34}{est:>9.1f}M{peak:>9.1f}Mi{n:>11,}", flush=True)
