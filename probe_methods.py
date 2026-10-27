"""Paper estimate vs measured CUDA memory for TinyTL's six transfer methods."""
import os, sys
from pathlib import Path
ROOT = Path(r"C:\Users\zhita\Documents\Codex\2026-09-10\this-class-will-feature-a-quarter\outputs\TinyTL")
os.environ['TINYTL_CACHE'] = str(ROOT / 'cache'); os.environ['TORCH_HOME'] = str(ROOT / 'cache' / 'torch')
sys.path[:0] = [str(ROOT / 'upstream' / 'once-for-all'), str(ROOT / 'upstream' / 'tinytl')]
import torch, torch.nn as nn
from ofa.utils.layers import LinearLayer
from ofa.model_zoo import proxylessnas_mobile
from ofa.utils import replace_conv2d_with_my_conv2d, replace_bn_with_gn
from tinytl.utils import set_module_grad_status, enable_bias_update, enable_bn_update, profile_memory_cost
from tinytl.model import LiteResidualModule

METHODS = ['last', 'bn+last', 'tinytl-bias', 'tinytl-lite_residual', 'tinytl-lite_residual+bias', 'full']
PAPER = {'last': 31, 'bn+last': 192, 'tinytl-bias': 32, 'tinytl-lite_residual': 37,
         'tinytl-lite_residual+bias': 37, 'full': 391}


def build(method):
    net = proxylessnas_mobile(pretrained=False)
    LiteResidualModule.insert_lite_residual(net, 2, 'bilinear', 1.0, 5, 'relu', 2)
    replace_bn_with_gn(net, gn_channel_per_group=8)
    net.load_state_dict(torch.load(ROOT / 'cache' / 'proxylessnas_mobile+lite_residual@imagenet@ws+gn',
                                   map_location='cpu')['state_dict'])
    net.classifier = LinearLayer(net.classifier.in_features, 102, dropout_rate=0.2)
    set_module_grad_status(net, method == 'full')
    set_module_grad_status([net.classifier], True)
    if method == 'bn+last':
        enable_bn_update(net)
    if method in ('tinytl-bias', 'tinytl-lite_residual+bias'):
        enable_bias_update(net)
    if method.startswith('tinytl-lite_residual'):
        for m in net.modules():
            if isinstance(m, LiteResidualModule):
                set_module_grad_status(m.lite_residual, True)
                if method.endswith('+bias'):
                    m.lite_residual.final_bn.bias.requires_grad = False
    replace_conv2d_with_my_conv2d(net, 1e-5)
    return net


def measure(method, bs):
    net = build(method)
    est, _ = profile_memory_cost(net, (1, 3, 224, 224), True, 32, 32, 8, batch_size=bs)
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    net = net.cuda().train()
    weights = torch.cuda.memory_allocated() - base
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=2e-4)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(bs, 3, 224, 224, device='cuda'); y = torch.randint(0, 102, (bs,), device='cuda')
    for _ in range(2):
        opt.zero_grad(set_to_none=True); crit(net(x), y).backward(); opt.step()
    torch.cuda.synchronize()
    steady = torch.cuda.memory_allocated() - base          # weights + optimizer state + input batch
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        opt.zero_grad(set_to_none=True); crit(net(x), y).backward(); opt.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del net, opt, x, y; torch.cuda.empty_cache()
    return est / 1e6, peak / 2 ** 20, weights / 2 ** 20, steady / 2 ** 20, trainable


for bs in (8, 1):
    print(f'\n=== batch {bs} ===')
    print(f"{'method':<28}{'paper':>7}{'profiler':>10}{'measured':>10}{'weights':>9}{'steady':>9}{'trainable':>11}")
    rows = {}
    for m in METHODS:
        est, peak, w, st, tr = measure(m, bs)
        rows[m] = (est, peak)
        print(f"{m:<28}{PAPER[m] if bs == 8 else '':>7}{est:>9.1f}M{peak:>9.1f}M{w:>8.1f}M{st:>8.1f}M{tr:>11,}")
    lb = rows['tinytl-lite_residual+bias']
    print(f"Full / L+B ratio: profiler {rows['full'][0] / lb[0]:.2f}x   measured {rows['full'][1] / lb[1]:.2f}x")
