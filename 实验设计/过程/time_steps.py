"""GPU time per training step (synthetic data) at batch 1 and 8, to extrapolate run lengths."""
import sys, time
sys.path.insert(0, r"C:\Users\zhita\AppData\Local\Temp\claude\C--Users-zhita--claude-spares-spare-YrkVlb\a66d6077-f64b-4862-9bc2-80f8d7234878\scratchpad")
import torch, torch.nn as nn
from probe_depth import build

variants = [('L+B full', -1, True, True), ('L+B from stage 3', 13, True, True), ('head only', 10 ** 6 - 1, True, True)]
for label, start, b, l in variants:
    for bs in (1, 8):
        net, _ = build(start, b, l)
        net = net.cuda().train()
        opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=2e-4)
        crit = nn.CrossEntropyLoss()
        x = torch.randn(bs, 3, 224, 224, device='cuda'); y = torch.randint(0, 102, (bs,), device='cuda')
        for _ in range(10):
            opt.zero_grad(set_to_none=True); crit(net(x), y).backward(); opt.step()
        torch.cuda.synchronize(); t = time.time()
        n = 40
        for _ in range(n):
            opt.zero_grad(set_to_none=True); crit(net(x), y).backward(); opt.step()
        torch.cuda.synchronize()
        ms = (time.time() - t) / n * 1000
        print(f"{label:<20} batch {bs}: {ms:6.1f} ms/step  {ms / bs:6.1f} ms/image", flush=True)
        del net, opt; torch.cuda.empty_cache()
