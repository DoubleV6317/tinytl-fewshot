"""Shared by fewshot_depth.py and measure_memory.py: paths, network build and the arm definitions.

An arm only decides which parameters receive gradients. Every arm keeps all lite residuals in the forward pass.
  last   classifier only (upstream `last`)
  T0-T5  biases + lite residuals from the first block of stage s onward, plus feature-mix bias and classifier
  LB     TinyTL-L+B: all biases + all lite residuals + classifier (same parameter set as upstream)
  B      all biases except lite_residual.final_bn.bias + classifier (lite residuals frozen). Upstream's tinytl-bias
         also trains those 19 final_bn biases (2,016 params); B is only used in the memory table.
  full   every parameter
  S...   block set (block N4, 实验设计/补充实验设计.md): biases + lite residual of the listed blocks only, plus
         feature-mix bias and classifier. Gradients still flow through every block after the earliest listed one.
         Lite-residual blocks are 1, 2, 5-21; a set is named by its runs of consecutive lite blocks joined by '_',
         e.g. S16-21 or S13_17-21, and a set equal to a T arm is that T arm (blockset_arm)
Frozen conv/linear weights are 8-bit k-means quantized (quantize_frozen), so arms also differ in which layers are
quantized, as in upstream.
"""
from pathlib import Path
import hashlib
import os
import re
import sys

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / 'cache' / 'proxylessnas_mobile+lite_residual@imagenet@ws+gn'
QUANT_CACHE = ROOT / 'cache' / 'quant8_rs0.pth'

HEAD = 10 ** 6 - 1  # start index that leaves only feature_mix_layer and the classifier
ARM_START = {'LB': -1, 'B': -1, 'T0': 1, 'T1': 5, 'T2': 9, 'T3': 13, 'T4': 17, 'T5': 21, 'head': HEAD}
ARMS = ['last', 'T5', 'T4', 'T3', 'T2', 'T1', 'T0', 'LB', 'B', 'full', 'head']

# Trainable parameter counts with a 102-way head (proposal Table M); the head has 1281 params per class.
EXPECTED_102 = {'last': 130662, 'T5': 362598, 'T4': 1113126, 'T3': 1578246, 'T2': 1863766, 'T1': 1944182,
                'T0': 1962230, 'LB': 1962310, 'B': 149622, 'full': 4746590, 'head': 131942}

# Block-set arms: the blocks with a lite residual (block 0 has none; 3 and 4 are ZeroLayer) and the parameters each
# one trains (its biases + lite residual, as in the T arms). The T arms add the feature-mix bias and the head.
LITE_BLOCKS = [1, 2] + list(range(5, 22))
BLOCK_PARAMS = {1: 3904, 2: 14144, 5: 14416, 6: 22000, 7: 22000, 8: 22000, 9: 23920, 10: 87200, 11: 87200,
                12: 87200, 13: 88992, 14: 125376, 15: 125376, 16: 125376, 17: 135360, 18: 205824, 19: 204672,
                20: 204672, 21: 230656}
MIX_BIAS, HEAD_PER_CLASS = 1280, 1281
T_ARMS = ['T5', 'T4', 'T3', 'T2', 'T1', 'T0']
for _a in T_ARMS:
    assert EXPECTED_102[_a] == sum(v for b, v in BLOCK_PARAMS.items() if b >= ARM_START[_a]) + MIX_BIAS + \
        102 * HEAD_PER_CLASS, _a


def setup_paths():
    os.environ.setdefault('TINYTL_CACHE', str(ROOT / 'cache'))
    os.environ.setdefault('TORCH_HOME', str(ROOT / 'cache' / 'torch'))
    for p in (ROOT / 'upstream' / 'tinytl', ROOT / 'upstream' / 'once-for-all'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def blockset_arm(blocks):
    """Canonical arm of a set of lite-residual blocks: the T arm (or head) with the same parameters if there is one,
    else S + the runs of consecutive lite blocks."""
    bs = sorted(set(blocks))
    assert all(b in BLOCK_PARAMS for b in bs), bs
    if not bs:
        return 'head'
    for a in T_ARMS:
        if bs == [b for b in LITE_BLOCKS if b >= ARM_START[a]]:
            return a
    pos, runs = {b: i for i, b in enumerate(LITE_BLOCKS)}, []
    for b in bs:
        if runs and pos[b] == pos[runs[-1][1]] + 1:
            runs[-1][1] = b
        else:
            runs.append([b, b])
    return 'S' + '_'.join(str(lo) if lo == hi else '%d-%d' % (lo, hi) for lo, hi in runs)


def parse_blockset(arm):
    """The sorted blocks of a block-set arm, or None if arm is not one; ValueError if it is not canonical."""
    if not arm.startswith('S'):
        return None
    blocks = []
    for part in arm[1:].split('_'):
        m = re.fullmatch(r'(\d+)(?:-(\d+))?', part)
        if not m:
            raise ValueError('malformed block set %s' % arm)
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        blocks += [b for b in LITE_BLOCKS if lo <= b <= hi]
    if blockset_arm(blocks) != arm:
        raise ValueError('%s is not the canonical name of its block set (%s)' % (arm, blockset_arm(blocks)))
    return sorted(blocks)


def check_arm(arm):
    """argparse type: a named arm or a canonical block-set arm."""
    if arm not in ARMS and parse_blockset(arm) is None:
        raise ValueError('unknown arm %s' % arm)
    return arm


def expected_trainable(arm, n_classes):
    blocks = parse_blockset(arm)
    if blocks is not None:
        return sum(BLOCK_PARAMS[b] for b in blocks) + MIX_BIAS + n_classes * HEAD_PER_CLASS
    return EXPECTED_102[arm] - (102 - n_classes) * 1281


def lr_str(lr):
    mant, exp = ('%.3e' % lr).split('e')
    s = mant.rstrip('0').rstrip('.') + 'e' + str(int(exp))
    assert abs(float(s) - lr) <= 1e-12 * max(1.0, abs(lr)), 'lr %r does not round-trip through %s' % (lr, s)
    return s


HEAD_INITS = ['random', 'imprint']


def run_name(dataset, k, arm, batch, lr, seed, epochs=50, val_every=5, tag='', head_init='random'):
    # val_every is part of the name: the val loader is shuffled, so it changes the training data order.
    assert head_init in HEAD_INITS, head_init
    name = 'depth-%s-k%d-%s-b%d-lr%s-s%d' % (dataset, k, arm, batch, lr_str(lr), seed)
    if epochs != 50:
        name += '-e%d' % epochs
    if val_every != 5:
        name += '-v%d' % val_every
    if head_init == 'imprint':
        name += '-imp'
    return name + ('-' + tag if tag else '')


def sha1_tensors(tensors):
    h = hashlib.sha1()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def quantize_frozen(net, bits=8, max_iter=20):
    """Upstream weight_quantization (8-bit k-means of every frozen nn.Conv2d / nn.Linear weight), made
    deterministic and cached. sklearn's KMeans is not bit-reproducible with several OpenMP threads even with a fixed
    random_state, so each layer is quantized single-threaded with random_state=0. A layer's result depends only on
    its fp32 weights, so it is cached by layer name + weight hash and shared by every arm and seed that freezes it.
    Returns (number of quantized layers, number computed now)."""
    import sklearn
    import torch
    import torch.nn as nn
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    from tinytl.utils.common_utils import reconstruct_weight_from_k_means_result
    cache = torch.load(QUANT_CACHE, map_location='cpu', weights_only=False) if QUANT_CACHE.exists() else {}
    n_done, n_new = 0, 0
    for name, m in net.named_modules():
        if not isinstance(m, (nn.Conv2d, nn.Linear)) or m.weight.requires_grad:
            continue
        w = m.weight.data
        key = '%s|%s|%d|%d|sklearn%s' % (name, sha1_tensors([w]), bits, max_iter, sklearn.__version__)
        if key not in cache:
            flat = w.cpu().numpy().reshape(-1, 1)
            with threadpool_limits(1):
                km = KMeans(n_clusters=min(2 ** bits, flat.size), init='k-means++', n_init=1, max_iter=max_iter,
                            random_state=0).fit(flat)
            centroids = torch.from_numpy(km.cluster_centers_).view(1, -1)
            labels = torch.from_numpy(km.labels_.reshape(w.shape)).int()
            cache[key] = reconstruct_weight_from_k_means_result(centroids, labels).float()
            n_new += 1
        m.weight.data = cache[key].clone().to(w.device)
        n_done += 1
    if n_new:
        tmp = QUANT_CACHE.with_suffix('.tmp')
        torch.save(cache, tmp)
        os.replace(tmp, QUANT_CACHE)
    return n_done, n_new


def block_index(name):
    if name.startswith('first_conv'):
        return -1
    m = re.match(r'blocks\.(\d+)\.', name)
    return int(m.group(1)) if m else 10 ** 6  # feature_mix_layer / classifier


def build_net(n_classes, dropout=0.2):
    """ProxylessNAS-Mobile + official r=2 lite residuals + GN, loaded from the ws+gn checkpoint, new head.
    Weight standardization is not applied yet (call replace_conv2d_with_my_conv2d after quantization)."""
    setup_paths()
    import torch
    from ofa.model_zoo import proxylessnas_mobile
    from ofa.utils import replace_bn_with_gn, init_models
    from ofa.utils.layers import LinearLayer
    from tinytl.model import LiteResidualModule
    net = proxylessnas_mobile(pretrained=False)
    LiteResidualModule.insert_lite_residual(net, 2, 'bilinear', 1.0, 5, 'relu', 2)
    replace_bn_with_gn(net, gn_channel_per_group=8)
    net.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=False)['state_dict'])
    net.classifier = LinearLayer(net.classifier.in_features, n_classes, dropout_rate=dropout)
    init_models([net.classifier])
    return net


def set_trainable(net, arm):
    """Set requires_grad for an arm and return the trainable parameter count."""
    from tinytl.utils import set_module_grad_status
    set_module_grad_status(net, arm == 'full')
    blocks = parse_blockset(arm)
    if arm in ARM_START or blocks is not None:
        if blocks is not None:
            keep = lambda b: b in blocks or b == 10 ** 6  # the listed blocks, feature_mix_layer and the classifier
        else:
            keep = lambda b: b >= ARM_START[arm]
        lite = arm != 'B'
        for name, p in net.named_parameters():
            if not keep(block_index(name)):
                continue
            if name.endswith('.bias') and 'lite_residual.final_bn' not in name:
                p.requires_grad = True
            if lite and '.lite_residual.' in name and 'lite_residual.final_bn.bias' not in name:
                p.requires_grad = True
    elif arm not in ('last', 'full'):
        raise ValueError('unknown arm %s' % arm)
    set_module_grad_status([net.classifier], True)
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
