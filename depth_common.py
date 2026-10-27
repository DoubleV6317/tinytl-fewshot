"""Shared by fewshot_depth.py and measure_memory.py: paths, network build and the arm definitions.

An arm only decides which parameters receive gradients. Every arm keeps all lite residuals in the forward pass.
  last   classifier only (upstream `last`)
  T0-T5  biases + lite residuals from the first block of stage s onward, plus feature-mix bias and classifier
  LB     TinyTL-L+B: all biases + all lite residuals + classifier (same parameter set as upstream)
  B      all biases except lite_residual.final_bn.bias + classifier (lite residuals frozen). Upstream's tinytl-bias
         also trains those 19 final_bn biases (2,016 params); B is only used in the memory table.
  full   every parameter
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


def setup_paths():
    os.environ.setdefault('TINYTL_CACHE', str(ROOT / 'cache'))
    os.environ.setdefault('TORCH_HOME', str(ROOT / 'cache' / 'torch'))
    for p in (ROOT / 'upstream' / 'tinytl', ROOT / 'upstream' / 'once-for-all'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def expected_trainable(arm, n_classes):
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
    if arm in ARM_START:
        start, lite = ARM_START[arm], arm != 'B'
        for name, p in net.named_parameters():
            if block_index(name) < start:
                continue
            if name.endswith('.bias') and 'lite_residual.final_bn' not in name:
                p.requires_grad = True
            if lite and '.lite_residual.' in name and 'lite_residual.final_bn.bias' not in name:
                p.requires_grad = True
    elif arm not in ('last', 'full'):
        raise ValueError('unknown arm %s' % arm)
    set_module_grad_status([net.classifier], True)
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
