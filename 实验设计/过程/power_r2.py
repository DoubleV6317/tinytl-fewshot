"""r2 planning: Monte Carlo of the four-label rules with sample-SD variability, 93% two-sided CIs at every look,
one extension look (n1 -> n2) only for 'inconclusive'. Normal paired differences."""
import numpy as np
from scipy import stats
rng = np.random.default_rng(7)
R = 100_000
CONF = 0.93

def ci(x):
    n = x.shape[1]; m = x.mean(1); s = x.std(1, ddof=1)
    h = stats.t.ppf(1 - (1 - CONF) / 2, n - 1) * s / np.sqrt(n)
    return m - h, m + h

def labels(L, U, cat=None):
    lab = np.full(L.shape, 'unres', dtype=object)          # U<0, L<=-2, U>=-2 : significant loss, size unresolved
    lab[L > -2] = 'retains'
    lab[(L <= -2) & (U >= 0)] = 'inconcl'
    lab[U < -2] = 'inferior'
    if cat:
        lab[(U < -2) & (U >= -5)] = 'moderate'
        lab[U < -5] = 'catastr'
    return lab

def run(d, sd, n1, n2, cat=False):
    x = rng.normal(d, sd, size=(R, n2))
    L, U = ci(x[:, :n1]); lab = labels(L, U, cat)
    inc = lab == 'inconcl'
    L2, U2 = ci(x); lab2 = labels(L2, U2, cat)
    final = np.where(inc, lab2, lab)
    keys = ['retains', 'unres', 'inconcl'] + (['moderate', 'catastr'] if cat else ['inferior'])
    return inc.mean(), {k: (final == k).mean() for k in keys}

def show(title, n1, n2, ds, sds, cat=False):
    print(title)
    for sd in sds:
        for d in ds:
            pinc, f = run(d, sd, n1, n2, cat)
            print(f'  sd {sd} d {d:+5.1f}: P(ext fires) {pinc:.2f} | final ' + ' '.join(f'{k} {v:.2f}' for k, v in f.items()))

show('H1 Flowers n=10 -> 15', 10, 15, (0, -1, -2, -3, -5), (1.8, 2.5, 3.0))
show('H1 Aircraft n=5 -> 10', 5, 10, (0, -2, -3, -6, -10, -17), (2.5, 3.5))
show('H2 n=6 -> 10 (moderate / catastrophic labels)', 6, 10, (0, -1, -3, -6, -10), (1.8, 2.5, 3.0), cat=True)
# type-I error of 'retains' at d = -2 (the margin), with extension
for n1, n2 in ((10, 15), (5, 10), (6, 10)):
    for sd in (1.8, 2.5, 3.0):
        _, f = run(-2.0, sd, n1, n2)
        print(f'type-I(retains | d=-2) n {n1}->{n2} sd {sd}: {f["retains"]:.3f}')
