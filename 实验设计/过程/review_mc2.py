"""Reviewer check: per-look CI level that keeps the two-look (n=10 -> 15) NI type-I error at or below 5%."""
import numpy as np
from scipy import stats
rng = np.random.default_rng(1)
R = 200_000
def ci(x, conf):
    n = x.shape[1]; m = x.mean(1); s = x.std(1, ddof=1)
    h = stats.t.ppf(1 - (1 - conf) / 2, n - 1) * s / np.sqrt(n)
    return m - h, m + h
for conf in (0.90, 0.92, 0.93, 0.94):
    out = []
    for sd in (1.8, 2.5, 3.0):
        x = rng.normal(-2.0, sd, size=(R, 15))
        L, U = ci(x[:, :10], conf); L2, _ = ci(x, conf)
        ret = (L > -2) | ((L <= -2) & (U >= 0) & (L2 > -2))
        out.append(f'sd {sd}: {ret.mean():.3f}')
    # power at d=0, sd 1.8 / 2.5
    pw = []
    for sd in (1.8, 2.5):
        x = rng.normal(0.0, sd, size=(R, 15))
        L, U = ci(x[:, :10], conf); L2, _ = ci(x, conf)
        ret = (L > -2) | ((L <= -2) & (U >= 0) & (L2 > -2))
        pw.append(f'P(retain|d=0,sd {sd}) {ret.mean():.3f}')
    print(f'conf {conf}: type-I ' + ', '.join(out) + ' | ' + ', '.join(pw))
