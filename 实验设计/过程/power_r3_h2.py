"""H2 primary rule at n=5: 'not catastrophic' L > -5, 'catastrophic' U < -5, else undetermined (93% t-CI)."""
import numpy as np
from scipy import stats
rng = np.random.default_rng(5); R = 100_000; n = 5
h = stats.t.ppf(0.965, n - 1)
print(f't(0.965, {n-1}) = {h:.3f}')
for sd in (2.5, 3.5):
    for d in (0, -2, -3, -5, -7, -10):
        x = rng.normal(d, sd, (R, n)); m = x.mean(1); s = x.std(1, ddof=1) * h / np.sqrt(n)
        L, U = m - s, m + s
        print(f'sd {sd} d {d:+3d}: not-catastrophic {np.mean(L > -5):.2f}  catastrophic {np.mean(U < -5):.2f}  '
              f'undetermined {np.mean((L <= -5) & (U >= -5)):.2f}  | within-2pp {np.mean(L > -2):.2f}')
