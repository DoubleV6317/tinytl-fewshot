"""Reviewer check: Monte Carlo of the H1 / H2 decision rules with sample-SD variability and the extension look."""
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
R = 200_000


def ci(x, conf=0.90):
    n = x.shape[1]
    m = x.mean(1); s = x.std(1, ddof=1)
    h = stats.t.ppf(1 - (1 - conf) / 2, n - 1) * s / np.sqrt(n)
    return m - h, m + h


def h1(d, sd, n1=10, n2=15, delta=2.0):
    x = rng.normal(d, sd, size=(R, n2))
    L, U = ci(x[:, :n1])
    ret1 = L > -delta
    inc1 = (L <= -delta) & (U >= 0)
    L2, U2 = ci(x)
    ret = ret1 | (inc1 & (L2 > -delta))
    inc_final = inc1 & (L2 <= -delta) & (U2 >= 0)
    return ret1.mean(), inc1.mean(), ret.mean(), inc_final.mean()


print('H1: P(retain at n=10), P(inconclusive at n=10), P(retain after extension), P(still inconclusive)')
for sd in (1.8, 2.5, 3.0):
    for d in (0.0, -0.5, -1.0, -1.5, -2.0):
        r1, i1, r, i = h1(d, sd)
        print(f' sd {sd} d {d:+.1f}: {r1:.3f} {i1:.3f} {r:.3f} {i:.3f}')

# Corrected labels: 'loses more than margin' should need U < -2; how often does the proposal's label
# (U < 0 and L <= -2) fire when the true loss is only 1 pp?
print('\nP(proposal label "loses more than the margin" | n=10) vs P(U < -2)')
for sd in (1.8, 2.5):
    for d in (-1.0, -1.5, -2.5, -3.0):
        x = rng.normal(d, sd, size=(R, 10)); L, U = ci(x)
        print(f' sd {sd} d {d:+.1f}: proposal {((U < 0) & (L <= -2)).mean():.3f}   U<-2 {(U < -2).mean():.3f}')

# H2 with extension 6 -> 10
print('\nH2: P(no meaningful degradation, after ext.), P(catastrophic U<-5 at n=6)')
for sd in (1.8, 2.5, 3.0):
    for d in (0.0, -1.0, -2.0, -5.0, -8.0):
        x = rng.normal(d, sd, size=(R, 10))
        L, U = ci(x[:, :6]); L2, U2 = ci(x)
        ok1 = L > -2; inc1 = (L <= -2) & (U >= 0)
        ok = ok1 | (inc1 & (L2 > -2))
        print(f' sd {sd} d {d:+.1f}: noDeg {ok.mean():.3f}  cat {(U < -5).mean():.3f}  inconcl@6 {inc1.mean():.3f}')

# Planning SD rests on 3 paired differences (2 df): CI for sigma
s, df = 1.83, 2
lo, hi = [s * np.sqrt(df / stats.chi2.ppf(q, df)) for q in (0.95, 0.05)]
print(f'\n90% CI for paired SD from 3 seeds (s=1.83): [{lo:.2f}, {hi:.2f}]')
print('implied unpaired SD of r1-r2 if independent:', np.sqrt(0.82 ** 2 + 1.10 ** 2).round(2))
