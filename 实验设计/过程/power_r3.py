"""r3 planning MC for the FIXED design (no extension looks). 93% two-sided t-CIs, four labels,
Holm (one-sided 0.035) over the NI claims of {T3, T4, FT-Last}; T2 exploratory (unadjusted).
Noise model: acc = mu_arm + seed_effect + eps_arm, so paired d shares LB's eps (corr 0.5)."""
import numpy as np
from scipy import stats
rng = np.random.default_rng(3)
R = 50_000; A = 0.035

def ci(d):
    n = d.shape[-1]; m = d.mean(-1); s = d.std(-1, ddof=1)
    h = stats.t.ppf(1 - A, n - 1) * s / np.sqrt(n); return m - h, m + h

def p_ni(d):
    n = d.shape[-1]; m = d.mean(-1); s = d.std(-1, ddof=1)
    return 1 - stats.t.cdf((m + 2) / (s / np.sqrt(n)), n - 1)

def lab(L, U):
    return np.where(L > -2, 'retains', np.where(U < -2, 'inferior', np.where(U >= 0, 'inconcl', 'unres')))

def holm(p):
    order = np.argsort(p, 1); m = p.shape[1]; rej = np.zeros_like(p, bool); alive = np.ones(len(p), bool)
    for j in range(m):
        idx = order[:, j]; pj = p[np.arange(len(p)), idx]; ok = alive & (pj < A / (m - j))
        rej[np.arange(len(p))[ok], idx[ok]] = True; alive &= ok
    return rej

def h1(mu, ns, sd_pair, names, holm_set):
    nmax = max(ns); s = sd_pair / np.sqrt(2)
    eps_lb = rng.normal(0, s, (R, nmax))
    labs, ps = {}, {}
    for nm, m_, n in zip(names, mu, ns):
        d = m_ + rng.normal(0, s, (R, n)) - eps_lb[:, :n]
        L, U = ci(d); labs[nm] = lab(L, U); ps[nm] = p_ni(d)
    P = np.stack([ps[nm] for nm in holm_set], 1); rej = holm(P)
    for i, nm in enumerate(holm_set):
        labs[nm] = np.where((labs[nm] == 'retains') & ~rej[:, i], 'retains-unadj', labs[nm])
    for nm in names:
        v, c = np.unique(labs[nm], return_counts=True)
        print(f'    {nm:8s} (n={ns[names.index(nm)]}, prior {mu[names.index(nm)]:+.1f}): ' +
              ', '.join(f'{a} {b / R:.2f}' for a, b in zip(v, c)))

for sdp in (1.8, 2.5):
    print(f'Flowers, paired SD {sdp}')
    h1([-1.0, -1.9, -0.7, -0.8], [10, 10, 10, 5], sdp, ['T3', 'T4', 'FT-Last', 'T2'], ['T3', 'T4', 'FT-Last'])
for sdp in (2.5, 3.5):
    print(f'Aircraft, paired SD {sdp}')
    h1([-3.1, -10.6, -17.6, -2.6], [6, 6, 5, 6], sdp, ['T3', 'T4', 'FT-Last', 'T2'], ['T3', 'T4', 'FT-Last'])

def h2(n):
    print(f'H2 (LB-b1 - LB-b8), n={n}')
    for sd in (2.5, 3.5):
        for d in (0, -1, -3, -6, -10):
            x = rng.normal(d, sd, (R, n)); L, U = ci(x)
            l = np.where(L > -2, 'noDeg', np.where(U < -5, 'catastr', np.where(U < -2, 'moderate',
                         np.where(U >= 0, 'inconcl', 'unres'))))
            v, c = np.unique(l, return_counts=True)
            print(f'  sd {sd} d {d:+3d}: ' + ', '.join(f'{a} {b / R:.2f}' for a, b in zip(v, c)))
h2(5)
print('H2b (LB-b1 - T4-b8 on Aircraft), n=5')
for sd in (2.5, 3.5):
    for e in (10.6, 5.0, 2.0, 0.0):
        x = rng.normal(e, sd, (R, 5)); L, U = ci(x)
        print(f'  sd {sd} e {e:+5.1f}: batch better {np.mean(L > 0):.2f}, depth better {np.mean(U < 0):.2f}')
for n in (10, 6, 5):
    x = rng.normal(-2.0, 2.5, (R, n)); L, _ = ci(x)
    print(f'type-I of a single NI claim at d=-2, n={n}, sd 2.5: {np.mean(L > -2):.3f}')
