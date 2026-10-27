"""r2 planning, part 2: (a) H1 under the pre-registered AdapterDrop priors with Holm over {T3, T4, FT-Last}, 93% CIs,
one extension look for inconclusive arms (LB re-run on the extra seeds too); arm noise model
acc = mu_arm + seed_effect + eps_arm  ->  paired d share LB's eps (corr 0.5).  (b) H2 with n = 5 -> 8."""
import numpy as np
from scipy import stats
rng = np.random.default_rng(11)
R = 40_000
A = 0.035  # one-sided per look (= two-sided 93% CI)

def ci(d):
    n = d.shape[-1]; m = d.mean(-1); s = d.std(-1, ddof=1)
    h = stats.t.ppf(1 - A, n - 1) * s / np.sqrt(n)
    return m - h, m + h, m, s

def pval_ni(d):
    n = d.shape[-1]; m = d.mean(-1); s = d.std(-1, ddof=1)
    return 1 - stats.t.cdf((m + 2) / (s / np.sqrt(n)), n - 1)

def holm(p):  # p: (R, m) -> rejected mask
    order = np.argsort(p, 1); m = p.shape[1]
    rej = np.zeros_like(p, dtype=bool); alive = np.ones(p.shape[0], bool)
    for j in range(m):
        idx = order[:, j]; pj = p[np.arange(p.shape[0]), idx]
        ok = alive & (pj < A / (m - j))
        rej[np.arange(p.shape[0])[ok], idx[ok]] = True
        alive &= ok
    return rej

def label(L, U):
    lab = np.where(L > -2, 'retains', np.where(U < -2, 'inferior', np.where(U >= 0, 'inconcl', 'unres')))
    return lab

def h1(mu, sd_eps, n1, n2, names):
    k = len(mu)
    eps = rng.normal(0, sd_eps, size=(R, k + 1, n2))  # index 0 = LB
    d = (np.array(mu)[None, :, None] + eps[:, 1:, :]) - eps[:, :1, :]
    L, U, _, _ = ci(d[..., :n1]); lab = label(L, U)
    ext = lab == 'inconcl'
    L2, U2, _, _ = ci(d); lab2 = label(L2, U2)
    p1 = pval_ni(d[..., :n1]); p2 = pval_ni(d)
    p = np.where(ext, p2, p1)
    final = np.where(ext, lab2, lab)
    rej = holm(p)
    final = np.where((final == 'retains') & ~rej, 'retains(unadj only)', final)
    out = {}
    for i, nm in enumerate(names):
        vals, cnt = np.unique(final[:, i], return_counts=True)
        out[nm] = {v: round(c / R, 2) for v, c in zip(vals, cnt)}
    n_ext = ext.any(1).mean()
    return out, n_ext, ext.sum(1).mean()

sd_pair = {'Flowers': (1.8, 2.5), 'Aircraft': (2.5, 3.5)}
priors = {'Flowers': ([-1.0, -1.9, -0.7], 10, 15), 'Aircraft': ([-3.1, -10.6, -17.6], 5, 10)}
for ds, (mu, n1, n2) in priors.items():
    for sdp in sd_pair[ds]:
        out, pe, ne = h1(mu, sdp / np.sqrt(2), n1, n2, ['T3', 'T4', 'FT-Last'])
        print(f'{ds} prior d={mu} paired-SD {sdp} n {n1}->{n2}: P(any extension) {pe:.2f}, mean #arms extended {ne:.2f}')
        for nm, o in out.items():
            print(f'    {nm}: {o}')

# H2: LB-b1 - LB-b8, n = 5 -> 8, labels incl. moderate/catastrophic
def h2(d, sd, n1, n2):
    x = rng.normal(d, sd, size=(R, n2))
    def lab(L, U):
        return np.where(L > -2, 'noDeg', np.where(U < -5, 'catastr', np.where(U < -2, 'moderate',
                        np.where(U >= 0, 'inconcl', 'unres'))))
    L, U, _, _ = ci(x[:, :n1]); l1 = lab(L, U); ext = l1 == 'inconcl'
    L2, U2, _, _ = ci(x); l2 = lab(L2, U2)
    f = np.where(ext, l2, l1)
    v, c = np.unique(f, return_counts=True)
    return ext.mean(), {a: round(b / R, 2) for a, b in zip(v, c)}
print('H2 n=5->8')
for sd in (2.0, 2.5, 3.5):
    for d in (0, -1, -3, -6, -10):
        pe, f = h2(d, sd, 5, 8)
        print(f'  sd {sd} d {d:+3d}: P(ext) {pe:.2f} final {f}')
# H2b: e = LB-b1 - T4 on Aircraft, n=5 (no extension): P(L>0) if LB-b1 retains and T4 is -10.6
for sd in (2.5, 3.5):
    for e in (10.6, 5.0, 2.0, 0.0):
        x = rng.normal(e, sd, size=(R, 5)); L, U, _, _ = ci(x)
        print(f'H2b sd {sd} e {e:+5.1f}: batch better {np.mean(L > 0):.2f}  depth better {np.mean(U < 0):.2f}')
