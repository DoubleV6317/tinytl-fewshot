"""Planning numbers for the decision rules (normal approx for the mean, t critical values)."""
from scipy import stats
import math
def hw(sd, n, conf=0.90):
    return stats.t.ppf(1 - (1 - conf) / 2, n - 1) * sd / math.sqrt(n)
def p_ni(sd, n, d, delta=2.0):
    se = sd / math.sqrt(n); thr = -delta + hw(sd, n)
    return 1 - stats.norm.cdf((thr - d) / se)
print('NI (delta=2, 90% CI lower > -2)')
for sd in (1.5, 1.8, 2.5):
    for n in (6, 10, 15):
        print(f' sd {sd} n {n:2d}: hw {hw(sd,n):.2f}  P(NI|d=0) {p_ni(sd,n,0):.2f}  P(NI|d=-1) {p_ni(sd,n,-1):.2f}  P(NI|d=-3) {p_ni(sd,n,-3):.3f}')
# inconclusive zone: L<=-2 and U>=0  ->  d in [-2+hw, -hw] if 2*hw>2
for sd in (1.8, 2.5):
    for n in (6, 10, 15):
        h = hw(sd, n); print(f' sd {sd} n {n}: inconclusive zone ' + (f'[{-h:.2f}, {-2+h:.2f}]' if 2*h > 2 else 'none'))
# H2 catastrophic: U < -5
for sd in (1.8, 2.5):
    n = 6; se = sd/math.sqrt(n); h = hw(sd, n)
    print(f'H2 sd {sd} n 6: hw {h:.2f}; P(U<-5|d=-8) {stats.norm.cdf((-5-h+8)/se):.2f}; P(L>-2|d=0) {p_ni(sd,6,0):.2f}')
# H3 Welch k=5 vs k=2, n=10 each
for s2, s5 in ((2.0, 2.0), (2.5, 2.0), (3.0, 2.5)):
    se = math.sqrt(s2**2/10 + s5**2/10)
    df = se**4 / ((s2**2/10)**2/9 + (s5**2/10)**2/9)
    tc = stats.t.ppf(0.95, df)
    mde = (tc + stats.norm.ppf(0.8)) * se
    print(f'H3 s2 {s2} s5 {s5}: SE {se:.2f} df {df:.1f} 90%CI hw {tc*se:.2f}  MDE80 {mde:.2f}')
# H2b / k=10 descriptive n=3
for sd in (1.0, 1.8):
    print(f'n=3 sd {sd}: 90% hw {hw(sd,3):.2f}')
