"""integerize: exact integer labels from an approximate (drifting) L1 solution."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sheetgraph import integerize

rng = np.random.default_rng(0)
# two chains (components) plus cross edges; true labels integers
n = 2000
truth = np.concatenate([np.cumsum(rng.integers(0, 2, 1200)), 50 + np.cumsum(rng.integers(0, 2, 800))]).astype(float)
a = np.r_[np.arange(0, 1199), np.arange(1200, 1999)]
b = a + 1
extra = rng.integers(0, 1200, 300); a = np.r_[a, extra]; b = np.r_[b, np.minimum(extra + rng.integers(2, 6, 300), 1199)]
d = truth[b] - truth[a]
wt = np.ones(len(a))
# approximate solution: truth + slow drift (fractional) + a global non-integer offset
x = truth + 0.3 * np.sin(np.arange(n) / 150.0) + 0.37
res = np.abs(x[b] - x[a] - d)
k, frac_ok = integerize(n, a, b, d, wt, x, res)
assert np.allclose(k, np.rint(k)), 'labels not integral'
assert np.allclose(k[b] - k[a], d), 'edge differences not reproduced'
# offset per component: closest integer to x
assert np.abs(np.median(x[:1200] - k[:1200])) < 0.5 and np.abs(np.median(x[1200:] - k[1200:])) < 0.5
# an edge whose d is wrong (outlier) must not break integrality
d2 = d.copy(); d2[5] += 3
res2 = np.abs(x[b] - x[a] - d2)
k2, _ = integerize(n, a, b, d2, wt, x, res2)
assert np.allclose(k2, np.rint(k2)) and np.allclose(k2[:1200] - k2[0], truth[:1200] - truth[0])
print('ok', frac_ok)
