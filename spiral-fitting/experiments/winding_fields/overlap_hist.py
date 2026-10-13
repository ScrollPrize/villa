"""Histogram of w_A - w_B on the shared slices of two neighbouring slabs."""
import sys
import numpy as np
A = np.load(sys.argv[1]); B = np.load(sys.argv[2])
common = np.intersect1d(A['zs'], B['zs'])
ya0, ya1, xa0, xa1 = A['crop']; yb0, yb1, xb0, xb1 = B['crop']
y0, y1, x0, x1 = max(ya0, yb0), min(ya1, yb1), max(xa0, xb0), min(xa1, xb1)
for z in common:
    ia = int(np.nonzero(A['zs'] == z)[0][0]); ib = int(np.nonzero(B['zs'] == z)[0][0])
    wa = A['w'][ia, y0 - ya0:y1 - ya0, x0 - xa0:x1 - xa0]; wb = B['w'][ib, y0 - yb0:y1 - yb0, x0 - xb0:x1 - xb0]
    m = A['mask'][ia, y0 - ya0:y1 - ya0, x0 - xa0:x1 - xa0] & B['mask'][ib, y0 - yb0:y1 - yb0, x0 - xb0:x1 - xb0]
    d = (wa - wb)[m]
    print(f'z {z} (A slice {ia}/{len(A["zs"]) - 1}, B slice {ib}): median {np.median(d):+.2f}  '
          f'|d|<0.5 {np.mean(np.abs(d) < 0.5):.2f}  |d|<1 {np.mean(np.abs(d) < 1):.2f}  |d|<1.5 {np.mean(np.abs(d) < 1.5):.2f}  |d|<2.5 {np.mean(np.abs(d) < 2.5):.2f}')
h = np.histogram(d, bins=np.arange(-3.25, 3.3, 0.5))[0] / d.size
print('last slice, bins of 0.5 from -3 to +3:', ' '.join(f'{v:.2f}' for v in h))
