"""Between consecutive annotated windings: is there an H layer, a V layer, both?"""
import json, sys
import numpy as np
d = np.load(sys.argv[1]); ann = json.load(open(sys.argv[2]))['collections']
s, zs, crop = d['s'].astype(np.float32), d['zs'], d['crop']; scale = 2 ** (int(d['level']) - 2)
y0, y1, x0, x1 = crop; dz = int(zs[1] - zs[0])
stats = []
for cid, c in ann.items():
    P = {}
    for p in c['points'].values():
        x, y, z = p['p'][0] / scale - x0, p['p'][1] / scale - y0, p['p'][2] / scale
        k = (z - zs[0]) / dz
        if -0.5 <= k <= len(zs) - 0.5 and 0 <= y < s.shape[1] and 0 <= x < s.shape[2]:
            P[p['wind_a']] = (x, y, int(round(k)))
    for wa in P:
        if wa + 1 in P and P[wa][2] == P[wa + 1][2]:
            (xa, ya, k), (xb, yb, _) = P[wa], P[wa + 1]
            n = int(np.hypot(xb - xa, yb - ya) * 2) + 2
            t = np.linspace(0, 1, n)
            v = s[k, np.rint(ya + t * (yb - ya)).astype(int), np.rint(xa + t * (xb - xa)).astype(int)]
            stats.append((np.hypot(xb - xa, yb - ya), v.max() > 0.1, v.min() < -0.1))
a = np.array(stats, float)
print(f'consecutive annotated windings on same slice: {len(a)}; spacing L3 px p10/50/90 {np.percentile(a[:,0],[10,50,90]).round(1)}')
print(f'H layer crossed: {a[:,1].mean():.2f}   V layer crossed: {a[:,2].mean():.2f}   both: {(a[:,1]*a[:,2]).mean():.2f}   neither: {((1-a[:,1])*(1-a[:,2])).mean():.2f}')
