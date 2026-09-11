"""Estimate an umbilicus.json (scroll core polyline) for a scroll without a published one, from the organizers' L2 surface
prediction: at N heights, take the largest connected component of the sheet mask, fill holes, and take the point of the
distance-to-boundary plateau (within 0.9 of the maximum) that is nearest the centroid of the component, and use that as
the core; write control points {x,y,z,score} at level 0. The plain argmax of the distance transform is an argmax over a
wide, nearly flat plateau, so it jumps between lobes from slice to slice; measured against the published umbilici of
PHerc0125, PHerc0211 and PHerc0826 the point of that plateau nearest the centroid is closer on all three. Pass
core=argmax for the previous behaviour.
Usage: estimate_umbilicus.py <scroll> <surf_zarr_rel_path/> <out.json> [n_z=12] [core=plateau|argmax]"""
import sys, json, urllib.request, urllib.error, numpy as np, numcodecs
from scipy import ndimage as ndi
B = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
scroll, rel, out = sys.argv[1:4]; n_z = int(sys.argv[4]) if len(sys.argv) > 4 else 12
core = sys.argv[5] if len(sys.argv) > 5 else 'plateau'
if core.startswith('core='): core = core.split('=', 1)[1]      # the usage line invites the prefixed form
if core not in ('plateau', 'argmax'): sys.exit(f"core must be 'plateau' or 'argmax', got {core!r}")
url = B + rel
def read_slice(level, z):
    meta = json.loads(urllib.request.urlopen(url + f'{level}/.zarray', timeout=60).read().decode())
    shape, chunks = meta['shape'], meta['chunks']; codec = numcodecs.get_codec(meta['compressor']) if meta['compressor'] else None
    cz, cy, cx = chunks; iz = z // cz; outp = np.zeros((shape[1], shape[2]), np.uint8)
    for iy in range((shape[1] + cy - 1) // cy):
        for ix in range((shape[2] + cx - 1) // cx):
            try: buf = urllib.request.urlopen(url + f'{level}/{iz}/{iy}/{ix}', timeout=60).read()
            except urllib.error.HTTPError as e:
                if e.code != 404: raise      # 404 is an all-zero chunk zarr never wrote; anything else is a real failure
                continue
            arr = np.frombuffer(codec.decode(buf) if codec else buf, np.dtype(meta['dtype'])).reshape(chunks)
            y1, x1 = min(shape[1], (iy + 1) * cy), min(shape[2], (ix + 1) * cx)
            outp[iy * cy:y1, ix * cx:x1] = arr[z % cz, :y1 - iy * cy, :x1 - ix * cx]
    return outp, shape
level = '3'; f = 8
meta = json.loads(urllib.request.urlopen(url + f'{level}/.zarray', timeout=60).read().decode()); Z = meta['shape'][0]
pts = []
for frac in np.linspace(0.08, 0.92, n_z):
    z = int(frac * Z); sl, shape = read_slice(level, z)
    mask = ndi.binary_closing(sl > 40, iterations=4)
    lab, n = ndi.label(mask)
    if n == 0: continue
    sizes = ndi.sum(mask, lab, range(1, n + 1)); mask = ndi.binary_fill_holes(lab == (1 + int(np.argmax(sizes))))
    dist = ndi.distance_transform_edt(mask)
    cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
    if core == 'plateau':                       # the plateau point nearest the section centroid
        gy, gx = ndi.center_of_mass(mask); ys, xs = np.where(dist >= 0.9 * dist.max())
        j = int(np.argmin((xs - gx) ** 2 + (ys - gy) ** 2)); cy, cx = ys[j], xs[j]
    pts.append({'x': int(cx * f + f // 2), 'y': int(cy * f + f // 2), 'z': int(z * f), 'score': 60})
    print(f'z={z*f}: core at x={cx*f} y={cy*f} (depth {dist[cy, cx]*f*9.4/1000:.1f} mm)', flush=True)
SOURCE = {'argmax': 'max distance-to-boundary', 'plateau': 'distance plateau nearest the centroid'}   # argmax keeps the old literal, so core=argmax reproduces existing files byte for byte
json.dump({'control_points': pts, 'metadata': {'source': f'estimate_umbilicus.py ({SOURCE[core]} of sheet mask, L3)', 'scroll': scroll}}, open(out, 'w'), indent=1)
print('wrote', out, len(pts), 'points')
