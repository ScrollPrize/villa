"""Estimate winding counts for scrolls from one mid-height cross-section of the organizers' L2 surface prediction:
polar-transform around the scroll-mask centroid (or umbilicus if given) and count sheet crossings along radial rays.
Usage: winding_counts.py <scroll:surf_zarr_rel_path/> [...]   (paths without trailing '0')"""
import sys, json, urllib.request, urllib.parse, numpy as np, zarr, numcodecs
from scipy import ndimage as ndi
from scipy.signal import find_peaks
B = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
def read_slice(url, z, level='2'):
    meta = json.loads(urllib.request.urlopen(url + f'{level}/.zarray', timeout=60).read().decode())
    shape, chunks = meta['shape'], meta['chunks']; codec = numcodecs.get_codec(meta['compressor']) if meta['compressor'] else None
    cz, cy, cx = chunks; iz = z // cz
    out = np.zeros((shape[1], shape[2]), np.uint8)
    for iy in range((shape[1] + cy - 1) // cy):
        for ix in range((shape[2] + cx - 1) // cx):
            try:
                buf = urllib.request.urlopen(url + f'{level}/{iz}/{iy}/{ix}', timeout=60).read()
            except Exception: continue
            arr = np.frombuffer(codec.decode(buf) if codec else buf, np.dtype(meta['dtype'])).reshape(chunks)
            y1, x1 = min(shape[1], (iy + 1) * cy), min(shape[2], (ix + 1) * cx)
            out[iy * cy:y1, ix * cx:x1] = arr[z % cz, :y1 - iy * cy, :x1 - ix * cx]
    return out, shape
for spec in sys.argv[1:]:
    scroll, rel = spec.split(':', 1); url = B + rel
    attrs = json.loads(urllib.request.urlopen(url + '.zattrs', timeout=60).read().decode())
    meta2 = json.loads(urllib.request.urlopen(url + '2/.zarray', timeout=60).read().decode()); Z2 = meta2['shape'][0]
    counts_all = []
    for frac in (0.35, 0.5, 0.65):
        sl, shape = read_slice(url, int(frac * Z2))
        mask = ndi.binary_fill_holes(ndi.binary_closing(sl > 40, iterations=8))
        if mask.sum() < 1000: continue
        cy, cx = ndi.center_of_mass(mask)
        nr, nt = 1500, 360
        rmax = min(cx, cy, shape[2] - cx, shape[1] - cy) * 0.98
        rr = np.linspace(2, rmax, nr); tt = np.linspace(-np.pi, np.pi, nt, endpoint=False)
        R, T = np.meshgrid(rr, tt, indexing='ij')
        pol = ndi.map_coordinates(sl.astype(np.float32), [(cy + R * np.sin(T)).ravel(), (cx + R * np.cos(T)).ravel()], order=1).reshape(nr, nt)
        counts = []
        for j in range(nt):
            prof = ndi.gaussian_filter1d(pol[:, j], 1.5); pk, _ = find_peaks(prof, height=60, distance=3); counts.append(len(pk))
        counts_all.append((int(frac * Z2 * 4), int(np.median(counts)), int(np.percentile(counts, 75)), int(np.max(counts))))
    print(scroll, 'z(level0), median, p75, max crossings per ray:', counts_all, flush=True)
