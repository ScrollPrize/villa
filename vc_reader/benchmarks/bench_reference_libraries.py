"""vc_reader CPU/CUDA image processing vs SciPy, scikit-image and OpenCV on real CT chunks.

Every case runs on each --start chunk (default: ten dense level-2 PHercParis4 chunks near
z=13000 whose mask, CT > 127, covers 0.5% to 36%). 3D cases use the (Z, Y, X) crop; 2D cases treat it as Z independent YX images. All CPU code is
single-threaded: vc_reader transforms are, components get threads=1, cv2.setNumThreads(1).
CUDA resident: DeviceArray in and out, synchronized inside the timed region. CUDA uploaded: host in,
host out. Medians of --reps calls after --warmup. Each row records max/mean differences.
"""
import os
# Single-threaded comparison: pin every BLAS/OpenMP pool before NumPy, SciPy, scikit-image or
# OpenCV load. Each timing also records process CPU time / wall time to catch hidden threads.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse, json, platform, statistics, subprocess, sys, time
from pathlib import Path

from extension_build import load_extensions

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build-dir', type=Path, required=True)
parser.add_argument('--array', required=True)
parser.add_argument('--level', default='2')
CHUNKS = ((12672, 4224, 7296), (13184, 3712, 3456), (12672, 4608, 5632), (13184, 5120, 4096),
          (13184, 3072, 3072), (13184, 4352, 3328), (13184, 4608, 4608), (12672, 4736, 6912),
          (12928, 2816, 4992), (12928, 3328, 2048))
parser.add_argument('--start', type=int, nargs=3, action='append', metavar=('Z', 'Y', 'X'),
                    help='chunk origin; repeat for several (default: the ten CHUNKS)')
parser.add_argument('--size', type=int, default=128)
parser.add_argument('--reps', type=int, default=5)
parser.add_argument('--warmup', type=int, default=1)
parser.add_argument('--dims', choices=('all', '3D', '2D'), default='all')
parser.add_argument('--only', nargs='+', default=(), help='case names to run')
parser.add_argument('--skip-references', action='store_true', help='time vc_reader only (e.g. build variants)')
parser.add_argument('--skip-cuda', action='store_true', help='CPU only (CPU-only builds)')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
load_extensions(args.build_dir, names=("_catalog", "_reader", "_transforms", "_thinning"))

import numpy as np, cv2, scipy.ndimage as ndi, skimage, scipy, torch
from skimage import exposure, filters, feature, measure, morphology
from vc_reader import transforms as A, cuda, geometry, thinning
from vc_reader.reader import ZarrReader

cv2.setNumThreads(1)
torch.set_num_threads(1)
try:
    from threadpoolctl import threadpool_info, threadpool_limits
    threadpool_limits(1)
    THREADPOOLS = [{k: v for k, v in p.items() if k in ('internal_api', 'num_threads', 'filepath')} for p in threadpool_info()]
except ImportError:
    THREADPOOLS = 'threadpoolctl not installed'
assert cv2.getNumThreads() == 1
ZARR, N, WARMUP, REPS = args.array, args.size, args.warmup, args.reps
STARTS = [tuple(v) for v in args.start] if args.start else list(CHUNKS)
only, only_dims = set(args.only), args.dims
chunk = None  # index of the chunk being measured

sync = torch.cuda.synchronize  # also covers work on vc_reader's own streams


def timeit(fn, sync_after=False):
    out = None
    times, cpu = [], []
    for i in range(WARMUP + REPS):
        c = time.process_time_ns()
        t = time.perf_counter_ns()
        out = fn()
        if sync_after:
            sync()
        dt = (time.perf_counter_ns() - t) / 1e6
        if i >= WARMUP:
            times.append(dt)
            cpu.append((time.process_time_ns() - c) / 1e6)
    # cpu_per_wall well above 1 means the call used more than one thread.
    return dict(p50=statistics.median(times), min=min(times), max=max(times),
                cpu_per_wall=sum(cpu) / sum(times)), out


def host(x):
    if isinstance(x, tuple):
        x = x[0]
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    if cuda.is_device_array(x):
        return x.numpy()
    return np.asarray(x)


def diff(a, b):
    a, b = host(a), host(b)
    a = a.reshape(b.shape) if a.size == b.size else a
    if a.shape != b.shape:
        return f'shape {a.shape} vs {b.shape}'
    if a.dtype == bool or b.dtype == bool or a.dtype.kind in 'iu' and b.dtype.kind in 'iu':
        n = int(np.count_nonzero(a.astype(np.int64) != b.astype(np.int64)))
        return 'exact' if n == 0 else f'{n} differ ({n / a.size:.2%})'
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return f'max {d.max():.2e}, mean {d.mean():.2e}'


results = []   # one row per case and chunk


def case(name, dims, vc_cpu, vc_dev=None, dev_inputs=(), refs=(), vc_upload=None, note=''):
    """vc_cpu(): host result. vc_dev(*dev): device result. refs: [(label, fn)]."""
    if (only and name not in only) or (only_dims and only_dims != 'all' and not dims.startswith(only_dims)):
        return
    row = dict(name=name, dims=dims, chunk=chunk, note=note, timings={}, checks={})
    t, cpu_out = timeit(vc_cpu)
    row['timings']['vc CPU'] = t
    if args.skip_cuda:
        vc_dev = vc_upload = None
    if args.skip_references:
        refs = ()
    if vc_dev is not None:
        to_dev = cuda.asarray
        try:
            vc_dev(*[to_dev(x) for x in dev_inputs])
        except TypeError:
            # Builds without DeviceArray batches: their CUDA batch input was a Torch tensor.
            to_dev = lambda x: torch.as_tensor(x, device='cuda')
            row['note'] = (row['note'] + '; ' if row['note'] else '') + 'CUDA batch via Torch tensors'
        dev = [to_dev(x) for x in dev_inputs]
        t, dev_out = timeit(lambda: vc_dev(*dev), sync_after=True)
        row['timings']['vc CUDA resident'] = t
        row['checks']['CUDA vs CPU'] = diff(dev_out, cpu_out)
        up = vc_upload or (lambda: host(vc_dev(*[to_dev(x) for x in dev_inputs])))
        t, _ = timeit(up, sync_after=True)
        row['timings']['vc CUDA uploaded'] = t
        del dev, dev_out
    elif vc_upload is not None:
        t, up_out = timeit(vc_upload, sync_after=True)
        row['timings']['vc CUDA uploaded'] = t
        row['checks']['CUDA vs CPU'] = diff(up_out, cpu_out)
    for label, fn in refs:
        t, ref_out = timeit(fn)
        row['timings'][label] = t
        row['checks'][f'CPU vs {label}'] = diff(cpu_out, ref_out)
    results.append(row)
    print(chunk, name, dims, {k: round(v['p50'], 3) for k, v in row['timings'].items()}, row['checks'], flush=True)


def per_slice(fn, stack):
    return np.stack([fn(s) for s in stack])


def run_chunk(u8, ref_u8):
    img3 = np.ascontiguousarray(u8.astype(np.float32) / np.float32(255))
    mask3 = np.ascontiguousarray(u8 > 127)
    mask3_u8 = mask3.astype(np.uint8)
    mask3_i16 = mask3.astype(np.int16)
    refimg3 = ref_u8.astype(np.float32) / np.float32(255)
    c = img3[None]                     # (1, Z, Y, X)
    m16 = mask3_u8[None]  # uint8 labels, as OpenCV gets
    ball3 = morphology.ball(3).astype(bool)
    cross = ndi.generate_binary_structure(3, 1)
    full = np.ones((3, 3, 3), bool)

    # ----------------------------------------------------------------------------------- 3D vs scipy / skimage
    g = A.GaussianBlur(sigma=1, truncate=6, boundary='mirror', full_width=False)
    case('GaussianBlur σ=1', '3D', lambda: g(c), g, [c],
         [('scipy gaussian_filter', lambda: ndi.gaussian_filter(img3, 1, mode='mirror', truncate=6))])
    g3 = A.GaussianBlur(sigma=3, truncate=4, boundary='mirror', full_width=False)
    case('GaussianBlur σ=3', '3D', lambda: g3(c), g3, [c],
         [('scipy gaussian_filter', lambda: ndi.gaussian_filter(img3, 3, mode='mirror', truncate=4))])
    med = A.MedianFilter(size=3)
    case('MedianFilter 3', '3D', lambda: med(c), med, [c],
         [('scipy median_filter', lambda: ndi.median_filter(img3, size=3, mode='nearest'))])
    med5 = A.MedianFilter(size=5)
    case('MedianFilter 5', '3D', lambda: med5(c), med5, [c],
         [('scipy median_filter', lambda: ndi.median_filter(img3, size=5, mode='nearest'))])
    us = A.UnsharpMask(strength=0.5, sigma=1)
    case('UnsharpMask', '3D', lambda: us(c), us, [c],
         [('skimage unsharp_mask', lambda: filters.unsharp_mask(img3, radius=1, amount=0.5, preserve_range=True))])
    for op, fn in (('dilate', ndi.binary_dilation), ('erode', ndi.binary_erosion),
                   ('open', ndi.binary_opening), ('close', ndi.binary_closing)):
        mo = A.Morphology(operation=op, radius=1, footprint='cross')
        case(f'Morphology {op} cross r=1', '3D', lambda mo=mo: mo(m16), mo, [m16],
             [(f'scipy binary_{"dilation" if op == "dilate" else "erosion" if op == "erode" else op + "ing"}',
               lambda fn=fn: fn(mask3, structure=cross))])
    mb = A.Morphology(operation='dilate', radius=3, footprint='ball')
    case('Morphology dilate ball r=3', '3D', lambda: mb(m16), mb, [m16],
         [('scipy binary_dilation', lambda: ndi.binary_dilation(mask3, structure=ball3)),
          ('skimage dilation', lambda: morphology.dilation(mask3_u8, ball3))])
    eq = A.Equalize(bins=256)
    case('Equalize', '3D', lambda: eq(c), eq, [c],
         [('skimage equalize_hist', lambda: exposure.equalize_hist(img3, nbins=256))])
    cl = A.CLAHE(clip_limit=2, tile_shape=(16, 16, 16), bins=256)
    case('CLAHE tiles 16³', '3D', lambda: cl(c), cl, [c],
         [('skimage equalize_adapthist', lambda: exposure.equalize_adapthist(img3, kernel_size=16, clip_limit=0.01, nbins=256))],
         note='clip-limit conventions differ; timing comparison only')
    hm = A.HistogramMatching(reference=refimg3[None], blend=1, bins=256)
    case('HistogramMatching', '3D', lambda: hm(c), hm, [c],
         [('skimage match_histograms', lambda: exposure.match_histograms(img3, refimg3))],
         note='vc uses 256 bins; skimage matches exact quantiles')
    # Fixed 5° rotation about Z + 0.95 scale, linear: same work as A.Affine(degrees=5, scale=0.95) per call.
    th = np.deg2rad(5)
    R = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]]) / 0.95
    ctr = np.array([N / 2 - .5] * 3)
    af = A.Affine(degrees=5, scale=0.95)
    case('Affine (5°, ×0.95, linear)', '3D', lambda: af(c), af, [c],
         [('scipy affine_transform order=1', lambda: ndi.affine_transform(img3, R, ctr - R @ ctr, order=1))],
         note='vc samples a random rotation axis per call; timing comparison only')
    dt = A.DistanceTransform(mode='outside')
    case('Euclidean distance (EDT)', '3D', lambda: dt(m16), dt, [m16],
         [('scipy distance_transform_edt', lambda: ndi.distance_transform_edt(~mask3))])
    dts = A.DistanceTransform(mode='signed')
    case('Signed EDT', '3D', lambda: dts(m16), dts, [m16],
         [('scipy edt ×2', lambda: ndi.distance_transform_edt(~mask3) - ndi.distance_transform_edt(mask3))])
    case('Manhattan distance (uint8)', '3D', lambda: thinning.manhattan_distance(mask3_u8),
         vc_upload=lambda: thinning.manhattan_distance(mask3_u8, backend='cuda'),
         refs=[('scipy distance_transform_cdt taxicab',
                lambda: np.minimum(ndi.distance_transform_cdt(mask3, metric='taxicab'), 255))],
         note='Python CUDA entry is host-in/host-out')


    def scipy_structure_tensor(x, sigma=1, rho=1.5):
        gz = ndi.gaussian_filter(x, sigma, order=(1, 0, 0), mode='nearest', truncate=3)
        gy = ndi.gaussian_filter(x, sigma, order=(0, 1, 0), mode='nearest', truncate=3)
        gx = ndi.gaussian_filter(x, sigma, order=(0, 0, 1), mode='nearest', truncate=3)
        return np.stack([ndi.gaussian_filter(a * b, rho, mode='nearest', truncate=3)
                         for a, b in ((gz, gz), (gz, gy), (gz, gx), (gy, gy), (gy, gx), (gx, gx))])


    st = A.StructureTensor(sigma=1, rho=1.5, derivative='gaussian')
    case('StructureTensor (Gaussian)', '3D', lambda: st(c), st, [c],
         [('scipy gaussian derivs + 6 smooths', lambda: scipy_structure_tensor(img3)),
          ('skimage structure_tensor', lambda: feature.structure_tensor(img3, sigma=1.5, order='rc'))],
         note='skimage uses np.gradient without σ pre-smoothing')
    rd = A.Ridges(sigma=6)
    case('Ridges (Hessian, σ=6)', '3D', lambda: rd(c), rd, [c],
         [('skimage frangi σ=6', lambda: filters.frangi(img3, sigmas=[6], black_ridges=False)),
          ('skimage hessian_matrix+eigvals σ=6',
           lambda: feature.hessian_matrix_eigvals(feature.hessian_matrix(img3, sigma=6, order='rc', use_gaussian_derivatives=False)))],
         note='response formulas differ; timing comparison only')
    case('Connected components 26', '3D', lambda: geometry.label_components(mask3, connectivity=26, threads=1),
         lambda d: geometry.label_components(d, connectivity=26), [mask3_u8],
         [('scipy label', lambda: ndi.label(mask3, structure=full)),
          ('skimage label', lambda: measure.label(mask3, connectivity=3))])
    case('Connected components 6', '3D', lambda: geometry.label_components(mask3, connectivity=6, threads=1),
         lambda d: geometry.label_components(d, connectivity=6), [mask3_u8],
         [('scipy label', lambda: ndi.label(mask3, structure=cross)),
          ('skimage label', lambda: measure.label(mask3, connectivity=1))])
    case('Fill holes', '3D', lambda: geometry.fill_holes(mask3, threads=1),
         lambda d: geometry.fill_holes(d), [mask3_u8],
         [('scipy binary_fill_holes', lambda: ndi.binary_fill_holes(mask3))])

    # ------------------------------------------------------------------- 2D: the same chunk as 128 YX slices vs cv2
    D2 = f'2D×{N}'
    img2b = img3[:, None]                     # (B=Z, C=1, Y, X)
    mask2b = mask3_u8[:, None]
    u8_slices = u8
    k3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    g2 = A.GaussianBlur(sigma=1, truncate=6, boundary='mirror', full_width=False)
    case('GaussianBlur σ=1', D2, lambda: g2.batch(img2b), g2.batch, [img2b],
         [('cv2 GaussianBlur', lambda: per_slice(lambda s: cv2.GaussianBlur(s, (13, 13), 1, borderType=cv2.BORDER_REFLECT_101), img3)),
          ('scipy gaussian_filter axes=(1,2)', lambda: ndi.gaussian_filter(img3, 1, mode='mirror', truncate=6, axes=(1, 2)))])
    m2 = A.MedianFilter(size=3)
    case('MedianFilter 3', D2, lambda: m2.batch(img2b), m2.batch, [img2b],
         [('cv2 medianBlur', lambda: per_slice(lambda s: cv2.medianBlur(s, 3), img3)),
          ('scipy median_filter', lambda: ndi.median_filter(img3, size=(1, 3, 3), mode='nearest'))])
    for op, cvop in (('dilate', cv2.MORPH_DILATE), ('erode', cv2.MORPH_ERODE), ('close', cv2.MORPH_CLOSE)):
        mo2 = A.Morphology(operation=op, radius=1, footprint='cross')
        case(f'Morphology {op} cross r=1', D2, lambda mo2=mo2: mo2.batch(mask2b), mo2.batch, [mask2b],
             [('cv2 morphologyEx', lambda cvop=cvop: per_slice(
                 lambda s: cv2.morphologyEx(s, cvop, k3, borderType=cv2.BORDER_CONSTANT, borderValue=0), mask3_u8))])
    e2 = A.Equalize(bins=256)
    case('Equalize', D2, lambda: e2.batch(img2b), e2.batch, [img2b],
         [('cv2 equalizeHist (uint8)', lambda: per_slice(cv2.equalizeHist, u8_slices) / np.float32(255))])
    cl2 = A.CLAHE(clip_limit=2, tile_shape=(16, 16), bins=256)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    case('CLAHE tiles 16²', D2, lambda: cl2.batch(img2b), cl2.batch, [img2b],
         [('cv2 CLAHE (uint8)', lambda: per_slice(clahe.apply, u8_slices) / np.float32(255))],
         note='cv2 works on uint8; timing comparison only')
    dt2 = A.DistanceTransform(mode='outside')
    case('Euclidean distance (EDT)', D2, lambda: dt2.batch(mask2b), dt2.batch, [mask2b],
         [('cv2 distanceTransform L2 precise', lambda: per_slice(
             lambda s: cv2.distanceTransform(s, cv2.DIST_L2, cv2.DIST_MASK_PRECISE), mask3_u8 ^ 1)),
          ('scipy edt per slice', lambda: per_slice(lambda s: ndi.distance_transform_edt(s), ~mask3))])
    case('Connected components 8', D2,
         lambda: np.stack([geometry.label_components(s, connectivity=8, threads=1)[0] for s in mask3]),
         refs=[('cv2 connectedComponents', lambda: per_slice(lambda s: cv2.connectedComponents(s, connectivity=8, ltype=cv2.CV_32S)[1], mask3_u8)),
               ('scipy label per slice', lambda: per_slice(lambda s: ndi.label(s, structure=np.ones((3, 3)))[0], mask3))],
         note='vc called per slice from Python (no batched 2D labelling entry); cv2 numbers components in a different order')
    sk_out = None if args.skip_cuda else cuda.empty(mask3_u8.shape, np.uint8)
    case('Skeletonize Zhang–Suen', D2, lambda: thinning.skeletonize(mask3_u8, preserve_border=False),
         lambda d: (thinning.skeletonize_into(d, sk_out, backend='cuda', preserve_border=False), sk_out)[1], [mask3_u8],
         [('skimage thin per slice', lambda: per_slice(lambda s: morphology.thin(s).astype(np.uint8) * 255, mask3)),
          ('skimage skeletonize per slice', lambda: per_slice(lambda s: morphology.skeletonize(s).astype(np.uint8) * 255, mask3))],
         vc_upload=lambda: (lambda d, o: (thinning.skeletonize_into(d, o, backend='cuda', preserve_border=False), o.numpy())[1])(
             cuda.asarray(mask3_u8), cuda.empty(mask3_u8.shape, np.uint8)))
    case('Manhattan distance (uint8)', D2,
         lambda: np.stack([thinning.manhattan_distance(s) for s in mask3_u8]),
         refs=[('cv2 distanceTransform L1', lambda: per_slice(lambda s: np.minimum(cv2.distanceTransform(s, cv2.DIST_L1, 3), 255), mask3_u8))],
         note='vc called per slice')

chunks = []
with ZarrReader.open(ZARR, level=args.level, threads=8) as reader:
    for chunk, start in enumerate(STARTS):
        u8 = reader.read_region(start, (N,) * 3)
        # Histogram-matching reference: the crop 3 chunks deeper.
        ref_u8 = reader.read_region((start[0] + 3 * N, *start[1:]), (N,) * 3)
        stats = dict(start=start, nonzero=float(np.mean(u8 > 0)), mean=float(u8.mean()), std=float(u8.std()),
                     mask_fraction=float(np.mean(u8 > 127)))
        chunks.append(stats)
        print(f'chunk {chunk}', stats, flush=True)
        run_chunk(u8, ref_u8)


def summarize(rows):
    """Per case: median over chunks of each implementation's p50, and the vc/reference ratio range."""
    table = {}
    for row in rows:
        table.setdefault((row['name'], row['dims']), []).append(row)
    out = []
    for (name, dims), group in table.items():
        labels = list(group[0]['timings'])
        entry = dict(name=name, dims=dims, note=group[0]['note'], p50_ms={}, worst_chunk_ratio={}, checks=group[0]['checks'])
        for label in labels:
            entry['p50_ms'][label] = statistics.median(r['timings'][label]['p50'] for r in group)
        refs = [l for l in labels if not l.startswith('vc')]
        for vc in ('vc CPU', 'vc CUDA resident'):
            if vc in labels and refs:
                # >1 means vc is slower than the fastest reference on that chunk.
                entry['worst_chunk_ratio'][vc] = max(r['timings'][vc]['p50'] / min(r['timings'][l]['p50'] for l in refs)
                                                    for r in group)
        entry['checks_by_chunk'] = [r['checks'] for r in group]
        entry['cpu_per_wall'] = {l: max(r['timings'][l]['cpu_per_wall'] for r in group) for l in labels}
        out.append(entry)
    return out


summary = summarize(results)
for entry in summary:
    print(entry['dims'], entry['name'], {k: round(v, 3) for k, v in entry['p50_ms'].items()},
          {k: round(v, 2) for k, v in entry['worst_chunk_ratio'].items()},
          {k: round(v, 2) for k, v in entry['cpu_per_wall'].items() if not k.startswith('vc CUDA') and v > 1.2} or '', flush=True)
out = dict(command=[sys.executable, *sys.argv], zarr=ZARR, level=args.level, chunks=chunks, size=N, warmup=WARMUP, reps=REPS,
           git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
           cpu=next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), platform.processor()),
           versions=dict(numpy=np.__version__, scipy=scipy.__version__, skimage=skimage.__version__,
                         cv2=cv2.__version__, torch=torch.__version__),
           gpu=torch.cuda.get_device_name(), threadpools=THREADPOOLS, summary=summary, results=results)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(out, indent=1) + '\n')
