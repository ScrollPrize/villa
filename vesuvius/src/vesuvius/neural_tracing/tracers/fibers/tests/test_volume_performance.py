import json
from collections import Counter

import numpy as np
from types import SimpleNamespace

from vesuvius.neural_tracing.tracers.fibers.tracing import volume

def test_heading_queued_chunks_exact():
    import torch
    from vesuvius.neural_tracing.tracers.fibers.heading_model.model import HeadingConfig, HeadingNet, HeadingPredictor
    torch.manual_seed(84)
    rng = np.random.default_rng(85)
    cfg = HeadingConfig(width=1, predict_normals=True, predict_frames=True)
    def crops(items, ct, spec, crop):
        patch = torch.arange(32**3, dtype=torch.float32).reshape(1,32,32,32)/32768.
        return torch.stack([patch+float(item['pos'][0])/100 for item in items])
    predictor = HeadingPredictor(HeadingNet(cfg), device='cpu', crops=crops)
    positions = rng.normal(size=(130,3))
    priors = rng.normal(size=(130,3))
    paths = [pos+np.arange(-8., 1.)[:, None]*prior for pos, prior in zip(positions, priors)]
    families = ['H', 'V']*65
    previous = [None if i%2 else np.eye(3) for i in range(130)]
    expected = []
    for start in range(0,130,64):
        s = slice(start,start+64)
        expected.extend(predictor.predict_frames(None, None, positions[s], priors[s], paths[s], families[s], previous[s]))
    actual = predictor.predict_frames_batched(None, None, positions, priors, paths, families, previous)
    np.testing.assert_array_equal(actual, expected)
    assert predictor.predict_frames_batched(None, None, [], [], [], []) == []

def test_parallel_advancement_whole_trace_exact(monkeypatch, tmp_path):
    import torch
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from vc_reader.reader import ZarrWriter, open_zarr
    from vesuvius.neural_tracing.tracers.fibers.tracing import trace
    from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolumeSpec
    from vesuvius.neural_tracing.tracers.shared.geometry import CropSpec
    # An empty 100^3 CT: only its shape (the trace bounds) is read; frames come from the patched ct_frame.
    ZarrWriter.create(tmp_path/'ct'/'0', shape=(100, 100, 100), chunks=(50, 50, 50), dtype='u1', fill_value=0,
                      compressor=None)
    spec = CTVolumeSpec(str(tmp_path/'ct'), ct_level=0, ct_grid_scale=8., grid_scale=8.)
    ct = open_zarr(spec.ct_zarr, level=spec.ct_level)
    monkeypatch.setattr(trace, 'ct_frame', lambda ct, spec, pos, *a, **k:
                        np.diag([1., -1., -1.]) if pos[0] == 13 and pos[2] >= 24 else np.eye(3))
    class Policy(torch.nn.Module):
        cfg = SimpleNamespace(n_future=4, max_recovery_distance=6.)
        def forward(self, x, hist, mask):
            points = torch.tensor([[0., 0., 1.], [0., 0., 2.], [0., 0., 3.], [0., 0., 4.]])
            points = points[None].repeat(len(hist), 1, 1)
            confidence = torch.ones(len(hist), 4)
            confidence[x[:, 0] == 10] = 0.
            points[x[:, 0] == 11] = float('nan')
            return dict(points=points, confidence=confidence)
    class Tracer(trace.ModelTracer):
        def build_inputs(self, pos, *args):
            return torch.from_numpy(pos)
    seeds = np.column_stack((np.arange(24)+10., np.full(24, 20.), np.full(24, 20.)))
    seeds[2, 2] = 99.
    headings = np.tile([0., 0., 1.], (24, 1))
    results = []
    for parallel, callback in ((False, False), (True, False), (True, True)):
        tracer = Tracer(Policy(), ct, spec,
                        CropSpec(depth=12, width=9, behind=2), 8,
                        trace.TraceParams(n_commit=4, max_len=10., gate='prefix', loop_skip=2), device='cpu')
        pool = ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) if parallel else None
        tracer.axis_pool = pool
        try:
            results.append(tracer.trace(seeds, headings, on_decision=(lambda *a: True) if callback else None))
        finally:
            tracer.close()
            if pool:
                pool.shutdown()
    for paths, reasons in results[1:]:
        assert results[0][1] == reasons
        for serial, parallel in zip(results[0][0], paths):
            np.testing.assert_array_equal(serial, parallel)
    assert {'confidence', 'bounds', 'loop', 'max_len'} <= set(results[0][1])

def test_parallel_geometry_matches_serial():
    import torch
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from vesuvius.neural_tracing.tracers.models.path_geometry import path_geometry_inputs
    rng = np.random.default_rng(142)
    items = []
    for row, size in enumerate((1, 2, 10, 100, 1000)*6):
        path = np.cumsum(rng.normal(size=(size, 3)), axis=0)
        frame, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        items.append(dict(observed_path=path, pos=path[-1], frame=frame,
                          seed_valid=bool(row % 2), seed_pos=path[0]))
    expected = path_geometry_inputs(items)
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool:
        actual = path_geometry_inputs(items, pool=pool)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

def test_seed_dispatch_preserves_order_and_filter(monkeypatch):
    dots = dict(xyz=np.arange(15).reshape(5, 3), family=np.array(['H']*5),
                direction=np.tile([1., 0., 0.], (5, 1)))
    def heading(task):
        return None if task[0][0] == 6 else task[0].astype(float)
    monkeypatch.setattr(volume, '_seed_heading', heading)
    class Pool:
        def map(self, fn, tasks, *, chunksize):
            assert chunksize == 1
            return map(fn, tasks)
    output, keep = volume.seed_headings(dots, Pool())
    np.testing.assert_array_equal(keep, [True, True, False, True, True])
    np.testing.assert_array_equal(output['xyz'], dots['xyz'][keep])
    np.testing.assert_array_equal(output['heading'], dots['xyz'][keep])


def reference_tensor(cube, center_zyx, *, sample_spacing=1., derivative_sigma=1., integration_sigma=4.):
    from scipy.ndimage import gaussian_filter
    image = np.asarray(cube, dtype=np.float64)/255.
    gradients = []
    for axis in range(3):
        order = [0, 0, 0]
        order[axis] = 1
        gradients.append(gaussian_filter(image, derivative_sigma/sample_spacing,
            order=order, mode='nearest', truncate=3.)/sample_spacing)
    weights = [np.exp(-.5*((np.arange(n)-c)/(integration_sigma/sample_spacing))**2)
               for n, c in zip(image.shape, center_zyx)]
    weight = weights[0][:, None, None]*weights[1][None, :, None]*weights[2][None, None, :]
    weight /= weight.sum()
    return np.array([[np.sum(weight*a*b) for b in gradients] for a in gradients])[::-1, ::-1]


def test_structure_tensor_exact():
    from vesuvius.neural_tracing.tracers.shared.ct_structure import ct_structure_tensor
    rng = np.random.default_rng(41)
    for shape in ((77, 77, 77), (19, 24, 31)):
        cube = rng.integers(0, 256, size=shape, dtype=np.uint8)
        for spacing in (1., 2., .75):
            center = np.asarray(shape)/2 + [.1, -.3, .2]
            options = dict(sample_spacing=spacing, derivative_sigma=2., integration_sigma=8.)
            np.testing.assert_array_equal(ct_structure_tensor(cube, center, **options),
                                          reference_tensor(cube, center, **options))
        for data in (np.asfortranarray(cube), cube[:, ::2, ::-1]):
            for sigma in (2., 1e-16):
                options = dict(derivative_sigma=sigma, integration_sigma=8.)
                np.testing.assert_array_equal(ct_structure_tensor(data, center, **options),
                                              reference_tensor(data, center, **options))


def test_inference_graph_cpu_falls_back():
    import torch
    from vesuvius.neural_tracing.tracers.shared.inference_graph import InferenceGraph
    class Model(torch.nn.Module):
        def forward(self, x, hist, hmask, scale):
            return {'points': (x + hist)*scale, 'mask': hmask}
    model = Model().eval()
    call = InferenceGraph(model, 2)
    x = torch.ones(2, 3)
    out = call(x, x, x, scale=2)
    torch.testing.assert_close(out['points'], model(x, x, x, scale=2)['points'], rtol=0, atol=0)
    assert call.graph is None


def reference_cone(centre, max_angle, step):
    e1 = np.cross(centre, np.eye(3)[np.argmin(np.abs(centre))])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(centre, e1)
    out = [centre]
    for a in np.radians(np.arange(step, max_angle+1e-9, step)):
        n = max(6, int(np.ceil(360*np.sin(a)/step)))
        b = np.linspace(0., 2*np.pi, n, endpoint=False)
        out.append(np.cos(a)*centre+np.sin(a)*(np.cos(b)[:, None]*e1+np.sin(b)[:, None]*e2))
    return np.vstack([np.atleast_2d(o) for o in out])


def test_cached_cone_exact():
    from vesuvius.neural_tracing.tracers.fibers.frames.learned import _cone
    for centre in np.random.default_rng(42).normal(size=(64, 3)):
        centre /= np.linalg.norm(centre)
        for angle, step in ((75., 2.5), (3., .25), (0., 1.)):
            np.testing.assert_array_equal(_cone(centre, angle, step), reference_cone(centre, angle, step))


def benchmark_axes():
    import time
    from vesuvius.neural_tracing.tracers.fibers.frames import learned
    optimized = learned._cone
    rng = np.random.default_rng(42)
    paths = np.cumsum(rng.normal(scale=.1, size=(128, 35, 3)) + [0., 0., 1.], axis=1)
    results, expected = {}, None
    try:
        for name, cone in [('original', reference_cone), ('cached', optimized)]:
            learned._cone = cone
            times = []
            for _ in range(3):
                start = time.perf_counter()
                values = np.array([learned.predicted_axis(p, np.zeros(3), 16.) for p in paths])
                times.append(time.perf_counter()-start)
                if expected is None:
                    expected = values
                np.testing.assert_array_equal(values, expected)
            results[name] = dict(seconds=times, median=float(np.median(times)), paths=len(paths))
    finally:
        learned._cone = optimized
    return results


def test_restore_reads_each_npz_member_once(tmp_path, monkeypatch):
    tiles = tmp_path / 'tiles'
    tiles.mkdir()
    points = np.arange(30, dtype=np.float32).reshape(10, 3)
    np.savez(tiles / 'tile.npz', points=points, offsets=[0, 3, 7, 10], id=[2, 5, 9])
    (tiles / 'ignored.partial.npz').write_bytes(b'incomplete')
    (tmp_path / 'progress.json').write_text(json.dumps({'next_id': 15}))
    real_load, reads = np.load, Counter()
    class Archive:
        def __init__(self, path): self.value = real_load(path)
        def __enter__(self): return self
        def __exit__(self, *args): self.value.close()
        def __getitem__(self, key):
            reads[key] += 1
            return self.value[key]
    class Owner:
        def __init__(self): self.rows = []
        def add(self, path, fid): self.rows.append((path, fid))
        def flush(self): pass
    monkeypatch.setattr(volume.np, 'load', Archive)
    owner = Owner()
    done, next_id = volume.load_done(tmp_path, owner)
    assert done == {'tile'} and next_id == 15
    assert reads == {'points': 1, 'offsets': 1, 'id': 1}
    assert [fid for _, fid in owner.rows] == [2, 5, 9]
    np.testing.assert_array_equal(np.concatenate([p for p, _ in owner.rows]), points)
