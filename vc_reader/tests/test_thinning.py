"""Public thinning, connected-component, and resident pipeline contracts."""
import numpy as np
import pytest
from build_support import load_build
load_build(names=("_thinning", "_transforms"))
from vc_reader import thinning as T
from vc_reader import transforms as A


def backends():
    return ["cpu", "cuda"] if A._transforms.cuda_available() else ["cpu"]


@pytest.mark.parametrize("shape", [(1, 1), (31, 33), (2, 3, 31, 33)])
@pytest.mark.parametrize("method", ["zhang_suen", "guo_hall", "zhang_suen_guo_hall"])
def test_skeleton_masks_batches_and_aliasing(shape, method):
    x = (np.random.default_rng(4).random(shape) > .4).astype(np.uint8)
    for border in (False, True):
        ref = T.skeletonize(x, method=method, preserve_border=border)
        for backend in backends():
            out = x.copy()
            T.skeletonize_into(out, out, method=method, backend=backend, preserve_border=border)
            np.testing.assert_array_equal(out, ref)
        assert np.all((ref == 0) | (ref == 255))
        assert np.all((ref == 0) | (x != 0))


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("full", [False, True])
def test_components_against_scipy(dim, full):
    from scipy.ndimage import label, generate_binary_structure
    x = (np.random.default_rng(32).random((11,) * dim) > .65).astype(np.uint8)
    labels, count = label(x, generate_binary_structure(dim, dim if full else 1))
    expected = np.zeros_like(labels, dtype=np.uint32)
    for c in range(1, count + 1):
        indices = np.flatnonzero(labels.ravel() == c)
        expected.ravel()[indices] = indices[0] + 1
    for backend in backends():
        np.testing.assert_array_equal(T.connected_components(x, backend=backend, full_connectivity=full), expected)


@pytest.mark.parametrize("dim", [2, 3])
def test_manhattan_against_scipy(dim):
    from scipy.ndimage import distance_transform_cdt
    x = (np.random.default_rng(7).random((17,) * dim) > .1).astype(np.uint8)
    expected = np.minimum(distance_transform_cdt(x, metric="taxicab"), 255)
    for backend in backends():
        np.testing.assert_array_equal(T.manhattan_distance(x, backend=backend), expected)
        np.testing.assert_array_equal(T.manhattan_distance(np.ones_like(x), backend=backend), 255)


def test_traces_pruning_workspace_and_empty():
    x = np.zeros((41, 63), np.uint8)
    x[5:15, 3:40] = 255
    x[25:37, 20:60] = 255
    expected, paths = T.trace_ridges(x, return_traces=True)
    for backend in backends():
        workspace = T.TraceWorkspace()
        for _ in range(2):
            out, actual = T.trace_ridges(x, backend=backend, return_traces=True, workspace=workspace)
            np.testing.assert_array_equal(out, expected)
            assert len(actual) == len(paths)
            for a, b in zip(actual, paths):
                np.testing.assert_array_equal(a, b)
        out, actual = T.trace_ridges(x, backend=backend, return_traces=True, min_length=10000)
        assert not actual and not out.any()
    assert T.skeletonize(np.empty((0, 4), np.uint8)).shape == (0, 4)
    with pytest.raises(ValueError):
        T.trace_ridges(x, min_length=-1)
    with pytest.raises(ValueError):
        T.skeletonize(x, method="invalid")
    with pytest.raises(TypeError):
        T.skeletonize(x.astype(np.float32))


def operations():
    # The chained mask transforms exercise queued producers/consumers in batch.
    return [A.Morphology(on="mask", operation="dilate", radius=1),
            A.ConnectedComponents(on="mask", full_connectivity=True),
            A.RidgeTrace(on="mask"),
            A.MedialSurface(on="mask", method="guo_hall", preserve_border=True,
                            tube=False, close=False),
            A.ReplaceLabel(on="mask_traces", value=1, replacement=255)]


def test_pipeline_cpu_2d_3d():
    for shape in ((1, 17, 23), (2, 3, 17, 23)):
        x = (np.random.default_rng(6).random(shape) > .65).astype(np.uint8)
        out = A.Pipeline(operations(), backend="cpu")({"mask": x}, seed=0)
        for c in range(x.shape[0]):
            np.testing.assert_array_equal(out["mask_components"][c], T.connected_components(out["mask"][c]))
            np.testing.assert_array_equal(out["mask_skel"][c], T.skeletonize(out["mask"][c], method="guo_hall") / 255)


def test_cuda_resident_and_pipeline_batch(torch_cuda):
    torch = torch_cuda
    x = (np.random.default_rng(8).random((2, 1, 31, 33)) > .65).astype(np.uint8)
    device = torch.as_tensor(x, device="cuda")
    out = torch.empty_like(device)
    torch.cuda.synchronize()
    T.skeletonize_into(device, out, backend="cuda", method="guo_hall")
    np.testing.assert_array_equal(out.cpu(), T.skeletonize(x, method="guo_hall"))
    traced = torch.empty_like(device[0, 0])
    T.trace_ridges_into(device[0, 0], traced)
    np.testing.assert_array_equal(traced.cpu(), T.trace_ridges(x[0, 0]))
    pipeline = A.Pipeline(operations(), backend="cuda")
    result = pipeline.batch({"mask": device}, seeds=[3, 4])
    cpu = A.Pipeline(operations(), backend="cpu")
    samples = [cpu({"mask": sample}, seed=seed) for sample, seed in zip(x, [3, 4])]
    expected = {name: np.stack([s[name] for s in samples]) for name in samples[0]}
    for name, value in result.items():
        assert value.device.type == "cuda"
        np.testing.assert_array_equal(value.cpu(), expected[name])
