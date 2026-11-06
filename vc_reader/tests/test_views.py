"""Patch-lattice views: orientations, placement, correspondence, masks."""
import itertools

import numpy as np
import pytest
from build_support import load_build
load_build(names=("_views",))
from vc_reader import views as V

PATCH = 2


def oriented(crop, view_id):
    """numpy.transpose then flip of the three leading axes."""
    perm, flips = V.orientation(view_id)
    out = np.transpose(crop, list(perm) + list(range(3, crop.ndim)))
    return np.flip(out, [k for k in range(3) if flips[k]])


def canonical(blocks, view_id):
    """Undo an orientation of (tokens, P, P, P) blocks."""
    perm, flips = V.orientation(view_id)
    blocks = np.flip(blocks, [k + 1 for k in range(3) if flips[k]])
    return np.transpose(blocks, [0] + [1 + perm.index(a) for a in range(3)])


def token_blocks(voxels):
    d, h, w = (s // PATCH for s in voxels.shape)
    blocks = voxels.reshape(d, PATCH, h, PATCH, w, PATCH).transpose(0, 2, 4, 1, 3, 5)
    return blocks.reshape(d * h * w, PATCH, PATCH, PATCH)


def crop(volume, origin, shape):
    return volume[tuple(slice(PATCH * o, PATCH * (o + n)) for o, n in zip(origin, shape))]


def reference_check(ids, support, so, ns, sr, to, nt, tr, teacher_index, student_valid):
    """Brute force against unique voxel IDs for one pair."""
    student = token_blocks(oriented(crop(ids, so, ns), sr))
    teacher = token_blocks(oriented(crop(ids, to, nt), tr))
    valid = token_blocks(oriented(crop(np.repeat(np.repeat(np.repeat(support, PATCH, 0), PATCH, 1), PATCH, 2), so, ns), sr))
    np.testing.assert_array_equal(student_valid, valid.reshape(len(valid), -1).min(1))
    teacher_ids = set(teacher.reshape(-1).tolist())
    for i, j in enumerate(teacher_index):
        if j >= 0:
            assert student_valid[i]
            assert set(student[i].reshape(-1).tolist()) == set(teacher[j].reshape(-1).tolist())
            np.testing.assert_array_equal(canonical(student[i:i + 1], sr), canonical(teacher[j:j + 1], tr))
        else:
            assert not student_valid[i] or not (set(student[i].reshape(-1).tolist()) & teacher_ids)


@pytest.fixture
def fixture():
    rng = np.random.default_rng(3)
    grid = (5, 4, 6)
    support = (rng.random(grid) > .2).astype(np.uint8)
    ids = np.arange(np.prod([PATCH * g for g in grid])).reshape([PATCH * g for g in grid])
    return rng, grid, support, ids


def test_orientations():
    assert V.ORIENTATION_COUNT == 48
    perms = list(itertools.permutations(range(3)))
    for view_id in range(48):
        perm, flips = V.orientation(view_id)
        assert perm == perms[view_id // 8] and flips == tuple(bool(view_id >> k & 1) for k in range(3))
        assert V.orientation_id(perm, flips) == view_id
    with pytest.raises(ValueError):
        V.orientation(48)
    ids = V.draw_orientations(np.arange(48 * 400, dtype=np.uint64))
    assert ids.dtype == np.int32
    counts = np.bincount(ids, minlength=48)
    assert counts.min() > 0 and ((counts - 400) ** 2 / 400).sum() < 100  # chi-square, 47 dof
    np.testing.assert_array_equal(V.draw_orientations(np.arange(48 * 400, dtype=np.uint64)[::-1]), ids[::-1])


def test_correspondence_all_orientation_pairs(fixture):
    rng, grid, support, ids = fixture
    ns, nt = (3, 2, 3), (4, 3, 4)
    pairs = [(a, b) for a in range(48) for b in range(48)]
    so = np.stack([rng.integers(0, np.subtract(grid, ns) + 1) for _ in pairs])
    to = np.stack([rng.integers(0, np.subtract(grid, nt) + 1) for _ in pairs])
    sr, tr = np.array(pairs).T
    out = V.correspondence(support, np.zeros(len(pairs), np.int64), so, ns, sr, to, nt, tr, lattice=True)
    assert out["teacher_index"].dtype == np.int32 and out["student_valid"].dtype == np.uint8
    assert out["teacher_index"].shape == (len(pairs), 18) and out["lattice"].shape == (len(pairs), 18, 3)
    for p in range(len(pairs)):
        reference_check(ids, support, so[p], ns, sr[p], to[p], nt, tr[p], out["teacher_index"][p], out["student_valid"][p])
        lattice = np.stack(np.meshgrid(*[np.arange(n) for n in ns], indexing="ij"), -1) + so[p]
        np.testing.assert_array_equal(out["lattice"][p], oriented(lattice, sr[p]).reshape(-1, 3))
    assert (out["teacher_index"] >= 0).any() and (out["teacher_index"] < 0).any()


@pytest.mark.parametrize("ns,nt,so,to", [
    ((2, 2, 2), (4, 4, 5), (1, 1, 2), (0, 0, 0)),   # contained
    ((3, 3, 3), (3, 3, 3), (1, 0, 2), (1, 0, 2)),   # coincident
    ((3, 4, 4), (4, 2, 3), (0, 0, 1), (1, 1, 3)),   # partial overlap
    ((2, 2, 2), (2, 2, 2), (0, 0, 0), (3, 2, 4)),   # disjoint
])
def test_correspondence_shapes_and_shifts(fixture, ns, nt, so, to):
    rng, grid, support, ids = fixture
    sr, tr = rng.integers(0, 48, 20), rng.integers(0, 48, 20)
    out = V.correspondence(support[None].repeat(2, 0), np.ones(20, np.int64), [so] * 20, ns, sr, [to] * 20, nt, tr)
    for p in range(20):
        reference_check(ids, support, so, ns, sr[p], to, nt, tr[p], out["teacher_index"][p], out["student_valid"][p])


def test_correspondence_identity_is_a_shift():
    support = np.ones((6, 6, 6), np.uint8)
    support[2, 3, 1] = 0
    ns, nt, so, to = (3, 4, 2), (4, 3, 5), (1, 2, 0), (0, 1, 1)
    out = V.correspondence(support, [0], [so], ns, [0], [to], nt, [0], lattice=True)
    for i, u in enumerate(itertools.product(*map(range, ns))):
        g = np.add(u, so)
        t = g - to
        inside = (t >= 0).all() and (t < nt).all() and support[tuple(g)]
        expected = np.ravel_multi_index(t, nt) if inside else -1
        assert out["teacher_index"][0, i] == expected
        assert out["student_valid"][0, i] == support[tuple(g)]


def test_correspondence_rejects_bad_pairs():
    support = np.ones((1, 4, 4, 4), np.uint8)
    with pytest.raises(ValueError):
        V.correspondence(support, [1], [(0, 0, 0)], (2, 2, 2), [0], [(0, 0, 0)], (2, 2, 2), [0])
    with pytest.raises(ValueError):
        V.correspondence(support, [0], [(0, 0, 0)], (2, 2, 2), [48], [(0, 0, 0)], (2, 2, 2), [0])


def test_token_support():
    rng = np.random.default_rng(1)
    valid = rng.random((3, 8, 6, 4)) > .03
    out = V.token_support(valid, (4, 2, 2))
    expected = valid.reshape(3, 2, 4, 3, 2, 2, 2).all((2, 4, 6))
    assert out.dtype == np.uint8 and out.shape == (3, 2, 3, 2)
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(V.token_support(valid[0].astype(np.uint8) * 7, 2), valid[0].reshape(4, 2, 3, 2, 2, 2).all((1, 3, 5)))
    with pytest.raises(ValueError):
        V.token_support(valid, 3)


def plan(support, **kwargs):
    options = dict(n_sets=64, min_overlap_fraction=.5, min_overlap_count=4, max_student_attempts=32,
                   max_teacher_attempts=16, seed=11)
    options.update(kwargs)
    return V.plan_view_sets(support, options.pop("teacher", (8, 8, 8)),
                            options.pop("students", [(8, 8, 8), (4, 4, 4), (4, 4, 4)]), **options)


def test_planner_overlap_rule_and_determinism():
    rng = np.random.default_rng(5)
    support = (rng.random((12, 11, 13)) > .3).astype(np.uint8)
    a, b = plan(support), plan(support)
    assert a["ok"] is True and a["set_ok"].all()
    for key in ("teacher_origins", "student_origins", "student_overlap", "student_attempts", "teacher_attempts", "rejections"):
        np.testing.assert_array_equal(a[key], b[key])
    assert a["teacher_origins"].dtype == np.int32 and a["student_origins"].shape == (64, 3, 3)
    assert a["student_overlap"].dtype == np.int64 and a["rejections"].dtype == np.int64
    assert a["student_attempts"].dtype == np.int32 and a["teacher_attempts"].dtype == np.int32
    assert not np.array_equal(plan(support, seed=12)["student_origins"], a["student_origins"])
    shapes = [(8, 8, 8), (4, 4, 4), (4, 4, 4)]
    for s in range(64):
        t = a["teacher_origins"][s]
        for k, n in enumerate(shapes):
            o = a["student_origins"][s, k]
            lo, hi = np.maximum(o, t), np.minimum(o + n, t + 8)
            overlap = int(support[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].sum()) if (hi > lo).all() else 0
            assert overlap == a["student_overlap"][s, k]
            assert overlap >= 4 and overlap >= .5 * np.prod(n)
            assert (o >= 0).all() and (o + n <= support.shape).all()
            assert 1 <= a["student_attempts"][s, k] <= 32


def test_planner_failure_and_errors():
    empty = np.zeros((10, 10, 10), np.uint8)
    result = plan(empty)
    assert result["ok"] is False and not result["set_ok"].any()
    assert (result["teacher_attempts"] == 16).all() and (result["rejections"] == 16 * 32).all()
    with pytest.raises(ValueError):
        plan(empty, teacher=(11, 4, 4))
    with pytest.raises(ValueError):
        plan(empty, students=[(4, 0, 4)])


def test_planner_proposal_law_and_containment():
    support = np.ones((20, 20, 20), np.uint8)
    free = plan(support, n_sets=6000, min_overlap_fraction=0, min_overlap_count=0, teacher=(8, 8, 8),
                students=[(4, 4, 4)])
    assert (free["student_attempts"] == 1).all() and (free["rejections"] == 0).all()
    for origins in (free["teacher_origins"], free["student_origins"][:, 0]):
        for axis, span in zip(origins.T, (13, 13, 13) if origins is free["teacher_origins"] else (17, 17, 17)):
            counts = np.bincount(axis, minlength=span)
            assert len(counts) == span and ((counts - 6000 / span) ** 2 / (6000 / span)).sum() < 70
    contained = plan(support, min_overlap_fraction=1.0, min_overlap_count=0)
    o, t = contained["student_origins"][:, 1:], contained["teacher_origins"][:, None]
    assert ((o >= t) & (o + 4 <= t + 8)).all()
    loose = plan(support, n_sets=200, min_overlap_fraction=.25)
    o, t = loose["student_origins"][:, 1:], loose["teacher_origins"][:, None]
    assert ((o >= t) & (o + 4 <= t + 8)).all(-1).any()


@pytest.mark.parametrize("mode", ["random", "multiblock"])
def test_patch_masks_counts_and_independence(mode):
    seeds = np.random.default_rng(2).integers(0, 2**63, 300, dtype=np.uint64)
    options = dict(mode=mode, probability=.6, fraction=.37, max_block_tokens=168)
    out = V.patch_masks((6, 7, 8), seeds, **options)
    assert out["masks"].dtype == np.uint8 and out["applied"].dtype == np.bool_ and out["counts"].dtype == np.int64
    n = int(np.floor(.37 * 336 + .5))
    np.testing.assert_array_equal(out["counts"], np.where(out["applied"], n, 0))
    np.testing.assert_array_equal(out["masks"].sum((1, 2, 3)), out["counts"])
    assert set(np.unique(out["masks"])) <= {0, 1}
    reversed_ = V.patch_masks((6, 7, 8), seeds[::-1], **options)
    np.testing.assert_array_equal(reversed_["masks"][::-1], out["masks"])
    chunks = [V.patch_masks((6, 7, 8), part, **options)["masks"] for part in np.array_split(seeds, 7)]
    np.testing.assert_array_equal(np.concatenate(chunks), out["masks"])
    single = V.patch_masks((6, 7, 8), seeds[42:43], **options)
    np.testing.assert_array_equal(single["masks"][0], out["masks"][42])


def test_patch_masks_probability_and_uniform_marginal():
    seeds = np.arange(4000, dtype=np.uint64)
    out = V.patch_masks((4, 4, 4), seeds, mode="random", probability=.3, fraction=.25)
    assert abs(out["applied"].mean() - .3) < .03
    marginal = out["masks"][out["applied"]].mean(0)
    assert abs(marginal.mean() - .25) < 1e-9 and np.abs(marginal - .25).max() < .07
    assert not V.patch_masks((4, 4, 4), seeds[:10], mode="random", probability=0, fraction=.5)["masks"].any()
    full = V.patch_masks((4, 4, 4), seeds[:10], mode="multiblock", probability=1, fraction=1)
    assert full["masks"].all()


def adjacency(masks):
    pairs = sum((masks & np.roll(masks, 1, a)).sum() for a in (1, 2, 3))
    return pairs / masks.sum()


def test_multiblock_structure_and_shift():
    seeds = np.arange(200, dtype=np.uint64)
    options = dict(probability=1, fraction=.3, max_block_tokens=.5 * 16 ** 3)
    blocks = V.patch_masks((16, 16, 16), seeds, mode="multiblock", cyclic_shift=False, **options)["masks"]
    shifted = V.patch_masks((16, 16, 16), seeds, mode="multiblock", cyclic_shift=True, **options)["masks"]
    uniform = V.patch_masks((16, 16, 16), seeds, mode="random", **options)["masks"]
    assert adjacency(blocks) > 2.0 > 1.2 > adjacency(uniform)
    np.testing.assert_array_equal(blocks.sum((1, 2, 3)), shifted.sum((1, 2, 3)))
    small = dict(probability=1, fraction=.4, max_block_tokens=84)
    for seed in range(20):
        a = V.patch_masks((6, 7, 8), [seed], mode="multiblock", cyclic_shift=False, **small)["masks"][0]
        b = V.patch_masks((6, 7, 8), [seed], mode="multiblock", cyclic_shift=True, **small)["masks"][0]
        assert any(np.array_equal(np.roll(a, s, (0, 1, 2)), b) for s in itertools.product(range(6), range(7), range(8)))


def test_mask_patch_cells():
    rng = np.random.default_rng(8)
    images = rng.random((2, 3, 4, 6, 8), dtype=np.float32)
    masks = rng.random((2, 2, 3, 4)) > .5
    copy = images.copy()
    out = V.mask_patch_cells(images, masks, 2, fill=-1.5)
    voxel_mask = masks.repeat(2, 1).repeat(2, 2).repeat(2, 3)[:, None]
    np.testing.assert_array_equal(out, np.where(voxel_mask, np.float32(-1.5), images))
    np.testing.assert_array_equal(images, copy)
    result = V.mask_patch_cells(images, masks.astype(np.uint8), (2, 2, 2), fill=-1.5, out=images)
    assert result is images
    np.testing.assert_array_equal(images, out)
    with pytest.raises(ValueError):
        V.mask_patch_cells(images, masks[:, :1], 2)


def test_torch_cpu_matches_numpy():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    valid = rng.random((2, 8, 8, 8)) > .02
    np.testing.assert_array_equal(V.token_support(torch.from_numpy(valid), 2).numpy(), V.token_support(valid, 2))
    images = torch.from_numpy(rng.random((2, 1, 8, 8, 8), dtype=np.float32))
    masks = torch.from_numpy(rng.random((2, 4, 4, 4)) > .5)
    out = V.mask_patch_cells(images, masks, 2, fill=3)
    assert isinstance(out, torch.Tensor)
    np.testing.assert_array_equal(out.numpy(), V.mask_patch_cells(images.numpy(), masks.numpy(), 2, fill=3))
    for dtype in (torch.float16, torch.bfloat16):
        half = V.mask_patch_cells(images.to(dtype), masks, 2, fill=0.1)
        expected = torch.where(masks.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3)[:, None],
                               torch.tensor(0.1, dtype=dtype), images.to(dtype))
        assert torch.equal(half, expected)


def test_cuda_matches_cpu(torch_cuda, fixture):
    torch = torch_cuda
    rng, grid, support, ids = fixture
    valid = rng.random((3, 10, 8, 12)) > .02
    gpu = V.token_support(torch.from_numpy(valid).cuda(), 2)
    assert gpu.is_cuda and gpu.dtype == torch.uint8
    np.testing.assert_array_equal(gpu.cpu().numpy(), V.token_support(valid, 2))

    supports = np.stack([support, (rng.random(grid) > .5).astype(np.uint8)])
    count, ns, nt = 3000, (3, 2, 3), (4, 3, 4)
    args = (rng.integers(0, 2, count), rng.integers(-1, 4, (count, 3)), ns, rng.integers(0, 48, count),
            rng.integers(0, 3, (count, 3)), nt, rng.integers(0, 48, count))
    cpu = V.correspondence(supports, *args, lattice=True)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        device_args = [torch.from_numpy(a).cuda() if isinstance(a, np.ndarray) else a for a in args]
        gpu = V.correspondence(torch.from_numpy(supports).cuda(), *device_args, lattice=True)
    stream.synchronize()
    for key in cpu:
        assert gpu[key].is_cuda
        np.testing.assert_array_equal(gpu[key].cpu().numpy(), cpu[key])

    images = rng.random((2, 3, 8, 6, 10), dtype=np.float32)
    masks = rng.random((2, 4, 3, 5)) > .5
    expected = V.mask_patch_cells(images, masks, 2, fill=-4)
    with torch.cuda.stream(stream):
        source = torch.from_numpy(images).cuda()
        out = V.mask_patch_cells(source, torch.from_numpy(masks).cuda(), 2, fill=-4)
        np.testing.assert_array_equal(out.cpu().numpy(), expected)
        np.testing.assert_array_equal(source.cpu().numpy(), images)
        V.mask_patch_cells(source, torch.from_numpy(masks).cuda(), 2, fill=-4, out=source)
        np.testing.assert_array_equal(source.cpu().numpy(), expected)
        for dtype in (torch.float16, torch.bfloat16):
            half = torch.from_numpy(images).to(dtype)
            assert torch.equal(V.mask_patch_cells(half.cuda(), torch.from_numpy(masks).cuda(), 2, fill=0.1).cpu(),
                               V.mask_patch_cells(half, torch.from_numpy(masks), 2, fill=0.1))
