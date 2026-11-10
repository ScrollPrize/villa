"""uint8 slice output (to_uint8 on either backend, converted on the device by the reconstructor) and the progress
bar."""
import os

import numpy as np
import pytest

import helirec
from helirec import OutputMask, ProgressBar, ReferenceMask, _core, to_uint8
from helirec.omezarr import OmeZarrWriter
from test_sirt_pipeline import helical_scan, recon


def numpy_uint8(img, lo, hi, disk=None):
    """the conversion written out in numpy: float32 window, round half to even, 0 outside the disk"""
    with np.errstate(invalid="ignore"):
        u = np.rint(np.clip((np.asarray(img, np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    if disk:
        n, x0, y0 = disk
        c = (n - 1) / 2
        yy, xx = np.mgrid[y0:y0 + img.shape[-2], x0:x0 + img.shape[-1]]
        u[..., (xx - c) ** 2 + (yy - c) ** 2 > (n // 2) ** 2] = 0
    return u


def coarse_lookup(mask, shape, scale, offset):
    """mask[(i + offset) // scale] per axis, indices clamped: what an OutputMask means for an output of `shape`"""
    idx = [np.clip((np.arange(n) + o).clip(0) // f, 0, m - 1) for n, f, o, m in zip(shape, scale, offset, mask.shape)]
    return mask[np.ix_(*idx)]


@pytest.mark.parametrize("device", helirec.devices())
def test_to_uint8_with_a_coarser_mask(device):
    rng = np.random.default_rng(3)
    img = rng.random((7, 37, 53), dtype=np.float32)
    plain = _core.to_uint8(img, 0.0, 1.0, device=device)
    for shape, scale, offset in [((7, 37, 53), (1, 1, 1), (0, 0, 0)), ((3, 10, 14), (3, 4, 4), (0, 0, 0)),
                                 ((2, 5, 6), (4, 8, 8), (1, 3, 5)),        # ends beyond the mask: clamped
                                 ((3, 6, 9), (2, 5, 7), (-3, -2, -9))]:    # starts before it
        mask = (rng.random(shape) > 0.4).astype(np.uint8)
        got = _core.to_uint8(img, 0.0, 1.0, device=device, mask=OutputMask(mask, scale, offset))
        np.testing.assert_array_equal(got, np.where(coarse_lookup(mask, img.shape, scale, offset), plain, 0))
        got = _core.to_uint8(img, 0.0, 1.0, (64, 5, 9), device, OutputMask(mask, scale, offset))      # with the disk
        np.testing.assert_array_equal(got, np.where(coarse_lookup(mask, img.shape, scale, offset),
                                                    numpy_uint8(img, 0.0, 1.0, (64, 5, 9)), 0))
    with pytest.raises(RuntimeError, match="scale < 1"):
        OutputMask(mask, (0, 1, 1))


def test_reference_mask_reads_any_pyramid_level(tmp_path):
    rng = np.random.default_rng(4)
    vol = np.zeros((21, 34, 27), np.uint8)
    vol[3:17, 6:30, 4:20] = rng.integers(1, 255, (14, 24, 16))        # a masked volume: 0 outside its support
    w = OmeZarrWriter(str(tmp_path / "ref.zarr"), vol.shape, np.uint8, chunks=(8, 8, 8), levels=3)
    w.write(0, vol)
    w.close()
    ones = np.ones(vol.shape, np.float32)
    for scale in range(3):
        ref = ReferenceMask(str(tmp_path / "ref.zarr"), scale, vol.shape)
        level = w.read(scale, 0, w.level_shapes[scale][0])
        assert tuple(ref.level_shape) == level.shape and tuple(ref.factors) == (2 ** scale,) * 3 and ref.scale == scale
        want = coarse_lookup(level, vol.shape, ref.factors, (0, 0, 0)) != 0
        if scale == 0:
            np.testing.assert_array_equal(want, vol != 0)
        assert want[vol != 0].all()                                   # a coarser level never masks the support's inside
        for (z0, z1), roi in [((0, 21), (0, 27, 0, 34)), ((5, 12), (3, 19, 7, 33)), ((16, 17), (26, 27, 0, 1))]:
            mask = ref.read(z0, z1, roi, slot=scale)
            got = to_uint8(ones[z0:z1, roi[2]:roi[3], roi[0]:roi[1]], 0.0, 1.0, mask=mask)
            np.testing.assert_array_equal(got != 0, want[z0:z1, roi[2]:roi[3], roi[0]:roi[1]])
    with pytest.raises(RuntimeError, match="same grid"):
        ReferenceMask(str(tmp_path / "ref.zarr"), 0, (21, 34, 28))
    with pytest.raises(RuntimeError, match="same grid"):
        ReferenceMask(str(tmp_path / "ref.zarr"), 0, (23, 34, 27))    # 10 % more slices: not the same scan
    with pytest.raises(RuntimeError, match="no pyramid level 5"):
        ReferenceMask(str(tmp_path / "ref.zarr"), 5, vol.shape)


def test_reference_mask_with_a_few_more_slices(tmp_path):
    """a reference of 400 slices on a reconstruction of 404 (another pitch calibration): its slices spread evenly"""
    ref_n, n = 400, 404
    vol = np.zeros((ref_n, 20, 24), np.uint8)
    edge = (60 + 0.5 * np.arange(20)).astype(int)                      # the support's first slice varies with y
    for y in range(20):
        vol[edge[y]:edge[y] + 250, y, 3:21] = 200
    w = OmeZarrWriter(str(tmp_path / "ref.zarr"), vol.shape, np.uint8, chunks=(16, 8, 8), levels=3)
    w.write(0, vol)
    w.close()
    ones = np.ones((n, 20, 24), np.float32)
    for scale in (0, 2):
        ref = ReferenceMask(str(tmp_path / "ref.zarr"), scale, (n, 20, 24))
        assert abs(ref.slice_ratio - ref_n / n) < 1e-12 and tuple(ref.factors) == (2 ** scale,) * 3
        level = w.read(scale, 0, w.level_shapes[scale][0])
        f = 2 ** scale
        for z0, z1 in ((0, n), (37, 101), (300, 404), (203, 204)):
            got = to_uint8(ones[z0:z1], 0.0, 1.0, mask=ref.read(z0, z1, (0, 24, 0, 20))) != 0
            z = np.arange(z0, z1)
            middle = (z // f) * f + (f - 1) / 2                         # of the f slices that share a mask row
            src = np.clip(np.floor((((middle + 0.5) * ref_n / n - 0.5) - (f - 1) / 2) / f + 0.5).astype(int), 0, level.shape[0] - 1)
            want = level[src][:, np.minimum(np.arange(20) // f, level.shape[1] - 1)][:, :, np.minimum(np.arange(24) // f, level.shape[2] - 1)] != 0
            np.testing.assert_array_equal(got, want)
        # the support begins where the reference's does, stretched: within a slice and the level's voxel
        whole = to_uint8(ones, 0.0, 1.0, mask=ref.read(0, n, (0, 24, 0, 20))) != 0
        first = whole[:, 10, 10].argmax()
        assert abs(first - (edge[10] + 0.5) * n / ref_n) <= 1 + f


@pytest.mark.parametrize("device", helirec.devices())
def test_to_uint8_matches_numpy(device):
    rng = np.random.default_rng(0)
    img = rng.random((3, 37, 53), dtype=np.float32) * 0.6 - 0.15
    img[0, 0, :5] = [np.nan, np.inf, -np.inf, -0.06, 0.29]
    got = _core.to_uint8(img, -0.06, 0.29, device=device)
    np.testing.assert_array_equal(got, numpy_uint8(img, -0.06, 0.29))
    np.testing.assert_array_equal(got[0, 0, :5], [0, 255, 0, 0, 255])
    np.testing.assert_array_equal(_core.to_uint8(img[1, 2], -0.06, 0.29, device=device), got[1, 2])   # one row
    for disk in [(64, 5, 9), (65, 0, 20), (40, 0, 0)]:     # even / odd slices, ROIs, a row wholly outside
        np.testing.assert_array_equal(_core.to_uint8(img, -0.06, 0.29, disk, device), numpy_uint8(img, -0.06, 0.29, disk))
    mask = (rng.random(img.shape) > 0.4).astype(np.uint8) * 7      # any nonzero value keeps the pixel
    for disk in [None, (64, 5, 9)]:
        np.testing.assert_array_equal(_core.to_uint8(img, -0.06, 0.29, disk, device, mask),
                                      np.where(mask, numpy_uint8(img, -0.06, 0.29, disk), 0))
    with pytest.raises(RuntimeError, match="hi > lo"):
        _core.to_uint8(img, 1.0, 1.0, device=device)
    with pytest.raises(RuntimeError, match="shape of img"):
        _core.to_uint8(img, 0.0, 1.0, device=device, mask=np.ascontiguousarray(mask[:2]))


@pytest.mark.parametrize("device", helirec.devices())
@pytest.mark.parametrize("backprojector", ["fbp", "ghbp"])
@pytest.mark.parametrize("roi", [None, (3, 14, 2, 17)])
def test_reconstruct_uint8_is_the_window_of_the_float_slices(tmp_path, device, backprojector, roi):
    scan, *_ = helical_scan(tmp_path)
    r = recon(scan, device, backprojector=backprojector, clip_outer_circle=False)
    lo, hi = r.info["doable"]
    h0, h1 = lo + 1, min(hi, lo + 4)
    f = r.reconstruct(h0, h1, roi=roi)
    window = (float(np.percentile(f, 20)), float(np.percentile(f, 90)))
    N = r.info["grid_size"]
    disk = (N, roi[0], roi[2]) if roi else (N, 0, 0)
    for zero_outside in (False, True):
        both, u = r.reconstruct(h0, h1, roi=roi, uint8=(*window, zero_outside))
        np.testing.assert_array_equal(both, f)
        np.testing.assert_array_equal(u, numpy_uint8(f, *window, disk if zero_outside else None))
        np.testing.assert_array_equal(r.reconstruct(h0, h1, roi=roi, uint8=(*window, zero_outside), float32=False), u)
        np.testing.assert_array_equal(to_uint8(f, *window, disk if zero_outside else None), u)
    assert u.dtype == np.uint8 and len(np.unique(u)) > 10
    mask = (np.random.default_rng(1).random(u.shape) > 0.5).astype(np.uint8)
    np.testing.assert_array_equal(r.reconstruct(h0, h1, roi=roi, uint8=(*window, True), float32=False, mask=mask),
                                  np.where(mask, u, 0))
    with pytest.raises(RuntimeError, match="no output"):
        r.reconstruct(h0, h1, float32=False)
    coarse = (np.random.default_rng(2).random((2, 5, 4)) > 0.4).astype(np.uint8)
    scale, offset = (2, 4, 5), (1, 2, 3)
    np.testing.assert_array_equal(r.reconstruct(h0, h1, roi=roi, uint8=(*window, True), float32=False,
                                                mask=OutputMask(coarse, scale, offset)),
                                  np.where(coarse_lookup(coarse, u.shape, scale, offset), u, 0))
    with pytest.raises(RuntimeError, match="shape of the output"):
        r.reconstruct(h0, h1, roi=roi, uint8=(*window, True), mask=mask[:1])
    with pytest.raises(RuntimeError, match="uint8 output"):
        r.reconstruct(h0, h1, roi=roi, mask=mask)


@pytest.mark.parametrize("device", helirec.devices())
@pytest.mark.parametrize("backprojector", ["fbp", "ghbp"])
def test_reconstruct_only_the_pixels_needed(tmp_path, device, backprojector):
    scan, *_ = helical_scan(tmp_path)
    r = recon(scan, device, backprojector=backprojector, clip_outer_circle=False)
    lo, hi = r.info["doable"]
    h0, h1 = lo + 1, min(hi, lo + 4)
    N = r.info["grid_size"]
    whole = r.reconstruct(h0, h1)
    window = (float(np.percentile(whole, 20)), float(np.percentile(whole, 90)))
    whole_u8 = numpy_uint8(whole, *window, (N, 0, 0))

    def inside(shape_roi, region):      # the pixels of `region` in an output of roi (x0, x1, y0, y1)
        x0, x1, y0, y1 = shape_roi
        keep = np.zeros((y1 - y0, x1 - x0), bool)
        keep[max(region[2], y0) - y0:max(region[3] - y0, 0), max(region[0], x0) - x0:max(region[1] - x0, 0)] = True
        return keep

    for roi in [None, (2, 15, 1, 16)]:
        box = roi or (0, N, 0, N)
        crop = (slice(None), slice(box[2], box[3]), slice(box[0], box[1]))
        for region in [(4, 11, 3, 9), (0, N, 0, N), (0, 5, 12, N), (12, 14, 0, 2)]:
            keep = inside(box, region)
            f, u = r.reconstruct(h0, h1, roi=roi, region=region, uint8=(*window, True))
            np.testing.assert_array_equal(f, np.where(keep, whole[crop], 0))        # the same values, 0 elsewhere
            np.testing.assert_array_equal(u, np.where(keep, whole_u8[crop], 0))
        if roi:                                                                     # a region wholly outside the ROI
            f, u = r.reconstruct(h0, h1, roi=roi, region=(0, 2, 0, 1), uint8=(*window, True))
            assert not f.any() and not u.any()
        # uint8 alone with a mask: only the bounds of what it keeps are computed, the result is the same
        rng = np.random.default_rng(5)
        for scale, offset in [((1, 1, 1), (0, 0, 0)), ((2, 3, 4), (1, 2, 1))]:
            shape = tuple(-(-(n + o) // f) for n, f, o in zip(whole[crop].shape, scale, offset))
            for fill in (0.0, 0.3, 1.0):                                            # none kept, some, all
                coarse = (rng.random(shape) < fill).astype(np.uint8)
                if fill == 0.3:
                    coarse[:, :1], coarse[:, :, -1:] = 0, 0                          # bounds inside the output
                kept = coarse_lookup(coarse, whole[crop].shape, scale, offset) != 0
                want = np.where(kept, whole_u8[crop], 0)
                got = r.reconstruct(h0, h1, roi=roi, uint8=(*window, True), float32=False, mask=OutputMask(coarse, scale, offset))
                np.testing.assert_array_equal(got, want)
                f, u = r.reconstruct(h0, h1, roi=roi, uint8=(*window, True), mask=OutputMask(coarse, scale, offset))
                np.testing.assert_array_equal(u, want)
                np.testing.assert_array_equal(f, whole[crop])                       # float slices are whole
    with pytest.raises(RuntimeError, match="empty"):
        r.reconstruct(h0, h1, region=(3, 3, 0, 4))


@pytest.mark.parametrize("device", helirec.devices())
def test_reconstruct_into_a_writer(tmp_path, device):
    """a writer takes the slices as the blocks computed: what it stores is what the arrays hold"""
    scan, *_ = helical_scan(tmp_path)
    r = recon(scan, device, backprojector="fbp", clip_outer_circle=False, grid_size=70)     # several backprojector tiles
    lo, hi = r.info["doable"]
    h0, h1 = lo + 1, min(hi, lo + 6)
    n = h1 - h0
    whole = r.reconstruct(h0, h1)
    window = (float(np.percentile(whole, 20)), float(np.percentile(whole, 90)))
    rng = np.random.default_rng(7)
    k = 0
    for roi, region, masked in [(None, None, False), ((3, 60, 8, 70), (10, 50, 20, 44), False), (None, None, True),
                                ((3, 60, 8, 70), None, True)]:
        shape = (n, roi[3] - roi[2], roi[1] - roi[0]) if roi else whole.shape
        mask = rest = None
        if masked:      # a coarse mask keeping patches: some tiles hold nothing, slices of the same mask element share it
            coarse = (rng.random(tuple(-(-s // f) for s, f in zip(shape, (2, 8, 8)))) < 0.3).astype(np.uint8)
            mask, rest = OutputMask(coarse, (2, 8, 8)), OutputMask(coarse, (2, 8, 8), (2, 0, 0))    # (of the slices from 2 on)
        kw = dict(roi=roi, region=region, uint8=(*window, True), mask=mask)
        f, u = r.reconstruct(h0, h1, **kw)
        assert u.any() and (not masked or (u == 0).mean() > 0.5)
        vol = (n + 3,) + shape[1:]                                                 # the chunk's slices at z 1.., in two calls
        k += 1
        wf = OmeZarrWriter(str(tmp_path / f"f{k}.zarr"), vol, np.float32, chunks=(4, 16, 32))
        wu = OmeZarrWriter(str(tmp_path / f"u{k}.zarr"), vol, np.uint8, chunks=(4, 16, 32), levels=2)
        assert r.reconstruct(h0, h0 + 2, float32_to=wf, uint8_to=wu, z_to=1, **kw) == (None, None)
        both = r.reconstruct(h0 + 2, h1, uint8_to=wu, z_to=3, **dict(kw, mask=rest))   # a writer for one, an array for the other
        assert both[1] is None
        np.testing.assert_array_equal(both[0], f[2:])
        wf.write(3, both[0])
        only = OmeZarrWriter(str(tmp_path / f"o{k}.zarr"), shape, np.uint8, chunks=(4, 16, 32))
        assert r.reconstruct(h0, h1, float32=False, uint8_to=only, **kw) is None    # uint8 alone: bounds and tiles of the mask
        for w in (wf, wu, only):
            w.close()
        pad = lambda a: np.concatenate([np.zeros_like(a[:1]), a, np.zeros_like(a[:2])])
        np.testing.assert_array_equal(wf.read(0, 0, n + 3), pad(f))
        np.testing.assert_array_equal(wu.read(0, 0, n + 3), pad(u))
        np.testing.assert_array_equal(only.read(0, 0, n), u)
        assert wu.read(1, 0, wu.level_shapes[1][0]).any()
    with pytest.raises(RuntimeError, match="dtype"):
        r.reconstruct(h0, h1, float32_to=only)
    with pytest.raises(RuntimeError, match="does not hold"):
        r.reconstruct(h0, h1, uint8=(*window, True), float32=False, uint8_to=only, z_to=1)
    with pytest.raises(RuntimeError, match="written twice"):
        again = OmeZarrWriter(str(tmp_path / "again.zarr"), whole.shape, np.float32, chunks=(8, 16, 32))
        r.reconstruct(h0, h1, float32_to=again)
        r.reconstruct(h0, h1, float32_to=again)


def test_remote_reference_reads_through_a_cache(tmp_path):
    """an http(s):// / s3:// Zarr is read chunk by chunk, each kept in the cache directory once read"""
    import functools
    import http.server
    import json
    import shutil
    import threading
    rng = np.random.default_rng(6)
    vol = np.zeros((21, 34, 27), np.uint8)
    vol[3:17, 6:30, 4:20] = rng.integers(1, 255, (14, 24, 16))
    root = tmp_path / "served"
    w = OmeZarrWriter(str(root / "ref.zarr"), vol.shape, np.uint8, chunks=(8, 8, 8), levels=3)
    w.write(0, vol)
    w.close()
    # the same pyramid without the translations of its levels: taken as block-reduced, which is what they say
    shutil.copytree(root / "ref.zarr", root / "plain.zarr")
    attrs = json.loads((root / "plain.zarr" / ".zattrs").read_text())
    for d in attrs["multiscales"][0]["datasets"]:
        d["coordinateTransformations"] = [t for t in d["coordinateTransformations"] if t["type"] == "scale"]
    (root / "plain.zarr" / ".zattrs").write_text(json.dumps(attrs))

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    ones = np.ones(vol.shape, np.float32)
    read = lambda ref, slot=0: to_uint8(ones, 0.0, 1.0, mask=ref.read(0, 21, (0, 27, 0, 34), slot))
    cache = str(tmp_path / "cache")
    try:
        for scale in range(3):
            want = read(ReferenceMask(str(root / "ref.zarr"), scale, vol.shape))
            np.testing.assert_array_equal(read(ReferenceMask(str(root / "plain.zarr"), scale, vol.shape)), want)
            np.testing.assert_array_equal(read(ReferenceMask(f"{url}/ref.zarr", scale, vol.shape)), want)           # no cache
            np.testing.assert_array_equal(read(ReferenceMask(f"{url}/plain.zarr", scale, vol.shape, cache)), want)
        with pytest.raises(RuntimeError, match="no Zarr v2 array or OME-Zarr group"):
            ReferenceMask(f"{url}/nothing.zarr", 0, vol.shape)
    finally:
        server.shutdown()
        server.server_close()
    # the server gone: chunks, the fact that others do not exist, and the metadata all come from the cache
    np.testing.assert_array_equal(read(ReferenceMask(f"{url}/plain.zarr", 2, vol.shape, cache)), want)
    with pytest.raises(RuntimeError, match="cannot read"):
        ReferenceMask(f"{url}/ref.zarr", 2, vol.shape, cache)


def test_progress_text():
    f = ProgressBar.format
    assert f(2272, 5398, 192, 11.8, "reconstruct", "slice", "backproject", 100, False) == \
        "reconstruct:  42%|############6                 | 2272/5398 [03:12<04:25, 11.8 slice/s, backproject]"
    assert "█▋" in f(2272, 5398, 192, 11.8, "reconstruct", "slice", "backproject", 100, True)
    assert f(3, 8, 4000, 0.4, "slices", "slice", "", 40, False).endswith("3/8 [1:06:40<00:13, 2.5 s/slice]")
    assert f(8, 8, 5, 250, "", "it", "", 39, False) == "100%|#####| 8/8 [00:05<00:00, 250 it/s]"
    assert f(0, 8, 0, 0, "", "it", "", 33, False) == "  0%|          | 0/8 [00:00<?, ?]"
    assert f(8, 8, 5, 250, "", "it", "", 30, False) == "100% 8/8 [00:05<00:00, 250 it/s]"         # no room for a bar
    assert f(17, 0, 12, 0, "files", "file", "", 70, True) == "files: 17 file [00:12, ?]"    # unknown total
    for width in range(0, 120):                                    # never wider than asked, once the text fits
        s = f(5, 9, 1, 2, "d", "it", "p", width, False)
        assert len(s) <= max(width, len(f(5, 9, 1, 2, "d", "it", "p", 0, False)))


def read_all(fd):
    out = b""
    while chunk := os.read(fd, 65536):
        out += chunk
    return out.decode()


def test_progress_lines_when_not_a_terminal():
    r, w = os.pipe()
    with ProgressBar(4, "demo", "it", fd=w, log_interval=0.0, width=60) as bar:
        for _ in range(4):
            bar.update()
        bar.set_postfix("write")            # no line of its own
        bar.write("a message")
        assert bar.count == 4 and bar.total == 4
    bar.update()                            # after close: counted, nothing printed
    os.close(w)
    lines = read_all(r).splitlines()
    assert [l.split("|")[0] for l in lines if "|" in l] == ["demo:   0%", "demo:  25%", "demo:  50%", "demo:  75%",
                                                           "demo: 100%", "demo: 100%"]
    assert lines[-2] == "a message" and lines[-1].endswith(", write]") and all(len(l) <= 60 for l in lines)
    assert "\r" not in "".join(lines)


def test_progress_redraws_in_place_on_a_terminal():
    r, w = os.pipe()
    bar = ProgressBar(2, "demo", fd=w, terminal=True, min_interval=0.0, width=40, leave=False)
    bar.update(2)
    bar.close()
    os.close(w)
    text = read_all(r)
    assert text.startswith("\rdemo:   0%|") and "\n" not in text and text.endswith("\r\x1b[K")


@pytest.mark.parametrize("device", helirec.devices())
def test_reconstruct_advances_the_bar_per_slice(tmp_path, device):
    scan, *_ = helical_scan(tmp_path)
    r = recon(scan, device, backprojector="fbp")
    lo, hi = r.info["doable"]
    rd, wr = os.pipe()
    bar = ProgressBar(hi - lo, "reconstruct", "slice", fd=wr, log_interval=1e9)
    for h in range(lo, hi, 3):
        r.reconstruct(h, min(hi, h + 3), progress=bar)
    assert bar.count == hi - lo
    bar.close()
    os.close(wr)
    assert read_all(rd).splitlines()[-1].endswith(", backproject]")
