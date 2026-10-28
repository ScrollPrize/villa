import json
import math
import os
import sys

import numpy as np
import pytest
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import check_winding_annotations as cwa  # noqa: E402
from umbilicus import json_umbilicus_z_to_yx  # noqa: E402

PITCH, R0, CX, CY, NZ, NYX = 12.0, 20.0, 100.0, 100.0, 16, 200


def spiral_radius(w, theta):
    """Archimedean spiral r = R0 + PITCH * (w + theta / 2pi), theta in [0, 2pi)."""
    return R0 + PITCH * (w + (theta % (2 * math.pi)) / (2 * math.pi))


@pytest.fixture(scope='module')
def scene(tmp_path_factory):
    d = tmp_path_factory.mktemp('spiral')
    y, x = np.mgrid[0:NYX, 0:NYX].astype(float)
    r, th = np.hypot(x - CX, y - CY), np.arctan2(y - CY, x - CX) % (2 * math.pi)
    s = (r - R0) / PITCH - th / (2 * math.pi)
    sheet = (np.abs(s - np.round(s)) * PITCH < 1.5) & (s > 0.5)
    vol = np.repeat(sheet[None], NZ, axis=0).astype(np.uint8) * 255
    arr = zarr.create_array(str(d / 'pred.zarr'), shape=vol.shape, chunks=(8, 64, 64), dtype='uint8')
    arr[:] = vol
    for name, v in (('pred01.zarr', vol // 255), ('predf.zarr', vol.astype('float32') / 255 * 0.9)):
        a = zarr.create_array(str(d / name), shape=vol.shape, chunks=(8, 64, 64), dtype=v.dtype)
        a[:] = v
    with open(d / 'umbilicus.json', 'w') as f:
        json.dump(dict(control_points=[dict(x=CX, y=CY, z=0), dict(x=CX, y=CY, z=NZ)]), f)
    return d


def write_pcl(path, chains):
    """chains: list of lists of (theta, winding_on_sheet, wind_a)."""
    cols, pid = {}, 0
    for cid, chain in enumerate(chains):
        pts = {}
        for k, (theta, w, wa) in enumerate(chain):
            r = spiral_radius(w, theta)
            pts[str(pid)] = dict(p=[CX + r * math.cos(theta), CY + r * math.sin(theta), NZ / 2], wind_a=wa, creation_time=k)
            pid += 1
        cols[str(cid)] = dict(name=f'col{cid}', points=pts, metadata={})
    with open(path, 'w') as f:
        json.dump({'vc_pointcollections_json_version': '1', 'collections': cols}, f)
    return str(path)


def run(scene, chains, name, pred='pred.zarr'):
    pcl = write_pcl(scene / name, chains)
    umb = json_umbilicus_z_to_yx(str(scene / 'umbilicus.json'))
    steps, meta = cwa.check(cwa.load_chains([pcl]), umb, [cwa.Prediction(str(scene / pred))])
    return steps, meta


def radial_chain(theta, ws, labels=None):
    return [(theta, w, float(l)) for w, l in zip(ws, labels if labels is not None else range(1, len(ws) + 1))]


def test_clean_spiral_has_no_flags(scene):
    steps, meta = run(scene, [radial_chain(t, [1, 2, 3, 4, 5]) for t in (0.3, 1.2, 2.0)], 'clean.json')
    assert meta[0]['frame_ok'] and meta[0]['steps_snapped_frac'] == 1.0
    assert len(steps) == 12
    assert all(s['cross'][0]['n'] == 1 for s in steps)
    assert not any(s['over'] or s['under'] for s in steps)


def test_skipped_sheet_is_flagged_with_suggested_delta(scene):
    # points on sheets 1, 2, 4, 5 but labelled 1, 2, 3, 4: the 2 -> 4 step skips a sheet
    bad = radial_chain(0.8, [1, 2, 4, 5], [1, 2, 3, 4])
    steps, _ = run(scene, [radial_chain(0.3, [1, 2, 3, 4, 5]), bad], 'skip.json')
    flagged = [(s['collection'], s['step']) for s in steps if s['over_strict']]
    assert flagged == [('col1', 1)]
    s = [s for s in steps if s['collection'] == 'col1' and s['step'] == 1][0]
    assert s['cross'][0]['n'] == 2 and s['suggested_delta'] == 2


def test_chain_on_atan2_branch_cut_is_clean(scene):
    # theta = pi is the atan2 branch cut; points alternate just above / below it
    seam = [(math.pi + (1e-3 if k % 2 else -1e-3), w, float(w)) for k, w in enumerate([1, 2, 3, 4])]
    steps, _ = run(scene, [seam, radial_chain(math.pi - 0.2, [1, 2, 3, 4])], 'seam.json')
    assert not any(s['over'] or s['under'] for s in steps)
    assert all(s['cross'][0]['n'] == 1 for s in steps)
    assert not any(s['global_suspect'] for s in steps)


@pytest.mark.parametrize('pred', ['pred01.zarr', 'predf.zarr'])
def test_dtype_aware_threshold(scene, pred):
    steps, meta = run(scene, [radial_chain(0.3, [1, 2, 3, 4])], 'dtype.json', pred)
    assert meta[0]['steps_snapped_frac'] == 1.0 and all(s['cross'][0]['n'] == 1 for s in steps)


def test_off_sheet_endpoint_counts_half_sheets(scene):
    # second point sits mid-gap between sheets 2 and 3: index 1.5 -> steps count 1.5 and 1.5, never flagged
    chain = [(0.3, 1, 1.0), (0.3, 2.5, 2.0), (0.3, 4, 3.0)]
    steps, _ = run(scene, [chain], 'off.json')
    assert [s['cross'][0]['n'] for s in steps] == [1.5, 1.5]
    assert not any(s['over'] or s['under'] or s['over_strict'] for s in steps)
    # labelled one winding apart but 2.5 sheets away: over-count (not strict, one endpoint off-sheet)
    steps, _ = run(scene, [[(0.3, 1, 1.0), (0.3, 3.5, 2.0)]], 'off2.json')
    assert steps[0]['cross'][0]['n'] == 2.5 and steps[0]['over'] and not steps[0]['over_strict']


def test_frame_check_rejects_negative_coordinates(scene):
    _, meta = run(scene, [[(0.3, 1, 1.0), (0.3, 2, 2.0)], [(math.pi, 7.5, 1.0), (math.pi, 8.5, 2.0)]], 'neg.json')
    assert meta[0]['frame_ok'] is False  # r ~ 122 > CX: x < 0


def test_point_ids_and_two_sided_local_null(scene):
    steps, _ = run(scene, [radial_chain(0.3 + 0.01 * k, [1, 2, 3, 4]) for k in range(4)], 'ids.json')
    assert all(s['point1'] == s['point0'] + 1 for s in steps)
    assert not any(s['local_suspect'] for s in steps)
    s = dict(p0=np.array([CX + 30.0, CY, NZ / 2]), p1=np.array([CX + 31.0, CY, NZ / 2]), delta=1, chain=-1)
    refs = [dict(p0=np.array([CX + 30.0, CY + 0.1 * k, NZ / 2]), p1=np.array([CX + 42.0, CY + 0.1 * k, NZ / 2]), delta=1, chain=k)
            for k in range(6)]
    cwa.radial_checks([s] + refs, json_umbilicus_z_to_yx(str(scene / 'umbilicus.json')))
    assert s['local_suspect'] and not any(r['local_suspect'] for r in refs)  # 1 vox vs 12 vox local: too small


def test_warns_on_non_binary_integer_volume(scene, capsys):
    vol = (np.arange(8 * 64 * 64) % 7).astype('uint8').reshape(8, 64, 64)
    a = zarr.create_array(str(scene / 'multi.zarr'), shape=vol.shape, chunks=(8, 64, 64), dtype='uint8')
    a[:] = vol
    p = cwa.Prediction(str(scene / 'multi.zarr'))
    p.prefetch(np.array([[0, 0, 0]]))
    assert 'WARNING' in capsys.readouterr().err
