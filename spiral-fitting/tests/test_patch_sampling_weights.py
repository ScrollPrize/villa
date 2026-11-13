"""Name/core patch draw weights and the patch exclusion list."""
from types import SimpleNamespace

import pytest

from fit_spiral import patch_sampling_factors


def test_longest_name_prefix_wins_and_core_patches_are_scaled():
    patches = [SimpleNamespace(name=name, _umbilicus_distance=distance)
               for name, distance in [('strip-net-a', 500.0), ('strip-b', 500.0),
                                      ('strip-c', 100.0), ('band-d', 100.0)]]
    config = {'patch_sampling_name_weights': {'strip-': 0.3, 'strip-net-': 1.0},
              'patch_sampling_core_radius': 230.0, 'patch_sampling_core_weight': 0.2}
    assert patch_sampling_factors(patches, config).tolist() == pytest.approx(
        [1.0, 0.3, 0.06, 0.2])


@pytest.mark.usefixtures('cpu')
def test_excluded_patches_are_left_out_of_the_fit(store_path):
    pytest.importorskip('vc_object_store')
    import object_store_inputs as osi
    from object_store_fixture import fit_config, names
    near = names(store_path)['near']
    kept = osi.load_fit_inputs(fit_config(store_path), log=lambda _: None)
    inputs = osi.load_fit_inputs(fit_config(
        store_path, object_store_patch_exclude_uuids=[near]), log=lambda _: None)
    assert near in kept.patches and near not in inputs.patches
    assert inputs.counts['patches excluded'] == 1
    assert inputs.counts['pcl attachments'] < kept.counts['pcl attachments']


def test_tracks_with_both_endpoints_inside_the_core_are_dropped():
    import numpy as np
    from tracks import filter_tracks_inside_umbilicus_radius
    centre = lambda z: np.zeros((len(z), 2))  # noqa: E731
    tracks = [np.array([[0, 10, 0], [1, 20, 0]], np.float32),    # both inside 50
              np.array([[0, 10, 0], [1, 80, 0]], np.float32),    # one outside
              np.array([[0, 90, 0], [1, 0, 95]], np.float32)]    # both outside
    kept, families = filter_tracks_inside_umbilicus_radius(tracks, centre, 50.0, ['a', 'b', 'c'])
    assert families == ['b', 'c'] and len(kept) == 2
