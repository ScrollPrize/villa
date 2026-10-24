import numpy as np

from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_kollesis import (
    Maps, conditional_abs, conditional_discrepancy, length_maps, ratio, edge_maps, write_plots)


def test_cumulative_expectation_matches_bruteforce():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(13, 9)); w = rng.uniform(size=x.shape); w /= w.sum(1)[:, None]
    t = rng.normal(size=(13, 7)); m = rng.uniform(0, 3, size=13)
    expected = (abs(x[:, None, :]-t[:, :, None])*w[:, None, :]).sum(2)
    np.testing.assert_allclose(conditional_abs(x, w, t), expected, atol=1e-14)
    expected = (abs(abs(x[:, None, :]-t[:, :, None])-m[:, None, None])*w[:, None, :]).sum(2)
    np.testing.assert_allclose(conditional_discrepancy(x, w, t, m), expected, atol=1e-14)
    # Equal coordinates, zero mass entries, unnormalized weights and zero distance.
    np.testing.assert_allclose(conditional_discrepancy(np.zeros((1, 3)), np.array([[0., 2., 0.]]), np.array([[0., 4.]]), np.array([0.])), [[0., 8.]])


def test_length_mass_entropy_rejection_and_reversal():
    d = dict(lengths=np.array([2000., 1000.]), family=np.array([0, 1]), node_ds=np.array([[0, 1000, 2000], [0, -500, -1000]]))
    u = np.array([[100., 3100.], [100., 3100.]])
    v = np.full((2, 2), 2100.)
    p = np.array([[.4, .4, .2], [.75, 0., .25]])
    maps = Maps((0, 0, 6000, 4000), 1000)
    length_maps(maps, d, u, v, p, 250)
    np.testing.assert_allclose(maps.values['length_support'].sum(), 3000)
    np.testing.assert_allclose(maps.values['rejection_sum'].sum(), 650)
    np.testing.assert_allclose(maps.values['entropy_sum'].sum(), 1600)
    np.testing.assert_allclose(maps.values['retained_length'].sum(), 2350)
    assert np.isnan(ratio(np.array([0.]), np.array([0.]))[0])


def test_profiles_use_ratio_of_sums():
    numerator = np.array([[1., 0.], [0., 0.]])
    denominator = np.array([[1., 0.], [9., 0.]])
    result = ratio(numerator.sum(0), denominator.sum(0))
    assert result[0] == .1 and np.isnan(result[1])


def test_actual_edge_deposition_matches_all_pairs():
    widths = np.array([1000., 2000.]); boundaries = np.array([0., 1000., 3000.])
    u = np.array([[100., 1500.], [350., 1750.], [500., 2100.]])
    v = np.array([[500., 550.], [800., 850.], [1000., 1200.]])
    p = np.array([[.5, .3, .2], [.2, .7, .1], [.4, .3, .3]])
    d = dict(i=np.array([0, 1]), j=np.array([1, 2]), family=np.array([0, 0, 1]),
        ds_a=np.array([150., -50.]), ds_b=np.array([-100., 50.]), phase_delta=np.array([.2, -.1]),
        distance=np.array([.4, .7]), parallel=np.array([True, False]), source_id=np.array([10, 11, 12]))
    geom = np.array([[12., 2., 1.], [25., 5., 1.]])
    maps = Maps((0., 0., 4000., 2000.), 1000.)
    edge_maps(maps, d, u, v, p, widths, boundaries, geom, {'event_batch': 1, 'event_workers': 2})
    serial = Maps((0., 0., 4000., 2000.), 1000.)
    edge_maps(serial, d, u, v, p, widths, boundaries, geom, {'event_batch': 1, 'event_workers': 1})
    for key in maps.values:
        np.testing.assert_array_equal(maps.values[key], serial.values[key])
    expected = Maps((0., 0., 4000., 2000.), 1000.)
    def q(x):
        k = np.clip(np.searchsorted(boundaries, x, side='right')-1, 0, 1)
        return k+(x-boundaries[k])/widths[k]
    for e, (i, j) in enumerate(zip(d['i'], d['j'])):
        for a in range(2):
            for b in range(2):
                x = u[i, a]+(d['family'][i] == 0)*d['ds_a'][e]
                y = v[i, a]+(d['family'][i] == 1)*d['ds_a'][e]
                xx = u[j, b]+(d['family'][j] == 0)*d['ds_b'][e]
                yy = v[j, b]+(d['family'][j] == 1)*d['ds_b'][e]
                delta = q(xx)-q(x)-d['phase_delta'][e]+.5*(d['family'][j]-d['family'][i])
                weight = p[i, a]*p[j, b]/2
                for px, py in ((x, y), (xx, yy)):
                    expected.add('event_support', np.array([px]), np.array([py]), weight)
                    expected.add('discrepancy_sum', np.array([px]), np.array([py]), weight*abs(abs(delta)-d['distance'][e]))
    for name in expected.values:
        np.testing.assert_allclose(maps.values[name], expected.values[name], atol=1e-13)
    np.testing.assert_allclose(maps.values['crowding_support'].sum(), .8*.9)
    np.testing.assert_allclose(sum(maps.values[f'normal_{a}_{b}'] for a, b in zip([0, 8, 16, 32, 64], [8, 16, 32, 64, 128])), maps.values['crowding_support'])


def test_plot_and_numeric_export(tmp_path):
    import json
    from PIL import Image
    maps = Maps((0, 0, 3000, 1000), 1000)
    for name in ('length_support', 'retained_length', 'event_support', 'crowding_support', 'parallel_support', 'perpendicular_support'):
        maps.values[name] = np.full(maps.shape, 2000.)
    maps.values['rejection_sum'] = np.full(maps.shape, 500.)
    maps.values['entropy_sum'] = np.full(maps.shape, 1000.)
    maps.values['discrepancy_sum'] = np.full(maps.shape, 100.)
    maps.values['normal_0_8'] = np.full(maps.shape, 1000.)
    stats = write_plots(tmp_path, maps, {'heatmap_panels': 2}, 'Synthetic')
    assert stats['rejection']['global_mean'] == 25.
    assert (tmp_path/'profiles.svg').exists()
    assert (tmp_path/'profiles.csv').exists()
    with Image.open(tmp_path/'profiles.tif') as img:
        assert img.tag_v2[259] == 5  # LZW
    assert json.loads((tmp_path/'metrics.json').read_text())['uncertainty']['global_mean'] == 50.
    with np.load(tmp_path/'maps.npz') as data:
        np.testing.assert_allclose(data['rejection'], 25.)
