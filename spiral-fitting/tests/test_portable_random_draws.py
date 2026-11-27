"""The native samplers draw the same values from a seed on every platform.

spiral_sampling and track_crossings draw through cpp/portable_random.hpp
rather than std::uniform_int_distribution and std::generate_canonical, whose
results differ between standard libraries. The expected values are the draws
of libstdc++ in GCC 11 to 15 on 64-bit Linux, from before the helpers and
after; before them, macOS (libc++) drew other cells, offsets and partners from
these seeds, and GCC 16 other offsets.
"""
import numpy as np

from vc_spiral import spiral_sampling, track_crossings


def test_patch_points_are_the_same_on_every_platform():
    mask = np.ones((31, 37), dtype=bool)
    mask[8:14, 10:22] = False
    atlas = spiral_sampling.PatchSamplingAtlas([mask])
    result = atlas.sample_patch_points(np.zeros(2, dtype=np.int64), 5, 1234)
    np.testing.assert_array_equal(
        result['node_ordinals'],
        [[559, 342, 904, 752, 654], [299, 812, 579, 142, 857]])
    np.testing.assert_array_equal(result['ijs'], np.array([
        [[17.606197, 2.5610185], [9.96053, 33.851498],
         [26.493006, 14.338529], [22.132221, 10.399055],
         [19.067186, 23.622856]],
        [[8.883776, 3.8865564], [23.283567, 33.51807],
         [17.810873, 22.981533], [3.3138986, 31.165152],
         [25.591251, 4.1012487]],
    ], dtype=np.float32))


def test_crossing_partners_are_the_same_on_every_platform():
    # Track 0 crosses tracks 1 to 9 once each; every crossing is recorded on
    # both tracks, as the sampler requires.
    index = track_crossings.prepare_crossing_index(
        np.array([0, 9, *range(10, 19)], dtype=np.int64),
        np.array([*range(1, 10), *[0] * 9], dtype=np.int32),
        np.array([*range(9), *[0] * 9], dtype=np.int32),
        np.array([*[0] * 9, *range(9)], dtype=np.int32),
        np.array([9, *[1] * 9], dtype=np.int32))
    result = track_crossings.sample_crossing_partners(
        index, np.zeros(3, dtype=np.int32), maximum=4, seed=1234)
    np.testing.assert_array_equal(
        result['partners'], [[8, 4, 5, 2], [2, 3, 7, 6], [5, 7, 2, 4]])
