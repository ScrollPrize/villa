import numpy as np

from vesuvius.utils.plotting import _choose_preview_slice_index


def test_choose_preview_slice_prefers_ground_truth_foreground():
    input_volume = np.zeros((4, 8, 8), dtype=np.float32)
    targets = {"surface": np.zeros((1, 4, 8, 8), dtype=np.float32)}
    preds = {"surface": np.zeros((2, 4, 8, 8), dtype=np.float32)}

    targets["surface"][0, 2, 2:6, 2:6] = 1.0
    preds["surface"][1, 1, 1:7, 1:7] = 0.75

    preview_idx = _choose_preview_slice_index(
        input_volume,
        targets,
        preds,
        is_2d_run=False,
    )

    assert preview_idx == 2


def test_choose_preview_slice_falls_back_to_prediction_mass():
    input_volume = np.zeros((5, 6, 6), dtype=np.float32)
    targets = {"surface": np.zeros((1, 5, 6, 6), dtype=np.float32)}
    preds = {"surface": np.zeros((2, 5, 6, 6), dtype=np.float32)}

    preds["surface"][1, 4, :, :] = 0.5

    preview_idx = _choose_preview_slice_index(
        input_volume,
        targets,
        preds,
        is_2d_run=False,
    )

    assert preview_idx == 4
