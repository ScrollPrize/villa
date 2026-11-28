"""The trainer's validation metrics accept the 2D patches of the napari trainer's 2D workflow."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vc_reader")

from vesuvius.models.evaluation.voi import compute_voi
from vesuvius.models.training.train import BaseTrainer


def _batch():
    torch.manual_seed(0)
    gt = torch.zeros(2, 1, 32, 32)
    gt[:, :, 4:12, 4:28] = 1
    gt[:, :, 20:28, 4:14] = 1
    gt[:, :, :, 30:] = 2  # ignore label
    pred = torch.randn(2, 2, 32, 32)
    pred[:, 1] += 4 * (gt[:, 0] == 1)
    return pred, gt


def test_validation_metrics_accept_2d_patches():
    pred, gt = _batch()
    trainer = SimpleNamespace(mgr=SimpleNamespace(targets={"ink": {"ignore_label": 2}}))
    metrics = BaseTrainer._initialize_evaluation_metrics(trainer)["ink"]

    for metric in metrics:
        metric.update(pred=pred, gt=gt, mask=None)

    assert {type(m).__name__ for m in metrics} >= {"VOIMetric"}


def test_2d_voi_equals_voi_of_the_same_patch_as_a_one_slice_volume():
    pred, gt = _batch()

    voi_2d = compute_voi(pred, gt, ignore_index=2)
    voi_3d = compute_voi(pred[:, :, None], gt[:, :, None], ignore_index=2)

    assert voi_2d == pytest.approx(voi_3d)
    assert voi_2d["voi_total"] > 0
