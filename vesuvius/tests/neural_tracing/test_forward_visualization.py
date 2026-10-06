"""CPU checks for visualization of existing loss forwards, without replay."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from tests.neural_tracing.test_fiber_trace_3d import (
    FiberTrace3DModelConfig,
    FiberTrace3DNet,
    _load_materialized_batch,
    _loader_with_mapping,
)
from vesuvius.neural_tracing.fiber_trace_3d import train
from vesuvius.neural_tracing.fiber_trace_3d.annotations import AnnotationConfig
from vesuvius.neural_tracing.fiber_trace_3d.direction import encode_lasagna_direction_3x2


def test_width_modulates_only_direction_presence_views():
    direction = torch.tensor([[1., 0., 1.]]) / np.sqrt(2)
    encoded = encode_lasagna_direction_3x2(direction)[0]
    prediction = torch.cat((encoded, torch.tensor([.8]))).expand(1, 5, 7)
    kwargs = dict(normal_zyx=np.array([0.,0.,1.]), tangent_zyx=np.array([1.,0.,0.]))
    original = train._sample_sheet_presence_views_from_sampled_output(prediction, **kwargs)
    gated = train._sample_sheet_presence_views_from_sampled_output(prediction, **kwargs,
        width_values=torch.tensor([[80.,90.,100.,110.,120.]]), center_width=torch.tensor(100.))
    np.testing.assert_array_equal(gated[0], original[0])
    for actual, before in zip(gated[1:], original[1:]):
        np.testing.assert_allclose(actual, before * np.array([[0.,.5,1.,.5,0.]]))


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("width,gray", [(0., 0), (40., 80), (80., 160), (160., 255)])
def test_width_column_base_scale_fixed_grayscale_all_planes(stride, width, gray):
    batch = train._slice_batch(_batch(), 0, 1)
    batch = replace(batch, volume_spacing_base=torch.tensor([2.]))
    output = torch.zeros((1, 7, 8 // stride, 8 // stride, 8 // stride))
    widths = torch.full((1, 1, *output.shape[-3:]), width)
    old = train._make_train_sample_3d_sheet(batch, output)
    new = train._make_train_sample_3d_sheet(batch, output, width_input=widths)
    assert new.shape == (old.shape[0], old.shape[1] + 12, 3)
    np.testing.assert_array_equal(new[:, :old.shape[1]], old)
    for plane in range(5):
        panel = new[plane * 12:plane * 12 + 8, -8:]
        assert np.any(np.all(panel == gray, axis=-1))
    assert np.all(new[0, -8] == gray)


@pytest.mark.parametrize("normal", [0., 10.])
def test_width_observer_routing_softplus_bounded_and_legacy(monkeypatch, normal):
    batch = _batch()
    cfg = AnnotationConfig(normal_weight=normal, width_weight=.01, hv_weight=.01)
    output = torch.zeros((2, 7, 8, 8, 8), requires_grad=True)
    aux = torch.zeros((2, cfg.channels, 8, 8, 8), requires_grad=True)
    rendered, legacy = [], []
    def render(physical, prediction, *, width_input):
        assert not width_input.requires_grad
        torch.testing.assert_close(width_input, torch.full_like(width_input, np.log(2)))
        rendered.append(physical)
        return np.zeros((2, 3, 3), dtype=np.uint8)
    monkeypatch.setattr(train, "_make_train_sample_3d_sheet", render)
    capture = train._ForwardVisualization(1)
    observer = train._combine_prediction_observers(lambda b, o: legacy.append(o), capture)
    for _ in range(2):
        train._notify_prediction_observer(observer, batch, output, aux, cfg)
    assert len(rendered) == 1 and len(legacy) == 2
    assert legacy[0] is output


def _batch():
    return _load_materialized_batch(_loader_with_mapping({
        "augment_enabled": False, "batch_size": 2, "patch_shape_zyx": [8, 8, 8],
    }), 0)


class _Writer:
    def __init__(self):
        self.images = []

    def add_image(self, tag, image, step, *, dataformats):
        self.images.append((tag, image, step, dataformats))


@pytest.mark.parametrize("statistics", [False, True])
@pytest.mark.parametrize("conditioned,annotated", [(False, False), (True, False), (False, True)])
def test_forward_capture_reuses_predictions_preserves_gradients_and_mode(
    monkeypatch, statistics, conditioned, annotated,
):
    batch = _batch()
    torch.manual_seed(17)
    model = FiberTrace3DNet(FiberTrace3DModelConfig(
        conditioned_decoder_enabled=conditioned, conditioned_latent_channels=8,
        conditioned_decoder_hidden_channels=8, conditioned_decoder_layers=2,
        features_per_stage=(4, 8), strides=((1, 1, 1), (2, 2, 2)),
        normalization="group",
        annotations=AnnotationConfig(width_weight=.01 if annotated else 0.),
    ))
    oracle = copy.deepcopy(model)
    forwarded = []
    handle = model.register_forward_hook(lambda _model, _args, output: forwarded.append(output))
    rendered = []

    def render(physical, prediction, **kwargs):
        assert not torch.is_grad_enabled()
        assert not prediction.requires_grad
        index = len(rendered)
        torch.testing.assert_close(physical.volume, batch.volume[index:index + 1])
        # Slicing the real already-augmented input does not allocate a replay batch.
        assert physical.volume.untyped_storage().data_ptr() == batch.volume.untyped_storage().data_ptr()
        rendered.append(prediction.clone())
        if annotated:
            width = torch.nn.functional.softplus(forwarded[0]["annotations"][index:index + 1].detach().float())
            torch.testing.assert_close(kwargs["width_input"], width)
        return np.zeros((3, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(train, "_make_train_sample_3d_sheet", render)
    capture = train._ForwardVisualization(2)

    def forward_loss(network, observer=None):
        if statistics:
            stats = train._forward_loss_statistics(network, batch, prediction_observer=observer)
            return train._finalize_loss_statistics(stats, direction_weight=1.0, presence_weight=1.0)["total"]
        return train._forward_loss_tensors(network, batch, direction_weight=1.0,
            presence_weight=1.0, training_loss=True, prediction_observer=observer)["total"]

    expected = forward_loss(oracle)
    expected.backward()
    actual = forward_loss(model, capture)
    actual.backward()
    handle.remove()
    assert len(forwarded) == 1
    assert model.training
    assert len(rendered) == 2
    output = forwarded[0]
    if conditioned:
        output = torch.cat((output["zero_output"], output["random_output"]), dim=1)
    elif annotated:
        output = output["prediction"]
    for index, image_prediction in enumerate(rendered):
        torch.testing.assert_close(image_prediction, output[index:index + 1].detach().float())
    torch.testing.assert_close(actual, expected)
    for parameter, expected_parameter in zip(model.parameters(), oracle.parameters()):
        if expected_parameter.grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected_parameter.grad)


def test_capture_count_across_microbatches_and_empty_writer(monkeypatch):
    batch = _batch()
    calls = []

    def render(physical, output):
        calls.append(float(output[0, 0, 0, 0, 0]))
        return np.full((2, 3, 3), len(calls), dtype=np.uint8)

    monkeypatch.setattr(train, "_make_train_sample_3d_sheet", render)
    capture = train._ForwardVisualization(3)
    single = train._slice_batch(batch, 0, 1)
    for value in range(5):
        capture(single, torch.full((1, 7, 8, 8, 8), float(value), requires_grad=True))
    assert calls == [0.0, 1.0, 2.0]
    assert len(capture.sheets) == 3
    assert all(isinstance(sheet, torch.Tensor) and sheet.device.type == "cpu"
               and not sheet.requires_grad for sheet in capture.sheets)
    writer = _Writer()
    train._ForwardVisualization(0).write(writer, "train", 0)
    train._ForwardVisualization(4).write(writer, "test", 0)
    assert writer.images == []
    capture.write(writer, "train", 500)
    assert len(writer.images) == 1
    assert writer.images[0][0::2] == ("train", 500)
    assert writer.images[0][3] == "HWC"


def test_resume_cpu_sheets_retains_completed_capture_without_replay(monkeypatch):
    sheets = [np.full((2, 3, 3), 9, dtype=np.uint8)]
    restored = train._ForwardVisualization(1, copy.deepcopy(sheets))

    def forbidden(*_args):
        raise AssertionError("completed visualization must not be captured again")

    monkeypatch.setattr(train, "_make_train_sample_3d_sheet", forbidden)
    batch = _batch()
    restored(batch, torch.zeros((2, 7, 8, 8, 8)))
    writer = _Writer()
    restored.write(writer, "test", 0)
    np.testing.assert_array_equal(writer.images[0][1], sheets[0])
    assert len(restored.sheets) == 1
