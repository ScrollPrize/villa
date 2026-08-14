"""Napari visualization helpers for winding-model dataset samples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from vesuvius.neural_tracing.winding_models.winding_model_dataset import (
    WindingModelDataset,
)


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _slab_affine_zyx(sample: dict[str, torch.Tensor]) -> np.ndarray:
    """Map the [H, W, L] slab's local (i, j, k) indices into volume ZYX."""
    spacing = float(sample["spacing"])
    affine = np.eye(4, dtype=np.float32)
    affine[:3, 0] = spacing * _as_numpy(sample["slab_axis_a_zyx"])
    affine[:3, 1] = spacing * _as_numpy(sample["slab_axis_b_zyx"])
    affine[:3, 2] = spacing * _as_numpy(sample["ray_direction_zyx"])
    affine[:3, 3] = _as_numpy(sample["slab_origin_zyx"])
    return affine


def _column_crossing_points(
    sample: dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    """World-ZYX crossing points of every supervised column, with indices."""
    counts = _as_numpy(sample["num_crossings"])
    crossing_t = _as_numpy(sample["crossing_t"])
    crossing_indices = _as_numpy(sample["crossing_indices"])
    stride = int(sample["column_stride"])
    spacing = float(sample["spacing"])
    origin = _as_numpy(sample["slab_origin_zyx"]).astype(np.float64)
    axis_a = _as_numpy(sample["slab_axis_a_zyx"]).astype(np.float64)
    axis_b = _as_numpy(sample["slab_axis_b_zyx"]).astype(np.float64)
    direction = _as_numpy(sample["ray_direction_zyx"]).astype(np.float64)

    points, indices = [], []
    for row, col in np.argwhere(counts > 0):
        ts = crossing_t[row, col, : counts[row, col], None].astype(np.float64)
        base = origin + spacing * stride * (row * axis_a + col * axis_b)
        points.append(base[None] + ts * direction[None])
        indices.append(crossing_indices[row, col, : counts[row, col]])
    if not points:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    return np.concatenate(points), np.concatenate(indices)


def display_napari_sample(viewer, sample: dict[str, torch.Tensor]) -> None:
    """Replace the viewer layers with one dataset sample in volume ZYX space."""
    viewer.layers.clear()
    slab_image = _as_numpy(sample["slab_image"])
    affine = _slab_affine_zyx(sample)
    viewer.add_image(
        slab_image,
        name="slab",
        affine=affine,
        colormap="gray",
        blending="translucent",
        rendering="attenuated_mip",
        contrast_limits=(0, 255),
    )
    # The transverse slice containing the central ray, for a plane view
    # matching the training visualization.
    viewer.add_image(
        slab_image[slab_image.shape[0] // 2][None],
        name="center_slice",
        affine=affine
        @ np.array(
            [
                [1, 0, 0, slab_image.shape[0] // 2],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        ),
        colormap="gray",
        blending="translucent",
        rendering="translucent",
        contrast_limits=(0, 255),
    )

    crossings, winding = _column_crossing_points(sample)
    if len(crossings):
        viewer.add_points(
            crossings,
            name="column_crossings",
            size=2,
            features={"winding": winding.astype(np.float32)},
            face_color="winding",
            face_colormap="turbo",
            blending="translucent_no_depth",
        )

    origin = _as_numpy(sample["ray_origin_zyx"])
    direction = _as_numpy(sample["ray_direction_zyx"])
    ray_end = origin + float(sample["ray_extent"]) * direction
    viewer.add_shapes(
        [np.stack((origin, ray_end))],
        shape_type="path",
        name="ray",
        edge_color="cyan",
        edge_width=2,
        face_color="transparent",
    )
    viewer.reset_view()


def run_napari_inspector(config_path: Path) -> None:
    """Launch the interactive inspector for winding-model dataset samples."""
    try:
        import napari
        from napari.qt.threading import thread_worker
        from qtpy.QtCore import QTimer
        from qtpy.QtWidgets import QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget
    except ImportError as exc:
        raise RuntimeError(
            "The winding dataset viewer requires the vesuvius GUI dependencies"
        ) from exc

    # Initialize Qt/Vispy before loading meshes. Some native geometry/image
    # dependencies initialize worker runtimes while the dataset is constructed;
    # doing that before Vispy creates its first Qt canvas can crash in the Qt
    # show event on Linux.
    viewer = napari.Viewer(ndisplay=3, title="Winding model dataset inspector")
    controls = QWidget()
    layout = QVBoxLayout(controls)
    ray_length_label = QLabel("Ray length (samples)")
    ray_length_spin = QSpinBox()
    ray_length_spin.setRange(2, 1_000_000)
    ray_length_spin.setSingleStep(128)
    transverse_label = QLabel("Transverse size (samples)")
    transverse_spin = QSpinBox()
    transverse_spin.setRange(8, 65_536)
    transverse_spin.setSingleStep(8)
    next_button = QPushButton("Next")
    status = QLabel("Loading dataset…")
    layout.addWidget(ray_length_label)
    layout.addWidget(ray_length_spin)
    layout.addWidget(transverse_label)
    layout.addWidget(transverse_spin)
    layout.addWidget(next_button)
    layout.addWidget(status)
    viewer.window.add_dock_widget(controls, area="right", name="Dataset")

    with config_path.open() as config_file:
        dataset = WindingModelDataset(json.load(config_file))
    ray_length_spin.setValue(dataset.ray_length)
    transverse_spin.setValue(dataset.transverse_size)
    status.setText("Loading first sample…")

    sample_iterator = iter(dataset)
    sample_number = 0
    active_worker = None

    def take_next_sample() -> dict[str, torch.Tensor]:
        nonlocal sample_iterator
        try:
            return next(sample_iterator)
        except StopIteration:
            sample_iterator = iter(dataset)
            return next(sample_iterator)

    def show_sample(sample: dict[str, torch.Tensor]) -> None:
        nonlocal sample_number
        sample_number += 1
        display_napari_sample(viewer, sample)
        volume_idx = int(sample["volume_idx"])
        supervised = int((_as_numpy(sample["num_crossings"]) > 0).sum())
        total = sample["num_crossings"].numel()
        status.setText(
            f"Sample {sample_number} · volume {volume_idx} · {supervised}/{total} "
            f"columns · {int(sample['transverse_size'])}²×{int(sample['ray_length'])}"
        )

    def show_error(error: BaseException) -> None:
        status.setText(f"Sampling failed: {error}")

    def sampling_finished() -> None:
        nonlocal active_worker
        next_button.setEnabled(True)
        ray_length_spin.setEnabled(True)
        transverse_spin.setEnabled(True)
        active_worker = None

    def request_next_sample(_checked: bool = False) -> None:
        nonlocal active_worker
        if active_worker is not None:
            return
        dataset.set_slab_dimensions(
            ray_length=ray_length_spin.value(),
            transverse_size=transverse_spin.value(),
        )
        next_button.setEnabled(False)
        ray_length_spin.setEnabled(False)
        transverse_spin.setEnabled(False)
        status.setText("Sampling…")
        active_worker = thread_worker(take_next_sample)()
        active_worker.returned.connect(show_sample)
        active_worker.errored.connect(show_error)
        active_worker.finished.connect(sampling_finished)
        active_worker.start()

    next_button.clicked.connect(request_next_sample)
    QTimer.singleShot(0, next_button.click)
    napari.run()
