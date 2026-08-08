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


def _plane_affine_zyx(sample: dict[str, torch.Tensor], plane_idx: int) -> np.ndarray:
    """Map a [1, H, W] plane image's local ZYX indices into volume ZYX."""
    origin = _as_numpy(sample["plane_origins_zyx"])[plane_idx]
    x_step = _as_numpy(sample["plane_x_steps_zyx"])[plane_idx]
    y_step = _as_numpy(sample["plane_y_steps_zyx"])[plane_idx]
    normal = np.cross(y_step, x_step)
    normal *= np.sqrt(np.linalg.norm(x_step) * np.linalg.norm(y_step)) / np.linalg.norm(
        normal
    )

    affine = np.eye(4, dtype=np.float32)
    affine[:3, 0] = normal
    affine[:3, 1] = y_step
    affine[:3, 2] = x_step
    affine[:3, 3] = origin
    return affine


def display_napari_sample(viewer, sample: dict[str, torch.Tensor]) -> None:
    """Replace the viewer layers with one dataset sample in volume ZYX space."""
    viewer.layers.clear()
    plane_images = _as_numpy(sample["plane_images"])
    winding_indices = _as_numpy(sample["winding_indices"])
    for plane_idx in range(len(plane_images)):
        viewer.add_image(
            plane_images[plane_idx][None],
            name=f"ray_plane_{plane_idx}",
            affine=_plane_affine_zyx(sample, plane_idx),
            colormap="gray",
            blending="translucent",
            rendering="translucent",
            contrast_limits=(0, 255),
        )

    crossings = _as_numpy(sample["crossing_zyx"])
    viewer.add_points(
        crossings,
        name="crossings",
        size=5,
        face_color="magenta",
        blending="translucent_no_depth",
        features={"winding": winding_indices},
        text={
            "string": "{winding}",
            "color": "yellow",
            "size": 18,
            "blending": "translucent_no_depth",
        },
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

    invalid = ~_as_numpy(sample["winding_valid"]).astype(bool)
    if invalid.any():
        spacing = float(sample["ray_extent"]) / (int(sample["ray_length"]) - 1)
        edges = np.diff(invalid.astype(np.int8), prepend=0, append=0)
        spans = [
            np.stack(
                (
                    origin + start * spacing * direction,
                    origin + end * spacing * direction,
                )
            )
            for start, end in zip(
                np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1
            )
        ]
        viewer.add_shapes(
            spans,
            shape_type="path",
            name="unlabeled",
            edge_color="red",
            edge_width=3,
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
    plane_height_label = QLabel("Plane height (samples)")
    plane_height_spin = QSpinBox()
    plane_height_spin.setRange(2, 65_536)
    plane_height_spin.setSingleStep(32)
    next_button = QPushButton("Next")
    status = QLabel("Loading dataset…")
    layout.addWidget(ray_length_label)
    layout.addWidget(ray_length_spin)
    layout.addWidget(plane_height_label)
    layout.addWidget(plane_height_spin)
    layout.addWidget(next_button)
    layout.addWidget(status)
    viewer.window.add_dock_widget(controls, area="right", name="Dataset")

    with config_path.open() as config_file:
        dataset = WindingModelDataset(json.load(config_file))
    ray_length_spin.setValue(dataset.ray_length)
    plane_height_spin.setValue(dataset.plane_height)
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
        crossing_count = len(sample["crossing_t"])
        status.setText(
            f"Sample {sample_number} · volume {volume_idx} · {crossing_count} "
            f"crossings · {int(sample['plane_height'])}×{int(sample['ray_length'])}"
        )

    def show_error(error: BaseException) -> None:
        status.setText(f"Sampling failed: {error}")

    def sampling_finished() -> None:
        nonlocal active_worker
        next_button.setEnabled(True)
        ray_length_spin.setEnabled(True)
        plane_height_spin.setEnabled(True)
        active_worker = None

    def request_next_sample(_checked: bool = False) -> None:
        nonlocal active_worker
        if active_worker is not None:
            return
        dataset.set_plane_dimensions(
            ray_length=ray_length_spin.value(),
            plane_height=plane_height_spin.value(),
        )
        next_button.setEnabled(False)
        ray_length_spin.setEnabled(False)
        plane_height_spin.setEnabled(False)
        status.setText("Sampling…")
        active_worker = thread_worker(take_next_sample)()
        active_worker.returned.connect(show_sample)
        active_worker.errored.connect(show_error)
        active_worker.finished.connect(sampling_finished)
        active_worker.start()

    next_button.clicked.connect(request_next_sample)
    QTimer.singleShot(0, next_button.click)
    napari.run()
