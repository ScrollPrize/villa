from __future__ import annotations

from pathlib import Path
import os
from typing import Any

import numpy as np


def _resolve_umbilicus(checkpoint: Path, explicit: str | Path | None) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"umbilicus does not exist: {path}")
        return path
    for parent in (checkpoint.parent, *checkpoint.parents):
        candidate = parent / "umbilicus.json"
        if candidate.is_file():
            return candidate
    dataset = os.environ.get("SPIRAL_DATASET")
    candidates = [] if not dataset else [Path(dataset) / "umbilicus.json"]
    candidates.extend(
        [
            Path.home() / "Documents/volpkgs/s1_2um.volpkg/umbilicus.json",
            Path.home() / "Documents/volpkgs/s1_ds2.volpkg/umbilicus.json",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "checkpoint does not embed its umbilicus; pass umbilicus=..."
    )


class SpiralThetaProvider:
    """Load a Spiral checkpoint and return wrapped theta for native batches."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        umbilicus: str | Path | None = None,
        device: str = "cuda",
        batch_size: int = 1_048_576,
    ) -> None:
        import torch

        from checkpoint_io import load_checkpoint_cpu
        from config import Config
        from transforms import SpiralAndTransform
        from umbilicus import json_umbilicus_z_to_yx

        checkpoint_path = Path(checkpoint).expanduser().resolve()
        checkpoint_data: dict[str, Any] = load_checkpoint_cpu(checkpoint_path)
        config = Config().as_dict()
        config.update(checkpoint_data["cfg"])
        z_begin = int(checkpoint_data["z_begin"])
        z_end = int(checkpoint_data["z_end"])
        if z_end <= z_begin:
            raise ValueError("checkpoint has an invalid z range")
        umbilicus_path = _resolve_umbilicus(checkpoint_path, umbilicus)
        checkpoint_stat = checkpoint_path.stat()
        umbilicus_stat = umbilicus_path.stat()
        self.cache_key = (
            f"spiral-checkpoint-v1:{checkpoint_path}:"
            f"{checkpoint_stat.st_size}:{checkpoint_stat.st_mtime_ns}|"
            f"{umbilicus_path}:{umbilicus_stat.st_size}:{umbilicus_stat.st_mtime_ns}"
        )
        z_to_yx = json_umbilicus_z_to_yx(umbilicus_path)
        self._z_to_yx = z_to_yx
        all_z = np.arange(z_begin, z_end, dtype=np.float32)
        umbilicus_zyx = np.concatenate(
            (all_z[:, None], z_to_yx(all_z).astype(np.float32)), axis=-1
        )

        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        radius = config["model_flow_bounds_radius"]
        margin = config["model_flow_bounds_z_margin"]
        model = SpiralAndTransform(
            flow_integration_steps=config["model_num_flow_integration_steps"],
            flow_integration_solver=config["model_flow_integration_solver"],
            umbilicus_zyx=torch.from_numpy(umbilicus_zyx).to(self.device),
            flow_min_corner_zyx=torch.tensor(
                [z_begin - margin, -radius, -radius],
                dtype=torch.int64,
                device=self.device,
            ),
            flow_max_corner_zyx=torch.tensor(
                [z_end + margin, radius, radius],
                dtype=torch.int64,
                device=self.device,
            ),
            config=config,
            spiral_outward_sense=(
                checkpoint_data.get("spiral_outward_sense") or "CW"
            ),
        ).to(self.device)
        model.load_state_dict(checkpoint_data["spiral_and_transform"])
        model.eval()
        self._torch = torch
        self._transform = model.get_slice_to_spiral_transform()
        self._z_begin = z_begin
        self._z_end = z_end
        self.z_begin = z_begin
        self.z_end = z_end

    def __call__(self, zyx: np.ndarray) -> np.ndarray:
        from sample_spiral import get_theta

        array = np.asarray(zyx, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("theta input must have shape (N, 3) in z/y/x order")
        if array.size and (
            np.nanmin(array[:, 0]) < self._z_begin
            or np.nanmax(array[:, 0]) >= self._z_end
        ):
            raise ValueError(
                f"theta input extends outside checkpoint z range "
                f"[{self._z_begin}, {self._z_end})"
            )
        outputs: list[np.ndarray] = []
        with self._torch.inference_mode():
            for begin in range(0, len(array), self.batch_size):
                points = self._torch.from_numpy(
                    np.ascontiguousarray(array[begin : begin + self.batch_size])
                ).to(self.device)
                spiral = self._transform(points)
                theta, _ = get_theta(spiral[..., 1:])
                outputs.append(theta.float().cpu().numpy())
        return (
            np.concatenate(outputs).astype(np.float32, copy=False)
            if outputs
            else np.empty(0, dtype=np.float32)
        )

    def geometric_theta(self, zyx: np.ndarray) -> np.ndarray:
        """Return raw polar angle around the configured umbilicus."""
        array = np.asarray(zyx, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("theta input must have shape (N, 3) in z/y/x order")
        if not len(array):
            return np.empty(0, dtype=np.float32)
        center_yx = np.asarray(self._z_to_yx(array[:, 0]), dtype=np.float32)
        relative_yx = array[:, 1:] - center_yx
        return np.mod(
            np.arctan2(relative_yx[:, 0], relative_yx[:, 1]),
            2.0 * np.pi,
        ).astype(np.float32, copy=False)
