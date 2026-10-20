"""Rendering for the DINO method; never participates in optimization."""
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class DinoMonitor:
    def __init__(self, method, directory, config):
        self.method = method
        self.monitor_dir = directory
        self.monitor_batch_size = max(5, int(config.get("monitor_batch_size", 5)))
        self.monitor_pca_sets = max(1, int(config.get("monitor_pca_sets", 4)))

    @staticmethod
    def _normalize_image(array: np.ndarray) -> np.ndarray:
        array = array.astype(np.float32)
        min_value = float(array.min())
        max_value = float(array.max())
        if max_value <= min_value:
            return np.zeros_like(array, dtype=np.uint8)
        scaled = (array - min_value) / (max_value - min_value)
        return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)

    @staticmethod
    def _center_slice(volume: torch.Tensor) -> np.ndarray:
        array = volume.detach().cpu().float()
        if array.ndim == 4:
            depth = array.shape[1] // 2
            return array[0, depth].numpy()
        if array.ndim == 3:
            return array[0].numpy()
        raise ValueError(f"unexpected tensor shape for visualization: {tuple(array.shape)}")

    def _patch_pca_slice(
        self,
        patch_tokens: torch.Tensor,
        sample_index: int,
        target_hw: tuple[int, int],
        input_spatial_shape: tuple[int, int, int],
    ) -> list[np.ndarray]:
        feature_shape = self.method._feature_map_shape(
            self.method.model_module.student.backbone,
            input_spatial_shape,
            view_kind="global",
        )
        feature_map = patch_tokens[sample_index].reshape(*feature_shape, patch_tokens.shape[-1])
        depth = feature_map.shape[0] // 2
        resized = []
        for result in self._pca_rgb(feature_map[depth]):
            image = F.interpolate(
                torch.from_numpy(result).permute(2, 0, 1).float()[None],
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
            resized.append(image[0].permute(1, 2, 0).numpy().astype(np.uint8))
        return resized

    def _pca_rgb(self, features: torch.Tensor, sets: int | None = None) -> list[np.ndarray]:
        """[H, W, C] features -> one [H, W, 3] uint8 image per set of three principal
        components (1-3, 4-6, ...), `sets` (default `monitor_pca_sets`) images in all."""
        h, w, c = features.shape
        flat = features.float().reshape(h * w, c)
        flat = flat - flat.mean(dim=0)
        n_sets = self.monitor_pca_sets if sets is None else int(sets)
        components = 3 * n_sets
        rank = min(components, h * w, c)
        _, _, V = torch.pca_lowrank(flat, q=rank)
        projected = (flat @ V[:, :rank]).reshape(h, w, rank).detach().cpu().numpy()
        channels = [self._normalize_image(projected[..., i]) if i < rank else np.zeros((h, w), dtype=np.uint8)
                    for i in range(components)]
        return [np.stack(channels[i:i + 3], axis=-1) for i in range(0, components, 3)]

    def _ink_rgb(self, features: torch.Tensor) -> np.ndarray:
        """Map [H, W, C] unit-norm ink embeddings to a 2-color [H, W, 3] RGB image
        using the running positive/negative ink prototypes in self.method.object_loss."""
        h, w, c = features.shape
        flat = F.normalize(features.float().reshape(h * w, c), dim=-1)
        pos_n = int(self.method.object_loss.ink_pos_count.item())
        neg_n = int(self.method.object_loss.ink_neg_count.item())
        if pos_n > 0 and neg_n > 0:
            pos_proto = F.normalize(self.method.object_loss.ink_pos_queue[:pos_n].float().mean(dim=0), dim=0)
            neg_proto = F.normalize(self.method.object_loss.ink_neg_queue[:neg_n].float().mean(dim=0), dim=0)
            temp = max(float(self.method.object_loss.ink_temperature), 1e-3)
            logits = (flat @ pos_proto - flat @ neg_proto) / temp
            prob = torch.sigmoid(logits).reshape(h, w).detach().cpu().numpy()
        else:
            centered = flat - flat.mean(dim=0, keepdim=True)
            _, _, V = torch.pca_lowrank(centered, q=1)
            proj = (centered @ V[:, 0]).reshape(h, w).detach().cpu().numpy()
            prob = np.clip(0.5 + 0.5 * proj, 0.0, 1.0)
        c0 = np.array([28.0, 64.0, 150.0], dtype=np.float32)
        c1 = np.array([250.0, 185.0, 25.0], dtype=np.float32)
        rgb = c0[None, None, :] + prob[..., None] * (c1 - c0)[None, None, :]
        return np.clip(np.round(rgb), 0, 255).astype(np.uint8)

    @staticmethod
    def _center_slice_points(n_views: int, spatial: tuple[int, int, int], device: torch.device) -> torch.Tensor:
        """[n_views, H * W, 3] voxel coordinates of every voxel on the center z slice."""
        depth, height, width = spatial
        y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        z = torch.full_like(y, depth // 2)
        points = torch.stack([z, y, x], dim=-1).reshape(1, height * width, 3).float()
        return points.expand(n_views, -1, -1).contiguous()

    def save_monitor_image(self, monitor_batch: Mapping[str, Any], step: int, metrics: Mapping[str, float]) -> Path:
        self.method.model.eval()
        # Per view: the input's center slice, then PCA sets of the patch tokens and, with
        # the dense point head, of its object embedding and ink embedding at every voxel
        # of that slice (colors are comparable only within one panel).
        dense_panels = self.method.model_module.dense_enabled
        with torch.no_grad(), torch.autocast(device_type=self.method.device.type, dtype=self.method.amp_dtype, enabled=self.method.use_amp):
            global_crops = monitor_batch["collated_global_crops"].to(self.method.device, non_blocking=True)
            student_outputs = self.method.model_module(
                student_input=global_crops,
                student_masks=None,
                return_teacher=False,
                point_coordinates=(
                    self._center_slice_points(global_crops.shape[0], tuple(global_crops.shape[2:]), self.method.device)
                    if dense_panels
                    else None
                ),
            )["student"]

        global_views = monitor_batch["collated_global_crops"]
        n_global_views = int(monitor_batch["n_global_views"])
        batch_size = int(monitor_batch["batch_size"])

        rows: list[np.ndarray] = []
        sample_count = min(batch_size, self.monitor_batch_size)
        for sample_index in range(sample_count):
            panels: list[np.ndarray] = []
            for view_index in range(n_global_views):
                tensor_index = view_index * batch_size + sample_index
                center_slice = self._center_slice(global_views[tensor_index])
                token_panels = self._patch_pca_slice(
                    student_outputs["patch_tokens"],
                    tensor_index,
                    target_hw=center_slice.shape,
                    input_spatial_shape=tuple(int(dim) for dim in global_views[tensor_index].shape[1:]),
                )
                input_rgb = np.stack([self._normalize_image(center_slice)] * 3, axis=-1)
                panels.extend([input_rgb, *token_panels])
                if dense_panels:
                    dense = student_outputs["object_points"]["dense"][tensor_index]
                    panels.extend(self._pca_rgb(dense.reshape(*center_slice.shape, dense.shape[-1])))
                    if "ink" in student_outputs["object_points"]:
                        ink_feat = student_outputs["object_points"]["ink"][tensor_index]
                        panels.append(
                            self._ink_rgb(ink_feat.reshape(*center_slice.shape, ink_feat.shape[-1]))
                        )
            rows.append(np.concatenate(panels, axis=1))

        canvas = np.concatenate(rows, axis=0) if rows else np.zeros((256, 256, 3), dtype=np.uint8)
        image_path = self.monitor_dir / f"monitor_step_{step:06d}.jpg"
        Image.fromarray(canvas).save(image_path, quality=90)
        print(
            f"step={step} monitor_image={image_path.name} "
            f"loss={metrics['loss']:.4f} glob={metrics['dino_global_loss']:.4f} "
            f"loc={metrics['dino_local_loss']:.4f} ibot={metrics['ibot_loss']:.4f} "
            f"koleo={metrics['koleo_loss']:.4f} gram={metrics['gram_loss']:.4f} "
            f"point={metrics.get('point_loss', 0.0):.4f}"
        )
        return image_path


