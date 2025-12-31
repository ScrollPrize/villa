
import json
import zarr
import cc3d
import torch
import accelerate
import numpy as np
from typing import Dict, Optional, Tuple

from vesuvius.neural_tracing.dataset import get_crop_from_volume, build_localiser, make_heatmaps
from vesuvius.models.run.tta import infer_with_tta


def is_slots_model(config: dict) -> bool:
    """Detect if model uses slots-based conditioning."""
    if config.get('dataset_variant') == 'slotted':
        return True
    if config.get('masked_conditioning', False):
        return True
    return False


class Inference:

    def __init__(self, model, config, volume_zarr, volume_scale):

        self.accelerator = accelerate.Accelerator(
            mixed_precision=config['mixed_precision'],
        )

        self.model = self.accelerator.prepare(model)
        self.device = self.accelerator.device
        self.config = config
        self.do_tta = config.get('do_tta', False)
        self.tta_type = config.get('tta_type', "rotation")
        self.tta_batched = bool(config.get('tta_batched', True))

        if config.get('compile', True):
            self.model.compile()

        self.model.eval()

        print(f"loading volume zarr {volume_zarr}...")
        ome_zarr = zarr.open_group(volume_zarr, mode='r')
        self.volume = ome_zarr[str(volume_scale)]
        with open(f'{volume_zarr}/meta.json', 'rt') as meta_fp:
            self.voxel_size_um = json.load(meta_fp)['voxelsize']
        print(f"volume shape: {self.volume.shape}, dtype: {self.volume.dtype}, voxel-size: {self.voxel_size_um * 2 ** volume_scale}um")

    def get_heatmaps_at(self, zyx, prev_u, prev_v, prev_diag):
        if isinstance(zyx, torch.Tensor) and zyx.ndim == 1:
            zyx = zyx[None]
            prev_u = [prev_u]
            prev_v = [prev_v]
            prev_diag = [prev_diag]
            not_originally_batched = True
        else:
            not_originally_batched = False
        crop_size = self.config['crop_size']
        use_localiser = bool(self.config.get('use_localiser', True))
        zeros = torch.zeros([1, crop_size, crop_size, crop_size])
        volume_crops = []
        min_corner_zyxs = []
        localisers = []
        prev_u_heatmaps = []
        prev_v_heatmaps = []
        prev_diag_heatmaps = []
        for idx in range(len(zyx)):
            volume_crop, min_corner_zyx = get_crop_from_volume(self.volume, zyx[idx], crop_size)
            volume_crops.append(volume_crop)
            min_corner_zyxs.append(min_corner_zyx)
            if use_localiser:
                localisers.append(build_localiser(zyx[idx], min_corner_zyx, crop_size))
            prev_u_heatmaps.append(make_heatmaps([prev_u[idx][None]], min_corner_zyx, crop_size) if prev_u[idx] is not None else zeros)
            prev_v_heatmaps.append(make_heatmaps([prev_v[idx][None]], min_corner_zyx, crop_size) if prev_v[idx] is not None else zeros)
            prev_diag_heatmaps.append(make_heatmaps([prev_diag[idx][None]], min_corner_zyx, crop_size) if prev_diag[idx] is not None else zeros)
        input_parts = [
            torch.stack(volume_crops)[:, None]
        ] + ([torch.stack(localisers)[:, None]] if use_localiser else []) + [
            torch.stack(prev_u_heatmaps),
            torch.stack(prev_v_heatmaps),
            torch.stack(prev_diag_heatmaps),
        ]
        inputs = torch.cat(input_parts, dim=1).to(self.device, non_blocking=True)
        min_corner_zyxs = torch.stack(min_corner_zyxs)

        def forward(model_inputs):
            outputs = self.model(model_inputs)
            logits = outputs['uv_heatmaps'] if isinstance(outputs, dict) else outputs
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            return logits

        def run_with_tta(model_inputs):
            if self.do_tta:
                return infer_with_tta(forward, model_inputs, self.tta_type, batched=self.tta_batched)
            return forward(model_inputs)

        with torch.no_grad():
            logits = run_with_tta(inputs)
            logits = logits.reshape(len(zyx), 2, self.config['step_count'], crop_size, crop_size, crop_size)  # u/v, step, z, y, x
            probs = torch.sigmoid(logits)

        if not_originally_batched:
            probs = probs.squeeze(0)
            min_corner_zyxs = min_corner_zyxs.squeeze(0)

        return probs, min_corner_zyxs

    def get_heatmaps_at_slots(
        self,
        zyx: torch.Tensor,
        known_slots: Dict[int, Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get heatmap predictions for slots-based model.

        Args:
            zyx: Center point coordinates, shape [3] or [batch, 3]
            known_slots: Dictionary mapping slot indices (0-4) to known positions.
                - Keys: 0=u_neg (above), 1=u_pos (below), 2=v_neg (left), 3=v_pos (right), 4=diag_in
                - Values: zyx coordinates [3] or None if unknown
                - Missing keys or None values are treated as unknown (zero input)

        Returns:
            probs: Probabilities for all 6 output slots, shape [6, crop_size, crop_size, crop_size]
                   or [batch, 6, crop_size, crop_size, crop_size]
            min_corner_zyxs: Minimum corner of crop in volume coordinates

        Slot output order:
            0: u_neg (above), 1: u_pos (below), 2: v_neg (left), 3: v_pos (right),
            4: diag_in, 5: diag_out
        """
        if isinstance(zyx, torch.Tensor) and zyx.ndim == 1:
            zyx = zyx[None]
            known_slots = [known_slots]
            not_originally_batched = True
        else:
            not_originally_batched = False

        crop_size = self.config['crop_size']
        use_localiser = bool(self.config.get('use_localiser', True))
        step_count = self.config.get('step_count', 1)
        include_diag = self.config.get('masked_include_diag', True)

        # Number of conditioning slots (excluding diag_out which is always unknown)
        num_cond_slots = 4 * step_count + (1 if include_diag else 0)  # 5 with diag for step_count=1
        num_out_slots = 4 * step_count + (2 if include_diag else 0)  # 6 with diag for step_count=1

        zeros = torch.zeros([1, crop_size, crop_size, crop_size])

        volume_crops = []
        min_corner_zyxs = []
        localisers = []
        slot_heatmaps_list = []

        for idx in range(len(zyx)):
            volume_crop, min_corner_zyx = get_crop_from_volume(self.volume, zyx[idx], crop_size)
            volume_crops.append(volume_crop)
            min_corner_zyxs.append(min_corner_zyx)

            if use_localiser:
                localisers.append(build_localiser(zyx[idx], min_corner_zyx, crop_size))

            # Build conditioning heatmaps for each slot
            batch_slot_heatmaps = []
            batch_known_slots = known_slots[idx] if known_slots[idx] is not None else {}
            for slot_idx in range(num_cond_slots):
                slot_pos = batch_known_slots.get(slot_idx)
                if slot_pos is not None:
                    heatmap = make_heatmaps([slot_pos[None]], min_corner_zyx, crop_size)
                else:
                    heatmap = zeros
                batch_slot_heatmaps.append(heatmap)
            slot_heatmaps_list.append(torch.cat(batch_slot_heatmaps, dim=0))

        # Assemble input tensor
        input_parts = [
            torch.stack(volume_crops)[:, None]
        ]
        if use_localiser:
            input_parts.append(torch.stack(localisers)[:, None])
        input_parts.append(torch.stack(slot_heatmaps_list))

        inputs = torch.cat(input_parts, dim=1).to(self.device, non_blocking=True)
        min_corner_zyxs = torch.stack(min_corner_zyxs)

        def forward(model_inputs):
            outputs = self.model(model_inputs)
            logits = outputs['uv_heatmaps'] if isinstance(outputs, dict) else outputs
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            return logits

        def run_with_tta(model_inputs):
            if self.do_tta:
                return infer_with_tta(forward, model_inputs, self.tta_type, batched=self.tta_batched)
            return forward(model_inputs)

        with torch.no_grad():
            logits = run_with_tta(inputs)
            logits = logits.reshape(len(zyx), num_out_slots, crop_size, crop_size, crop_size)
            probs = torch.sigmoid(logits)

        if not_originally_batched:
            probs = probs.squeeze(0)
            min_corner_zyxs = min_corner_zyxs.squeeze(0)

        return probs, min_corner_zyxs

    def get_blob_coordinates(self, heatmap, min_corner_zyx, threshold=0.5, min_size=8):
        # Find up to four blobs of sufficient size; return their centroids in descending order of blob size
        # TODO: strip blobs that are further than K * step_size in euclidean space
        cc_labels = cc3d.connected_components((heatmap > threshold).cpu().numpy(), connectivity=18, binary_image=True)
        cc_labels, num_ccs = cc3d.dust(cc_labels, threshold=min_size, precomputed_ccl=True, return_N=True)
        cc_stats = cc3d.statistics(cc_labels)
        centroid_zyxs = cc_stats['centroids'][1:] + min_corner_zyx.numpy()
        size_order = np.argsort(-cc_stats['voxel_counts'][1:])[:4]
        centroid_zyxs = centroid_zyxs[size_order]
        return torch.from_numpy(centroid_zyxs.astype(np.float32))
