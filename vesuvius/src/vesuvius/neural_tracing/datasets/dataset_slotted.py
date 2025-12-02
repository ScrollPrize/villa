"""Slot-based dataset variant with masked conditioning for neural tracing."""

import torch
import random

from vesuvius.neural_tracing.dataset import HeatmapDatasetV2, get_zyx_from_patch


class HeatmapDatasetSlotted(HeatmapDatasetV2):
    """
    Variant of HeatmapDatasetV2 that uses fixed slots + masking for conditioning.

    Instead of conditioning on u/v directions, this creates a fixed number of slots
    (one per direction step) and randomly masks some as "known" (input) vs "unknown" (to predict).
    """

    def _build_final_heatmaps(
        self,
        patch,
        min_corner_zyx,
        crop_size,
        heatmap_sigma,
        u_pos_shifted_ijs,
        u_neg_shifted_ijs,
        v_pos_shifted_ijs,
        v_neg_shifted_ijs,
        u_pos_shifted_zyxs,
        u_neg_shifted_zyxs,
        v_pos_shifted_zyxs,
        v_neg_shifted_zyxs,
        u_pos_valid,
        u_neg_valid,
        v_pos_valid,
        v_neg_valid,
        use_multistep=False,
        include_center=False,
    ):
        """Build slot-based heatmaps with masking."""
        slot_heatmaps = []
        slot_valids = []

        def _append_slots(points_zyx, valid_flags):
            for idx in range(points_zyx.shape[0]):
                slot_heatmaps.append(self.make_heatmaps([points_zyx[idx:idx+1]], min_corner_zyx, crop_size, sigma=heatmap_sigma))
                slot_valids.append(valid_flags[idx])

        _append_slots(u_neg_shifted_zyxs, u_neg_valid)
        _append_slots(u_pos_shifted_zyxs, u_pos_valid)
        _append_slots(v_neg_shifted_zyxs, v_neg_valid)
        _append_slots(v_pos_shifted_zyxs, v_pos_valid)

        if self._config.get("masked_include_diag", True):
            diag_prob = float(self._config.get("masked_diag_prob", 0.5))
            diag_candidates = [
                (torch.stack([u_neg_shifted_ijs[0, 0], v_pos_shifted_ijs[0, 1]]), u_neg_valid[0] & v_pos_valid[0]),
                (torch.stack([u_pos_shifted_ijs[0, 0], v_neg_shifted_ijs[0, 1]]), u_pos_valid[0] & v_neg_valid[0]),
                (torch.stack([u_neg_shifted_ijs[0, 0], v_neg_shifted_ijs[0, 1]]), u_neg_valid[0] & v_neg_valid[0]),
                (torch.stack([u_pos_shifted_ijs[0, 0], v_pos_shifted_ijs[0, 1]]), u_pos_valid[0] & v_pos_valid[0]),
            ]
            valid_diags = [(ij, ok) for ij, ok in diag_candidates if ok]
            diag_valid = False
            diag_heatmap = torch.zeros_like(slot_heatmaps[0])
            if valid_diags and torch.rand([]) < diag_prob:
                diag_ij, _ = random.choice(valid_diags)
                diag_zyx = get_zyx_from_patch(diag_ij, patch)
                diag_heatmap = self.make_heatmaps([diag_zyx[None]], min_corner_zyx, crop_size, sigma=heatmap_sigma)
                diag_valid = True
            slot_heatmaps.append(diag_heatmap)
            slot_valids.append(torch.tensor(diag_valid, device=diag_heatmap.device))

        valid_mask = torch.as_tensor(slot_valids, device=slot_heatmaps[0].device, dtype=torch.bool) if slot_heatmaps else torch.tensor([], device=u_pos_shifted_zyxs.device, dtype=torch.bool)
        if valid_mask.numel() == 0 or not valid_mask.any():
            return None

        known_prob = float(self._config.get("masked_condition_known_prob", 0.5))
        known_mask = (torch.rand_like(valid_mask, dtype=torch.float32) < known_prob) & valid_mask
        unknown_mask = valid_mask & ~known_mask
        if not unknown_mask.any():
            # Ensure at least one slot remains to supervise
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
            chosen = valid_indices[torch.randint(len(valid_indices), size=[])]
            known_mask[chosen] = False
            unknown_mask[chosen] = True

        uv_heatmaps_out_all = torch.cat(slot_heatmaps, dim=0)
        uv_heatmaps_in_all = uv_heatmaps_out_all * known_mask[:, None, None, None].to(dtype=uv_heatmaps_out_all.dtype)
        out_channel_mask = unknown_mask

        include_condition_mask = bool(self._config.get("include_condition_mask_channel", True))
        condition_mask_for_batch = known_mask[:, None, None, None].to(dtype=uv_heatmaps_out_all.dtype, device=uv_heatmaps_out_all.device)
        condition_mask_for_batch = condition_mask_for_batch.expand_as(uv_heatmaps_out_all)
        condition_mask_channels = condition_mask_for_batch.shape[0] if include_condition_mask else 0

        condition_channels = uv_heatmaps_in_all.shape[0]
        uv_heatmaps_both = torch.cat(
            [uv_heatmaps_in_all, uv_heatmaps_out_all] + ([condition_mask_for_batch] if include_condition_mask else []),
            dim=0
        )
        uv_heatmaps_out_all_channels = uv_heatmaps_out_all.shape[0]

        maybe_center_heatmap = {}
        if include_center:
            maybe_center_heatmap['center_heatmap'] = self.make_heatmaps(
                [torch.full([1, 3], crop_size / 2)],
                torch.zeros([3]),
                crop_size,
                sigma=heatmap_sigma,
            )

        return {
            'uv_heatmaps_both': uv_heatmaps_both,
            'condition_channels': condition_channels,
            'uv_heatmaps_out_all_channels': uv_heatmaps_out_all_channels,
            'out_channel_mask': out_channel_mask,
            'condition_mask_channels': condition_mask_channels,
            **maybe_center_heatmap,
        }

    def _build_batch_dict(
        self,
        volume_crop,
        localiser,
        uv_heatmaps_in,
        uv_heatmaps_out,
        out_channel_mask,
        condition_mask_aug,
        seg,
        seg_mask,
        normals,
        normals_mask,
        center_heatmaps=None,
    ):
        """Build batch dict with masking for slotted dataset."""
        out_channel_mask_expanded = out_channel_mask.to(
            device=uv_heatmaps_out.device, dtype=uv_heatmaps_out.dtype
        ).view(1, 1, 1, -1)
        uv_heatmaps_out_mask = out_channel_mask_expanded.expand_as(uv_heatmaps_out)
        # Apply channel mask so known slots are zeroed; keep background in unknown slots.
        uv_heatmaps_out = uv_heatmaps_out * uv_heatmaps_out_mask

        batch_dict = {
            'volume': volume_crop,
            'localiser': localiser,
            'uv_heatmaps_in': uv_heatmaps_in,
            'uv_heatmaps_out': uv_heatmaps_out,
            'uv_heatmaps_out_mask': uv_heatmaps_out_mask,
            **({'center_heatmaps': center_heatmaps} if center_heatmaps is not None else {}),
        }
        if condition_mask_aug is not None:
            batch_dict['condition_mask'] = condition_mask_aug

        if self._config.get("aux_segmentation", False) and seg is not None:
            batch_dict.update({'seg': seg, 'seg_mask': seg_mask})
        if self._config.get("aux_normals", False) and normals is not None:
            batch_dict.update({'normals': normals, 'normals_mask': normals_mask})

        return batch_dict
