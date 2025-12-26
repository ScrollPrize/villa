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

    def _decide_conditioning(self, use_multistep, u_neg_valid, v_neg_valid,
                             u_pos_shifted_ijs, u_neg_shifted_ijs, v_pos_shifted_ijs, v_neg_shifted_ijs, patch):
        """Override to store ijs/valids/patch for use in _build_final_heatmaps."""
        # Store data needed by _build_final_heatmaps
        self._slotted_patch = patch
        self._slotted_u_pos_shifted_ijs = u_pos_shifted_ijs
        self._slotted_u_neg_shifted_ijs = u_neg_shifted_ijs
        self._slotted_v_pos_shifted_ijs = v_pos_shifted_ijs
        self._slotted_v_neg_shifted_ijs = v_neg_shifted_ijs
        self._slotted_u_neg_valid = u_neg_valid
        self._slotted_v_neg_valid = v_neg_valid
        # Compute center_ij for perturbation (average of first points in each direction)
        self._slotted_center_ij = (u_neg_shifted_ijs[0] + u_pos_shifted_ijs[0] +
                                   v_neg_shifted_ijs[0] + v_pos_shifted_ijs[0]) / 4

        # Return dummy conditioning result (slotted doesn't use directional conditioning)
        return {
            'u_cond': False,
            'v_cond': False,
            'suppress_out_u': None,
            'suppress_out_v': None,
            'diag_zyx': None,
        }

    def _should_swap_uv_axes(self):
        """Disable UV swap for slotted - would need different implementation."""
        return False

    def _build_final_heatmaps(
        self,
        min_corner_zyx,
        crop_size,
        heatmap_sigma,
        u_pos_shifted_zyxs,
        u_neg_shifted_zyxs,
        v_pos_shifted_zyxs,
        v_neg_shifted_zyxs,
        u_neg_shifted_zyxs_unperturbed=None,
        v_neg_shifted_zyxs_unperturbed=None,
        u_cond=None,
        v_cond=None,
        suppress_out_u=None,
        suppress_out_v=None,
        diag_zyx=None,
        center_zyx_unperturbed=None,
    ):
        """Build slot-based heatmaps with masking."""
        # Retrieve stored data from _decide_conditioning
        patch = self._slotted_patch
        u_pos_shifted_ijs = self._slotted_u_pos_shifted_ijs
        u_neg_shifted_ijs = self._slotted_u_neg_shifted_ijs
        v_pos_shifted_ijs = self._slotted_v_pos_shifted_ijs
        v_neg_shifted_ijs = self._slotted_v_neg_shifted_ijs
        center_ij = self._slotted_center_ij

        # Parent guarantees all steps are valid when we reach here
        step_count = u_pos_shifted_zyxs.shape[0]
        u_pos_valid = torch.ones(step_count, dtype=torch.bool)
        u_neg_valid = torch.ones(step_count, dtype=torch.bool)
        v_pos_valid = torch.ones(step_count, dtype=torch.bool)
        v_neg_valid = torch.ones(step_count, dtype=torch.bool)

        # Collect all slot data: (ij, zyx, valid) for each slot
        slot_data = []  # list of (ij, zyx_unperturbed, valid)

        def _append_slot_data(ijs, zyxs, valids):
            for idx in range(zyxs.shape[0]):
                slot_data.append((ijs[idx], zyxs[idx], valids[idx]))

        _append_slot_data(u_neg_shifted_ijs, u_neg_shifted_zyxs, u_neg_valid)
        _append_slot_data(u_pos_shifted_ijs, u_pos_shifted_zyxs, u_pos_valid)
        _append_slot_data(v_neg_shifted_ijs, v_neg_shifted_zyxs, v_neg_valid)
        _append_slot_data(v_pos_shifted_ijs, v_pos_shifted_zyxs, v_pos_valid)

        # Handle diagonal slots
        diag_in_ij = diag_out_ij = None
        diag_in_zyx = diag_out_zyx = None
        if self._config.get("masked_include_diag", True):
            diag_prob = float(self._config.get("masked_diag_prob", 0.5))
            # Opposite diagonal pairs: (diag_in, diag_out)
            diag_pairs = [
                # (u_neg, v_neg) <-> (u_pos, v_pos)
                (
                    (torch.stack([u_neg_shifted_ijs[0, 0], v_neg_shifted_ijs[0, 1]]), u_neg_valid[0] & v_neg_valid[0]),
                    (torch.stack([u_pos_shifted_ijs[0, 0], v_pos_shifted_ijs[0, 1]]), u_pos_valid[0] & v_pos_valid[0]),
                ),
                # (u_neg, v_pos) <-> (u_pos, v_neg)
                (
                    (torch.stack([u_neg_shifted_ijs[0, 0], v_pos_shifted_ijs[0, 1]]), u_neg_valid[0] & v_pos_valid[0]),
                    (torch.stack([u_pos_shifted_ijs[0, 0], v_neg_shifted_ijs[0, 1]]), u_pos_valid[0] & v_neg_valid[0]),
                ),
            ]
            valid_pairs = [(d_in, d_out) for (d_in, d_out) in diag_pairs if d_in[1] and d_out[1]]

            diag_in_valid = torch.tensor(False)
            diag_out_valid = torch.tensor(False)

            if valid_pairs and torch.rand([]) < diag_prob:
                (diag_in_ij, _), (diag_out_ij, _) = random.choice(valid_pairs)
                if torch.rand([]) < 0.5:
                    diag_in_ij, diag_out_ij = diag_out_ij, diag_in_ij
                diag_in_zyx = get_zyx_from_patch(diag_in_ij, patch)
                diag_out_zyx = get_zyx_from_patch(diag_out_ij, patch)
                diag_in_valid = torch.tensor(True)
                diag_out_valid = torch.tensor(True)

            slot_data.append((diag_in_ij, diag_in_zyx, diag_in_valid))
            slot_data.append((diag_out_ij, diag_out_zyx, diag_out_valid))

        valid_mask = torch.stack([s[2] for s in slot_data]) if slot_data else torch.tensor([], dtype=torch.bool)
        if valid_mask.numel() == 0 or not valid_mask.any():
            return None

        known_prob = float(self._config.get("masked_condition_known_prob", 0.5))
        known_mask = (torch.rand_like(valid_mask, dtype=torch.float32) < known_prob) & valid_mask
        unknown_mask = valid_mask & ~known_mask

        # Force diagonal slots: diag_in (slot -2) is always known, diag_out (slot -1) is always unknown
        if self._config.get("masked_include_diag", True):
            diag_in_idx = len(slot_data) - 2
            diag_out_idx = len(slot_data) - 1
            if valid_mask[diag_in_idx]:
                known_mask[diag_in_idx] = True
                unknown_mask[diag_in_idx] = False
            if valid_mask[diag_out_idx]:
                known_mask[diag_out_idx] = False
                unknown_mask[diag_out_idx] = True

        if not unknown_mask.any():
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
            chosen = valid_indices[torch.randint(len(valid_indices), size=[])]
            known_mask[chosen] = False
            unknown_mask[chosen] = True

        # Apply perturbation to known slots for input heatmaps
        should_perturb = torch.rand([]) < self._perturb_prob
        slot_zyxs_for_input = []  # perturbed positions for known slots
        slot_zyxs_for_output = []  # unperturbed positions for all slots

        for idx, (ij, zyx_unperturbed, valid) in enumerate(slot_data):
            # Output always uses unperturbed
            if zyx_unperturbed is not None:
                slot_zyxs_for_output.append(zyx_unperturbed)
            else:
                # Invalid slot - use zeros
                slot_zyxs_for_output.append(torch.zeros(3))

            # Input: perturb if known and should_perturb
            if known_mask[idx] and should_perturb and ij is not None:
                perturbed_zyx = self._get_perturbed_zyx_from_patch(
                    ij, patch, center_ij, min_corner_zyx, crop_size, is_center_point=False
                )
                slot_zyxs_for_input.append(perturbed_zyx)
            elif zyx_unperturbed is not None:
                slot_zyxs_for_input.append(zyx_unperturbed)
            else:
                slot_zyxs_for_input.append(torch.zeros(3))

        # Build heatmaps
        slot_heatmaps_out = []
        for zyx in slot_zyxs_for_output:
            slot_heatmaps_out.append(self.make_heatmaps([zyx[None]], min_corner_zyx, crop_size, sigma=heatmap_sigma))

        slot_heatmaps_in = []
        for zyx in slot_zyxs_for_input:
            slot_heatmaps_in.append(self.make_heatmaps([zyx[None]], min_corner_zyx, crop_size, sigma=heatmap_sigma))

        uv_heatmaps_out_all = torch.cat(slot_heatmaps_out, dim=0)

        # Input channels exclude diag_out (last slot) since it's never conditioned
        if self._config.get("masked_include_diag", True):
            input_slot_heatmaps = slot_heatmaps_in[:-1]  # exclude diag_out
            input_known_mask = known_mask[:-1]
        else:
            input_slot_heatmaps = slot_heatmaps_in
            input_known_mask = known_mask

        uv_heatmaps_in_all = torch.cat(input_slot_heatmaps, dim=0)
        uv_heatmaps_in_all = uv_heatmaps_in_all * input_known_mask[:, None, None, None].to(dtype=uv_heatmaps_in_all.dtype)

        condition_channels = uv_heatmaps_in_all.shape[0]
        uv_heatmaps_both = torch.cat([uv_heatmaps_in_all, uv_heatmaps_out_all], dim=0)

        # Store valid_mask for _build_batch_dict - supervise all valid slots (not just unknown)
        self._slotted_valid_mask = valid_mask

        return {
            'uv_heatmaps_both': uv_heatmaps_both,
            'condition_channels': condition_channels,
        }

    def _build_batch_dict(
        self,
        volume_crop,
        localiser,
        uv_heatmaps_in,
        uv_heatmaps_out,
        seg,
        seg_mask,
        normals,
        normals_mask,
        center_heatmaps,
    ):
        """Build batch dict for slotted dataset."""
        valid_mask = self._slotted_valid_mask

        # Expand valid_mask to match output shape for loss masking
        valid_mask_expanded = valid_mask.to(
            device=uv_heatmaps_out.device, dtype=uv_heatmaps_out.dtype
        ).view(1, 1, 1, -1)
        uv_heatmaps_out_mask = valid_mask_expanded.expand_as(uv_heatmaps_out)

        batch_dict = {
            'volume': volume_crop,
            'localiser': localiser,
            'uv_heatmaps_in': uv_heatmaps_in,
            'uv_heatmaps_out': uv_heatmaps_out,
            'uv_heatmaps_out_mask': uv_heatmaps_out_mask,
        }

        if self._config.get("aux_segmentation", False) and seg is not None:
            batch_dict.update({'seg': seg, 'seg_mask': seg_mask})
        if self._config.get("aux_normals", False) and normals is not None:
            batch_dict.update({'normals': normals, 'normals_mask': normals_mask})

        return batch_dict
