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

    def __init__(self, config, patches_for_split, multistep_count, bidirectional):
        config = dict(config)
        aug_config = dict(config.get("augmentation", {}) or {})
        if not config.get("slotted_allow_spatial_transforms", True):
            # Slot channels encode direction; spatial flips/transposes would require channel remapping.
            aug_config["allow_transposes"] = False
            aug_config["allow_mirroring"] = False
        config["augmentation"] = aug_config
        super().__init__(config, patches_for_split, multistep_count, bidirectional)

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

        # Collect cardinal slot data first: (ij, zyx, valid) for slots 0-3
        # Slot mapping: 0=u_neg, 1=u_pos, 2=v_neg, 3=v_pos
        slot_data = []  # list of (ij, zyx_unperturbed, valid)

        def _append_slot_data(ijs, zyxs, valids):
            for idx in range(zyxs.shape[0]):
                slot_data.append((ijs[idx], zyxs[idx], valids[idx]))

        _append_slot_data(u_neg_shifted_ijs, u_neg_shifted_zyxs, u_neg_valid)
        _append_slot_data(u_pos_shifted_ijs, u_pos_shifted_zyxs, u_pos_valid)
        _append_slot_data(v_neg_shifted_ijs, v_neg_shifted_zyxs, v_neg_valid)
        _append_slot_data(v_pos_shifted_ijs, v_pos_shifted_zyxs, v_pos_valid)

        # Compute masking for cardinal slots first (needed for diagonal selection)
        cardinal_valid_mask = torch.stack([s[2] for s in slot_data])
        if not cardinal_valid_mask.any():
            return None

        # Match inference: pick 1 target (unknown), ensure opposite is known, randomly mask perpendiculars
        # Slot mapping: 0=u_neg, 1=u_pos, 2=v_neg, 3=v_pos
        # Opposite pairs: 0↔1, 2↔3 (XOR with 1)
        valid_indices = torch.nonzero(cardinal_valid_mask, as_tuple=False).flatten()
        target_idx = valid_indices[torch.randint(len(valid_indices), size=[])].item()
        opposite_idx = target_idx ^ 1  # flip 0↔1, 2↔3

        # Perpendicular indices (the other axis)
        if target_idx in [0, 1]:  # u-axis target, perpendiculars are v-axis
            perp_indices = [2, 3]
        else:  # v-axis target, perpendiculars are u-axis
            perp_indices = [0, 1]

        # Build masks:
        # - Target: unknown
        # - Opposite: known (if valid), unless force_zero_known
        # - Perpendiculars: randomly masked
        cardinal_known_mask = torch.zeros_like(cardinal_valid_mask)
        cardinal_unknown_mask = torch.zeros_like(cardinal_valid_mask)

        # With small probability, mask all inputs (zero known case for robustness)
        zero_known_prob = float(self._config.get("masked_zero_known_prob", 0.05))
        force_zero_known = (torch.rand([]) < zero_known_prob).item()

        # Target is unknown
        cardinal_unknown_mask[target_idx] = True

        # Opposite is known with probability (not always) - matches inference where
        # opposite might not exist yet at patch edges during bootstrapping
        opposite_known_prob = float(self._config.get("masked_opposite_known_prob", 0.7))
        if cardinal_valid_mask[opposite_idx] and not force_zero_known:
            if torch.rand([]).item() < opposite_known_prob:
                cardinal_known_mask[opposite_idx] = True
            else:
                cardinal_unknown_mask[opposite_idx] = True
        elif cardinal_valid_mask[opposite_idx]:
            cardinal_unknown_mask[opposite_idx] = True

        # Perpendiculars randomly known, otherwise unknown (supervised) since not "behind"
        perp_known_prob = float(self._config.get("masked_condition_known_prob", 0.5))
        for perp_idx in perp_indices:
            if cardinal_valid_mask[perp_idx]:
                if not force_zero_known and torch.rand([]) < perp_known_prob:
                    cardinal_known_mask[perp_idx] = True
                else:
                    cardinal_unknown_mask[perp_idx] = True

        # Handle diagonal slots - select based on which cardinals are unknown (geometric heuristic)
        # This matches inference: diagonal should be OPPOSITE to the gap direction
        if self._config.get("masked_include_diag", True):
            diag_prob = float(self._config.get("masked_diag_prob", 0.5))
            diag_in_ij = diag_out_ij = None
            diag_in_zyx = diag_out_zyx = None
            diag_in_valid = torch.tensor(False)
            diag_out_valid = torch.tensor(False)

            if torch.rand([]) < diag_prob:
                # Pick primary unknown direction (random from unknown cardinals)
                unknown_indices = torch.nonzero(cardinal_unknown_mask, as_tuple=False).flatten()
                primary_target = unknown_indices[torch.randint(len(unknown_indices), size=[])].item()

                # Select diagonal OPPOSITE to primary target direction
                # Slot mapping: 0=u_neg(i-1), 1=u_pos(i+1), 2=v_neg(j-1), 3=v_pos(j+1)
                if primary_target == 0:  # predicting above (u_neg) → diagonal from below (u_pos side)
                    diag_i = u_pos_shifted_ijs[0, 0]
                    diag_j_options = [(v_neg_shifted_ijs[0, 1], v_neg_valid[0]), (v_pos_shifted_ijs[0, 1], v_pos_valid[0])]
                    opposite_diag_i = u_neg_shifted_ijs[0, 0]
                elif primary_target == 1:  # predicting below (u_pos) → diagonal from above (u_neg side)
                    diag_i = u_neg_shifted_ijs[0, 0]
                    diag_j_options = [(v_neg_shifted_ijs[0, 1], v_neg_valid[0]), (v_pos_shifted_ijs[0, 1], v_pos_valid[0])]
                    opposite_diag_i = u_pos_shifted_ijs[0, 0]
                elif primary_target == 2:  # predicting left (v_neg) → diagonal from right (v_pos side)
                    diag_j = v_pos_shifted_ijs[0, 1]
                    diag_i_options = [(u_neg_shifted_ijs[0, 0], u_neg_valid[0]), (u_pos_shifted_ijs[0, 0], u_pos_valid[0])]
                    opposite_diag_j = v_neg_shifted_ijs[0, 1]
                else:  # primary_target == 3: predicting right (v_pos) → diagonal from left (v_neg side)
                    diag_j = v_neg_shifted_ijs[0, 1]
                    diag_i_options = [(u_neg_shifted_ijs[0, 0], u_neg_valid[0]), (u_pos_shifted_ijs[0, 0], u_pos_valid[0])]
                    opposite_diag_j = v_pos_shifted_ijs[0, 1]

                # Build diag_in and diag_out positions
                if primary_target in [0, 1]:  # u-direction target, fixed i for diagonal
                    # Shuffle j options and pick first valid
                    random.shuffle(diag_j_options)
                    for diag_j, j_valid in diag_j_options:
                        if j_valid:
                            diag_in_ij = torch.stack([diag_i, diag_j])
                            diag_out_ij = torch.stack([opposite_diag_i, diag_j])
                            diag_in_zyx = get_zyx_from_patch(diag_in_ij, patch)
                            diag_out_zyx = get_zyx_from_patch(diag_out_ij, patch)
                            diag_in_valid = torch.tensor(True)
                            diag_out_valid = torch.tensor(True)
                            break
                else:  # v-direction target, fixed j for diagonal
                    # Shuffle i options and pick first valid
                    random.shuffle(diag_i_options)
                    for diag_i, i_valid in diag_i_options:
                        if i_valid:
                            diag_in_ij = torch.stack([diag_i, diag_j])
                            diag_out_ij = torch.stack([diag_i, opposite_diag_j])
                            diag_in_zyx = get_zyx_from_patch(diag_in_ij, patch)
                            diag_out_zyx = get_zyx_from_patch(diag_out_ij, patch)
                            diag_in_valid = torch.tensor(True)
                            diag_out_valid = torch.tensor(True)
                            break

            slot_data.append((diag_in_ij, diag_in_zyx, diag_in_valid))
            slot_data.append((diag_out_ij, diag_out_zyx, diag_out_valid))

        # Build full masks including diagonal slots
        valid_mask = torch.stack([s[2] for s in slot_data])
        if self._config.get("masked_include_diag", True):
            # Extend cardinal masks with diagonal masks
            # diag_in randomly known (like perpendiculars), diag_out always unknown
            diag_in_valid = valid_mask[-2]
            diag_out_valid = valid_mask[-1]
            # diag_in: randomly known (maskable to prevent over-reliance), never in output
            diag_in_known = diag_in_valid.item() and not force_zero_known and torch.rand([]).item() < perp_known_prob
            # known_mask has 6 entries (matches slot_data), unknown_mask has 5 (matches output channels)
            known_mask = torch.cat([cardinal_known_mask, torch.tensor([diag_in_known, False])])
            unknown_mask = torch.cat([cardinal_unknown_mask, torch.tensor([diag_out_valid.item()])])
        else:
            known_mask = cardinal_known_mask
            unknown_mask = cardinal_unknown_mask

        # Apply perturbation to known slots for input heatmaps
        should_perturb = (torch.rand([]) < self._perturb_prob).item()
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

        # Output channels exclude diag_in (index 4) - it's only conditioning, never predicted
        if self._config.get("masked_include_diag", True):
            output_slot_heatmaps = slot_heatmaps_out[:4] + [slot_heatmaps_out[5]]  # cardinals + diag_out
        else:
            output_slot_heatmaps = slot_heatmaps_out
        uv_heatmaps_out_all = torch.cat(output_slot_heatmaps, dim=0)

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

        # Store both masks for hybrid supervision in _build_batch_dict
        # Unknown slots get full weight, known slots get reduced weight (stabilizing anchor)
        self._slotted_unknown_mask = unknown_mask
        self._slotted_known_mask = known_mask & valid_mask  # known AND valid

        # Generate srf_overlap mask if enabled - only valid when all 4 cardinal slots are valid
        srf_overlap_mask = None
        srf_overlap_valid = False
        if self._config.get('aux_srf_overlap', False):
            cardinal_all_valid = valid_mask[:4].all()  # slots 0-3 are cardinals
            if cardinal_all_valid:
                from vesuvius.neural_tracing.surf_overlap_loss import render_surf_overlap_mask
                srf_overlap_thickness = self._config.get('srf_overlap_thickness', 2.0)
                # Use slot positions: 0=u_neg, 1=u_pos, 2=v_neg, 3=v_pos
                srf_overlap_mask = render_surf_overlap_mask(
                    slot_zyxs_for_output[0], slot_zyxs_for_output[1],
                    slot_zyxs_for_output[2], slot_zyxs_for_output[3],
                    min_corner_zyx, crop_size, thickness=srf_overlap_thickness
                )
                srf_overlap_valid = True
        self._slotted_srf_overlap_mask = srf_overlap_mask
        self._slotted_srf_overlap_valid = srf_overlap_valid
        # Store cardinal GT positions (in local crop coordinates) for srf_overlap loss
        self._slotted_cardinal_positions = torch.stack([
            slot_zyxs_for_output[0] - min_corner_zyx,  # u_neg
            slot_zyxs_for_output[1] - min_corner_zyx,  # u_pos
            slot_zyxs_for_output[2] - min_corner_zyx,  # v_neg
            slot_zyxs_for_output[3] - min_corner_zyx,  # v_pos
        ])  # [4, 3]

        return {
            'uv_heatmaps_both': uv_heatmaps_both,
            'condition_channels': condition_channels,
            'srf_overlap_mask': srf_overlap_mask,
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
        center_heatmap,
        srf_overlap_mask=None,
    ):
        """Build batch dict for slotted dataset."""
        # Only supervise unknown slots (known slots are conditioning inputs only)
        # This avoids conflict with perturbation: conditioning uses perturbed positions,
        # but targets use unperturbed - we don't want to supervise that mismatch
        loss_mask = self._slotted_unknown_mask.float()

        # Expand to match output shape for loss masking
        loss_mask_expanded = loss_mask.to(
            device=uv_heatmaps_out.device, dtype=uv_heatmaps_out.dtype, non_blocking=True
        ).view(1, 1, 1, -1)
        uv_heatmaps_out_mask = loss_mask_expanded.expand_as(uv_heatmaps_out)

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
        if self._config.get("aux_srf_overlap", False):
            # Use the stored srf_overlap mask and validity from _build_final_heatmaps
            srf_overlap_mask = self._slotted_srf_overlap_mask
            srf_overlap_valid = self._slotted_srf_overlap_valid
            if srf_overlap_mask is not None:
                batch_dict.update({
                    'srf_overlap_mask': srf_overlap_mask,
                    'srf_overlap_valid': torch.tensor(srf_overlap_valid),
                    'cardinal_positions': self._slotted_cardinal_positions,  # [4, 3] GT positions
                })
            else:
                batch_dict.update({'srf_overlap_valid': torch.tensor(False)})

        return batch_dict
