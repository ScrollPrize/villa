from __future__ import annotations

import copy
import math
import unittest

import torch

from dinovol_2.model.dinov2_eva import Eva


def make_model(spatial=(8, 12, 16), patch=(2, 4, 4), **kwargs):
    return Eva(
        input_channels=2, global_crops_size=spatial, local_crops_size=spatial,
        patch_size=patch, embed_dim=24, depth=1, num_heads=2,
        num_reg_tokens=2, use_rot_pos_emb=False, **kwargs,
    )


class VoxelMaskingTests(unittest.TestCase):
    def test_default_matches_token_only_outputs_and_gradients(self):
        for spatial, patch in (((8, 12, 16), (2, 4, 4)), ((8, 12), (2, 4))):
            for mask_kind in ("none", "empty", "mixed", "full"):
                with self.subTest(spatial=spatial, mask=mask_kind):
                    torch.manual_seed(17)
                    model = make_model(spatial, patch)
                    reference = copy.deepcopy(model)
                    x = torch.randn(2, 2, *spatial, requires_grad=True)
                    original = x.detach().clone()
                    ref_x = x.detach().clone().requires_grad_()
                    count = math.prod(size // stride for size, stride in zip(spatial, patch))
                    masks = torch.zeros(2, count, dtype=torch.bool)
                    if mask_kind == "mixed":
                        masks[0, ::3] = True
                        masks[1, 1::2] = True
                    elif mask_kind == "full":
                        masks[:] = True
                    elif mask_kind == "none":
                        masks = None

                    actual = model.forward_features(x, masks)["x_norm_patchtokens"]
                    # Independent reference for the previous token-only path.
                    tokens = reference.down_projection(ref_x).flatten(2).transpose(1, 2).contiguous()
                    if masks is not None:
                        tokens = torch.where(masks.unsqueeze(-1), reference.mask_token, tokens)
                    tokens, rope = reference._pos_embed(tokens, *spatial)
                    for block in reference.blocks:
                        tokens = block(tokens, rope=rope)
                    expected = reference._format_features(tokens, masks, view_kind="global")["x_norm_patchtokens"]
                    torch.testing.assert_close(actual, expected)
                    weights = torch.randn_like(actual)
                    (actual * weights).sum().backward()
                    (expected * weights).sum().backward()
                    torch.testing.assert_close(x.grad, ref_x.grad)
                    for (name, parameter), (_, ref_parameter) in zip(model.named_parameters(), reference.named_parameters()):
                        with self.subTest(parameter=name):
                            if parameter.grad is None or ref_parameter.grad is None:
                                self.assertIsNone(parameter.grad)
                                self.assertIsNone(ref_parameter.grad)
                            else:
                                torch.testing.assert_close(parameter.grad, ref_parameter.grad)
                    torch.testing.assert_close(x.detach(), original, rtol=0, atol=0)

    def test_overlapping_encoder_cannot_observe_hidden_voxels(self):
        for batch_chunk_size in (None, 1):
            with self.subTest(batch_chunk_size=batch_chunk_size):
                torch.manual_seed(19)
                model = make_model(
                    (8, 8, 8), (2, 2, 2), embedding_type="deeper",
                    deeper_embed_batch_chunk_size=batch_chunk_size,
                )
                x = torch.randn(2, 2, 8, 8, 8)
                altered = x.clone()
                masks = torch.zeros(2, 64, dtype=torch.bool)
                masks[:, 21] = True  # Output-grid coordinate (1, 1, 1).
                altered[:, :, 2:4, 2:4, 2:4] += 100
                captured = []
                handle = model.down_projection.stem.register_forward_pre_hook(
                    lambda module, args: captured.append(args[0].detach().clone())
                )
                try:
                    with torch.no_grad():
                        actual = model.forward_features(x, masks)["x_norm_patchtokens"]
                        expected = model.forward_features(altered, masks)["x_norm_patchtokens"]
                finally:
                    handle.remove()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                masked_input = x.clone()
                masked_input[:, :, 2:4, 2:4, 2:4] = 0
                torch.testing.assert_close(torch.cat(captured), masked_input.repeat(2, 1, 1, 1, 1), rtol=0, atol=0)
                # The same change must remain visible to an unmasked teacher.
                with torch.no_grad():
                    teacher = model.forward_features(x)["x_norm_patchtokens"]
                    changed_teacher = model.forward_features(altered)["x_norm_patchtokens"]
                self.assertFalse(torch.allclose(teacher, changed_teacher))

    def test_mask_shape_and_dtype_are_validated(self):
        model = make_model()
        x = torch.randn(2, 2, 8, 12, 16)
        for masks in (torch.zeros(2, 47, dtype=torch.bool), torch.zeros(2, 48)):
            with self.assertRaisesRegex(ValueError, "masks must be boolean with shape"):
                model.prepare_tokens_with_masks(x, masks)


if __name__ == "__main__":
    unittest.main()
