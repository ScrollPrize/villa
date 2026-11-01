from __future__ import annotations

import tempfile
import unittest

import torch
import torch.nn.functional as F

from dinovol_2.loss import iBOTPatchLoss
from dinovol_2.pretrain import DinoIBOTPretrainer


def reference_sinkhorn(teacher_output, teacher_temp, n_masked, n_iterations=3):
    Q = torch.exp(teacher_output.float() / teacher_temp).t()
    B, K = n_masked, Q.shape[0]
    Q /= torch.sum(Q)
    for _ in range(n_iterations):
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    Q *= B
    return Q.t()


def reference_masked_loss(student, targets, masks, weights, temp):
    rows = torch.sum(targets.float() * F.log_softmax(student.float() / temp, dim=-1), dim=-1)
    return -(rows * weights).sum() / masks.shape[0]


class IBOTLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.masks = torch.zeros(4, 16, dtype=torch.bool)
        self.masks[0, :5] = True
        self.masks[2, 3:12] = True
        n_masked = int(self.masks.sum())
        self.weights = (1. / self.masks.sum(1).clamp(min=1)).unsqueeze(1).expand_as(self.masks)[self.masks]
        self.teacher = torch.randn(n_masked, 40)
        self.student = torch.randn(n_masked, 40)

    def test_sinkhorn_matches_transposed_reference_without_modifying_input(self):
        teacher = self.teacher.clone()
        n_masked = torch.tensor([teacher.shape[0]])
        actual = iBOTPatchLoss(40).sinkhorn_knopp_teacher(teacher, 0.07, n_masked)
        torch.testing.assert_close(actual, reference_sinkhorn(self.teacher, 0.07, teacher.shape[0]))
        torch.testing.assert_close(teacher, self.teacher, rtol=0, atol=0)
        self.assertTrue(actual.is_contiguous())

    def test_chunked_loss_matches_autograd_reference(self):
        targets = reference_sinkhorn(self.teacher, 0.07, self.teacher.shape[0])
        expected_student = self.student.clone().requires_grad_()
        expected = reference_masked_loss(expected_student, targets, self.masks, self.weights, 0.1)
        expected.backward()
        for chunk_size in (None, 3, 4096):
            student = self.student.clone().requires_grad_()
            loss = iBOTPatchLoss(40, masked_loss_chunk_size=chunk_size).forward_masked(
                student, targets, self.masks, n_masked_patches=student.shape[0], masks_weight=self.weights,
            )
            loss.backward()
            torch.testing.assert_close(loss, expected)
            torch.testing.assert_close(student.grad, expected_student.grad)

    def test_half_precision_logits_receive_half_precision_gradients(self):
        targets = reference_sinkhorn(self.teacher, 0.07, self.teacher.shape[0])
        student = self.student.to(torch.bfloat16).requires_grad_()
        loss = iBOTPatchLoss(40, masked_loss_chunk_size=5).forward_masked(
            student, targets, self.masks, masks_weight=self.weights,
        )
        loss.backward()
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(student.grad.dtype, torch.bfloat16)
        reference = student.detach().float().requires_grad_()
        reference_masked_loss(reference, targets, self.masks, self.weights, 0.1).backward()
        torch.testing.assert_close(student.grad.float(), reference.grad, rtol=1e-2, atol=1e-4)

    def test_fused_last_layer_matches_materialized_logits(self):
        features = F.normalize(torch.randn(self.student.shape[0], 6), dim=-1)
        teacher_features = F.normalize(torch.randn(self.student.shape[0], 6), dim=-1)
        weight = torch.randn(40, 6)
        teacher_weight = torch.randn(40, 6)
        n_masked = torch.tensor([features.shape[0]])

        expected_features = features.clone().requires_grad_()
        expected_weight = weight.clone().requires_grad_()
        loss = iBOTPatchLoss(40)
        targets = loss.sinkhorn_knopp_teacher(teacher_features @ teacher_weight.t(), 0.05, n_masked)
        expected = reference_masked_loss(expected_features @ expected_weight.t(), targets, self.masks, self.weights, 0.1)
        expected.backward()

        for chunk_size in (None, 4):
            loss = iBOTPatchLoss(40, masked_loss_chunk_size=chunk_size)
            actual_features = features.clone().requires_grad_()
            actual_weight = weight.clone().requires_grad_()
            log_scales = loss.sinkhorn_knopp_prototype_scales(teacher_features, teacher_weight, 0.05, n_masked)
            actual = loss.forward_masked_from_features(
                actual_features, actual_weight, teacher_features, teacher_weight, log_scales, 0.05,
                self.masks, self.weights,
            )
            actual.backward()
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual_features.grad, expected_features.grad)
            torch.testing.assert_close(actual_weight.grad, expected_weight.grad)

    def test_fused_training_step_matches_materialized_step(self):
        config = {
            "device": "cpu", "use_amp": False, "output_dir": None, "max_iterations": 3,
            "warmup_steps": 0, "freeze_last_layer_steps": 0, "batch_size": 2, "lr": 1e-3,
            "ibot_masked_loss_chunk_size": 7,
            "model": {
                "model_type": "v3", "embed_dim": 36, "num_heads": 3, "depth": 2,
                "global_crops_size": [16] * 3, "local_crops_size": [8] * 3, "patch_size": [4] * 3,
                "drop_path_rate": 0., "rope_rescale_coords": None,
                "dino_out_dim": 64, "ibot_out_dim": 48,
                "dino_head_hidden_dim": 32, "ibot_head_hidden_dim": 32,
                "dino_head_bottleneck_dim": 12, "ibot_head_bottleneck_dim": 12,
            },
        }
        masks = torch.zeros(4, 64, dtype=torch.bool)
        masks[:2, ::3] = True
        indices = masks.flatten().nonzero().flatten()
        batch = {
            "collated_global_crops": torch.randn(4, 1, 16, 16, 16),
            "collated_local_crops": torch.randn(4, 1, 8, 8, 8),
            "collated_masks": masks, "mask_indices_list": indices,
            "n_masked_patches": torch.tensor(indices.numel()),
            "masks_weight": (1. / masks.sum(1).clamp(min=1)).unsqueeze(1).expand_as(masks)[masks],
            "n_global_views": 2, "n_local_views": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            trainers = []
            for fused in (False, True):
                torch.manual_seed(3)
                trainers.append(DinoIBOTPretrainer({**config, "output_dir": directory, "ibot_fused_last_layer": fused}))
            materialized, fused = trainers
            fused.model_module.load_state_dict(materialized.model_module.state_dict())
            for step in range(2):
                expected = materialized.train_step(batch, step)
                actual = fused.train_step(batch, step)
                for name in ("loss", "ibot_loss", "dino_global_loss", "koleo_loss"):
                    torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6, msg=name)
                for name, tensor in materialized.model_module.state_dict().items():
                    torch.testing.assert_close(fused.model_module.state_dict()[name], tensor,
                                               rtol=1e-4, atol=1e-6, msg=name)


if __name__ == "__main__":
    unittest.main()
