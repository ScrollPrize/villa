from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from dinovol_2.loss import iBOTPatchLoss


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


if __name__ == "__main__":
    unittest.main()
