"""CPU executable specifications for JEM reductions and mask-status diagnostics.

Run from dinovol/:
    AGENTS_AGENT_MODE=1 .venv/bin/python plans/check_jem_reductions.py

These are planning references, not production JEM tests. The future production
losses must pass the same fixed value/gradient fixtures. Entropy arithmetic is
reused from check_jem_entropy_scope.py, whose estimator decisions are unchanged.
"""

import math
import unittest

import torch

from check_jem_entropy_scope import (
    SLOT_GROUPS,
    TYPE_GROUPS,
    dense_loss,
    entropy,
    marginal,
    position_mean,
    totals,
)


def alignment_reference(kl, mask):
    """One depth/view; kl contains only valid corresponding patch losses."""
    zero = kl.sum() * 0
    visible = kl[~mask].mean() if (~mask).any() else zero
    masked = kl[mask].mean() if mask.any() else zero
    return 0.2 * visible + 0.8 * masked


def depth_sum_reference(values):
    """Semantic depths [4, 8, 12], with no normalization by weight sum."""
    if values.shape != (3,):
        raise ValueError("Expected three scalar losses at depths [4, 8, 12]")
    return (values * values.new_tensor([0.25, 0.25, 1.0])).sum()


def categorical_reductions_reference(alignment_by_depth, global_by_depth):
    return (
        depth_sum_reference(alignment_by_depth),
        depth_sum_reference(global_by_depth),
    )


@torch.no_grad()
def entropy_mask_contributions(q, valid, view_was_masked, groups):
    """Attribute one depth's true entropy loss using its unchanged marginals.

    q: [source, slot, position, category]; valid: [source, slot, position].
    view_was_masked: [source, slot], computed over the entire student mask.
    Returned values include crop-type/slot weights, before depth/schedule weights.
    """
    result = {False: q.new_zeros(()), True: q.new_zeros(())}
    for views, weight in groups:
        sums, _, counts = totals(q, valid, views)
        mu = marginal(sums, counts)
        selected_valid = valid[:, views]
        # Zero invalid vectors before logarithms, including zero-mass categories.
        selected_q = q[:, views].masked_fill(~selected_valid[..., None], 0)
        per_entry = entropy(selected_q) + torch.special.xlogy(selected_q, mu).sum(-1)
        for masked_status in result:
            membership = (view_was_masked[:, views] == masked_status)[..., None]
            values = (per_entry * membership).sum((0, 1)) / counts.clamp_min(1)
            result[masked_status] += weight * position_mean(values, counts)
    return result


class ReductionChecks(unittest.TestCase):
    def assert_close(self, actual, expected):
        expected = torch.as_tensor(expected, dtype=actual.dtype, device=actual.device)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_unmasked_view_keeps_visible_coefficient(self):
        kl = torch.tensor([1., 3.], requires_grad=True)
        loss = alignment_reference(kl, torch.tensor([False, False]))
        self.assert_close(loss, 0.4)
        loss.backward()
        self.assert_close(kl.grad, [0.1, 0.1])

    def test_mixed_all_masked_and_empty_subsets(self):
        for values, mask, expected, gradient in (
            ([1., 3., 5.], [False, False, True], 4.4, [0.1, 0.1, 0.8]),
            ([1., 3.], [True, True], 1.6, [0.4, 0.4]),
            ([], [], 0., []),
        ):
            with self.subTest(mask=mask):
                kl = torch.tensor(values, requires_grad=True)
                loss = alignment_reference(kl, torch.tensor(mask, dtype=torch.bool))
                self.assert_close(loss, expected)
                loss.backward()  # Empty subsets must still be graph connected.
                self.assert_close(kl.grad, gradient)

    def test_both_categorical_losses_keep_unnormalized_depth_weights(self):
        alignment = torch.tensor([2., 4., 8.], requires_grad=True)
        global_collapse = torch.tensor([-1., -2., -4.], requires_grad=True)
        align_loss, global_loss = categorical_reductions_reference(alignment, global_collapse)
        self.assert_close(align_loss, 9.5)
        self.assert_close(global_loss, -4.75)
        (align_loss + global_loss).backward()
        self.assert_close(alignment.grad, [0.25, 0.25, 1.])
        self.assert_close(global_collapse.grad, [0.25, 0.25, 1.])

    def test_unmasked_view_across_three_depths(self):
        means = torch.tensor([2., 2., 2.], requires_grad=True)
        per_depth = torch.stack([
            alignment_reference(mean.expand(2), torch.tensor([False, False]))
            for mean in means
        ])
        loss = depth_sum_reference(per_depth)
        self.assert_close(loss, 0.6)
        loss.backward()
        self.assert_close(means.grad, [0.05, 0.05, 0.2])

    def test_view_contributions_use_original_denominator(self):
        masks = torch.tensor([[False, False]] * 2 + [[False, True]] * 8)
        losses = torch.stack([
            alignment_reference(torch.tensor([2., 2.]), mask) for mask in masks
        ])
        was_masked = masks.any(-1)
        unmasked_contribution = losses[~was_masked].sum() / 10
        masked_contribution = losses[was_masked].sum() / 10
        self.assert_close(unmasked_contribution, 0.08)
        self.assert_close(masked_contribution, 1.60)
        self.assert_close(losses.mean(), 1.68)
        self.assert_close(unmasked_contribution + masked_contribution, losses.mean())

    def test_mask_status_includes_patches_outside_overlap(self):
        full_mask = torch.tensor([False, False, True, False])
        overlap = torch.tensor([0, 1])
        self.assertTrue(full_mask.any().item())
        self.assertFalse(full_mask[overlap].any().item())
        loss = alignment_reference(torch.tensor([1., 3.]), full_mask[overlap])
        self.assert_close(loss, 0.4)  # Still belongs in masked-view diagnostics.

    @staticmethod
    def opposite_categories():
        q = torch.eye(2)[:, None, None, :].repeat(1, 10, 1, 1)
        return q, torch.ones(q.shape[:-1], dtype=torch.bool)

    def test_entropy_formula_with_three_depths(self):
        q, valid = self.opposite_categories()
        for groups in (TYPE_GROUPS, SLOT_GROUPS):
            with self.subTest(scopes=len(groups)):
                per_depth = dense_loss(q, valid, groups)
                self.assert_close(per_depth, -math.log(2))
                self.assert_close(depth_sum_reference(per_depth.repeat(3)), -1.5 * math.log(2))

    def test_entropy_groups_use_full_batch_marginal(self):
        q, valid = self.opposite_categories()
        status = torch.tensor([[False] * 10, [True] * 10])
        for groups in (TYPE_GROUPS, SLOT_GROUPS):
            with self.subTest(scopes=len(groups)):
                contributions = entropy_mask_contributions(q, valid, status, groups)
                self.assert_close(contributions[False], -math.log(2) / 2)
                self.assert_close(contributions[True], -math.log(2) / 2)
                self.assert_close(sum(contributions.values()), dense_loss(q, valid, groups))
                for source in (slice(0, 1), slice(1, 2)):
                    # Recomputing either subgroup's marginal loses all this MI.
                    self.assert_close(dense_loss(q[source], valid[source], groups), 0.)
                scheduled = 0.1 * sum(
                    depth_sum_reference(contribution.repeat(3))
                    for contribution in contributions.values()
                )
                self.assert_close(scheduled, -0.15 * math.log(2))

    def test_entropy_group_additivity_with_invalid_support_and_empty_groups(self):
        rng = torch.Generator().manual_seed(2801)
        q = torch.randn((3, 10, 3, 5), generator=rng).softmax(-1)
        valid = torch.rand((3, 10, 3), generator=rng) > 0.3
        valid[:, :, -1] = False
        valid[0, :2] = False
        mixed = torch.rand((3, 10), generator=rng) > 0.5
        for groups in (TYPE_GROUPS, SLOT_GROUPS):
            for status in (mixed, torch.zeros_like(mixed), torch.ones_like(mixed)):
                contributions = entropy_mask_contributions(q, valid, status, groups)
                self.assert_close(sum(contributions.values()), dense_loss(q, valid, groups))
                if not status.any():
                    self.assert_close(contributions[True], 0.)
                if status.all():
                    self.assert_close(contributions[False], 0.)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
