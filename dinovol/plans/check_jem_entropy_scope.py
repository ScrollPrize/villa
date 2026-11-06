"""CPU planning checks for JEM entropy scope; no training or data loading.

Run from dinovol/:
    AGENTS_AGENT_MODE=1 .venv/bin/python plans/check_jem_entropy_scope.py

Distributed checks simulate DDP's parameter-gradient average. They do not test
collectives, model stochastic replay, or the future production JEM implementation.
"""

import math
import unittest

import torch


TYPE_GROUPS = [(list(range(2)), 0.2), (list(range(2, 10)), 0.8)]
SLOT_GROUPS = [([v], 0.1) for v in range(10)]


def entropy(q):
    return -torch.special.xlogy(q, q).sum(-1)


def totals(q, valid, views):
    a = valid[:, views].to(q.dtype)
    return (
        (q[:, views] * a[..., None]).sum((0, 1)),
        (entropy(q[:, views]) * a).sum((0, 1)),
        a.sum((0, 1)),
    )


def marginal(sums, counts):
    return torch.where(
        counts[:, None] > 0,
        sums / counts.clamp_min(1)[:, None],
        torch.full_like(sums, 1 / sums.shape[-1]),
    )


def position_mean(values, counts):
    present = counts > 0
    return values[present].mean() if present.any() else values.sum() * 0


def loss_from_totals(stats, groups):
    result = stats[0][0].sum() * 0
    for (sums, conditional, counts), (_, weight) in zip(stats, groups):
        values = conditional / counts.clamp_min(1) - entropy(marginal(sums, counts))
        result = result + weight * position_mean(values, counts)
    return result


def dense_loss(q, valid, groups):
    return loss_from_totals([totals(q, valid, views) for views, _ in groups], groups)


def streamed_stats(features, weight, valid, groups, partitions):
    """Detached first pass, summing sufficient statistics across rank/chunks."""
    stats = None
    with torch.no_grad():
        for partition in partitions:
            for chunk in partition:
                q = (features[chunk] @ weight).softmax(-1)
                current = [totals(q, valid[chunk], views) for views, _ in groups]
                if stats is None:
                    stats = current
                else:
                    stats = [
                        tuple(a + b for a, b in zip(previous, new))
                        for previous, new in zip(stats, current)
                    ]
    return stats


def local_surrogate(q, valid, stats, groups, world_size):
    """Exact first derivative at collected q; scalar value is not the loss."""
    result = q.sum() * 0
    for (sums, _, counts), (views, group_weight) in zip(stats, groups):
        local_sum, local_conditional, _ = totals(q, valid, views)
        coefficient = (marginal(sums, counts).log() + 1).detach()
        values = world_size * (
            local_conditional + (local_sum * coefficient).sum(-1)
        ) / counts.clamp_min(1)
        result = result + group_weight * position_mean(values, counts)
    return result


class EntropyScopeChecks(unittest.TestCase):
    def setUp(self):
        self.rng = torch.Generator().manual_seed(1701)

    def random_q(self, shape):
        return torch.randn(shape, generator=self.rng, dtype=torch.float64).softmax(-1)

    def assert_close(self, actual, expected, atol=1e-12, rtol=1e-10):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_pooling_difference_is_slot_marginal_kl(self):
        q = self.random_q((7, 10, 3, 11))
        valid = torch.ones(q.shape[:-1], dtype=torch.bool)
        slot_loss = dense_loss(q, valid, SLOT_GROUPS)
        type_loss = dense_loss(q, valid, TYPE_GROUPS)
        gap = q.new_zeros(())
        for views, weight in TYPE_GROUPS:
            mu_slot = q[:, views].mean(0)
            mu_type = mu_slot.mean(0)
            kl = (mu_slot * (mu_slot.log() - mu_type.log())).sum(-1).mean()
            gap += weight * kl
        self.assertGreater(gap.item(), 0)
        self.assert_close(slot_loss - type_loss, gap)
        print(f"\nType-pooling MI increase = slot-marginal KL: {gap.item():.8f} nats")

    def test_unequal_validity_identity_uses_count_weights(self):
        q = self.random_q((7, 8, 3, 11))
        valid = torch.rand((7, 8, 3), generator=self.rng) > 0.3
        valid[0] = True
        count_slot = valid.to(q.dtype).sum(0)
        count = count_slot.sum(0)
        mu_slot = (q * valid[..., None]).sum(0) / count_slot[..., None]
        conditional_slot = (entropy(q) * valid).sum(0) / count_slot
        weights = count_slot / count
        slot_mi = (weights * (entropy(mu_slot) - conditional_slot)).sum(0).mean()
        type_mi = -dense_loss(q, valid, [(list(range(8)), 1.0)])
        mu = (weights[..., None] * mu_slot).sum(0)
        gap = (
            weights * (mu_slot * (mu_slot.log() - mu.log())).sum(-1)
        ).sum(0).mean()
        self.assert_close(type_mi - slot_mi, gap)

    def test_duplicate_views_add_no_information(self):
        q = self.random_q((48, 1, 3, 512))
        one = -dense_loss(q, torch.ones(q.shape[:-1], dtype=torch.bool), [([0], 1.0)])
        for copies in (2, 8):
            repeated = q.repeat(1, copies, 1, 1)
            many = -dense_loss(
                repeated, torch.ones(repeated.shape[:-1], dtype=torch.bool),
                [(list(range(copies)), 1.0)],
            )
            self.assert_close(one, many)

    def test_position_codes_stay_zero_but_slot_codes_expose_shortcut(self):
        q_position = torch.eye(3, dtype=torch.float64)[None, None].repeat(7, 10, 1, 1)
        valid = torch.ones(q_position.shape[:-1], dtype=torch.bool)
        self.assert_close(dense_loss(q_position, valid, TYPE_GROUPS), q_position.new_zeros(()))
        q_slot = torch.eye(8, dtype=torch.float64)[None, :, None].repeat(7, 1, 3, 1)
        valid = torch.ones(q_slot.shape[:-1], dtype=torch.bool)
        pooled = -dense_loss(q_slot, valid, [(list(range(8)), 1.0)])
        separated = -dense_loss(q_slot, valid, [([v], 1 / 8) for v in range(8)])
        self.assert_close(pooled, q_slot.new_tensor(math.log(8)))
        self.assert_close(separated, q_slot.new_zeros(()))

    def test_rank_local_entropy_is_a_different_objective(self):
        q = torch.tensor([[1., 0.]] * 4 + [[0., 1.]] * 4, dtype=torch.float64)[:, None, None]
        valid = torch.ones(q.shape[:-1], dtype=torch.bool)
        groups = [([0], 1.0)]
        global_mi = -dense_loss(q, valid, groups)
        segregated = -sum(dense_loss(q[s], valid[s], groups) for s in (slice(0, 4), slice(4, 8))) / 2
        mixed = -sum(dense_loss(q[s], valid[s], groups) for s in (slice(None, None, 2), slice(1, None, 2))) / 2
        self.assert_close(global_mi, q.new_tensor(math.log(2)))
        self.assert_close(segregated, q.new_zeros(()))
        self.assert_close(mixed, global_mi)

    def test_finite_mixture_bound_is_not_a_marginal_entropy_bound(self):
        q = torch.full((1, 1, 1, 512), 1 / 512, dtype=torch.float64)
        valid = torch.ones(q.shape[:-1], dtype=torch.bool)
        self.assert_close(dense_loss(q, valid, [([0], 1.0)]), q.new_zeros(()))
        self.assert_close(entropy(q).squeeze(), q.new_tensor(math.log(512)))

    def test_exact_parameter_gradients_for_uneven_ranks_and_chunks(self):
        # Shared linear categorical head, source-varying inputs, partial/empty support.
        partitions = [[slice(0, 1), slice(1, 3)], [slice(3, 4)], [slice(4, 6), slice(6, 7)]]
        for dtype in (torch.float64, torch.float32):
            x = torch.randn((7, 10, 4, 5), generator=self.rng, dtype=dtype)
            initial = torch.randn((5, 11), generator=self.rng, dtype=dtype) * 0.4
            valid = torch.rand((7, 10, 4), generator=self.rng) > 0.25
            valid[:, :, -1] = False  # A globally empty position.
            valid[3, :2] = False  # One rank has no valid global-crop entries.
            for groups in (TYPE_GROUPS, SLOT_GROUPS):
                with self.subTest(dtype=dtype, groups=len(groups)):
                    weight = initial.clone().requires_grad_()
                    reference = dense_loss((x @ weight).softmax(-1), valid, groups)
                    reference_gradient, = torch.autograd.grad(reference, weight)
                    stats = streamed_stats(x, initial, valid, groups, partitions)
                    reported = loss_from_totals(stats, groups)
                    gradients = []
                    for rank_chunks in partitions:
                        local_weight = initial.clone().requires_grad_()
                        for chunk in rank_chunks:
                            q = (x[chunk] @ local_weight).softmax(-1)
                            local_surrogate(q, valid[chunk], stats, groups, len(partitions)).backward()
                        gradients.append(local_weight.grad)
                    averaged_gradient = torch.stack(gradients).mean(0)
                    tol = dict(atol=2e-7, rtol=2e-5) if dtype == torch.float32 else {}
                    self.assert_close(reported, reference, **tol)
                    self.assert_close(averaged_gradient, reference_gradient, **tol)
                    error = (averaged_gradient - reference_gradient).abs().max().item()
                    print(f"\n{dtype}, {len(groups)} groups, 3 uneven ranks / 5 chunks: max gradient error {error:.3e}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
