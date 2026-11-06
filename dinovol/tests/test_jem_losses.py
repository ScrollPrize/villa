"""Production JEM reductions against the fixed fixtures of plans/JEM_impl.md §5 and dense references."""
import math
import unittest

import torch

from dinovol_2.loss.jem_losses import (
    EntropyStatistics, alignment_from_means, band_pair_weights, distance_metric, entropy, entropy_attribution,
    entropy_groups, entropy_surrogate, global_collapse_weight, kl_divergence, nonaffine_target,
    structural_band_counts, structural_dense_reference, structural_distance_stratified,
    structural_uniform_contraction, structural_uniform_tiled, structural_weight, subset_means,
)

DEPTHS = [4, 8, 12]
DEPTH_WEIGHTS = {4: 0.25, 8: 0.25, 12: 1.0}


def alignment(kl, masked):
    rows = torch.zeros(len(kl), dtype=torch.long)
    visible, masked_mean, _, _ = subset_means(kl, rows, masked, 1)
    return alignment_from_means(visible, masked_mean)[0]


def depth_sum(values):
    return sum(DEPTH_WEIGHTS[d] * v for d, v in zip(DEPTHS, values))


def dense_global_loss(q, valid, groups, positions_first=False):
    """q [source, slot, position, K]; groups as (slots, weight)."""
    total = q.sum() * 0
    for slots, weight in groups:
        a = valid[:, slots].to(q.dtype)
        selected = q[:, slots]
        counts = a.sum((0, 1))
        mu = torch.where(counts[:, None] > 0, (selected * a[..., None]).sum((0, 1)) / counts.clamp_min(1)[:, None],
                         torch.full_like(selected[0, 0], 1.0 / selected.shape[-1]))
        conditional = (-torch.special.xlogy(selected, selected).sum(-1) * a).sum((0, 1)) / counts.clamp_min(1)
        values = conditional + torch.special.xlogy(mu, mu).sum(-1)
        present = counts > 0
        total = total + weight * (values[present].mean() if present.any() else values.sum() * 0)
    return total


TYPE_GROUPS = [(list(range(2)), 0.2), (list(range(2, 10)), 0.8)]
SLOT_GROUPS = [([v], 0.1) for v in range(10)]


def statistics_for(groups_scope, n_positions, K, device="cpu"):
    groups = entropy_groups(groups_scope, n_global_slots=2, n_local_slots=8)
    return EntropyStatistics(groups, [12], {"global": n_positions, "local": n_positions}, K, device), groups


def accumulate_q(stats, log_q, valid):
    """log_q [source, slot(10), position, K]: slots 0-1 global, 2-9 local."""
    S = log_q.shape[0]
    for kind, slots in (("global", range(2)), ("local", range(2, 10))):
        slots = list(slots)
        rows_log_q = log_q[:, slots].transpose(0, 1).reshape(-1, *log_q.shape[2:])
        rows_valid = valid[:, slots].transpose(0, 1).reshape(-1, valid.shape[2])
        local_slots = torch.arange(len(slots)).repeat_interleave(S)
        sources = torch.arange(S).repeat(len(slots))
        stats.accumulate(12, kind, rows_log_q, rows_valid, local_slots, sources)


class AlignmentFixtures(unittest.TestCase):
    def check(self, actual, expected):
        torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=actual.dtype), atol=1e-6, rtol=1e-6)

    def test_subset_fixtures(self):
        for values, mask, expected, gradient in (
            ([1., 3.], [False, False], 0.4, [0.1, 0.1]),
            ([1., 3.], [True, True], 1.6, [0.4, 0.4]),
            ([1., 3., 5.], [False, False, True], 4.4, [0.1, 0.1, 0.8]),
            ([], [], 0., []),
        ):
            with self.subTest(mask=mask):
                kl = torch.tensor(values, requires_grad=True)
                loss = alignment(kl, torch.tensor(mask, dtype=torch.bool))
                self.check(loss, expected)
                loss.backward()  # empty subsets stay graph connected
                self.check(kl.grad, gradient)

    def test_unnormalized_depth_weights(self):
        values = torch.tensor([2., 4., 8.], requires_grad=True)
        loss = depth_sum(values)
        self.check(loss, 9.5)
        loss.backward()
        self.check(values.grad, [0.25, 0.25, 1.])
        values = torch.tensor([-1., -2., -4.], requires_grad=True)
        loss = depth_sum(values)
        self.check(loss, -4.75)
        loss.backward()
        self.check(values.grad, [0.25, 0.25, 1.])

    def test_unmasked_views_across_three_depths(self):
        means = torch.tensor([2., 2., 2.], requires_grad=True)
        loss = depth_sum([alignment(mean.expand(2), torch.tensor([False, False])) for mean in means])
        self.check(loss, 0.6)
        loss.backward()
        self.check(means.grad, [0.05, 0.05, 0.2])

    def test_ten_view_mean_and_mask_status_contributions(self):
        masks = torch.tensor([[False, False]] * 2 + [[False, True]] * 8)
        kl = torch.full((10, 2), 2.0)
        rows = torch.arange(10).repeat_interleave(2)
        visible, masked, _, _ = subset_means(kl.flatten(), rows, masks.flatten(), 10)
        losses = alignment_from_means(visible, masked)
        was_masked = masks.any(-1)
        self.check(losses[~was_masked].sum() / 10, 0.08)
        self.check(losses[was_masked].sum() / 10, 1.60)
        self.check(losses.mean(), 1.68)

    def test_kl_is_zero_safe_and_gather_commutes_with_softmax(self):
        teacher = torch.tensor([[0., -1e4, 3.], [1., 2., 3.]])
        student = torch.log_softmax(torch.randn(2, 3), -1)
        kl = kl_divergence(teacher, student, 0.1)
        self.assertTrue(torch.isfinite(kl).all())
        probabilities = torch.softmax(teacher / 0.1, -1)
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(2))
        index = torch.tensor([1, 0, 1])
        torch.testing.assert_close(torch.softmax(teacher[index] / 0.1, -1), probabilities[index])


class EntropyTests(unittest.TestCase):
    def test_opposite_categories_and_mask_status_attribution(self):
        # Two sources with opposite one-hot categories at every slot/position.
        log_q = torch.log(torch.eye(2)[:, None, None, :].repeat(1, 10, 1, 1).clamp_min(1e-30))
        log_q = torch.log_softmax(log_q, -1)
        valid = torch.ones(log_q.shape[:-1], dtype=torch.bool)
        for scope in ("crop_type", "slot"):
            with self.subTest(scope=scope):
                stats, groups = statistics_for(scope, 1, 2)
                accumulate_q(stats, log_q, valid)
                total = sum(group.weight * stats.group_loss((12, group.name))["loss"] for group in groups)
                torch.testing.assert_close(total, torch.tensor(-math.log(2)), atol=1e-6, rtol=0)
                torch.testing.assert_close(depth_sum([total] * 3), torch.tensor(-1.5 * math.log(2)), atol=1e-6, rtol=0)
                # Source 0 wholly unmasked, source 1 masked: each gets half, from the full-batch marginal.
                shares = torch.zeros(2)
                for group in groups:
                    coef, _, log_mu = stats.coefficients((12, group.name), group.weight)
                    slots = [s if group.view_type == "global" else 2 + s for s in group.slots]
                    for source in range(2):
                        rows = log_q[source, slots]
                        shares[source] += entropy_attribution(rows, valid[source, slots], coef, log_mu).sum()
                torch.testing.assert_close(shares, torch.full((2,), -math.log(2) / 2), atol=1e-6, rtol=0)
                torch.testing.assert_close(0.1 * depth_sum([shares.sum()] * 3), torch.tensor(-0.15 * math.log(2)),
                                           atol=1e-6, rtol=0)

    def test_statistics_match_dense_reference_with_invalid_support(self):
        g = torch.Generator().manual_seed(3)
        log_q = torch.log_softmax(torch.randn(5, 10, 4, 7, generator=g), -1)
        valid = torch.rand(5, 10, 4, generator=g) > 0.3
        valid[:, :, -1] = False  # empty position, excluded from the mean
        for scope, groups in (("crop_type", TYPE_GROUPS), ("slot", SLOT_GROUPS)):
            stats, named = statistics_for(scope, 4, 7)
            accumulate_q(stats, log_q, valid)
            total = sum(group.weight * stats.group_loss((12, group.name))["loss"] for group in named)
            torch.testing.assert_close(total, dense_global_loss(log_q.exp(), valid, groups), atol=1e-5, rtol=1e-5)

    def test_surrogate_gradient_matches_dense_autograd_across_ranks_and_chunks(self):
        """Uneven simulated ranks and chunks: averaged surrogate gradients equal the dense gradient."""
        g = torch.Generator().manual_seed(11)
        dtype = torch.float64
        x = torch.randn(7, 10, 4, 5, generator=g, dtype=dtype)
        initial = torch.randn(5, 6, generator=g, dtype=dtype) * 0.5
        valid = torch.rand(7, 10, 4, generator=g) > 0.25
        valid[:, :, -1] = False
        valid[3, :2] = False
        partitions = [[slice(0, 1), slice(1, 3)], [slice(3, 4)], [slice(4, 6), slice(6, 7)]]
        world = len(partitions)
        for scope, groups in (("crop_type", TYPE_GROUPS), ("slot", SLOT_GROUPS)):
            with self.subTest(scope=scope):
                weight = initial.clone().requires_grad_()
                reference = dense_global_loss(torch.softmax(x @ weight, -1), valid, groups)
                reference_grad, = torch.autograd.grad(reference, weight)
                stats, named = statistics_for(scope, 4, 6)
                for part in partitions:
                    for chunk in part:
                        accumulate_q(stats, torch.log_softmax(x[chunk] @ initial, -1), valid[chunk])
                reported = sum(group.weight * stats.group_loss((12, group.name))["loss"] for group in named)
                torch.testing.assert_close(reported.double(), reference.detach(), atol=1e-6, rtol=1e-5)
                gradients = []
                for part in partitions:
                    local = initial.clone().requires_grad_()
                    for chunk in part:
                        log_q = torch.log_softmax(x[chunk] @ local, -1)
                        surrogate = 0
                        for group in named:
                            coef, d, _ = stats.coefficients((12, group.name), world * group.weight)
                            slots = [s if group.view_type == "global" else 2 + s for s in group.slots]
                            rows = log_q[:, slots].reshape(-1, *log_q.shape[2:])
                            surrogate = surrogate + entropy_surrogate(
                                rows, valid[chunk][:, slots].reshape(-1, valid.shape[2]), coef.double(), d.double())
                        surrogate.backward()
                    gradients.append(local.grad)
                torch.testing.assert_close(torch.stack(gradients).mean(0), reference_grad, atol=1e-6, rtol=1e-5)

    def test_position_codes_have_zero_mutual_information(self):
        log_q = torch.log_softmax(torch.eye(3)[None, None].repeat(7, 10, 1, 1) * 50, -1)
        valid = torch.ones(log_q.shape[:-1], dtype=torch.bool)
        stats, named = statistics_for("crop_type", 3, 3)
        accumulate_q(stats, log_q, valid)
        total = sum(group.weight * stats.group_loss((12, group.name))["loss"] for group in named)
        self.assertAlmostEqual(float(total), 0.0, places=5)

    def test_duplicate_slots_add_no_information(self):
        g = torch.Generator().manual_seed(5)
        one = torch.log_softmax(torch.randn(6, 1, 2, 9, generator=g), -1)
        repeated = one.repeat(1, 10, 1, 1)
        valid = torch.ones(repeated.shape[:-1], dtype=torch.bool)
        stats, named = statistics_for("crop_type", 2, 9)
        accumulate_q(stats, repeated, valid)
        pooled = [stats.group_loss((12, group.name))["loss"] for group in named]
        single = dense_global_loss(one.exp(), torch.ones(one.shape[:-1], dtype=torch.bool), [([0], 1.0)])
        for value in pooled:
            torch.testing.assert_close(value, single.float(), atol=1e-6, rtol=1e-5)

    def test_entropy_underflow_is_zero_safe(self):
        log_q = torch.log_softmax(torch.tensor([[[[0., -200., 0.]]]]).repeat(2, 10, 1, 1), -1)
        valid = torch.ones(log_q.shape[:-1], dtype=torch.bool)
        stats, named = statistics_for("crop_type", 1, 3)
        accumulate_q(stats, log_q, valid)
        for group in named:
            coef, d, log_mu = stats.coefficients((12, group.name), 1.0)
            self.assertTrue(torch.isfinite(d).all() and torch.isfinite(log_mu).all())
        self.assertTrue(torch.isfinite(entropy(log_q)).all())


class StructuralTests(unittest.TestCase):
    def setUp(self):
        self.g = torch.Generator().manual_seed(7)

    def unit(self, n, d=8, dtype=torch.float64):
        return torch.nn.functional.normalize(torch.randn(n, d, generator=self.g, dtype=dtype), dim=-1)

    def test_backends_match_dense_reference_values_and_gradients(self):
        for n, near in ((11, False), (37, False), (37, True)):
            a = self.unit(n)
            b = (a + 1e-4 * self.unit(n)) if near else self.unit(n)
            b = torch.nn.functional.normalize(b, dim=-1)
            reference_input = a.clone().requires_grad_()
            reference = structural_dense_reference(reference_input, b)
            reference.backward()
            for name, function in (("contraction", structural_uniform_contraction),
                                   ("tiled", lambda x, y: structural_uniform_tiled(x, y, tile=8))):
                with self.subTest(n=n, near=near, backend=name):
                    x = a.clone().requires_grad_()
                    value = function(x, b)
                    value.backward()
                    torch.testing.assert_close(value, reference.detach(), atol=1e-12, rtol=1e-6)
                    torch.testing.assert_close(x.grad, reference_input.grad, atol=1e-12, rtol=1e-6)

    def test_float32_contraction_error_near_equality(self):
        # Production FP32 against an FP64 dense reference, including near-equal features.
        for scale in (1.0, 1e-2, 1e-3):
            a = self.unit(4096, 64)
            b = torch.nn.functional.normalize(a + scale * self.unit(4096, 64), dim=-1)
            reference = structural_dense_reference(a, b)
            value = structural_uniform_contraction(a.float(), b.float()).double()
            relative = float((value - reference).abs() / reference)
            self.assertLess(relative, 2e-3, f"scale={scale}: relative error {relative:.2e}")

    def test_fewer_than_two_tokens_give_connected_zero(self):
        for n in (0, 1):
            a = self.unit(n).requires_grad_()
            for function in (structural_uniform_contraction, structural_uniform_tiled):
                value = function(a, self.unit(n))
                self.assertEqual(float(value), 0.0)
                value.backward()

    def test_distance_bands(self):
        coords = torch.tensor([[0, 0, 0], [0, 0, 2], [0, 0, 3], [0, 0, 9], [0, 1, 1]])
        metric, ell = distance_metric((7.91,) * 3, (8,) * 3)
        self.assertEqual(ell, 8 * 7.91)
        self.assertTrue(torch.equal(metric, torch.ones(3, dtype=torch.float64)))
        counts = structural_band_counts(coords, metric, [2.0, 8.0])
        # Ordered pairs: distance exactly 2 is "near" (inclusive edge).
        distances = torch.cdist(coords.double(), coords.double())
        off = ~torch.eye(5, dtype=torch.bool)
        expected = torch.tensor([((distances <= 2) & off).sum(), ((distances > 2) & (distances <= 8)).sum(),
                                 (distances > 8).sum()], dtype=torch.float64)
        torch.testing.assert_close(counts, expected)
        # Anisotropic spacing changes membership through physical distances.
        metric, ell = distance_metric((1.0, 1.0, 4.0), (8,) * 3)
        self.assertAlmostEqual(ell, 8 * 4 ** (1 / 3))
        anisotropic = structural_band_counts(coords, metric, [2.0, 8.0])
        self.assertEqual(float(anisotropic.sum()), 20.0)
        self.assertFalse(torch.equal(anisotropic, counts))

    def test_stratified_weighting(self):
        n = 30
        coords = torch.randint(0, 12, (n, 3), generator=self.g)
        coords = torch.unique(coords, dim=0)
        n = len(coords)
        a, b = self.unit(n), self.unit(n)
        metric, _ = distance_metric((1.0,) * 3, (8,) * 3)
        edges = [2.0, 8.0]
        counts = structural_band_counts(coords, metric, edges, tile=7)
        # Weights proportional to pair counts recover the uniform loss and gradient.
        x = a.clone().requires_grad_()
        value, _, _ = structural_distance_stratified(x, b, coords, metric, edges, counts.tolist(), tile=7)
        value.backward()
        y = a.clone().requires_grad_()
        uniform = structural_dense_reference(y, b)
        uniform.backward()
        torch.testing.assert_close(value, uniform.detach(), atol=1e-12, rtol=1e-6)
        torch.testing.assert_close(x.grad, y.grad, atol=1e-12, rtol=1e-6)
        # Equal weights: each populated band contributes its mean error times 1/3.
        value, counts, weights = structural_distance_stratified(a, b, coords, metric, edges, [1, 1, 1], tile=7)
        error = (a @ a.T - b @ b.T).square()
        distances = torch.cdist(coords.double(), coords.double())
        off = ~torch.eye(n, dtype=torch.bool)
        bands = [(distances <= 2) & off, (distances > 2) & (distances <= 8), distances > 8]
        populated = [band for band in bands if band.any()]
        expected = sum(error[band].mean() for band in populated) / len(populated)
        torch.testing.assert_close(value, expected, atol=1e-12, rtol=1e-6)
        # Band membership is invariant to permuting the student tokens (independent orientations).
        order = torch.randperm(n, generator=self.g)
        permuted, _, _ = structural_distance_stratified(a[order], b[order], coords[order], metric, edges, [1, 1, 1], tile=5)
        torch.testing.assert_close(permuted, value, atol=1e-12, rtol=1e-6)

    def test_empty_bands_and_zero_weights(self):
        coords = torch.tensor([[0, 0, 0], [0, 0, 1]])
        counts = structural_band_counts(coords, torch.ones(3, dtype=torch.float64), [2.0, 8.0])
        torch.testing.assert_close(counts, torch.tensor([2., 0., 0.], dtype=torch.float64))
        weights = band_pair_weights(counts, [0.0, 1.0, 1.0])
        self.assertEqual(float(weights.sum()), 0.0)
        a = self.unit(2).requires_grad_()
        value, _, _ = structural_distance_stratified(a, self.unit(2), coords, torch.ones(3, dtype=torch.float64),
                                                     [2.0, 8.0], [0.0, 1.0, 1.0])
        self.assertEqual(float(value), 0.0)
        value.backward()

    def test_nonaffine_target(self):
        features = torch.randn(5, 16, generator=self.g) * 3 + 2
        target = nonaffine_target(features)
        torch.testing.assert_close(target.norm(dim=-1), torch.ones(5))


class ScheduleTests(unittest.TestCase):
    def test_endpoints(self):
        self.assertEqual(global_collapse_weight(0, 12500), 1.0)
        self.assertAlmostEqual(global_collapse_weight(12500, 12500), 0.1)
        self.assertAlmostEqual(global_collapse_weight(500000, 12500), 0.1)
        self.assertEqual(structural_weight(0, 12500, 500000), 0.0)
        self.assertAlmostEqual(structural_weight(12500, 12500, 500000), 0.05)
        self.assertAlmostEqual(structural_weight(12501, 12500, 500000), 0.05 + 0.95 / 487500)
        self.assertAlmostEqual(structural_weight(500000, 12500, 500000), 1.0)
        self.assertAlmostEqual(structural_weight(6250, 12500, 500000), 0.025)
        with self.assertRaises(ValueError):
            structural_weight(0, 100, 100)


if __name__ == "__main__":
    unittest.main()
