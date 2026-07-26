"""Unit tests for the batched three-state pair-HMM soft aligner."""

import math

import pytest
import torch

from soft_alignment import soft_align_sequences


INF = float('inf')


def align(cost, model_valid=None, band_valid=None, **kwargs):
    cost = torch.as_tensor(cost, dtype=torch.float32)
    if cost.dim() == 2:
        cost = cost[None]
    batch, n_model, n_bands = cost.shape
    if model_valid is None:
        model_valid = torch.ones([batch, n_model], dtype=torch.bool)
    if band_valid is None:
        band_valid = torch.ones([batch, n_bands], dtype=torch.bool)
    defaults = dict(missing_open=1.0, missing_extend=1.0, extra_open=1.0,
                    extra_extend=1.0, temperature=0.25,
                    compute_map_path=True)
    defaults.update(kwargs)
    return soft_align_sequences(cost, model_valid, band_valid, **defaults)


def brute_force(cost, *, missing_open, missing_extend, extra_open,
                extra_extend, temperature):
    """Enumerate every monotonic alignment path and its log potential.

    ``extra_open``/``extra_extend`` may be scalars or per-band sequences.
    """
    cost = torch.as_tensor(cost, dtype=torch.float64)
    n_model, n_bands = cost.shape

    def per_band(value, band):
        if isinstance(value, (int, float)):
            return float(value)
        return float(value[band])

    paths = []

    def recurse(i, j, previous, potential, matches):
        if i == n_model and j == n_bands:
            paths.append((potential, tuple(matches)))
            return
        if i < n_model and j < n_bands and math.isfinite(float(cost[i, j])):
            recurse(i + 1, j + 1, 'M',
                    potential - float(cost[i, j]) / temperature,
                    matches + [(i, j)])
        if i < n_model:
            gap = missing_extend if previous == 'X' else missing_open
            recurse(i + 1, j, 'X', potential - gap / temperature, matches)
        if j < n_bands:
            gap = per_band(
                extra_extend if previous == 'Y' else extra_open, j)
            recurse(i, j + 1, 'Y', potential - gap / temperature, matches)

    recurse(0, 0, 'start', 0.0, [])
    potentials = torch.tensor([p for p, _ in paths], dtype=torch.float64)
    log_partition = torch.logsumexp(potentials, dim=0)
    marginal = torch.zeros([n_model, n_bands], dtype=torch.float64)
    for potential, matches in paths:
        weight = math.exp(potential - float(log_partition))
        for i, j in matches:
            marginal[i, j] += weight
    return float(log_partition), marginal


def shifted_stack_cost(n_model, n_bands, offsets, scale=1.0):
    """Quadratic cost between winding positions i and band positions."""
    model = torch.arange(n_model, dtype=torch.float32)
    bands = torch.as_tensor(offsets, dtype=torch.float32)
    return scale * (model[:, None] - bands[None, :]).abs()


class TestAgainstBruteForce:
    @pytest.mark.parametrize('seed', [0, 1, 2, 3])
    def test_log_partition_and_marginals_match_enumeration(self, seed):
        generator = torch.Generator().manual_seed(seed)
        n_model, n_bands = 3, 4
        cost = torch.rand([n_model, n_bands], generator=generator) * 3.0
        cost[torch.rand([n_model, n_bands], generator=generator) < 0.2] = INF
        params = dict(missing_open=0.8, missing_extend=0.8, extra_open=1.1,
                      extra_extend=1.1, temperature=0.35)
        result = align(cost, **params)
        expected_log_z, expected_marginal = brute_force(cost, **params)
        assert float(result['log_partition'][0]) == pytest.approx(
            expected_log_z, abs=1e-4)
        torch.testing.assert_close(
            result['match_posterior'][0].double(), expected_marginal,
            atol=1e-4, rtol=1e-3)

    def test_affine_costs_match_enumeration(self):
        cost = torch.tensor([
            [0.2, INF, 2.0],
            [INF, 0.4, 1.5],
            [1.0, INF, 0.3],
            [INF, 2.0, 0.8],
        ])
        params = dict(missing_open=1.4, missing_extend=0.3, extra_open=1.2,
                      extra_extend=0.4, temperature=0.5)
        result = align(cost, **params)
        expected_log_z, expected_marginal = brute_force(cost, **params)
        assert float(result['log_partition'][0]) == pytest.approx(
            expected_log_z, abs=1e-4)
        torch.testing.assert_close(
            result['match_posterior'][0].double(), expected_marginal,
            atol=1e-4, rtol=1e-3)


class TestAlignmentBehavior:
    def test_perfect_diagonal_has_near_unit_diagonal_marginal(self):
        cost = shifted_stack_cost(5, 5, [0, 1, 2, 3, 4], scale=4.0)
        result = align(cost, temperature=0.1)
        posterior = result['match_posterior'][0]
        assert float(posterior.diagonal().min()) > 0.98
        assert float(result['expected_missing'][0]) < 0.05
        assert float(result['expected_extra'][0]) < 0.05
        assert (result['map_match'][0] == torch.arange(5)).all()

    def test_one_missing_band_resumes_matching_afterwards(self):
        # Bands for windings 0, 1, 3, 4 - winding 2's band is missing.
        cost = shifted_stack_cost(5, 4, [0, 1, 3, 4], scale=4.0)
        result = align(cost, temperature=0.1)
        expected = torch.tensor([0, 1, -1, 2, 3])
        assert (result['map_match'][0] == expected).all()
        posterior = result['match_posterior'][0]
        assert float(posterior[3, 2]) > 0.9
        assert float(posterior[4, 3]) > 0.9
        assert float(result['missing_posterior'][0, 2]) > 0.9

    def test_one_extra_band_is_skipped_and_matching_resumes(self):
        # An extra band at 1.5 sits between windings 1 and 2.
        cost = shifted_stack_cost(4, 5, [0, 1, 1.5, 2, 3], scale=6.0)
        result = align(cost, temperature=0.1)
        expected = torch.tensor([0, 1, 3, 4])
        assert (result['map_match'][0] == expected).all()
        assert bool(result['map_extra'][0, 2])
        assert float(result['interior_extra_posterior'][0, 2]) > 0.5

    def test_consecutive_missing_and_extra_stay_ordered_and_finite(self):
        cost = shifted_stack_cost(6, 6, [0, 0.4, 0.8, 4.0, 4.4, 5.0],
                                  scale=3.0)
        result = align(cost)
        assert torch.isfinite(result['log_partition']).all()
        matched = result['map_match'][0]
        previous = -1
        for value in matched.tolist():
            if value >= 0:
                assert value > previous
                previous = value

    def test_crossed_and_many_to_one_assignments_are_impossible(self):
        cost = torch.zeros([3, 3])
        result = align(cost, temperature=0.5)
        posterior = result['match_posterior'][0]
        # Monotonicity: each band's total match mass is at most one, and the
        # joint posterior of a crossing pair is zero by construction.
        assert float(posterior.sum(dim=0).max()) <= 1.0 + 1e-5
        assert float(posterior.sum(dim=1).max()) <= 1.0 + 1e-5

    def test_half_winding_ambiguity_is_low_margin_not_averaged(self):
        # One winding equidistant from two bands: the marginal splits and the
        # top-2 margin flags it instead of inventing a confident midpoint.
        cost = torch.tensor([[1.0, 1.0]])
        result = align(cost, temperature=0.5, missing_open=3.0,
                       missing_extend=3.0)
        posterior = result['match_posterior'][0]
        assert float((posterior[0, 0] - posterior[0, 1]).abs()) < 1e-4
        assert float(result['top2_margin'][0, 0]) < 0.05
        assert float(result['winding_entropy'][0, 0]) > 0.5

    def test_low_temperature_approaches_map(self):
        cost = shifted_stack_cost(4, 4, [0.1, 1.2, 1.9, 3.05], scale=2.0)
        result = align(cost, temperature=0.01)
        posterior = result['match_posterior'][0]
        assert float(posterior.diagonal().min()) > 0.999
        assert float(result['map_clearance'][0]) > 5.0

    def test_leading_and_trailing_padding_bands_consume_free_end_gaps(self):
        # Padding bands (outside the central interval) carry zero skip cost;
        # they must be consumed by the free end gaps without changing the
        # core alignment and without attracting bogus end matches.
        core = shifted_stack_cost(3, 3, [0, 1, 2], scale=4.0)
        padded = torch.full([3, 7], INF)
        padded[:, 2:5] = core
        extra_cost = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
        result_core = align(core, temperature=0.1)
        result_padded = align(padded, temperature=0.1,
                              extra_open=extra_cost, extra_extend=extra_cost)
        torch.testing.assert_close(
            result_padded['match_posterior'][0, :, 2:5],
            result_core['match_posterior'][0], atol=1e-4, rtol=1e-4)
        # Padding bands are consumed by end gaps, not by bogus end matches.
        assert float(result_padded['match_posterior'][0, :, :2].sum()) < 1e-6
        assert float(result_padded['match_posterior'][0, :, 5:].sum()) < 1e-6
        assert float(result_padded['log_partition'][0]) == pytest.approx(
            float(result_core['log_partition'][0]), abs=1e-3)

    def test_per_band_costs_match_enumeration(self):
        cost = torch.tensor([
            [0.3, INF, 1.0, 2.0],
            [1.0, 0.2, INF, 1.5],
            [2.0, 1.0, 0.4, 0.6],
        ])
        extra_open = [0.0, 1.2, 0.8, 0.0]
        extra_extend = [0.0, 0.5, 0.9, 0.0]
        params = dict(missing_open=0.9, missing_extend=0.4, temperature=0.3)
        result = align(cost, extra_open=torch.tensor([extra_open]),
                       extra_extend=torch.tensor([extra_extend]), **params)
        expected_log_z, expected_marginal = brute_force(
            cost, extra_open=extra_open, extra_extend=extra_extend, **params)
        assert float(result['log_partition'][0]) == pytest.approx(
            expected_log_z, abs=1e-4)
        torch.testing.assert_close(
            result['match_posterior'][0].double(), expected_marginal,
            atol=1e-4, rtol=1e-3)

    def test_banded_out_cells_receive_zero_probability_mass(self):
        cost = torch.zeros([3, 3])
        cost[0, 2] = INF
        cost[2, 0] = INF
        result = align(cost, temperature=0.5)
        posterior = result['match_posterior'][0]
        assert float(posterior[0, 2]) == 0.0
        assert float(posterior[2, 0]) == 0.0

    def test_untied_affine_prefers_one_contiguous_missing_run(self):
        # One band, three candidate windings: matching the middle winding
        # forces two separated single-winding gaps, matching either end
        # winding forces one contiguous two-winding run. Tied gap costs make
        # the three alignments equal (the transition-matrix pair-HMM
        # reproduces the constant-cost alignment); untied costs with cheap
        # extension prefer the contiguous runs.
        cost = torch.zeros([3, 1])
        tied = align(cost, missing_open=1.0, missing_extend=1.0,
                     temperature=0.5, extra_open=5.0, extra_extend=5.0)
        affine = align(cost, missing_open=1.5, missing_extend=0.25,
                       temperature=0.5, extra_open=5.0, extra_extend=5.0)
        tied_posterior = tied['match_posterior'][0]
        assert float(tied_posterior[1, 0]) == pytest.approx(
            float(tied_posterior[0, 0]), abs=1e-4)
        assert float(tied_posterior[1, 0]) == pytest.approx(
            float(tied_posterior[2, 0]), abs=1e-4)
        affine_posterior = affine['match_posterior'][0]
        assert float(affine_posterior[0, 0]) > float(affine_posterior[1, 0])
        assert float(affine_posterior[2, 0]) > float(affine_posterior[1, 0])

    def test_empty_and_degenerate_inputs_are_defined(self):
        no_bands = align(torch.zeros([2, 0]))
        assert torch.isfinite(no_bands['log_partition']).all()
        assert float(no_bands['expected_matches'][0]) == 0.0
        assert float(no_bands['missing_posterior'].sum()) == pytest.approx(
            2.0, abs=1e-5)

        all_invalid = align(torch.full([3, 3], INF))
        assert torch.isfinite(all_invalid['log_partition']).all()
        assert float(all_invalid['match_posterior'].sum()) == 0.0

        one_band = align(torch.tensor([[0.1], [2.0], [3.0]]))
        assert torch.isfinite(one_band['log_partition']).all()
        assert float(one_band['match_posterior'].sum()) <= 1.0 + 1e-5

        masked = align(
            torch.zeros([2, 3]),
            model_valid=torch.tensor([[True, False]]),
            band_valid=torch.tensor([[True, True, False]]))
        assert torch.isfinite(masked['log_partition']).all()
        assert float(masked['match_posterior'][0, 1, :].sum()) == 0.0
        assert float(masked['match_posterior'][0, :, 2].sum()) == 0.0

    def test_variable_lengths_in_one_batch(self):
        cost = torch.full([2, 4, 5], INF)
        cost[0, :4, :4] = shifted_stack_cost(4, 4, [0, 1, 2, 3], scale=4.0)
        cost[1, :2, :3] = shifted_stack_cost(2, 3, [0, 0.5, 1], scale=4.0)
        model_valid = torch.tensor([[True] * 4, [True, True, False, False]])
        band_valid = torch.tensor(
            [[True, True, True, True, False], [True, True, True, False,
                                               False]])
        result = align(cost, model_valid=model_valid, band_valid=band_valid,
                       temperature=0.1)
        assert torch.isfinite(result['log_partition']).all()
        assert (result['map_match'][0, :4] == torch.arange(4)).all()
        assert (result['map_match'][1, 2:] == -1).all()
        assert float(result['match_posterior'][1, 2:, :].sum()) == 0.0


class TestGradients:
    def test_gradients_point_toward_selected_band(self):
        # Bands at integers; model targets shifted +0.3: the expected-cost
        # gradient with respect to the shift must be positive (pull back).
        shift = torch.tensor(0.3, requires_grad=True)
        model = torch.arange(4, dtype=torch.float32) + shift
        bands = torch.arange(4, dtype=torch.float32)
        cost = (model[:, None] - bands[None, :]).square()
        result = align(cost, temperature=0.2)
        posterior = result['match_posterior'][0]
        expected_cost = (posterior * cost[None][0]).sum()
        expected_cost.backward()
        assert torch.isfinite(shift.grad)
        assert float(shift.grad) > 0.0

    def test_gradcheck_log_partition_and_marginals(self):
        torch.manual_seed(0)
        base = torch.rand([2, 3, 4], dtype=torch.float64) * 2.0

        def wrapped(cost):
            result = soft_align_sequences(
                cost,
                torch.ones([2, 3], dtype=torch.bool),
                torch.ones([2, 4], dtype=torch.bool),
                missing_open=0.9, missing_extend=0.9, extra_open=1.1,
                extra_extend=1.1, temperature=0.4)
            return result['log_partition'], result['match_posterior']

        assert torch.autograd.gradcheck(
            wrapped, (base.clone().requires_grad_(True),), atol=1e-6)

    def test_long_sequences_stay_finite(self):
        torch.manual_seed(1)
        cost = torch.rand([4, 40, 48]) * 5.0
        cost[torch.rand([4, 40, 48]) < 0.5] = INF
        result = align(cost, temperature=0.1)
        assert torch.isfinite(result['log_partition']).all()
        assert torch.isfinite(result['match_posterior']).all()
        total = (result['match_posterior'].sum(dim=2)
                 + result['missing_posterior'])
        torch.testing.assert_close(
            total, torch.ones_like(total), atol=1e-3, rtol=1e-3)
