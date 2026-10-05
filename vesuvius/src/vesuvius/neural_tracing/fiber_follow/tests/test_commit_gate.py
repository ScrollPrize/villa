"""Commit gates: 'full' (default: n_commit points iff last-plane confidence passes) and 'prefix' (the earlier rule)."""
from types import SimpleNamespace

import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.tracing.policy import (OperatingPolicy, checkpoint_policy, commit_count,
                                                                 commit_prefix, selection_window)
from vesuvius.neural_tracing.fiber_follow.tracing.trace import TraceParams


def proposal(first=(0., 0., 1.)):
    points = torch.zeros(1, 16, 3)
    points[..., 2] = torch.arange(1., 17.)
    points[:, 0] = torch.tensor(first)
    return points


def test_full_gate_commits_n_points_only_when_the_last_plane_passes():
    falling = torch.linspace(.9, .3, 16)[None]  # passes 0.4 for the first planes only
    assert commit_count(proposal(), falling, .4, 8, 6., 'full')[0].item() == 0
    assert commit_count(proposal(), falling, .4, 8, 6., 'prefix')[0].item() == 8  # prefix commits the confident part
    assert commit_count(proposal(), falling, .25, 8, 6., 'full')[0].item() == 8
    # The recovery limit still blocks a far first connection.
    counts, allowed = commit_count(proposal((9., 0., 1.)), torch.ones(1, 16), .4, 8, 6., 'full')
    assert counts.item() == 0 and not allowed.item()
    torch.testing.assert_close(commit_count(proposal(), falling, .5, 16, 6., 'prefix')[0],
                               commit_prefix(proposal(), falling, .5, 16, 6.)[0].to(torch.int64), check_dtype=False)
    assert selection_window(8, 16, 'full') == 16 and selection_window(8, 16, 'prefix') == 8


def test_defaults_and_legacy_records():
    assert (OperatingPolicy().confidence, OperatingPolicy().n_commit, OperatingPolicy().gate) == (.4, 8, 'full')
    assert TraceParams.from_policy(OperatingPolicy(gate='prefix')).gate == 'prefix'
    cfg = SimpleNamespace(n_future=16, max_recovery_distance=6., recurrent_refinement_steps=3)
    legacy = dict(operating_policy=dict(confidence=.5, n_commit=16, max_recovery_distance=6., refinement_steps=3))
    # Policies recorded before gates existed keep their prefix semantics (e.g. collectors of runs started earlier).
    assert checkpoint_policy(legacy, cfg).gate == 'prefix'
    assert checkpoint_policy(legacy, cfg, gate='full', confidence=.4, n_commit=8) == OperatingPolicy(
        confidence=.4, n_commit=8, max_recovery_distance=6., refinement_steps=3, gate='full')
    with pytest.raises(ValueError):
        OperatingPolicy(gate='other')


def test_resuming_a_run_from_before_gates_requires_its_prefix_policy():
    from model_fixtures import REQUIRED
    from vesuvius.neural_tracing.fiber_follow.train.train import build_parser, validate_resume_options
    recorded = {k: v for k, v in vars(build_parser().parse_args(REQUIRED)).items() if k not in ('gate', 'trace_confidence')}
    with pytest.raises(ValueError, match='gate|trace_confidence'):
        validate_resume_options(build_parser().parse_args(REQUIRED), recorded)
    validate_resume_options(build_parser().parse_args(REQUIRED+['--gate', 'prefix', '--trace-confidence', '.5']), recorded)
