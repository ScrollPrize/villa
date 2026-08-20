import argparse

from vesuvius.models.training.cli import _add_training_control_args


def test_schedule_flags_leave_yaml_values_owned_when_omitted():
    parser = argparse.ArgumentParser()
    _add_training_control_args(parser)

    defaults = parser.parse_args([])
    assert defaults.max_epoch is None
    assert defaults.max_steps_per_epoch is None
    assert defaults.max_val_steps_per_epoch is None
    assert defaults.val_every_n is None

    explicit = parser.parse_args([
        "--max-epoch", "50",
        "--max-steps-per-epoch", "150",
        "--max-val-steps-per-epoch", "20",
        "--val-every-n", "5",
    ])
    assert explicit.max_epoch == 50
    assert explicit.max_steps_per_epoch == 150
    assert explicit.max_val_steps_per_epoch == 20
    assert explicit.val_every_n == 5
