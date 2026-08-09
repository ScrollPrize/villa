import torch

from koine_machines.training.loss.losses import create_loss_from_config


def test_koine_half_epsilon_matches_prior_soft_bce_quarter_targets():
    loss = create_loss_from_config(
        {
            "loss": {
                "bce_label_smoothing": 0.5,
                "dice_label_smoothing": 0.0,
            }
        }
    )
    base_loss = loss.terms[0].module

    smoothed = base_loss._smooth_bce_targets(torch.tensor([0.0, 1.0]))

    torch.testing.assert_close(smoothed, torch.tensor([0.25, 0.75]))
