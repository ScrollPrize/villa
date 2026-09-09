"""The shared feature API preserves the existing canonical inference path."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_resnet3d_3d_decoder import Decoder3DUNet  # noqa: E402


def test_feature_api_preserves_eval_and_auxiliary_training_outputs():
    torch.manual_seed(9)
    decoder = Decoder3DUNet(encoder_dims=(8, 16, 32, 64), decoder_dims=(4, 8, 16, 32))
    features = [torch.randn(2, c, s, s, s) for c, s in zip((8, 16, 32, 64), (8, 4, 2, 1))]
    decoder.eval()
    decoded, auxiliary = decoder.forward_features(features)
    assert not auxiliary
    expected = decoder.logit(decoder.depth_collapse(decoded))
    torch.testing.assert_close(decoder(features), expected, atol=0, rtol=0)
    decoder.train()
    decoded, auxiliary = decoder.forward_features(features)
    actual, actual_auxiliary = decoder(features)
    torch.testing.assert_close(actual, decoder.logit(decoder.depth_collapse(decoded)), atol=0, rtol=0)
    assert len(auxiliary) == len(actual_auxiliary) == 2
    for a, b in zip(auxiliary, actual_auxiliary):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
