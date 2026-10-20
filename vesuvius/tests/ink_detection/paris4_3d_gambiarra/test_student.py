"""The student network (vesuvius NetworkFromConfig), its deterministic pooling and the native-voxel resampling. CPU."""
import hashlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from torch.nn import functional as F

from vesuvius.ink_detection.paris4_3d_gambiarra import student as S

# sha256 over (name, float32 bytes) of the state dict of NetworkFromConfig(manager()) built under torch.manual_seed(27);
# equal to the network the original student run built (training-time vesuvius runtime).
INIT_SHA256 = 'e577dcd4f1c91bbc62aede0147a39044737e22a55b8f052d90ff71040a51a773'


def state_sha256(network):
    h = hashlib.sha256()
    for k, v in network.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def has_vesuvius():
    try:
        return importlib.util.find_spec('vesuvius.models.build.build_network_from_config') is not None
    except ModuleNotFoundError:
        return False


@unittest.skipUnless(has_vesuvius(), 'needs vesuvius (villa/vesuvius)')
class Network(unittest.TestCase):
    def test_initialization(self):
        rng = torch.random.get_rng_state()
        network = S.build_network(27)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))  # the global RNG is untouched
        self.assertEqual(sum(p.numel() for p in network.parameters()), S.PARAMETERS)
        self.assertEqual(state_sha256(network), INIT_SHA256)
        self.assertEqual(sum(isinstance(m, S.DeterministicPool2) for m in network.modules()), 4)
        self.assertFalse(any(isinstance(m, torch.nn.AvgPool3d) for m in network.modules()))
        with tempfile.TemporaryDirectory() as tmp:
            S.save_weights(network, Path(tmp) / 'w.safetensors')
            with safe_open(Path(tmp) / 'w.safetensors', 'pt') as f:  # release format: no alias copies
                self.assertFalse(any('.all_modules.' in k for k in f.keys()))
            other = S.load_weights(S.build_network(28), Path(tmp) / 'w.safetensors')
            self.assertEqual(state_sha256(other), INIT_SHA256)

    @unittest.skipUnless(os.environ.get('PARIS4_INK_WEIGHTS'), 'set PARIS4_INK_WEIGHTS to a released model.safetensors')
    def test_released_weights(self):
        path = Path(os.environ['PARIS4_INK_WEIGHTS'])
        network = S.load_weights(S.build_network(27), path)
        with safe_open(path, 'pt') as f:
            for k in f.keys():
                self.assertTrue(torch.equal(network.state_dict()[k], f.get_tensor(k)), k)


class Pool(unittest.TestCase):
    def test_matches_avg_pool(self):
        x = torch.randn(2, 3, 8, 6, 4, dtype=torch.float64, requires_grad=True)
        y = S.DeterministicPool2()(x)
        ref = F.avg_pool3d(x, 2, 2)
        self.assertTrue(torch.equal(y, ref))
        grad = torch.randn_like(y)
        a, = torch.autograd.grad(y, x, grad)
        b, = torch.autograd.grad(ref, x, grad)
        self.assertTrue(torch.equal(a, b))


class Resample(unittest.TestCase):
    def test_trilinear_upsampling(self):
        low = torch.randn(2, 1, 64, 64, 64)
        ref = F.interpolate(low, scale_factor=2, mode='trilinear', align_corners=False)
        torch.testing.assert_close(S.resample(low), ref, rtol=0, atol=1e-5)

    def test_window(self):
        low = torch.randn(1, 1, 128, 128, 128)
        full = F.interpolate(low, scale_factor=2, mode='trilinear', align_corners=False)
        torch.testing.assert_close(S.resample(low, start=(64, 0, 32)), full[:, :, 64:192, 0:128, 32:160], rtol=0, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
