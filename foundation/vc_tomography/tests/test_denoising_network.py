"""networks/denoising: the network's tiling and the spool's bookkeeping. Needs PyTorch (`uv sync --extra denoise`; skipped
without it); on the host, small sizes."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

try:
    import torch
    import torch.nn.functional as F
except ImportError as e:      # (not there, or another Python's build of it on the path)
    pytest.skip("PyTorch cannot be imported here: %s" % str(e).splitlines()[0], allow_module_level=True)

DENOISING = Path(__file__).resolve().parents[1] / "networks" / "denoising"


def load(name):
    spec = importlib.util.spec_from_file_location(name, DENOISING / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


n2i = load("noise2inverse")
sp = load("denoise_spool")


def random_net(scales=2, width=4, seed=0):
    torch.manual_seed(seed)
    net = n2i.UNet3d(width, scales)
    torch.nn.init.normal_(net.out.weight, std=0.5)      # (a new network's last layer is 0: it gives its input back)
    net.amp = "bf16"
    return net.eval()


def test_new_network_gives_its_input_back():
    x = torch.randn(1, 1, 8, 16, 16)
    with torch.no_grad():
        assert torch.equal(n2i.UNet3d(4, 2)(x), x)


def test_tiles_agree_with_the_whole_volume_away_from_its_ends():
    # the halo (16) is wider than the network's reach, so a tile's kept part is what the whole volume gives
    net = random_net()
    v = torch.randn(8, 80, 72)
    with torch.no_grad():
        whole = net(v[None, None] * n2i.SCALE)[0, 0] / n2i.SCALE
    tiled = n2i.run_tiled(net, v, tile=32, halo=16)
    assert tiled.shape == v.shape
    assert float((whole - v).abs().max()) > 0.01        # the network does change its input
    np.testing.assert_allclose(tiled[:, 16:-16, 16:-16].numpy(), whole[:, 16:-16, 16:-16].numpy(), atol=1e-5)


def test_spool_slab_is_run_tiled_in_the_window():
    net = random_net(seed=1)
    v = torch.randn(8, 80, 72) * 0.1
    lo, hi = -0.2, 0.3
    tiled = n2i.run_tiled(net, v, tile=32, halo=16).numpy()
    want = np.clip((tiled[2:6] - lo) / (hi - lo), 0, 1) * 255
    got = sp.slab_to_uint8(net, v.numpy(), 2, 6, lo, hi, torch.device("cpu"), n2i, torch, F, tile=32, halo=16)
    assert got.dtype == np.uint8 and got.shape == (4, 80, 72)
    assert np.abs(got.astype(np.float64) - want).max() <= 0.5 + 1e-3


def test_saved_network_loads_with_its_size_and_the_first_models_layer_names(tmp_path):
    net = random_net(scales=3)
    x = torch.randn(1, 1, 8, 16, 16)
    with torch.no_grad():
        want = net(x)
    torch.save(dict(state=net.state_dict(), width=4, scales=3, amp="bf16"), tmp_path / "model.pt")
    loaded = n2i.load_net(tmp_path / "model.pt", torch.device("cpu"))
    assert loaded.amp == "bf16" and not loaded.training
    with torch.no_grad():
        assert torch.equal(loaded(x), want)
    # the first models: three scales under the names e1, e2, mid, d1, d2 and no `scales`
    old = {"enc.0": "e1", "enc.1": "e2", "enc.2": "mid", "dec.0": "d1", "dec.1": "d2"}
    state = {(old[k[:5]] + k[5:] if k[:5] in old else k): v for k, v in net.state_dict().items()}
    torch.save(dict(state=state, width=4), tmp_path / "old.pt")
    with torch.no_grad():
        assert torch.equal(n2i.load_net(tmp_path / "old.pt", torch.device("cpu"))(x), want)


def test_spool_files_are_whole_and_in_order(tmp_path):
    sp.save(tmp_path / "recon_j00016_00032.npy", np.ones((16, 2, 2), np.float32))
    sp.save(tmp_path / "recon_j00000_00016.npy", np.zeros((16, 2, 2), np.float32))
    np.save(tmp_path / "recon_j00032_00048.tmp.npy", np.zeros(1))       # one still being written
    sp.save(tmp_path / "den_j00000_00008.npy", np.zeros((8, 2, 2), np.uint8))
    assert [(a, b) for a, b, _ in sp.spool_files(tmp_path, "recon")] == [(0, 16), (16, 32)]
    assert [(a, b) for a, b, _ in sp.spool_files(tmp_path, "den")] == [(0, 8)]
    assert sorted(f.name for f in tmp_path.glob("*.tmp*")) == ["recon_j00032_00048.tmp.npy"]      # save left none of its own
