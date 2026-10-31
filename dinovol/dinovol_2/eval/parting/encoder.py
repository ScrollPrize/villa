"""Base dinovol tokens for a CT window, from the released `scrollprize/dinovol_ps8_9.6um_101626` checkpoint.

The checkpoint (`teacher_305k.pt`) is downloaded from Hugging Face at run time, pinned by revision and sha256, and
loaded with `torch.load(weights_only=True)`, so no pickled code in it runs. The backbone is built from the
checkpoint's own model config by this package's `DinoVitStudentTeacher._build_backbone`, and each window is
normalized with `normalize_robust` (percentile clip 1 to 99, then median and 1.4826 MAD), the dataset default the
checkpoint was pretrained with: its config names no `normalization_scheme`.
"""
import hashlib
import subprocess
from pathlib import Path

import numpy as np

DV_REPO = "scrollprize/dinovol_ps8_9.6um_101626"
DV_REV = "02164a5b60bc948fad038617cd7d5c5255fa4333"
DV_FILE = "teacher_305k.pt"
DV_SHA = "875306424046f69ac2767b1447ea99f9dbc6a9c7d28147d8a03ce0346e00dd77"
STRIDE = 8              # the encoder's patch size: token i is centered at voxel STRIDE * i + (STRIDE - 1) / 2


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def fetch_checkpoint(path=None):
    """The pinned base dinovol checkpoint: PATH when given (checked), else downloaded from the Hub (cached)."""
    if path:
        got = sha256_file(path)
        if got != DV_SHA:
            raise SystemExit(f"{path}: sha256 {got} is not the pinned {DV_SHA} ({DV_REPO}/{DV_FILE})")
        return str(path)
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(DV_REPO, DV_FILE, revision=DV_REV)
    got = sha256_file(p)
    if got != DV_SHA:
        raise SystemExit(f"{DV_REPO}/{DV_FILE}@{DV_REV}: sha256 {got} is not the pinned {DV_SHA}")
    return p


def build_backbone(checkpoint):
    """The teacher backbone of a dinovol checkpoint dict ({"config", "teacher"}), all weights loaded, in eval mode."""
    from dinovol_2.model.model import DinoVitStudentTeacher, _upgrade_weight_norm_state_dict_keys
    net = DinoVitStudentTeacher._build_backbone(checkpoint["config"]["model"])
    state = {k[len("backbone."):]: v for k, v in checkpoint["teacher"].items() if k.startswith("backbone.")}
    net.load_state_dict(_upgrade_weight_norm_state_dict_keys(state), strict=True)
    net.return_embed_skips = False
    return net.eval()


def code_commit():
    """The git commit of this checkout's dinovol code, or None outside a git checkout."""
    try:
        out = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def normalize(ct):
    """A CT window as the encoder saw its pretraining crops: float32, robust-normalized, the input left unchanged."""
    from dinovol_2.dataset.normalization import normalize_robust
    return normalize_robust(np.array(ct))


class Encoder:
    """Base dinovol on one device: tokens(ct) gives the (288, 16, 16, 16) token grid of a 128^3 uint8 window."""

    def __init__(self, device="cpu", checkpoint=None, threads=None):
        import torch
        if threads:
            torch.set_num_threads(int(threads))
        if device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--device cuda: no CUDA GPU here; use --device cpu")
        self.torch = torch
        ck = torch.load(fetch_checkpoint(checkpoint), map_location="cpu", weights_only=True)
        self.device = device
        self.net = build_backbone(ck).to(device)
        self.bf16 = device == "cuda"

    def tokens(self, ct):
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(normalize(ct), np.float32))[None, None].to(self.device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16):
            tok = self.net.forward_features(x, view_kind="global")["x_norm_patchtokens"]
        g = round(tok.shape[1] ** (1 / 3))
        return tok[0].float().T.reshape(-1, g, g, g).cpu().numpy()

    def record(self):
        return {"repo": DV_REPO, "rev": DV_REV, "file": DV_FILE, "sha256": DV_SHA,
                "code": {"repo": "https://github.com/ScrollPrize/villa", "path": "dinovol", "commit": code_commit()},
                "device": self.device, "bf16": self.bf16, "torch": self.torch.__version__}
