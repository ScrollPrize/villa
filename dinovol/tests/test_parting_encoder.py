"""Parting's encoder on a tiny random dinovol checkpoint: no network, no GPU, no released weights."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from dinovol_2.eval.parting import encoder as E
from dinovol_2.model.model import DinoVitStudentTeacher


def tiny_config():
    return {"model_type": "v3", "embed_dim": 24, "depth": 2, "num_heads": 2, "patch_size": [4, 4, 4],
            "global_crops_size": [16, 16, 16], "local_crops_size": [8, 8, 8], "drop_path_rate": 0.0,
            "embedding_type": "deeper", "deeper_embed_stage_channels": [4, 8, 16],
            "deeper_embed_stage_strides": [1, 2, 2], "deeper_embed_depth_per_level": [1, 1, 1],
            "deeper_embed_use_stem": True, "deeper_embed_stem_channels": 4, "deeper_embed_stem_type": "conv",
            "deeper_embed_final_projection": True, "deeper_embed_batch_chunk_size": None,
            "rope_type": "mixed", "num_reg_tokens": 4}


def write_checkpoint(folder):
    torch.manual_seed(0)
    net = DinoVitStudentTeacher._build_backbone(tiny_config())
    path = Path(folder) / "tiny.pt"
    torch.save({"config": {"model": tiny_config()}, "step": 0,
                "teacher": {f"backbone.{k}": v for k, v in net.state_dict().items()}}, path)
    return path, net.eval()


class EncoderTests(unittest.TestCase):
    def test_tokens_match_the_backbone_built_directly(self):
        with tempfile.TemporaryDirectory() as td:
            path, net = write_checkpoint(td)
            with mock.patch.object(E, "DV_SHA", E.sha256_file(path)):
                enc = E.Encoder("cpu", str(path))
            ct = np.random.default_rng(0).integers(0, 255, (16, 16, 16)).astype(np.uint8)
            tok = enc.tokens(ct)
            self.assertEqual(tok.shape, (24, 4, 4, 4))
            x = torch.from_numpy(E.normalize(ct))[None, None]
            with torch.no_grad():
                ref = net.forward_features(x, view_kind="global")["x_norm_patchtokens"][0]
            np.testing.assert_array_equal(tok, ref.T.reshape(24, 4, 4, 4).numpy())
            self.assertEqual(enc.record()["code"]["path"], "dinovol")

    def test_a_checkpoint_with_another_hash_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            path, _ = write_checkpoint(td)
            with self.assertRaises(SystemExit) as cm:
                E.Encoder("cpu", str(path))
            self.assertIn("is not the pinned", str(cm.exception))

    def test_a_checkpoint_with_pickled_code_is_not_run(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "code.pt"
            torch.save({"config": {"model": tiny_config()}, "teacher": {}, "hook": mock.sentinel}, path)
            with mock.patch.object(E, "DV_SHA", E.sha256_file(path)):
                with self.assertRaises(Exception) as cm:
                    E.Encoder("cpu", str(path))
            self.assertNotIsInstance(cm.exception, SystemExit)

    def test_normalize_is_the_robust_clip_median_mad_and_leaves_its_input_alone(self):
        rng = np.random.default_rng(2)
        for ct in (rng.integers(0, 255, (16, 16, 16)).astype(np.uint8),
                   rng.normal(100, 20, (16, 16, 16)).astype(np.float32)):
            before = ct.copy()
            got = E.normalize(ct)
            np.testing.assert_array_equal(ct, before)
            v = ct.astype(np.float32)
            v = np.clip(v, np.percentile(v, 1), np.percentile(v, 99))
            med = np.median(v)
            want = (v - med) / (1.4826 * np.median(np.abs(v - med)))
            self.assertEqual(got.dtype, np.float32)
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
