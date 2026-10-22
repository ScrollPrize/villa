import copy
import tempfile
import unittest

import torch

from dinovol_2.model.rope import CompassRopePositionEmbedding, MixedRopePositionEmbedding
from dinovol_2.training.trainer import Trainer
from test_training import tiny_batch, tiny_config


class CompassConversionTests(unittest.TestCase):
    def test_compass_reproduces_mixed_phases(self):
        torch.manual_seed(0)
        for ndim, head_dim, use_dft in ((3, 24, True), (3, 24, False), (2, 16, True)):
            mixed = MixedRopePositionEmbedding(head_dim, ndim=ndim, num_heads=3)
            compass = CompassRopePositionEmbedding(head_dim, ndim=ndim, num_heads=3, use_dft=use_dft)
            compass.set_frequencies(mixed.mix_frequencies.detach())
            coords = torch.rand(50, ndim) * 2 - 1
            for got, want in zip(compass.get_embed_from_coords(coords), mixed.get_embed_from_coords(coords)):
                torch.testing.assert_close(got, want, rtol=0, atol=2e-5)


class ResumeAsCompassWithQkNormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_mixed_checkpoint_resumes_as_compass_with_calibrated_qk_norms(self):
        with tempfile.TemporaryDirectory() as folder:
            torch.manual_seed(0)
            config = tiny_config(folder)
            config["model"].update(rope_type="mixed", qk_norm=False)
            old = Trainer(config)
            batch = tiny_batch()
            old.train_step(batch, 0)
            path = old.save_checkpoint(0)

            new_config = copy.deepcopy(config)
            new_config["model"].update(rope_type="compass", qk_norm=True)
            new = Trainer(new_config)
            new.load_checkpoint(path)

            old_backbone, new_backbone = old.method.model_module.student.backbone, new.method.model_module.student.backbone
            for old_block, new_block in zip(old_backbone.blocks, new_backbone.blocks):
                coords = torch.rand(20, 3) * 2 - 1
                for got, want in zip(new_block.rope_embed.get_embed_from_coords(coords),
                                     old_block.rope_embed.get_embed_from_coords(coords)):
                    torch.testing.assert_close(got, want, rtol=0, atol=2e-5)

            names = dict(new.method.model_module.student.named_parameters())
            old_names = dict(old.method.model_module.student.named_parameters())
            kept = "backbone.blocks.0.attn.qkv.weight"
            torch.testing.assert_close(new.optimizer.state[names[kept]]["exp_avg"],
                                       old.optimizer.state[old_names[kept]]["exp_avg"], rtol=0, atol=0)
            for fresh in ("backbone.blocks.0.rope_embed.radii", "backbone.blocks.0.rope_embed.angles",
                          "backbone.blocks.0.attn.q_norm.weight", "backbone.blocks.0.attn.k_norm.weight"):
                self.assertNotIn(names[fresh], new.optimizer.state, fresh)

            self.assertEqual(len(new.method.qk_norms_to_calibrate), 2 * len(new_backbone.blocks))
            x = batch["collated_global_crops"]

            def tokens(weight=None):
                backbone = copy.deepcopy(new_backbone).eval()
                for name, module in backbone.named_modules():
                    if name.endswith(("q_norm", "k_norm")):
                        if weight is None:
                            module.forward = lambda q: q  # no Q/K norm: the converted 640k-style model
                        elif weight != "calibrated":
                            module.weight.data.fill_(weight)
                with torch.no_grad():
                    return backbone.forward_features(x, view_kind="global")["x_norm_patchtokens"]

            reference, unit = tokens(), tokens(1.0)
            new.method.calibrate_qk_norms([batch])
            calibrated = tokens("calibrated")
            self.assertEqual(new.method.qk_norms_to_calibrate, [])
            self.assertLess((calibrated - reference).norm(), (unit - reference).norm())
            teacher_norm = new.method.model_module.teacher.backbone.blocks[0].attn.q_norm.weight
            self.assertTrue(torch.isfinite(teacher_norm).all() and (teacher_norm != 1).all())
            new.train_step(batch, 1)


class RewarmupTests(unittest.TestCase):
    def test_lr_rewarms_linearly_from_its_start_step(self):
        with tempfile.TemporaryDirectory() as folder:
            config = tiny_config(folder)
            config.update(rewarmup_start_step=3, rewarmup_steps=4, max_iterations=10)
            trainer = Trainer(config)
            lrs = [trainer._apply_optim_scheduler(step)[0] for step in range(9)]
            expected = [1e-4] * 3 + [1e-4 * k / 4 for k in (1, 2, 3, 4)] + [1e-4] * 2
            for got, want in zip(lrs, expected):
                self.assertAlmostEqual(got, want, places=12)


if __name__ == "__main__":
    unittest.main()
