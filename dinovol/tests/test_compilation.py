from __future__ import annotations

import tempfile
import unittest

import torch
from torch._functorch import config as functorch_config

from dinovol_2.pretrain import DinoIBOTPretrainer
from dinovol_2.eval.task_eval import TaskEvalRunner
from dinovol_2.ops.precision import resolve_amp_dtype


class CompilationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        torch.compiler.reset()
        torch.manual_seed(47)

    def tearDown(self):
        torch.compiler.reset()
        self.tempdir.cleanup()

    def config(self, compile_settings=False):
        return {
            "device": "cpu", "use_amp": False, "compile": compile_settings,
            "output_dir": self.tempdir.name, "max_iterations": 3, "warmup_steps": 0,
            "freeze_last_layer_steps": 0, "batch_size": 2, "lr": 1e-4,
            "ibot_masked_loss_chunk_size": 8,
            "model": {
                "model_type": "v3", "embed_dim": 36, "num_heads": 3, "depth": 2,
                "global_crops_size": [16]*3, "local_crops_size": [8]*3, "patch_size": [4]*3,
                "embedding_type": "deeper", "deeper_embed_stage_channels": [4, 8, 36],
                "deeper_embed_stage_strides": [1, 2, 2], "deeper_embed_depth_per_level": [1]*3,
                "deeper_embed_use_stem": True, "deeper_embed_stem_type": "conv",
                "deeper_embed_stem_channels": 4, "deeper_embed_final_projection": False,
                "grad_checkpointing": True, "drop_path_rate": 0., "rope_rescale_coords": None,
                "dino_out_dim": 64, "ibot_out_dim": 64,
                "dino_head_hidden_dim": 32, "ibot_head_hidden_dim": 32,
                "dino_head_bottleneck_dim": 12, "ibot_head_bottleneck_dim": 12,
            },
        }

    @staticmethod
    def batch(mask_every=3):
        masks = torch.zeros(4, 64, dtype=torch.bool)
        if mask_every:
            masks[:2, ::mask_every] = True
        indices = masks.flatten().nonzero().flatten()
        weights = (1. / masks.sum(1).clamp(min=1)).unsqueeze(1).expand_as(masks)[masks]
        return {
            "collated_global_crops": torch.randn(4, 1, 16, 16, 16),
            "collated_local_crops": torch.randn(4, 1, 8, 8, 8),
            "collated_masks": masks, "mask_indices_list": indices,
            "n_masked_patches": torch.tensor(indices.numel()), "masks_weight": weights,
            "n_global_views": 2, "n_local_views": 2,
        }

    def assert_models_close(self, first, second):
        for name, tensor in first.model_module.state_dict().items():
            torch.testing.assert_close(tensor, second.model_module.state_dict()[name], rtol=2e-4, atol=1e-5, msg=name)

    def test_compiled_training_matches_eager_and_checkpoint_resume(self):
        # AOTAutograd compiles forward/backward graphs without requiring a native compiler in CI.
        compiled = DinoIBOTPretrainer(self.config({"enabled": True, "backend": "aot_eager", "fullgraph": True}))
        eager = DinoIBOTPretrainer(self.config())
        eager.model_module.load_state_dict(compiled.model_module.state_dict(), strict=True)
        keys = set(compiled.model_module.state_dict())
        self.assertFalse(any("_orig_mod" in name for name in keys))
        self.assertTrue(compiled.compile_enabled)
        for branch in (compiled.model_module.student, compiled.model_module.teacher):
            for module in branch.values():
                self.assertIsNotNone(module._compiled_call_impl)
        teacher_before = {name: p.detach().clone() for name, p in compiled.model_module.teacher.named_parameters()}
        for step, mask_every in enumerate((3, 5, 0)):
            batch = self.batch(mask_every)
            expected = eager.train_step(batch, step)
            actual = compiled.train_step(batch, step)
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], rtol=1e-4, atol=2e-5, msg=name)
            self.assert_models_close(compiled, eager)
        self.assertTrue(any(not torch.equal(p, teacher_before[name])
                            for name, p in compiled.model_module.teacher.named_parameters()))
        self.assertTrue(all(p.grad is None for p in compiled.model_module.teacher.parameters()))
        self.assertEqual(keys, set(compiled.model_module.state_dict()))
        checkpoint = compiled.save_checkpoint(2)
        self.assertEqual(eager.load_checkpoint(checkpoint), 2)
        self.assert_models_close(compiled, eager)
        eager_checkpoint = eager.save_checkpoint(3)
        self.assertEqual(compiled.load_checkpoint(eager_checkpoint), 3)
        self.assert_models_close(compiled, eager)
        self.assertEqual(len(compiled.optimizer.state), len(eager.optimizer.state))
        compiled.model.eval()
        eager.model.eval()
        with torch.no_grad():
            x = torch.randn(1, 1, 8, 8, 8)
            actual = compiled.model_module.student.backbone(x)["x_norm_patchtokens"]
            expected = eager.model_module.student.backbone(x)["x_norm_patchtokens"]
        torch.testing.assert_close(actual, expected)

    def test_compiled_amp_backward_matches_eager(self):
        # Retain explicit FP16 coverage after switching the default to BF16.
        self._check_amp_backward("float16" if torch.cuda.is_available() else "bfloat16")

    def test_compiled_default_bf16_backward_matches_eager(self):
        self._check_amp_backward()

    def _check_amp_backward(self, amp_dtype=None):
        backend = "inductor" if amp_dtype is None and torch.cuda.is_available() else "aot_eager"
        config = self.config({"enabled": True, "backend": backend, "fullgraph": True})
        # CPU oneDNN lacks half-precision convolution backward on some platforms.
        config.update(use_amp=True, koleo_loss_weight=0., device="cuda" if torch.cuda.is_available() else "cpu")
        if amp_dtype is not None:
            config["amp_dtype"] = amp_dtype
        compiled = DinoIBOTPretrainer(config)
        eager = DinoIBOTPretrainer({**config, "compile": False})
        eager.model_module.load_state_dict(compiled.model_module.state_dict())
        expected_dtype = resolve_amp_dtype(amp_dtype or "bfloat16")
        self.assertEqual(compiled.amp_dtype, expected_dtype)
        self.assertEqual(compiled.scaler.is_enabled(), expected_dtype == torch.float16 and compiled.device.type == "cuda")
        if amp_dtype is None:
            self.assertEqual(compiled.config["amp_dtype"], "bfloat16")
        observed_dtypes = []
        handle = eager.model_module.student.dino_head.mlp[0].register_forward_hook(
            lambda module, inputs, output: observed_dtypes.append(output.dtype)
        )
        self.addCleanup(handle.remove)
        # This toy head overflows even on the original eager path. Let the
        # scaler back off, then compare finite gradients and an actual update.
        for trainer in (compiled, eager):
            trainer.scaler = torch.amp.GradScaler(
                "cuda", init_scale=1., enabled=trainer.scaler.is_enabled()
            )
        batch = self.batch()
        previous_autocast = functorch_config.backward_pass_autocast
        for _ in range(8):
            expected = eager.train_step(batch, 0)
            actual = compiled.train_step(batch, 0)
            self.assertEqual(eager.scaler.get_scale(), compiled.scaler.get_scale())
            eager_grad = torch.cat([p.grad.flatten() for p in eager.model_module.student.parameters() if p.grad is not None])
            compiled_grad = torch.cat([p.grad.flatten() for p in compiled.model_module.student.parameters() if p.grad is not None])
            if torch.isfinite(eager_grad).all() and torch.isfinite(compiled_grad).all():
                break
        self.assertEqual(functorch_config.backward_pass_autocast, previous_autocast)
        torch.testing.assert_close(actual["loss"], expected["loss"], rtol=0.01, atol=0.01)
        self.assertTrue(torch.isfinite(eager_grad).all())
        self.assertTrue(torch.isfinite(compiled_grad).all())
        self.assertTrue(observed_dtypes)
        self.assertEqual(set(observed_dtypes), {expected_dtype})
        self.assertGreater(torch.nn.functional.cosine_similarity(eager_grad, compiled_grad, dim=0).item(), 0.99)

    def test_amp_dtype_configuration_and_checkpoint_resume(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = {**self.config(), "device": device, "use_amp": True}
        bf16 = DinoIBOTPretrainer(config)
        fp16 = DinoIBOTPretrainer({**config, "amp_dtype": "fp16"})
        self.assertEqual(bf16.amp_dtype, torch.bfloat16)
        self.assertFalse(bf16.scaler.is_enabled())
        self.assertEqual(fp16.amp_dtype, torch.float16)
        self.assertEqual(fp16.scaler.is_enabled(), device == "cuda")
        # Resume in either direction: BF16 ignores an FP16 scaler state, while
        # FP16 starts its own scaler when the saved BF16 state is absent.
        self.assertEqual(bf16.load_checkpoint(fp16.save_checkpoint(0)), 0)
        self.assertFalse(bf16.scaler.is_enabled())
        self.assertEqual(fp16.load_checkpoint(bf16.save_checkpoint(1)), 1)
        self.assertEqual(fp16.scaler.is_enabled(), device == "cuda")
        runner = TaskEvalRunner(bf16.config, output_dir=bf16.output_dir,
                                device=bf16.device, use_amp=True)
        self.assertEqual(runner.amp_dtype, bf16.amp_dtype)
        preview = runner._probability_preview(torch.rand(5, 7).bfloat16())
        self.assertEqual(preview.shape, (5, 7, 3))
        for name in ("bf16", "bfloat16", "BF16"):
            self.assertEqual(resolve_amp_dtype(name), torch.bfloat16)
        for name in ("fp16", "float16"):
            self.assertEqual(resolve_amp_dtype(name), torch.float16)
        for value in ("float32", "typo", None, 16):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_amp_dtype(value)

    def test_compile_configuration_and_disabled_default(self):
        trainer = DinoIBOTPretrainer(self.config())
        self.assertFalse(trainer.compile_enabled)
        self.assertIsNone(trainer.model_module.student.backbone._compiled_call_impl)
        for setting in ("yes", {"enabled": "yes"}, {"enabled": True, "dynamic": "yes"},
                        {"enabled": True, "fullgraph": "yes"}, {"enabled": True, "typo": 1},
                        {"enabled": True, "dynamic": {"typo": False}},
                        {"enabled": True, "dynamic": {"backbone": "yes"}}):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                trainer.config["compile"] = setting
                trainer._configure_model_compilation()


if __name__ == "__main__":
    unittest.main()
