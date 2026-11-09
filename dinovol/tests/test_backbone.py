import unittest
import torch
from dinovol_2.model import build_backbone
from dinovol_2.model.model import DinoVitStudentTeacher


class BackboneTests(unittest.TestCase):
    def test_public_factory_matches_dino_backbone(self):
        config = dict(model_type='v3', embed_dim=36, depth=1, num_heads=3,
            global_crops_size=[8]*3, local_crops_size=[4]*3, patch_size=[4]*3,
            drop_path_rate=0., rope_rescale_coords=None,
            dino_out_dim=16, ibot_out_dim=16, dino_head_hidden_dim=16,
            dino_head_bottleneck_dim=8, ibot_head_hidden_dim=16, ibot_head_bottleneck_dim=8)
        torch.manual_seed(5)
        backbone = build_backbone(config)
        torch.manual_seed(5)
        wrapper = DinoVitStudentTeacher(config)
        self.assertEqual(backbone.state_dict().keys(), wrapper.student.backbone.state_dict().keys())
        for name, value in backbone.state_dict().items():
            torch.testing.assert_close(value,wrapper.student.backbone.state_dict()[name],rtol=0,atol=0)
        backbone.eval(); wrapper.eval()
        for kind,size in [('global',8),('local',4)]:
            x = torch.randn(2,1,size,size,size)
            with torch.no_grad():
                a = backbone(x,is_training=True,view_kind=kind)
                b = wrapper.student.backbone(x,is_training=True,view_kind=kind)
            torch.testing.assert_close(a['x_norm_clstoken'],b['x_norm_clstoken'],rtol=0,atol=0)
            torch.testing.assert_close(a['x_norm_patchtokens'],b['x_norm_patchtokens'],rtol=0,atol=0)

    def test_channels_last_stem_sees_channels_last_single_channel_input(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        from dinovol_2.model.patch_encode_decode import PatchEmbedDeeper
        torch.manual_seed(0)
        stem = PatchEmbedDeeper(patch_size=(4, 4, 4), input_channels=1, embed_dim=16, stage_channels=(4, 8),
                                stage_strides=((2, 2, 2), (2, 2, 2)), depth_per_level=(1, 1), use_stem=True,
                                stem_type="residual", stem_channels=4, final_projection=True, channels_last=True)
        stem.to(memory_format=stem.memory_format)
        x = torch.randn(2, 1, 8, 8, 8)
        seen = []

        class Inputs(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func is torch.ops.aten.convolution.default and args[0].shape[1] == 1:
                    seen.append(args[0].stride())
                return func(*args, **(kwargs or {}))

        with Inputs():
            out = stem(x)
        # One-channel inputs carry explicit channels-last strides (same bytes, no copy).
        self.assertTrue(seen)
        self.assertTrue(all(stride == (512, 1, 64, 8, 1) for stride in seen), seen)
        self.assertEqual(stem(x).shape, out.shape)


if __name__ == "__main__":
    unittest.main()
