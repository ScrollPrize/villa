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
