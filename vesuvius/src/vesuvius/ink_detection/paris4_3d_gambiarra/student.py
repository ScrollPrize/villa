"""The student: a vesuvius NetworkFromConfig U-Net ("deep5": 5 stages, 32-320 channels, 24,500,769 parameters) on
the normalized 128^3 crop average-pooled to 64^3 (4.8 um voxels). Its logits are a 64^3 field on the pooled grid;
for training they are resampled (linear, align_corners=False) to the 128^3 native voxels of the view.
"""
import contextlib
import copy
import io
import re
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

PARAMETERS = 24500769
CONFIG = dict(architecture_type='unet', features_per_stage=[32, 64, 128, 256, 320], n_blocks_per_stage=[1, 2, 2, 2, 2],
              n_conv_per_stage_decoder=[1] * 4, kernel_sizes=[[3] * 3] * 5, strides=[[1] * 3] + [[2] * 3] * 4,
              pool_op_kernel_sizes=[[1] * 3] + [[2] * 3] * 4, basic_encoder_block='BasicBlockD', basic_decoder_block='ConvBlock',
              bottleneck_block='BasicBlockD', norm_op='nn.InstanceNorm3d', norm_op_kwargs={'affine': True, 'eps': 1e-5},
              conv_bias=True, nonlin='nn.LeakyReLU', nonlin_kwargs={'negative_slope': .01, 'inplace': True},
              do_stem=False, return_skips=True, separate_decoders=False, pretrained_backbone=None, guide_backbone=None)


def manager(cfg=CONFIG):
    """The vesuvius model-manager fields NetworkFromConfig reads (model/config.json 'vesuvius' of the release)."""
    return SimpleNamespace(targets={'ink': {'out_channels': 1, 'activation': 'none'}}, train_patch_size=[128] * 3,
                           train_batch_size=1, in_channels=1, autoconfigure=False, op_dims=3, spacing=[2.4] * 3,
                           enable_deep_supervision=False, model_config=copy.deepcopy(cfg), model_name='paris4-native-student')


class _Pool2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert x.ndim == 5 and all(n % 2 == 0 for n in x.shape[-3:])
        return F.avg_pool3d(x, 2, 2)

    @staticmethod
    def backward(ctx, grad):
        # Each input voxel belongs to exactly one 2x2x2 block: no atomic accumulation in backward.
        return (grad / 8).repeat_interleave(2, -3).repeat_interleave(2, -2).repeat_interleave(2, -1)


class DeterministicPool2(nn.Module):
    """AvgPool3d(2, 2) with the same forward and a deterministic backward."""
    def forward(self, x):
        return _Pool2.apply(x)


def build_network(seed=27, deterministic_pool=True):
    """NetworkFromConfig(manager()) initialized under torch.manual_seed(seed) without touching the global RNG."""
    from vesuvius.models.build.build_network_from_config import NetworkFromConfig
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            network = NetworkFromConfig(manager())
    assert sum(p.numel() for p in network.parameters()) == PARAMETERS
    if deterministic_pool:
        keys = tuple(network.state_dict())
        pools = [name for name, m in network.named_modules() if isinstance(m, nn.AvgPool3d)]
        assert len(pools) == 4, pools
        def triple(v):
            return (v,) * 3 if isinstance(v, int) else tuple(v)
        for name in pools:
            m = network.get_submodule(name)
            assert triple(m.kernel_size) == triple(m.stride) == (2, 2, 2) and triple(m.padding) == (0, 0, 0)
            assert not m.ceil_mode and m.divisor_override is None
            parent, leaf = name.rsplit('.', 1)
            network.get_submodule(parent)._modules[leaf] = DeterministicPool2()
        assert tuple(network.state_dict()) == keys
    return network


def resample(low, factor=2, start=(0, 0, 0), size=128):
    """Values of low [B, C, n, n, n] at native voxel centres start..start+size-1: q = (i + .5) / factor - .5, edge
    clamped, separable index_select."""
    x = low
    for axis, s in zip((2, 3, 4), start):
        n = x.shape[axis]
        q = ((torch.arange(size, device=x.device, dtype=torch.float32) + float(s) + .5) / factor - .5).clamp(0, n - 1)
        lo = q.floor().long()
        hi = (lo + 1).clamp_max(n - 1)
        shape = [1] * 5
        shape[axis] = size
        w = (q - lo).reshape(shape)
        x = x.index_select(axis, lo) * (1 - w) + x.index_select(axis, hi) * w
    return x


class Student(nn.Module):
    def __init__(self, seed=27, deterministic_pool=True):
        super().__init__()
        self.network = build_network(seed, deterministic_pool).cuda()

    def low(self, x):
        """[B, 1, 128^3] normalized crops -> [B, 1, 64^3] float32 logits on the 4.8 um grid (bf16 autocast)."""
        x = F.avg_pool3d(x, 2)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return self.network(x)['ink'].float()

    def forward(self, x):
        """[B, 1, 128^3] logits at the native voxels of the crops (resampled per crop, as in training)."""
        low = self.low(x)
        return torch.cat([resample(low[k:k + 1]) for k in range(len(low))])


def load_weights(network, path):
    """Load a model.safetensors (release format, see save_weights) into network (strict; the aliases share the
    loaded tensors)."""
    from safetensors.torch import load_model
    load_model(network, str(path), strict=True)
    return network


def save_weights(network, path, metadata=None):
    """Write the network weights in the release format: the vesuvius build registers every conv / norm twice
    (`.conv` / `.norm` and the alias `.all_modules.0` / `.all_modules.1`, the same tensors); the aliases are dropped."""
    from safetensors.torch import save
    from .runtime import atomic_file
    sd = network.state_dict()
    for k in [k for k in sd if '.all_modules.' in k]:
        canon = re.sub(r'all_modules\.0\.', 'conv.', re.sub(r'all_modules\.1\.', 'norm.', k))
        assert sd[k].data_ptr() == sd[canon].data_ptr(), k
    data = save({k: v.detach().cpu().contiguous() for k, v in sd.items() if '.all_modules.' not in k}, metadata=metadata)
    with atomic_file(path) as f:
        f.write(data)
