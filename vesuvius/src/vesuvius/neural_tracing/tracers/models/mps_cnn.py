"""The tracers' CropCNN on Apple GPUs as Metal 4 tensor-op kernels (``mps_cnn.metal``), dispatched on PyTorch MPS
buffers

:class:`MetalCropCNN` replaces ``CropCNN.features`` on MPS without autograd. Activations stay channels-last and
zero-padded between convs, each conv's epilogue writes the InstanceNorm partials of its output, and one kernel per
norm applies it with the block's residual and LeakyReLU, straight into the next conv's padded input. Parameters are
the module's own; any structure the kernels do not cover runs the stock forward.
"""
from __future__ import annotations

import platform
from importlib import resources

import torch
import torch.nn.functional as F
from torch import nn

from vesuvius.models.build.resblocks import BasicBlockD
from vesuvius.models.build.simple_conv_blocks import ConvDropoutNormReLU
from vesuvius.neural_tracing.tracers.models.blocks import CropCNN
from vesuvius.neural_tracing.tracers.models.mps import DepthTapConv3d, DirectConv3d, FusedBasicBlockD

_LIBRARIES = {}


def metal_tensor_ops():
    """Whether this Mac's Metal has the tensor ops ``mps_cnn.metal`` includes (``<metal_tensor>``, Metal 4): macOS 26
    and later. Below that :func:`~vesuvius.neural_tracing.tracers.models.mps.optimize_for_mps` keeps the CropCNN on
    the MPS layers, whose strips match within float noise (mean 2.3e-6 over 234,375 nodes, same kept nodes)"""
    release = platform.mac_ver()[0]
    return bool(release) and int(release.split('.')[0]) >= 26


def _library():
    if 'float' not in _LIBRARIES:
        source = resources.files(__package__).joinpath('mps_cnn.metal').read_text()
        _LIBRARIES['float'] = torch.mps.compile_shader(source)
    return _LIBRARIES['float']


def _conv(module, kernel, stride, *, bias=None):
    """Whether ``module`` is a plain Conv3d with a cube kernel, cube stride and padding kernel // 2"""
    return (type(module) in _CONVS and module.kernel_size == (kernel,)*3 and module.stride == (stride,)*3
            and module.padding == (kernel//2,)*3 and module.dilation == (1, 1, 1) and module.groups == 1
            and module.padding_mode == 'zeros' and (bias is None or (module.bias is not None) == bias))


def _norm(module):
    return (type(module) is nn.InstanceNorm3d and module.affine and not module.track_running_stats
            and module.weight is not None)


def _conv_norm(block, activation):
    """A ConvDropoutNormReLU of a conv, an affine InstanceNorm3d and (``activation``) a LeakyReLU, nothing else"""
    if type(block) is not ConvDropoutNormReLU or hasattr(block, 'dropout') or not _norm(getattr(block, 'norm', None)):
        return False
    return type(getattr(block, 'nonlin', None)) is nn.LeakyReLU if activation else not hasattr(block, 'nonlin')


def _is_identity(skip):
    """nn.Identity, or the ``lambda x: x`` that resblocks.BasicBlockD sets when the skip needs no conv"""
    if type(skip) is nn.Identity:
        return True
    code = getattr(skip, '__code__', None)
    return (code is not None and code.co_argcount == 1 and code.co_code == (lambda x: x).__code__.co_code
            and not getattr(skip, '__defaults__', None))


def _skip_kind(block, stride):
    """How a block's skip runs here: 'identity', 'pool' (AvgPool3d(2)), 'pool_conv' (AvgPool3d(2), 1x1 conv, norm),
    'conv' (1x1 conv, norm); None when the kernels do not cover it"""
    skip = block.skip
    if not isinstance(skip, nn.Sequential):
        return 'identity' if stride == 1 and _is_identity(skip) else None
    # checkpoints store the pool's sizes as lists ([2, 2, 2]), so compare them as tuples
    cube2 = lambda v: (tuple(v) if isinstance(v, (list, tuple)) else (v,)*3) == (2, 2, 2)
    pool = lambda m: (type(m) is nn.AvgPool3d and cube2(m.kernel_size) and cube2(m.stride) and m.padding in (0, (0,)*3, [0]*3)
                      and not m.ceil_mode and m.count_include_pad and m.divisor_override is None)
    parts = list(skip)
    if stride == 2 and len(parts) == 1 and pool(parts[0]):
        return 'pool'
    if stride == 2 and len(parts) == 2 and pool(parts[0]) and _conv_norm(parts[1], False) and _conv(parts[1].conv, 1, 1):
        return 'pool_conv'
    if stride == 1 and len(parts) == 1 and _conv_norm(parts[0], False) and _conv(parts[0].conv, 1, 1):
        return 'conv'
    return None


def _block_stride(block):
    """A block's stride when the kernels cover it (a ResNet-D BasicBlockD of conv, IN, LeakyReLU; conv, IN; skip;
    LeakyReLU, channels a multiple of 4), else None"""
    if type(block) not in _BLOCKS or block.apply_se or block.apply_stochastic_depth:
        return None
    if not (_conv_norm(block.conv1, True) and _conv_norm(block.conv2, False) and type(block.nonlin2) is nn.LeakyReLU):
        return None
    stride = block.conv1.conv.stride[0]
    out = block.conv2.conv.out_channels
    if not (_conv(block.conv1.conv, 3, stride) and stride in (1, 2) and _conv(block.conv2.conv, 3, 1)
            and block.conv1.conv.out_channels == out and out % 4 == 0):
        return None
    kind = _skip_kind(block, stride)
    if kind is None or (kind in ('identity', 'pool') and block.conv1.conv.in_channels != out):
        return None
    return stride


def _blocks(cnn):
    return [block for stage in cnn.stages for block in stage.blocks]


class MetalCropCNN(CropCNN):
    """A :class:`CropCNN` whose ``features`` run as ``mps_cnn.metal`` kernels on MPS (module docstring); the stock
    forward elsewhere, with autograd, or for crops whose sizes the strides do not divide"""

    @staticmethod
    def suits(module):
        return type(module) is CropCNN and metal_tensor_ops() and MetalCropCNN.covers(module)

    @staticmethod
    def covers(module):
        """Whether the kernels cover ``module``'s structure, before or after optimize_for_mps swaps its children"""
        if not isinstance(module, CropCNN) or hasattr(module, 'stem') or hasattr(module, 'bottleneck'):
            return False
        blocks = _blocks(module)
        if not blocks or any(_block_stride(b) is None for b in blocks):
            return False
        first = blocks[0]
        # the stem kernel: 2 -> 12 channels at stride 2, skip AvgPool3d(2) + 1x1 conv + norm
        if not (first.conv1.conv.in_channels == 2 and first.conv1.conv.out_channels == 12
                and _block_stride(first) == 2 and _skip_kind(first, 2) == 'pool_conv'):
            return False
        if hasattr(module, 'context_stage'):
            context = list(module.context_stage.blocks)
            fuse = module.context_fuse
            if (not context or any(_block_stride(b) is None for b in context) or _block_stride(context[0]) != 2
                    or not _conv(fuse, 1, 1, bias=True) or fuse.out_channels % 4):
                return False
        return True

    def features(self, image):
        if (image.device.type != 'mps' or image.dtype != torch.float32 or torch.is_grad_enabled()
                or image.shape[1] != 2 or not _even(image.shape[2:], self._halvings())):
            return super().features(image)
        return _Runner(self).features(image.contiguous())

    def _halvings(self):
        strides = [_block_stride(b) for b in _blocks(self)]
        if hasattr(self, 'context_stage'):
            strides += [_block_stride(b) for b in self.context_stage.blocks]
        return sum(s == 2 for s in strides)


# optimize_for_mps swaps a MetalCropCNN's children too (they serve its fallback); the swapped classes keep the
# parameters and outputs, so the kernels read them the same way
_BLOCKS = (BasicBlockD, FusedBasicBlockD)
_CONVS = (nn.Conv3d, DepthTapConv3d, DirectConv3d)


def _even(sizes, halvings):
    return all(int(s) % (1 << halvings) == 0 for s in sizes)


def _stamp(tensor):
    """What identifies ``tensor``'s current values: its storage and in-place version (inference tensors keep no
    version counter, and nothing outside inference mode can write them in place)"""
    return tensor.data_ptr(), None if tensor.is_inference() else tensor._version


def _padded_rows(b, d, h, w):
    """Rows (of C values) of a padded [B, D + 2, H + 2, W + 2, C] buffer plus the tail every tap view may read"""
    return b*(d+2)*(h+2)*(w+2)+2*(h+2)*(w+2)+2*(w+2)+2


class _Runner:
    """One forward of a :class:`MetalCropCNN`. Weights in kernel layouts and the padded buffers (zero borders, written
    only inside) are cached on the module, the buffers per shape"""

    def __init__(self, cnn):
        self.cnn, self.lib = cnn, _library()
        cache = cnn.__dict__.setdefault('_metal_cache', {})
        self.weights = cache.setdefault('weights', {})
        self.buffers = cache.setdefault('buffers', {})
        self.ints = cache.setdefault('ints', {})

    # host-side tables

    def weight(self, conv, kind='taps'):
        """``conv``'s weight in the layout of its kernel: taps [O, k, k, k, C], 'pool' [O, 2, 2, 2, C] (w / 8), 'raw'"""
        key = (id(conv), kind)
        # load_state_dict and optimizer steps write the weight in place (a new version), and assigning a new
        # Parameter changes its storage, so either one rebuilds the layout
        stamp = _stamp(conv.weight)
        cached = self.weights.get(key)
        if cached is None or cached[0] != stamp:
            w = conv.weight.detach().float()
            if kind == 'pool':
                w = (w[:, :, 0, 0, 0]/8.)[:, None, None, None, :].expand(-1, 2, 2, 2, -1)
            elif kind == 'taps':
                w = w.permute(0, 2, 3, 4, 1)
            cached = self.weights[key] = stamp, w.contiguous()
        return cached[1]

    def int_table(self, *values):
        if values not in self.ints:
            self.ints[values] = torch.tensor(values, dtype=torch.int32, device='mps')
        return self.ints[values]

    def padded(self, name, b, d, h, w, c):
        """A cached zero-bordered [B, D + 2, H + 2, W + 2, C] buffer (flat, with its tail rows)"""
        key = (name, b, d, h, w, c)
        if key not in self.buffers:
            self.buffers[key] = torch.zeros(_padded_rows(b, d, h, w)*c, dtype=torch.float32, device='mps')
        return self.buffers[key]

    # kernels

    def finalize(self, part, tiles, b, o, norm):
        stats = torch.empty((b, o, 2), dtype=torch.float32, device='mps')
        self.lib.in_finalize(stats, part, self.int_table(tiles, o), float(norm.eps),
                             threads=(-(-o//16)*256, b), group_size=(256, 1))
        return stats

    def conv_s1(self, xp, shape, conv, kernel, stats=True):
        """y [B, D, H, W, O] and (stats) the partials and their tile count, of a stride-1 conv on padded xp"""
        b, d, h, w, c = shape
        o = conv.out_channels
        bn = 64 if o > 32 else 32 if o > 16 else 16
        bm = 64
        tiles = -(-d*(h+2)*(w+2)//bm)
        y = torch.empty((b, d, h, w, o), dtype=torch.float32, device='mps')
        part = torch.empty((b, tiles, o, 4) if stats else (1, 4), dtype=torch.float32, device='mps')
        kernel_name = f'conv_s1_{bm}x{bn}'
        getattr(self.lib, kernel_name)(xp, self.weight(conv), y, part, self.int_table(b, d, h, w, c, o, kernel, int(stats)),
                                       threads=(-(-o//bn)*128, b*tiles), group_size=(128, 1))
        return y, part, tiles

    def conv_s2(self, xp, shape, conv, kernel):
        """y [B, D/2, H/2, W/2, O], partials and tile count of a stride-2 conv (kernel 3, or 2 for a pooled 1x1)"""
        b, d, h, w, c = shape
        o = conv.out_channels
        bn = 64 if o > 32 else 32
        bm = 32
        tiles = (d//2)*(h//2)*-(-(w//2)//bm)
        y = torch.empty((b, d//2, h//2, w//2, o), dtype=torch.float32, device='mps')
        part = torch.empty((b, tiles, o, 4), dtype=torch.float32, device='mps')
        getattr(self.lib, f'conv_s2_{bm}x{bn}')(xp, self.weight(conv, 'taps' if kernel == 3 else 'pool'), y, part,
                                                self.int_table(b, d, h, w, c, o, kernel),
                                                threads=(-(-o//bn)*128, b*tiles), group_size=(128, 1))
        return y, part, tiles

    def apply(self, y, stats, norm, out, *, mode=0, residual=None, residual_stats=None, residual_norm=None,
              slope=None):
        b, d, h, w, o = y.shape
        r = y if residual is None else residual
        rs = stats if residual_stats is None else residual_stats
        rn = norm if residual_norm is None else residual_norm
        count = b*d*h*w*o//4
        self.lib.in_apply(y, stats, norm.weight, norm.bias, r, rs, rn.weight, rn.bias, out,
                          self.int_table(b, d, h, w, o, mode, int(slope is not None)),
                          float(0. if slope is None else slope), threads=count, group_size=256)
        return out

    # network

    def stem_block(self, block, image):
        """Stage 0's first block on the crops: stem kernel (conv1 and the pooled skip conv), then conv2"""
        b, _, d, h, w = image.shape
        do, ho, wo = d//2, h//2, w//2
        count = do*ho*wo
        tiles = -(-count//192)
        y = torch.empty((b, do, ho, wo, 12), dtype=torch.float32, device='mps')
        ys = torch.empty_like(y)
        part = torch.empty((b, tiles, 12, 4), dtype=torch.float32, device='mps')
        parts = torch.empty_like(part)
        skip_conv = block.skip[1]
        self.lib.stem(image, self.weight(block.conv1.conv, 'raw'), self.weight(skip_conv.conv, 'raw'), y, ys, part,
                      parts, self.int_table(b, d, h, w), threads=(tiles*192, b), group_size=(192, 1))
        stats = self.finalize(part, tiles, b, 12, block.conv1.norm)
        skip_stats = self.finalize(parts, tiles, b, 12, skip_conv.norm)
        return self.finish(block, 'stem', y, stats, (b, do, ho, wo, 12), dict(
            mode=2, residual=ys, residual_stats=skip_stats, residual_norm=skip_conv.norm))

    def block(self, name, block, xp, shape):
        """One block on the padded xp of ``shape`` (B, D, H, W, C): its padded output and output shape"""
        b, d, h, w, c = shape
        stride = _block_stride(block)
        o = block.conv2.conv.out_channels
        if stride == 1:
            y, part, tiles = self.conv_s1(xp, shape, block.conv1.conv, 3)
        else:
            y, part, tiles = self.conv_s2(xp, shape, block.conv1.conv, 3)
        stats = self.finalize(part, tiles, b, o, block.conv1.norm)
        out_shape = (b, d//stride, h//stride, w//stride, o)
        kind = _skip_kind(block, stride)
        if kind == 'identity':
            residual = dict(mode=1, residual=xp)
        elif kind == 'pool':
            residual = dict(mode=3, residual=xp)
        else:
            skip = block.skip[-1]
            if kind == 'pool_conv':
                ys, sp, st = self.conv_s2(xp, shape, skip.conv, 2)
            else:
                ys, sp, st = self.conv_s1(xp, shape, skip.conv, 1)
            residual = dict(mode=2, residual=ys, residual_stats=self.finalize(sp, st, b, o, skip.norm),
                            residual_norm=skip.norm)
        return self.finish(block, name, y, stats, out_shape, residual)

    def finish(self, block, name, y, stats, shape, residual):
        """conv1's norm and activation into a padded buffer, conv2, its norm with the residual: (padded out, shape)"""
        b, d, h, w, o = shape
        hidden = self.apply(y, stats, block.conv1.norm, self.padded(f'{name}.hidden', *shape),
                            slope=block.conv1.nonlin.negative_slope)
        y2, part, tiles = self.conv_s1(hidden, shape, block.conv2.conv, 3)
        stats2 = self.finalize(part, tiles, b, o, block.conv2.norm)
        out = self.apply(y2, stats2, block.conv2.norm, self.padded(f'{name}.out', *shape),
                         slope=block.nonlin2.negative_slope, **residual)
        return out, shape

    def features(self, image):
        blocks = _blocks(self.cnn)
        xp, shape = self.stem_block(blocks[0], image)
        for index, block in enumerate(blocks[1:], 1):
            xp, shape = self.block(f'b{index}', block, xp, shape)
        b, d, h, w, c = shape
        if not hasattr(self.cnn, 'context_stage'):
            n = b*(d+2)*(h+2)*(w+2)*c
            return xp[:n].view(b, d+2, h+2, w+2, c)[:, 1:-1, 1:-1, 1:-1].permute(0, 4, 1, 2, 3)
        cp, cshape = xp, shape
        for index, block in enumerate(self.cnn.context_stage.blocks):
            cp, cshape = self.block(f'c{index}', block, cp, cshape)
        fuse = self.cnn.context_fuse
        f, _, _ = self.conv_s1(cp, cshape, fuse, 1, stats=False)
        out = torch.empty((b, d, h, w, fuse.out_channels), dtype=torch.float32, device='mps')
        self.lib.upsample_add(xp, f, fuse.bias, out, self.int_table(b, d, h, w, fuse.out_channels),
                              threads=out.numel()//4, group_size=256)
        return out.permute(0, 4, 1, 2, 3)
