"""Building blocks shared by the crop transformer (models/crop_transformer.py) and the sequence follower
(models/sequence.py): the residual crop CNN and the pre-norm transformer layer's weights and head split/merge. Each
model's layer subclass defines which tokens attend to which."""
from torch import nn

from vesuvius.models.build.resblocks import BasicBlockD, StackedResidualBlocks


class CropCNN(nn.Module):
    """Residual CNN stages (channels, initial stride, blocks per stage) over a one-channel crop; the first stage's
    convolutions carry biases."""
    def __init__(self, channels, strides, blocks):
        super().__init__()
        options = dict(conv_op=nn.Conv3d, kernel_size=3, norm_op=nn.InstanceNorm3d,
                       norm_op_kwargs=dict(eps=1e-5, affine=True), nonlin=nn.ReLU, nonlin_kwargs=dict(inplace=True),
                       block=BasicBlockD)
        inputs = (1, *channels[:-1])
        self.stages = nn.ModuleList(
            StackedResidualBlocks(n_blocks=n, input_channels=i, output_channels=c, initial_stride=s, conv_bias=not index,
                                  **options)
            for index, (i, c, s, n) in enumerate(zip(inputs, channels, strides, blocks)))

    def forward(self, image):
        x = image
        for stage in self.stages:
            x = stage(x)
        return x


class TransformerLayer(nn.Module):
    """One pre-norm transformer layer's weights (attention and FFN branches) and its head split/merge."""
    def __init__(self, width, heads, ffn):
        super().__init__()
        if width % heads:
            raise ValueError('Transformer width must divide by its heads')
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width)
        self.out = nn.Linear(width, width)
        self.ffn = nn.Sequential(nn.Linear(width, ffn), nn.GELU(), nn.Linear(ffn, width))

    def split(self, x):
        b, n, width = x.shape
        shape = lambda t: t.reshape(b, n, self.heads, width//self.heads).transpose(1, 2)
        return tuple(map(shape, self.qkv(x).chunk(3, -1)))

    def merge(self, value):
        b, heads, n, d = value.shape
        return self.out(value.transpose(1, 2).reshape(b, n, heads*d))
