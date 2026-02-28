
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Union, List, Tuple, Type

from vesuvius.models.utilities.utils import (
    maybe_convert_scalar_to_list,
    get_matching_pool_op,
    get_matching_convtransp,
    DropPath,
    SqueezeExcite,
)


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional embeddings for timestep conditioning."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class FiLM(nn.Module):
    """Feature-wise Linear Modulation for timestep conditioning."""
    def __init__(self, channels, time_emb_dim):
        super().__init__()
        self.channels = channels
        self.time_emb_dim = time_emb_dim
        
        # Project time embedding to scale and shift parameters
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2)
        )

    def forward(self, x, time_emb):
        """
        Args:
            x: Feature tensor of shape (B, C, ...)
            time_emb: Time embedding of shape (B, time_emb_dim)
        """
        # Get scale and shift parameters
        params = self.time_proj(time_emb)  # (B, 2*C)
        scale, shift = params.chunk(2, dim=1)  # Each: (B, C)
        
        # Reshape for broadcasting
        for _ in range(x.dim() - 2):
            scale = scale.unsqueeze(-1)
            shift = shift.unsqueeze(-1)
        
        # Apply FiLM: x = scale * x + shift
        return scale * x + shift


class ConvDropoutNormReLU(nn.Module):
    """Convolution + Dropout + Normalization + ReLU block with optional FiLM conditioning."""
    def __init__(self,
                 conv_op: Type[nn.Module],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[nn.Module]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False,
                 time_emb_dim: int = None):
        super(ConvDropoutNormReLU, self).__init__()
        
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        # Convolution
        self.conv = conv_op(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding=[(i - 1) // 2 for i in kernel_size],
            dilation=1,
            bias=conv_bias,
        )

        # Dropout
        self.dropout = dropout_op(**dropout_op_kwargs) if dropout_op is not None else None

        # Normalization
        self.norm = norm_op(output_channels, **norm_op_kwargs) if norm_op is not None else None

        # Activation
        self.nonlin = nonlin(**nonlin_kwargs) if nonlin is not None else None

        # FiLM conditioning
        self.film = FiLM(output_channels, time_emb_dim) if time_emb_dim is not None else None

    def forward(self, x, time_emb):
        x = self.conv(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
            
        if self.norm is not None:
            x = self.norm(x)
            
        # Apply FiLM conditioning after normalization
        if self.film is not None:
            x = self.film(x, time_emb)
            
        if self.nonlin is not None:
            x = self.nonlin(x)
            
        return x


class StackedConvBlocks(nn.Module):
    """Stack multiple ConvDropoutNormReLU blocks with timestep conditioning."""
    def __init__(self,
                 num_convs: int,
                 conv_op: Type[nn.Module],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[nn.Module]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False,
                 time_emb_dim: int = None):
        super().__init__()
        
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs = nn.ModuleList([
            ConvDropoutNormReLU(
                conv_op, input_channels, output_channels[0], kernel_size, initial_stride, 
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, 
                nonlin, nonlin_kwargs, nonlin_first, time_emb_dim
            )
        ] + [
            ConvDropoutNormReLU(
                conv_op, output_channels[i - 1], output_channels[i], kernel_size, 1, 
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, 
                nonlin, nonlin_kwargs, nonlin_first, time_emb_dim
            )
            for i in range(1, num_convs)
        ])

        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x, time_emb):
        for conv in self.convs:
            x = conv(x, time_emb)
        return x


class BasicBlockD(nn.Module):
    """ResNet-D Basic Block with squeeze-excitation and timestep conditioning support."""
    def __init__(self,
                 conv_op: Type[nn.Module],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[nn.Module]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16,
                 time_emb_dim: int = None):
        super().__init__()
        
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        # Main path
        self.conv1 = ConvDropoutNormReLU(
            conv_op, input_channels, output_channels, kernel_size, stride, conv_bias,
            norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
            time_emb_dim=time_emb_dim
        )
        self.conv2 = ConvDropoutNormReLU(
            conv_op, output_channels, output_channels, kernel_size, 1, conv_bias, 
            norm_op, norm_op_kwargs, None, None, None, None, time_emb_dim=time_emb_dim
        )

        # Final activation
        if nonlin is not None:
            self.nonlin2 = nonlin(**nonlin_kwargs)
        else:
            self.nonlin2 = lambda x: x

        # Stochastic Depth
        self.apply_stochastic_depth = stochastic_depth_p > 0.0
        if self.apply_stochastic_depth:
            self.drop_path = DropPath(drop_prob=stochastic_depth_p)

        # Squeeze Excitation
        self.apply_se = squeeze_excitation
        if self.apply_se:
            self.squeeze_excitation = SqueezeExcite(
                output_channels, conv_op, rd_ratio=squeeze_excitation_reduction_ratio, rd_divisor=8
            )

        # Skip connection
        has_stride = (isinstance(stride, int) and stride != 1) or any([i != 1 for i in stride])
        requires_projection = (input_channels != output_channels)

        if has_stride or requires_projection:
            ops = []
            if has_stride:
                ops.append(get_matching_pool_op(conv_op=conv_op, adaptive=False, pool_type='avg')(stride, stride))
            if requires_projection:
                ops.append(
                    ConvDropoutNormReLU(
                        conv_op, input_channels, output_channels, 1, 1, False, 
                        norm_op, norm_op_kwargs, None, None, None, None, time_emb_dim=time_emb_dim
                    )
                )
            self.skip = nn.Sequential(*ops)
        else:
            self.skip = lambda x: x

    def forward(self, x, time_emb):
        # Handle skip connection - pooling doesn't need time_emb, conv does
        if isinstance(self.skip, nn.Sequential):
            residual = x
            for module in self.skip:
                if isinstance(module, ConvDropoutNormReLU):
                    residual = module(residual, time_emb)
                else:
                    residual = module(residual)
        else:
            residual = self.skip(x)
            
        out = self.conv2(self.conv1(x, time_emb), time_emb)
        
        if self.apply_stochastic_depth:
            out = self.drop_path(out)
        if self.apply_se:
            out = self.squeeze_excitation(out)
            
        out += residual
        return self.nonlin2(out)


class StackedResidualBlocks(nn.Module):
    """Stack multiple residual blocks with timestep conditioning."""
    def __init__(self,
                 n_blocks: int,
                 conv_op: Type[nn.Module],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[nn.Module]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block: Type[BasicBlockD] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16,
                 time_emb_dim: int = None):
        super().__init__()
        
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * n_blocks

        self.blocks = nn.ModuleList([
            block(
                conv_op, input_channels, output_channels[0], kernel_size, initial_stride, 
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, 
                nonlin, nonlin_kwargs, stochastic_depth_p, squeeze_excitation, 
                squeeze_excitation_reduction_ratio, time_emb_dim
            )
        ] + [
            block(
                conv_op, output_channels[n - 1], output_channels[n], kernel_size, 1, 
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, 
                nonlin, nonlin_kwargs, stochastic_depth_p, squeeze_excitation, 
                squeeze_excitation_reduction_ratio, time_emb_dim
            ) for n in range(1, n_blocks)
        ])
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)
        self.output_channels = output_channels[-1]

    def forward(self, x, time_emb):
        for block in self.blocks:
            x = block(x, time_emb)
        return x


class Encoder(nn.Module):
    """3D U-Net Encoder with BasicBlockD residual blocks and timestep conditioning."""
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int]],
                 n_blocks_per_stage: Union[int, List[int]],
                 conv_op: Type[nn.Module],
                 strides: Union[int, List[int]],
                 kernel_sizes: Union[int, List[int]],
                 conv_bias: bool,
                 norm_op: Union[None, Type[nn.Module]],
                 norm_op_kwargs: dict,
                 dropout_op: Union[None, Type[nn.Module]],
                 dropout_op_kwargs: dict,
                 nonlin: Union[None, Type[nn.Module]],
                 nonlin_kwargs: dict,
                 do_stem: bool = True,
                 stem_channels: int = None,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16,
                 stochastic_depth_p: float = 0.0,
                 return_skips: bool = False,
                 bottleneck_block = BasicBlockD,
                 pool_type: str = 'conv',
                 bottleneck_channels: Union[int, List[int]] = None,
                 n_conv_per_stage: Union[int, List[int]] = None,
                 time_emb_dim: int = None):
        super().__init__()

        # Convert scalars to lists
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if bottleneck_channels is None or isinstance(bottleneck_channels, int):
            bottleneck_channels = [bottleneck_channels] * n_stages

        # Always use BasicBlockD
        block = BasicBlockD
        is_residual = True

        # Stem
        if do_stem:
            if stem_channels is None:
                stem_channels = features_per_stage[0]
            self.stem = StackedConvBlocks(
                1, conv_op, input_channels, stem_channels, kernel_sizes[0], 1, 
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, 
                nonlin, nonlin_kwargs, time_emb_dim=time_emb_dim
            )
            input_channels = stem_channels
        else:
            self.stem = None

        # Stages
        stages = []
        for s in range(n_stages):
            stride_for_conv = strides[s]
            
            stage = StackedResidualBlocks(
                n_blocks_per_stage[s],
                conv_op,
                input_channels,
                features_per_stage[s],
                kernel_sizes[s],
                stride_for_conv,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                block=block,
                bottleneck_channels=bottleneck_channels[s],
                stochastic_depth_p=stochastic_depth_p,
                squeeze_excitation=squeeze_excitation,
                squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio,
                time_emb_dim=time_emb_dim
            )
            stages.append(stage)
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x, time_emb):
        if self.stem is not None:
            x = self.stem(x, time_emb)
        
        ret = []
        for s in self.stages:
            x = s(x, time_emb)
            ret.append(x)
            
        if self.return_skips:
            return ret
        else:
            return ret[-1]


class Decoder(nn.Module):
    """3D U-Net Decoder with BasicBlockD residual blocks, transpose convolutions, skip connections, and timestep conditioning."""
    def __init__(self,
                 encoder,
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[nn.Module]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None,
                 time_emb_dim: int = None):
        super().__init__()
        
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        
        assert len(n_conv_per_stage) == n_stages_encoder - 1, \
            f"n_conv_per_stage must have {n_stages_encoder - 1} entries"

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs

        # Build decoder stages
        stages = []
        transpconvs = []
        seg_layers = []
        
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            
            # Transpose convolution
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, 
                stride_for_transpconv, bias=encoder.conv_bias
            ))
            
            # Decoder stage - always use BasicBlockD
            stages.append(StackedResidualBlocks(
                n_blocks=n_conv_per_stage[s - 1],
                conv_op=encoder.conv_op,
                input_channels=2 * input_features_skip,
                output_channels=input_features_skip,
                kernel_size=encoder.kernel_sizes[-(s + 1)],
                initial_stride=1,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                time_emb_dim=time_emb_dim
            ))

            # Segmentation layer
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
        
        # Final transpose convolution to match input size
        self.final_transpconv = transpconv_op(
            encoder.output_channels[0], encoder.output_channels[0], 
            encoder.strides[0], encoder.strides[0], bias=encoder.conv_bias
        )
        
        self.final_seg_layer = encoder.conv_op(encoder.output_channels[0], num_classes, 1, 1, 0, bias=True)

    def forward(self, skips, time_emb):
        """Forward pass with skip connections and timestep conditioning."""
        lres_input = skips[-1]
        seg_outputs = []
        
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x, time_emb)
            
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        lres_input = self.final_transpconv(lres_input)
        seg_outputs.append(self.final_seg_layer(lres_input))
        
        # Invert seg outputs so largest prediction is first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            return seg_outputs[0]
        else:
            return seg_outputs


class Vesuvius3dUnetModel(nn.Module):
    """
    3D U-Net model for denoising diffusion with timestep conditioning.
    
    This model supports denoising diffusion by conditioning on timesteps (0-1000)
    using sinusoidal positional embeddings and FiLM (Feature-wise Linear Modulation).
    
    Architecture features:
    - 3D convolutions with InstanceNorm3d normalization
    - BasicBlockD residual blocks with squeeze-excitation in both encoder and decoder
    - 6-stage encoder with features: [32, 64, 128, 256, 320, 320]
    - Skip connections between encoder and decoder
    - Transpose convolutions for upsampling in decoder
    - Timestep conditioning via FiLM layers
    - Configurable input/output channels
    """
    
    def __init__(self, in_channels, out_channels, config):
        super().__init__()

        model_config = config['model_config']
        
        # Network parameters - configurable input/output channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = model_config.get('time_emb_dim', None)
        if self.time_emb_dim == 0:
            self.time_emb_dim = None
        
        conv_op = nn.Conv3d
        norm_op = nn.InstanceNorm3d
        
        # Architecture parameters
        self.features_per_stage = model_config.get('features_per_stage', [32, 64, 128, 256, 320, 320])
        self.num_stages = len(self.features_per_stage)
        
        # Number of blocks per stage
        self.n_blocks_per_stage = self._get_n_blocks_per_stage(self.num_stages)
        
        # Kernel sizes and strides
        self.kernel_sizes = [[3, 3, 3]] * self.num_stages
        self.strides = [[2, 2, 2]] * self.num_stages
        
        # Timestep embedding
        self.time_embedding = SinusoidalPositionEmbeddings(self.time_emb_dim) if self.time_emb_dim else None
        
        # Build encoder
        self.encoder = Encoder(
            input_channels=self.in_channels,
            n_stages=self.num_stages,
            features_per_stage=self.features_per_stage,
            n_blocks_per_stage=self.n_blocks_per_stage,
            conv_op=conv_op,
            strides=self.strides,
            kernel_sizes=self.kernel_sizes,
            conv_bias=True,
            norm_op=norm_op,
            norm_op_kwargs={'affine': True, 'eps': 1e-5},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            do_stem=True,
            stem_channels=self.features_per_stage[0],
            squeeze_excitation=model_config.get('squeeze_excitation', True),
            squeeze_excitation_reduction_ratio=1.0/16.0,
            stochastic_depth_p=0.0,
            return_skips=True,
            pool_type='conv',
            time_emb_dim=self.time_emb_dim
        )
        
        # Build decoder
        self.decoder = Decoder(
            encoder=self.encoder,
            num_classes=self.out_channels,
            n_conv_per_stage=[1] * (self.num_stages - 1),
            deep_supervision=False,
            time_emb_dim=self.time_emb_dim
        )

    def _get_n_blocks_per_stage(self, num_stages):
        """Get number of blocks per stage (nnU-Net style)."""
        blocks = []
        for i in range(num_stages):
            if i == 0:
                blocks.append(1)
            elif i == 1:
                blocks.append(3)
            elif i == 2:
                blocks.append(4)
            else:
                blocks.append(6)
        return blocks
    
    def forward(self, x, timesteps=None):
        """
        Forward pass for denoising diffusion.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W) - noisy data
            timesteps: Timestep tensor of shape (B,) with values 0-1000
            
        Returns:
            Output tensor of shape (B, C, D, H, W) - predicted noise
        """
        # Get timestep embeddings
        time_emb = self.time_embedding(timesteps) if self.time_embedding is not None else None  # (B, time_emb_dim)
        
        # Get skip connections from encoder
        skips = self.encoder(x, time_emb)
        
        # Decode with skip connections and timestep conditioning
        output = self.decoder(skips, time_emb)
        
        return output

