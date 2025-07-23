# this is basically copy/pasted from dynamic_network_architectures : https://github.com/MIC-DKFZ/dynamic-network-architectures/tree/main
# reworked a bit to function here. again probably easier to import , but now gives me a good baseline i can modify without having
# to rely on a 3rd party folder with a custom module. i baked the split conv and res encoder/decoder into one shared one

from typing import Tuple, List, Union, Type
import numpy as np
import torch.nn
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.dropout import _DropoutNd
from torch.nn.modules.instancenorm import _InstanceNorm
from torch.nn.modules.pooling import _MaxPoolNd

from models.utils import (maybe_convert_scalar_to_list,
                   get_matching_dropout,
                   get_matching_convtransp,
                   get_matching_batchnorm,
                   get_matching_instancenorm,
                   get_matching_pool_op)

from .simple_conv_blocks import ConvDropoutNormReLU, StackedConvBlocks
from .resblocks import (BasicBlockD,
                                BottleneckD,
                                StackedResidualBlocks)

class Encoder(nn.Module,):
    def __init__(self,
                 input_channels: int,
                 basic_block: str,
                 n_stages: int,
                 features_per_stage: Union[int, List[int]],
                 n_blocks_per_stage: Union[int, List[int]],
                 conv_op: Type[_ConvNd],
                 strides: Union[int, List[int]],
                 kernel_sizes: Union[int, List[int]],
                 conv_bias: bool,
                 norm_op: Union[None, Type[_BatchNorm], Type[_InstanceNorm]],
                 norm_op_kwargs: dict,
                 dropout_op: Union[None, Type[_DropoutNd]],
                 dropout_op_kwargs: dict,
                 nonlin: Union[None, Type[torch.nn.Module]],
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

                 ):

        super().__init__()

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

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        if basic_block == 'BottleneckBlockD':
            block = BottleneckD
            is_residual = True
        elif basic_block == 'BasicBlockD':
            block = BasicBlockD
            is_residual = True
        elif basic_block == 'ConvBlock':
            block = None
            is_residual = False
        else:
            raise ValueError(f"Unknown block type: {basic_block}")

        if do_stem:
            if stem_channels is None:
                stem_channels = features_per_stage[0]
            self.stem = StackedConvBlocks(1, conv_op, input_channels, stem_channels, kernel_sizes[0], 1, conv_bias,
                                          norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
            input_channels = stem_channels
        else:
            self.stem = None

        stages = []
        for s in range(n_stages):

            stride_for_conv = strides[s] if pool_op is None else 1
            if is_residual:
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
                    squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio
                )

                if pool_op is not None:
                    stage = nn.Sequential(pool_op(strides[s]), stage)

                stages.append(stage)
                input_channels = features_per_stage[s]

            else:
                stage_modules = []
                conv_stride = strides[s]
                stage_modules.append(StackedConvBlocks(
                    n_blocks_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
                ))
                stages.append(nn.Sequential(*stage_modules))
                input_channels = features_per_stage[s]


        # we store some things that a potential decoder needs
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

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        return output


