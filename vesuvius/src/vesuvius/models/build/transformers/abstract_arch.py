from copy import deepcopy
from typing import Literal, Sequence
from torch import nn
import torch
from dataclasses import dataclass


class AbstractDynamicNetworkArchitectures(nn.Module):

    def __init__(self):
        super(AbstractDynamicNetworkArchitectures, self).__init__()
        # Key to the position holding all the encoder weights
        self.key_to_encoder: str
        # Key to the full stem -- Can be located within or outside the encoder
        self.key_to_stem: str
        # Not sure yet if we need anything but this -- but minor redundancy is okay I suppose
        # Key to the weights that are dependent on the input channels.
        #   Can hold multiple weights (e.g. for bad weight mappings like in this repo >.<' )
        self.keys_to_in_proj: Sequence[str]
        self.key_to_lpe: str | None = None  # LPE == Learnable Positional Embedding


def test_submodules_loadable(model: AbstractDynamicNetworkArchitectures):
    encoder_key = model.key_to_encoder
    stem_key = model.key_to_stem
    stem_weights_key = model.keys_to_in_proj
    # Check if the encoder submodule is loadable
    # Throws an error otherwise.
    _ = model.get_submodule(encoder_key)
    _ = model.get_submodule(stem_key)
    prev_shape = None
    for swk in stem_weights_key:
        stem_weights_submodule = model.get_submodule(swk).weight
        if prev_shape is None:
            prev_shape = stem_weights_submodule.shape
        else:
            assert stem_weights_submodule.shape == prev_shape, f"Stem weights submodule {swk} has different shape"
            prev_shape = stem_weights_submodule.shape
        assert stem_weights_submodule is not None, f"Stem weights submodule {swk} is not loadable"
    return
