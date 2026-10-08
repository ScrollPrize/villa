"""Model weights independent of any model type: the convolution memory layout and loading matching tensors."""
import torch


def conv_memory_format(device):
    """Use the shared encoder convolution layout on CUDA."""
    return torch.channels_last_3d if torch.device(device).type == 'cuda' else torch.contiguous_format


# Weight names of earlier layouts -> current ones (the same tensors): the crop transformer's and the sequence
# follower's cell-token Linear became the 'cells' head of models.blocks.CropCNN.
RENAMED_WEIGHTS = (('cell_token.', 'cnn.head.'),)


def upgrade_state(state: dict) -> dict:
    """``state`` with weights of earlier layouts renamed (``RENAMED_WEIGHTS``)."""
    def rename(key):
        for old, new in RENAMED_WEIGHTS:
            if key.startswith(old):
                return new+key[len(old):]
        return key
    return {rename(k): v for k, v in state.items()}


def load_matching_weights(model, state, exclude=()):
    """Load tensors whose names and shapes match, except names starting with an ``exclude`` prefix.

    Returns a report: tensors left at initialization (``fresh``), checkpoint tensors with no counterpart
    (``unexpected``) and same-name tensors of another shape (``mismatched``)."""
    own = model.state_dict()
    matched = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape
               and not any(k.startswith(prefix) for prefix in exclude)}
    model.load_state_dict(matched, strict=False)
    return dict(fresh=sorted(set(own)-set(matched)), unexpected=sorted(set(state)-set(own)),
                mismatched=sorted(k for k, v in state.items() if k in own and own[k].shape != v.shape))
