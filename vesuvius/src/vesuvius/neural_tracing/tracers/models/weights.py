"""Model weights independent of any model type: the convolution memory layout and loading matching tensors."""
import torch


def conv_memory_format(device):
    """Use the shared encoder convolution layout on CUDA."""
    return torch.channels_last_3d if torch.device(device).type == 'cuda' else torch.contiguous_format


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
