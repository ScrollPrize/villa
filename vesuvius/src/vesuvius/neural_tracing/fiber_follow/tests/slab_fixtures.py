import torch
from test_identity import config as cfg, batch as base_batch


def slab_inputs(b):
    valid = torch.zeros(b, 8, dtype=torch.bool)
    valid[:, :2] = True
    return dict(history_slabs=torch.rand(b, 8, 2, 8, 65, 65), history_valid=valid,
                history_pose=torch.zeros(b, 8, 14), history_ages=torch.zeros(b, 8),
                history_overlap=torch.zeros(b, 8), history_load_seconds=torch.zeros(b))


def slab_batch(c, b=2, step=0):
    return base_batch(c, b)
