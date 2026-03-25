from __future__ import annotations

import torch


def compute_group_avg(values, group_idx, *, n_groups):
    n_groups = int(n_groups)
    group_idx = group_idx.long()
    group_map = (group_idx == torch.arange(n_groups, device=group_idx.device).unsqueeze(1).long()).float()
    group_count = group_map.sum(1)
    group_denom = group_count + (group_count == 0).float()
    group_avg = (group_map @ values.view(-1)) / group_denom
    return group_avg, group_count
