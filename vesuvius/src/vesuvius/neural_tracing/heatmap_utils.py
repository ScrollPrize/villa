import torch


def expected_heatmap_centroid(heatmap, temperature=1.0, apply_sigmoid=True, eps=1.0e-8):
    """Compute centroid of a heatmap differentiably."""
    squeeze_output = False
    if heatmap.ndim == 3:
        heatmap = heatmap[None, ...]
        squeeze_output = True

    if apply_sigmoid:
        prob_map = torch.sigmoid(heatmap / temperature)
    else:
        prob_map = heatmap

    prob_normalized = prob_map / (prob_map.sum(dim=(1, 2, 3), keepdim=True) + eps)

    shape = prob_map.shape[1:]
    device = prob_map.device
    dtype = torch.float32

    z_coords = torch.arange(shape[0], device=device, dtype=dtype)
    y_coords = torch.arange(shape[1], device=device, dtype=dtype)
    x_coords = torch.arange(shape[2], device=device, dtype=dtype)
    zyx_grid = torch.stack(torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij'), dim=-1)

    centroid = (prob_normalized.unsqueeze(-1) * zyx_grid).sum(dim=(1, 2, 3))
    return centroid.squeeze(0) if squeeze_output else centroid
