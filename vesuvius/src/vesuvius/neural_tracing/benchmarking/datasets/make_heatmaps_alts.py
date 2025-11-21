import torch
import torch.nn.functional as F
from einops import rearrange
from fft_conv_pytorch import fft_conv
from vesuvius.neural_tracing.dataset import HeatmapDatasetV2, kernel, kernel_size

# ---- benchmarks for make_heatmaps method ----- #
# make heatmaps takes up a fair bit of our dataloader time
#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#        35    0.001    0.000   37.365    1.068 dataloader.py:729(__next__)
#        35    0.001    0.000   37.358    1.067 dataloader.py:788(_next_data)
#        35    0.001    0.000   37.357    1.067 fetch.py:25(fetch)
#       175    0.000    0.000   36.880    0.211 {built-in method builtins.next}
#       141    0.099    0.001   36.880    0.262 dataset.py:375(__iter__)
#       399    0.147    0.000   21.268    0.053 dataset.py:301(_get_perturbed_zyx_from_patch)
#       399    0.049    0.000   19.402    0.049 dataset.py:255(_get_cached_patch_points)
#     14466    0.013    0.000   12.498    0.001 dataset.py:225(_get_patch_points_in_crop)
#     15264    6.867    0.000   11.626    0.001 dataset.py:221(_get_quads_in_crop)
#       554    0.017    0.000    9.640    0.017 dataset.py:579(make_heatmaps)
# default timings: mean 18.54 ms | median 17.96 ms | min 14.77 ms | max 31.23 ms

def _scatter_heatmaps(all_zyxs, min_corner_zyx, crop_size):
    crop_size_int = int(crop_size)
    dtype = all_zyxs[0].dtype
    device = all_zyxs[0].device
    min_corner = min_corner_zyx.to(device=device, dtype=dtype)
    heatmaps = torch.zeros((crop_size_int, crop_size_int, crop_size_int, all_zyxs[0].shape[0]), device=device, dtype=dtype)

    def scatter(zyxs):
        coords = torch.cat([
            (zyxs.to(device=device, dtype=dtype) - min_corner + 0.5).int(),
            torch.arange(zyxs.shape[0], device=device)[:, None]
        ], dim=1)
        coords = coords[(coords[..., :3] >= 0).all(dim=1) & (coords[..., :3] < crop_size_int).all(dim=1)]
        if len(coords) > 0:
            heatmaps[*coords.T] = 1.

    for zyxs in all_zyxs:
        scatter(zyxs)

    return heatmaps

# slower , 24ms or so dif
class SeparableConv1d_fft(HeatmapDatasetV2):
    @classmethod
    def make_heatmaps(cls, all_zyxs, min_corner_zyx, crop_size):
        heatmaps = _scatter_heatmaps(all_zyxs, min_corner_zyx, crop_size)
        heatmaps_5d = rearrange(heatmaps, 'z y x c -> 1 c z y x')
        c = heatmaps_5d.shape[1]

        gaussian_1d = kernel[kernel_size // 2, kernel_size // 2].to(device=heatmaps_5d.device, dtype=heatmaps_5d.dtype)
        weight_z = gaussian_1d.view(1, 1, -1, 1, 1).expand(c, -1, -1, -1, -1)
        weight_y = gaussian_1d.view(1, 1, 1, -1, 1).expand(c, -1, -1, -1, -1)
        weight_x = gaussian_1d.view(1, 1, 1, 1, -1).expand(c, -1, -1, -1, -1)

        convolved = F.conv3d(heatmaps_5d, weight_z, padding=(kernel_size // 2, 0, 0), groups=c)
        convolved = F.conv3d(convolved, weight_y, padding=(0, kernel_size // 2, 0), groups=c)
        convolved = F.conv3d(convolved, weight_x, padding=(0, 0, kernel_size // 2), groups=c)

        # Match HeatmapDatasetV2 return layout: (channels, z, y, x)
        return rearrange(convolved, '1 c z y x -> c z y x')

# way way way fuckin slower
class Depthwise3dConv_fft(HeatmapDatasetV2):
    @classmethod
    def make_heatmaps(cls, all_zyxs, min_corner_zyx, crop_size):
        heatmaps = _scatter_heatmaps(all_zyxs, min_corner_zyx, crop_size)
        heatmaps_5d = rearrange(heatmaps, 'z y x c -> 1 c z y x')
        c = heatmaps_5d.shape[1]

        kernel_3d = kernel.to(device=heatmaps_5d.device, dtype=heatmaps_5d.dtype).unsqueeze(0).unsqueeze(0)
        weight = kernel_3d.expand(c, -1, -1, -1, -1)
        convolved = F.conv3d(heatmaps_5d, weight, padding=kernel_size // 2, groups=c)

        # Match HeatmapDatasetV2 return layout: (channels, z, y, x)
        return rearrange(convolved, '1 c z y x -> c z y x')


# mean 0.50 ms | median 0.50 ms | min 0.25 ms | max 0.80 ms
class SparseGaussianSplat(HeatmapDatasetV2):
    @classmethod
    def make_heatmaps(cls, all_zyxs, min_corner_zyx, crop_size):
        crop_size_int = int(crop_size)
        radius = kernel_size // 2
        dtype = all_zyxs[0].dtype
        device = all_zyxs[0].device
        min_corner = min_corner_zyx.to(device=device, dtype=dtype)

        heatmaps = torch.zeros((crop_size_int, crop_size_int, crop_size_int, all_zyxs[0].shape[0]), device=device, dtype=dtype)
        kernel_t = kernel.to(device=device, dtype=dtype)

        for zyxs in all_zyxs:
            coords = (zyxs.to(device=device, dtype=dtype) - min_corner + 0.5).int()
            valid_mask = (coords >= 0).all(dim=1) & (coords < crop_size_int).all(dim=1)
            if not torch.any(valid_mask):
                continue
            coords = coords[valid_mask]
            channel_indices = torch.arange(zyxs.shape[0], device=device)[valid_mask]

            for coord, channel_idx in zip(coords, channel_indices):
                z, y, x = int(coord[0]), int(coord[1]), int(coord[2])
                z0 = max(0, z - radius)
                y0 = max(0, y - radius)
                x0 = max(0, x - radius)
                z1 = min(crop_size_int, z + radius + 1)
                y1 = min(crop_size_int, y + radius + 1)
                x1 = min(crop_size_int, x + radius + 1)

                kz0 = z0 - (z - radius)
                ky0 = y0 - (y - radius)
                kx0 = x0 - (x - radius)
                kz1 = kz0 + (z1 - z0)
                ky1 = ky0 + (y1 - y0)
                kx1 = kx0 + (x1 - x0)

                heatmaps[z0:z1, y0:y1, x0:x1, int(channel_idx)] += kernel_t[kz0:kz1, ky0:ky1, kx0:kx1]

        # Match HeatmapDatasetV2 return layout: (channels, z, y, x)
        return rearrange(heatmaps, 'z y x c -> c z y x')

# mean 1.38 ms | median 1.34 ms | min 1.23 ms | max 1.89 ms
class CroppedFFTConv(HeatmapDatasetV2):
    @classmethod
    def make_heatmaps(cls, all_zyxs, min_corner_zyx, crop_size):
        crop_size_int = int(crop_size)
        radius = kernel_size // 2
        dtype = all_zyxs[0].dtype
        device = all_zyxs[0].device
        min_corner = min_corner_zyx.to(device=device, dtype=dtype)

        all_coords = []
        all_channels = []
        for zyxs in all_zyxs:
            coords = (zyxs.to(device=device, dtype=dtype) - min_corner + 0.5).int()
            valid_mask = (coords >= 0).all(dim=1) & (coords < crop_size_int).all(dim=1)
            if not torch.any(valid_mask):
                continue
            all_coords.append(coords[valid_mask])
            all_channels.append(torch.arange(zyxs.shape[0], device=device)[valid_mask])

        if len(all_coords) == 0:
            return torch.zeros((all_zyxs[0].shape[0], crop_size_int, crop_size_int, crop_size_int), device=device, dtype=dtype)

        coords_cat = torch.cat(all_coords, dim=0)
        channels_cat = torch.cat(all_channels, dim=0)

        z_min = max(0, int(coords_cat[:, 0].min().item()) - radius)
        y_min = max(0, int(coords_cat[:, 1].min().item()) - radius)
        x_min = max(0, int(coords_cat[:, 2].min().item()) - radius)
        z_max = min(crop_size_int, int(coords_cat[:, 0].max().item()) + radius + 1)
        y_max = min(crop_size_int, int(coords_cat[:, 1].max().item()) + radius + 1)
        x_max = min(crop_size_int, int(coords_cat[:, 2].max().item()) + radius + 1)

        sub_shape = (z_max - z_min, y_max - y_min, x_max - x_min)
        heatmaps_sub = torch.zeros((*sub_shape, all_zyxs[0].shape[0]), device=device, dtype=dtype)

        rel_coords = coords_cat - torch.tensor([z_min, y_min, x_min], device=device)
        heatmaps_sub[rel_coords[:, 0], rel_coords[:, 1], rel_coords[:, 2], channels_cat] = 1.

        convolved_sub = fft_conv(
            rearrange(heatmaps_sub, 'z y x c -> c 1 z y x'),
            kernel[None, None].to(device=device, dtype=dtype),
            padding=radius
        )
        convolved_sub = rearrange(convolved_sub, 'c 1 z y x -> c z y x')

        full_heatmaps = torch.zeros((all_zyxs[0].shape[0], crop_size_int, crop_size_int, crop_size_int), device=device, dtype=dtype)
        full_heatmaps[:, z_min:z_max, y_min:y_max, x_min:x_max] = convolved_sub

        return full_heatmaps

# mean 0.75 ms | median 0.52 ms | min 0.33 ms | max 4.97 ms
class CroppedSparseGaussianSplat(HeatmapDatasetV2):
    @classmethod
    def make_heatmaps(cls, all_zyxs, min_corner_zyx, crop_size):
        crop_size_int = int(crop_size)
        radius = kernel_size // 2
        dtype = all_zyxs[0].dtype
        device = all_zyxs[0].device
        min_corner = min_corner_zyx.to(device=device, dtype=dtype)

        all_coords = []
        all_channels = []
        for zyxs in all_zyxs:
            coords = (zyxs.to(device=device, dtype=dtype) - min_corner + 0.5).int()
            valid_mask = (coords >= 0).all(dim=1) & (coords < crop_size_int).all(dim=1)
            if not torch.any(valid_mask):
                continue
            all_coords.append(coords[valid_mask])
            all_channels.append(torch.arange(zyxs.shape[0], device=device)[valid_mask])

        if len(all_coords) == 0:
            return torch.zeros((all_zyxs[0].shape[0], crop_size_int, crop_size_int, crop_size_int), device=device, dtype=dtype)

        coords_cat = torch.cat(all_coords, dim=0)
        channels_cat = torch.cat(all_channels, dim=0)

        z_min = max(0, int(coords_cat[:, 0].min().item()) - radius)
        y_min = max(0, int(coords_cat[:, 1].min().item()) - radius)
        x_min = max(0, int(coords_cat[:, 2].min().item()) - radius)
        z_max = min(crop_size_int, int(coords_cat[:, 0].max().item()) + radius + 1)
        y_max = min(crop_size_int, int(coords_cat[:, 1].max().item()) + radius + 1)
        x_max = min(crop_size_int, int(coords_cat[:, 2].max().item()) + radius + 1)

        sub_shape = (z_max - z_min, y_max - y_min, x_max - x_min)
        channels = all_zyxs[0].shape[0]

        heatmaps_sub = torch.zeros((*sub_shape, channels), device=device, dtype=dtype)
        kernel_t = kernel.to(device=device, dtype=dtype)
        rel_coords = coords_cat - torch.tensor([z_min, y_min, x_min], device=device, dtype=coords_cat.dtype)

        for coord, channel_idx in zip(rel_coords, channels_cat):
            z, y, x = int(coord[0]), int(coord[1]), int(coord[2])
            z0 = max(0, z - radius)
            y0 = max(0, y - radius)
            x0 = max(0, x - radius)
            z1 = min(sub_shape[0], z + radius + 1)
            y1 = min(sub_shape[1], y + radius + 1)
            x1 = min(sub_shape[2], x + radius + 1)

            kz0 = z0 - (z - radius)
            ky0 = y0 - (y - radius)
            kx0 = x0 - (x - radius)
            kz1 = kz0 + (z1 - z0)
            ky1 = ky0 + (y1 - y0)
            kx1 = kx0 + (x1 - x0)

            heatmaps_sub[z0:z1, y0:y1, x0:x1, int(channel_idx)] += kernel_t[kz0:kz1, ky0:ky1, kx0:kx1]

        full_heatmaps = torch.zeros((crop_size_int, crop_size_int, crop_size_int, channels), device=device, dtype=dtype)
        full_heatmaps[z_min:z_max, y_min:y_max, x_min:x_max] = heatmaps_sub

        # Match HeatmapDatasetV2 return layout: (channels, z, y, x)
        return rearrange(full_heatmaps, 'z y x c -> c z y x')
