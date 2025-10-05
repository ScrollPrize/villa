import os
import cv2
import json
import cc3d
import click
import torch
import diffusers
import accelerate
import numpy as np
import random
import zarr
import sklearn.cluster
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt
import torch.nn.functional as F

from vesuvius_unet3d import Vesuvius3dUnetModel
from song_unet3d import SongUnet3dModel
from dataset import get_crop_from_volume, build_localiser, make_heatmaps
from tifxyz import save_tifxyz


@click.command()
@click.option('--config_path', type=click.Path(exists=True), required=True, help='Path to config file')
@click.option('--checkpoint_path', type=click.Path(exists=True), required=True, help='Path to checkpoint file')
@click.option('--out_path', type=click.Path(), required=True, help='Path to write surface to')
@click.option('--start_xyz', nargs=3, type=int, required=True, help='Starting XYZ coordinates')
@click.option('--volume_zarr', type=click.Path(exists=True), required=True, help='Path to ome-zarr folder')
@click.option('--volume_scale', type=int, required=True, help='OME scale to use')
@click.option('--steps_per_crop', type=int, required=True, help='Number of steps to take before sampling a new crop')
def trace(config_path, checkpoint_path, out_path, start_xyz, volume_zarr, volume_scale, steps_per_crop):

    with open(config_path, 'r') as f:
        config = json.load(f)

    assert steps_per_crop <= config['step_count']
    crop_size = config['crop_size']
    
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    
    accelerator = accelerate.Accelerator(
        mixed_precision=config['mixed_precision'],
    )

    # TODO: share this code with train.py!
    model = Vesuvius3dUnetModel(in_channels=5, out_channels=config['step_count'] * 2, config=config)
    # model = SongUnet3dModel(
    #     img_resolution=crop_size,
    #     in_channels=5,
    #     out_channels=config['step_count'] * 2,
    #     model_channels=32,
    #     channel_mult=[1, 2, 4, 8],
    #     num_blocks=2,  # 4
    #     encoder_type='residual',
    #     dropout=0.1,
    #     attn_resolutions=[16, 8],
    # )

    print('loading checkpoint')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    noise_scheduler = diffusers.DDPMScheduler.from_config(checkpoint.get('noise_scheduler', None))
    del checkpoint
    
    model = accelerator.prepare(model)
    model.eval()
    
    print("loading volume zarr")
    ome_zarr = zarr.open_group(volume_zarr, mode='r')
    volume = ome_zarr[str(volume_scale)]
    with open(f'{volume_zarr}/meta.json', 'rt') as meta_fp:
        voxel_size_um = json.load(meta_fp)['voxelsize']
    print(f"volume shape: {volume.shape}, dtype: {volume.dtype}, voxel-size: {voxel_size_um * 2 ** volume_scale}um")

    def get_heatmaps_at(zyx, prev_u, prev_v, prev_diag):
        volume_crop, min_corner_zyx = get_crop_from_volume(volume, zyx, crop_size)
        localiser = build_localiser(zyx, min_corner_zyx, crop_size)
        prev_u_heatmap = make_heatmaps([prev_u[None]], min_corner_zyx, crop_size) if prev_u is not None else torch.zeros([1, crop_size, crop_size, config['crop_size']])
        prev_v_heatmap = make_heatmaps([prev_v[None]], min_corner_zyx, crop_size) if prev_v is not None else torch.zeros([1, crop_size, crop_size, config['crop_size']])
        prev_diag_heatmap = make_heatmaps([prev_diag[None]], min_corner_zyx, crop_size) if prev_diag is not None else torch.zeros([1, crop_size, crop_size, config['crop_size']])
        inputs = torch.cat([
            volume_crop[None, None].to(accelerator.device),
            localiser[None, None].to(accelerator.device),
            prev_u_heatmap[None].to(accelerator.device),
            prev_v_heatmap[None].to(accelerator.device),
            prev_diag_heatmap[None].to(accelerator.device),
        ], dim=1)
        timesteps = torch.zeros([1], dtype=torch.long, device=accelerator.device)
        logits = model(inputs, timesteps).squeeze(0).reshape(2, config['step_count'], crop_size, crop_size, crop_size)  # u/v, step, z, y, x
        # TODO: test-time augmentation! as well as flip/rotate also consider very small spatial jitters etc
        return F.sigmoid(logits[:, :steps_per_crop]), min_corner_zyx

    def get_blob_coordinates(heatmap, min_corner_zyx, threshold=0.5, min_size=40):
        # Find up to four blobs of sufficient size; return their centroids in descending order of blob size
        # TODO: strip blobs that are further than K * step_size in euclidean space
        cc_labels = cc3d.connected_components((heatmap > threshold).cpu().numpy(), connectivity=18, binary_image=True)
        cc_labels, num_ccs = cc3d.dust(cc_labels, threshold=min_size, precomputed_ccl=True, return_N=True)
        cc_stats = cc3d.statistics(cc_labels)
        centroid_zyxs = cc_stats['centroids'][1:] + min_corner_zyx.numpy()
        size_order = np.argsort(-cc_stats['voxel_counts'][1:])[:4]
        centroid_zyxs = centroid_zyxs[size_order]
        return torch.from_numpy(centroid_zyxs.astype(np.float32))

    def trace_strip(start_zyx, num_steps, direction):

        assert steps_per_crop == 1  # TODO...

        # Get hopefully-4 adjacent points; take the one with min or max z-displacement depending on required direction
        heatmaps, min_corner_zyx = get_heatmaps_at(start_zyx, prev_u=None, prev_v=None, prev_diag=None)
        coordinates = get_blob_coordinates(heatmaps[:, 0].amax(dim=0), min_corner_zyx)
        save_point_collection('coordinates.json', torch.cat([start_zyx[None], coordinates], dim=0))
        if direction == 'u':  # use maximum delta-z
            best_idx = torch.argmax((coordinates - start_zyx)[:, 0].abs())
        elif direction == 'v':  # use minimum delta-z
            best_idx = torch.argmin((coordinates - start_zyx)[:, 0].abs())
        else:
            assert False
        trace_zyxs = [start_zyx, coordinates[best_idx]]

        for step in tqdm(range(num_steps // steps_per_crop), desc='tracing strip'):  # loop over planning windows

            # Query the model 'along' the relevant direction, with crop centered at current point and previous as conditioning
            if direction == 'u':
                prev_uv = {'prev_u': trace_zyxs[-2], 'prev_v': None}
            else:
                prev_uv = {'prev_v': trace_zyxs[-2], 'prev_u': None}
            heatmaps, min_corner_zyx = get_heatmaps_at(trace_zyxs[-1], **prev_uv, prev_diag=None)
            coordinates = get_blob_coordinates(heatmaps[0 if direction == 'u' else 1].squeeze(0), min_corner_zyx)

            if len(coordinates) == 0 or coordinates[0].isnan().any():
                break

            # Take the largest (0th) blob centroid as the next point
            next_zyx = coordinates[0]
            trace_zyxs.append(next_zyx)

        return torch.stack(trace_zyxs, dim=0)

    def trace_patch(first_row_zyxs, num_steps, direction):
        # For simplicity we denote the start_zyxs as a 'row', with 'columns' ordered 'left' to 'right'; however this has no geometric significance!
        # direction parameter controls which direction we're growing in; the initial strip should be perpendicular

        assert steps_per_crop == 1  # TODO!
        rows = [first_row_zyxs]
        for row_idx in tqdm(range(1, num_steps // steps_per_crop), desc='tracing patch'):
            next_row = []
            for col_idx in tqdm(range(rows[-1].shape[0]), 'tracing row'):

                # TODO: could grow more 'triangularly' -- would increase the support for the top row, hence maybe more robust

                # TODO: could grow bidirectionally

                # TODO: more generally, can just be a single loop over points in *arbitrary* order, and we figure out what already exists in the patch to use as conditioning
                #  that'd nicely separate the growing strategy from constructing the conditioning signal

                # TODO: fallback conditioning inputs for fail (no-blob / NaN) cases

                assert direction == 'u'  # TODO! need to conditionally flip u/v in each of the following cases
                if row_idx == 1 and col_idx == 0:  # first point of second row
                    # Conditioned on center and right, predict below & above
                    # Note this one is ambiguous for +/- direction and we choose arbitrarily below, based on largest blob
                    heatmaps, min_corner_zyx = get_heatmaps_at(rows[0][0], prev_u=None, prev_v=rows[0][1], prev_diag=None)
                elif row_idx == 1:  # later points of second row
                    # Conditioned on center and left and below-left, predict below
                    heatmaps, min_corner_zyx = get_heatmaps_at(rows[0][col_idx], prev_u=None, prev_v=rows[0][col_idx - 1], prev_diag=next_row[-1])
                elif col_idx == 0:  # first point of later rows
                    # Conditioned on center and above and right, predict below
                    heatmaps, min_corner_zyx = get_heatmaps_at(rows[-1][0], prev_u=rows[-2][0], prev_v=rows[-1][1], prev_diag=None)
                else:  # later points of later rows
                    # Conditioned on center and left and above and below-left, predict below
                    heatmaps, min_corner_zyx = get_heatmaps_at(rows[-1][col_idx], prev_u=rows[-2][col_idx], prev_v=rows[-1][col_idx - 1], prev_diag=next_row[-1])

                coordinates = get_blob_coordinates(heatmaps[0 if direction == 'u' else 1].squeeze(0), min_corner_zyx)
                if len(coordinates) == 0 or coordinates[0].isnan().any():
                    print('warning: no point found!')
                    next_row.extend([torch.tensor([-1, -1, -1])] * (first_row_zyxs.shape[0] - len(next_row)))
                    break

                # Take the largest (0th) blob centroid as the next point
                next_zyx = coordinates[0]
                next_row.append(next_zyx)

            rows.append(torch.stack(next_row, dim=0))
        return torch.stack(rows, dim=0)

    def save_point_collection(filename, zyxs):
        with open(filename, 'wt') as fp:
            json.dump({
                'collections': {
                    '0': {
                        'name': 'strip',
                        'color': [1.0, 0.5, 0.5],
                        'metadata': {'winding_is_absolute': False},
                        'points': {
                            str(idx): {
                                'creation_time': 1000,
                                'p': (zyxs[idx].flip(0) * 2 ** volume_scale).tolist(),
                                'wind_a': 1.,
                            }
                            for idx in range(zyxs.shape[0])
                        }
                    }
                },
                'vc_pointcollections_json_version': '1'
            }, fp)

    with torch.inference_mode():

        strip_direction = 'v'
        start_zyx = torch.tensor(start_xyz).flip(0) / 2 ** volume_scale
        strip_zyxs = trace_strip(start_zyx, num_steps=50, direction=strip_direction)
        save_point_collection(f'points_{strip_direction}.json', strip_zyxs)

        patch_zyxs = trace_patch(strip_zyxs, num_steps=50, direction='v' if strip_direction == 'u' else 'u')
        save_point_collection(f'points_patch.json', patch_zyxs.view(-1, 3))
        save_tifxyz((patch_zyxs * 2 ** volume_scale).numpy(), f'{out_path}', 'neural-trace-patch', config['step_size'] * 2 ** volume_scale, voxel_size_um, 'neural-tracer')
        plt.plot(*patch_zyxs.view(-1, 3)[:, [0, 1]].T, 'r.')
        plt.savefig('patch.png')
        plt.close()


if __name__ == '__main__':
    trace()
