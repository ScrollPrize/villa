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
from dataset import get_crop_from_volume, build_localiser
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
    
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    
    accelerator = accelerate.Accelerator(
        mixed_precision=config['mixed_precision'],
    )

    # TODO: share this code with train.py!
    model = Vesuvius3dUnetModel(in_channels=8, out_channels=6, config=config)
    # model = SongUnet3dModel(
    #     img_resolution=config['crop_size'],
    #     in_channels=8,
    #     out_channels=6,
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
    print(f"volume shape: {volume.shape}, dtype: {volume.dtype}, voxel-size: {voxel_size_um}um")

    def get_heatmaps_at(zyx):
        volume_crop, min_corner_zyx = get_crop_from_volume(volume, zyx, config['crop_size'])
        localiser = build_localiser(zyx, min_corner_zyx, config['crop_size'])
        inputs = torch.cat([
            volume_crop[None, None].to(accelerator.device),
            localiser[None, None].to(accelerator.device),
            torch.zeros([1, config['step_count'], *volume_crop.shape], device=accelerator.device),
        ], dim=1)
        timesteps = torch.zeros([1], dtype=torch.long, device=accelerator.device)
        logits = model(inputs, timesteps).squeeze(0)[:steps_per_crop]
        # TODO: test-time augmentation! as well as flip/rotate also consider very small spatial jitters etc
        return F.sigmoid(logits), min_corner_zyx

    def get_blob_coordinates(heatmaps, min_corner_zyx):
        centroids_by_step = []
        for heatmap in heatmaps:
            if False:
                points = torch.stack(torch.where(heatmap > 0.5), dim=-1).cpu().numpy()
                kmeans = sklearn.cluster.KMeans(n_clusters=4)
                kmeans.fit(points)
                print(kmeans.cluster_centers_)
            else:
                cc_labels = cc3d.connected_components((heatmap > 0.5).cpu().numpy(), connectivity=18, binary_image=True)
                cc_labels, num_ccs = cc3d.dust(cc_labels, threshold=40, precomputed_ccl=True, return_N=True)
                cc_stats = cc3d.statistics(cc_labels)
                centroid_zyxs = cc_stats['centroids'][1:] + min_corner_zyx.numpy()
                if num_ccs > 4:
                    size_order = np.argsort(-cc_stats['voxel_counts'][1:])
                    centroid_zyxs = centroid_zyxs[size_order[:4]]
                centroids_by_step.append(torch.from_numpy(centroid_zyxs))
                # if len(centroids_by_distance) == 0:
                #     centroids_by_distance.append(centroid_zyxs)
                # else:
                #     # TODO: this is non-trivial when some blobs are missing then/now
                #     #  ideally we want to look back further in time
                #     #  also can use CW/ACW-ness to resolve some points if we have others
                #     #  trickiest case is if we have two blobs in previous and two in current, but they're disjoint!
                #     distances = centroids_by_distance[-1] - centroid_zyxs
                #     centroids_by_distance.append(centroid_zyxs)
        # TODO: *might* be useful to 'connect' the blobs across planes, associating with the nearest adjacent ones (hungarian / greedy)
        #  ideally would globally optimise this across all four points and all planes; however maybe overkill!
        #  in that case, discretely assign each blob to of t>1 to
        return centroids_by_step

    with torch.inference_mode():

        start_zyx = torch.tensor(start_xyz).flip(0) / 2 ** volume_scale
        trace_zyxs = [start_zyx]
        for global_step in tqdm(range(50)):  # loop over planning windows
            heatmaps, min_corner_zyx = get_heatmaps_at(trace_zyxs[-1])
            coordinates = get_blob_coordinates(heatmaps, min_corner_zyx)
            assert steps_per_crop == 1  # FIXME: the logic for deciding which candidate to take otherwise changes -- for 'later' points, either always just take nearest to one before, or track correspondences properly in get_blob_coordinates, or think about 'long distance' symmetry
            for step_in_crop in range(steps_per_crop):  # loop over steps within the current planning window
                candidates = coordinates[step_in_crop]
                if len(trace_zyxs) == 1:
                    # For very first step, move in the -z direction for now
                    # TODO: should allow the user to specify the starting direction
                    assert step_in_crop == 0
                    delta_zs = (candidates - trace_zyxs[-1])[0]
                    best_candidate_idx = torch.argmin(delta_zs)
                else:
                    previous = trace_zyxs[-2]
                    nearest_to_previous = torch.argmin(torch.linalg.norm(candidates - previous, dim=-1))
                    # TODO: if candidates[nearest_to_previous] is too far from previous (i.e. the min is too large), don't trust this step
                    # TODO: tricky to find the correct point among the four; the right one is the one that continues the current line in the geodesic sense (i.e. in the unrolled surface)
                    #  - consider geometry in lstsq plane (still this fails to account for geodesic-ness)
                    #  - no possibility of working with full surface (because we don't know it); maybe could at least use symmetry (maybe not in planar projection) to account for the case of a 'squashed diamond'
                    #  - could use similarity of directions -- presumably the current-to-next direction should be more parallel with the previous-to-current direction than the others
                    #  - in the extreme case of a right-angle bend (with fold at current point, perpendicular to direction from previous), there is no 'per point' local measure that'll work
                    # Assume the most-distant point is the 'next' point to continue with; this is blind to the fact that the geodesic distances/directions are what matter
                    best_candidate_idx = torch.argmax(torch.linalg.norm(candidates - candidates[nearest_to_previous], dim=-1))
                trace_zyxs.append(candidates[best_candidate_idx])

        trace_zyxs = torch.stack(trace_zyxs, dim=0)
        quads = torch.stack([trace_zyxs + torch.tensor([0, -24, 0]), trace_zyxs + torch.tensor([0, -18, 0]), trace_zyxs + torch.tensor([0, -12, 0]), trace_zyxs + torch.tensor([0, -6, 0]), trace_zyxs, trace_zyxs + torch.tensor([0, 0, 6]), trace_zyxs + torch.tensor([0, 0, 12]), trace_zyxs + torch.tensor([0, 0, 18]), trace_zyxs + torch.tensor([0, 0, 24])], dim=1)
        quads *= 2 ** volume_scale
        save_tifxyz(quads.numpy(), f'{out_path}', 'neural-trace', config['step_size'], voxel_size_um, 'neural-tracer')


if __name__ == '__main__':
    trace()
