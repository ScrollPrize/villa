import json
import shutil
import click
import torch
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from models import make_model, load_checkpoint
from infer import Inference
from tifxyz import save_tifxyz, get_area


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
    step_size = config['step_size'] * 2 ** volume_scale
    
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    model = make_model(config)
    load_checkpoint(checkpoint_path, model)
    inference = Inference(model, config, volume_zarr, volume_scale)

    partial_uuids = []

    def trace_strip(start_zyx, num_steps, direction):

        assert steps_per_crop == 1  # TODO...

        # Get hopefully-4 adjacent points; take the one with min or max z-displacement depending on required direction
        heatmaps, min_corner_zyx = inference.get_heatmaps_at(start_zyx, prev_u=None, prev_v=None, prev_diag=None)
        coordinates = inference.get_blob_coordinates(heatmaps[:, 0].amax(dim=0), min_corner_zyx)
        if len(coordinates) == 0 or coordinates[0].isnan().any():
            print('no blobs found at step 0')
            return torch.empty([0, 0, 3])
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
            heatmaps, min_corner_zyx = inference.get_heatmaps_at(trace_zyxs[-1], **prev_uv, prev_diag=None)
            coordinates = inference.get_blob_coordinates(heatmaps[0 if direction == 'u' else 1, 0], min_corner_zyx)

            if len(coordinates) == 0 or coordinates[0].isnan().any():
                print(f'no blobs found at step {step + 1}')
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
                #  this would allow working around holes, in theory)
                #  could prioritise maximally-supported points where possible, which effectively makes it do something triangle-like

                # TODO: fallback conditioning inputs for fail (no-blob / NaN) cases
                #  ...so try to grow a point rightwards instead of downwards

                if row_idx == 1 and col_idx == 0:  # first point of second row
                    # Conditioned on center and right, predict below & above
                    # Note this one is ambiguous for +/- direction and we choose arbitrarily below, based on largest blob
                    prev_u = None
                    prev_v = rows[0][1]
                    prev_diag = None
                elif row_idx == 1:  # later points of second row
                    # Conditioned on center and left and below-left, predict below
                    prev_u = None
                    prev_v = rows[0][col_idx - 1]
                    prev_diag = next_row[-1]
                elif col_idx == 0:  # first point of later rows
                    # Conditioned on center and above and right, predict below
                    prev_u=rows[-2][0]
                    prev_v = rows[-1][1]
                    prev_diag = None
                else:  # later points of later rows
                    # Conditioned on center and left and above and below-left, predict below
                    prev_u = rows[-2][col_idx]
                    prev_v = rows[-1][col_idx - 1]
                    prev_diag = next_row[-1]

                if direction == 'v':
                    prev_u, prev_v = prev_v, prev_u
                heatmaps, min_corner_zyx = inference.get_heatmaps_at(rows[-1][col_idx], prev_u=prev_u, prev_v=prev_v, prev_diag=prev_diag)
                coordinates = inference.get_blob_coordinates(heatmaps[0 if direction == 'u' else 1, 0], min_corner_zyx)
                if len(coordinates) == 0 or coordinates[0].isnan().any():
                    print('warning: no point found!')
                    next_row.extend([torch.tensor([-1, -1, -1])] * (first_row_zyxs.shape[0] - len(next_row)))
                    break

                # Take the largest (0th) blob centroid as the next point
                next_zyx = coordinates[0]
                next_row.append(next_zyx)

            rows.append(torch.stack(next_row, dim=0))
        return torch.stack(rows, dim=0)

    def trace_patch_v4(start_zyx, max_size):

        patch = np.full([max_size, max_size, 3], -1.)
        center_uv = np.array([max_size // 2, max_size // 2])
        patch[*center_uv] = start_zyx

        def in_patch(u, v):
            return 0 <= u < max_size and 0 <= v < max_size and not (patch[u, v] == -1).all()

        candidate_centers_and_gaps = []

        def enqueue_gaps_around_center(center_uv):
            center_uv = np.asarray(center_uv)
            candidate_centers_and_gaps.extend([
                (tuple(center_uv), tuple(center_uv + delta))
                for delta in [[0, -1], [0, 1], [-1, 0], [1, 0]]
                if not in_patch(*(center_uv + delta))
                    and (center_uv + delta >= 0).all() and (center_uv + delta < max_size).all()
            ])

        enqueue_gaps_around_center(center_uv)

        def get_conditioning(center, gap):
            assert in_patch(*center)
            assert not in_patch(*gap)
            center_u, center_v = center
            gap_u, gap_v = gap

            if gap_u == center_u:
                if in_patch(center_u - 1, center_v):
                    prev_u = center_u - 1
                elif in_patch(center_u + 1, center_v):
                    prev_u = center_u + 1
                else:
                    prev_u = None
            elif gap_u > center_u:
                if in_patch(center_u - 1, center_v):
                    prev_u = center_u - 1
                else:
                    prev_u = None
            else:  # gap_u < center_u
                if in_patch(center_u + 1, center_v):
                    prev_u = center_u + 1
                else:
                    prev_u = None

            if gap_v == center_v:
                if in_patch(center_u, center_v - 1):
                    prev_v = center_v - 1
                elif in_patch(center_u, center_v + 1):
                    prev_v = center_v + 1
                else:
                    prev_v = None
            elif gap_v > center_v:
                if in_patch(center_u, center_v - 1):
                    prev_v = center_v - 1
                else:
                    prev_v = None
            else:  # gap_v < center_v
                if in_patch(center_u, center_v + 1):
                    prev_v = center_v + 1
                else:
                    prev_v = None

            # TODO: missing any cases here?
            if prev_u is not None and prev_v is None:
                if gap_v != center_v and in_patch(prev_u, gap_v):
                    prev_diag = prev_u, gap_v
                else:
                    prev_diag = None
            elif prev_v is not None and prev_u is None:
                if gap_u != center_u and in_patch(gap_u, prev_v):
                    prev_diag = gap_u, prev_v
                else:
                    prev_diag = None
            elif prev_u is not None and prev_v is not None:
                if gap_u != center_u:
                    assert gap_v == center_v
                    if in_patch(gap_u, prev_v):
                        prev_diag = gap_u, prev_v
                    else:
                        prev_diag = None
                elif gap_v != center_v:
                    assert gap_u == center_u
                    if in_patch(prev_u, gap_v):
                        prev_diag = prev_u, gap_v
                    else:
                        prev_diag = None
                else:
                    assert False
            else:
                prev_diag = None

            return (
                torch.tensor([prev_u, center_v]) if prev_u is not None else None,
                torch.tensor([center_u, prev_v]) if prev_v is not None else None,
                torch.tensor(prev_diag) if prev_diag is not None else None,
            )

        def count_conditionings(conditionings):
            prev_u, prev_v, prev_diag = conditionings
            return sum([prev_u is not None, prev_v is not None, prev_diag is not None])

        tried_cag_and_conditionings = set()
        num_vertices = 1

        while len(candidate_centers_and_gaps) > 0 and num_vertices < max_size**2:

            # For each center-and-gap, get available conditioning points; choose the center-and-gap
            # with most support that we've not already tried (with its current conditionings)
            cag_to_conditionings = {(center, gap): get_conditioning(center, gap) for center, gap in candidate_centers_and_gaps}
            cag_to_conditionings = {(center, gap): conditioning for (center, gap), conditioning in cag_to_conditionings.items() if (center, gap, conditioning) not in tried_cag_and_conditionings}
            if len(cag_to_conditionings) == 0:
                print('halting: no untried center-gap-conditioning combinations')
                break
            center_and_gap = max(cag_to_conditionings, key=lambda center_and_gap: count_conditionings(cag_to_conditionings[center_and_gap]))
            tried_cag_and_conditionings.add((*center_and_gap, cag_to_conditionings[center_and_gap]))
            center_zyx = patch[*center_and_gap[0]]
            center_uv, gap_uv = center_and_gap
            assert in_patch(*center_uv)
            assert not in_patch(*gap_uv)
            def get_prev_zyx(uv):
                if uv is None or (uv < 0).any() or (uv >= max_size).any():
                    return None
                assert in_patch(*uv)
                return torch.from_numpy(patch[*uv])
            prev_u, prev_v, prev_diag = map(get_prev_zyx, cag_to_conditionings[center_and_gap])

            # Query model at this point; take the largest (0th) blob centroid to fill the gap-point
            heatmaps, min_corner_zyx = inference.get_heatmaps_at(torch.from_numpy(center_zyx), prev_u=prev_u, prev_v=prev_v, prev_diag=prev_diag)
            coordinates = inference.get_blob_coordinates(heatmaps[0 if gap_uv[0] != center_uv[0] else 1, 0], min_corner_zyx)
            if len(coordinates) == 0 or coordinates[0].isnan().any():
                print('warning: no point found!')
                continue
            patch[*gap_uv] = coordinates[0]
            num_vertices += 1

            # Update the set of candidate center-and-gap points -- remove any for the same gap
            # Add those that have the former gap as a center (and a new gap adjacent)
            candidate_centers_and_gaps = [
                (other_center_uv, other_gap_uv)
                for other_center_uv, other_gap_uv in candidate_centers_and_gaps
                if other_gap_uv != gap_uv
            ]
            enqueue_gaps_around_center(gap_uv)

            _, area_cm2 = get_area(patch, step_size, inference.voxel_size_um)
            print(f'vertex count = {num_vertices}, area = {area_cm2:.2f}cm2, queue size = {len(candidate_centers_and_gaps)} of which {len(candidate_centers_and_gaps) - len(cag_to_conditionings)} already tried')

            if num_vertices > 0 and num_vertices % 1000 == 0:
                partial_uuid = f'neural-trace-patch_{timestamp}_{num_vertices//1000:03}Kvert'
                partial_uuids.append(partial_uuid)
                save_tifxyz(
                    (np.where((patch == -1).all(-1, keepdims=True), -1, patch * 2 ** volume_scale)),
                    f'{out_path}',
                    partial_uuid,
                    step_size,
                    inference.voxel_size_um,
                    'neural-tracer',
                    {'seed': start_xyz}
                )

        return patch

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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        start_zyx = torch.tensor(start_xyz).flip(0) / 2 ** volume_scale

        if True:  # strip-then-extrude

            strip_direction = 'u'
            strip_zyxs = trace_strip(start_zyx, num_steps=100, direction=strip_direction)
            save_point_collection(f'points_{strip_direction}_{timestamp}.json', strip_zyxs)
            patch_zyxs = trace_patch(strip_zyxs, num_steps=100, direction='v' if strip_direction == 'u' else 'u')

        else:  # freeform 2D growth

            patch_zyxs = trace_patch_v4(start_zyx, max_size=100)

        print(f'saving with timestamp {timestamp}')
        save_tifxyz(
            patch_zyxs * 2 ** volume_scale,
            f'{out_path}',
            f'neural-trace-patch_{timestamp}',
            step_size,
            inference.voxel_size_um,
            'neural-tracer',
            {'seed': start_xyz}
        )
        for partial_uuid in partial_uuids:
            shutil.rmtree(f'{out_path}/{partial_uuid}')
        if False:  # useful for debugging
            save_point_collection(f'points_patch_{timestamp}.json', patch_zyxs.view(-1, 3))
            plt.plot(*patch_zyxs.view(-1, 3)[:, [0, 1]].T, 'r.')
            plt.show()
            plt.savefig('patch.png')
            plt.close()


if __name__ == '__main__':
    trace()
