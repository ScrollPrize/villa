import torch
import random
import networkx as nx
from einops import rearrange
from vesuvius.neural_tracing.dataset import load_datasets, get_zyx_from_patch, get_crop_from_volume, build_localiser
import vesuvius.neural_tracing.augmentation as augmentation

class PatchInCubeDataset(torch.utils.data.IterableDataset):

    def __init__(self, config):
        self._config = config
        self._patches = load_datasets(config)
        self._augmentations = augmentation.get_training_augmentations(config['crop_size'],
                                                                      config['augmentation']['no_spatial'],
                                                                      config['augmentation'][
                                                                          'only_spatial_and_intensity'])

    def __iter__(self):

        areas = torch.tensor([patch.area for patch in self._patches])
        area_weights = areas / areas.sum()
        # context_point_distance = self._config['context_point_distance']
        # num_context_points = self._config['num_context_points']
        # assert num_context_points >= 1  # ...since we always include one point at the center of the patch
        crop_size = torch.tensor(self._config['crop_size'])

        while True:
            patch = random.choices(self._patches, weights=area_weights)[0]

            # Sample a random valid quad in the patch, then a point in that quad
            random_idx = torch.randint(len(patch.valid_quad_indices) - 1, size=[])
            start_quad_ij = patch.valid_quad_indices[random_idx]
            center_ij = start_quad_ij + torch.rand(size=[2])
            # TODO: maybe erode inwards before doing this, so we can directly
            #  avoid sampling points that are too near the edge of the patch

            # # Sample other nearby random points as additional context; reject if outside patch
            # angle = torch.rand(size=[num_context_points]) * 2 * torch.pi
            # distance = context_point_distance * patch.scale
            # context_ij = center_ij + distance * torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
            # if torch.any(context_ij < 0) or torch.any(context_ij >= torch.tensor(patch.zyxs.shape[:2])):
            #     continue
            # if not patch.valid_quad_mask[*context_ij.int().T].all():
            #     continue

            center_zyx = get_zyx_from_patch(center_ij, patch)
            # context_zyxs = [self._get_zyx_from_patch(context_ij, patch) for context_ij in context_ij]

            # Crop ROI out of the volume; mark context points
            volume_crop, min_corner_zyx = get_crop_from_volume(patch.volume, center_zyx, crop_size)
            # self._mark_context_point(volume_crop, center_zyx - min_corner_zyx)
            # for context_zyx in context_zyxs:
            #     self._mark_context_point(volume_crop, context_zyx - min_corner_zyx)

            # Find quads that are in the volume crop, and reachable from the start quad without leaving the crop

            # FIXME: instead check any corner in crop, and clamp to bounds later
            quad_centers = 0.5 * (patch.zyxs[1:, 1:] + patch.zyxs[:-1, :-1])
            quad_in_crop = patch.valid_quad_mask & torch.all(quad_centers >= min_corner_zyx, dim=-1) & torch.all(
                quad_centers < min_corner_zyx + crop_size, dim=-1)

            # Build neighbor graph of quads that are in the volume crop
            G = nx.Graph()
            quad_indices = torch.stack(torch.where(quad_in_crop), dim=-1)

            # Add nodes for each quad in crop using (i, j) as node ID
            for i, j in quad_indices:
                G.add_node((i.item(), j.item()))

            # Add edges between neighboring quads
            for i, j in quad_indices:
                # Check 4-connected neighbors
                neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                for ni, nj in neighbors:
                    if (0 <= ni < quad_in_crop.shape[0] and
                            0 <= nj < quad_in_crop.shape[1] and
                            quad_in_crop[ni, nj]):
                        G.add_edge((i.item(), j.item()), (ni.item(), nj.item()))

            # Find reachable quads starting from start_quad_ij
            start_node = (start_quad_ij[0].item(), start_quad_ij[1].item())
            if not G.has_node(start_node):
                print('WARNING: start_quad_ij not in crop')
                continue
            reachable_quads = nx.node_connected_component(G, start_node)
            reachable_quads = torch.tensor(list(reachable_quads))

            # Create new mask with only reachable quads
            quad_reachable_in_crop = torch.zeros_like(quad_in_crop)
            quad_reachable_in_crop[*reachable_quads.T] = True

            # Rasterise the (cropped, reachable) surface patch
            filtered_quads_zyxs = torch.stack([
                torch.stack([
                    patch.zyxs[:-1, :-1][quad_reachable_in_crop],
                    patch.zyxs[:-1, 1:][quad_reachable_in_crop],
                ], dim=1),
                torch.stack([
                    patch.zyxs[1:, :-1][quad_reachable_in_crop],
                    patch.zyxs[1:, 1:][quad_reachable_in_crop],
                ], dim=1),
            ], dim=1)  # quad, top/bottom, left/right, zyx
            oversample_factor = 2
            points_per_side = (1 / patch.scale + 0.5).int() * oversample_factor
            v_points = torch.arange(points_per_side[0], dtype=torch.float32) / points_per_side[0]
            u_points = torch.arange(points_per_side[1], dtype=torch.float32) / points_per_side[1]
            points_covering_quads = torch.lerp(filtered_quads_zyxs[:, None, 0, :], filtered_quads_zyxs[:, None, 1, :],
                                               v_points[None, :, None, None])
            points_covering_quads = torch.lerp(points_covering_quads[:, :, None, 0],
                                               points_covering_quads[:, :, None, 1], u_points[None, None, :, None])
            indices_in_crop = (points_covering_quads - min_corner_zyx + 0.5).int().clip(0, crop_size - 1)
            rasterised = torch.zeros([crop_size, crop_size, crop_size], dtype=torch.float32)
            rasterised[*indices_in_crop.view(-1, 3).T] = 1.

            # Construct UV map on the surface
            filtered_quads_uvs = torch.stack(torch.where(quad_reachable_in_crop), dim=-1)
            uv_grid = torch.stack(torch.meshgrid(u_points, v_points, indexing='ij'), dim=-1)
            interpolated_uvs = filtered_quads_uvs[:, None, None, :] + uv_grid
            uvs = torch.zeros([crop_size, crop_size, crop_size, 2], dtype=torch.float32)
            uvs[*indices_in_crop.view(-1, 3).T] = interpolated_uvs.view(-1, 2)

            # Extend into free space: find EDT, and use feature transform to get nearest UV
            edt, ft = scipy.ndimage.morphology.distance_transform_edt((rasterised == 0).numpy(), return_indices=True)
            edt /= ((
                                crop_size // 2) ** 2 * 3) ** 0.5 + 1.  # worst case: only center point is fg, hence max(edt) is 'radius' to corners
            edt = torch.exp(-edt / 0.25)
            edt = edt.to(torch.float32) * 2 - 1  # ...so it's "signed" but the zero point is arbitrary
            uvs = uvs[*ft]
            uvs = (uvs - center_ij) / patch.scale / (
                        crop_size * 2)  # *2 is somewhat arbitrary; worst-ish-case = patch wrapping round three sides of cube
            uvws = torch.cat([uvs, edt[..., None]], dim=-1).to(torch.float32)

            localiser = build_localiser(center_zyx, min_corner_zyx, crop_size)

            # TODO: include full 2d slices for additional context
            #  if so, need to augment them consistently with the 3d crop -> tricky for geometric transforms

            # FIXME: the loop is a hack because some augmentation sometimes randomly returns None
            #  we should instead just remove the relevant augmentation (or fix it!)
            while True:
                augmented = self._augmentations(image=volume_crop[None], dist_map=torch.cat(
                    [localiser[None], rearrange(uvws, 'z y x c -> c z y x')], dim=0))
                if augmented['dist_map'] is not None:
                    break
            volume_crop = augmented['image'].squeeze(0)
            localiser = augmented['dist_map'][0]
            uvws = rearrange(augmented['dist_map'][1:], 'c z y x -> z y x c')
            if torch.any(torch.isnan(volume_crop)) or torch.any(torch.isnan(localiser)) or torch.any(torch.isnan(uvws)):
                # FIXME: why do these NaNs happen occasionally?
                continue

            yield {
                'volume': volume_crop,
                'localiser': localiser,
                'uvw': uvws,
            }
