from copy import deepcopy
from typing import Tuple, List, Union

import math

import SimpleITK
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import fourier_gaussian
from torch import Tensor
from torch.nn.functional import grid_sample

from models.augmentation.helpers.scalar_type import RandomScalar, sample_scalar
from models.augmentation.transforms.base.basic_transform import BasicTransform
from models.augmentation.transforms.utils.cropping import crop_tensor


class SpatialTransform(BasicTransform):
    def __init__(self,
                 patch_size: Tuple[int, ...],
                 patch_center_dist_from_border: Union[int, List[int], Tuple[int, ...]],
                 random_crop: bool,
                 p_elastic_deform: float = 0,
                 elastic_deform_scale: RandomScalar = (0, 0.2),
                 elastic_deform_magnitude: RandomScalar = (0, 0.2),
                 p_synchronize_def_scale_across_axes: float = 0,
                 p_rotation: float = 0,
                 rotation: RandomScalar = (0, 2 * np.pi),
                 p_scaling: float = 0,
                 scaling: RandomScalar = (0.7, 1.3),
                 p_synchronize_scaling_across_axes: float = 0,
                 bg_style_seg_sampling: bool = True,
                 mode_seg: str = 'bilinear'
                 ):
        """
        magnitude must be given in pixels!
        """
        super().__init__()
        self.patch_size = patch_size
        if not isinstance(patch_center_dist_from_border, (tuple, list)):
            patch_center_dist_from_border = [patch_center_dist_from_border] * len(patch_size)
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.random_crop = random_crop
        self.p_elastic_deform = p_elastic_deform
        self.elastic_deform_scale = elastic_deform_scale  # sigma for blurring offsets, in % of patch size. Larger values mean coarser deformation
        self.elastic_deform_magnitude = elastic_deform_magnitude  # determines the maximum displacement, measured in % of patch size
        self.p_rotation = p_rotation
        self.rotation = rotation
        self.p_scaling = p_scaling
        self.scaling = scaling  # larger numbers = smaller objects!
        self.p_synchronize_scaling_across_axes = p_synchronize_scaling_across_axes
        self.p_synchronize_def_scale_across_axes = p_synchronize_def_scale_across_axes
        self.bg_style_seg_sampling = bg_style_seg_sampling
        self.mode_seg = mode_seg

    def get_parameters(self, **data_dict) -> dict:
        dim = data_dict['image'].ndim - 1

        do_rotation = np.random.uniform() < self.p_rotation
        do_scale = np.random.uniform() < self.p_scaling
        do_deform = np.random.uniform() < self.p_elastic_deform

        if do_rotation:
            angles = [sample_scalar(self.rotation, image=data_dict['image'], dim=i) for i in range(0, 3)]
        else:
            angles = [0] * dim
        if do_scale:
            if np.random.uniform() <= self.p_synchronize_scaling_across_axes:
                scales = [sample_scalar(self.scaling, image=data_dict['image'], dim=None)] * dim
            else:
                scales = [sample_scalar(self.scaling, image=data_dict['image'], dim=i) for i in range(0, 3)]
        else:
            scales = [1] * dim

        # affine matrix
        if do_scale or do_rotation:
            if dim == 3:
                affine = create_affine_matrix_3d(angles, scales)
            elif dim == 2:
                affine = create_affine_matrix_2d(angles[-1], scales)
            else:
                raise RuntimeError(f'Unsupported dimension: {dim}')
        else:
            affine = None  # this will allow us to detect that we can skip computations

        # elastic deformation. We need to create the displacement field here
        # we use the method from augment_spatial_2 in batchgenerators
        if do_deform:
            if np.random.uniform() <= self.p_synchronize_def_scale_across_axes:
                deformation_scales = [
                    sample_scalar(self.elastic_deform_scale, image=data_dict['image'], dim=None, patch_size=self.patch_size)
                    ] * dim
            else:
                deformation_scales = [
                    sample_scalar(self.elastic_deform_scale, image=data_dict['image'], dim=i, patch_size=self.patch_size)
                    for i in range(0, 3)
                    ]

            # sigmas must be in pixels, as this will be applied to the deformation field
            sigmas = [i * j for i, j in zip(deformation_scales, self.patch_size)]

            magnitude = [
                sample_scalar(self.elastic_deform_magnitude, image=data_dict['image'], patch_size=self.patch_size,
                              dim=i, deformation_scale=deformation_scales[i])
                for i in range(0, 3)]
            # doing it like this for better memory layout for blurring
            offsets = torch.normal(mean=0, std=1, size=(dim, *self.patch_size))

            # all the additional time elastic deform takes is spent here
            for d in range(dim):
                # fft torch, slower
                # for i in range(offsets.ndim - 1):
                #     offsets[d] = blur_dimension(offsets[d][None], sigmas[d], i, force_use_fft=True, truncate=6)[0]

                # fft numpy, this is faster o.O
                tmp = np.fft.fftn(offsets[d].numpy())
                tmp = fourier_gaussian(tmp, sigmas[d])
                offsets[d] = torch.from_numpy(np.fft.ifftn(tmp).real)

                mx = torch.max(torch.abs(offsets[d]))
                offsets[d] /= (mx / np.clip(magnitude[d], a_min=1e-8, a_max=np.inf))
            offsets = torch.permute(offsets, (1, 2, 3, 0))
        else:
            offsets = None

        shape = data_dict['image'].shape[1:]
        if not self.random_crop:
            center_location_in_pixels = [i / 2 for i in shape]
        else:
            center_location_in_pixels = []
            for d in range(0, 3):
                mn = self.patch_center_dist_from_border[d]
                mx = shape[d] - self.patch_center_dist_from_border[d]
                if mx < mn:
                    center_location_in_pixels.append(shape[d] / 2)
                else:
                    center_location_in_pixels.append(np.random.uniform(mn, mx))
        return {
            'affine': affine,
            'elastic_offsets': offsets,
            'center_location_in_pixels': center_location_in_pixels
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if params['affine'] is None and params['elastic_offsets'] is None:
            # No spatial transformation is being done. Round grid_center and crop without having to interpolate.
            # This saves compute.
            # cropping requires the center to be given as integer coordinates
            img = crop_tensor(img, [math.floor(i) for i in params['center_location_in_pixels']], self.patch_size, pad_mode='constant',
                              pad_kwargs={'value': 0})
            return img
        else:
            grid = _create_centered_identity_grid2(self.patch_size)

            # we deform first, then rotate
            if params['elastic_offsets'] is not None:
                grid += params['elastic_offsets']
            if params['affine'] is not None:
                grid = torch.matmul(grid, torch.from_numpy(params['affine']).float())

            # we center the grid around the center_location_in_pixels. We should center the mean of the grid, not the center position
            # only do this if we elastic deform
            if params['elastic_offsets'] is not None:
                mn = grid.mean(dim=list(range(img.ndim - 1)))
            else:
                mn = 0

            new_center = torch.Tensor([c - s / 2 for c, s in zip(params['center_location_in_pixels'], img.shape[1:])])
            grid += (new_center - mn)
            return grid_sample(img[None], _convert_my_grid_to_grid_sample_grid(grid, img.shape[1:])[None],
                               mode='bilinear', padding_mode="zeros", align_corners=False)[0]

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        segmentation = segmentation.contiguous()
        if params['affine'] is None and params['elastic_offsets'] is None:
            # No spatial transformation is being done. Round grid_center and crop without having to interpolate.
            # This saves compute.
            # cropping requires the center to be given as integer coordinates
            segmentation = crop_tensor(segmentation,
                                       [math.floor(i) for i in params['center_location_in_pixels']],
                                       self.patch_size,
                                       pad_mode='constant',
                                       pad_kwargs={'value': 0})
            return segmentation
        else:
            grid = _create_centered_identity_grid2(self.patch_size)

            # we deform first, then rotate
            if params['elastic_offsets'] is not None:
                grid += params['elastic_offsets']
            if params['affine'] is not None:
                grid = torch.matmul(grid, torch.from_numpy(params['affine']).float())

            # we center the grid around the center_location_in_pixels. We should center the mean of the grid, not the center coordinate
            if params['elastic_offsets'] is not None:
                mn = grid.mean(dim=list(range(segmentation.ndim - 1)))
            else:
                mn = 0

            new_center = torch.Tensor([c - s / 2 for c, s in zip(params['center_location_in_pixels'], segmentation.shape[1:])])

            grid += (new_center - mn)
            grid = _convert_my_grid_to_grid_sample_grid(grid, segmentation.shape[1:])

            if self.mode_seg == 'nearest':
                result_seg = grid_sample(
                                segmentation[None].float(),
                                grid[None],
                                mode=self.mode_seg,
                                padding_mode="zeros",
                                align_corners=False
                            )[0].to(segmentation.dtype)
            else:
                result_seg = torch.zeros((segmentation.shape[0], *self.patch_size), dtype=segmentation.dtype)
                if self.bg_style_seg_sampling:
                    for c in range(segmentation.shape[0]):
                        labels = torch.from_numpy(np.sort(pd.unique(segmentation[c].numpy().ravel())))
                        # if we only have 2 labels then we can save compute time
                        if len(labels) == 2:
                            out = grid_sample(
                                    ((segmentation[c] == labels[1]).float())[None, None],
                                    grid[None],
                                    mode=self.mode_seg,
                                    padding_mode="zeros",
                                    align_corners=False
                                )[0][0] >= 0.5
                            result_seg[c][out] = labels[1]
                            result_seg[c][~out] = labels[0]
                        else:
                            for i, u in enumerate(labels):
                                result_seg[c][
                                    grid_sample(
                                        ((segmentation[c] == u).float())[None, None],
                                        grid[None],
                                        mode=self.mode_seg,
                                        padding_mode="zeros",
                                        align_corners=False
                                    )[0][0] >= 0.5] = u
                else:
                    for c in range(segmentation.shape[0]):
                        labels = torch.from_numpy(np.sort(pd.unique(segmentation[c].numpy().ravel())))
                        #torch.where(torch.bincount(segmentation.ravel()) > 0)[0].to(segmentation.dtype)
                        tmp = torch.zeros((len(labels), *self.patch_size), dtype=torch.float16)
                        scale_factor = 1000
                        done_mask = torch.zeros(*self.patch_size, dtype=torch.bool)
                        for i, u in enumerate(labels):
                            tmp[i] = grid_sample(((segmentation[c] == u).float() * scale_factor)[None, None], grid[None],
                                                 mode=self.mode_seg, padding_mode="zeros", align_corners=False)[0][0]
                            mask = tmp[i] > (0.7 * scale_factor)
                            result_seg[c][mask] = u
                            done_mask = done_mask | mask
                        if not torch.all(done_mask):
                            result_seg[c][~done_mask] = labels[tmp[:, ~done_mask].argmax(0)]
                        del tmp
            del grid
            return result_seg.contiguous()

    def _apply_to_dist_map(self, dist_map: torch.Tensor, **params) -> torch.Tensor:
        # Distance maps should be transformed like images (using bilinear interpolation)
        return self._apply_to_image(dist_map, **params)

    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        return self._apply_to_image(regression_target, **params)

    def _apply_to_keypoints(self, keypoints, **params):
        raise NotImplementedError

    def _apply_to_bbox(self, bbox, **params):
        raise NotImplementedError


def create_affine_matrix_3d(rotation_angles, scaling_factors):
    # Rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rotation_angles[0]), -np.sin(rotation_angles[0])],
                   [0, np.sin(rotation_angles[0]), np.cos(rotation_angles[0])]])

    Ry = np.array([[np.cos(rotation_angles[1]), 0, np.sin(rotation_angles[1])],
                   [0, 1, 0],
                   [-np.sin(rotation_angles[1]), 0, np.cos(rotation_angles[1])]])

    Rz = np.array([[np.cos(rotation_angles[2]), -np.sin(rotation_angles[2]), 0],
                   [np.sin(rotation_angles[2]), np.cos(rotation_angles[2]), 0],
                   [0, 0, 1]])

    # Scaling matrix
    S = np.diag(scaling_factors)

    # Combine rotation and scaling
    RS = Rz @ Ry @ Rx @ S
    return RS


def create_affine_matrix_2d(rotation_angle, scaling_factors):
    # Rotation matrix
    R = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                  [np.sin(rotation_angle), np.cos(rotation_angle)]])

    # Scaling matrix
    S = np.diag(scaling_factors)

    # Combine rotation and scaling
    RS = R @ S
    return RS


# def _create_identity_grid(size: List[int]) -> Tensor:
#     space = [torch.linspace((-s + 1) / s, (s - 1) / s, s) for s in size[::-1]]
#     grid = torch.meshgrid(space, indexing="ij")
#     grid = torch.stack(grid, -1)
#     spatial_dims = list(range(len(size)))
#     grid = grid.permute((*spatial_dims[::-1], len(size)))
#     return grid


def _create_centered_identity_grid2(size: Union[Tuple[int, ...], List[int]]) -> torch.Tensor:
    space = [torch.linspace((1 - s) / 2, (s - 1) / 2, s) for s in size]
    grid = torch.meshgrid(space, indexing="ij")
    grid = torch.stack(grid, -1)
    return grid


def _convert_my_grid_to_grid_sample_grid(my_grid: torch.Tensor, original_shape: Union[Tuple[int, ...], List[int]]):
    # rescale
    for d in range(len(original_shape)):
        s = original_shape[d]
        my_grid[..., d] /= (s / 2)
    my_grid = torch.flip(my_grid, (len(my_grid.shape) - 1, ))
    # my_grid = my_grid.flip((len(my_grid.shape) - 1,))
    return my_grid


# size = (4, 5, 6)
# grid_old = _create_identity_grid(size)
# grid_new = _create_centered_identity_grid2(size)
# grid_new_converted = _convert_my_grid_to_grid_sample_grid(grid_new, size)
# torch.all(torch.isclose(grid_new_converted, grid_old))

# An alternative way of generating the displacement fieldQ
# def displacement_field(data: torch.Tensor):
#     downscaling_global = np.random.uniform() ** 2 * 4 + 2
#     # local downscaling can vary a bit relative to global
#     granularity_scale_local = np.random.uniform(round(max(downscaling_global - 1.5, 2)),
#                                                 round(downscaling_global + 1.5), size=3)
#
#     B, _, D, H, W = data.size()
#     random_field_size = [round(j / i) for i, j in zip(granularity_scale_local, data.shape[2:])]
#     pool_kernel_size = [min(i // 4 * 2 + 1, round(7 / 4 * downscaling_global) // 2 * 2 + 1) for i in
#                         random_field_size]  # must be odd
#     pool_padding = [(i - 1) // 2 for i in pool_kernel_size]
#     aug1 = F.avg_pool3d(
#         F.avg_pool3d(
#             torch.randn((B, 2, *random_field_size), device=data.device),
#             pool_kernel_size, stride=1, padding=pool_padding),
#         pool_kernel_size, stride=1, padding=pool_padding)


if __name__ == '__main__':
    # torch.set_num_threads(1)
    #
    # shape = (128, 128, 128)
    # patch_size = (128, 128, 128)
    # labels = 2
    #
    #
    # # seg = torch.rand([i // 32 for i in shape]) * labels
    # # seg_up = torch.round(torch.nn.functional.interpolate(seg[None, None], size=shape, mode='trilinear')[0],
    # #                      decimals=0).to(torch.int16)
    # # img = torch.ones((1, *shape))
    # # img[tuple([slice(img.shape[0])] + [slice(i // 4, i // 4 * 2) for i in shape])] = 200
    #
    #
    # import SimpleITK as sitk
    # # img = camera()
    # # seg = None
    # img = sitk.GetArrayFromImage(sitk.ReadImage('/media/isensee/raw_data/nnUNet_raw/Dataset137_BraTS2021/imagesTr/BraTS2021_00000_0000.nii.gz'))
    # seg = sitk.GetArrayFromImage(sitk.ReadImage('/media/isensee/raw_data/nnUNet_raw/Dataset137_BraTS2021/labelsTr/BraTS2021_00000.nii.gz'))
    #
    # patch_size = (192, 192, 192)
    # sp = SpatialTransform(
    #     patch_size=(192, 192, 192),
    #     patch_center_dist_from_border=[i / 2 for i in patch_size],
    #     random_crop=True,
    #     p_elastic_deform=0,
    #     elastic_deform_magnitude=(0.1, 0.1),
    #     elastic_deform_scale=(0.1, 0.1),
    #     p_synchronize_def_scale_across_axes=0.5,
    #     p_rotation=1,
    #     rotation=(-30 / 360 * np.pi, 30 / 360 * np.pi),
    #     p_scaling=1,
    #     scaling=(0.75, 1),
    #     p_synchronize_scaling_across_axes=0.5,
    #     bg_style_seg_sampling=True,
    #     mode_seg='bilinear'
    # )
    #
    # data_dict = {'image': torch.from_numpy(deepcopy(img[None])).float()}
    # if seg is not None:
    #     data_dict['segmentation'] = torch.from_numpy(deepcopy(seg[None]))
    # # out = sp(**data_dict)
    # #
    # # view_batch(out['image'], out['segmentation'])
    #
    # from time import time
    # times = []
    # for _ in range(10):
    #     data_dict = {'image': torch.from_numpy(deepcopy(img[None])).float()}
    #     if seg is not None:
    #         data_dict['segmentation'] = torch.from_numpy(deepcopy(seg[None]))
    #     st = time()
    #     out = sp(**data_dict)
    #     times.append(time() - st)
    # print(np.median(times))


    #################
    # with this part we can qualitatively test that the correct axes are ebing augmented. Just set one of the probs to 1 and off you go
    #################

    def eldef_scale(image, dim, patch_size):
        return 0.1

    def eldef_magnitude(image, dim, patch_size, deformation_scale):
        return 10 if dim == 2 else 0

    def rot(image, dim):
        return 45/360 * 2 * np.pi if dim == 0 else 0

    def scaling(image, dim):
        return 0.5 if dim == 0 else 1

    # lines
    patch = torch.zeros((1, 64, 60, 68))
    patch[:, :, 10, 30] = 1
    patch[:, 50, :, 30] = 1
    patch[:, 40, 20, :] = 1

    # patch_block
    patch_block = torch.zeros((1, 64, 60, 68))
    patch_block[:, 22:42, 20:40, 24:44] = 1

    patch_line = torch.zeros((1, 64, 60, 128))
    patch_line[:, 22:24, 30:32, 10:-10] = 1
    use = patch_line

    sp = SpatialTransform(
        patch_size=patch.shape[1:],
        patch_center_dist_from_border=0,
        random_crop=False,
        p_elastic_deform=0,
        p_rotation=1,
        p_scaling=0,
        elastic_deform_scale=eldef_scale,
        elastic_deform_magnitude=eldef_magnitude,
        p_synchronize_def_scale_across_axes=0,
        rotation=rot,
        scaling=scaling,
        p_synchronize_scaling_across_axes=0,
        bg_style_seg_sampling=False,
        mode_seg='bilinear'
    )


    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(use[0].numpy()), 'orig.nii.gz')

    params = sp.get_parameters(image=use)
    transformed = sp._apply_to_image(use, **params)

    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(transformed[0].numpy()), 'transformed.nii.gz')

    # p = torch.zeros((1, 1, 8, 16, 32))
    # p[:, :, 2:6, 10:16, 10:24] = 1
    # grid = _create_identity_grid(p.shape[2:])
    # grid[:, :, :, 0] *= 0.5
    # out = grid_sample(p, grid[None], mode='bilinear', padding_mode="zeros", align_corners=False)
    # torch.all(out == p)
    # SimpleITK.WriteImage(SimpleITK.GetImageFromArray(p[0, 0].numpy()), 'orig.nii.gz')
    # SimpleITK.WriteImage(SimpleITK.GetImageFromArray(out[0, 0].numpy()), 'transformed.nii.gz')

    #################
    # with this part I verify that the crop through spatialtransforms grid sample yields the same result as crop_tensor
    #################

    # sp = SpatialTransform(
    #     patch_size=(48, 52, 54),
    #     patch_center_dist_from_border=0,
    #     random_crop=True,
    #     p_elastic_deform=0,
    #     p_rotation=1,
    #     p_scaling=0,
    #     rotation=0
    # )
    # sp2 = SpatialTransform(
    #     patch_size=(48, 52, 54),
    #     patch_center_dist_from_border=0,
    #     random_crop=True,
    #     p_elastic_deform=0,
    #     p_rotation=0,
    #     p_scaling=0,
    # )
    #
    # patch = torch.zeros((1, 64, 60, 68))
    # patch[:, :, 10, 30] = 1
    # patch[:, 50, :, 30] = 1
    # patch[:, 40, 20, :] = 1
    # SimpleITK.WriteImage(SimpleITK.GetImageFromArray(patch[0].numpy()), 'orig.nii.gz')
    #
    # center_coords = [50, 10, 16]
    # params = sp.get_parameters(image=patch)
    # params['center_location_in_pixels'] = center_coords
    # params2 = sp2.get_parameters(image=patch)
    # params2['center_location_in_pixels'] = center_coords
    # transformed = sp._apply_to_image(patch, **params)
    # transformed2 = sp._apply_to_image(patch, **params2)
    #
    # SimpleITK.WriteImage(SimpleITK.GetImageFromArray(transformed[0].numpy()), 'transformed.nii.gz')
    # SimpleITK.WriteImage(SimpleITK.GetImageFromArray(transformed2[0].numpy()), 'transformed2.nii.gz')



    ####################
    # This is exploraroty code to check how to retrieve coordinates. I used it to verify that grid_sample does in fact
    # use coordinates in reversed dimension order (zyx and not xyz)
    ####################
    # # create a dummy input which has a unique shape in each exis
    # p = torch.zeros((1, 1, 8, 16, 32))
    # # set one pixel to 1
    # p[:, :, 4, 0, 31] = 1
    # # now create an identity grid. I have verified that this grid yields the same image as the input when used in grid_sample. So the grid is correct
    # grid = _create_identity_grid((8, 16, 32)).contiguous() # grid is shape torch.Size([8, 16, 32, 3])
    # out = grid_sample(p, grid[None], mode='bilinear', padding_mode="zeros", align_corners=False)
    # assert torch.all(out == p)  # this passes
    # # reduce the grid to the location we are interested in. That are the coordinates where we placed the 1. The 4:5 etc is only so that we keep the number of dimensions
    # grid = grid[4:5, 0:1, 31:32]
    # # What coordinate would we expect? Note that grid is [-1, 1]
    # # For the first dimension, coordinate 4 out of shape 8 is approximately in the middle, so about 0
    # # For the second dimension, coordinate 0 out of shape 16 is very low, so we expect -1 ish (remember there is aligned corners and shit)
    # # For the third dimension, coordinate 31 out of shape 32 is very high, so we expect 1 ish (remember there is aligned corners and shit)
    # # So we expect [0, -1, 1]
    # # What do we get?
    # print(grid)
    # # > tensor([[[[ 0.9688, -0.9375,  0.1250]]]])
    # # not what we expect
    # out = grid_sample(p, grid[None], mode='bilinear', padding_mode="zeros", align_corners=False)
    # assert out.item() == 1
