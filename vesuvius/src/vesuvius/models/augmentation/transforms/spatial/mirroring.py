from typing import Optional, Set, Tuple

import torch

from vesuvius.models.augmentation.transforms.base.basic_transform import BasicTransform


class MirrorTransform(BasicTransform):

    _is_spatial = True  # Skip per-transform padding restoration

    def __init__(self, allowed_axes: Tuple[int, ...], normal_keys: Optional[Set[str]] = None):
        super().__init__()
        self.allowed_axes = allowed_axes
        self.normal_keys = set(normal_keys or [])
        self._skip_when_vector = True

    def get_parameters(self, **data_dict) -> dict:
        axes = [i for i in self.allowed_axes if torch.rand(1) < 0.5]
        params = {'axes': axes}
        # Pass crop_shape through for keypoint transforms
        if 'crop_shape' in data_dict:
            params['crop_shape'] = data_dict['crop_shape']
        return params

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if len(params['axes']) == 0:
            return img
        axes = [i + 1 for i in params['axes']]
        return torch.flip(img, axes)

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        if len(params['axes']) == 0:
            return segmentation
        axes = [i + 1 for i in params['axes']]
        return torch.flip(segmentation, axes)

    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        if len(params['axes']) == 0:
            return regression_target
        axes = [i + 1 for i in params['axes']]
        return torch.flip(regression_target, axes)

    def _apply_to_bbox(self, bbox, **params):
        raise NotImplementedError

    def _apply_to_keypoints(self, keypoints, **params):
        """Flip keypoint coordinates along mirrored axes."""
        if keypoints is None:
            return None
        axes = params['axes']
        if len(axes) == 0:
            return keypoints
        crop_shape = params.get('crop_shape')
        if crop_shape is None:
            raise ValueError("MirrorTransform._apply_to_keypoints requires 'crop_shape' in params")

        keypoints = keypoints.clone()
        for axis in axes:
            keypoints[:, axis] = crop_shape[axis] - 1 - keypoints[:, axis]
        return keypoints

    def _apply_to_vectors(self, vectors, **params):
        """Negate vector components along mirrored axes."""
        if vectors is None:
            return None
        axes = params['axes']
        if len(axes) == 0:
            return vectors

        vectors = vectors.clone()
        for axis in axes:
            vectors[:, axis] = -vectors[:, axis]
        return vectors

    def apply(self, data_dict: dict, **params) -> dict:
        """
        Extend base apply to also flip and negate normal vectors along mirrored axes.
        """
        if data_dict.get('_skip_spatial_transforms', False) and self._skip_when_vector:
            return data_dict

        # Remove normals so base class doesn't treat them as scalar labels
        normals_cache = {}
        for key in self.normal_keys:
            if key in data_dict:
                normals_cache[key] = data_dict.pop(key)

        data_dict = super().apply(data_dict, **params)

        # Handle normals: flip spatially and negate the corresponding component
        if normals_cache:
            axes = params.get('axes', [])
            for key, normals in normals_cache.items():
                if normals is None or len(axes) == 0:
                    data_dict[key] = normals
                    continue
                # Flip spatially (axes are 0-indexed spatial, need +1 for tensor dims)
                spatial_axes = [i + 1 for i in axes]
                normals = torch.flip(normals, spatial_axes)
                # Negate the normal components for flipped axes
                # normals shape is (3, Z, Y, X) where channel 0=Z, 1=Y, 2=X
                for axis in axes:
                    normals[axis] = -normals[axis]
                data_dict[key] = normals

        # Handle vector_keys
        vector_keys = set(data_dict.get('vector_keys', []) or [])
        for key in vector_keys:
            if data_dict.get(key) is not None:
                data_dict[key] = self._apply_to_vectors(data_dict[key], **params)

        # Transform padding_mask with the same flip
        if data_dict.get('padding_mask') is not None:
            data_dict['padding_mask'] = self._apply_to_image(data_dict['padding_mask'], **params)

        return data_dict
