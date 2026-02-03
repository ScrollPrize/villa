import abc
import time
from typing import Optional

import torch

# Value to use for padded regions - should remain unchanged after transforms
PADDING_VALUE = 0.0


def _restore_padded_regions(img: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Restore padded regions to PADDING_VALUE after a transform."""
    if padding_mask is None:
        return img
    mask = padding_mask.to(img.device, dtype=img.dtype)
    return img * mask + PADDING_VALUE * (1 - mask)


class BasicTransform(abc.ABC):
    """
    Transforms are applied to each sample individually. The dataloader is responsible for collating, or we might consider a CollateTransform

    We expect (C, X, Y) or (C, X, Y, Z) shaped inputs for image and seg (yes seg can have more color channels)

    No idea what keypoint and bbox will look like, this is Michaels turf
    """
    def __init__(self):
        pass

    def __call__(self, **data_dict) -> dict:
        perf = data_dict.get('_aug_perf')
        perf_name = getattr(self, '_perf_name', type(self).__name__)
        should_time = (
            perf is not None
            and not getattr(self, '_perf_exclude', False)
            and perf_name in perf
        )

        if should_time:
            start = time.perf_counter()
            params = self.get_parameters(**data_dict)
            out = self.apply(data_dict, **params)
            perf[perf_name] += time.perf_counter() - start
            return out

        params = self.get_parameters(**data_dict)
        return self.apply(data_dict, **params)

    def apply(self, data_dict, **params):
        # Check if this is unlabeled data
        is_unlabeled = data_dict.get('is_unlabeled', False)

        # Special handling for known keys
        if data_dict.get('image') is not None:
            # Always transform images, even for unlabeled data
            data_dict['image'] = self._apply_to_image(data_dict['image'], **params)
            # Restore padded regions for non-spatial transforms
            if not getattr(self, '_is_spatial', False):
                padding_mask = data_dict.get('padding_mask')
                if padding_mask is not None:
                    data_dict['image'] = _restore_padded_regions(data_dict['image'], padding_mask)

        # Skip all label transforms for unlabeled data
        if not is_unlabeled:
            if data_dict.get('regression_target') is not None:
                data_dict['regression_target'] = self._apply_to_segmentation(data_dict['regression_target'], **params)

            if data_dict.get('segmentation') is not None:
                data_dict['segmentation'] = self._apply_to_segmentation(data_dict['segmentation'], **params)

            if data_dict.get('dist_map') is not None:
                data_dict['dist_map'] = self._apply_to_dist_map(data_dict['dist_map'], **params)

            if data_dict.get('geols_labels') is not None:
                data_dict['geols_labels'] = self._apply_to_dist_map(data_dict['geols_labels'], **params)

            if data_dict.get('keypoints') is not None:
                data_dict['keypoints'] = self._apply_to_keypoints(data_dict['keypoints'], **params)

            if data_dict.get('bbox') is not None:
                data_dict['bbox'] = self._apply_to_bbox(data_dict['bbox'], **params)

            # Dynamic handling for any other keys (e.g., custom targets like 'ink', 'normals')
            # Skip 'ignore_masks' as it shouldn't be transformed
            regression_keys = set(data_dict.get('regression_keys', []) or [])
            vector_keys = set(data_dict.get('vector_keys', []) or [])
            known_keys = {'image', 'regression_target', 'segmentation', 'dist_map',
                          'geols_labels', 'keypoints', 'bbox', 'ignore_masks', 'is_unlabeled',
                          'regression_keys', 'vector_keys', 'crop_shape', 'patch_info', 'padding_mask',
                          '_aug_perf'}
            # Also skip vector_keys since they're handled by specific transforms
            known_keys.update(vector_keys)

            for key in list(data_dict.keys()):
                if key in known_keys or data_dict.get(key) is None:
                    continue
                # Choose interpolation mode based on whether the key is marked as regression
                if key in regression_keys:
                    data_dict[key] = self._apply_to_regr_target(data_dict[key], **params)
                else:
                    data_dict[key] = self._apply_to_segmentation(data_dict[key], **params)

        return data_dict

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return img

    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        return regression_target

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        return segmentation

    def _apply_to_dist_map(self, dist_map: torch.Tensor, **params) -> torch.Tensor:
        return dist_map

    def _apply_to_geols_labels(self, geols_labels: torch.Tensor, **params) -> torch.Tensor:
        return geols_labels

    def _apply_to_keypoints(self, keypoints, **params):
        """Default: pass keypoints through unchanged."""
        return keypoints

    def _apply_to_bbox(self, bbox, **params):
        """Default: pass bbox through unchanged."""
        return bbox

    def _apply_to_vectors(self, vectors, **params):
        """Apply directional transformation to (N, 3) vectors (rotation/flip only, no translation).
        Default: pass vectors through unchanged."""
        return vectors

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def __repr__(self):
        ret_str = str(type(self).__name__) + "( " + ", ".join(
            [key + " = " + repr(val) for key, val in self.__dict__.items()]) + " )"
        return ret_str


class ImageOnlyTransform(BasicTransform):
    def apply(self, data_dict: dict, **params) -> dict:
        if data_dict.get('image') is not None:
            data_dict['image'] = self._apply_to_image(data_dict['image'], **params)
            # Restore padded regions for non-spatial transforms
            if not getattr(self, '_is_spatial', False):
                padding_mask = data_dict.get('padding_mask')
                if padding_mask is not None:
                    data_dict['image'] = _restore_padded_regions(data_dict['image'], padding_mask)
        return data_dict


class SegOnlyTransform(BasicTransform):
    def apply(self, data_dict: dict, **params) -> dict:
        if data_dict.get('segmentation') is not None:
            data_dict['segmentation'] = self._apply_to_segmentation(data_dict['segmentation'], **params)
        return data_dict


if __name__ == '__main__':
    pass
