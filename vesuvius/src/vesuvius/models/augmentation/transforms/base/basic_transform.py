import abc
import torch


class BasicTransform(abc.ABC):
    """
    Transforms are applied to each sample individually. The dataloader is responsible for collating, or we might consider a CollateTransform

    We expect (C, X, Y) or (C, X, Y, Z) shaped inputs for image and seg (yes seg can have more color channels)

    No idea what keypoint and bbox will look like, this is Michaels turf
    """
    def __init__(self):
        pass

    def __call__(self, **data_dict) -> dict:
        params = self.get_parameters(**data_dict)
        return self.apply(data_dict, **params)

    def apply(self, data_dict, **params):
        # Special handling for known keys
        if data_dict.get('image') is not None:
            data_dict['image'] = self._apply_to_image(data_dict['image'], **params)

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
        known_keys = {'image', 'regression_target', 'segmentation', 'dist_map', 
                      'geols_labels', 'keypoints', 'bbox', 'ignore_masks'}
        
        for key in data_dict.keys():
            if key not in known_keys and data_dict[key] is not None:
                # Assume custom targets should be treated as segmentation
                data_dict[key] = self._apply_to_segmentation(data_dict[key], **params)

        return data_dict

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        pass

    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        pass

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        pass

    def _apply_to_dist_map(self, dist_map: torch.Tensor, **params) -> torch.Tensor:
        pass

    def _apply_to_geols_labels(self, geols_labels: torch.Tensor, **params) -> torch.Tensor:
        pass

    def _apply_to_keypoints(self, keypoints, **params):
        pass

    def _apply_to_bbox(self, bbox, **params):
        pass

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
        return data_dict


class SegOnlyTransform(BasicTransform):
    def apply(self, data_dict: dict, **params) -> dict:
        if data_dict.get('segmentation') is not None:
            data_dict['segmentation'] = self._apply_to_segmentation(data_dict['segmentation'], **params)
        return data_dict


if __name__ == '__main__':
    pass
