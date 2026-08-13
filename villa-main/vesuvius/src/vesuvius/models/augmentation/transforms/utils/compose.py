from typing import List

from vesuvius.models.augmentation.transforms.base.basic_transform import BasicTransform


class ComposeTransforms(BasicTransform):
    def __init__(self, transforms: List[BasicTransform]):
        super().__init__()
        self.transforms = transforms
        # Skip timing for the wrapper; leaf transforms record their own timings.
        self._perf_exclude = True

    def apply(self, data_dict, **params):
        for t in self.transforms:
            if data_dict.get('_skip_spatial_transforms', False) and getattr(t, '_skip_when_vector', False):
                continue
            data_dict = t(**data_dict)
        return data_dict
