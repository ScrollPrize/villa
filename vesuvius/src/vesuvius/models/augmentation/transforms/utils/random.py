import torch

from vesuvius.models.augmentation.transforms.base.basic_transform import BasicTransform


class RandomTransform(BasicTransform):
    def __init__(self, transform: BasicTransform, apply_probability: float = 1):
        super().__init__()
        self.transform = transform
        self.apply_probability = apply_probability
        self._skip_when_vector = getattr(transform, "_skip_when_vector", False)
        # Skip timing for the wrapper; leaf transforms record their own timings.
        self._perf_exclude = True

    def get_parameters(self, **data_dict) -> dict:
        return {"apply_transform": torch.rand(1).item() < self.apply_probability}

    def apply(self, data_dict: dict, **params) -> dict:
        if data_dict.get('_skip_spatial_transforms', False) and self._skip_when_vector:
            return data_dict
        if params['apply_transform']:
            return self.transform(**data_dict)
        else:
            return data_dict
    
    def __repr__(self):
        ret_str = f"{type(self).__name__}(p={self.apply_probability}, transform={self.transform})"
        return ret_str
