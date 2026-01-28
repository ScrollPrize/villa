import numpy as np
from typing import List
from vesuvius.models.augmentation.transforms.base.basic_transform import BasicTransform


class OneOfTransform(BasicTransform):
    def __init__(self, list_of_transforms: List[BasicTransform]):
        """
        Randomly selects one of the transforms given in list_of_transforms and applies it. Remember that
        probabilities of the individual transforms for being applied still exist and apply!
        :param list_of_transforms: List of BasicTransform instances
        """
        super().__init__()
        self.list_of_transforms = list_of_transforms
        # Skip timing for the wrapper; leaf transforms record their own timings.
        self._perf_exclude = True

    def apply(self, data_dict, **params):
        # Randomly select one transform from the list
        selected_transform = self.list_of_transforms[params['selected_index']]
        # Apply the selected transform
        return selected_transform(**data_dict)
    
    def get_parameters(self, **data_dict) -> dict:
        # Randomly choose which transform to apply
        return {'selected_index': np.random.choice(len(self.list_of_transforms))}
