import torch
import numpy as np
from scipy.ndimage import binary_closing
from skimage.morphology import binary_closing as skimage_binary_closing, disk, ball

from vesuvius.models.augmentation.transforms.base.basic_transform import BasicTransform


class MorphologicalClosingTransform(BasicTransform):
    """
    Apply morphological closing to segmentation labels to clean up artifacts from bilinear interpolation.
    This is particularly useful after spatial transforms.
    """
    
    def __init__(self, structure_size: int = 3):
        """
        Initialize the morphological closing transform.
        
        Parameters
        ----------
        structure_size : int
            Size of the structuring element. For 2D, this creates a disk with given radius.
            For 3D, this creates a ball with given radius.
        """
        super().__init__()
        self.structure_size = structure_size
    
    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        # Don't modify images
        return img
    
    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        """
        Apply morphological closing to each channel of the segmentation.
        """
        seg_np = segmentation.numpy()
        result = np.zeros_like(seg_np)
        
        # Check if 2D or 3D
        is_3d = segmentation.ndim == 4  # C, D, H, W for 3D or C, H, W for 2D
        
        # Process each channel
        for c in range(seg_np.shape[0]):
            if is_3d:
                # For 3D, create a ball structuring element
                from scipy.ndimage import generate_binary_structure
                structure = generate_binary_structure(3, 1)
                structure = np.repeat(structure, self.structure_size, axis=0)
                structure = np.repeat(structure, self.structure_size, axis=1)
                structure = np.repeat(structure, self.structure_size, axis=2)
                
                # Apply binary closing
                result[c] = binary_closing(seg_np[c] > 0, structure=structure).astype(seg_np.dtype)
                # Restore original values where closing kept the pixels
                result[c] *= seg_np[c].max()
            else:
                # For 2D, use disk structuring element
                structure = disk(self.structure_size)
                result[c] = skimage_binary_closing(seg_np[c] > 0, structure).astype(seg_np.dtype)
                # Restore original values where closing kept the pixels
                result[c] *= seg_np[c].max()
        
        return torch.from_numpy(result)
    
    def _apply_to_dist_map(self, dist_map: torch.Tensor, **params) -> torch.Tensor:
        # Don't modify distance maps
        return dist_map
    
    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        # Don't modify regression targets
        return regression_target
    
    def _apply_to_keypoints(self, keypoints, **params):
        # Don't modify keypoints
        return keypoints
    
    def _apply_to_bbox(self, bbox, **params):
        # Don't modify bounding boxes
        return bbox