import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any, Type, Callable
import random
from .base import BaseAugmentation, ArrayLike

class Compose:
    """A class to compose multiple augmentations together.
    
    This class allows you to chain multiple augmentations and apply them
    sequentially to a volume and/or label, with optional probability for each augmentation.
    
    Attributes
    ----------
    augmentations : List[Dict]
        List of dictionaries containing augmentation class, parameters, and probability
    """
    
    def __init__(self, augmentations: List[Dict]):
        """Initialize the compose with a list of augmentation specifications.
        
        Parameters
        ----------
        augmentations : List[Dict]
            List of dictionaries, each containing:
            - 'aug_class': The augmentation class (subclass of BaseAugmentation)
            - 'params': Dict of parameters to pass to the augmentation (optional)
            - 'probability': Float between 0 and 1 for chance to apply (default: 1.0)
            
        Example
        -------
        >>> from vesuvius.augmentation.transforms import RandomFlip, RandomRotate
        >>> compose = Compose([
        ...     {'aug_class': RandomFlip, 'params': {'axis': -1}, 'probability': 0.5},
        ...     {'aug_class': RandomRotate, 'params': {'angles': [90, 180, 270]}, 'probability': 0.3}
        ... ])
        """
        self.augmentations = []
        
        for aug_spec in augmentations:
            # Validate the augmentation specification
            if 'aug_class' not in aug_spec:
                raise ValueError("Each augmentation must specify 'aug_class'")
            
            # Extract class, parameters, and probability
            aug_class = aug_spec['aug_class']
            params = aug_spec.get('params', {})
            probability = aug_spec.get('probability', 1.0)
            
            # Validate the augmentation class
            if not issubclass(aug_class, BaseAugmentation):
                raise TypeError(f"Augmentation class must be a subclass of BaseAugmentation, got {aug_class}")
            
            # Validate probability
            if not 0 <= probability <= 1:
                raise ValueError(f"Probability must be between 0 and 1, got {probability}")
            
            self.augmentations.append({
                'aug_class': aug_class,
                'params': params,
                'probability': probability
            })
    
    def __call__(self, 
                volume: Optional[ArrayLike] = None, 
                label: Optional[ArrayLike] = None,
                device: Optional[Union[str, torch.device]] = 'cuda'
               ) -> Dict[str, Optional[ArrayLike]]:
        """Apply the chain of augmentations to the input volume and/or label.
        
        Parameters
        ----------
        volume : Optional[ArrayLike], default=None
            Input volume data as numpy array, torch tensor, or cupy array
        label : Optional[ArrayLike], default=None
            Input label data as numpy array, torch tensor, or cupy array
        device : Optional[Union[str, torch.device]], default='cuda'
            Device to place tensor data on (only used for torch tensors)
            
        Returns
        -------
        Dict[str, Optional[ArrayLike]]
            Dictionary containing the augmented volume and/or label
            
        Example
        -------
        >>> result = compose(volume=my_volume, label=my_label)
        >>> augmented_volume = result['volume']
        >>> augmented_label = result['label']
        """
        result = {'volume': volume, 'label': label}
        
        for aug_spec in self.augmentations:
            # Determine if we should apply this augmentation based on probability
            if random.random() < aug_spec['probability']:
                # Create the augmentation instance with current results and parameters
                aug_class = aug_spec['aug_class']
                params = aug_spec['params']
                
                aug_instance = aug_class(
                    volume=result.get('volume'),
                    label=result.get('label'),
                    device=device,
                    **params
                )
                
                # Apply the augmentation and update the results
                aug_result = aug_instance.apply()
                
                # Update our results dict only for keys that exist in aug_result
                for key, value in aug_result.items():
                    if value is not None:
                        result[key] = value
        
        return result
    
    def apply_to_volume(self, 
                       volume: ArrayLike,
                       device: Optional[Union[str, torch.device]] = 'cuda'
                      ) -> ArrayLike:
        """Apply the chain of augmentations to just the volume.
        
        Parameters
        ----------
        volume : ArrayLike
            Input volume data as numpy array, torch tensor, or cupy array
        device : Optional[Union[str, torch.device]], default='cuda'
            Device to place tensor data on (only used for torch tensors)
            
        Returns
        -------
        ArrayLike
            Augmented volume in the same format as the input
        """
        result = self(volume=volume, device=device)
        return result['volume']
    
    def apply_to_label(self, 
                      label: ArrayLike,
                      device: Optional[Union[str, torch.device]] = 'cuda'
                     ) -> ArrayLike:
        """Apply the chain of augmentations to just the label.
        
        Parameters
        ----------
        label : ArrayLike
            Input label data as numpy array, torch tensor, or cupy array
        device : Optional[Union[str, torch.device]], default='cuda'
            Device to place tensor data on (only used for torch tensors)
            
        Returns
        -------
        ArrayLike
            Augmented label in the same format as the input
        """
        result = self(label=label, device=device)
        return result['label']