import torch
import numpy as np
from typing import Union, Optional, Any, Tuple, List, Dict

# Define type alias for clarity
ArrayLike = Union[np.ndarray, "torch.Tensor", Any]  # 'Any' will cover cupy arrays without needing import

class BaseAugmentation(object):
    """Base class for all augmentations.
    
    This class handles different array types (numpy, torch, cupy) and provides
    a common interface for augmentation operations.
    """
    def __init__(
        self,
        volume: Optional[ArrayLike] = None,
        label: Optional[ArrayLike] = None,
        device: Optional[Union[str, torch.device]] = 'cuda',
        **kwargs
    ):
        """Initialize the augmentation with input volume and/or label.
        
        Parameters
        ----------
        volume : Optional[ArrayLike], default=None
            Input volume data as numpy array, torch tensor, or cupy array
        label : Optional[ArrayLike], default=None
            Input label data as numpy array, torch tensor, or cupy array
        device : Optional[Union[str, torch.device]], default='cuda'
            Device to place tensor data on (only used for torch tensors)
        **kwargs
            Additional keyword arguments for specific augmentations
        """
        # Store data type information
        self.volume_original_type = None
        self.volume = None
        if volume is not None:
            self.volume_original_type = self._determine_type(volume)
            self.volume = self._convert_to_working_format(volume, device)
            
        self.label_original_type = None
        self.label = None
        if label is not None:
            self.label_original_type = self._determine_type(label)
            self.label = self._convert_to_working_format(label, device)
            
        self.device = device
        self.kwargs = kwargs
        
    def _determine_type(self, data: ArrayLike) -> str:
        """Determine the type of the input data.
        
        Parameters
        ----------
        data : ArrayLike
            Input data
            
        Returns
        -------
        str
            Type of data: 'numpy', 'torch', or 'cupy'
        """
        if isinstance(data, torch.Tensor):
            return "torch"
        elif isinstance(data, np.ndarray):
            return "numpy"
        else:
            # Check if it's a cupy array without importing cupy
            module_name = type(data).__module__
            if module_name.startswith('cupy'):
                return "cupy"
            else:
                raise TypeError(f"Unsupported data type: {type(data)}. Must be numpy.ndarray, torch.Tensor, or cupy.ndarray")
                
    def _convert_to_working_format(self, data: ArrayLike, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """Convert input data to the format used for internal operations.
        
        Parameters
        ----------
        data : ArrayLike
            Input data
        device : Optional[Union[str, torch.device]], default=None
            Device to place tensor on
            
        Returns
        -------
        torch.Tensor
            Data converted to torch tensor for internal processing
        """
        if isinstance(data, torch.Tensor):
            return data.to(device) if device is not None else data
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
            return tensor.to(device) if device is not None else tensor
        else:
            # Assuming it's a cupy array
            try:
                # First convert cupy to numpy, then to torch
                np_array = data.get()  # Standard method to convert cupy to numpy
                tensor = torch.from_numpy(np_array)
                return tensor.to(device) if device is not None else tensor
            except AttributeError:
                # If .get() doesn't work, try alternative approaches
                if hasattr(data, 'astype'):
                    # Try numpy-like conversion
                    np_array = np.array(data)
                    tensor = torch.from_numpy(np_array)
                    return tensor.to(device) if device is not None else tensor
                else:
                    raise TypeError(f"Unable to convert {type(data)} to torch.Tensor")
                
    def _convert_back(self, result: torch.Tensor, original_type: str) -> ArrayLike:
        """Convert results back to the original data type.
        
        Parameters
        ----------
        result : torch.Tensor
            Result data as torch tensor
        original_type : str
            Original data type to convert back to
            
        Returns
        -------
        ArrayLike
            Data converted back to the original type
        """
        if original_type == "torch":
            return result
        elif original_type == "numpy":
            return result.detach().cpu().numpy()
        elif original_type == "cupy":
            np_array = result.detach().cpu().numpy()
            # Try to import cupy and convert
            try:
                import cupy as cp
                return cp.array(np_array)
            except ImportError:
                raise ImportError("Could not import cupy to convert results back to cupy array. "
                                 "Please ensure cupy is installed.")
        else:
            raise ValueError(f"Unknown original type: {original_type}")
    
    def _apply_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the actual transformation to the data.
        
        This method should be implemented by subclasses.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data to transform
            
        Returns
        -------
        torch.Tensor
            Transformed data
        """
        raise NotImplementedError("Subclasses must implement _apply_transform()")
            
    def apply_to_volume(self) -> Optional[ArrayLike]:
        """Apply the augmentation to the volume and return the result.
        
        Returns
        -------
        Optional[ArrayLike]
            Augmented volume in the same format as the input, or None if no volume was provided
        """
        if self.volume is None:
            return None
            
        transformed_volume = self._apply_transform(self.volume)
        return self._convert_back(transformed_volume, self.volume_original_type)
    
    def apply_to_label(self) -> Optional[ArrayLike]:
        """Apply the augmentation to the label and return the result.
        
        Returns
        -------
        Optional[ArrayLike]
            Augmented label in the same format as the input, or None if no label was provided
        """
        if self.label is None:
            return None
            
        transformed_label = self._apply_transform(self.label)
        return self._convert_back(transformed_label, self.label_original_type)
    
    def apply(self) -> Dict[str, Optional[ArrayLike]]:
        """Apply the augmentation to both volume and label if provided.
        
        Returns
        -------
        Dict[str, Optional[ArrayLike]]
            Dictionary containing the augmented volume and label
        """
        result = {}
        
        if self.volume is not None:
            result['volume'] = self.apply_to_volume()
            
        if self.label is not None:
            result['label'] = self.apply_to_label()
            
        return result