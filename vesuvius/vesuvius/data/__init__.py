# Define modules and classes to expose
__all__ = ['Volume', 'VCDataset']

# Import key classes to make them available at the data package level
from vesuvius.vesuvius.data.volume import Volume
from vesuvius.vesuvius.data.vc_dataset import VCDataset
from .vc_dataset import VCDataset
