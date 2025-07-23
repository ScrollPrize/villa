"""
Vesuvius module - provides access to the package components.
"""

# Import key modules
from vesuvius import models, data, utils, setup

# Import specific classes for direct access
from vesuvius.data import Volume
from vesuvius.data.vc_dataset import VCDataset

# Import utility functions for direct access
from vesuvius.utils import list_files, list_cubes, is_aws_ec2_instance

# Define what to expose
__all__ = ['data', 'models', 'utils', 'setup', 'Volume', 'VCDataset', 
           'list_files', 'list_cubes', 'is_aws_ec2_instance']
