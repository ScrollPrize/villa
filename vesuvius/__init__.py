"""
Vesuvius module - provides access to the package components.
"""

# Import key modules
import data
import models
import utils
import setup

# Import specific classes for direct access
from data.volume import Volume
from data.vc_dataset import VCDataset

# Import utility functions for direct access
from utils import list_files, list_cubes, is_aws_ec2_instance

# Define what to expose
__all__ = ['data', 'models', 'utils', 'setup', 'Volume', 'VCDataset', 
           'list_files', 'list_cubes', 'is_aws_ec2_instance']
