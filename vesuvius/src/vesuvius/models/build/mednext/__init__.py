"""
Minimal vendored MedNeXt implementation for Vesuvius.

Source adapted from the official MIC-DKFZ MedNeXt repository:
https://github.com/MIC-DKFZ/MedNeXt

The initial landing intentionally scopes this package to 3D usage only.
"""

from .factory import create_mednext_v1, get_mednext_v1_config
from .v1 import MedNeXtV1

__all__ = [
    "MedNeXtV1",
    "create_mednext_v1",
    "get_mednext_v1_config",
]
