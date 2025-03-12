"""
Volume Cartographer surface tracing algorithms.

This package contains the Python implementation of the Volume Cartographer
surface tracing algorithms, which grow surfaces from seed points through
optimization in 3D volumes.
"""

from .grid import PointGrid, GridState, STATE_NONE, STATE_PROCESSING
from .grid import STATE_LOC_VALID, STATE_COORD_VALID