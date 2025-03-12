"""
Test package for the tracer module.

This package contains tests for all components of the tracer module,
organized into subdirectories that mirror the structure of the main package.

Subdirectories:
- core: Tests for core functionality (interpolation, grid, etc.)
- cost_functions: Tests for all cost functions
- surfaces: Tests for surface representation classes
- utils: Tests for utility functions

To run all tests:
    python -m unittest discover tests

To run tests in a specific directory:
    python -m unittest discover tests/cost_functions
"""

import sys
import os
from pathlib import Path

# Add the project root to the path for proper imports
# This allows tests to be run from any directory
project_root = Path(__file__).parent.parent
if project_root not in sys.path:
    sys.path.append(str(project_root))