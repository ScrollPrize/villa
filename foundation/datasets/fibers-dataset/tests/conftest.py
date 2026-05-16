"""Pytest configuration for fibers-dataset tests.

Adds the fibers-dataset directory to ``sys.path`` so ``import tools`` and
``import generate_fiber_labels_from_ct`` work despite the hyphenated parent
directory name (which prevents normal package imports).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
