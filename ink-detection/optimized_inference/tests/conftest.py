"""Shared pytest setup for optimized_inference.

The package modules call ``os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "0")`` before importing cv2, and OpenCV
reads that variable when it is imported, treating ``"0"`` as a zero-pixel limit. Whichever test module imports a package
module first therefore decides whether ``cv2.imread`` works for the rest of the session. Give OpenCV an explicit large
limit before any test module is imported so the suite does not depend on collection order.
"""

import os

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2**40))
