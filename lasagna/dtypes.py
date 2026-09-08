# Set LASAGNA_MAX_PRECISION_FLOAT=64 to make lasagna use 64 bit precision where it did before.
#
# TODO: If float32 works well, replace float_hi with float32 and remove float_hi.
# see: https://github.com/ScrollPrize/villa/pull/1639

import os
import torch
import numpy as np


_float_bits = os.environ.get("LASAGNA_MAX_PRECISION_FLOAT", "32")
torch_float_hi = getattr(torch, "float" + _float_bits)
numpy_float_hi = getattr(np, "float" + _float_bits)
