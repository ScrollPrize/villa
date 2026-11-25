"""Test-session configuration for the lasagna suite."""
from __future__ import annotations

import torch._inductor.config as _inductor_config

# torch 2.12.1's vectorized CPU inductor codegen emits C++ that does not compile
# (``operands to '?:' have different types VecMask<int> / VecMask<float>``) for the
# compiled CPU kernels these tests exercise. Scalar codegen compiles and gives the
# same results, so the suite runs with it; it does not change the tested numerics.
_inductor_config.cpp.simdlen = 1
