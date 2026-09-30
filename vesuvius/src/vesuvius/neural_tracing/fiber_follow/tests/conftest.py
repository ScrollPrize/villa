import pytest
import torch


@pytest.fixture(autouse=True)
def isolate_compiler_cache():
    # Each test constructs different model shapes/backends. Their specializations
    # must not consume the next test's per-function recompilation budget.
    torch._dynamo.reset()
    dynamic_shapes = torch._dynamo.config.capture_dynamic_output_shape_ops
    emulate_casts = torch._inductor.config.emulate_precision_casts
    try:
        yield
    finally:
        torch._dynamo.config.capture_dynamic_output_shape_ops = dynamic_shapes
        torch._inductor.config.emulate_precision_casts = emulate_casts
        torch._dynamo.reset()
