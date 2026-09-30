import pytest
import torch


@pytest.fixture(autouse=True)
def isolate_compiler_cache():
    # Each test constructs different model shapes/backends. Their specializations
    # must not consume the next test's per-function recompilation budget.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()
