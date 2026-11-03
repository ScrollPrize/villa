import importlib.util
from pathlib import Path

import numpy as np
import pytest

# The adapter itself has no dependency on VC's GUI/native Python modules.
spec = importlib.util.spec_from_file_location(
    "vcz1", Path(__file__).parents[1] / "vc/compression/vcz1.py"
)
vcz1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vcz1)


@pytest.mark.parametrize("shape", [(32, 32, 32), (17, 19, 23)])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("quant", [1, 3])
def test_vcz1_keeps_its_wire_format(shape, dtype, quant):
    data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    payload = vcz1.compress_array(data, quant=quant)
    assert payload[:4] == b"VCZ1"
    assert payload == vcz1.compress(data.tobytes(), *shape, data.itemsize, quant=quant)
    expected = np.minimum(
        (data.astype(np.uint32) + quant // 2) // quant * quant, np.iinfo(dtype).max
    ).astype(dtype)
    assert vcz1.decompress(payload, data.nbytes) == expected.tobytes()
    output = bytearray(data.nbytes)
    vcz1.decompress_into(payload, output)
    assert output == expected.tobytes()
