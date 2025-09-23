import numpy as np
from vesuvius.structure_tensor.vf_format import encode_dir_to_u8, encode_conf_to_u8


def test_encode_dir_to_u8_roundtrip_edges():
    xs = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    u8 = encode_dir_to_u8(xs)
    # exact endpoints under spec d*127 + 128  ⇒  -1→1, 1→255
    assert u8[0] == 1 and u8[4] == 255
    # center maps to ~128 (allow 1 ulp tolerance due to rounding)
    assert abs(int(u8[2]) - 128) <= 1
    # NaN/±Inf are clamped
    assert u8[5] == 128 and u8[6] == 255 and u8[7] == 0


def test_encode_conf_to_u8_clamps():
    cs = np.array([-0.1, 0.0, 0.5, 1.0, 1.7, np.nan], dtype=np.float32)
    u8 = encode_conf_to_u8(cs)
    assert u8.tolist() == [0, 0, 128, 255, 255, 0]