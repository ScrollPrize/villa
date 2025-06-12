import pytest
import torch
import numpy as np
from optimize_fields.utils import rotate_by_quaternion, matrix_to_quaternion

def rotation_matrix(axis: str, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'Z':  # your Z=(1,0,0)
        return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)
    if axis == 'Y':  # your Y=(0,1,0)
        return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float64)
    if axis == 'X':  # your X=(0,0,1)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)
    raise ValueError(f"Unknown axis {axis}")

@pytest.mark.parametrize("axis,vec_name,vec", [
    ("Z","Z",[1,0,0]),
    ("Z","Y",[0,1,0]),
    ("Z","X",[0,0,1]),
    ("Y","Z",[1,0,0]),
    ("Y","Y",[0,1,0]),
    ("Y","X",[0,0,1]),
    ("X","Z",[1,0,0]),
    ("X","Y",[0,1,0]),
    ("X","X",[0,0,1]),
])
def test_rotate_against_matrix(axis, vec_name, vec):
    theta = np.pi / 4
    R = rotation_matrix(axis, theta)
    R_t = torch.from_numpy(R).unsqueeze(0)   # (1,3,3), dtype float64
    q = matrix_to_quaternion(R_t).squeeze(0) # (4,), dtype float64

    # rotate vector
    v = torch.tensor(vec, dtype=torch.float64)
    v_rot = rotate_by_quaternion(q, v).numpy()
    v_exp = R.dot(np.array(vec, np.float64))

    # numeric check
    np.testing.assert_allclose(v_rot, v_exp, atol=1e-10)

    # quaternion unit‐norm and w>=0 using pure‐Python scalars
    norm = q.norm().item()
    assert abs(norm - 1.0) < 1e-10
    assert q[0].item() >= 0

def test_matrix_to_quaternion_identity():
    I = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    q = matrix_to_quaternion(I).squeeze(0)
    q_np = q.numpy()
    # identity quaternion is [1,0,0,0]
    np.testing.assert_allclose(q_np, [1.0, 0.0, 0.0, 0.0], atol=1e-10)
