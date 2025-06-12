import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from optimize_fields.utils import rotate_by_quaternion, matrix_to_quaternion

def rotation_matrix(axis: str, theta: float) -> np.ndarray:
    """
    3×3 rotation matrix for rotation by `theta` about world-axis:
      'Z'→x, 'Y'→y, 'X'→z (since you’ve re-labeled Z=(1,0,0), Y=(0,1,0), X=(0,0,1))
    """
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'Z':  # around x-axis
        return np.array([[1, 0,  0],
                         [0, c, -s],
                         [0, s,  c]], dtype=np.float32)
    if axis == 'Y':  # around y-axis
        return np.array([[ c, 0, s],
                         [ 0, 1, 0],
                         [-s, 0, c]], dtype=np.float32)
    if axis == 'X':  # around z-axis
        return np.array([[c, -s, 0],
                         [s,  c, 0],
                         [0,  0, 1]], dtype=np.float32)
    raise ValueError(f"Unknown axis {axis}")

# 1) Define 45° in radians
theta = np.pi / 4

# 2) Canonical “axes” per your naming: Z=(1,0,0), Y=(0,1,0), X=(0,0,1)
axes = {
    'Z': torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
    'Y': torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
    'X': torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
}

# 3) Build rotation matrices & convert to quaternions via utils.matrix_to_quaternion
rot_mats = {ax: rotation_matrix(ax, theta) for ax in axes}
quaternions = {}
for ax, R in rot_mats.items():
    # matrix_to_quaternion expects a torch.Tensor of shape (...,3,3)
    R_t = torch.from_numpy(R).unsqueeze(0)  # (1,3,3)
    q = matrix_to_quaternion(R_t)           # (1,4)
    quaternions[ax] = q.squeeze(0)          # (4,)

# 4) Numeric checks
print("=== Numeric checks for 45° rotations ===")
for ax, q in quaternions.items():
    print(f"\n-- Rotation about {ax}-axis --")
    R = rot_mats[ax]
    for name, v in axes.items():
        orig = v.numpy()
        rotated_v = rotate_by_quaternion(q, v).numpy()
        expected_v = R.dot(orig)
        diff = np.max(np.abs(rotated_v - expected_v))
        norm = np.linalg.norm(rotated_v)
        print(f" {name}-axis: max|rot–exp|={diff:.2e}, norm={norm:.4f}")

# 5) Visualization: 3 rows (Z,Y,X), 2 cols (before, after)
fig = plt.figure(figsize=(12, 18))
colors = {'Z':'r', 'Y':'g', 'X':'b'}

for i, ax_name in enumerate(['Z','Y','X']):
    q = quaternions[ax_name]

    # Before
    ax1 = fig.add_subplot(3, 2, 2*i+1, projection='3d')
    for name, v in axes.items():
        v_np = v.numpy()
        ax1.quiver(0,0,0, v_np[0],v_np[1],v_np[2],
                   length=1, color=colors[name], linewidth=2)
    ax1.set_title("Original Axes")
    ax1.set_xlabel('Z'); ax1.set_ylabel('Y'); ax1.set_zlabel('X')
    ax1.set_xlim(-1,1); ax1.set_ylim(-1,1); ax1.set_zlim(-1,1)
    if i == 0: ax1.legend(axes.keys())

    # After
    ax2 = fig.add_subplot(3, 2, 2*i+2, projection='3d')
    for name, v in axes.items():
        v_rot = rotate_by_quaternion(q, v).numpy()
        ax2.quiver(0,0,0, v_rot[0],v_rot[1],v_rot[2],
                   length=1, color=colors[name], linewidth=2)
    ax2.set_title(f"Rotated 45° about {ax_name}-axis")
    ax2.set_xlabel('Z'); ax2.set_ylabel('Y'); ax2.set_zlabel('X')
    ax2.set_xlim(-1,1); ax2.set_ylim(-1,1); ax2.set_zlim(-1,1)
    if i == 0: ax2.legend(axes.keys())

plt.tight_layout()
plt.show()
