import numpy as np
import random


class RandomFlipWithNormals:
    """
    Flip 3D volumes (and their normal vectors) along each axis with probability p.
    Handles arrays that may be (C, Z, Y, X) or (Z, Y, X).
    If it's (Z, Y, X), we temporarily add a dummy channel.
    """

    def __init__(self, p=0.5, p_transform=1.0, normal_keys=("normals",)):
        """
        Args:
            p (float): Probability of flipping along each axis independently.
            p_transform (float): Probability of performing this entire transform at all.
            normal_keys (tuple): which dictionary keys hold normal-vector data.
        """
        self.p = p
        self.p_transform = p_transform
        self.normal_keys = set(normal_keys)

    def __call__(self, data_dict):
        """
        data_dict: dict of {key: np.ndarray}
          Each array can be either shape (Z, Y, X), (C, Z, Y, X).
          We'll do the flip, and if we add a dummy channel, we'll remove it afterwards.
        """

        # First decide whether we apply this transform at all.
        if random.random() >= self.p_transform:
            # Skip entirely
            return data_dict

        # Then flip along each axis with probability p.
        for axis_id in [1, 2, 3]:  # Z=1, Y=2, X=3
            if random.random() < self.p:
                for k, arr in data_dict.items():
                    # 1) Check shape
                    remove_dummy = False
                    if arr.ndim == 3:
                        # shape is (Z, Y, X), so add dummy channel
                        arr = arr[None, ...]  # -> (1, Z, Y, X)
                        remove_dummy = True

                    # Now we expect arr has shape (C, Z, Y, X)
                    # Flip along the correct axis
                    arr = np.flip(arr, axis=axis_id).copy()

                    # If k is a normal vector, fix sign on the correct component
                    if k in self.normal_keys:
                        # Nx = arr[0], Ny = arr[1], Nz = arr[2]
                        if axis_id == 1:
                            # flipping along Z -> arr[2] *= -1
                            arr[2] *= -1
                        elif axis_id == 2:
                            # flipping along Y -> arr[1] *= -1
                            arr[1] *= -1
                        elif axis_id == 3:
                            # flipping along X -> arr[0] *= -1
                            arr[0] *= -1

                    # Remove dummy channel if we added it
                    if remove_dummy:
                        arr = arr[0]  # from (1, Z, Y, X) back to (Z, Y, X)

                    data_dict[k] = arr

        return data_dict

class RandomRotate90WithNormals:
    def __init__(self, axes=('x', 'y', 'z'), p=0.5, p_transform=1.0, normal_keys=("normals",)):
        """
        Args:
            axes (tuple): Which axes to consider rotating around (e.g. ('x','y','z')).
            p (float): Probability of performing a rotation (once we decide to transform at all).
            p_transform (float): Probability of performing this entire transform at all.
            normal_keys (tuple): Which dictionary keys hold normal-vector data.
        """
        self.axes = axes
        self.p = p
        self.p_transform = p_transform
        self.normal_keys = set(normal_keys)

    def __call__(self, data_dict):
        if random.random() >= self.p_transform:
            return data_dict

        if random.random() >= self.p:
            return data_dict

        axis = random.choice(self.axes)  # 'x', 'y', or 'z'
        k = random.choice([1, 2, 3])     # 90°, 180°, 270°

        for key, arr in data_dict.items():
            # 1) If it's (Z, Y, X), add a dummy channel
            remove_dummy = False
            if arr.ndim == 3:
                arr = arr[None, ...]  # (1, Z, Y, X)
                remove_dummy = True

            # 2) Now we know arr is (C, Z, Y, X). We can rotate in (Z, Y, X) => axes=(1,2,3).
            if axis == 'z':
                # rotating in the Y,X plane => axes=(2, 3)
                arr = np.rot90(arr, k=k, axes=(2, 3)).copy()
            elif axis == 'y':
                # rotating in the Z,X plane => axes=(1, 3)
                arr = np.rot90(arr, k=k, axes=(1, 3)).copy()
            else:  # axis == 'x'
                # rotating in the Z,Y plane => axes=(1, 2)
                arr = np.rot90(arr, k=k, axes=(1, 2)).copy()

            # Handle normal vector rotations
            if key in self.normal_keys:
                nx = arr[0].copy()
                ny = arr[1].copy()
                nz = arr[2].copy()

                if axis == 'z':
                    # Compute final rotation based on k
                    if k == 1:  # 90°
                        arr[0], arr[1], arr[2] = ny, -nx, nz
                    elif k == 2:  # 180°
                        arr[0], arr[1], arr[2] = -nx, -ny, nz
                    elif k == 3:  # 270°
                        arr[0], arr[1], arr[2] = -ny, nx, nz
                elif axis == 'y':
                    if k == 1:  # 90°
                        arr[0], arr[1], arr[2] = nz, ny, -nx
                    elif k == 2:  # 180°
                        arr[0], arr[1], arr[2] = -nx, ny, -nz
                    elif k == 3:  # 270°
                        arr[0], arr[1], arr[2] = -nz, ny, nx
                else:  # axis == 'x'
                    if k == 1:  # 90°
                        arr[0], arr[1], arr[2] = nx, nz, -ny
                    elif k == 2:  # 180°
                        arr[0], arr[1], arr[2] = nx, -ny, -nz
                    elif k == 3:  # 270°
                        arr[0], arr[1], arr[2] = nx, -nz, ny

            # 3) Remove dummy channel if we added it
            if remove_dummy:
                arr = arr[0]  # from (1, Z, Y, X) back to (Z, Y, X)

            data_dict[key] = arr

        return data_dict
