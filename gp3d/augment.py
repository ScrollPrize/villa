import numpy as np
import numba as nb
import random

import skimage.exposure
from scipy.ndimage import rotate,  map_coordinates, gaussian_filter
from scipy.interpolate import RegularGridInterpolator

from config import AUGMENT_CHANCE, ISO_THRESHOLD

@nb.jit(nopython=True, fastmath=True)
def random_noise_3d_uint8(volume, noise_std=8):  # ~3% of 255
    """Fast 3D Gaussian noise addition for uint8"""
    result = np.empty_like(volume)
    flat_vol = volume.flatten()
    flat_result = result.flatten()

    for i in range(flat_vol.size):
        noise = np.random.normal(0.0, noise_std)
        new_val = int(flat_vol[i]) + int(noise)
        flat_result[i] = max(0, min(255, new_val))

    return result


def anisotropic_gaussian_blur_3d_uint8(volume):
    """Anisotropic Gaussian blur for uint8"""
    sigma_z = random.uniform(0.5, 1.0)
    sigma_y = random.uniform(0.5, 1.0)
    sigma_x = random.uniform(0.5, 1.0)

    # scipy can handle uint8 directly
    volume = gaussian_filter(volume.astype(np.float32), sigma=(sigma_z, sigma_y, sigma_x))
    return np.clip(volume, 0, 255).astype(np.uint8)


def random_rotation_3d_uint8(volume, mask):
    """Random rotation in 3D space for uint8"""
    angle_x = random.uniform(-180, 180)
    angle_y = random.uniform(-180, 180)
    angle_z = random.uniform(-180, 180)

    # Convert to float32 for rotation, then back to uint8
    volume_f = volume.astype(np.float32)
    volume_f = rotate(volume_f, angle_x, axes=(1, 2), reshape=False, order=1)
    volume_f = rotate(volume_f, angle_y, axes=(0, 2), reshape=False, order=1)
    volume_f = rotate(volume_f, angle_z, axes=(0, 1), reshape=False, order=1)
    volume = np.clip(volume_f, 0, 255).astype(np.uint8)

    # Rotate mask with nearest neighbor
    mask = rotate(mask, angle_x, axes=(1, 2), reshape=False, order=0)
    mask = rotate(mask, angle_y, axes=(0, 2), reshape=False, order=0)
    mask = rotate(mask, angle_z, axes=(0, 1), reshape=False, order=0)

    return volume, mask


def elastic_transform_3d_uint8(volume, mask, alpha=500, sigma=20):
    """3D Elastic deformation for uint8"""
    shape = volume.shape

    random_state = np.random.RandomState(None)
    dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(z + dz, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    # Convert to float32 for map_coordinates, then back to uint8
    volume_f = volume.astype(np.float32)
    volume_f = map_coordinates(volume_f, indices, order=1, mode='reflect').reshape(shape)
    volume = np.clip(volume_f, 0, 255).astype(np.uint8)

    mask = map_coordinates(mask, indices, order=0, mode='reflect').reshape(shape)

    return volume, mask


def grid_distortion_3d_uint8(volume, mask, num_steps=5, distort_limit=0.3):
    """3D Grid distortion for uint8"""
    shape = volume.shape

    grid_z = np.linspace(0, shape[0] - 1, num_steps)
    grid_y = np.linspace(0, shape[1] - 1, num_steps)
    grid_x = np.linspace(0, shape[2] - 1, num_steps)

    distort_z = np.random.uniform(-distort_limit, distort_limit, (num_steps, num_steps, num_steps))
    distort_y = np.random.uniform(-distort_limit, distort_limit, (num_steps, num_steps, num_steps))
    distort_x = np.random.uniform(-distort_limit, distort_limit, (num_steps, num_steps, num_steps))

    z_coords = np.arange(shape[0])
    y_coords = np.arange(shape[1])
    x_coords = np.arange(shape[2])

    f_z = RegularGridInterpolator((grid_z, grid_y, grid_x), distort_z)
    f_y = RegularGridInterpolator((grid_z, grid_y, grid_x), distort_y)
    f_x = RegularGridInterpolator((grid_z, grid_y, grid_x), distort_x)

    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    points = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=-1)

    dz = f_z(points).reshape(shape) * shape[0]
    dy = f_y(points).reshape(shape) * shape[1]
    dx = f_x(points).reshape(shape) * shape[2]

    indices = np.reshape(zz + dz, (-1, 1)), np.reshape(yy + dy, (-1, 1)), np.reshape(xx + dx, (-1, 1))

    # Convert to float32 for map_coordinates, then back to uint8
    volume_f = volume.astype(np.float32)
    volume_f = map_coordinates(volume_f, indices, order=1, mode='reflect').reshape(shape)
    volume = np.clip(volume_f, 0, 255).astype(np.uint8)

    mask = map_coordinates(mask, indices, order=0, mode='reflect').reshape(shape)

    return volume, mask


def temporal_slice_augment_uint8(volume, mask):
    """Generalized slice reordering and cutout - works with uint8"""
    axis = random.randint(0, 2)
    axis_size = volume.shape[axis]

    volume_tmp = np.zeros_like(volume)
    mask_tmp = np.zeros_like(mask)

    cropping_num = random.randint(12, 20)
    cropping_num = min(cropping_num, axis_size - 1)

    start_idx = random.randint(0, axis_size - cropping_num)
    crop_indices = np.arange(start_idx, start_idx + cropping_num)

    start_paste_idx = random.randint(0, axis_size - cropping_num)

    if axis == 0:
        volume_tmp[start_paste_idx:start_paste_idx + cropping_num] = volume[crop_indices]
        mask_tmp[start_paste_idx:start_paste_idx + cropping_num] = mask[crop_indices]
    elif axis == 1:
        volume_tmp[:, start_paste_idx:start_paste_idx + cropping_num] = volume[:, crop_indices]
        mask_tmp[:, start_paste_idx:start_paste_idx + cropping_num] = mask[:, crop_indices]
    else:
        volume_tmp[:, :, start_paste_idx:start_paste_idx + cropping_num] = volume[:, :, crop_indices]
        mask_tmp[:, :, start_paste_idx:start_paste_idx + cropping_num] = mask[:, :, crop_indices]

    if random.random() > 0.4:
        cutout_idx = random.randint(0, 3)
        if cutout_idx > 0:
            tmp_indices = np.arange(start_paste_idx, start_paste_idx + cropping_num)
            np.random.shuffle(tmp_indices)
            cutout_slices = tmp_indices[:cutout_idx]

            if axis == 0:
                volume_tmp[cutout_slices] = 0
                mask_tmp[cutout_slices] = 0
            elif axis == 1:
                volume_tmp[:, cutout_slices] = 0
                mask_tmp[:, cutout_slices] = 0
            else:
                volume_tmp[:, :, cutout_slices] = 0
                mask_tmp[:, :, cutout_slices] = 0

    return volume_tmp, mask_tmp


@nb.jit(nopython=True, fastmath=True)
def random_gamma_uint8(volume, gamma_min=0.8, gamma_max=1.2):
    """Fast gamma correction for uint8"""
    gamma = gamma_min + (gamma_max - gamma_min) * np.random.random()

    # Create lookup table for efficiency
    lut = np.empty(256, dtype=np.uint8)
    for i in range(256):
        val = (i / 255.0) ** gamma
        lut[i] = min(255, max(0, int(val * 255)))

    result = np.empty_like(volume)
    flat_vol = volume.flatten()
    flat_result = result.flatten()

    for i in range(flat_vol.size):
        flat_result[i] = lut[flat_vol[i]]

    return result


@nb.jit(nopython=True, fastmath=True)
def random_brightness_contrast_uint8(volume, brightness_range=51, contrast_range=0.2):  # 51 = 20% of 255
    """Fast brightness/contrast adjustment for uint8"""
    brightness = int(-brightness_range + 2 * brightness_range * np.random.random())
    contrast = 0.8 + 0.4 * np.random.random()

    mean_val = 0.0
    count = 0
    flat_vol = volume.flatten()
    for i in range(flat_vol.size):
        mean_val += flat_vol[i]
        count += 1
    mean_val = mean_val / count

    result = np.empty_like(volume)
    flat_result = result.flatten()

    for i in range(flat_vol.size):
        val = (flat_vol[i] - mean_val) * contrast + mean_val + brightness
        flat_result[i] = max(0, min(255, int(val)))

    return result


@nb.jit(nopython=True, fastmath=True)
def fast_gradient_magnitude_uint8(volume):
    """Fast 3D gradient magnitude computation for uint8"""
    d, h, w = volume.shape
    gradient_mag = np.zeros_like(volume, dtype=np.uint8)

    for z in range(d):
        for y in range(h):
            for x in range(w):
                gz = 0
                gy = 0
                gx = 0

                # Z gradient
                if z > 0 and z < d - 1:
                    gz = abs(int(volume[z + 1, y, x]) - int(volume[z - 1, y, x])) // 2
                elif z > 0:
                    gz = abs(int(volume[z, y, x]) - int(volume[z - 1, y, x]))
                elif z < d - 1:
                    gz = abs(int(volume[z + 1, y, x]) - int(volume[z, y, x]))

                # Y gradient
                if y > 0 and y < h - 1:
                    gy = abs(int(volume[z, y + 1, x]) - int(volume[z, y - 1, x])) // 2
                elif y > 0:
                    gy = abs(int(volume[z, y, x]) - int(volume[z, y - 1, x]))
                elif y < h - 1:
                    gy = abs(int(volume[z, y + 1, x]) - int(volume[z, y, x]))

                # X gradient
                if x > 0 and x < w - 1:
                    gx = abs(int(volume[z, y, x + 1]) - int(volume[z, y, x - 1])) // 2
                elif x > 0:
                    gx = abs(int(volume[z, y, x]) - int(volume[z, y, x - 1]))
                elif x < w - 1:
                    gx = abs(int(volume[z, y, x + 1]) - int(volume[z, y, x]))

                total_grad = gz + gy + gx
                gradient_mag[z, y, x] = min(255, total_grad)

    return gradient_mag


@nb.jit(nopython=True, fastmath=True)
def gradient_based_dropout_uint8(volume, dropout_factor=128):  # 0.5 * 255
    """Fast gradient-based dropout for uint8"""
    gradient_mag = fast_gradient_magnitude_uint8(volume)

    # Fast percentile approximation
    flat_grad = gradient_mag.flatten()
    flat_grad_sorted = np.sort(flat_grad)
    threshold_idx = int(0.8 * len(flat_grad_sorted))
    threshold = flat_grad_sorted[threshold_idx]

    result = volume.copy()
    d, h, w = volume.shape

    for z in range(d):
        for y in range(h):
            for x in range(w):
                if gradient_mag[z, y, x] > threshold:
                    result[z, y, x] = (int(result[z, y, x]) * dropout_factor) // 255

    return result


@nb.jit(nopython=True, fastmath=True)
def coarse_dropout_3d_uint8(volume, n_holes=2, hole_size_ratio=0.2):
    """Fast 3D coarse dropout for uint8"""
    d, h, w = volume.shape
    result = volume.copy()

    hole_size = int(hole_size_ratio * min(d, h, w))

    for _ in range(n_holes):
        z_start = int(np.random.random() * (d - hole_size))
        y_start = int(np.random.random() * (h - hole_size))
        x_start = int(np.random.random() * (w - hole_size))

        for z in range(z_start, min(z_start + hole_size, d)):
            for y in range(y_start, min(y_start + hole_size, h)):
                for x in range(x_start, min(x_start + hole_size, w)):
                    result[z, y, x] = 0

    return result


@nb.jit(nopython=True, fastmath=True)
def fast_intensity_shift_3d_uint8(volume, shift_strength=26, block_size=16):  # 26 = 10% of 255
    """Fast spatially-varying intensity shift for uint8"""
    d, h, w = volume.shape
    result = volume.copy()

    blocks_d = (d + block_size - 1) // block_size
    blocks_h = (h + block_size - 1) // block_size
    blocks_w = (w + block_size - 1) // block_size

    shifts = np.random.randint(-shift_strength, shift_strength + 1, (blocks_d, blocks_h, blocks_w))

    for z in range(d):
        for y in range(h):
            for x in range(w):
                block_z = min(z // block_size, blocks_d - 1)
                block_y = min(y // block_size, blocks_h - 1)
                block_x = min(x // block_size, blocks_w - 1)

                new_val = int(result[z, y, x]) + shifts[block_z, block_y, block_x]
                result[z, y, x] = max(0, min(255, new_val))

    return result


@nb.jit(nopython=True, fastmath=True)
def motion_blur_z_axis_uint8(volume, kernel_size=5):
    """Fast 1D convolution along Z-axis for uint8"""
    d, h, w = volume.shape
    result = np.zeros_like(volume)
    half_kernel = kernel_size // 2

    for y in range(h):
        for x in range(w):
            for z in range(d):
                sum_val = 0
                count = 0

                for k in range(-half_kernel, half_kernel + 1):
                    z_idx = z + k
                    if 0 <= z_idx < d:
                        sum_val += int(volume[z_idx, y, x])
                        count += 1

                result[z, y, x] = sum_val // count if count > 0 else 0

    return result


# ============ NEW LIGHTWEIGHT AUGMENTATIONS ============

@nb.jit(nopython=True, fastmath=True)
def intensity_clipping_uint8(volume, clip_min_ratio=0.05, clip_max_ratio=0.95):
    """Random intensity clipping for uint8"""
    # Random clipping thresholds
    clip_min = int(255 * (clip_min_ratio + np.random.random() * 0.1))
    clip_max = int(255 * (clip_max_ratio - np.random.random() * 0.1))

    result = np.empty_like(volume)
    flat_vol = volume.flatten()
    flat_result = result.flatten()

    for i in range(flat_vol.size):
        val = flat_vol[i]
        if val < clip_min:
            flat_result[i] = clip_min
        elif val > clip_max:
            flat_result[i] = clip_max
        else:
            flat_result[i] = val

    return result


@nb.jit(nopython=True, fastmath=True)
def quantization_uint8(volume, levels=16):
    """Reduce bit depth (quantization) for uint8"""
    step = 255 // (levels - 1)

    result = np.empty_like(volume)
    flat_vol = volume.flatten()
    flat_result = result.flatten()

    for i in range(flat_vol.size):
        quantized = (flat_vol[i] // step) * step
        flat_result[i] = min(255, quantized)

    return result


@nb.jit(nopython=True, fastmath=True)
def multiplicative_scaling_uint8(volume, scale_min=0.7, scale_max=1.3):
    """Simple multiplicative intensity scaling for uint8"""
    scale = scale_min + (scale_max - scale_min) * np.random.random()

    result = np.empty_like(volume)
    flat_vol = volume.flatten()
    flat_result = result.flatten()

    for i in range(flat_vol.size):
        scaled = int(flat_vol[i] * scale)
        flat_result[i] = max(0, min(255, scaled))

    return result


@nb.jit(nopython=True, fastmath=True)
def uniform_noise_uint8(volume, noise_range=20):
    """Fast uniform noise addition for uint8"""
    result = np.empty_like(volume)
    flat_vol = volume.flatten()
    flat_result = result.flatten()

    for i in range(flat_vol.size):
        noise = int((np.random.random() - 0.5) * 2 * noise_range)
        new_val = int(flat_vol[i]) + noise
        flat_result[i] = max(0, min(255, new_val))

    return result


@nb.jit(nopython=True, fastmath=True)
def binary_noise_uint8(volume, noise_prob=0.02):
    """Random binary (salt & pepper) noise for uint8"""
    result = volume.copy()
    flat_result = result.flatten()

    for i in range(flat_result.size):
        if np.random.random() < noise_prob:
            if np.random.random() < 0.5:
                flat_result[i] = 0  # pepper
            else:
                flat_result[i] = 255  # salt

    return result


@nb.jit(nopython=True, fastmath=True)
def posterization_uint8(volume, levels=8):
    """Posterization (reduce intensity levels) for uint8"""
    step = 255 // levels

    result = np.empty_like(volume)
    flat_vol = volume.flatten()
    flat_result = result.flatten()

    for i in range(flat_vol.size):
        level = flat_vol[i] // step
        flat_result[i] = min(255, level * step)

    return result


@nb.jit(nopython=True, fastmath=True)
def threshold_transform_uint8(volume, threshold=128, invert_prob=0.3):
    """Threshold-based binary transform with optional inversion"""
    result = np.empty_like(volume)
    flat_vol = volume.flatten()
    flat_result = result.flatten()

    invert = np.random.random() < invert_prob

    for i in range(flat_vol.size):
        if flat_vol[i] > threshold:
            flat_result[i] = 0 if invert else 255
        else:
            flat_result[i] = 255 if invert else 0

    return result


def random_90_rotation_uint8(volume, mask):
    """Fast 90-degree rotations (no interpolation needed)"""
    axis = random.randint(0, 2)
    k = random.randint(1, 3)  # 90, 180, or 270 degrees

    if axis == 0:  # rotate in YZ plane
        volume = np.rot90(volume, k, axes=(1, 2))
        mask = np.rot90(mask, k, axes=(1, 2))
    elif axis == 1:  # rotate in XZ plane
        volume = np.rot90(volume, k, axes=(0, 2))
        mask = np.rot90(mask, k, axes=(0, 2))
    else:  # rotate in XY plane
        volume = np.rot90(volume, k, axes=(0, 1))
        mask = np.rot90(mask, k, axes=(0, 1))

    return volume, mask


def axis_swapping_uint8(volume, mask):
    """Random axis transposition"""
    axes_permutations = [
        (0, 1, 2),  # original
        (0, 2, 1),  # swap Y,Z
        (1, 0, 2),  # swap X,Z
        (1, 2, 0),  # cyclic
        (2, 0, 1),  # cyclic
        (2, 1, 0)  # reverse
    ]

    perm = random.choice(axes_permutations[1:])  # exclude original
    volume = np.transpose(volume, perm)
    mask = np.transpose(mask, perm)

    return volume, mask


def random_shift_uint8(volume, mask, max_shift=10):
    """Integer pixel translations (fast)"""
    shift_z = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)
    shift_x = random.randint(-max_shift, max_shift)

    volume = np.roll(volume, shift_z, axis=0)
    volume = np.roll(volume, shift_y, axis=1)
    volume = np.roll(volume, shift_x, axis=2)

    mask = np.roll(mask, shift_z, axis=0)
    mask = np.roll(mask, shift_y, axis=1)
    mask = np.roll(mask, shift_x, axis=2)

    return volume, mask


@nb.jit(nopython=True, fastmath=True)
def structured_dropout_uint8(volume, grid_size=8, dropout_prob=0.3):
    """Structured grid-based dropout"""
    d, h, w = volume.shape
    result = volume.copy()

    for z in range(0, d, grid_size):
        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                if np.random.random() < dropout_prob:
                    z_end = min(z + grid_size, d)
                    y_end = min(y + grid_size, h)
                    x_end = min(x + grid_size, w)

                    for zi in range(z, z_end):
                        for yi in range(y, y_end):
                            for xi in range(x, x_end):
                                result[zi, yi, xi] = 0

    return result


@nb.jit(nopython=True, fastmath=True)
def bit_shift_uint8(volume, max_shift=2):
    """Fast intensity scaling via bit shifts"""
    shift = np.random.randint(-max_shift, max_shift + 1)

    result = np.empty_like(volume)
    flat_vol = volume.flatten()
    flat_result = result.flatten()

    if shift >= 0:
        for i in range(flat_vol.size):
            shifted = int(flat_vol[i]) << shift
            flat_result[i] = min(255, shifted)
    else:
        for i in range(flat_vol.size):
            flat_result[i] = int(flat_vol[i]) >> abs(shift)

    return result


def subsampling_uint8(volume, mask, factor=2):
    """Spatial subsampling with interpolation back to original size"""
    # Subsample
    sub_vol = volume[::factor, ::factor, ::factor]
    sub_mask = mask[::factor, ::factor, ::factor]

    # Simple nearest neighbor upsampling
    d, h, w = volume.shape
    result_vol = np.zeros_like(volume)
    result_mask = np.zeros_like(mask)

    sub_d, sub_h, sub_w = sub_vol.shape

    for z in range(d):
        for y in range(h):
            for x in range(w):
                sub_z = min(z // factor, sub_d - 1)
                sub_y = min(y // factor, sub_h - 1)
                sub_x = min(x // factor, sub_w - 1)

                result_vol[z, y, x] = sub_vol[sub_z, sub_y, sub_x]
                result_mask[z, y, x] = sub_mask[sub_z, sub_y, sub_x]

    return result_vol, result_mask


@nb.jit(nopython=True, fastmath=True)
def additive_checkerboard_uint8(volume, block_size=8, intensity=30):
    """Add checkerboard pattern to volume"""
    d, h, w = volume.shape
    result = volume.copy()

    for z in range(d):
        for y in range(h):
            for x in range(w):
                block_z = z // block_size
                block_y = y // block_size
                block_x = x // block_size

                if (block_z + block_y + block_x) % 2 == 0:
                    new_val = int(result[z, y, x]) + intensity
                    result[z, y, x] = min(255, new_val)

    return result


@nb.jit(nopython=True, fastmath=True)
def modulo_intensity_uint8(volume, modulo=64):
    """Apply modulo operation to intensities"""
    result = np.empty_like(volume)
    flat_vol = volume.flatten()
    flat_result = result.flatten()

    for i in range(flat_vol.size):
        flat_result[i] = flat_vol[i] % modulo

    return result

def random_iso(volume, mask):
    """apply iso threshold between 0 and ISO_THRESHOLD"""
    threshold = random.randint(0,ISO_THRESHOLD)
    isomask = volume < threshold
    volume[isomask] = 0
    mask[isomask] = 0

    return volume, mask


def apply_all_augmentations(volume, mask):
    """Apply both existing and new lightweight augmentations"""
    #volume, mask = random_iso(volume, mask)

    if random.random() < AUGMENT_CHANCE:
        # Random flips
        if random.random() < 0.33:
            volume = np.ascontiguousarray(volume[::-1])
            mask = np.ascontiguousarray(mask[::-1])
        if random.random() < 0.33:
            volume = np.ascontiguousarray(volume[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if random.random() < 0.33:
            volume = np.ascontiguousarray(volume[:, :, ::-1])
            mask = np.ascontiguousarray(mask[:, :, ::-1])

    # Mix of 90-degree and arbitrary rotations
    if random.random() < AUGMENT_CHANCE:
        if random.random() < 0.5:
            volume, mask = random_90_rotation_uint8(volume, mask)
        else:
            volume, mask = random_rotation_3d_uint8(volume, mask)

    if random.random() < AUGMENT_CHANCE:
        dropout_choice = random.choice(['gradient', 'structured', 'coarse'])
        if dropout_choice == 'gradient':
            volume = gradient_based_dropout_uint8(volume, random.randint(77, 178))
        elif dropout_choice == 'structured':
            volume = structured_dropout_uint8(volume, random.choice([4, 8, 16]))
        else:
            volume = coarse_dropout_3d_uint8(volume)

    if random.random() < AUGMENT_CHANCE:
        volume, mask = axis_swapping_uint8(volume, mask)


    if random.random() < AUGMENT_CHANCE/4:
        if random.random() < 0.5:
            volume, mask = random_shift_uint8(volume, mask)
        else:
            volume, mask = temporal_slice_augment_uint8(volume, mask)

    if random.random() < AUGMENT_CHANCE/4:
        if random.random() < 0.5:
            volume, mask = elastic_transform_3d_uint8(volume, mask)
        else:
            volume, mask = grid_distortion_3d_uint8(volume, mask)

    if random.random() < AUGMENT_CHANCE/4:
        intensity_choice = random.choice(['brightness_contrast', 'clipping', 'scaling'])
        if intensity_choice == 'brightness_contrast':
            volume = random_brightness_contrast_uint8(volume)
        elif intensity_choice == 'clipping':
            volume = intensity_clipping_uint8(volume)
        else:
            volume = multiplicative_scaling_uint8(volume)

    if random.random() < AUGMENT_CHANCE/4:
        volume = random_gamma_uint8(volume)

    if random.random() < AUGMENT_CHANCE/4:
        if random.random() < 0.5:
            volume = fast_intensity_shift_3d_uint8(volume)
        else:
            choice = random.choice(['quantization', 'posterization', 'bit_shift'])
            if choice == 'quantization':
                volume = quantization_uint8(volume, random.choice([8, 16, 32]))
            elif choice == 'posterization':
                volume = posterization_uint8(volume, random.choice([4, 8, 16]))
            else:
                volume = bit_shift_uint8(volume)

    if random.random() < AUGMENT_CHANCE/4:
        kernel_size = random.choice([3, 5, 7])
        volume = motion_blur_z_axis_uint8(volume, kernel_size)



    if random.random() < AUGMENT_CHANCE/4:
        noise_type = random.choice(['gaussian', 'uniform', 'binary', 'anisotropic_blur'])
        if noise_type == 'gaussian':
            volume = random_noise_3d_uint8(volume, random.randint(3, 13))
        elif noise_type == 'uniform':
            volume = uniform_noise_uint8(volume, random.randint(10, 30))
        elif noise_type == 'binary':
            volume = binary_noise_uint8(volume, random.uniform(0.01, 0.05))
        else:
            volume = anisotropic_gaussian_blur_3d_uint8(volume)

    if random.random() < AUGMENT_CHANCE/4:
        special_choice = random.choice(['checkerboard', 'modulo', 'threshold'])
        if special_choice == 'checkerboard':
            volume = additive_checkerboard_uint8(volume, random.choice([4, 8, 16]))
        elif special_choice == 'modulo':
            volume = modulo_intensity_uint8(volume, random.choice([32, 64, 128]))
        else:
            volume = threshold_transform_uint8(volume, random.randint(64, 192))

    if random.random() < AUGMENT_CHANCE/4:
        volume, mask = subsampling_uint8(volume, mask, random.choice([2, 3]))

    return volume, mask