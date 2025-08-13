import numpy as np
import SimpleITK as sitk
import zarr

from transform_utils import (
    affine_matrix_to_sitk_transform,
    invert_affine_matrix,
    sitk_transform_to_affine_matrix,
    visualize_images_with_transform,
)


def align_zarrs(
    zarr1_path: str, zarr2_path: str, initial_transform: np.ndarray
) -> np.ndarray:
    """
    Align two zarr datasets given their paths and an initial transform.

    Args:
        zarr1_path: Path to the first zarr dataset
        zarr2_path: Path to the second zarr dataset
        initial_transform: Initial affine transformation matrix

    Returns:
        Refined affine transformation matrix
    """
    fixed_zarr = zarr.open(zarr1_path, mode="r")
    moving_zarr = zarr.open(zarr2_path, mode="r")

    fixed_level = int(list(fixed_zarr.array_keys())[-1])
    moving_level = int(list(moving_zarr.array_keys())[-1])

    fixed_image = fixed_zarr[fixed_level]
    moving_image = moving_zarr[moving_level]

    fixed_image = sitk.GetImageFromArray(fixed_image)
    moving_image = sitk.GetImageFromArray(moving_image)

    # Cast images to float32 for SimpleITK compatibility
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    # Normalize images to [0,1] range based on uint8 scale [0,255]
    fixed_image = sitk.ShiftScale(fixed_image, shift=0.0, scale=1.0 / 255.0)
    moving_image = sitk.ShiftScale(moving_image, shift=0.0, scale=1.0 / 255.0)

    # SimpleITK uses a different convention for the initial transform, so we need to invert it
    visualize_images_with_transform(fixed_image, moving_image, sitk_initial_transform)

    # Initialize the registration
    registration = sitk.ImageRegistrationMethod()
    registration.SetInitialTransform(sitk_initial_transform)
    registration.SetFixedInitialTransform(fixed_image_transform)
    registration.SetMovingInitialTransform(moving_image_transform)
    registration.SetMetricAsMattesMutualInformation()
    registration.SetOptimizerAsGradientDescent(
        learningRate=0.01, numberOfIterations=100
    )
    registration.SetInterpolator(sitk.sitkLinear)

    # Set the registration to run
    out_transform = registration.Execute(fixed_image, moving_image)
    out_matrix = out_transform.GetMatrix()
    # invert the matrix
    out_matrix = invert_affine_matrix(out_matrix)

    # In many cases you want the center of rotation to be the physical center of the fixed image (the CenteredTransformCenteredTransformInitializerFilter ensures this).

    return out_matrix
