import numpy as np
import SimpleITK as sitk
import zarr

from transform_utils import (
    affine_matrix_to_sitk_transform,
    invert_affine_matrix,
    sitk_affine_transform_to_matrix,
    matrix_swap_between_xyz_and_zyx,
    check_images_with_transform,
)


def align_zarrs(zarr1_path: str, zarr2_path: str, M_init_ng: np.ndarray) -> np.ndarray:
    """
    Align two zarr datasets given their paths and an initial transform.

    Args:
        zarr1_path: Path to the first zarr dataset
        zarr2_path: Path to the second zarr dataset
        M_init_ng: Initial affine transformation matrix in ZYX order (from neuroglancer)

    Returns:
        Refined affine transformation matrix
    """

    fixed_zarr = zarr.open(zarr1_path, mode="r")
    moving_zarr = zarr.open(zarr2_path, mode="r")

    fixed_level = int(list(fixed_zarr.array_keys())[-1])
    moving_level = int(list(moving_zarr.array_keys())[-1])

    fixed_image = fixed_zarr[fixed_level]
    moving_image = moving_zarr[moving_level]

    fixed_scale_factor = 2**fixed_level
    moving_scale_factor = 2**moving_level

    fixed_image = sitk.GetImageFromArray(fixed_image)
    moving_image = sitk.GetImageFromArray(moving_image)

    # Set image spacing based on scale factor
    fixed_image.SetSpacing([fixed_scale_factor] * 3)
    moving_image.SetSpacing([moving_scale_factor] * 3)

    # Cast images to float32 for SimpleITK compatibility
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    # Convert the neuroglancer matrix (ZYX) to a SimpleITK ordering (XYZ)
    # Also invert it so it maps from virtual to moving (expected by SimpleITK) instead of moving to fixed (neuroglancer)
    M_init = invert_affine_matrix(matrix_swap_between_xyz_and_zyx(M_init_ng))
    T_moving_init = affine_matrix_to_sitk_transform(M_init)

    check_images_with_transform(fixed_image, moving_image, T_moving_init)

    # Use a new identity affine as the initial transform
    registration = sitk.ImageRegistrationMethod()
    registration.SetMovingInitialTransform(T_moving_init)
    registration.SetInitialTransform(sitk.AffineTransform(3))
    registration.SetMetricAsMeanSquares()
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.1,
        minStep=0.00001,
        numberOfIterations=100,
    )
    registration.SetInterpolator(sitk.sitkLinear)

    # Add callback to print metric every iteration
    def print_metric(registration: sitk.ImageRegistrationMethod):
        print(
            f"Iteration: {registration.GetOptimizerIteration()}, "
            f"Metric: {registration.GetMetricValue():.6f}"
            f"Matrix: {registration.GetInitialTransform().GetParameters()}"
        )

    registration.AddCommand(sitk.sitkIterationEvent, lambda: print_metric(registration))

    # Set the registration to run
    T_opt = registration.Execute(fixed_image, moving_image)

    print(f"Final metric value: {registration.GetMetricValue()}")
    print(
        f"Optimizer's stopping condition, {registration.GetOptimizerStopConditionDescription()}"
    )

    M_opt = sitk_affine_transform_to_matrix(T_opt)

    # Compose the initial transform with the optimized transform
    # https://simpleitk.readthedocs.io/en/master/registrationOverview.html#transforms-and-image-spaces
    M_out = M_init @ M_opt

    # Invert the matrix to get the transform from moving to fixed
    # Also convert to neuroglancer ordering (ZYX)
    M_out_ng = matrix_swap_between_xyz_and_zyx(invert_affine_matrix(M_out))

    # Return a neuroglancer-suitable matrix
    return M_out_ng
