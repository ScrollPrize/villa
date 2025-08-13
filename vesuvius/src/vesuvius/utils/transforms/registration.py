import numpy as np
import SimpleITK as sitk
import zarr

from transform_utils import (
    affine_matrix_to_sitk_transform,
    invert_affine_matrix,
    sitk_transform_to_affine_matrix,
    check_images_with_transform,
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

    # SimpleITK transforms from fixed to moving, so we need to invert the neuroglancer transform
    inverted_initial_transform = invert_affine_matrix(initial_transform)
    sitk_initial_transform = affine_matrix_to_sitk_transform(inverted_initial_transform)

    check_images_with_transform(fixed_image, moving_image, sitk_initial_transform)

    # Initialize the registration
    registration = sitk.ImageRegistrationMethod()
    registration.SetInitialTransform(sitk_initial_transform)
    registration.SetMetricAsMeanSquares()
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.1,
        minStep=0.00001,
        numberOfIterations=1000,
    )
    registration.SetInterpolator(sitk.sitkLinear)

    # Add callback to print metric every iteration
    def print_metric(registration):
        print(
            f"Iteration: {registration.GetOptimizerIteration()}, "
            f"Metric: {registration.GetMetricValue():.6f}"
        )

    registration.AddCommand(sitk.sitkIterationEvent, lambda: print_metric(registration))

    # Set the registration to run
    out_transform = registration.Execute(fixed_image, moving_image)

    print(f"Final metric value: {registration.GetMetricValue()}")
    print(
        f"Optimizer's stopping condition, {registration.GetOptimizerStopConditionDescription()}"
    )

    out_matrix = sitk_transform_to_affine_matrix(out_transform)
    inverted_out_matrix = invert_affine_matrix(out_matrix)

    return inverted_out_matrix
