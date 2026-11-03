include(GNUInstallDirs)
find_package(Threads REQUIRED)

# Publish the existing Python-independent headers without maintaining copies.
# configure_file tracks edits and keeps build-tree and installed include paths identical.
file(GLOB_RECURSE _vc_native_headers CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/*.cuh")
list(FILTER _vc_native_headers EXCLUDE REGEX "(^|/)([^/]*_)?bindings\\.hpp$")
foreach(header IN LISTS _vc_native_headers)
    configure_file("src/cpp/${header}" "include/vc_reader/native/${header}" COPYONLY)
    get_filename_component(directory "${header}" DIRECTORY)
    install(FILES "src/cpp/${header}" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/vc_reader/native/${directory}")
endforeach()
add_library(vc_reader_headers INTERFACE)
add_library(vc_reader::headers ALIAS vc_reader_headers)
set_target_properties(vc_reader_headers PROPERTIES EXPORT_NAME headers)
target_compile_features(vc_reader_headers INTERFACE cxx_std_20)
target_include_directories(vc_reader_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(vc_reader_headers INTERFACE Threads::Threads)

add_library(vc_reader_transforms INTERFACE)
add_library(vc_reader::transforms ALIAS vc_reader_transforms)
set_target_properties(vc_reader_transforms PROPERTIES EXPORT_NAME transforms)
target_link_libraries(vc_reader_transforms INTERFACE vc_reader::headers)

if(VC_READER_ENABLE_CUDA)
    add_library(vc_reader_transform_cuda STATIC
        src/cpp/transforms/cuda/bias_field.cu
        src/cpp/transforms/cuda/plasma_brightness_contrast.cu
        src/cpp/transforms/cuda/plasma_shadow.cu
        src/cpp/transforms/cuda/illumination.cu
        src/cpp/transforms/cuda/vignetting.cu
        src/cpp/transforms/cuda/random_tone_curve.cu
        src/cpp/transforms/cuda/multiplicative_noise.cu
        src/cpp/transforms/cuda/shot_noise.cu
        src/cpp/transforms/cuda/additive_noise.cu
        src/cpp/transforms/cuda/salt_and_pepper.cu
        src/cpp/transforms/cuda/motion_blur.cu
        src/cpp/transforms/cuda/defocus.cu
        src/cpp/transforms/cuda/zoom_blur.cu
        src/cpp/transforms/cuda/equalize.cu
        src/cpp/transforms/cuda/clahe.cu
        src/cpp/transforms/cuda/histogram_matching.cu
        src/cpp/transforms/cuda/pixel_distribution_adaptation.cu
        src/cpp/transforms/cuda/fourier_domain_adaptation.cu
        src/cpp/transforms/cuda/unsharp_mask.cu
        src/cpp/transforms/cuda/gibbs_ringing.cu
        src/cpp/transforms/cuda/ringing_overshoot.cu
        src/cpp/transforms/cuda/foreground_crop.cu
        src/cpp/transforms/cuda/channel_misalignment.cu
        src/cpp/transforms/cuda/grid_dropout.cu
        src/cpp/transforms/cuda/stripe_masking.cu
        src/cpp/transforms/cuda/foreground_dropout.cu
        src/cpp/transforms/cuda/object_dropout.cu
        src/cpp/transforms/cuda/fourier.cu
        src/cpp/transforms/cuda/dropout_mask.cu
        src/cpp/transforms/cuda/clip_outliers.cu
        src/cpp/transforms/cuda/crop_pad.cu
        src/cpp/transforms/cuda/affine.cu
        src/cpp/transforms/cuda/sheet_compression.cu
        src/cpp/transforms/cuda/strip_geometry.cu
        src/cpp/transforms/cuda/surface_perturb.cu
        src/cpp/transforms/cuda/morphology.cu
        src/cpp/transforms/cuda/remove_components.cu
        src/cpp/transforms/cuda/replace_label.cu
        src/cpp/transforms/cuda/regions.cu
        src/cpp/transforms/cuda/one_hot.cu
        src/cpp/transforms/cuda/mask_image.cu
        src/cpp/transforms/cuda/deep_supervision.cu
        src/cpp/transforms/cuda/to_pseudo_2d.cu
        src/cpp/transforms/cuda/from_pseudo_2d.cu
        src/cpp/transforms/cuda/backend.cu
        src/cpp/transforms/cuda/blank_rectangle.cu
        src/cpp/transforms/cuda/brightness_add.cu
        src/cpp/transforms/cuda/brightness_gradient.cu
        src/cpp/transforms/cuda/brightness_multiply.cu
        src/cpp/transforms/cuda/contrast.cu
        src/cpp/transforms/cuda/directional.cu
        src/cpp/transforms/cuda/elastic.cu
        src/cpp/transforms/cuda/flip.cu
        src/cpp/transforms/cuda/gamma.cu
        src/cpp/transforms/cuda/gaussian_blur.cu
        src/cpp/transforms/cuda/gaussian_noise.cu
        src/cpp/transforms/cuda/geometry.cu
        src/cpp/transforms/cuda/invert.cu
        src/cpp/transforms/cuda/local_contrast.cu
        src/cpp/transforms/cuda/local_field.cu
        src/cpp/transforms/cuda/local_gamma.cu
        src/cpp/transforms/cuda/local_smoothing.cu
        src/cpp/transforms/cuda/low_resolution.cu
        src/cpp/transforms/cuda/median_filter.cu
        src/cpp/transforms/cuda/medial_surface.cu
        src/cpp/transforms/cuda/normalization.cu
        src/cpp/transforms/cuda/oriented_blur.cu
        src/cpp/transforms/cuda/random_binary_operator.cu
        src/cpp/transforms/cuda/resident.cu
        src/cpp/transforms/cuda/rician_noise.cu
        src/cpp/transforms/cuda/rotate90.cu
        src/cpp/transforms/cuda/sharpen.cu
        src/cpp/transforms/cuda/slice_illumination.cu
        src/cpp/transforms/cuda/smear.cu
        src/cpp/transforms/cuda/distance_transform.cu
        src/cpp/transforms/cuda/structure_tensor.cu
        src/cpp/transforms/cuda/statistics.cu
        src/cpp/transforms/cuda/thick_slice.cu
        src/cpp/transforms/cuda/transpose.cu
    )
    add_library(vc_reader::transforms_cuda ALIAS vc_reader_transform_cuda)
    set_target_properties(vc_reader_transform_cuda PROPERTIES EXPORT_NAME transforms_cuda
        POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    target_compile_definitions(vc_reader_transform_cuda PUBLIC VC_READER_CUDA=1)
    target_compile_options(vc_reader_transform_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_link_libraries(vc_reader_transform_cuda PUBLIC vc_reader::headers CUDA::cudart CUDA::cufft)
    target_link_libraries(vc_reader_transforms INTERFACE vc_reader::transforms_cuda)
endif()
