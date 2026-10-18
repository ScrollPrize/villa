cmake_minimum_required(VERSION 3.20)
project(vc_reader_native LANGUAGES CXX)

# Native read path for vc_reader (module vc_reader._native), built by
# scikit-build-core. Needs libcurl, c-blosc, zstd and SQLite development files:
#   Ubuntu: apt install libcurl4-openssl-dev libblosc-dev libzstd-dev libsqlite3-dev
#   macOS:  brew install c-blosc zstd sqlite   (libcurl comes with the SDK)
# VC-Delta3D is fetched at a pinned commit (the same codec VC3D links).

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(nanobind CONFIG REQUIRED)
if(nanobind_VERSION VERSION_LESS 2.10.2)
    message(FATAL_ERROR "vc_reader requires nanobind >= 2.10.2 for device-array ownership")
endif()
find_package(Threads REQUIRED)

# The augmentation module has no dependency on reader networking or codecs.
nanobind_add_module(_augment NB_STATIC NOMINSIZE cpp/augment/bindings.cpp)
install(TARGETS _augment LIBRARY DESTINATION vc_reader)

# CPU builds never require the toolkit or load the CUDA runtime.
option(VC_READER_ENABLE_CUDA "Build optional CUDA reader and augmentation backends" OFF)
if(VC_READER_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(_augment PRIVATE
        cpp/augment/cuda/bias_field.cu
        cpp/augment/cuda/plasma_brightness_contrast.cu
        cpp/augment/cuda/plasma_shadow.cu
        cpp/augment/cuda/illumination.cu
        cpp/augment/cuda/vignetting.cu
        cpp/augment/cuda/random_tone_curve.cu
        cpp/augment/cuda/multiplicative_noise.cu
        cpp/augment/cuda/shot_noise.cu
        cpp/augment/cuda/additive_noise.cu
        cpp/augment/cuda/salt_and_pepper.cu
        cpp/augment/cuda/motion_blur.cu
        cpp/augment/cuda/defocus.cu
        cpp/augment/cuda/zoom_blur.cu
        cpp/augment/cuda/equalize.cu
        cpp/augment/cuda/clahe.cu
        cpp/augment/cuda/histogram_matching.cu
        cpp/augment/cuda/pixel_distribution_adaptation.cu
        cpp/augment/cuda/fourier_domain_adaptation.cu
        cpp/augment/cuda/unsharp_mask.cu
        cpp/augment/cuda/gibbs_ringing.cu
        cpp/augment/cuda/ringing_overshoot.cu
        cpp/augment/cuda/foreground_crop.cu
        cpp/augment/cuda/channel_misalignment.cu
        cpp/augment/cuda/grid_dropout.cu
        cpp/augment/cuda/stripe_masking.cu
        cpp/augment/cuda/foreground_dropout.cu
        cpp/augment/cuda/object_dropout.cu
        cpp/augment/cuda/fourier.cu
        cpp/augment/cuda/dropout_mask.cu
        cpp/augment/cuda/clip_outliers.cu
        cpp/augment/cuda/crop_pad.cu
        cpp/augment/cuda/affine.cu
        cpp/augment/cuda/sheet_compression.cu
        cpp/augment/cuda/strip_geometry.cu
        cpp/augment/cuda/surface_perturb.cu
        cpp/augment/cuda/morphology.cu
        cpp/augment/cuda/remove_components.cu
        cpp/augment/cuda/replace_label.cu
        cpp/augment/cuda/regions.cu
        cpp/augment/cuda/one_hot.cu
        cpp/augment/cuda/mask_image.cu
        cpp/augment/cuda/deep_supervision.cu
        cpp/augment/cuda/to_pseudo_2d.cu
        cpp/augment/cuda/from_pseudo_2d.cu
        cpp/augment/cuda/backend.cu
        cpp/augment/cuda/blank_rectangle.cu
        cpp/augment/cuda/brightness_add.cu
        cpp/augment/cuda/brightness_gradient.cu
        cpp/augment/cuda/brightness_multiply.cu
        cpp/augment/cuda/contrast.cu
        cpp/augment/cuda/directional.cu
        cpp/augment/cuda/elastic.cu
        cpp/augment/cuda/flip.cu
        cpp/augment/cuda/gamma.cu
        cpp/augment/cuda/gaussian_blur.cu
        cpp/augment/cuda/gaussian_noise.cu
        cpp/augment/cuda/geometry.cu
        cpp/augment/cuda/invert.cu
        cpp/augment/cuda/local_contrast.cu
        cpp/augment/cuda/local_field.cu
        cpp/augment/cuda/local_gamma.cu
        cpp/augment/cuda/local_smoothing.cu
        cpp/augment/cuda/low_resolution.cu
        cpp/augment/cuda/median_filter.cu
        cpp/augment/cuda/medial_surface.cu
        cpp/augment/cuda/normalization.cu
        cpp/augment/cuda/oriented_blur.cu
        cpp/augment/cuda/random_binary_operator.cu
        cpp/augment/cuda/resident.cu
        cpp/augment/cuda/rician_noise.cu
        cpp/augment/cuda/rotate90.cu
        cpp/augment/cuda/sharpen.cu
        cpp/augment/cuda/slice_illumination.cu
        cpp/augment/cuda/smear.cu
        cpp/augment/cuda/distance_transform.cu
        cpp/augment/cuda/structure_tensor.cu
        cpp/augment/cuda/statistics.cu
        cpp/augment/cuda/thick_slice.cu
        cpp/augment/cuda/transpose.cu
    )
    target_compile_definitions(_augment PRIVATE VC_READER_CUDA=1)
    target_link_libraries(_augment PRIVATE CUDA::cudart CUDA::cufft)
    set_target_properties(_augment PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    target_compile_options(_augment PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

option(VC_READER_BUILD_READER "Build the native Zarr reader as well as augmentation" ON)
if(VC_READER_BUILD_READER)
    find_package(CURL REQUIRED)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(BLOSC REQUIRED IMPORTED_TARGET blosc)
    pkg_check_modules(ZSTD REQUIRED IMPORTED_TARGET libzstd)
    find_package(SQLite3 3.24 REQUIRED)

    include(FetchContent)
    set(VC_DELTA3D_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    set(VC_DELTA3D_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(VC_DELTA3D_INSTALL_CPP OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(vc_delta3d
        GIT_REPOSITORY https://github.com/ScrollPrize/vc-delta3d.git
        GIT_TAG 457f612aa0379a73fd750d8b8e61eb8b11bf45fd  # v0.1.1
        GIT_SHALLOW FALSE)
    FetchContent_MakeAvailable(vc_delta3d)
    set_target_properties(vc_delta3d PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # NOMINSIZE: nanobind otherwise appends -Os, which costs the point samplers ~8%.
    nanobind_add_module(_native NB_STATIC NOMINSIZE cpp/bindings.cpp)
    if(VC_READER_ENABLE_CUDA)
        target_sources(_native PRIVATE cpp/reader_cuda/kernels.cu cpp/aligned/kernels.cu cpp/surface/kernels.cu)
        # Geometric predicates and categorical ties use separate float32 rounding.
        set_source_files_properties(cpp/surface/kernels.cu PROPERTIES COMPILE_OPTIONS "--fmad=false")
        target_compile_definitions(_native PRIVATE VC_READER_CUDA=1)
        target_link_libraries(_native PRIVATE CUDA::cudart)
        set_target_properties(_native PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    endif()
    target_include_directories(_native PRIVATE cpp)
    target_link_libraries(_native PRIVATE CURL::libcurl PkgConfig::BLOSC PkgConfig::ZSTD SQLite::SQLite3 vc_delta3d::vc_delta3d Threads::Threads)
    install(TARGETS _native LIBRARY DESTINATION vc_reader)
endif()
