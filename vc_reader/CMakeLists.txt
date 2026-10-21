# CMake 3.25.2 adds CUDA C++20 compiler flags for NVCC 12 and newer.
cmake_minimum_required(VERSION 3.25.2)
project(vc_reader_extensions LANGUAGES CXX)

# C++ read path for vc_reader (module vc_reader._reader), built by
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

# Initialize CUDA architecture defaults before creating targets.
# An explicit CPU build skips CUDA discovery entirely.
if(NOT DEFINED VC_READER_ENABLE_CUDA OR VC_READER_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
        # Toolkit discovery also searches standard locations outside PATH.
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_NVCC_EXECUTABLE)
            set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}"
                CACHE FILEPATH "CUDA compiler")
        endif()
    endif()
    include(CheckLanguage)
    check_language(CUDA)
endif()
set(_vc_reader_cuda_default OFF)
if(CMAKE_CUDA_COMPILER)
    set(_vc_reader_cuda_default ON)
endif()
option(VC_READER_ENABLE_CUDA "Build optional CUDA reader and transform backends"
    ${_vc_reader_cuda_default})
if(VC_READER_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# The transform module has no dependency on reader networking or codecs.
nanobind_add_module(_transforms NB_STATIC NOMINSIZE src/cpp/transforms/bindings.cpp)
install(TARGETS _transforms LIBRARY DESTINATION vc_reader)

if(VC_READER_ENABLE_CUDA)
    target_sources(_transforms PRIVATE
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
    target_compile_definitions(_transforms PRIVATE VC_READER_CUDA=1)
    target_link_libraries(_transforms PRIVATE CUDA::cudart CUDA::cufft)
    set_target_properties(_transforms PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    target_compile_options(_transforms PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

option(VC_READER_BUILD_READER "Build the C++ Zarr reader as well as transform" ON)
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
    nanobind_add_module(_reader NB_STATIC NOMINSIZE src/cpp/bindings.cpp)
    if(VC_READER_ENABLE_CUDA)
        target_sources(_reader PRIVATE src/cpp/reader_cuda/kernels.cu src/cpp/aligned/kernels.cu src/cpp/surface/kernels.cu src/cpp/sheet_normal/kernels.cu src/cpp/geometry/kernels.cu)
        # Separate float32 rounding preserves geometric ties and filter agreement.
        set_source_files_properties(src/cpp/surface/kernels.cu src/cpp/sheet_normal/kernels.cu src/cpp/geometry/kernels.cu PROPERTIES COMPILE_OPTIONS "--fmad=false")
        target_compile_definitions(_reader PRIVATE VC_READER_CUDA=1)
        target_link_libraries(_reader PRIVATE CUDA::cudart)
        set_target_properties(_reader PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    endif()
    target_include_directories(_reader PRIVATE src/cpp)
    target_link_libraries(_reader PRIVATE CURL::libcurl PkgConfig::BLOSC PkgConfig::ZSTD SQLite::SQLite3 vc_delta3d::vc_delta3d Threads::Threads)
    install(TARGETS _reader LIBRARY DESTINATION vc_reader)
endif()
