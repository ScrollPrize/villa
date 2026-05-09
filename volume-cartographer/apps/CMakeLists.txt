############
# GUI Apps #
############
add_subdirectory(VC3D)

#################
# CMD Line Apps #
#################

add_executable(vc_render_video src/vc_render_video.cpp)
target_link_libraries(vc_render_video opencv_core vc_core opencv_videoio)

add_executable(vc_render_tifxyz src/vc_render_tifxyz.cpp)
target_link_libraries(vc_render_tifxyz vc_core vc_flattening Boost::program_options TIFF::TIFF)

add_executable(vc_objrefine src/ObjAlphaCompRefinement.cpp)
target_link_libraries(vc_objrefine vc_core)

add_executable(vc_grow_seg_from_seed src/vc_grow_seg_from_seed.cpp)
target_link_libraries(vc_grow_seg_from_seed vc_core vc_ui vc_tracer Boost::program_options)

add_executable(vc_seg_add_overlap src/vc_seg_add_overlap.cpp)
target_link_libraries(vc_seg_add_overlap vc_core)

add_executable(vc_grow_seg_from_segments src/vc_grow_seg_from_segments.cpp)
target_link_libraries(vc_grow_seg_from_segments vc_core vc_tracer Boost::program_options)

add_executable(vc_tifxyz2obj src/vc_tifxyz2obj.cpp)
target_link_libraries(vc_tifxyz2obj vc_core)

add_executable(vc_tifxyz2zarr_sparse src/vc_tifxyz2zarr_sparse.cpp)
target_link_libraries(vc_tifxyz2zarr_sparse vc_core Boost::program_options)

add_executable(vc_add_ignore_label src/vc_add_ignore_label.cpp)
target_link_libraries(vc_add_ignore_label
    vc_core
    Boost::program_options
    CGAL::CGAL
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
)

add_executable(vc_obj2tifxyz src/vc_obj2tifxyz.cpp)
target_link_libraries(vc_obj2tifxyz vc_core)

add_executable(vc_obj2tifxyz_legacy src/vc_obj2tifxyz_legacy.cpp)
target_link_libraries(vc_obj2tifxyz_legacy vc_core)

add_executable(vc_flatten src/vc_flatten.cpp)
target_link_libraries(vc_flatten vc_core vc_flattening Boost::program_options)

add_executable(vc_tifxyz_inp_mask src/vc_tifxyz_inp_mask.cpp)
target_link_libraries(vc_tifxyz_inp_mask vc_core)

add_executable(vc_tifxyz_trim src/vc_tifxyz_trim.cpp)
target_link_libraries(vc_tifxyz_trim vc_core opencv_core opencv_imgcodecs TIFF::TIFF)

add_executable(vc_cut_windings src/vc_cut_windings.cpp)
target_link_libraries(vc_cut_windings vc_core opencv_core)

add_executable(vc_tifxyz_winding src/vc_tifxyz_winding.cpp)
target_link_libraries(vc_tifxyz_winding vc_core)

add_executable(vc_calc_surface_metrics src/vc_calc_surface_metrics.cpp)
target_link_libraries(vc_calc_surface_metrics vc_core vc_ui Boost::program_options)

add_executable(vc_tifxyz_gengt src/vc_tifxyz_gengt.cpp)
target_link_libraries(vc_tifxyz_gengt vc_core vc_ui Boost::program_options)

set(vc_diffuse_winding_sources
    diffusion/continuous3d.cpp
    diffusion/continous.cpp
    diffusion/discrete.cpp
    diffusion/spiral_ceres.cpp
    diffusion/spiral_common.cpp
    diffusion/spiral.cpp
    diffusion/spiral2.cpp
    diffusion/spiral2cont.cpp
    diffusion/support.cpp
    diffusion/vc_diffuse_winding.cpp
)

add_executable(vc_diffuse_winding ${vc_diffuse_winding_sources})
target_include_directories(vc_diffuse_winding PRIVATE diffusion)
target_link_libraries(vc_diffuse_winding
        opencv_videoio
        opencv_ximgproc
        vc_core
        vc_ui
        Boost::program_options
        OpenMP::OpenMP_CXX
        Ceres::ceres
)

add_executable(vc_tifxyz_mmap_prepare src/vc_tifxyz_mmap_prepare.cpp)
target_link_libraries(vc_tifxyz_mmap_prepare TIFF::TIFF)

add_executable(vc_create_segment_mask src/vc_create_segment_mask.cpp)
target_link_libraries(vc_create_segment_mask vc_core Boost::program_options)

# Grid-resolution helpers split into a static lib so the unit tests can
# link them without spawning the binary. No behavior change vs. having the
# code inline in vc_merge_tifxyz.cpp.
add_library(vc_merge_grid STATIC src/vc_merge_tifxyz_grid.cpp)
target_link_libraries(vc_merge_grid PUBLIC vc_core)
target_include_directories(vc_merge_grid PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(vc_merge_tifxyz src/vc_merge_tifxyz.cpp)
target_link_libraries(vc_merge_tifxyz vc_merge_grid Boost::program_options opencv_imgcodecs opencv_imgproc opencv_flann Eigen3::Eigen)

add_executable(vc_visualize src/vc_visualize.cpp)
target_link_libraries(vc_visualize vc_core Boost::program_options)

add_executable(vc_volpkg_convert src/vc_volpkg_convert.cpp)
target_link_libraries(vc_volpkg_convert vc_core)


# Apply affine + scale to OBJ, single TIFXYZ, or batch TIFXYZ subfolders
add_executable(vc_transform_geom src/vc_transform_geom.cpp)
target_link_libraries(vc_transform_geom vc_core Boost::program_options)


add_executable(vc_gen_normalgrids src/vc_gen_normalgrids.cpp diffusion/support.cpp)
target_link_libraries(vc_gen_normalgrids vc_core Boost::program_options opencv_core opencv_imgproc opencv_ximgproc OpenMP::OpenMP_CXX Ceres::ceres)
target_include_directories(vc_gen_normalgrids PRIVATE diffusion)

# Read / visualize NormalGridVolume
add_executable(vc_ngrids src/vc_ngrids.cpp)
target_link_libraries(vc_ngrids vc_core Boost::program_options opencv_core Ceres::ceres OpenMP::OpenMP_CXX)

add_executable(vc_tifxyz2rgb src/vc_tifxyz2rgb.cpp)
target_link_libraries(vc_tifxyz2rgb
    vc_core
    opencv_core
    opencv_imgcodecs
)

add_executable(vc_rgb2tifxyz src/vc_rgb2tifxyz.cpp)
target_link_libraries(vc_rgb2tifxyz
    vc_core
    opencv_core
    opencv_imgcodecs
    # nlohmann_json::nlohmann_json
)

add_executable(vc_thinning src/vc_thinning.cpp)
target_link_libraries(vc_thinning vc_core opencv_ximgproc)

add_executable(vc_tifxyz src/vc_tifxyz.cpp)
target_link_libraries(vc_tifxyz vc_core Boost::program_options opencv_core)

add_executable(vc_project_tifxyz src/vc_project_tifxyz.cpp)
target_link_libraries(vc_project_tifxyz vc_core Boost::program_options opencv_core OpenMP::OpenMP_CXX)

add_executable(vc_zarr_to_tiff src/vc_zarr_to_tiff.cpp)
target_link_libraries(vc_zarr_to_tiff vc_core Boost::program_options TIFF::TIFF)

add_executable(vc_zarr_recompress src/vc_zarr_recompress.cpp)
target_link_libraries(vc_zarr_recompress vc_core blosc_static)

# Precompiled headers for CLI apps (reuse vc_core's PCH)
if(VC_USE_PCH)
    foreach(_app
        vc_render_video
        vc_render_tifxyz
        vc_objrefine
        vc_grow_seg_from_seed
        vc_seg_add_overlap
        vc_grow_seg_from_segments
        vc_tifxyz2obj
        vc_obj2tifxyz
        vc_obj2tifxyz_legacy
        vc_flatten
        vc_tifxyz_inp_mask
        vc_tifxyz_trim
        vc_cut_windings
        vc_tifxyz_winding
        vc_tifxyz2zarr_sparse
        vc_add_ignore_label
        vc_calc_surface_metrics
        vc_tifxyz_gengt
        vc_diffuse_winding
        vc_tifxyz_mmap_prepare
        vc_create_segment_mask
        vc_visualize
        vc_transform_geom
        vc_gen_normalgrids
        vc_ngrids
        vc_tifxyz2rgb
        vc_rgb2tifxyz
        vc_thinning
        vc_tifxyz
        vc_project_tifxyz
        vc_zarr_to_tiff
    )
        target_precompile_headers(${_app} REUSE_FROM vc_core)
    endforeach()
endif()
