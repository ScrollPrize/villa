############
# GUI Apps #
############
add_subdirectory(VC3D)

#################
# CMD Line Apps #
#################

add_executable(vc_render_video src/vc_render_video.cpp)
target_link_libraries(vc_render_video z5 opencv_core vc_core opencv_videoio)

add_executable(vc_render_tifxyz src/vc_render_tifxyz.cpp)
target_link_libraries(vc_render_tifxyz vc_core Boost::program_options tiff)

add_executable(vc_objrefine src/ObjAlphaCompRefinement.cpp)
target_link_libraries(vc_objrefine vc_core)

add_executable(vc_grow_seg_from_seed src/vc_grow_seg_from_seed.cpp)
target_link_libraries(vc_grow_seg_from_seed vc_core vc_ui vc_tracer Boost::program_options)

add_executable(vc_seg_add_overlap src/vc_seg_add_overlap.cpp)
target_link_libraries(vc_seg_add_overlap vc_core)

add_executable(vc_grow_seg_from_segments src/vc_grow_seg_from_segments.cpp)
target_link_libraries(vc_grow_seg_from_segments vc_core vc_tracer)

add_executable(vc_tifxyz2obj src/vc_tifxyz2obj.cpp)
target_link_libraries(vc_tifxyz2obj vc_core)

add_executable(vc_obj2tifxyz src/vc_obj2tifxyz.cpp)
target_link_libraries(vc_obj2tifxyz vc_core)

add_executable(vc_obj2tifxyz_legacy src/vc_obj2tifxyz_legacy.cpp)
target_link_libraries(vc_obj2tifxyz_legacy vc_core)

add_executable(vc_flatten src/vc_flatten.cpp)
target_link_libraries(vc_flatten vc_core Boost::program_options)

add_executable(vc_tifxyz_inp_mask src/vc_tifxyz_inp_mask.cpp)
target_link_libraries(vc_tifxyz_inp_mask vc_core)

add_executable(vc_tifxyz_winding src/vc_tifxyz_winding.cpp)
target_link_libraries(vc_tifxyz_winding vc_core)

add_executable(vc_fill_quadmesh src/vc_fill_quadmesh.cpp)
target_link_libraries(vc_fill_quadmesh vc_core vc_tracer)

add_executable(vc_tiffxyz_upscale_grounding src/vc_tiffxyz_upscale_grounding.cpp)
target_link_libraries(vc_tiffxyz_upscale_grounding vc_core vc_tracer)

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
        z5
        opencv_videoio
        opencv_ximgproc
        vc_core
        vc_ui
        Boost::program_options
        OpenMP::OpenMP_CXX
        Ceres::ceres
)

add_executable(vc_render_bench src/ZarrExtract.cpp)
target_link_libraries(vc_render_bench vc_core)

add_executable(vc_create_segment_mask src/vc_create_segment_mask.cpp)
target_link_libraries(vc_create_segment_mask vc_core Boost::program_options)

add_executable(vc_surface_diff src/vc_surface_diff.cpp)
target_link_libraries(vc_surface_diff vc_core Boost::program_options)

add_executable(vc_visualize src/vc_visualize.cpp)
target_link_libraries(vc_visualize vc_core Boost::program_options)


add_executable(vc_ced2d src/vc_ced2d.cpp)
target_link_libraries(vc_ced2d
    z5
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
    Boost::program_options
    OpenMP::OpenMP_CXX
    nlohmann_json::nlohmann_json
)

# Apply affine + scale to OBJ or TIFXYZ
add_executable(vc_transform_geom src/vc_transform_geom.cpp)
target_link_libraries(vc_transform_geom vc_core Boost::program_options)

add_executable(vc_gen_normalgrids src/vc_gen_normalgrids.cpp diffusion/support.cpp)
target_link_libraries(vc_gen_normalgrids vc_core z5 Boost::program_options opencv_core opencv_imgproc opencv_ximgproc OpenMP::OpenMP_CXX Ceres::ceres)
target_include_directories(vc_gen_normalgrids PRIVATE diffusion)

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

add_executable(vc_telea_inpaint src/vc_telea_inpaint.cpp)
target_link_libraries(vc_telea_inpaint
    vc_core
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
    opencv_photo
)

add_executable(vc_thinning src/vc_thinning.cpp)
target_link_libraries(vc_thinning vc_core opencv_ximgproc)

add_executable(vc_tifxyz src/vc_tifxyz.cpp)
target_link_libraries(vc_tifxyz vc_core Boost::program_options opencv_core)

add_executable(test_h264_zarr src/test_h264_zarr.cpp)
target_link_libraries(test_h264_zarr z5 nlohmann_json::nlohmann_json blosc2_static)

add_executable(test_blosc2_h264 src/test_blosc2_h264.cpp)
target_link_libraries(test_blosc2_h264 blosc2_static)

# Install targets
if(VC_INSTALL_APPS)
install(
    TARGETS
        vc_render_video
        vc_render_tifxyz
        vc_objrefine
        vc_grow_seg_from_seed
        vc_grow_seg_from_segments
        vc_seg_add_overlap
        vc_tifxyz2obj
        vc_obj2tifxyz
        vc_obj2tifxyz_legacy
        vc_flatten
        vc_tifxyz2rgb
        vc_rgb2tifxyz
        vc_tifxyz_winding
        vc_tifxyz_gengt
        vc_fill_quadmesh
        vc_calc_surface_metrics
        vc_diffuse_winding
        vc_ced2d
        vc_transform_geom
        vc_gen_normalgrids
        vc_telea_inpaint
        vc_thinning
        vc_tifxyz
        ${extra_apps}
    RUNTIME DESTINATION bin
    COMPONENT Programs
)
endif()
