############
# GUI Apps #
############
if(VC_BUILD_GUI)
add_subdirectory(VC3D)
endif()

#################
# CMD Line Apps #
#################

add_executable(vc_render_video src/vc_render_video.cpp)
target_link_libraries(vc_render_video z5 opencv_core vc_core)

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
)

# Apply affine + scale to OBJ or TIFXYZ
add_executable(vc_transform_geom src/vc_transform_geom.cpp)
target_link_libraries(vc_transform_geom vc_core Boost::program_options)

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
        vc_tifxyz2rgb
        vc_rgb2tifxyz
        vc_tifxyz_winding
        vc_tifxyz_gengt
        vc_fill_quadmesh
        vc_calc_surface_metrics
        vc_ced2d
        vc_transform_geom
        vc_telea_inpaint
        ${extra_apps}
    RUNTIME DESTINATION bin
    COMPONENT Programs
)
endif()
