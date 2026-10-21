include(CMakePackageConfigHelpers)
set(_vc_native_targets vc_reader_headers vc_reader_transforms)
if(VC_READER_ENABLE_CUDA)
    list(APPEND _vc_native_targets vc_reader_transform_cuda)
endif()
if(VC_READER_BUILD_READER)
    list(APPEND _vc_native_targets vc_reader_native)
    if(VC_READER_ENABLE_CUDA)
        list(APPEND _vc_native_targets vc_reader_reader_cuda)
    endif()
endif()
install(TARGETS ${_vc_native_targets} EXPORT vc_readerTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT vc_readerTargets NAMESPACE vc_reader:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vc_reader)
configure_package_config_file(cmake/vc_readerConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/vc_readerConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vc_reader)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vc_readerConfig.cmake" DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vc_reader)
