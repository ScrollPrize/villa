# Generated files directory (usually build/bin/generated)
set(gen_dir "${CMAKE_BINARY_DIR}/generated")

# Generated CMake project and version configuration files
set(version_config "${gen_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${gen_dir}/${PROJECT_NAME}Config.cmake")

# Configures VCConfigVersion.cmake
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${version_config}" COMPATIBILITY SameMajorVersion
)

# Configures VCConfig.cmake
set(config_in "${PROJECT_SOURCE_DIR}/cmake/VCConfig.cmake.in")
configure_package_config_file(
    "${config_in}"
    "${project_config}"
    INSTALL_DESTINATION "${config_install_dir}"
)

if(VC_INSTALL_LIBS)
# Install VC project config + version config
install(
    FILES "${project_config}" "${version_config}"
    DESTINATION "${config_install_dir}"
    COMPONENT Libraries
)

# Install exported targets file
install(
    EXPORT "${targets_export_name}"
    NAMESPACE "${namespace}"
    DESTINATION "${config_install_dir}"
    COMPONENT Libraries
)

# Install custom modules
install(
  FILES ${VC_CUSTOM_MODULES}
  DESTINATION "${config_install_dir}/Modules/"
  COMPONENT Libraries
)
endif()
