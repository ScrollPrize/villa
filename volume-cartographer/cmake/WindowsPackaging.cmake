# Windows bundling + installer.
#
# Produces a self-contained install tree (bin/ holds VC3D.exe, the vc_* CLI
# tools, every dependency DLL resolved from the MinGW prefix, and the Qt
# plugin folders), then wraps it in an NSIS installer + ZIP via CPack:
#
#   cmake --build <build>            # as usual
#   cpack --config <build>/CPackConfig.cmake -G NSIS
#
# Expects an MSYS2 MinGW environment (MSYSTEM_PREFIX set, e.g. /ucrt64).

if(NOT WIN32)
    return()
endif()

# ---- Dependency DLL closure --------------------------------------------------
# Targets are registered into the vc_runtime_deps set by the install() walk in
# the top-level CMakeLists. Resolve their transitive DLLs out of the MinGW
# prefix; Windows system DLLs stay excluded.
set(_vc_mingw_bin "$ENV{MSYSTEM_PREFIX}/bin")
if(NOT EXISTS "${_vc_mingw_bin}")
    message(WARNING "MSYSTEM_PREFIX is not set — dependency DLLs will not be bundled")
endif()
install(RUNTIME_DEPENDENCY_SET vc_runtime_deps
    COMPONENT vc_runtime
    PRE_EXCLUDE_REGEXES "^api-ms-" "^ext-ms-" "^hvsifiletrust" "^pdmutilities"
    POST_EXCLUDE_REGEXES ".*[/\\\\][Ww]indows[/\\\\].*" ".*[Ss]ystem32.*"
    DIRECTORIES "${_vc_mingw_bin}"
    RUNTIME DESTINATION bin
)

# ---- Qt runtime (plugins etc.) -----------------------------------------------
# windeployqt drops the platform/style/imageformat/tls plugins (and any Qt
# DLLs the dependency walk above may have classified oddly) next to VC3D.exe,
# which is where Qt looks for the "platforms/" folder at startup.
find_program(VC_WINDEPLOYQT
    NAMES windeployqt-qt6 windeployqt6 windeployqt
    HINTS "$ENV{MSYSTEM_PREFIX}/bin" "$ENV{MSYSTEM_PREFIX}/share/qt6/bin")
if(VC_WINDEPLOYQT)
    install(CODE "
        message(STATUS \"Running windeployqt: ${VC_WINDEPLOYQT}\")
        execute_process(
            COMMAND \"${VC_WINDEPLOYQT}\"
                --release --no-translations --no-compiler-runtime
                --dir \"\${CMAKE_INSTALL_PREFIX}/bin\"
                \"\${CMAKE_INSTALL_PREFIX}/bin/VC3D.exe\"
            RESULT_VARIABLE _wdq_rc)
        if(NOT _wdq_rc EQUAL 0)
            message(FATAL_ERROR \"windeployqt failed (rc=\${_wdq_rc})\")
        endif()
    " COMPONENT vc_runtime)
else()
    # Fallback: copy the plugin folders VC3D needs straight from the prefix.
    message(WARNING "windeployqt not found — copying Qt plugin dirs directly")
    foreach(_plugin_dir platforms styles imageformats iconengines tls)
        set(_src "$ENV{MSYSTEM_PREFIX}/share/qt6/plugins/${_plugin_dir}")
        if(EXISTS "${_src}")
            install(DIRECTORY "${_src}"
                DESTINATION bin
                COMPONENT vc_runtime
                FILES_MATCHING PATTERN "*.dll")
        endif()
    endforeach()
endif()

# ---- Installer ---------------------------------------------------------------
set(CPACK_PACKAGE_NAME "VC3D")
set(CPACK_PACKAGE_VENDOR "Vesuvius Challenge")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "VC3D — Volume Cartographer 3D")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/ScrollPrize/villa")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "VC3D")
set(CPACK_PACKAGE_FILE_NAME "VC3D-${VC_VERSION_STRING}-win64")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_GENERATOR "NSIS;ZIP")

# Single vc_runtime component, packaged monolithically.
set(CPACK_COMPONENTS_ALL vc_runtime)
set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)

set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
set(CPACK_NSIS_MODIFY_PATH OFF)
set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\VC3D.exe")
# Installer/uninstaller executable icon.
set(CPACK_NSIS_MUI_ICON "${CMAKE_SOURCE_DIR}/apps/VC3D/logo.ico")
set(CPACK_NSIS_MUI_UNIICON "${CMAKE_SOURCE_DIR}/apps/VC3D/logo.ico")
set(CPACK_NSIS_MUI_FINISHPAGE_RUN "VC3D.exe")
set(CPACK_NSIS_URL_INFO_ABOUT "https://github.com/ScrollPrize/villa")
# Start-menu + optional desktop shortcut to bin/VC3D.exe.
set(CPACK_PACKAGE_EXECUTABLES "VC3D" "VC3D")
set(CPACK_CREATE_DESKTOP_LINKS "VC3D")

include(CPack)
