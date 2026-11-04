# macOS bundling + DMG.
#
# Produces a self-contained VC3D.app (Contents/MacOS holds VC3D plus the vc_*
# CLI tools, Contents/Frameworks the full dylib closure resolved out of the
# Homebrew prefix, Contents/PlugIns the Qt plugins), then wraps it in a
# drag-and-drop DMG via CPack (CI derives its ZIP from that deployed DMG):
#
#   ninja -C <build>                # as usual
#   cpack --config <build>/CPackConfig.cmake
#
# Everything is ad-hoc signed (codesign -s -): arm64 macOS refuses to load
# binaries whose linker signature was invalidated by install-name rewriting.
# There is no Developer ID / notarization, so downloads stay quarantined —
# first launch of the app is right-click -> Open.

if(NOT APPLE OR NOT VC_BUILD_APPS)
    return()
endif()

set(_vc_app_dir "VC3D.app")

# ---- App bundle skeleton -------------------------------------------------
# VC3D builds as a plain executable everywhere; the install walk in the
# top-level CMakeLists already drops it and the CLI tools into
# VC3D.app/Contents/MacOS. Info.plist + icon make that tree a real
# double-clickable bundle (and give macdeployqt its entry point).
# The bundle contains Homebrew bottles built for the host OS, whose minimum
# version may be newer than our own compiler deployment target. Advertise the
# build host's major version so Finder does not claim unsupported compatibility.
execute_process(COMMAND sw_vers -productVersion
    OUTPUT_VARIABLE _vc_host_macos OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REGEX MATCH "^[0-9]+" VC_MACOS_MIN_VERSION "${_vc_host_macos}")
if(NOT VC_MACOS_MIN_VERSION)
    message(FATAL_ERROR "Could not determine the macOS host version")
endif()
configure_file("${CMAKE_SOURCE_DIR}/cmake/MacOSBundleInfo.plist.in"
    "${CMAKE_BINARY_DIR}/Info.plist" @ONLY)
install(FILES "${CMAKE_BINARY_DIR}/Info.plist"
    DESTINATION "${_vc_app_dir}/Contents" COMPONENT vc_runtime)
install(FILES "${CMAKE_SOURCE_DIR}/apps/VC3D/logo.icns"
    DESTINATION "${_vc_app_dir}/Contents/Resources" COMPONENT vc_runtime)

# ---- Dependency dylib closure + Qt runtime --------------------------------
# macdeployqt does both jobs windeployqt + RUNTIME_DEPENDENCY_SET split on
# Windows: it walks every executable's dependency closure, copies the dylibs
# and Qt frameworks into Contents/Frameworks, deploys the Qt plugins, and
# rewrites install names to @executable_path-relative paths.
find_program(VC_MACDEPLOYQT NAMES macdeployqt macdeployqt6
    HINTS "$ENV{HOMEBREW_PREFIX}/opt/qtbase/bin"
          "$ENV{HOMEBREW_PREFIX}/opt/qt/bin" "$ENV{HOMEBREW_PREFIX}/bin")
find_program(VC_QMAKE NAMES qmake qmake6
    HINTS "$ENV{HOMEBREW_PREFIX}/opt/qtbase/bin"
          "$ENV{HOMEBREW_PREFIX}/opt/qt/bin" "$ENV{HOMEBREW_PREFIX}/bin")
set(_vc_qt_plugin_dir "")
if(VC_QMAKE)
    execute_process(COMMAND "${VC_QMAKE}" -query QT_INSTALL_PLUGINS
        OUTPUT_VARIABLE _vc_qt_plugin_dir OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(VC_MACDEPLOYQT)
    find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter)
    # The helper is run on the staged bundle, not installed into the app.
    # Use configure_file so paths with spaces survive the install-time handoff.
    configure_file("${CMAKE_SOURCE_DIR}/cmake/DeployMacOS.cmake.in"
        "${CMAKE_BINARY_DIR}/DeployMacOS.cmake" @ONLY)
    install(SCRIPT "${CMAKE_BINARY_DIR}/DeployMacOS.cmake" COMPONENT vc_runtime)
else()
    message(WARNING "macdeployqt not found — the installed VC3D.app will not be self-contained")
endif()

# ---- DMG (CI derives ZIP without redeploying) -------------------------------
set(CPACK_PACKAGE_NAME "VC3D")
set(CPACK_PACKAGE_VENDOR "Vesuvius Challenge")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "VC3D — Volume Cartographer 3D")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/ScrollPrize/villa")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "VC3D-${VC_VERSION_STRING}-macos-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
# DragNDrop puts VC3D.app + an /Applications symlink in the DMG.
# Deploy the app once for the DMG. CI derives the ZIP from that finished DMG;
# listing both generators here would rerun every install-time deployment hook.
set(CPACK_GENERATOR "DragNDrop")
set(CPACK_DMG_VOLUME_NAME "VC3D")

# Single vc_runtime component, packaged monolithically.
set(CPACK_COMPONENTS_ALL vc_runtime)
set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)

include(CPack)
