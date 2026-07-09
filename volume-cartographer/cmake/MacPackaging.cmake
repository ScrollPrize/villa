# macOS bundling + DMG.
#
# Produces a self-contained VC3D.app (Contents/MacOS holds VC3D plus the vc_*
# CLI tools, Contents/Frameworks the full dylib closure resolved out of the
# Homebrew prefix, Contents/PlugIns the Qt plugins), then wraps it in a
# drag-and-drop DMG + ZIP via CPack:
#
#   cmake --build <build>            # as usual
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
if(CMAKE_OSX_DEPLOYMENT_TARGET)
    set(VC_MACOS_MIN_VERSION "${CMAKE_OSX_DEPLOYMENT_TARGET}")
else()
    # No explicit deployment target: the binaries (and the Homebrew bottles
    # they link) target the build host's macOS release.
    execute_process(COMMAND sw_vers -productVersion
        OUTPUT_VARIABLE _vc_host_macos OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "^[0-9]+" VC_MACOS_MIN_VERSION "${_vc_host_macos}")
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
    HINTS "$ENV{HOMEBREW_PREFIX}/opt/qt/bin" "$ENV{HOMEBREW_PREFIX}/bin")
find_program(VC_QMAKE NAMES qmake qmake6
    HINTS "$ENV{HOMEBREW_PREFIX}/opt/qt/bin" "$ENV{HOMEBREW_PREFIX}/bin")
set(_vc_qt_plugin_dir "")
if(VC_QMAKE)
    execute_process(COMMAND "${VC_QMAKE}" -query QT_INSTALL_PLUGINS
        OUTPUT_VARIABLE _vc_qt_plugin_dir OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(VC_MACDEPLOYQT)
    # Configure-time values the install-time code below needs.
    install(CODE "
        set(_vc_macdeployqt \"${VC_MACDEPLOYQT}\")
        set(_vc_qt_plugin_dir \"${_vc_qt_plugin_dir}\")
    " COMPONENT vc_runtime)

    install(CODE [==[
        set(_app "${CMAKE_INSTALL_PREFIX}/VC3D.app")

        # Every CLI tool rides along via -executable so macdeployqt pulls its
        # dependency closure and rewrites its install names too. (VC3D itself
        # is the bundle's main executable — covered implicitly.)
        file(GLOB _clis "${_app}/Contents/MacOS/*")
        set(_exe_args "")
        foreach(_cli IN LISTS _clis)
            if(NOT _cli MATCHES "/VC3D$")
                list(APPEND _exe_args "-executable=${_cli}")
            endif()
        endforeach()
        message(STATUS "Running macdeployqt: ${_vc_macdeployqt}")
        execute_process(
            COMMAND "${_vc_macdeployqt}" "${_app}" -always-overwrite ${_exe_args}
            RESULT_VARIABLE _mdq_rc)
        if(NOT _mdq_rc EQUAL 0)
            message(FATAL_ERROR "macdeployqt failed (rc=${_mdq_rc})")
        endif()

        # macdeployqt only deploys the cocoa platform plugin. Headless runs
        # (QT_QPA_PLATFORM=offscreen — used by the CI smoke test) need the
        # offscreen plugin as well: copy it and point its Homebrew-absolute
        # dependency paths at the frameworks macdeployqt just bundled.
        set(_offscreen "${_vc_qt_plugin_dir}/platforms/libqoffscreen.dylib")
        if(EXISTS "${_offscreen}")
            file(COPY "${_offscreen}"
                 DESTINATION "${_app}/Contents/PlugIns/platforms")
            set(_plugin "${_app}/Contents/PlugIns/platforms/libqoffscreen.dylib")
            execute_process(COMMAND install_name_tool -id
                "@rpath/libqoffscreen.dylib" "${_plugin}")
            execute_process(COMMAND otool -L "${_plugin}"
                OUTPUT_VARIABLE _deps_out)
            string(REPLACE "\n" ";" _dep_lines "${_deps_out}")
            foreach(_line IN LISTS _dep_lines)
                if(NOT _line MATCHES "^[ \t]+(/[^ ]+)")
                    continue()
                endif()
                set(_dep "${CMAKE_MATCH_1}")
                if(_dep MATCHES "^(/usr/lib|/System)")
                    continue()
                endif()
                if(_dep MATCHES "/([^/]+\\.framework/.+)$")
                    set(_new "@executable_path/../Frameworks/${CMAKE_MATCH_1}")
                else()
                    get_filename_component(_base "${_dep}" NAME)
                    set(_new "@executable_path/../Frameworks/${_base}")
                endif()
                execute_process(COMMAND install_name_tool
                    -change "${_dep}" "${_new}" "${_plugin}")
            endforeach()
        else()
            message(WARNING "offscreen Qt plugin not found at ${_offscreen}")
        endif()

        # Ad-hoc re-sign every Mach-O: install-name rewriting invalidated the
        # signatures, and arm64 macOS refuses to load unsigned binaries.
        # Mach-O detection by magic bytes (thin LE / fat BE / fat LE) — the
        # bundle also holds plists, qt.conf, icons etc. that must be skipped.
        file(GLOB_RECURSE _bundle_files LIST_DIRECTORIES false
            "${_app}/Contents/Frameworks/*"
            "${_app}/Contents/PlugIns/*"
            "${_app}/Contents/MacOS/*")
        foreach(_f IN LISTS _bundle_files)
            if(IS_SYMLINK "${_f}")
                continue()
            endif()
            file(READ "${_f}" _magic LIMIT 4 HEX)
            if(NOT _magic MATCHES "^(cffaedfe|cafebabe|bebafeca)")
                continue()
            endif()
            execute_process(COMMAND codesign --force --sign - "${_f}"
                RESULT_VARIABLE _cs_rc ERROR_VARIABLE _cs_err)
            if(NOT _cs_rc EQUAL 0)
                message(FATAL_ERROR "codesign failed for ${_f}: ${_cs_err}")
            endif()
        endforeach()
        # Bundle-level signature last (re-signs the main executable and seals
        # the bundle) so the per-file signatures above stay valid beneath it.
        execute_process(COMMAND codesign --force --sign - "${_app}"
            RESULT_VARIABLE _cs_rc ERROR_VARIABLE _cs_err)
        if(NOT _cs_rc EQUAL 0)
            message(FATAL_ERROR "codesign failed for ${_app}: ${_cs_err}")
        endif()
    ]==] COMPONENT vc_runtime)
else()
    message(WARNING "macdeployqt not found — the installed VC3D.app will not be self-contained")
endif()

# ---- DMG + ZIP -------------------------------------------------------------
set(CPACK_PACKAGE_NAME "VC3D")
set(CPACK_PACKAGE_VENDOR "Vesuvius Challenge")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "VC3D — Volume Cartographer 3D")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/ScrollPrize/villa")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "VC3D-${VC_VERSION_STRING}-macos-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
# DragNDrop puts VC3D.app + an /Applications symlink in the DMG.
set(CPACK_GENERATOR "DragNDrop;ZIP")
set(CPACK_DMG_VOLUME_NAME "VC3D")

# Single vc_runtime component, packaged monolithically.
set(CPACK_COMPONENTS_ALL vc_runtime)
set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)

include(CPack)
