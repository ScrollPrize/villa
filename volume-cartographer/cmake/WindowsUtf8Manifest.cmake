# Embed cmake/windows/utf8.manifest (activeCodePage UTF-8) into every
# executable target under a directory, recursively. MinGW only: MSVC writes
# its own manifest, and other platforms have no use for one (villa#1979).

function(vc_embed_utf8_manifest directory)
    get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS targets)
        get_target_property(type ${target} TYPE)
        if(type STREQUAL "EXECUTABLE")
            target_sources(${target} PRIVATE "${VC_UTF8_MANIFEST_RC}")
            # windres writes no depfile; source properties are per directory.
            set_property(SOURCE "${VC_UTF8_MANIFEST_RC}" TARGET_DIRECTORY ${target}
                         PROPERTY OBJECT_DEPENDS "${VC_UTF8_MANIFEST}")
        endif()
    endforeach()
    get_property(subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(subdirectory IN LISTS subdirectories)
        vc_embed_utf8_manifest("${subdirectory}")
    endforeach()
endfunction()

if(MINGW)
    # The .rc names the manifest by its bare file name, next to it in the
    # build tree: windres opens files through the ANSI code page, so an
    # absolute path into a source tree under a non-ASCII directory would not
    # open. configure_file re-runs CMake when the manifest changes.
    set(VC_UTF8_MANIFEST "${CMAKE_BINARY_DIR}/vc_utf8.manifest")
    configure_file("${CMAKE_SOURCE_DIR}/cmake/windows/utf8.manifest"
                   "${VC_UTF8_MANIFEST}" COPYONLY)
    set(VC_UTF8_MANIFEST_RC "${CMAKE_BINARY_DIR}/vc_utf8_manifest.rc")
    configure_file("${CMAKE_SOURCE_DIR}/cmake/windows/utf8_manifest.rc.in"
                   "${VC_UTF8_MANIFEST_RC}" COPYONLY)
    vc_embed_utf8_manifest("${CMAKE_SOURCE_DIR}")
endif()
