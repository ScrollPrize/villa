# Load the full reader before the object store, whose standalone build only needs
# the catalog when relationship detection is disabled. Leave CUDA discovery and
# an explicit VC_READER_ENABLE_CUDA choice to vc_reader.
function(vc_find_reader)
    if(TARGET vc_reader::reader)
        return()
    endif()
    find_package(vc_reader CONFIG QUIET)
    if(TARGET vc_reader::reader)
        return()
    endif()
    if(TARGET vc_reader::catalog)
        message(FATAL_ERROR "The installed vc_reader lacks the reader API; rebuild it with VC_READER_BUILD_READER=ON")
    endif()
    set(VC_READER_SOURCE "${CMAKE_SOURCE_DIR}/../vc_reader" CACHE PATH
        "vc_reader source checkout")
    if(NOT EXISTS "${VC_READER_SOURCE}/CMakeLists.txt")
        message(FATAL_ERROR "Install vc_reader or set VC_READER_SOURCE to its checkout")
    endif()
    set(VC_READER_BUILD_PYTHON OFF)
    set(VC_READER_BUILD_READER ON)
    add_subdirectory("${VC_READER_SOURCE}" "${CMAKE_BINARY_DIR}/reader")
endfunction()
vc_find_reader()

# Prefer an installed package, otherwise build the sibling store and native formats.
# No installation, FetchContent, or dependency download is performed here.
function(vc_find_object_store)
    if(TARGET vc_object_store::vc_object_geometry)
        return()
    endif()
    find_package(vc_object_store CONFIG QUIET)
    if(NOT TARGET vc_object_store::vc_object_geometry)
        set(VC_OBJECT_STORE_SOURCE "${CMAKE_SOURCE_DIR}/../vc_object_store" CACHE PATH
            "vc_object_store source checkout")
        if(NOT EXISTS "${VC_OBJECT_STORE_SOURCE}/CMakeLists.txt")
            message(FATAL_ERROR "Install vc_object_store or set VC_OBJECT_STORE_SOURCE to its checkout")
        endif()
        set(VC_OBJECT_STORE_BUILD_ADAPTERS ON)
        set(VC_OBJECT_STORE_BUILD_LINKING OFF)
        set(VC_OBJECT_STORE_BUILD_TOOLS OFF)
        set(VC_OBJECT_STORE_BUILD_PYTHON OFF)
        set(VC_OBJECT_STORE_BUILD_TESTS ${VC_TESTING})
        # Each vc_* DLL would otherwise embed and re-export its own copy of the
        # static store, and PE has no symbol interposition to merge them.
        set(VC_OBJECT_STORE_SHARED ${WIN32})
        add_subdirectory("${VC_OBJECT_STORE_SOURCE}" "${CMAKE_BINARY_DIR}/object_store")
    endif()
endfunction()
vc_find_object_store()
