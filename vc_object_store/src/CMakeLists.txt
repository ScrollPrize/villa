# Shared owned geometry and operations; no SQLite, OpenCV or application types.
add_library(vc_geometry STATIC geometry/TypedGeometry.cpp)
target_include_directories(vc_geometry PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>" "$<INSTALL_INTERFACE:include>"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(vc_geometry PRIVATE nlohmann_json::nlohmann_json)
target_compile_features(vc_geometry PUBLIC cxx_std_20)

# Primary SQLite object database; no dependency on vc_core or Qt.
add_library(vc_objects STATIC
    internal/Anchors.cpp
    internal/Sqlite.cpp
    internal/RecordSql.cpp
    internal/EdgeSql.cpp
    internal/RecordJson.cpp
    internal/JsonFields.cpp
    Database.cpp
    Transactions.cpp
    History.cpp
    Lifecycle.cpp
    Records.cpp
    Metadata.cpp
    SourcePaths.cpp
    Snapshots.cpp
    Queries.cpp
    Spatial.cpp
    FiberRegion.cpp
    sync/Synchronize.cpp
    sync/GeometryTransfer.cpp
    geometry/GeometryViews.cpp
    imports/SourceIdentityIndex.cpp
    geometry/DependencyIndex.cpp
    geometry/SequenceMetadata.cpp
    geometry/SharedGeometry.cpp
    geometry/StructuralGeometry.cpp
    geometry/Geometry.cpp
    geometry/GeometryDelta.cpp
    geometry/GeometryReader.cpp
    relationships/Edges.cpp
    relationships/Traversal.cpp
    relationships/EdgeGeometry.cpp
)
target_include_directories(vc_objects PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>" "$<INSTALL_INTERFACE:include>"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
if(TARGET SQLite3::SQLite3)
    target_link_libraries(vc_objects PUBLIC SQLite3::SQLite3)
else()
    target_link_libraries(vc_objects PUBLIC SQLite::SQLite3)
endif()
target_link_libraries(vc_objects PUBLIC vc_geometry PRIVATE nlohmann_json::nlohmann_json OpenSSL::Crypto zstd::libzstd)
target_compile_features(vc_objects PUBLIC cxx_std_20)

if(NOT VC_OBJECT_STORE_BUILD_ADAPTERS)
    return()
endif()
find_package(TIFF REQUIRED)
set(VC_OBJECT_STORE_READER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../vc_reader" CACHE PATH "vc_reader source checkout")
if(NOT TARGET vc_reader::reader)
    find_package(vc_reader CONFIG QUIET)
endif()
if(NOT TARGET vc_reader::reader)
    set(VC_READER_BUILD_PYTHON OFF)
    set(VC_READER_BUILD_READER ON)
    # This adapter uses CPU normals; an explicit reader CUDA option is still honored.
    if(NOT DEFINED VC_READER_ENABLE_CUDA)
        set(VC_READER_ENABLE_CUDA OFF)
    endif()
    add_subdirectory("${VC_OBJECT_STORE_READER_SOURCE}" "${CMAKE_CURRENT_BINARY_DIR}/reader")
endif()
add_library(vc_track_crossings STATIC TrackCrossings.cpp)
target_include_directories(vc_track_crossings PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>" "$<INSTALL_INTERFACE:include>")
target_link_libraries(vc_track_crossings PRIVATE OpenMP::OpenMP_CXX)
target_compile_features(vc_track_crossings PUBLIC cxx_std_20)
add_library(vc_object_formats STATIC Tiff.cpp MemMap.cpp PackedTrackStore.cpp spatial/PatchGrid.cpp)
target_include_directories(vc_object_formats PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>" "$<INSTALL_INTERFACE:include>" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(vc_object_formats PUBLIC opencv_core TIFF::TIFF PRIVATE nlohmann_json::nlohmann_json)
target_compile_features(vc_object_formats PUBLIC cxx_std_20)
add_library(vc_object_geometry STATIC geometry/GeometryIngest.cpp geometry/GeometryExport.cpp geometry/SourceMetadata.cpp imports/SourceRelationships.cpp
    formats/GeometryInput.cpp formats/FiberInput.cpp formats/PatchInput.cpp
    formats/PointCollectionInput.cpp formats/TrackInput.cpp formats/GeometryOutput.cpp)
target_include_directories(vc_object_geometry PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(vc_object_geometry PUBLIC vc_objects PRIVATE vc_object_formats nlohmann_json::nlohmann_json)
add_library(vc_object_links STATIC
    CtSheetNormal.cpp spatial/PatchIndex.cpp
    linking/LinkCandidates.cpp linking/PatchOverlap.cpp linking/FiberPatch.cpp
    linking/TrackTrack.cpp linking/StoredGeometry.cpp linking/LinePatch.cpp
    linking/TrackPatch.cpp linking/PointCollectionPatch.cpp)
target_include_directories(vc_object_links PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(vc_object_links PUBLIC vc_objects PRIVATE vc_track_crossings vc_object_formats vc_reader::reader opencv_core nlohmann_json::nlohmann_json)
