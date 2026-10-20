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
    Records.cpp
    Metadata.cpp
    SourcePaths.cpp
    Snapshots.cpp
    Queries.cpp
    Spatial.cpp
    sync/Synchronize.cpp
    sync/GeometryTransfer.cpp
    geometry/Geometry.cpp
    geometry/GeometryDelta.cpp
    geometry/GeometryReader.cpp
    relationships/Edges.cpp
    relationships/Traversal.cpp
    relationships/EdgeGeometry.cpp
)
target_include_directories(vc_objects PUBLIC "${PROJECT_SOURCE_DIR}/core/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
if(TARGET SQLite3::SQLite3)
    target_link_libraries(vc_objects PUBLIC SQLite3::SQLite3)
else()
    target_link_libraries(vc_objects PUBLIC SQLite::SQLite3)
endif()
target_link_libraries(vc_objects PRIVATE nlohmann_json::nlohmann_json OpenSSL::Crypto zstd::libzstd)
target_compile_features(vc_objects PUBLIC cxx_std_20)
set_target_properties(vc_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Native source geometry adapters depend on the database and VC's file readers.
add_library(vc_object_geometry STATIC geometry/GeometryIngest.cpp geometry/GeometryExport.cpp geometry/SourceMetadata.cpp imports/SourceRelationships.cpp)
target_include_directories(vc_object_geometry PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(vc_object_geometry PUBLIC vc_objects PRIVATE vc_core nlohmann_json::nlohmann_json)
target_compile_features(vc_object_geometry PUBLIC cxx_std_20)
set_target_properties(vc_object_geometry PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Detectors propose relationship metadata; they do not own a second database.
add_library(vc_object_links STATIC
    linking/LinkCandidates.cpp
    linking/PatchOverlap.cpp
    linking/FiberPatch.cpp
    linking/TrackTrack.cpp
    linking/StoredGeometry.cpp
    linking/LinePatch.cpp
    linking/TrackPatch.cpp
    linking/PointCollectionPatch.cpp
)
target_include_directories(vc_object_links PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(vc_object_links PUBLIC vc_objects PRIVATE vc_track_crossings vc_core vc_lasagna nlohmann_json::nlohmann_json)
target_compile_features(vc_object_links PUBLIC cxx_std_20)
set_target_properties(vc_object_links PROPERTIES POSITION_INDEPENDENT_CODE ON)
