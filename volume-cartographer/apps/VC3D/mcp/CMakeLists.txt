include(FetchContent)

set(VC_MCP_NNUNET_TEST_PYTHON "" CACHE FILEPATH "Python executable for the optional nnU-Net MCP integration test")
set(VC_MCP_NNUNET_TEST_MODEL "" CACHE PATH "Dataset 058 model directory for the optional nnU-Net MCP integration test")

# FastMCPP is beta, so pin an audited commit rather than following a branch or tag.
set(FASTMCPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FASTMCPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FASTMCPP_ENABLE_POST_STREAMING OFF CACHE BOOL "" FORCE)
set(FASTMCPP_FETCH_CURL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    fastmcpp
    GIT_REPOSITORY https://github.com/0xeb/fastmcpp.git
    GIT_TAG 9a3ee7125b1db9dc85a71f60661beb40ea1e7993
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(fastmcpp)

add_library(vc_mcp_phase1
    CpuDiscovery.cpp
    CpuDiscovery.hpp
    DiscoveryTools.cpp
    DiscoveryTools.hpp
    GrowRequest.cpp
    InspectionTools.cpp
    InspectionTools.hpp
    JobStore.cpp
    JobStore.hpp
    McpApplication.cpp
    McpApplication.hpp
    PredictionService.cpp
    PredictionService.hpp
    VolumeCartographer.cpp
    VolumeCartographer.hpp
)
target_include_directories(vc_mcp_phase1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(vc_mcp_phase1 PUBLIC cxx_std_23)
target_link_libraries(vc_mcp_phase1 PUBLIC
    fastmcpp_core
    Qt6::Core
    Qt6::Gui
    CURL::libcurl
    Blosc::blosc
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
)
qt_add_resources(vc_mcp_phase1 vc_mcp_ui
    PREFIX "/vc_mcp_ui"
    BASE "${CMAKE_CURRENT_SOURCE_DIR}/ui/dist"
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/ui/dist/inspector.html"
)
set_target_properties(vc_mcp_phase1 PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)

add_executable(vc_mcp_server main.cpp)
target_compile_features(vc_mcp_server PRIVATE cxx_std_23)
target_link_libraries(vc_mcp_server PRIVATE vc_mcp_phase1)
target_compile_definitions(vc_mcp_server PRIVATE
    VC_MCP_VC_VERSION="${VC_VERSION_STRING}"
    VC_MCP_VC_COMMIT="${GIT_SHA1}"
    VC_MCP_CONTAINER_DIGEST="${VC_MCP_CONTAINER_DIGEST}"
)
set_target_properties(vc_mcp_server PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)

if(VC_TESTING)
    # This subdirectory is configured before the repository-level enable_testing().
    enable_testing()
    function(vc_mcp_enable_assertions target)
        target_compile_options(${target} PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
    endfunction()

    add_executable(vc_mcp_job_store_test tests/JobStoreTest.cpp)
    target_compile_features(vc_mcp_job_store_test PRIVATE cxx_std_23)
    target_link_libraries(vc_mcp_job_store_test PRIVATE vc_mcp_phase1)
    vc_mcp_enable_assertions(vc_mcp_job_store_test)
    set_target_properties(vc_mcp_job_store_test PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)
    add_test(NAME vc_mcp_job_store COMMAND vc_mcp_job_store_test)

    add_executable(vc_mcp_fake_grow tests/FakeVcGrow.cpp)
    target_compile_features(vc_mcp_fake_grow PRIVATE cxx_std_23)
    target_link_libraries(vc_mcp_fake_grow PRIVATE nlohmann_json::nlohmann_json)
    set_target_properties(vc_mcp_fake_grow PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)

    add_executable(vc_mcp_local_worker_test tests/LocalWorkerTest.cpp)
    target_compile_features(vc_mcp_local_worker_test PRIVATE cxx_std_23)
    target_link_libraries(vc_mcp_local_worker_test PRIVATE vc_mcp_phase1)
    vc_mcp_enable_assertions(vc_mcp_local_worker_test)
    target_compile_definitions(vc_mcp_local_worker_test PRIVATE
        VC_MCP_FAKE_GROW_PATH="$<TARGET_FILE:vc_mcp_fake_grow>")
    add_dependencies(vc_mcp_local_worker_test vc_mcp_fake_grow)
    set_target_properties(vc_mcp_local_worker_test PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)
    add_test(NAME vc_mcp_local_worker COMMAND vc_mcp_local_worker_test)

    add_executable(vc_mcp_cpu_discovery_test tests/CpuDiscoveryTest.cpp)
    target_compile_features(vc_mcp_cpu_discovery_test PRIVATE cxx_std_23)
    target_link_libraries(vc_mcp_cpu_discovery_test PRIVATE vc_mcp_phase1)
    vc_mcp_enable_assertions(vc_mcp_cpu_discovery_test)
    set_target_properties(vc_mcp_cpu_discovery_test PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)
    add_test(NAME vc_mcp_cpu_discovery COMMAND vc_mcp_cpu_discovery_test)

    if(EXISTS "${VC_MCP_NNUNET_TEST_PYTHON}")
        add_test(NAME vc_mcp_volume_stager
            COMMAND "${VC_MCP_NNUNET_TEST_PYTHON}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/VolumeStagerTest.py"
                    --stager "${CMAKE_CURRENT_SOURCE_DIR}/volume_stager.py")
        set_tests_properties(vc_mcp_volume_stager PROPERTIES TIMEOUT 30)

        set(_vc_ink_model_test_args)
        if(EXISTS "${CMAKE_SOURCE_DIR}/../ink-detection/metadata.json")
            list(APPEND _vc_ink_model_test_args --ink-model-root "${CMAKE_SOURCE_DIR}/../ink-detection")
        endif()
        add_test(NAME vc_mcp_surface_bundle_adapter
            COMMAND "${VC_MCP_NNUNET_TEST_PYTHON}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/SurfaceBundleAdapterTest.py"
                    --adapter "${CMAKE_CURRENT_SOURCE_DIR}/surface_bundle_adapter.py"
                    --ink-model-contracts "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ink_model_surface_volume_contracts.json"
                    ${_vc_ink_model_test_args})
        set_tests_properties(vc_mcp_surface_bundle_adapter PROPERTIES TIMEOUT 30)

        add_test(NAME vc_mcp_structural_evidence_adapter
            COMMAND "${VC_MCP_NNUNET_TEST_PYTHON}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/StructuralEvidenceAdapterTest.py"
                    --adapter "${CMAKE_CURRENT_SOURCE_DIR}/structural_evidence_adapter.py")
        set_tests_properties(vc_mcp_structural_evidence_adapter PROPERTIES TIMEOUT 60)

        add_test(NAME vc_mcp_evidence_fusion_adapter
            COMMAND "${VC_MCP_NNUNET_TEST_PYTHON}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/EvidenceFusionAdapterTest.py"
                    --adapter "${CMAKE_CURRENT_SOURCE_DIR}/evidence_fusion_adapter.py")
        set_tests_properties(vc_mcp_evidence_fusion_adapter PROPERTIES TIMEOUT 60)

        add_test(NAME vc_mcp_review_adapter
            COMMAND "${VC_MCP_NNUNET_TEST_PYTHON}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/ReviewAdapterTest.py"
                    --adapter "${CMAKE_CURRENT_SOURCE_DIR}/review_adapter.py")
        set_tests_properties(vc_mcp_review_adapter PROPERTIES TIMEOUT 60)

        add_test(NAME vc_mcp_evidence_pipeline_e2e
            COMMAND "${VC_MCP_NNUNET_TEST_PYTHON}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/EndToEndEvidencePipelineTest.py"
                    --mcp-dir "${CMAKE_CURRENT_SOURCE_DIR}")
        set_tests_properties(vc_mcp_evidence_pipeline_e2e PROPERTIES TIMEOUT 120 LABELS "mcp;integration")

        add_test(NAME vc_mcp_evidence_boundaries
            COMMAND "${VC_MCP_NNUNET_TEST_PYTHON}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/McpEvidenceBoundaryTest.py"
                    --mcp-dir "${CMAKE_CURRENT_SOURCE_DIR}")
        set_tests_properties(vc_mcp_evidence_boundaries PROPERTIES TIMEOUT 60 LABELS "mcp;negative")

        add_executable(vc_mcp_registered_surface_test tests/RegisteredSurfaceMcpTest.cpp)
        target_compile_features(vc_mcp_registered_surface_test PRIVATE cxx_std_23)
        target_link_libraries(vc_mcp_registered_surface_test PRIVATE vc_mcp_phase1)
        target_compile_definitions(vc_mcp_registered_surface_test PRIVATE
            VC_MCP_FAKE_GROW_PATH="$<TARGET_FILE:vc_mcp_fake_grow>"
            VC_MCP_ANALYSIS_PYTHON="${VC_MCP_NNUNET_TEST_PYTHON}"
            VC_MCP_SURFACE_BUNDLE_ADAPTER="${CMAKE_CURRENT_SOURCE_DIR}/surface_bundle_adapter.py"
            VC_MCP_STRUCTURAL_EVIDENCE_ADAPTER="${CMAKE_CURRENT_SOURCE_DIR}/structural_evidence_adapter.py"
            VC_MCP_EVIDENCE_FUSION_ADAPTER="${CMAKE_CURRENT_SOURCE_DIR}/evidence_fusion_adapter.py"
            VC_MCP_REVIEW_ADAPTER="${CMAKE_CURRENT_SOURCE_DIR}/review_adapter.py"
            VC_MCP_VOLUME_STAGER="${CMAKE_CURRENT_SOURCE_DIR}/volume_stager.py")
        add_dependencies(vc_mcp_registered_surface_test vc_mcp_fake_grow)
        vc_mcp_enable_assertions(vc_mcp_registered_surface_test)
        set_target_properties(vc_mcp_registered_surface_test PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)
        add_test(NAME vc_mcp_registered_surface COMMAND vc_mcp_registered_surface_test)
        set_tests_properties(vc_mcp_registered_surface PROPERTIES TIMEOUT 60)
    endif()

    if(EXISTS "${VC_MCP_NNUNET_TEST_PYTHON}" AND EXISTS "${VC_MCP_NNUNET_TEST_MODEL}/fold_0/checkpoint_best.pth")
        add_executable(vc_mcp_segmentation_test tests/SegmentationMcpTest.cpp)
        target_compile_features(vc_mcp_segmentation_test PRIVATE cxx_std_23)
        target_link_libraries(vc_mcp_segmentation_test PRIVATE vc_mcp_phase1)
        target_compile_definitions(vc_mcp_segmentation_test PRIVATE
            VC_MCP_NNUNET_PYTHON="${VC_MCP_NNUNET_TEST_PYTHON}"
            VC_MCP_NNUNET_ADAPTER="${CMAKE_CURRENT_SOURCE_DIR}/nnunet_segmentation_adapter.py"
            VC_MCP_VOLUME_STAGER="${CMAKE_CURRENT_SOURCE_DIR}/volume_stager.py"
            VC_MCP_NNUNET_MODEL="${VC_MCP_NNUNET_TEST_MODEL}")
        vc_mcp_enable_assertions(vc_mcp_segmentation_test)
        set_target_properties(vc_mcp_segmentation_test PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)
        add_test(NAME vc_mcp_segmentation COMMAND vc_mcp_segmentation_test)
        set_tests_properties(vc_mcp_segmentation PROPERTIES TIMEOUT 180)
    endif()

    add_executable(vc_mcp_protocol_test tests/ProtocolTest.cpp)
    target_compile_features(vc_mcp_protocol_test PRIVATE cxx_std_23)
    target_link_libraries(vc_mcp_protocol_test PRIVATE vc_mcp_phase1)
    target_compile_definitions(vc_mcp_protocol_test PRIVATE
        VC_MCP_DINOV3_ADAPTER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/dinov3_headless_adapter.py")
    vc_mcp_enable_assertions(vc_mcp_protocol_test)
    set_target_properties(vc_mcp_protocol_test PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)
    add_test(NAME vc_mcp_protocol COMMAND vc_mcp_protocol_test)

    add_executable(vc_mcp_http_transport_test tests/HttpTransportTest.cpp)
    target_compile_features(vc_mcp_http_transport_test PRIVATE cxx_std_23)
    target_link_libraries(vc_mcp_http_transport_test PRIVATE vc_mcp_phase1)
    vc_mcp_enable_assertions(vc_mcp_http_transport_test)
    set_target_properties(vc_mcp_http_transport_test PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)
    add_test(NAME vc_mcp_http_transport COMMAND vc_mcp_http_transport_test)
    set_tests_properties(vc_mcp_http_transport PROPERTIES TIMEOUT 30)
endif()
