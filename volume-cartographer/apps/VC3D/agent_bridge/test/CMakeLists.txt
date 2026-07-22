# Standalone unit test for the Agent Bridge strict wire-parameter helpers
# (apps/VC3D/agent_bridge/AgentBridgeInternal.hpp). Deterministic, no process
# launch, no volume data — safe for the default ctest run. Built only under
# VC_TESTING (wired from the top-level testing block).
#
# The helpers are header-only and pure (QJsonValue in, QPointF/cv::Vec3f/int/…
# out, AgentBridgeError on a malformed value), so the test links only what those
# headers need: Qt6::Core (QJson*, QPointF, QString), opencv_core (cv::Vec3f),
# and vc_core (for vc/core headers reached transitively via ManualAddTool.hpp).
find_package(Qt6 REQUIRED COMPONENTS Core)

add_executable(test_agent_bridge_parse test_agent_bridge_parse.cpp)

# The included app headers declare Q_OBJECT classes, but the test never
# instantiates them (it only uses the free inline helpers), so no MOC is needed.
set_target_properties(test_agent_bridge_parse PROPERTIES
    AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_agent_bridge_parse PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_agent_bridge_parse PRIVATE
    Qt6::Core
    opencv_core
    vc_core)

add_test(NAME agent_bridge_parse COMMAND test_agent_bridge_parse)
set_property(GLOBAL APPEND PROPERTY VC_TEST_TARGETS test_agent_bridge_parse)
