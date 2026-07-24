# Standalone unit test for descriptor validation and the small set of parsing
# helpers used behind it. It needs no process launch or volume data.
find_package(Qt6 REQUIRED COMPONENTS Core)

add_executable(test_agent_bridge_contract
    test_agent_bridge_contract.cpp
    ../AgentBridgeMethod.cpp)

# The included app headers declare Q_OBJECT classes, but the test never
# instantiates them (it only uses the free inline helpers), so no MOC is needed.
set_target_properties(test_agent_bridge_contract PROPERTIES
    AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_agent_bridge_contract PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_agent_bridge_contract PRIVATE
    Qt6::Core
    opencv_core
    vc_core)

add_test(NAME agent_bridge_contract COMMAND test_agent_bridge_contract)
set_property(GLOBAL APPEND PROPERTY VC_TEST_TARGETS test_agent_bridge_contract)
