# Standalone unit test for descriptor validation and the small set of parsing
# helpers used behind it. It needs no process launch or volume data. QtGui is
# linked only for QKeySequence key parsing; no QGuiApplication is created.
find_package(Qt6 REQUIRED COMPONENTS Core Gui)

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
    Qt6::Gui
    opencv_core
    vc_core)

add_test(NAME agent_bridge_contract COMMAND test_agent_bridge_contract)
vc_register_test_target(test_agent_bridge_contract)
vc_register_ctest(agent_bridge_contract test_agent_bridge_contract)

# Offscreen widget test for the keyboard synthesis behind canvas.key:
# shortcuts and widgets receive synthesized keys, the cached modifiers are
# reset, dialogs and menus a key opens are answered as Escape would, and
# focus returns. It needs no VC3D window.
find_package(Qt6 REQUIRED COMPONENTS Test Widgets)

add_executable(test_agent_bridge_widgets
    test_agent_bridge_widgets.cpp
    ../AgentBridgeKeyboard.cpp)

set_target_properties(test_agent_bridge_widgets PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_agent_bridge_widgets PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_agent_bridge_widgets PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Test
    Qt6::Widgets)

add_test(NAME agent_bridge_widgets COMMAND test_agent_bridge_widgets)
vc_register_test_target(test_agent_bridge_widgets)
vc_register_ctest(agent_bridge_widgets test_agent_bridge_widgets)
set_tests_properties(agent_bridge_widgets PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen" TIMEOUT 60)
