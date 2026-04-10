#pragma once
// vc4d plugin interface.
// Each plugin is a shared library (.so) exporting a factory function.
// The host loads plugins from a directory and calls their factory to
// get a Plugin instance. Plugins can add dock widgets, menu items,
// toolbar buttons, and respond to viewer events.

#include <cstdint>
#include <string>

class QWidget;
class QDockWidget;
class QMenu;
class QToolBar;

namespace vc4d {

class App;  // forward — the host application

// Base class for all plugins. Override what you need.
class Plugin {
public:
    virtual ~Plugin() = default;

    // Called once after construction. The plugin can use app to:
    //   - get the viewer widget and volume
    //   - add dock widgets, menus, toolbar buttons
    //   - connect to signals
    virtual void init(App* app) = 0;

    // Plugin identity
    virtual const char* name() const = 0;
    virtual const char* version() const { return "0.1"; }
};

} // namespace vc4d

// Every plugin .so must export this function:
//   extern "C" vc4d::Plugin* vc4d_create_plugin();
using vc4d_create_plugin_fn = vc4d::Plugin* (*)();
