#pragma once
// vc4d::App — the host application interface visible to plugins.
// Plugins use this to access the viewer, volume, and add UI elements.

#include "volume.hpp"
#include "types.hpp"

#include <QMainWindow>
#include <QDockWidget>
#include <QMenuBar>
#include <QToolBar>
#include <QStatusBar>

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vc4d {

class Viewer;   // forward
class Plugin;   // forward

class App : public QMainWindow {
    Q_OBJECT

public:
    // --- Volume access ---
    virtual vc::Volume* volume() = 0;
    virtual void open_volume(const std::filesystem::path& path) = 0;

    // --- Viewer access ---
    virtual Viewer* viewer() = 0;

    // --- Plugin UI hooks ---
    // Plugins call these to add their UI elements to the host.
    virtual QDockWidget* add_dock(const QString& title, QWidget* widget,
                                   Qt::DockWidgetArea area = Qt::RightDockWidgetArea) = 0;
    virtual QMenu* add_menu(const QString& title) = 0;
    virtual QToolBar* add_toolbar(const QString& title) = 0;

    // --- Status bar ---
    virtual void set_status(const QString& msg, int timeout_ms = 3000) = 0;

signals:
    // Emitted when a volume is opened/closed
    void volume_opened();
    void volume_closed();
};

} // namespace vc4d
