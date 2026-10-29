#pragma once

#include "plugins/PluginManifest.hpp"

#include <QObject>
#include <QPointer>
#include <QString>

#include <functional>
#include <optional>

class CState;
class CommandLineToolRunner;
class InkDetectionOverlayController;
class QMenu;
class QMenuBar;
class QWidget;

// Lists manifest-declared plugins in a Plugins menu, runs one through the
// shared CommandLineToolRunner and loads what it wrote with VC3D's existing
// loaders. It never loads code: see docs/plugins.md.
class PluginController : public QObject
{
    Q_OBJECT

public:
    struct Host {
        QWidget* parent{nullptr};
        CState* state{nullptr};
        InkDetectionOverlayController* inkOverlay{nullptr};
        std::function<CommandLineToolRunner*()> runner;
        std::function<void(const QString&, int)> status;
    };

    explicit PluginController(Host host, QObject* parent = nullptr);

    void installMenu(QMenuBar* menuBar);
    void reload();

    [[nodiscard]] const QList<vc3d::plugins::PluginManifest>& plugins() const { return _scan.plugins; }
    [[nodiscard]] const QStringList& problems() const { return _scan.problems; }

    // Asks for inputs, shows the exact command, then starts the plugin.
    bool run(const QString& pluginId);

private:
    struct Run {
        vc3d::plugins::PluginManifest manifest;
        QString program;
        QStringList args;
        QString outputDir;
        QString resultPath;
        QString segmentId;
    };

    [[nodiscard]] vc3d::plugins::PluginContext gatherContext() const;
    void rebuildMenu();
    void showFolders();
    void finish(const Run& run, bool success, const QString& message);
    // Returns a one-line summary of what was loaded; throws std::runtime_error.
    QString loadResult(const Run& run);
    void writeRecord(const Run& run, const QString& state, const QString& message = {}) const;

    Host _host;
    QPointer<QMenu> _menu;
    vc3d::plugins::ScanResult _scan;
    QStringList _dirs;
};
