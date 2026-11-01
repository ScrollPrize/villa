#include "plugins/PluginController.hpp"

#include "CState.hpp"
#include "CommandLineToolRunner.hpp"
#include "overlays/InkDetectionOverlayController.hpp"

#include "vc/core/PointCollections.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/ui/VCCollection.hpp"

#include <QAction>
#include <QCoreApplication>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QFileInfo>
#include <QInputDialog>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLineEdit>
#include <QMap>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPushButton>
#include <QSaveFile>
#include <QUrl>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace vc3d::plugins;

namespace {

QString runsRoot()
{
    const QString overridden = qEnvironmentVariable("VC3D_PLUGIN_RUNS");
    return overridden.isEmpty() ? QDir::home().filePath(QStringLiteral(".vc3d/plugin-runs")) : overridden;
}

QString kindDescription(ResultKind kind)
{
    switch (kind) {
        case ResultKind::None: return QObject::tr("nothing is loaded");
        case ResultKind::File: return QObject::tr("VC3D shows the file it writes");
        case ResultKind::InkOverlay: return QObject::tr("the image it writes is shown as the active segment's ink overlay");
        case ResultKind::Points: return QObject::tr("the point collections it writes are added");
    }
    return {};
}

} // namespace

PluginController::PluginController(Host host, QObject* parent)
    : QObject(parent)
    , _host(std::move(host))
{
    reload();
}

void PluginController::installMenu(QMenuBar* menuBar)
{
    if (!menuBar || _menu) return;
    _menu = new QMenu(tr("&Plugins"), menuBar);
    const auto actions = menuBar->actions();
    // Keep Help last.
    if (actions.isEmpty()) menuBar->addMenu(_menu);
    else menuBar->insertMenu(actions.back(), _menu);
    rebuildMenu();
}

void PluginController::reload()
{
    _dirs = pluginDirectories(QDir::homePath(), qEnvironmentVariable("VC3D_PLUGIN_PATH"));
    _scan = scanPluginDirectories(_dirs);
    for (const auto& problem : _scan.problems) {
        qWarning().noquote() << "VC3D plugin skipped:" << problem;
    }
    rebuildMenu();
}

void PluginController::rebuildMenu()
{
    if (!_menu) return;
    _menu->clear();
    if (_scan.plugins.isEmpty()) {
        auto* empty = _menu->addAction(tr("No plugins installed"));
        empty->setEnabled(false);
    }
    QMap<QString, QMenu*> submenus;
    for (const auto& plugin : _scan.plugins) {
        QMenu* target = _menu;
        if (!plugin.menu.isEmpty()) {
            if (!submenus.contains(plugin.menu)) submenus.insert(plugin.menu, _menu->addMenu(plugin.menu));
            target = submenus.value(plugin.menu);
        }
        auto* action = target->addAction(plugin.name + QStringLiteral("..."));
        action->setToolTip(QStringLiteral("%1\n%2").arg(plugin.description, plugin.manifestPath).trimmed());
        action->setStatusTip(plugin.description);
        const QString id = plugin.id;
        connect(action, &QAction::triggered, this, [this, id]() { run(id); });
    }
    _menu->setToolTipsVisible(true);
    _menu->addSeparator();
    connect(_menu->addAction(tr("Reload Plugins")), &QAction::triggered, this, &PluginController::reload);
    const QString foldersLabel = _scan.problems.isEmpty()
        ? tr("Plugin Folders...")
        : tr("Plugin Folders (%1 skipped)...").arg(_scan.problems.size());
    connect(_menu->addAction(foldersLabel), &QAction::triggered, this, &PluginController::showFolders);
}

void PluginController::showFolders()
{
    QString text = tr("VC3D reads *.json plugin manifests from these folders, in order:\n\n%1\n\n"
                      "Set VC3D_PLUGIN_PATH to add folders. A plugin runs a program on this computer "
                      "with your permissions; only install manifests you trust.")
                       .arg(_dirs.join(QLatin1Char('\n')));
    if (!_scan.problems.isEmpty()) {
        text += tr("\n\nSkipped:\n%1").arg(_scan.problems.join(QLatin1Char('\n')));
    }
    QMessageBox::information(_host.parent, tr("Plugin Folders"), text);
}

PluginContext PluginController::gatherContext() const
{
    PluginContext ctx;
    CState* state = _host.state;
    if (!state) return ctx;
    if (auto volume = state->currentVolume()) {
        ctx.volume = QString::fromStdString(volume->isRemote() ? volume->remoteLocator()
                                                               : volume->path().string());
        ctx.volumeId = QString::fromStdString(state->currentVolumeId());
    }
    if (auto surface = state->activeSurface().lock()) {
        if (!surface->path.empty()) {
            ctx.segment = QString::fromStdString(surface->path.string());
            ctx.segmentId = QString::fromStdString(state->activeSurfaceId());
        }
    }
    if (state->hasVpkg()) {
        ctx.project = state->projectDirectory();
    }
    if (POI* focus = state->poi("focus")) {
        ctx.point = std::array<double, 3>{focus->p[0], focus->p[1], focus->p[2]};
    }
    if (auto bounds = state->activeFocusBounds()) {
        ctx.region = std::array<double, 6>{bounds->low[0], bounds->low[1], bounds->low[2],
                                           bounds->high[0], bounds->high[1], bounds->high[2]};
    }
    if (_host.inkOverlay && _host.inkOverlay->hasLoadedSelection()) {
        ctx.inkOverlay = QString::fromStdString(_host.inkOverlay->selectedPath().string());
    }
    return ctx;
}

bool PluginController::run(const QString& pluginId)
{
    const auto it = std::find_if(_scan.plugins.cbegin(), _scan.plugins.cend(),
                                 [&](const PluginManifest& p) { return p.id == pluginId; });
    if (it == _scan.plugins.cend()) return false;
    const PluginManifest manifest = *it;
    QWidget* parent = _host.parent;
    auto warn = [&](const QString& text) {
        QMessageBox::warning(parent, manifest.name, text);
        return false;
    };

    CommandLineToolRunner* runner = _host.runner ? _host.runner() : nullptr;
    if (!runner) return warn(tr("Command line tools are not available."));
    if (runner->isRunning()) return warn(tr("Another tool is still running."));

    const QString program = resolveProgram(manifest, QCoreApplication::applicationDirPath());
    if (program.isEmpty()) {
        return warn(tr("Cannot find the program \"%1\" next to VC3D or on PATH.\n\nManifest: %2")
                        .arg(manifest.program, manifest.manifestPath));
    }

    PluginContext ctx = gatherContext();
    for (const auto& input : manifest.inputs) {
        bool ok = false;
        const QString value = QInputDialog::getText(parent, manifest.name, input.label,
                                                    QLineEdit::Normal, input.defaultValue, &ok);
        if (!ok) return false;
        ctx.inputs.insert(input.name, value.trimmed());
    }

    const QString stamp = QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMdd-HHmmss-zzz"));
    const QString outputDir = QDir(runsRoot()).filePath(manifest.id + QLatin1Char('/') + stamp);
    ctx.output = outputDir;
    const Expansion expansion = expand(manifest, ctx);
    if (!expansion.error.isEmpty()) return warn(expansion.error);

    const QString command = displayCommand(program, expansion.args);
    QMessageBox confirm(QMessageBox::Question, manifest.name,
                        tr("Run this command?\n\n%1\n\nIt runs on this computer with your permissions. "
                           "Output folder: %2\nAfterwards %3.")
                            .arg(command, outputDir, kindDescription(manifest.resultKind)),
                        QMessageBox::Yes | QMessageBox::Cancel, parent);
    confirm.setInformativeText(tr("Manifest: %1").arg(manifest.manifestPath));
    if (confirm.exec() != QMessageBox::Yes) return false;

    if (!QDir().mkpath(outputDir)) return warn(tr("Cannot create the output folder %1").arg(outputDir));

    Run run{manifest, program, expansion.args, outputDir, expansion.resultPath, ctx.segmentId};
    writeRecord(run, QStringLiteral("started"));

    QPointer<PluginController> guard(this);
    auto connection = std::make_shared<QMetaObject::Connection>();
    *connection = connect(runner, &CommandLineToolRunner::toolFinished, this,
                          [this, guard, connection, run](CommandLineToolRunner::Tool tool, bool success,
                                                         const QString& message, const QString&, bool) {
        if (!guard) {
            disconnect(*connection);
            return;
        }
        if (tool != CommandLineToolRunner::Tool::CustomCommand) return;
        disconnect(*connection);
        finish(run, success, message);
    });

    // Silent keeps the runner's generic dialogs away; the console still shows.
    const CommandLineToolRunner::ExecutionOptions options{CommandLineToolRunner::Presentation::Silent, true};
    if (!runner->executeCustomCommand(program, expansion.args, manifest.name, options)) {
        disconnect(*connection);
        writeRecord(run, QStringLiteral("failed"), tr("did not start"));
        return warn(tr("%1 did not start. The console has details.").arg(manifest.name));
    }
    if (_host.status) _host.status(tr("Running plugin %1").arg(manifest.name), 0);
    return true;
}

void PluginController::finish(const Run& run, bool success, const QString& message)
{
    if (!success) {
        writeRecord(run, QStringLiteral("failed"), message);
        if (_host.status) _host.status(tr("Plugin %1 failed").arg(run.manifest.name), 5000);
        QMessageBox::critical(_host.parent, run.manifest.name,
                              tr("%1\n\nOutput folder: %2").arg(message, run.outputDir));
        return;
    }

    QString summary;
    try {
        summary = loadResult(run);
    } catch (const std::exception& e) {
        const QString why = QString::fromStdString(e.what());
        writeRecord(run, QStringLiteral("load_failed"), why);
        QMessageBox::warning(_host.parent, run.manifest.name,
                             tr("The program finished, but its result was not loaded:\n\n%1\n\nOutput folder: %2")
                                 .arg(why, run.outputDir));
        return;
    }
    writeRecord(run, QStringLiteral("finished"), summary);
    if (_host.status) _host.status(summary, 8000);

    if (run.manifest.resultKind == ResultKind::File) {
        QMessageBox box(QMessageBox::Information, run.manifest.name, summary, QMessageBox::Close, _host.parent);
        auto* open = box.addButton(tr("Open"), QMessageBox::AcceptRole);
        box.exec();
        if (box.clickedButton() == open) {
            QDesktopServices::openUrl(QUrl::fromLocalFile(run.resultPath));
        }
        return;
    }
    QMessageBox::information(_host.parent, run.manifest.name, summary);
}

QString PluginController::loadResult(const Run& run)
{
    const auto& m = run.manifest;
    if (m.resultKind == ResultKind::None) {
        return tr("%1 finished. Output folder: %2").arg(m.name, run.outputDir);
    }
    const QFileInfo result(run.resultPath);
    if (!result.exists()) {
        throw std::runtime_error(tr("expected result %1 was not written").arg(run.resultPath).toStdString());
    }
    CState* state = _host.state;
    const std::string path = run.resultPath.toStdString();

    switch (m.resultKind) {
        case ResultKind::None:
        case ResultKind::File:
            return tr("%1 wrote %2").arg(m.name, run.resultPath);

        case ResultKind::InkOverlay: {
            if (run.segmentId.isEmpty()) {
                throw std::runtime_error("no segment was active when the plugin started");
            }
            if (!state || state->activeSurfaceId() != run.segmentId.toStdString()) {
                throw std::runtime_error("the active segment changed while the plugin ran");
            }
            if (!_host.inkOverlay) throw std::runtime_error("the ink detection overlay is not available");
            // Published detections come from the surface catalog; a plugin's
            // image is shown for this session without being published.
            _host.inkOverlay->setSelectedPath(std::filesystem::path(path));
            if (!_host.inkOverlay->hasLoadedSelection()) {
                throw std::runtime_error("the image could not be shown on the active segment");
            }
            return tr("%1 showed an ink overlay on %2").arg(m.name, run.segmentId);
        }

        case ResultKind::Points: {
            VCCollection* live = state ? state->pointCollection() : nullptr;
            if (!live) throw std::runtime_error("no point collections are available");
            PointCollections loaded;
            if (!loaded.loadFromJSON(path)) throw std::runtime_error("not a VC3D point collection file");
            // Add, never replace: the user's collections stay as they are.
            std::vector<uint64_t> ids;
            for (const auto& entry : loaded.getAllCollections()) ids.push_back(entry.first);
            std::sort(ids.begin(), ids.end());
            int collections = 0;
            int points = 0;
            for (const auto id : ids) {
                const auto& col = loaded.getAllCollections().at(id);
                std::string name = col.name.empty() ? m.id.toStdString() : col.name;
                if (live->getCollectionId(name)) name = live->generateNewCollectionName(name + "-");
                std::vector<uint64_t> pointIds;
                for (const auto& entry : col.points) pointIds.push_back(entry.first);
                std::sort(pointIds.begin(), pointIds.end());
                for (const auto pid : pointIds) {
                    const auto& source = col.points.at(pid);
                    auto added = live->addPoint(name, source.p);
                    if (std::isfinite(source.winding_annotation)) {
                        added.winding_annotation = source.winding_annotation;
                        live->updatePoint(added);
                    }
                    ++points;
                }
                // addPoint made the collection with VC3D's default metadata (absolute windings);
                // keep the file's, so relative windings stay relative.
                if (const auto liveId = live->getCollectionId(name)) {
                    live->setCollectionMetadata(*liveId, col.metadata);
                }
                ++collections;
            }
            return tr("%1 added %2 point(s) in %3 collection(s)").arg(m.name).arg(points).arg(collections);
        }

    }
    return {};
}

void PluginController::writeRecord(const Run& run, const QString& state, const QString& message) const
{
    // plugin-run.json: what ran, from which manifest, and how it ended.
    QJsonObject record{
        {QStringLiteral("schema"), QStringLiteral("vc3d-plugin-run/1")},
        {QStringLiteral("plugin_id"), run.manifest.id},
        {QStringLiteral("manifest"), run.manifest.manifestPath},
        {QStringLiteral("manifest_sha256"), QString::fromLatin1(run.manifest.sha256Hex)},
        {QStringLiteral("program"), run.program},
        {QStringLiteral("args"), QJsonArray::fromStringList(run.args)},
        {QStringLiteral("result_kind"), resultKindName(run.manifest.resultKind)},
        {QStringLiteral("result_path"), run.resultPath},
        {QStringLiteral("state"), state},
        {QStringLiteral("updated_utc"), QDateTime::currentDateTimeUtc().toString(Qt::ISODate)},
    };
    if (!message.isEmpty()) record.insert(QStringLiteral("message"), message);
    QSaveFile file(QDir(run.outputDir).filePath(QStringLiteral("plugin-run.json")));
    if (file.open(QIODevice::WriteOnly)) {
        file.write(QJsonDocument(record).toJson(QJsonDocument::Indented));
        file.commit();
    }
}
