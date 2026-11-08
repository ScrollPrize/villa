#include "ink_labels/InkLabelController.hpp"

#include "CState.hpp"
#include "Keybinds.hpp"
#include "SpiralBrushCursorWidget.hpp"
#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "overlays/InkDetectionOverlayController.hpp"
#include "overlays/InkLabelOverlayController.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <QAction>
#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QEvent>
#include <QFutureInterface>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QFutureWatcher>
#include <QGraphicsView>
#include <QKeyEvent>
#include <QLockFile>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPushButton>
#include <QShortcut>
#include <QStandardPaths>
#include <QTimer>
#include <QWheelEvent>
#include <QWidget>
#include <QtConcurrent/QtConcurrent>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <system_error>
#include <utility>

namespace fs = std::filesystem;
using namespace vc3d::ink_labels;

namespace {

constexpr int kAutosaveTickMs = 2000;
constexpr qint64 kAutosaveQuietMs = 2000;
constexpr qint64 kAutosaveIntervalMs = 120000;
constexpr double kLassoStepDevicePx = 2.0;

qint64 nowMs()
{
    return QDateTime::currentMSecsSinceEpoch();
}

QString qpath(const fs::path& p)
{
    return QString::fromStdString(p.string());
}

QString className(PaintClass cls)
{
    switch (cls) {
        case PaintClass::Ink:
            return QObject::tr("ink");
        case PaintClass::Background:
            return QObject::tr("background");
        case PaintClass::Validation:
            return QObject::tr("validation");
    }
    return {};
}

QFuture<SaveOutcome> readyOutcome(SaveOutcome outcome)
{
    QFutureInterface<SaveOutcome> fi;
    fi.reportStarted();
    fi.reportResult(std::move(outcome));
    fi.reportFinished();
    return fi.future();
}

SaveOutcome failedOutcome(const QString& error)
{
    SaveOutcome o;
    o.error = error;
    return o;
}

QString actorKey(Actor actor)
{
    return actor == Actor::Agent ? QStringLiteral("agent") : QStringLiteral("user");
}

cv::Rect changedRect(const TileRaster& raster, const HistoryEntry& entry)
{
    cv::Rect r;
    for (const auto& change : entry.changes) {
        const cv::Rect t = raster.tileRect(change.index);
        r = r.empty() ? t : (r | t);
    }
    return r;
}

bool isVolumetricCamera(VolumeViewerBase* viewer)
{
    return viewer->isCompositeEnabled() && viewer->compositeRenderSettings().params.volumetric;
}

}  // namespace

struct InkLabelController::Session {
    SessionInfo info;
    TileRaster raster;
    History history;
    // Declared after the raster it edits.
    std::optional<EditTxn> txn;
    std::shared_ptr<QLockFile> lock;
    std::map<LabelKind, std::string> expectedSha;
    bool validationFileExists = false;
    std::weak_ptr<QuadSurface> surface;
    QuadSurface* surfaceRaw = nullptr;
    std::string geometrySha;
    // The segment's grid changed since the hash was taken.
    bool geometryStale = false;
    // The store's folder and lock are taken at the first edit, so opening
    // a segment that has no labels writes nothing to the project.
    bool lockPending = false;
    // The newest write_seq of the store's trainer sidecar or checkpoint.
    uint64_t writeSeq = 0;
    QJsonObject edits;
    QJsonArray provenance;
    std::optional<cv::Point2d> lastPoint;
    cv::Point2d strokePrev;
    QString strokeKind;
    qint64 lastCommitMs = 0;
    qint64 lastWriteMs = 0;
    uint64_t generation = 0;

    QString storeKey() const { return qpath(info.paths.dir); }
    bool writable() const { return !info.readOnly && !info.locked; }
};

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

InkLabelController::InkLabelController(Deps deps, QObject* parent)
    : QObject(parent)
    , _deps(std::move(deps))
    , _overlay(std::make_unique<InkLabelOverlayController>(*this))
{
    if (_deps.viewers) {
        _overlay->bindToViewerManager(_deps.viewers);
        connect(_deps.viewers, &ViewerManager::baseViewerCreated, this, &InkLabelController::bindViewer);
        connect(_deps.viewers, &ViewerManager::baseViewerClosing, this, &InkLabelController::unbindViewer);
        for (VolumeViewerBase* viewer : _deps.viewers->baseViewers())
            bindViewer(viewer);
    }
    if (_deps.state) {
        connect(_deps.state, &CState::surfaceChanged, this, [this](const std::string& name, std::shared_ptr<Surface>, bool) {
            if (name.empty() || name == "segmentation")
                onStateSurfaceChanged();
        });
        connect(_deps.state, &CState::vpkgChanged, this, [this](std::shared_ptr<VolumePkg>) { closeSession(); });
        connect(_deps.state, &CState::volumeClosing, this, [this]() { closeSession(); });
    }
    createShortcuts();
    // Enter, Esc and Shift+U follow the suggestions and running assists.
    connect(this, &InkLabelController::assistChanged, this, &InkLabelController::updateShortcutStates);
    _labelModeAction = new QAction(tr("Ink label mode") + QLatin1Char('\t')
                                       + vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::InkLabelMode)
                                             .toString(QKeySequence::NativeText),
                                   this);
    _labelModeAction->setObjectName(QStringLiteral("inkLabelModeAction"));
    _labelModeAction->setCheckable(true);
    _labelModeAction->setToolTip(tr("Paint ink, background and validation labels on the segment view"));
    connect(_labelModeAction, &QAction::triggered, this, [this](bool checked) {
        setLabelModeEnabled(checked);
        // Refused: the action shows the real state again.
        _labelModeAction->setChecked(_labelMode);
    });
    connect(this, &InkLabelController::labelModeChanged, _labelModeAction, &QAction::setChecked);
    _autosaveTimer = new QTimer(this);
    _autosaveTimer->setInterval(kAutosaveTickMs);
    connect(_autosaveTimer, &QTimer::timeout, this, &InkLabelController::onAutosaveTick);
    _autosaveTimer->start();
    updateShortcutStates();
}

InkLabelController::~InkLabelController()
{
    _autosaveTimer->stop();
    // The owner is being torn down: report nothing and touch no other tool.
    _deps.status = nullptr;
    _deps.otherSurfaceToolActive = nullptr;
    _deps.segmentationEditing = nullptr;
    _deps.stopSurfacePainting = nullptr;
    _deps.workspaceActive = nullptr;
    _deps.modeLine = nullptr;
    // Commit and checkpoint unsaved edits; the write keeps its own copies.
    closeSession();
    for (auto& bound : _bound) {
        if (bound.viewport)
            bound.viewport->removeEventFilter(this);
        if (bound.view)
            bound.view->removeEventFilter(this);
        if (bound.cursor)
            bound.cursor->deleteLater();
    }
    _bound.clear();
    for (QShortcut* shortcut : std::as_const(_shortcuts))
        delete shortcut;
    _shortcuts.clear();
}

void InkLabelController::createShortcuts()
{
    if (!_deps.shortcutHost)
        return;
    namespace keys = vc3d::keybinds::shortcuts;
    const auto make = [this](const vc3d::keybinds::ShortcutDef& def, auto&& slot) {
        auto* shortcut = new QShortcut(_deps.shortcutHost);
        shortcut->setKeys(vc3d::keybinds::sequencesFor(def));
        // Window scope, as CWindow's own keys: a click in a dock or the dock
        // bar's popup must not turn the label keys off. They are enabled only
        // while the segment workspace is current (updateShortcutStates).
        shortcut->setContext(Qt::WindowShortcut);
        shortcut->setAutoRepeat(false);
        shortcut->setObjectName(QString::fromUtf8(def.id));
        connect(shortcut, &QShortcut::activated, this, std::forward<decltype(slot)>(slot));
        _shortcuts.append(shortcut);
        return shortcut;
    };
    _toggleShortcut = make(keys::InkLabelMode, [this] { setLabelModeEnabled(!_labelMode); });
    make(keys::InkLabelInk, [this] {
        setErase(false);
        setPaintClass(PaintClass::Ink);
    });
    make(keys::InkLabelBackground, [this] {
        setErase(false);
        setPaintClass(PaintClass::Background);
    });
    make(keys::InkLabelValidation, [this] {
        setErase(false);
        setPaintClass(PaintClass::Validation);
    });
    make(keys::InkLabelEraser, [this] { setErase(!_erase); });
    make(keys::InkLabelTool, [this] {
        setTool(_tool == Tool::Brush ? Tool::Polygon : _tool == Tool::Polygon ? Tool::Fill : Tool::Brush);
    });
    make(keys::InkLabelSuggest, [this] { suggest(); });
    _pickAllShortcut = make(keys::InkLabelPickAll, [this] { pickAllSuggestions(); });
    _showShortcut = make(keys::InkLabelShow, [this] {
        setDisplay(!_showLabels, _showInk, _showSupervision, _showValidation, _opacity);
        statusMessage(_showLabels ? tr("Ink labels shown") : tr("Ink labels hidden"), 2000);
    });
    make(keys::InkLabelUndo, [this] {
        if (!undo(Actor::User, true))
            statusMessage(tr("Nothing to undo in the ink labels"), 2000);
    });
    make(keys::InkLabelRedo, [this] {
        if (!redo(Actor::User))
            statusMessage(tr("Nothing to redo in the ink labels"), 2000);
    });
    make(keys::InkLabelSave, [this] { saveInteractive(); });

    // Enter, Backspace and Esc act on the open polygon wherever focus is in
    // the window; they are enabled only while there is something to act on,
    // so they never take these keys from a dialog button or a text field
    // otherwise.
    for (const int key : {int(Qt::Key_Return), int(Qt::Key_Enter), int(Qt::Key_Backspace), int(Qt::Key_Escape)}) {
        auto* shortcut = new QShortcut(QKeySequence(key), _deps.shortcutHost);
        shortcut->setContext(Qt::WindowShortcut);
        shortcut->setAutoRepeat(key == Qt::Key_Backspace);
        shortcut->setObjectName(QStringLiteral("ink_label_edit_key_%1").arg(key, 0, 16));
        shortcut->setProperty("inkLabelKey", key);
        connect(shortcut, &QShortcut::activated, this, [this, key] { handleEditKey(key); });
        _editKeyShortcuts.append(shortcut);
    }
}

void InkLabelController::updateShortcutStates()
{
    const bool workspace = !_deps.workspaceActive || _deps.workspaceActive();
    for (QShortcut* shortcut : std::as_const(_shortcuts)) {
        if (shortcut == _toggleShortcut)
            shortcut->setEnabled(workspace);
        else if (shortcut == _showShortcut)
            shortcut->setEnabled(workspace && _session != nullptr);
        else
            shortcut->setEnabled(workspace && _labelMode && _session != nullptr);
    }
    // Enter closes a polygon or accepts suggestions, Esc cancels either (or
    // drops a running assist), Backspace shortens a polygon.
    const bool polygon = _labelMode && !_polygon.empty();
    for (QShortcut* shortcut : std::as_const(_editKeyShortcuts)) {
        const int key = shortcut->property("inkLabelKey").toInt();
        bool enabled = polygon;
        if (key == Qt::Key_Return || key == Qt::Key_Enter)
            enabled = enabled || _suggestions.has_value();
        else if (key == Qt::Key_Escape)
            enabled = enabled || _suggestions.has_value() || _assistRunning;
        shortcut->setEnabled(workspace && enabled);
    }
    if (_pickAllShortcut)
        _pickAllShortcut->setEnabled(workspace && _suggestions.has_value());
}

bool InkLabelController::handleEditKey(int key)
{
    if (_polygon.empty())
        return handleAssistKey(key);
    switch (key) {
        case Qt::Key_Return:
        case Qt::Key_Enter:
            closePolygon();
            return true;
        case Qt::Key_Backspace:
            _polygon.pop_back();
            updatePolygonPreview();
            return true;
        case Qt::Key_Escape:
            cancelPolygon();
            return true;
        default:
            return false;
    }
}

void InkLabelController::statusMessage(const QString& text, int ms) const
{
    if (_deps.status)
        _deps.status(text, ms);
}

// ---------------------------------------------------------------------------
// Label mode and tool state
// ---------------------------------------------------------------------------

void InkLabelController::setLabelModeRefusal(const QString& reason)
{
    if (_labelModeRefusal == reason)
        return;
    _labelModeRefusal = reason;
    emit labelModeRefusalChanged();
}

void InkLabelController::setLabelModeEnabled(bool enabled)
{
    if (enabled == _labelMode)
        return;
    if (enabled) {
        const auto refuse = [this](const QString& reason, int ms) {
            setLabelModeRefusal(reason);
            qWarning().noquote() << "Ink label mode refused:" << reason;
            statusMessage(reason, ms);
        };
        // Refuse before touching any other tool: pressing I while editing
        // changes nothing.
        if (_deps.segmentationEditing && _deps.segmentationEditing()) {
            refuse(tr("Turn off segmentation editing to label ink; editing changes the grid labels are attached to."), 6000);
            return;
        }
        if (_deps.stopSurfacePainting)
            _deps.stopSurfacePainting();
        if (_deps.otherSurfaceToolActive && _deps.otherSurfaceToolActive()) {
            refuse(tr("Turn off surface mask painting to label ink."), 6000);
            return;
        }
        if (!_session || _deps.state) {
            const OpenOutcome outcome = openForActiveSegment();
            if (!outcome.ok) {
                refuse(outcome.message, 8000);
                return;
            }
        }
        if (!_session)
            return;
        setLabelModeRefusal({});
        _labelMode = true;
        if (!_hintShown) {
            _hintShown = true;
            statusMessage(tr("Ink label mode: W ink, F background, V validation, Z eraser, L tool, H show or hide, Ctrl+Z undo, "
                             "Ctrl+S save, I to leave. Left drag paints, right drag pans."),
                          10000);
        } else {
            announceToolState();
        }
    } else {
        commitOpenStroke();
        cancelPolygon();
        _labelMode = false;
        _leftDown = false;
        hideCursors();
        statusMessage(tr("Ink label mode off"), 2000);
    }
    updateShortcutStates();
    updateModeLine();
    emit labelModeChanged(_labelMode);
    _overlay->refreshAll();
}

PaintOp InkLabelController::currentOp() const
{
    return paintOp(_class, _erase, _backgroundKeepsInk);
}

QString InkLabelController::toolName() const
{
    switch (_tool) {
        case Tool::Brush:
            return QStringLiteral("brush");
        case Tool::Polygon:
            return QStringLiteral("polygon");
        case Tool::Fill:
            return QStringLiteral("fill");
    }
    return {};
}

QColor InkLabelController::classColor() const
{
    return colorFor(_class, _erase);
}

QColor InkLabelController::colorFor(PaintClass cls, bool erase)
{
    if (erase)
        return QColor(200, 200, 200);
    switch (cls) {
        case PaintClass::Ink:
            return QColor(255, 64, 64);
        case PaintClass::Background:
            return QColor(64, 128, 255);
        case PaintClass::Validation:
            return QColor(255, 200, 0);
    }
    return QColor(255, 64, 64);
}

void InkLabelController::updateModeLine()
{
    if (!_deps.modeLine)
        return;
    if (!_labelMode || !_session) {
        _deps.modeLine(QString());
        return;
    }
    QString what = _erase ? tr("eraser") : className(_class);
    if (_session->info.readOnly || _session->info.locked)
        what += tr(", read-only");
    const QString tool = _tool == Tool::Brush ? tr("brush r %1").arg(QString::number(_radius, 'g', 3)) : toolName();
    _deps.modeLine(tr("Ink labels: %1, %2").arg(what, tool));
}

void InkLabelController::announceToolState()
{
    QString what;
    if (_erase) {
        what = _class == PaintClass::Validation ? tr("eraser, validation off") : tr("eraser, %1 to unknown").arg(className(_class));
    } else {
        what = className(_class);
        if (_class == PaintClass::Background && _backgroundKeepsInk)
            what += tr(" (keeps ink)");
    }
    statusMessage(tr("Ink labels: %1, %2, r %3").arg(what, toolName(), QString::number(_radius, 'g', 4)), 3000);
    updateModeLine();
}

void InkLabelController::setPaintClass(PaintClass cls)
{
    if (_class == cls)
        return;
    _class = cls;
    updatePolygonPreview();
    emit toolStateChanged();
    announceToolState();
}

void InkLabelController::setErase(bool erase)
{
    if (_erase == erase)
        return;
    _erase = erase;
    updatePolygonPreview();
    emit toolStateChanged();
    announceToolState();
}

void InkLabelController::setTool(Tool tool)
{
    if (_tool == tool)
        return;
    commitOpenStroke();
    cancelPolygon();
    _tool = tool;
    for (auto& bound : _bound)
        if (bound.cursor)
            bound.cursor->setCursorState(QPointF(), 0, false, false);
    emit toolStateChanged();
    announceToolState();
}

void InkLabelController::setRadius(double labelPx)
{
    if (!std::isfinite(labelPx))
        return;
    const double r = std::clamp(labelPx, kMinRadius, kMaxRadius);
    if (r == _radius)
        return;
    _radius = r;
    emit toolStateChanged();
    updateModeLine();
}

void InkLabelController::setBackgroundKeepsInk(bool keep)
{
    if (_backgroundKeepsInk == keep)
        return;
    _backgroundKeepsInk = keep;
    emit toolStateChanged();
}

void InkLabelController::setDisplay(bool showLabels, bool ink, bool supervision, bool validation, double opacity)
{
    opacity = std::clamp(std::isfinite(opacity) ? opacity : _opacity, 0.0, 1.0);
    if (showLabels == _showLabels && ink == _showInk && supervision == _showSupervision && validation == _showValidation
        && opacity == _opacity)
        return;
    _showLabels = showLabels;
    _showInk = ink;
    _showSupervision = supervision;
    _showValidation = validation;
    _opacity = opacity;
    _overlay->setStyle(_showInk, _showSupervision, _showValidation, _opacity);
    emit displayChanged();
    _overlay->refreshAll();
}

// ---------------------------------------------------------------------------
// Session access
// ---------------------------------------------------------------------------

const InkLabelController::SessionInfo* InkLabelController::session() const
{
    if (!_session)
        return nullptr;
    _session->info.dirty = _session->history.dirty();
    _session->info.stateSerial = _session->history.stateSerial();
    _session->info.counts = _session->info.locked ? std::array<uint64_t, 8>{} : _session->raster.counts();
    return &_session->info;
}

const TileRaster* InkLabelController::raster() const
{
    return _session && !_session->info.locked ? &_session->raster : nullptr;
}

const History* InkLabelController::history() const
{
    return _session ? &_session->history : nullptr;
}

QuadSurface* InkLabelController::sessionSurface() const
{
    if (!_session)
        return nullptr;
    if (_session->surface.expired())
        return nullptr;
    return _session->surfaceRaw;
}

bool InkLabelController::strokeOpen() const
{
    return _session && _session->txn.has_value();
}

bool InkLabelController::saveInFlight() const
{
    return _session && _inFlight.contains(_session->storeKey());
}

bool InkLabelController::trainerSaveInFlight() const
{
    if (!_session)
        return false;
    const auto it = _inFlight.constFind(_session->storeKey());
    return it != _inFlight.cend() && it->target == Target::Trainer;
}

// ---------------------------------------------------------------------------
// Opening and closing
// ---------------------------------------------------------------------------

std::optional<InkLabelController::OpenRequest> InkLabelController::activeSegmentRequest(OpenOutcome& failure) const
{
    const auto fail = [&failure](OpenError error, const QString& message) -> std::optional<OpenRequest> {
        failure.error = error;
        failure.message = message;
        return std::nullopt;
    };
    if (!_deps.state || !_deps.state->vpkg())
        return fail(OpenError::NoProject, tr("Open a project to label ink."));
    const fs::path projectFile = _deps.state->vpkg()->path();
    if (projectFile.empty() || !projectFile.has_parent_path())
        return fail(OpenError::NoProject, tr("The project has no folder on disk, so ink labels have nowhere to go."));
    // A catalog project lives in the remote cache, which can be wiped and is
    // shared by every sample: its labels go to the application data folder.
    const fs::path projectDir =
        labelRootFor(projectFile, vc3d::remoteCachePathFs(),
                     fs::path(QStandardPaths::writableLocation(QStandardPaths::AppDataLocation).toStdString()) / "open_data_labels");
    auto surface = std::dynamic_pointer_cast<QuadSurface>(_deps.state->surface("segmentation"));
    if (!surface)
        return fail(OpenError::NoSegment, tr("Select a segment to label ink."));
    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty())
        return fail(OpenError::NoSegment, tr("The segment has no grid to label."));
    const cv::Vec2f scale = surface->scale();
    if (!std::isfinite(scale[0]) || !std::isfinite(scale[1]) || scale[0] <= 0.f || scale[1] <= 0.f)
        return fail(OpenError::InvalidScale, tr("The segment's grid scale is not a positive number, so its render canvas is undefined."));

    OpenRequest request;
    request.canvas = vc::RenderCanvas::make(points->size(), scale, 1.0);
    request.projectDir = projectDir;
    request.segmentDir = surface->path;
    // Current object-store surfaces have no persistent tifxyz folder. Use the
    // stable object UUID for local trainer exports, never a hosted cache path.
    request.segmentUuid = surface->objectUuid();
    request.segment = !request.segmentUuid.empty() ? request.segmentUuid : surface->path.filename().string();
    if (request.segment.empty())
        request.segment = !_deps.state->activeSurfaceId().empty() ? _deps.state->activeSurfaceId() : surface->id;
    if (request.segment.empty())
        return fail(OpenError::NoSegment, tr("The segment has no folder name to name its label files after."));
    request.segmentUuid = surface->objectUuid();
    request.surface = surface;
    request.geometrySha = geometrySha256(*surface);
    return request;
}

InkLabelController::OpenOutcome InkLabelController::openForActiveSegment()
{
    return openForActiveSegment(OpenOptions{});
}

InkLabelController::OpenOutcome InkLabelController::openForActiveSegment(const OpenOptions& options)
{
    OpenOutcome outcome;
    if (_openPending) {
        outcome.error = OpenError::Busy;
        outcome.message = tr("Another ink label open is waiting for an answer.");
        return outcome;
    }
    auto request = activeSegmentRequest(outcome);
    if (!request)
        return outcome;
    if (_session) {
        const auto paths = storePathsFor(request->projectDir, request->segment);
        if (_session->info.paths.dir == paths.dir && _session->info.canvas == request->canvas
            && _session->surfaceRaw == request->surface.lock().get()
            && (options.version == 0 || options.version == _session->info.paths.version)) {
            outcome.ok = !_session->info.locked;
            outcome.readOnly = _session->info.readOnly;
            outcome.locked = _session->info.locked;
            if (_session->info.locked) {
                outcome.error = OpenError::LayerLocked;
                outcome.message = _session->info.lockReason;
                outcome.expected = _session->info.lockExpected;
                outcome.got = _session->info.lockGot;
            }
            return outcome;
        }
    }
    return openStore(*request, options);
}

bool InkLabelController::resolveVersion(StorePaths& paths, const OpenRequest& request, const OpenOptions& options,
                                        OpenOutcome& out) const
{
    const std::vector<int> existing = listVersions(paths.dir, paths.prefix);
    const int newest = existing.empty() ? 1 : existing.back();
    if (options.version <= 0) {
        paths.version = newest;
        return true;
    }
    if (options.version == 1 || std::find(existing.begin(), existing.end(), options.version) != existing.end()
        || (options.createVersion && options.version > newest)) {
        paths.version = options.version;
        return true;
    }
    QStringList list;
    for (const int v : existing)
        list << QString::number(v);
    out.error = OpenError::NoVersion;
    out.message = tr("Ink labels for %1 have no version %2 (versions: %3).")
                      .arg(QString::fromStdString(request.segment))
                      .arg(options.version)
                      .arg(existing.empty() ? tr("none") : list.join(QStringLiteral(", ")));
    return false;
}

bool InkLabelController::acquireLock(Session& session, const OpenOptions& options, std::shared_ptr<QLockFile> heldLock,
                                     OpenOutcome& out)
{
    if (heldLock) {
        session.lock = std::move(heldLock);
        return true;
    }
    std::error_code ec;
    if (!fs::is_directory(session.info.paths.dir, ec)) {
        // Nothing to lock yet: the folder and lock come with the first edit.
        session.lockPending = true;
        return true;
    }
    auto lock = std::make_shared<QLockFile>(qpath(session.info.paths.lock()));
    lock->setStaleLockTime(0);
    if (lock->tryLock(0)) {
        session.lock = std::move(lock);
        return true;
    }
    qint64 pid = 0;
    QString host, app;
    lock->getLockInfo(&pid, &host, &app);
    const QString holder = tr("pid %1 on %2").arg(pid).arg(host.isEmpty() ? tr("an unknown host") : host);
    if (!options.readOnlyIfLocked) {
        out.error = OpenError::LockedByOther;
        out.message = tr("Ink labels for %1 are open in another VC3D (%2).").arg(QString::fromStdString(session.info.segment), holder);
        return false;
    }
    session.info.readOnly = true;
    session.info.lockReason = tr("Read-only: these labels are open in another VC3D (%1).").arg(holder);
    out.warnings.append(session.info.lockReason);
    return true;
}

InkLabelController::OpenOutcome InkLabelController::lockLayer(std::unique_ptr<Session> session, OpenOutcome out,
                                                              const QString& reason, const QString& expected, const QString& got)
{
    session->info.locked = true;
    session->info.lockReason = reason;
    session->info.lockExpected = expected;
    session->info.lockGot = got;
    session->lock.reset();
    session->lockPending = false;
    out.ok = false;
    out.locked = true;
    out.error = OpenError::LayerLocked;
    out.message = reason;
    out.expected = expected;
    out.got = got;
    installSession(std::move(session));
    return out;
}

std::optional<InkLabelController::OpenOutcome> InkLabelController::checkSidecar(std::unique_ptr<Session>& session,
                                                                                const OpenRequest& request, const CanvasKey& key,
                                                                                Sidecars& sidecars, OpenOutcome& out)
{
    const StorePaths& paths = session->info.paths;
    std::error_code ec;
    if (fs::exists(paths.sidecar(), ec)) {
        QString error;
        sidecars.trainer = readSidecar(paths.sidecar(), &error);
        if (!sidecars.trainer)
            out.warnings.append(error);
    }
    const auto& trainer = sidecars.trainer;
    if (trainer && !trainer->segmentUuid.empty() && !request.segmentUuid.empty() && trainer->segmentUuid != request.segmentUuid) {
        return lockLayer(std::move(session), out,
                         tr("Labels in %1 were saved for the segment with uuid %2; this segment has uuid %3. Nothing is shown "
                            "or saved.")
                             .arg(qpath(paths.dir), QString::fromStdString(trainer->segmentUuid),
                                  QString::fromStdString(request.segmentUuid)),
                         QString::fromStdString(trainer->segmentUuid), QString::fromStdString(request.segmentUuid));
    }
    if (trainer && trainer->canvas != key) {
        return lockLayer(std::move(session), out,
                         tr("Labels were saved for %1. The segment now has %2. To carry labels across a re-render use "
                            "vesuvius.tifxyz_label_transfer.")
                             .arg(trainer->canvas.describe(), key.describe()),
                         trainer->canvas.describe(), key.describe());
    }
    return std::nullopt;
}

std::optional<Target> InkLabelController::chooseTarget(Session& session, const OpenRequest& request, const OpenOptions& options,
                                                       const CanvasKey& key, Sidecars& sidecars, OpenOutcome& out)
{
    const StorePaths& paths = session.info.paths;
    if (!targetExists(paths, Target::Autosave))
        return Target::Trainer;
    QString error;
    sidecars.autosave = readSidecar(paths.sidecar(Target::Autosave), &error);
    if (sidecars.autosave && sidecars.autosave->canvas != key)
        error = tr("it was written for %1, the segment has %2").arg(sidecars.autosave->canvas.describe(), key.describe());
    const bool usable = sidecars.autosave && sidecars.autosave->canvas == key;
    if (!usable) {
        // Never delete what cannot be read: keep it beside the store.
        sidecars.autosave.reset();
        if (session.writable() && moveTargetAside(paths, Target::Autosave))
            out.warnings.append(tr("An autosave of these labels cannot be restored (%1); it was kept as *.unusable in %2.")
                                    .arg(error, qpath(paths.dir)));
        return Target::Trainer;
    }
    // Every write of the store counts up write_seq; a checkpoint is newer
    // than the trainer save only if it was written after it. The state
    // serial cannot tell, because undo moves it back.
    const bool newer = !sidecars.trainer || sidecars.autosave->writeSeq > sidecars.trainer->writeSeq;
    if (!newer) {
        if (session.writable())
            removeTarget(paths, Target::Autosave);
        sidecars.autosave.reset();
        return Target::Trainer;
    }
    if (!session.writable()) {
        out.warnings.append(tr("A newer autosave exists; open these labels in the VC3D that holds them to restore it."));
        sidecars.autosave.reset();
        return Target::Trainer;
    }
    const QString storeKey = qpath(paths.dir);
    RecoverPolicy policy = options.recover;
    if (_checkpointedThisRun.contains(storeKey))
        policy = RecoverPolicy::Restore;
    if (policy == RecoverPolicy::Ask) {
        QMessageBox box(QMessageBox::Question, tr("Unsaved ink labels"),
                        tr("Unsaved ink labels for %1 from %2 were found. Restore them?")
                            .arg(QString::fromStdString(request.segment), QString::fromStdString(sidecars.autosave->savedAtUtc)),
                        QMessageBox::NoButton, _deps.shortcutHost);
        box.setObjectName(QStringLiteral("inkLabelRecoverPrompt"));
        QPushButton* restore = box.addButton(tr("Restore"), QMessageBox::AcceptRole);
        box.addButton(tr("Discard"), QMessageBox::DestructiveRole);
        box.setDefaultButton(restore);
        // The nested event loop still runs timers, bridge calls and segment
        // changes; _openPending refuses nested opens, and the segment is
        // checked again below.
        box.exec();
        policy = box.clickedButton() == restore ? RecoverPolicy::Restore : RecoverPolicy::Discard;
        if (_deps.state && _deps.state->surface("segmentation").get() != request.surface.lock().get()) {
            out.error = OpenError::Busy;
            out.message = tr("The segment changed while the restore question was open; nothing was opened.");
            return std::nullopt;
        }
    }
    switch (policy) {
        case RecoverPolicy::Restore:
            return Target::Autosave;
        case RecoverPolicy::Discard:
            removeTarget(paths, Target::Autosave);
            _checkpointedThisRun.remove(storeKey);
            sidecars.autosave.reset();
            return Target::Trainer;
        case RecoverPolicy::Fail:
        case RecoverPolicy::Ask:
            break;
    }
    out.error = OpenError::AutosavePending;
    out.autosaveAtUtc = QString::fromStdString(sidecars.autosave->savedAtUtc);
    out.message = tr("Ink labels for %1 have an autosave from %2 that was never saved; open with restore or discard.")
                      .arg(QString::fromStdString(request.segment), out.autosaveAtUtc);
    return std::nullopt;
}

bool InkLabelController::loadInto(Session& session, Target target, const CanvasKey& key, const Sidecars& sidecars, OpenOutcome& out)
{
    const StorePaths& paths = session.info.paths;
    std::error_code ec;
    LoadOutcome load = loadLabels(paths, key, target);
    if (load.canvasMismatch) {
        out.error = OpenError::LayerLocked;
        out.message = load.error;
        out.expected = load.sidecar ? load.sidecar->canvas.describe() : QString();
        out.got = key.describe();
        return false;
    }
    if (!load.ok) {
        out.error = OpenError::ReadFailed;
        out.message = tr("Cannot read the ink labels in %1: %2").arg(qpath(paths.dir), load.error);
        return false;
    }
    out.warnings.append(load.warnings);

    session.raster = std::move(load.raster);
    if (target == Target::Trainer) {
        session.expectedSha = load.sha256;
    } else {
        for (const LabelKind kind : kAllKinds)
            if (fs::exists(paths.file(kind), ec))
                session.expectedSha[kind] = fileSha256(paths.file(kind));
        out.restoredAutosave = true;
    }
    session.validationFileExists = fs::exists(paths.file(LabelKind::Validation), ec);
    const Sidecar* carried = target == Target::Autosave ? &*sidecars.autosave : (sidecars.trainer ? &*sidecars.trainer : nullptr);
    if (carried) {
        session.edits = carried->edits;
        session.provenance = carried->provenance;
    }
    session.writeSeq = std::max(sidecars.trainer ? sidecars.trainer->writeSeq : 0, sidecars.autosave ? sidecars.autosave->writeSeq : 0);
    if (target == Target::Autosave) {
        // A checkpoint is unsaved work by definition: dirty until saved.
        session.history.reset(sidecars.autosave->stateSerial);
        session.history.markClean(std::numeric_limits<uint64_t>::max());
    } else {
        session.history.reset(sidecars.trainer ? sidecars.trainer->stateSerial : 0);
    }
    if (sidecars.trainer && !sidecars.trainer->geometrySha256.empty() && !session.geometrySha.empty()
        && sidecars.trainer->geometrySha256 != session.geometrySha)
        out.warnings.append(tr("The segment's geometry was edited since these labels were saved; labels stay attached to the same "
                               "grid positions."));
    return true;
}

InkLabelController::OpenOutcome InkLabelController::openStore(const OpenRequest& request, const OpenOptions& options)
{
    OpenOutcome out;
    if (_openPending) {
        out.error = OpenError::Busy;
        out.message = tr("Another ink label open is waiting for an answer.");
        return out;
    }
    _openPending = true;
    struct PendingReset {
        bool& flag;
        ~PendingReset() { flag = false; }
    } pendingReset{_openPending};

    StorePaths paths = storePathsFor(request.projectDir, request.segment);
    const QString storeKey = qpath(paths.dir);
    if (!resolveVersion(paths, request, options, out))
        return out;

    if (_session)
        closeSession();
    // A write of this store from an earlier session finishes first, and its
    // lock (this process's) carries over to the new session.
    std::shared_ptr<QLockFile> heldLock;
    if (auto it = _inFlight.find(storeKey); it != _inFlight.end()) {
        it->future.waitForFinished();
        if (it->lock && it->lock->isLocked())
            heldLock = it->lock;
    }

    auto session = std::make_unique<Session>();
    session->info.canvas = request.canvas;
    session->info.paths = paths;
    session->info.segment = request.segment;
    session->info.segmentUuid = request.segmentUuid;
    session->info.segmentDir = request.segmentDir;
    session->surface = request.surface;
    session->surfaceRaw = request.surface.lock().get();
    session->geometrySha = request.geometrySha;
    if (!acquireLock(*session, options, std::move(heldLock), out))
        return out;

    const CanvasKey key = canvasKeyFor(request.canvas);
    Sidecars sidecars;
    if (auto locked = checkSidecar(session, request, key, sidecars, out))
        return *locked;
    const std::optional<Target> target = chooseTarget(*session, request, options, key, sidecars, out);
    if (!target)
        return out;
    if (!loadInto(*session, *target, key, sidecars, out)) {
        if (out.error == OpenError::LayerLocked)
            return lockLayer(std::move(session), out, out.message, out.expected, out.got);
        return out;
    }

    if (sidecars.trainer) {
        LastSave last;
        last.atUtc = QDateTime::fromString(QString::fromStdString(sidecars.trainer->savedAtUtc), Qt::ISODate);
        for (const auto& [kind, sha] : sidecars.trainer->sha256) {
            last.files.emplace_back(kind, paths.file(kind));
            last.sha256[kind] = sha;
        }
        _lastSave = last;
    } else {
        _lastSave.reset();
    }
    _lastAutosave = QDateTime();

    out.ok = true;
    out.readOnly = session->info.readOnly;
    installSession(std::move(session));
    for (const QString& warning : std::as_const(out.warnings))
        statusMessage(warning, 8000);
    return out;
}

bool InkLabelController::prepareEdit(QString* why)
{
    if (!canEdit(why))
        return false;
    if (!_session->lockPending)
        return true;
    const StorePaths& paths = _session->info.paths;
    std::error_code ec;
    fs::create_directories(paths.dir, ec);
    if (ec) {
        if (why)
            *why = tr("Cannot create the label folder %1: %2").arg(qpath(paths.dir), QString::fromStdString(ec.message()));
        return false;
    }
    auto lock = std::make_shared<QLockFile>(qpath(paths.lock()));
    lock->setStaleLockTime(0);
    _session->lockPending = false;
    if (!lock->tryLock(0)) {
        qint64 pid = 0;
        QString host, app;
        lock->getLockInfo(&pid, &host, &app);
        _session->info.readOnly = true;
        _session->info.lockReason = tr("Read-only: these labels were opened in another VC3D (pid %1 on %2).")
                                        .arg(pid)
                                        .arg(host.isEmpty() ? tr("an unknown host") : host);
        if (why)
            *why = _session->info.lockReason;
        emit sessionChanged();
        updateModeLine();
        return false;
    }
    _session->lock = std::move(lock);
    return true;
}

void InkLabelController::installSession(std::unique_ptr<Session> session)
{
    _session = std::move(session);
    setLabelModeRefusal(_session->info.locked ? _session->info.lockReason : QString{});
    _session->generation = ++_sessionGeneration;
    _session->lastWriteMs = nowMs();
    _polygon.clear();
    cancelAssist();
    _overlay->resetForSession();
    updateShortcutStates();
    updateModeLine();
    emit sessionChanged();
    emit historyChanged();
    emit dirtyChanged(_session->history.dirty());
    _overlay->refreshAll();
}

void InkLabelController::closeSession()
{
    if (!_session)
        return;
    commitOpenStroke();
    cancelPolygon();
    cancelAssist();
    if (_session->writable() && _session->history.dirty()) {
        startCheckpoint();
        statusMessage(tr("Unsaved ink labels for %1 kept as autosave").arg(QString::fromStdString(_session->info.segment)), 6000);
    } else if (_session->writable() && !_session->lockPending) {
        // Clean: the labels equal the trainer files (an undo can get back
        // there after a checkpoint), so any checkpoint is stale.
        const QString key = _session->storeKey();
        if (auto it = _inFlight.find(key); it != _inFlight.end())
            it->future.waitForFinished();
        removeTarget(_session->info.paths, Target::Autosave);
        _checkpointedThisRun.remove(key);
    }
    _session.reset();
    ++_sessionGeneration;
    const bool wasLabeling = _labelMode;
    _labelMode = false;
    _leftDown = false;
    hideCursors();
    _overlay->resetForSession();
    updateShortcutStates();
    updateModeLine();
    if (wasLabeling)
        emit labelModeChanged(false);
    emit sessionChanged();
    emit historyChanged();
    emit dirtyChanged(false);
    _overlay->refreshAll();
}

void InkLabelController::onStateSurfaceChanged()
{
    if (!_session || !_deps.state)
        return;
    auto surface = std::dynamic_pointer_cast<QuadSurface>(_deps.state->surface("segmentation"));
    if (surface && surface.get() == _session->surfaceRaw) {
        // Same segment: only a changed canvas ends the session.
        const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
        if (points && vc::RenderCanvas::make(points->size(), surface->scale(), 1.0) == _session->info.canvas) {
            // The grid may have moved: hash it again before the next write.
            _session->geometryStale = true;
            return;
        }
    }
    closeSession();
}

// ---------------------------------------------------------------------------
// Editing
// ---------------------------------------------------------------------------

bool InkLabelController::canEdit(QString* why) const
{
    if (!_session) {
        if (why)
            *why = tr("No ink label session is open.");
        return false;
    }
    if (_session->info.locked || _session->info.readOnly) {
        if (why)
            *why = _session->info.lockReason;
        return false;
    }
    return true;
}

void InkLabelController::notifyRasterChanged(const cv::Rect& labelRect)
{
    if (labelRect.empty())
        return;
    const QRect rect(labelRect.x, labelRect.y, labelRect.width, labelRect.height);
    _overlay->invalidate(rect);
    emit rasterChanged(rect);
    _overlay->refreshAll();
}

void InkLabelController::notifyHistory()
{
    emit historyChanged();
    emit dirtyChanged(_session && _session->history.dirty());
}

void InkLabelController::finishTransaction()
{
    if (!_session || !_session->txn)
        return;
    const cv::Rect dirty = _session->txn->takeDirtyRect();
    const Actor actor = _session->txn->actor();
    HistoryEntry entry = _session->txn->commit(_session->history.nextSerial());
    _session->txn.reset();
    if (!entry.changes.empty()) {
        QJsonObject byActor = _session->edits.value(actorKey(actor)).toObject();
        byActor[_session->strokeKind] = byActor.value(_session->strokeKind).toInt() + 1;
        _session->edits[actorKey(actor)] = byActor;
        _session->history.push(std::move(entry));
        _session->lastCommitMs = nowMs();
    }
    notifyRasterChanged(dirty);
    notifyHistory();
}

void InkLabelController::commitOpenStroke()
{
    finishTransaction();
}

bool InkLabelController::beginStroke(cv::Point2d labelPx, bool fromLastPoint, Actor actor)
{
    QString why;
    if (!prepareEdit(&why)) {
        statusMessage(why, 4000);
        return false;
    }
    if (!std::isfinite(labelPx.x) || !std::isfinite(labelPx.y))
        return false;
    finishTransaction();
    _session->txn.emplace(_session->raster, _erase ? "eraser stroke" : "brush stroke", actor);
    _session->strokeKind = QStringLiteral("brush");
    const PaintOp op = currentOp();
    if (fromLastPoint && _session->lastPoint)
        _session->txn->apply({Shape::Kind::Capsule, {*_session->lastPoint, labelPx}, _radius}, op);
    else
        _session->txn->apply({Shape::Kind::Disc, {labelPx}, _radius}, op);
    _session->strokePrev = labelPx;
    _session->lastPoint = labelPx;
    notifyRasterChanged(_session->txn->takeDirtyRect());
    return true;
}

void InkLabelController::extendStroke(cv::Point2d labelPx)
{
    if (!_session || !_session->txn || !std::isfinite(labelPx.x) || !std::isfinite(labelPx.y))
        return;
    _session->txn->apply({Shape::Kind::Capsule, {_session->strokePrev, labelPx}, _radius}, currentOp());
    _session->strokePrev = labelPx;
    _session->lastPoint = labelPx;
    notifyRasterChanged(_session->txn->takeDirtyRect());
}

void InkLabelController::endStroke()
{
    finishTransaction();
}

uint64_t InkLabelController::applyEdit(const std::function<uint64_t(EditTxn&)>& edit,
                                       Actor actor,
                                       const QString& label,
                                       const QString& editKind)
{
    QString why;
    if (!prepareEdit(&why)) {
        statusMessage(why, 4000);
        return 0;
    }
    finishTransaction();
    _session->strokeKind = editKind;
    _session->txn.emplace(_session->raster, label.toStdString(), actor);
    const uint64_t changed = edit(*_session->txn);
    finishTransaction();
    return changed;
}

uint64_t InkLabelController::applyShape(const Shape& shape, Actor actor)
{
    const QString kind = shape.kind == Shape::Kind::Polygon ? QStringLiteral("polygon") : QStringLiteral("brush");
    return applyShape(shape, currentOp(), actor, kind);
}

uint64_t InkLabelController::applyShape(const Shape& shape, PaintOp op, Actor actor, const QString& label)
{
    const QString kind = shape.kind == Shape::Kind::Polygon ? QStringLiteral("polygon") : QStringLiteral("stroke");
    return applyEdit([&](EditTxn& txn) { return txn.apply(shape, op); }, actor, label, kind);
}

uint64_t InkLabelController::applyMask(cv::Point origin, const cv::Mat_<uint8_t>& mask, Actor actor)
{
    return applyMask(origin, mask, currentOp(), actor, QStringLiteral("mask"));
}

uint64_t InkLabelController::applyMask(cv::Point origin, const cv::Mat_<uint8_t>& mask, PaintOp op, Actor actor, const QString& label)
{
    return applyEdit([&](EditTxn& txn) { return txn.applyMask(origin, mask, op); }, actor, label, QStringLiteral("mask"));
}

uint64_t InkLabelController::applyShapes(const std::vector<Shape>& shapes, PaintOp op, Actor actor, const QString& label)
{
    const QString kind = !shapes.empty() && shapes.front().kind == Shape::Kind::Polygon ? QStringLiteral("polygon") : QStringLiteral("stroke");
    return applyEdit(
        [&](EditTxn& txn) {
            uint64_t changed = 0;
            for (const Shape& shape : shapes)
                changed += txn.apply(shape, op);
            return changed;
        },
        actor, label, kind);
}

InkLabelController::ReadOutcome InkLabelController::readRect(const cv::Rect& rect, ReadFormat format, const fs::path& path) const
{
    ReadOutcome out;
    if (!_session || _session->info.locked) {
        out.error = tr("No readable ink label session is open.");
        return out;
    }
    if (rect.width > kMaxAssistWindow || rect.height > kMaxAssistWindow) {
        out.error = tr("The rectangle is larger than %1 x %1 label px.").arg(kMaxAssistWindow);
        return out;
    }
    out.rect = rect & cv::Rect(cv::Point(0, 0), _session->info.canvas.size);
    if (out.rect.empty()) {
        out.error = tr("The rectangle lies outside the %1 x %2 px label canvas.")
                        .arg(_session->info.canvas.size.width)
                        .arg(_session->info.canvas.size.height);
        return out;
    }
    cv::Mat_<uint8_t> values;
    _session->raster.readRect(out.rect, values);
    std::array<uint64_t, 8> byValue{};
    for (int y = 0; y < values.rows; ++y) {
        const uint8_t* row = values.ptr(y);
        for (int x = 0; x < values.cols; ++x)
            ++byValue[row[x] & kValueMask];
    }
    out.counts = LabelCounts::from(byValue);
    cv::Mat_<uint8_t> image;
    if (format == ReadFormat::Values) {
        image = values;
    } else {
        const uint8_t bit = format == ReadFormat::Ink ? kInk : format == ReadFormat::Supervision ? kSupervision : kValidation;
        cv::bitwise_and(values, cv::Scalar(bit), image);
        cv::compare(image, cv::Scalar(0), image, cv::CMP_NE);
    }
    try {
        if (!cv::imwrite(path.string(), image)) {
            out.error = tr("Cannot write %1").arg(qpath(path));
            return out;
        }
    } catch (const cv::Exception& e) {
        out.error = tr("Cannot write %1: %2").arg(qpath(path), QString::fromUtf8(e.what()));
        return out;
    }
    out.ok = true;
    return out;
}

uint64_t InkLabelController::applyValues(cv::Point origin, const cv::Mat_<uint8_t>& values, Actor actor, const QString& label)
{
    if (values.empty())
        return 0;
    return applyEdit(
        [&](EditTxn& txn) {
            // Each bit set where the image has it and cleared where it does
            // not, all inside one transaction.
            uint64_t changed = 0;
            cv::Mat_<uint8_t> where;
            for (const uint8_t bit : {kInk, kSupervision, kValidation}) {
                cv::bitwise_and(values, cv::Scalar(bit), where);
                cv::compare(where, cv::Scalar(0), where, cv::CMP_NE);
                changed += txn.applyMask(origin, where, PaintOp{bit, 0});
                cv::bitwise_not(where, where);
                changed += txn.applyMask(origin, where, PaintOp{0, bit});
            }
            return changed;
        },
        actor, label, QStringLiteral("values"));
}

bool InkLabelController::undo(Actor actor, bool force)
{
    if (!_session || _session->info.locked)
        return false;
    finishTransaction();
    const HistoryEntry* top = _session->history.top();
    if (!top)
        return false;
    if (!force && top->actor != actor)
        return false;
    const cv::Rect rect = changedRect(_session->raster, *top);
    if (!_session->history.undo(_session->raster))
        return false;
    _session->lastPoint.reset();
    notifyRasterChanged(rect);
    notifyHistory();
    return true;
}

bool InkLabelController::redo(Actor)
{
    if (!_session || _session->info.locked)
        return false;
    finishTransaction();
    const HistoryEntry* next = _session->history.nextRedo();
    if (!next)
        return false;
    const cv::Rect rect = changedRect(_session->raster, *next);
    if (!_session->history.redo(_session->raster))
        return false;
    notifyRasterChanged(rect);
    notifyHistory();
    return true;
}

// ---------------------------------------------------------------------------
// Saving
// ---------------------------------------------------------------------------

void InkLabelController::refreshGeometrySha()
{
    if (!_session || !_session->geometryStale)
        return;
    _session->geometryStale = false;
    if (auto surface = _session->surface.lock())
        _session->geometrySha = geometrySha256(*surface);
}

SaveRequest InkLabelController::makeSaveRequest(Target target, bool overwriteExternalEdits)
{
    refreshGeometrySha();
    SaveRequest req;
    req.snapshot = _session->raster.snapshot();
    req.paths = _session->info.paths;
    req.target = target;
    req.overwriteExternalEdits = overwriteExternalEdits;
    req.expectedSha256 = _session->expectedSha;
    const auto counts = _session->raster.counts();
    const bool anyValidation = counts[4] + counts[5] + counts[6] + counts[7] > 0;
    // A numbered version always has all three kinds, so the trainer's
    // newest-version pick per kind cannot mix versions.
    req.writeValidation = anyValidation || _session->validationFileExists || _session->info.paths.version > 1;

    Sidecar& s = req.sidecar;
    s.canvas = canvasKeyFor(_session->info.canvas);
    s.segmentName = _session->info.segment;
    s.segmentUuid = _session->info.segmentUuid;
    s.segmentPath = _session->info.segmentDir.string();
    s.geometrySha256 = _session->geometrySha;
    s.stateSerial = _session->history.stateSerial();
    s.writeSeq = ++_session->writeSeq;
    s.savedBy = QStringLiteral("VC3D %1").arg(QCoreApplication::applicationVersion()).trimmed().toStdString();
    s.edits = _session->edits;
    s.provenance = _session->provenance;
    if (_deps.viewers) {
        if (auto* ink = _deps.viewers->inkDetectionOverlay(); ink && !ink->selectedPath().empty())
            s.predictionShown = ink->selectedPath().string();
    }
    return req;
}

QFuture<SaveOutcome> InkLabelController::startWrite(SaveRequest request)
{
    const QString storeKey = qpath(request.paths.dir);
    const Target target = request.target;
    const int version = request.paths.version;
    const uint64_t generation = _session ? _session->generation : 0;
    std::shared_ptr<QLockFile> lock = _session ? _session->lock : nullptr;
    const uint64_t id = _nextWriteId++;
    QFuture<SaveOutcome> future = QtConcurrent::run([request = std::move(request)] {
        try {
            return saveLabels(request);
        } catch (const std::exception& e) {
            return failedOutcome(QString::fromUtf8(e.what()));
        }
    });
    _inFlight.insert(storeKey, InFlight{id, target, future, lock});
    if (target == Target::Autosave)
        _checkpointedThisRun.insert(storeKey);

    auto* watcher = new QFutureWatcher<SaveOutcome>(this);
    connect(watcher, &QFutureWatcherBase::finished, this, [this, watcher, storeKey, id, generation, lock, target, version]() {
        watcher->deleteLater();
        const SaveOutcome outcome = watcher->result();
        if (auto it = _inFlight.find(storeKey); it != _inFlight.end() && it->id == id)
            _inFlight.erase(it);
        if (_session && _session->generation == generation && outcome.ok) {
            if (target == Target::Trainer) {
                // The save deleted the checkpoint: a later open of this store
                // must not restore it without asking.
                _checkpointedThisRun.remove(storeKey);
                if (_session->info.paths.version != version) {
                    // Saved as a new version: the session moves to it, and
                    // the old version's checkpoint is superseded.
                    removeTarget(_session->info.paths, Target::Autosave);
                    _checkpointedThisRun.remove(_session->storeKey());
                    _session->info.paths.version = version;
                    _session->expectedSha.clear();
                }
                _session->history.markClean(outcome.stateSerial);
                for (const auto& [kind, sha] : outcome.sha256)
                    _session->expectedSha[kind] = sha;
                if (outcome.sha256.count(LabelKind::Validation))
                    _session->validationFileExists = true;
                _lastSave = LastSave{QDateTime::currentDateTimeUtc(), outcome.files, outcome.sha256};
                statusMessage(tr("Saved ink labels for %1 (%2 ms)")
                                  .arg(QString::fromStdString(_session->info.segment))
                                  .arg(qRound(outcome.ms)),
                              4000);
            } else {
                _lastAutosave = QDateTime::currentDateTimeUtc();
            }
            emit dirtyChanged(_session->history.dirty());
            emit sessionChanged();
        } else if (!outcome.ok && outcome.externallyEdited.isEmpty()) {
            statusMessage(tr("Ink labels were not saved: %1").arg(outcome.error), 8000);
        }
        emit saveFinished(outcome);
        Q_UNUSED(lock);
    });
    watcher->setFuture(future);
    return future;
}

QFuture<SaveOutcome> InkLabelController::save()
{
    return save(SaveOptions{});
}

QFuture<SaveOutcome> InkLabelController::save(const SaveOptions& options)
{
    QString why;
    if (!prepareEdit(&why))
        return readyOutcome(failedOutcome(why));
    finishTransaction();
    if (auto it = _inFlight.find(_session->storeKey()); it != _inFlight.end()) {
        if (it->target == Target::Trainer)
            return readyOutcome(failedOutcome(tr("A save of these labels is already running.")));
        it->future.waitForFinished();
    }
    SaveRequest request = makeSaveRequest(Target::Trainer, options.overwriteExternalEdits);
    _session->lastWriteMs = nowMs();
    return startWrite(std::move(request));
}

QFuture<SaveOutcome> InkLabelController::saveAsNewVersion()
{
    QString why;
    if (!canEdit(&why))
        return readyOutcome(failedOutcome(why));
    finishTransaction();
    if (auto it = _inFlight.find(_session->storeKey()); it != _inFlight.end()) {
        if (it->target == Target::Trainer)
            return readyOutcome(failedOutcome(tr("A save of these labels is already running.")));
        it->future.waitForFinished();
    }
    const std::vector<int> existing = versions();
    const int next = std::max(existing.empty() ? 1 : existing.back(), _session->info.paths.version) + 1;
    SaveRequest request = makeSaveRequest(Target::Trainer, false);
    request.paths.version = next;
    request.writeValidation = true;
    request.expectedSha256.clear();
    _session->lastWriteMs = nowMs();
    return startWrite(std::move(request));
}

std::vector<int> InkLabelController::versions() const
{
    if (!_session)
        return {};
    std::vector<int> list = listVersions(_session->info.paths.dir, _session->info.paths.prefix);
    if (std::find(list.begin(), list.end(), _session->info.paths.version) == list.end()) {
        list.push_back(_session->info.paths.version);
        std::sort(list.begin(), list.end());
    }
    return list;
}

InkLabelController::OpenOutcome InkLabelController::openVersion(int version)
{
    OpenOptions options;
    options.version = version;
    if (_session && _session->info.paths.version == version)
        return openForActiveSegment(options);
    // A different version of the same store: release this one first.
    closeSession();
    return openForActiveSegment(options);
}

InkLabelController::OpenOutcome InkLabelController::startNewVersion()
{
    OpenOutcome failure;
    const auto request = activeSegmentRequest(failure);
    if (!request)
        return failure;
    const StorePaths paths = storePathsFor(request->projectDir, request->segment);
    const std::vector<int> existing = listVersions(paths.dir, paths.prefix);
    OpenOptions options;
    options.version = (existing.empty() ? 1 : existing.back()) + 1;
    options.createVersion = true;
    closeSession();
    OpenOutcome out = openStore(*request, options);
    if (out.ok)
        statusMessage(tr("Ink labels: started empty version %1; Save writes it").arg(options.version), 5000);
    return out;
}

QJsonObject InkLabelController::datasetEntry() const
{
    if (!_session)
        return {};
    const fs::path projectDir = _session->info.paths.dir.parent_path().parent_path();
    return vc3d::ink_labels::datasetEntry(_session->info.paths, projectDir, versions());
}

QString InkLabelController::datasetEntryText() const
{
    if (!_session)
        return {};
    return QString::fromUtf8(QJsonDocument(datasetEntry()).toJson(QJsonDocument::Indented));
}

bool InkLabelController::importFile(const fs::path& path, const ImportOptions& options)
{
    return importFile(path, options, Actor::User);
}

bool InkLabelController::importFile(const fs::path& path, const ImportOptions& options, Actor actor)
{
    QString why;
    if (!canStartAssist(&why)) {
        refuseAssist(tr("import"), why);
        return false;
    }
    finishTransaction();
    const vc::RenderCanvas canvas = _session->info.canvas;
    const uint64_t serial = _session->history.nextSerial();
    const uint64_t baseSerial = _session->history.stateSerial();
    TileRaster snapshot = _session->raster.snapshot();
    const QString name = qpath(path.filename());
    startAssist([=, snapshot = std::move(snapshot)]() mutable {
        AssistResult r;
        r.what = tr("import %1").arg(name);
        r.editKind = QStringLiteral("import");
        r.importPath = qpath(path);
        r.importOptions = options;
        r.baseSerial = baseSerial;
        // The edit is made on the snapshot (its tiles are cloned on write) and
        // replayed onto the live raster if nothing changed meanwhile.
        EditTxn txn(snapshot, r.what.toStdString(), actor);
        r.import = importLabel(path, canvas, options, txn);
        if (r.import->ok)
            r.entry = txn.commit(serial);
        return r;
    });
    return true;
}

void InkLabelController::finishImport(const AssistResult& result)
{
    const ImportOutcome& outcome = *result.import;
    if (!outcome.ok) {
        if (!outcome.needsDecision)
            statusMessage(tr("Ink labels: %1 failed: %2").arg(result.what, outcome.error), 8000);
        emit assistChanged();
        emit importFinished(result.importPath, outcome, result.importOptions);
        return;
    }
    // Replay only onto the state the import read.
    bool current = result.entry && _session->history.stateSerial() == result.baseSerial && !_session->txn;
    if (current)
        for (const auto& change : result.entry->changes)
            current = current && _session->raster.tile(change.index).sameAs(change.before);
    if (!current) {
        ImportOutcome changed = outcome;
        changed.ok = false;
        changed.error = tr("The labels changed while %1 was read; import it again.").arg(QFileInfo(result.importPath).fileName());
        statusMessage(changed.error, 6000);
        emit assistChanged();
        emit importFinished(result.importPath, changed, result.importOptions);
        return;
    }
    // The first edit of a new store takes its folder and lock.
    QString why;
    if (!prepareEdit(&why)) {
        ImportOutcome refused = outcome;
        refused.ok = false;
        refused.error = why;
        statusMessage(why, 6000);
        emit assistChanged();
        emit importFinished(result.importPath, refused, result.importOptions);
        return;
    }
    cv::Rect rect;
    if (!result.entry->changes.empty()) {
        for (const auto& change : result.entry->changes) {
            _session->raster.setTile(change.index, change.after);
            const cv::Rect t = _session->raster.tileRect(change.index);
            rect = rect.empty() ? t : (rect | t);
        }
        HistoryEntry entry = *result.entry;
        const QString who = actorKey(entry.actor);
        QJsonObject byActor = _session->edits.value(who).toObject();
        byActor[QStringLiteral("import")] = byActor.value(QStringLiteral("import")).toInt() + 1;
        _session->edits[who] = byActor;
        _session->history.push(std::move(entry));
        _session->lastCommitMs = nowMs();
        _session->lastPoint.reset();
    }
    QJsonObject record;
    record["kind"] = QStringLiteral("import");
    record["source"] = result.importPath;
    record["format"] = QString::fromLatin1(formatName(outcome.format));
    record["label"] = QString::fromLatin1(kindName(result.importOptions.kind));
    record["source_shape"] = QJsonArray{outcome.sourceShape.width, outcome.sourceShape.height};
    record["resampled_from"] = outcome.resampledFrom;
    record["pixels"] = double(outcome.setPixels);
    record["at_utc"] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
    _session->provenance.append(record);
    while (_session->provenance.size() > kMaxProvenanceEntries)
        _session->provenance.removeFirst();
    notifyRasterChanged(rect);
    notifyHistory();
    for (const QString& warning : outcome.warnings)
        statusMessage(warning, 6000);
    statusMessage(tr("Ink labels: imported %1 (%2 px %3)")
                      .arg(QFileInfo(result.importPath).fileName())
                      .arg(outcome.setPixels)
                      .arg(QString::fromLatin1(kindName(result.importOptions.kind))),
                  5000);
    emit assistChanged();
    emit importFinished(result.importPath, outcome, result.importOptions);
}

void InkLabelController::saveInteractive()
{
    QString why;
    if (!canEdit(&why)) {
        statusMessage(why, 4000);
        return;
    }
    if (!_session->history.dirty() && _lastSave) {
        statusMessage(tr("Ink labels are already saved"), 2000);
        return;
    }
    QFuture<SaveOutcome> future = save();
    auto* watcher = new QFutureWatcher<SaveOutcome>(this);
    const uint64_t generation = _session->generation;
    connect(watcher, &QFutureWatcherBase::finished, this, [this, watcher, generation]() {
        watcher->deleteLater();
        const SaveOutcome outcome = watcher->result();
        if (outcome.externallyEdited.isEmpty() || !_session || _session->generation != generation)
            return;
        auto* box = new QMessageBox(QMessageBox::Warning,
                                    tr("Label files changed outside VC3D"),
                                    tr("%1 changed since VC3D loaded or saved it (an edit in another program, or an interrupted "
                                       "save). Overwrite it? The current file is kept as .prev.")
                                        .arg(outcome.externallyEdited.join(QStringLiteral(", "))),
                                    QMessageBox::Yes | QMessageBox::Cancel,
                                    _deps.shortcutHost);
        box->setAttribute(Qt::WA_DeleteOnClose);
        connect(box, &QMessageBox::finished, this, [this, generation](int result) {
            if (result == QMessageBox::Yes && _session && _session->generation == generation)
                save(SaveOptions{true});
        });
        box->open();
    });
    watcher->setFuture(future);
}

void InkLabelController::startCheckpoint()
{
    if (!_session || !_session->writable())
        return;
    finishTransaction();
    if (_inFlight.contains(_session->storeKey()))
        return;
    SaveRequest request = makeSaveRequest(Target::Autosave, false);
    _session->lastWriteMs = nowMs();
    startWrite(std::move(request));
}

void InkLabelController::onAutosaveTick()
{
    if (!_session || !_session->writable() || !_session->history.dirty() || _session->txn)
        return;
    if (_inFlight.contains(_session->storeKey()))
        return;
    const qint64 now = nowMs();
    if (now - _session->lastCommitMs < kAutosaveQuietMs || now - _session->lastWriteMs < kAutosaveIntervalMs)
        return;
    startCheckpoint();
}

bool InkLabelController::readyToClose(std::function<void()> retryClose)
{
    if (_closeBypass) {
        // One close only: if another guard cancels it, the next close asks
        // again.
        _closeBypass = false;
        return true;
    }
    finishTransaction();
    const bool dirty = _session && _session->writable() && _session->history.dirty();
    if (!dirty) {
        if (_inFlight.isEmpty())
            return true;
        // Let pending writes land, then try again.
        QTimer::singleShot(100, this, [retryClose]() { retryClose(); });
        return false;
    }
    auto* box = new QMessageBox(QMessageBox::Question,
                                tr("Unsaved ink labels"),
                                tr("Save the ink labels for %1 before closing?").arg(QString::fromStdString(_session->info.segment)),
                                QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel,
                                _deps.shortcutHost);
    box->setObjectName(QStringLiteral("inkLabelCloseGuard"));
    box->setAttribute(Qt::WA_DeleteOnClose);
    connect(box, &QMessageBox::finished, this, [this, retryClose](int result) {
        if (result == QMessageBox::Save) {
            QFuture<SaveOutcome> future = save();
            auto* watcher = new QFutureWatcher<SaveOutcome>(this);
            connect(watcher, &QFutureWatcherBase::finished, this, [this, watcher, retryClose]() {
                watcher->deleteLater();
                if (watcher->result().ok) {
                    _closeBypass = true;
                    retryClose();
                } else {
                    statusMessage(tr("Ink labels were not saved: %1").arg(watcher->result().error), 8000);
                }
            });
            watcher->setFuture(future);
        } else if (result == QMessageBox::Discard) {
            if (_session) {
                removeTarget(_session->info.paths, Target::Autosave);
                _checkpointedThisRun.remove(_session->storeKey());
                _session->history.markClean(_session->history.stateSerial());
            }
            _closeBypass = true;
            retryClose();
        }
    });
    box->open();
    return false;
}

// ---------------------------------------------------------------------------
// Viewer binding and mapping
// ---------------------------------------------------------------------------

void InkLabelController::bindViewer(VolumeViewerBase* viewer)
{
    if (!viewer || std::any_of(_bound.begin(), _bound.end(), [viewer](const BoundViewer& b) { return b.viewer == viewer; }))
        return;
    BoundViewer bound;
    bound.viewer = viewer;
    if (auto* view = viewer->graphicsView()) {
        bound.view = view;
        bound.viewport = view->viewport();
        view->installEventFilter(this);
        view->viewport()->installEventFilter(this);
    }
    _bound.push_back(bound);
}

void InkLabelController::unbindViewer(VolumeViewerBase* viewer)
{
    for (auto it = _bound.begin(); it != _bound.end(); ++it) {
        if (it->viewer != viewer)
            continue;
        if (it->viewport)
            it->viewport->removeEventFilter(this);
        if (it->view)
            it->view->removeEventFilter(this);
        if (it->cursor)
            it->cursor->deleteLater();
        _bound.erase(it);
        return;
    }
}

InkLabelController::BoundViewer* InkLabelController::boundFor(QObject* watched)
{
    for (auto& bound : _bound)
        if (watched && (watched == bound.viewport || watched == bound.view))
            return &bound;
    return nullptr;
}

bool InkLabelController::viewerShowsSession(VolumeViewerBase* viewer) const
{
    QuadSurface* surface = sessionSurface();
    return viewer && surface && !_session->info.locked && viewer->currentSurface() == surface && !isVolumetricCamera(viewer);
}

std::optional<QPointF> InkLabelController::labelToScene(VolumeViewerBase* viewer, cv::Point2d labelPx) const
{
    QuadSurface* surface = sessionSurface();
    if (!viewer || !surface || viewer->currentSurface() != surface)
        return std::nullopt;
    const cv::Vec2d grid = _session->info.canvas.canvasToGrid({labelPx.x, labelPx.y});
    const cv::Vec2d surf = surface->gridToSurface(grid);
    if (!std::isfinite(surf[0]) || !std::isfinite(surf[1]))
        return std::nullopt;
    return viewer->surfaceCoordsToScene(float(surf[0]), float(surf[1]));
}

std::optional<cv::Point2d> InkLabelController::sceneToLabel(VolumeViewerBase* viewer, const QPointF& scenePos) const
{
    QuadSurface* surface = sessionSurface();
    if (!viewer || !surface || viewer->currentSurface() != surface)
        return std::nullopt;
    const cv::Vec2f surf = viewer->sceneToSurfaceCoords(scenePos);
    const cv::Vec2d grid = surface->surfaceToGrid({double(surf[0]), double(surf[1])});
    const cv::Vec2d p = _session->info.canvas.gridToCanvas(grid);
    if (!std::isfinite(p[0]) || !std::isfinite(p[1]))
        return std::nullopt;
    return cv::Point2d(p[0], p[1]);
}

double InkLabelController::devicePxPerLabelPx(VolumeViewerBase* viewer, cv::Point2d labelPx) const
{
    auto* view = viewer ? viewer->graphicsView() : nullptr;
    const auto a = labelToScene(viewer, labelPx);
    const auto b = labelToScene(viewer, labelPx + cv::Point2d(1.0, 0.0));
    if (!view || !a || !b)
        return 1.0;
    const QTransform t = view->viewportTransform();
    const QPointF d = t.map(*b) - t.map(*a);
    const double len = std::hypot(d.x(), d.y());
    return len > 1e-9 ? len : 1.0;
}

void InkLabelController::updateCursor(BoundViewer& bound, const QPointF& devicePos)
{
    auto* viewport = qobject_cast<QWidget*>(bound.viewport.data());
    if (!viewport)
        return;
    if (!bound.cursor) {
        bound.cursor = new SpiralBrushCursorWidget(viewport);
        bound.cursor->setGeometry(viewport->rect());
        bound.cursor->show();
    }
    bound.cursor->raise();
    const bool visible = _labelMode && _tool == Tool::Brush && _session && viewerShowsSession(bound.viewer);
    double d = 1.0;
    if (visible) {
        auto* view = bound.viewer->graphicsView();
        const QPointF scene = view->viewportTransform().inverted().map(devicePos);
        if (const auto label = sceneToLabel(bound.viewer, scene))
            d = devicePxPerLabelPx(bound.viewer, *label);
    }
    const int diameter = std::max(2, int(std::lround(2.0 * _radius * d)));
    bound.cursor->setCursorState(devicePos, diameter, visible, false, QColor(50, 255, 215), classColor());
}

void InkLabelController::hideCursors()
{
    for (auto& bound : _bound)
        if (bound.cursor)
            bound.cursor->setCursorState(QPointF(), 0, false, false);
}

// ---------------------------------------------------------------------------
// Mouse and keys
// ---------------------------------------------------------------------------

bool InkLabelController::handlePress(cv::Point2d labelPx, Qt::KeyboardModifiers modifiers, double deviceScale)
{
    if (_deps.otherSurfaceToolActive && _deps.otherSurfaceToolActive()) {
        setLabelModeEnabled(false);
        statusMessage(tr("Ink label mode off: another segment tool is active"), 4000);
        return false;
    }
    QString why;
    if (!canEdit(&why)) {
        statusMessage(why, 4000);
        return true;
    }
    if (_suggestions) {
        // While suggestions are shown a click picks or unpicks one.
        toggleSuggestionAt(labelPx);
        return true;
    }
    if (_tool == Tool::Fill) {
        fillAt(labelPx, _pressBound);
        return true;
    }
    _leftDown = true;
    if (_tool == Tool::Brush) {
        beginStroke(labelPx, modifiers.testFlag(Qt::ShiftModifier), Actor::User);
        return true;
    }
    if (_polygon.empty()) {
        _lassoCandidate = true;
        _lasso = false;
        _lastLassoPoint = labelPx;
        _lassoMinStep = kLassoStepDevicePx / std::max(deviceScale, 1e-6);
    } else {
        _lassoCandidate = false;
    }
    _polygon.push_back(labelPx);
    updatePolygonPreview();
    return true;
}

void InkLabelController::handleDrag(cv::Point2d labelPx, double deviceScale)
{
    if (_tool == Tool::Brush) {
        extendStroke(labelPx);
        return;
    }
    if (!_lassoCandidate && !_lasso)
        return;
    _lassoMinStep = kLassoStepDevicePx / std::max(deviceScale, 1e-6);
    if (cv::norm(labelPx - _lastLassoPoint) < _lassoMinStep)
        return;
    _lasso = true;
    _polygon.push_back(labelPx);
    _lastLassoPoint = labelPx;
    updatePolygonPreview();
}

void InkLabelController::handleRelease()
{
    _leftDown = false;
    if (_tool == Tool::Brush) {
        endStroke();
        return;
    }
    if (_lasso) {
        if (_polygon.size() >= 3)
            closePolygon();
        else
            cancelPolygon();
    }
    _lassoCandidate = false;
    _lasso = false;
}

void InkLabelController::closePolygon()
{
    if (_polygon.size() >= 3) {
        const Shape shape{Shape::Kind::Polygon, _polygon, 0.0};
        _polygon.clear();
        applyEdit([&](EditTxn& txn) { return txn.apply(shape, currentOp()); }, Actor::User, QStringLiteral("polygon"),
                  QStringLiteral("polygon"));
    }
    cancelPolygon();
}

void InkLabelController::cancelPolygon()
{
    _polygon.clear();
    _lassoCandidate = false;
    _lasso = false;
    updatePolygonPreview();
}

void InkLabelController::updatePolygonPreview(std::optional<cv::Point2d> hover)
{
    std::vector<cv::Point2d> points = _polygon;
    if (hover && !points.empty() && !_lasso)
        points.push_back(*hover);
    _overlay->setPolygonPreview(std::move(points), classColor());
    // Enter, Backspace and Esc follow whether a polygon is open.
    updateShortcutStates();
}

bool InkLabelController::eventFilter(QObject* watched, QEvent* event)
{
    BoundViewer* bound = boundFor(watched);
    if (!bound || !bound->viewer || !event)
        return false;
    const QEvent::Type type = event->type();
    if (type == QEvent::FocusOut || type == QEvent::WindowDeactivate || type == QEvent::Hide) {
        if (strokeOpen())
            endStroke();
        _leftDown = false;
        return false;
    }
    auto* viewport = qobject_cast<QWidget*>(bound->viewport.data());
    if (type == QEvent::Resize && watched == bound->viewport && bound->cursor && viewport) {
        bound->cursor->setGeometry(viewport->rect());
        return false;
    }
    if (type == QEvent::Leave) {
        if (bound->cursor)
            bound->cursor->setCursorState(QPointF(), 0, false, false);
        return false;
    }
    if (!_labelMode || !_session)
        return false;
    VolumeViewerBase* viewer = bound->viewer;
    if (!viewerShowsSession(viewer)) {
        if (bound->cursor)
            bound->cursor->setCursorState(QPointF(), 0, false, false);
        return false;
    }
    auto* view = viewer->graphicsView();
    const bool onViewport = watched == bound->viewport;
    const auto devicePosition = [&](const QPointF& position, const QPointF& globalPosition) {
        return onViewport || !viewport ? position : QPointF(viewport->mapFromGlobal(globalPosition.toPoint()));
    };
    const auto labelAt = [&](const QPointF& devicePos) -> std::optional<cv::Point2d> {
        return sceneToLabel(viewer, view->viewportTransform().inverted().map(devicePos));
    };

    switch (type) {
        case QEvent::Wheel: {
            auto* wheel = static_cast<QWheelEvent*>(event);
            if (wheel->modifiers() != Qt::ControlModifier)
                return false;
            const double steps = wheel->angleDelta().y() / 120.0;
            if (steps != 0.0) {
                setRadius(_radius * std::pow(1.1, steps));
                updateCursor(*bound, devicePosition(wheel->position(), wheel->globalPosition()));
                statusMessage(tr("Ink label brush radius %1 px (voxels)").arg(QString::number(_radius, 'g', 4)), 1500);
            }
            return true;
        }
        case QEvent::MouseMove: {
            auto* mouse = static_cast<QMouseEvent*>(event);
            const QPointF devicePos = devicePosition(mouse->position(), mouse->globalPosition());
            updateCursor(*bound, devicePos);
            const auto label = labelAt(devicePos);
            if (_leftDown && mouse->buttons().testFlag(Qt::LeftButton)) {
                if (label)
                    handleDrag(*label, devicePxPerLabelPx(viewer, *label));
                return true;
            }
            if (!_polygon.empty() && label)
                updatePolygonPreview(*label);
            return false;
        }
        case QEvent::MouseButtonPress:
        case QEvent::MouseButtonDblClick: {
            auto* mouse = static_cast<QMouseEvent*>(event);
            if (mouse->button() != Qt::LeftButton)
                return false;
            const QPointF devicePos = devicePosition(mouse->position(), mouse->globalPosition());
            const auto label = labelAt(devicePos);
            if (!label)
                return false;
            if (type == QEvent::MouseButtonDblClick && _tool == Tool::Polygon) {
                _leftDown = false;
                closePolygon();
                return true;
            }
            _pressBound = visibleLabelRect(viewer);
            return handlePress(*label, mouse->modifiers(), devicePxPerLabelPx(viewer, *label));
        }
        case QEvent::MouseButtonRelease: {
            auto* mouse = static_cast<QMouseEvent*>(event);
            if (mouse->button() != Qt::LeftButton || !_leftDown)
                return false;
            handleRelease();
            return true;
        }
        case QEvent::KeyPress:
            // Normally the window shortcuts take these keys first.
            return handleEditKey(static_cast<QKeyEvent*>(event)->key());
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// Assisted labeling
// ---------------------------------------------------------------------------

void InkLabelController::setAssistSettings(const AssistSettings& settings)
{
    AssistSettings s = settings;
    s.connectivity = s.connectivity == 8 ? 8 : 4;
    s.median = s.median <= 1 ? 1 : (s.median >= 5 ? 5 : 3);
    s.minArea = std::max(0, s.minArea);
    s.cleanupSize = std::clamp(s.cleanupSize, 1, 256);
    if (s == _assist)
        return;
    _assist = s;
    emit toolStateChanged();
}

std::optional<PredictionGuide> InkLabelController::predictionGuide() const
{
#ifdef VC_TESTING
    if (_guideForTesting)
        return _guideForTesting;
#endif
    auto* ink = _deps.viewers ? _deps.viewers->inkDetectionOverlay() : nullptr;
    if (!ink || ink->selectedPath().empty() || !ink->selectedIsSingleChannel() || ink->selectedScalar().empty())
        return std::nullopt;
    PredictionGuide guide;
    guide.scalar = ink->selectedScalar();
    guide.hflip = ink->horizontalFlip();
    guide.vflip = ink->verticalFlip();
    guide.source = ink->selectedPath().string();
    return guide;
}

std::optional<cv::Rect> InkLabelController::visibleLabelRect(VolumeViewerBase* viewer) const
{
    if (!viewer || !_session || !viewerShowsSession(viewer))
        return std::nullopt;
    auto* view = viewer->graphicsView();
    if (!view || !view->viewport())
        return std::nullopt;
    const QRect device = view->viewport()->rect();
    const QTransform toScene = view->viewportTransform().inverted();
    double x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    bool first = true;
    for (const QPoint& corner : {device.topLeft(), device.topRight(), device.bottomLeft(), device.bottomRight()}) {
        const auto label = sceneToLabel(viewer, toScene.map(QPointF(corner)));
        if (!label)
            return std::nullopt;
        x0 = first ? label->x : std::min(x0, label->x);
        x1 = first ? label->x : std::max(x1, label->x);
        y0 = first ? label->y : std::min(y0, label->y);
        y1 = first ? label->y : std::max(y1, label->y);
        first = false;
    }
    const cv::Rect rect(cv::Point(int(std::floor(x0)), int(std::floor(y0))), cv::Point(int(std::ceil(x1)) + 1, int(std::ceil(y1)) + 1));
    const cv::Rect clipped = rect & cv::Rect(cv::Point(0, 0), _session->info.canvas.size);
    if (clipped.empty())
        return std::nullopt;
    return clipped;
}

cv::Rect InkLabelController::defaultBound() const
{
    for (const auto& bound : _bound)
        if (const auto rect = visibleLabelRect(bound.viewer))
            return *rect;
    return _session ? cv::Rect(cv::Point(0, 0), _session->info.canvas.size) : cv::Rect();
}

bool InkLabelController::canStartAssist(QString* why) const
{
    if (!canEdit(why))
        return false;
    if (_assistRunning) {
        if (why)
            *why = tr("An ink label assist is still running; press Esc to drop it.");
        return false;
    }
    return true;
}

void InkLabelController::refuseAssist(const QString& what, const QString& why)
{
    _lastAssistReport = AssistReport{what, 0, false, why};
    statusMessage(why, 4000);
}

InkLabelController::AssistReport InkLabelController::runAssistNow(const std::function<bool()>& start)
{
    _lastAssistReport = AssistReport{};
    _assistNow = true;
    const bool started = start();
    _assistNow = false;
    if (!started && _lastAssistReport.error.isEmpty())
        _lastAssistReport.error = tr("The assist was refused.");
    return _lastAssistReport;
}

void InkLabelController::startAssist(std::function<AssistResult()> work)
{
    const uint64_t generation = ++_assistGeneration;
    const uint64_t sessionGeneration = _session ? _session->generation : 0;
    if (_assistNow) {
        // Same work and same landing as the worker path, on this thread.
        AssistResult result;
        try {
            result = work();
        } catch (const std::exception& e) {
            result.error = QString::fromUtf8(e.what());
        }
        finishAssist(result);
        return;
    }
    _assistRunning = true;
    auto result = std::make_shared<AssistResult>();
    _assistFuture = QtConcurrent::run([work = std::move(work), result] {
        try {
            *result = work();
        } catch (const std::exception& e) {
            result->error = QString::fromUtf8(e.what());
        }
    });
    auto* watcher = new QFutureWatcher<void>(this);
    connect(watcher, &QFutureWatcherBase::finished, this, [this, watcher, result, generation, sessionGeneration]() {
        watcher->deleteLater();
        if (generation != _assistGeneration)
            return;  // dropped (Esc, a newer assist, or a session change)
        _assistRunning = false;
        if (!_session || _session->generation != sessionGeneration) {
            emit assistChanged();
            return;
        }
        finishAssist(*result);
    });
    watcher->setFuture(_assistFuture);
    emit assistChanged();
}

void InkLabelController::finishAssist(const AssistResult& result)
{
    _lastAssistReport = AssistReport{result.what, 0, result.mask && result.mask->reachedBound, result.error};
    if (!result.error.isEmpty()) {
        statusMessage(tr("Ink labels: %1 failed: %2").arg(result.what, result.error), 6000);
        emit assistChanged();
        return;
    }
    if (result.import) {
        finishImport(result);
        return;
    }
    if (result.suggestions) {
        _suggestions = result.suggestions;
        _selectedSuggestions.clear();
        refreshSuggestionPreview();
        statusMessage(_suggestions->empty() ? tr("No suggestions above the threshold here")
                                            : tr("%n ink suggestion(s): click to pick, Shift+U picks all, Enter accepts, Esc discards", nullptr,
                                                 int(_suggestions->items.size())),
                      5000);
        emit assistChanged();
        emit assistFinished(result.what, 0);
        return;
    }
    uint64_t changed = 0;
    if (result.mask && result.mask->count() > 0) {
        changed = applyEdit([&](EditTxn& txn) { return txn.applyMask(result.mask->origin, result.mask->mask, result.op); },
                            _assistNow ? Actor::Agent : Actor::User, result.what, result.editKind);
    }
    _lastAssistReport.changedPixels = changed;
    QString message = changed > 0 ? tr("Ink labels: %1 changed %2 px").arg(result.what).arg(changed)
                                  : tr("Ink labels: %1 changed nothing").arg(result.what);
    if (result.mask && result.mask->reachedBound)
        message += tr("; it stopped at the edge of the view, zoom out to fill further");
    statusMessage(message, result.mask && result.mask->reachedBound ? 6000 : 3000);
    emit assistChanged();
    emit assistFinished(result.what, changed);
}

bool InkLabelController::fillAt(cv::Point2d labelPx, std::optional<cv::Rect> bound)
{
    return fillAt(labelPx, bound, currentOp());
}

bool InkLabelController::fillAt(cv::Point2d labelPx, std::optional<cv::Rect> bound, PaintOp op)
{
    QString why;
    if (!canStartAssist(&why)) {
        refuseAssist(tr("fill"), why);
        return false;
    }
    finishTransaction();
    std::optional<PredictionGuide> guide;
    if (_assist.guide == GuideKind::Prediction) {
        guide = predictionGuide();
        if (!guide) {
            refuseAssist(tr("fill"), tr("Select a single-channel ink prediction in the Ink Detection dock to fill from it."));
            return false;
        }
    }
    const cv::Point seed(int(std::floor(labelPx.x)), int(std::floor(labelPx.y)));
    FloodOptions flood;
    flood.connectivity = _assist.connectivity;
    flood.threshold = _assist.threshold;
    flood.above = _assist.above;
    flood.unlabeledOnly = _assist.unlabeledOnly;
    const GuideKind kind = _assist.guide;
    const vc::RenderCanvas canvas = _session->info.canvas;
    const cv::Rect area = bound.value_or(defaultBound());
    TileRaster snapshot = _session->raster.snapshot();
    const QString what = tr("fill");
    startAssist([=, snapshot = std::move(snapshot)]() {
        AssistResult r;
        r.what = what;
        r.editKind = QStringLiteral("fill");
        r.op = op;
        r.mask = floodRegion(snapshot, guide ? &*guide : nullptr, kind, canvas, seed, area, flood);
        return r;
    });
    return true;
}

bool InkLabelController::suggest(std::optional<cv::Rect> bound)
{
    QString why;
    if (!canStartAssist(&why)) {
        refuseAssist(tr("suggestions"), why);
        return false;
    }
    finishTransaction();
    std::optional<PredictionGuide> guide = predictionGuide();
    if (!guide) {
        refuseAssist(tr("suggestions"), tr("Select a single-channel ink prediction in the Ink Detection dock to get suggestions."));
        return false;
    }
    _suggestions.reset();
    _selectedSuggestions.clear();
    refreshSuggestionPreview();
    const vc::RenderCanvas canvas = _session->info.canvas;
    const cv::Rect area = bound.value_or(defaultBound());
    const AssistSettings s = _assist;
    const QString what = tr("suggestions");
    // Suggestions leave out pixels already labeled ink or background.
    TileRaster snapshot = _session->raster.snapshot();
    startAssist([=, g = std::move(*guide), snapshot = std::move(snapshot)]() {
        AssistResult r;
        r.what = what;
        r.suggestions = suggestComponents(g, canvas, area, s.threshold, s.above, s.median, s.minArea, &snapshot);
        return r;
    });
    return true;
}

bool InkLabelController::runCleanup(Cleanup kind, std::optional<cv::Rect> bound)
{
    QString why;
    if (!canStartAssist(&why)) {
        refuseAssist(QString::fromLatin1(cleanupName(kind)), why);
        return false;
    }
    finishTransaction();
    const cv::Rect area = bound.value_or(defaultBound());
    const int n = _assist.cleanupSize;
    TileRaster snapshot = _session->raster.snapshot();
    const QString what = QString::fromLatin1(cleanupName(kind));
    startAssist([=, snapshot = std::move(snapshot)]() {
        AssistResult r;
        r.what = what;
        r.editKind = QStringLiteral("cleanup");
        CleanupResult c = cleanup(snapshot, area, kind, n);
        r.op = c.op;
        r.mask = MaskResult{c.origin, std::move(c.mask)};
        return r;
    });
    return true;
}

bool InkLabelController::toggleSuggestionAt(cv::Point2d labelPx)
{
    if (!_suggestions)
        return false;
    const int id = _suggestions->idAt(cv::Point(int(std::floor(labelPx.x)), int(std::floor(labelPx.y))));
    if (id <= 0)
        return false;
    setSuggestionSelected(id, !_selectedSuggestions.count(id));
    return true;
}

void InkLabelController::setSuggestionSelected(int id, bool selected)
{
    if (!_suggestions || id <= 0 || id > int(_suggestions->items.size()))
        return;
    if (selected == bool(_selectedSuggestions.count(id)))
        return;
    if (selected)
        _selectedSuggestions.insert(id);
    else
        _selectedSuggestions.erase(id);
    refreshSuggestionPreview();
    emit assistChanged();
}

uint64_t InkLabelController::acceptSuggestions()
{
    return acceptSuggestions(Actor::User);
}

uint64_t InkLabelController::acceptSuggestions(Actor actor)
{
    if (!_suggestions)
        return 0;
    QString why;
    if (!canEdit(&why)) {
        statusMessage(why, 4000);
        return 0;
    }
    if (_selectedSuggestions.empty()) {
        // Keep them: Enter before picking must not throw the work away.
        statusMessage(tr("Nothing picked: click suggestions to pick them, Shift+U picks all, Esc discards."), 5000);
        return 0;
    }
    const SuggestionSet set = std::move(*_suggestions);
    const std::vector<int> ids(_selectedSuggestions.begin(), _selectedSuggestions.end());
    _suggestions.reset();
    _selectedSuggestions.clear();
    refreshSuggestionPreview();
    uint64_t changed = 0;
    if (!ids.empty()) {
        MaskResult mask = set.maskFor(ids);
        // Labels may have changed since the suggestions were made: never turn
        // a pixel the annotator labeled background into ink.
        dropSupervised(_session->raster, mask);
        changed = applyEdit([&](EditTxn& txn) { return txn.applyMask(mask.origin, mask.mask, paintOp(PaintClass::Ink, false, false)); },
                            actor, QStringLiteral("accept suggestions"), QStringLiteral("suggestion"));
        if (changed > 0 && _session) {
            QJsonObject entry;
            entry["kind"] = QStringLiteral("suggestion");
            entry["source"] = QString::fromStdString(set.source);
            entry["threshold"] = int(set.threshold);
            entry["above"] = set.above;
            entry["median"] = set.median;
            entry["min_area"] = set.minArea;
            entry["components"] = int(ids.size());
            entry["pixels"] = double(mask.count());
            entry["at_utc"] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
            _session->provenance.append(entry);
            while (_session->provenance.size() > kMaxProvenanceEntries)
                _session->provenance.removeFirst();
        }
    }
    statusMessage(tr("Ink labels: accepted %n suggestion(s), %1 px changed", nullptr, int(ids.size())).arg(changed), 4000);
    emit assistChanged();
    return changed;
}

void InkLabelController::pickAllSuggestions()
{
    if (!_suggestions || _suggestions->empty())
        return;
    for (const Suggestion& item : _suggestions->items)
        _selectedSuggestions.insert(item.id);
    refreshSuggestionPreview();
    emit assistChanged();
}

void InkLabelController::discardSuggestions()
{
    if (!_suggestions)
        return;
    _suggestions.reset();
    _selectedSuggestions.clear();
    refreshSuggestionPreview();
    emit assistChanged();
}

void InkLabelController::cancelAssist()
{
    const bool had = _assistRunning || _suggestions;
    ++_assistGeneration;
    _assistRunning = false;
    _suggestions.reset();
    _selectedSuggestions.clear();
    refreshSuggestionPreview();
    if (had)
        emit assistChanged();
}

void InkLabelController::refreshSuggestionPreview()
{
    if (_overlay)
        _overlay->setSuggestionPreview(_suggestions ? &*_suggestions : nullptr, _selectedSuggestions);
}

bool InkLabelController::handleAssistKey(int key)
{
    if (!_suggestions && !_assistRunning)
        return false;
    switch (key) {
        case Qt::Key_Return:
        case Qt::Key_Enter:
            if (!_suggestions)
                return false;
            acceptSuggestions();
            return true;
        case Qt::Key_Escape:
            cancelAssist();
            statusMessage(tr("Ink labels: assist dropped"), 2000);
            return true;
        default:
            return false;
    }
}

#ifdef VC_TESTING
InkLabelController::OpenOutcome InkLabelController::openForTesting(const vc::RenderCanvas& canvas,
                                                                   const fs::path& projectDir,
                                                                   const std::string& segment)
{
    return openForTesting(canvas, projectDir, segment, OpenOptions{});
}

InkLabelController::OpenOutcome InkLabelController::openForTesting(const vc::RenderCanvas& canvas,
                                                                   const fs::path& projectDir,
                                                                   const std::string& segment,
                                                                   const OpenOptions& options)
{
    OpenRequest request;
    request.canvas = canvas;
    request.projectDir = projectDir;
    request.segment = segment;
    return openStore(request, options);
}

bool InkLabelController::keyForTesting(int key)
{
    return handleEditKey(key);
}

bool InkLabelController::pressForTesting(cv::Point2d labelPx, Qt::KeyboardModifiers modifiers)
{
    if (!_labelMode || !_session)
        return false;
    return handlePress(labelPx, modifiers, 1.0);
}

void InkLabelController::releaseForTesting()
{
    if (_leftDown)
        handleRelease();
}

void InkLabelController::runAutosaveTickForTesting(bool ignoreTimers)
{
    if (ignoreTimers && _session) {
        _session->lastCommitMs = 0;
        _session->lastWriteMs = 0;
    }
    onAutosaveTick();
}
#endif
