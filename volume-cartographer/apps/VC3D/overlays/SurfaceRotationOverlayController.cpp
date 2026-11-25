#include "SurfaceRotationOverlayController.hpp"
#include "SurfaceRotationTransform.hpp"

#include "../CState.hpp"
#include "../volume_viewers/CVolumeViewerView.hpp"
#include "../ViewerManager.hpp"
#include "../volume_viewers/VolumeViewerBase.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <QApplication>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QFutureWatcher>
#include <QGraphicsProxyWidget>
#include <QGraphicsScene>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QPointer>
#include <QPushButton>
#include <QSignalBlocker>
#include <QStatusBar>
#include <QVBoxLayout>
#include <QWidget>
#include <QtConcurrent>

#include <algorithm>
#include <cmath>
#include <exception>

namespace
{
constexpr const char* kOverlayGroupKey = "surface_transform_rotate";
constexpr const char* kHorizontalFlipObjectName = "surfaceHorizontalFlipCheckBox";
constexpr const char* kApplyTransformObjectName = "surfaceTransformApplyButton";
constexpr double kPi = 3.14159265358979323846;

struct RotationSaveResult {
    std::shared_ptr<QuadSurface> surface;
    QString error;
};

vc3d::surface_rotation::TransformPersistenceCompatibility surfaceTransformPersistenceCompatibility(
    const std::shared_ptr<QuadSurface>& surface)
{
    if (!surface) {
        return {};
    }
    return {surface->meta.contains("components") && surface->meta["components"].is_array() &&
            surface->meta["components"].size() > 0};
}

QString transformPersistenceUnavailableTooltip(const vc3d::surface_rotation::TransformPersistenceCompatibility&)
{
    return QObject::tr(
        "Apply is unavailable because disconnected component ranges cannot yet be "
        "transformed safely.");
}

void warnTransformNotApplied()
{
    QMessageBox::information(
        nullptr,
        QObject::tr("Surface Transform Cancelled"),
        QObject::tr("The transform was not applied: the segment was deselected, reopened or saved again "
                    "while Rotate was open, or its stored revision could not be read. Open Rotate again "
                    "on the segment as it is now."));
}

class RotationDial final : public QWidget
{
public:
    explicit RotationDial(QWidget* parent = nullptr)
        : QWidget(parent)
    {
        setFixedSize(78, 78);
        setMouseTracking(true);
    }

    void setAngle(double angleDeg)
    {
        _angleDeg = normalizeAngle(angleDeg);
        update();
    }

    std::function<void(double)> angleChanged;

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);

        const QPointF c(width() * 0.5, height() * 0.5);
        constexpr double radius = 28.0;
        p.setPen(QPen(QColor(255, 255, 255, 210), 2.0));
        p.setBrush(QColor(20, 20, 20, 170));
        p.drawEllipse(c, radius, radius);

        const double radians = _angleDeg * kPi / 180.0;
        const QPointF handle(c.x() + std::cos(radians) * radius,
                             c.y() + std::sin(radians) * radius);
        p.setPen(QPen(QColor(40, 28, 0), 1.0));
        p.setBrush(QColor(255, 214, 42));
        p.drawEllipse(handle, 7.0, 7.0);
    }

    void mousePressEvent(QMouseEvent* event) override
    {
        if (event->button() != Qt::LeftButton) {
            return;
        }
        _dragging = true;
        updateFromPos(event->position());
        event->accept();
    }

    void mouseMoveEvent(QMouseEvent* event) override
    {
        if (!_dragging) {
            return;
        }
        updateFromPos(event->position());
        event->accept();
    }

    void mouseReleaseEvent(QMouseEvent* event) override
    {
        if (event->button() == Qt::LeftButton && _dragging) {
            updateFromPos(event->position());
            _dragging = false;
            event->accept();
        }
    }

private:
    static double normalizeAngle(double angleDeg)
    {
        while (angleDeg > 360.0) {
            angleDeg -= 360.0;
        }
        while (angleDeg < -360.0) {
            angleDeg += 360.0;
        }
        return angleDeg;
    }

    void updateFromPos(const QPointF& pos)
    {
        const QPointF c(width() * 0.5, height() * 0.5);
        const double angle = std::atan2(pos.y() - c.y(), pos.x() - c.x()) * 180.0 / kPi;
        setAngle(angle);
        if (angleChanged) {
            angleChanged(_angleDeg);
        }
    }

    double _angleDeg{0.0};
    bool _dragging{false};
};

class RotationWidget final : public QWidget
{
public:
    explicit RotationWidget(QWidget* parent = nullptr)
        : QWidget(parent)
    {
        setAutoFillBackground(false);
        setAttribute(Qt::WA_TranslucentBackground);

        auto* layout = new QHBoxLayout(this);
        layout->setContentsMargins(10, 8, 10, 8);
        layout->setSpacing(8);

        _dial = new RotationDial(this);
        layout->addWidget(_dial);

        auto* controls = new QVBoxLayout();
        controls->setContentsMargins(0, 0, 0, 0);
        controls->setSpacing(5);

        auto* label = new QLabel(tr("Rotate"), this);
        label->setStyleSheet("QLabel { color: white; }");
        controls->addWidget(label);

        _spin = new QDoubleSpinBox(this);
        _spin->setRange(-360.0, 360.0);
        _spin->setDecimals(2);
        _spin->setSingleStep(1.0);
        _spin->setSuffix(QString::fromUtf8("°"));
        _spin->setFixedWidth(92);
        controls->addWidget(_spin);

        _horizontalFlip = new QCheckBox(tr("Flip horizontally"), this);
        _horizontalFlip->setObjectName(QString::fromLatin1(kHorizontalFlipObjectName));
        _horizontalFlip->setStyleSheet("QCheckBox { color: white; }");
        controls->addWidget(_horizontalFlip);

        auto* apply = new QPushButton(tr("Apply"), this);
        apply->setObjectName(QString::fromLatin1(kApplyTransformObjectName));
        controls->addWidget(apply);
        layout->addLayout(controls);

        setStyleSheet(
            "RotationWidget { background: rgba(24, 24, 24, 210); border: 1px solid rgba(255,255,255,80); border-radius: 6px; }"
            "QDoubleSpinBox { background: rgba(255,255,255,235); color: black; }"
            "QPushButton { padding: 4px 10px; }");

        _dial->angleChanged = [this](double angle) {
            {
                QSignalBlocker blocker(_spin);
                _spin->setValue(angle);
            }
            if (angleChanged) {
                angleChanged(angle);
            }
        };
        QObject::connect(_spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                         this, [this](double value) {
                             _dial->setAngle(value);
                             if (angleChanged) {
                                 angleChanged(value);
                             }
                         });
        QObject::connect(_horizontalFlip, &QCheckBox::toggled, this, [this](bool enabled) {
            if (horizontalFlipChanged) {
                horizontalFlipChanged(enabled);
            }
        });
        QObject::connect(apply, &QPushButton::clicked, this, [this]() {
            if (applyRequested) {
                applyRequested();
            }
        });
    }

    void setAngle(double angle)
    {
        QSignalBlocker blocker(_spin);
        _spin->setValue(angle);
        _dial->setAngle(angle);
    }

    std::function<void(double)> angleChanged;
    std::function<void(bool)> horizontalFlipChanged;
    std::function<void()> applyRequested;

protected:
    void paintEvent(QPaintEvent* event) override
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        p.setPen(QPen(QColor(255, 255, 255, 80), 1.0));
        p.setBrush(QColor(24, 24, 24, 210));
        p.drawRoundedRect(rect().adjusted(0, 0, -1, -1), 6, 6);
        QWidget::paintEvent(event);
    }

private:
    RotationDial* _dial{nullptr};
    QDoubleSpinBox* _spin{nullptr};
    QCheckBox* _horizontalFlip{nullptr};
};
} // namespace

SurfaceRotationOverlayController::SurfaceRotationOverlayController(CState* state, QObject* parent)
    : QObject(parent)
    , _state(state)
{
}

SurfaceRotationOverlayController::~SurfaceRotationOverlayController()
{
    clearWidgets();
    if (_viewerManager) {
        QObject::disconnect(_viewerCreatedConn);
        QObject::disconnect(_viewerClosingConn);
        QObject::disconnect(_managerDestroyedConn);
    }
}

void SurfaceRotationOverlayController::setViewerManager(ViewerManager* manager)
{
    if (_viewerManager == manager) {
        return;
    }
    clearWidgets();
    if (_viewerManager) {
        QObject::disconnect(_viewerCreatedConn);
        QObject::disconnect(_viewerClosingConn);
        QObject::disconnect(_managerDestroyedConn);
    }

    _viewerManager = manager;
    if (!_viewerManager) {
        return;
    }

    _viewerCreatedConn = QObject::connect(_viewerManager, &ViewerManager::baseViewerCreated,
                                          this, [this](auto* viewer) { attachViewer(viewer); });
    _viewerClosingConn = QObject::connect(_viewerManager, &ViewerManager::baseViewerClosing,
                                          this, [this](auto* viewer) { detachViewer(viewer); });
    _managerDestroyedConn = QObject::connect(_viewerManager, &QObject::destroyed,
                                             this, [this]() {
                                                 clearWidgets();
                                                 _viewerManager = nullptr;
                                             });
    _viewerManager->forEachBaseViewer([this](auto* viewer) { attachViewer(viewer); });
}

void SurfaceRotationOverlayController::beginRotate()
{
    auto source = currentSourceSurface();
    if (!source) {
        return;
    }

    if (_rotateActive && _sourceSurface == source) {
        ensureWidgetForTarget();
        return;
    }

    const auto sourceSurfaceId = _state->activeSurfaceId();
    cancelRotate();
    _sourceSurface = std::move(source);
    _sourceSurfaceId = sourceSurfaceId;
    // The revision of the geometry this tool transforms (Apply saves only over it).
    _sourceRevision = _sourceRevisionReader ? _sourceRevisionReader(_sourceSurface.get())
                      : _revisionReader     ? _revisionReader(_sourceSurfaceId)
                                            : std::nullopt;
    _shownSurface = _state->surface("segmentation");
    _angleDeg = 0.0;
    _horizontalFlip = false;
    _rotateActive = true;
    ensureWidgetForTarget();
}

void SurfaceRotationOverlayController::cancelRotate()
{
    // Restore the un-previewed source surface — but only if no save
    // worker is currently mutating it. Publishing _sourceSurface
    // while applyRotation()'s background transform is running would point surfaceChanged consumers at a cv::Mat
    // the worker is concurrently writing, racing on _points and the
    // ancillary channels. Whatever was previously active (typically
    // the last _previewSurface from a dial drag) stays in place; the
    // worker's finished slot already drops its own setSurface via
    // the stale-completion guard, so the UI never sees the
    // cancelled-session's transformed output either. A user who
    // wants the on-disk transformed state can reopen the segment.
    if (_rotateActive && _state && _sourceSurface && !_saveInFlight) {
        releaseSource();
    }
    // Invalidate any in-flight save worker — the session it was
    // started for is over, so its completion callback will see a
    // mismatched session id and drop its state changes. We do NOT
    // clear _saveInFlight here: the worker is still running and a
    // new Apply on a freshly begun session must wait for it to
    // finish, otherwise two workers could race the transform on the
    // same QuadSurface (the new session may target the very
    // same segment). The finished slot clears _saveInFlight whether
    // it applies or drops the result.
    ++_saveSessionId;
    closeRotate();
}

void SurfaceRotationOverlayController::closeRotate()
{
    _rotateActive = false;
    _angleDeg = 0.0;
    _horizontalFlip = false;
    _sourceSurface.reset();
    _previewSurface.reset();
    _shownSurface.reset();
    _sourceSurfaceId.clear();
    _sourceRevision.reset();
    clearWidgets();
}

bool SurfaceRotationOverlayController::isOpen() const
{
    return _rotateActive && _state && _state->surface("segmentation") == _shownSurface;
}

bool SurfaceRotationOverlayController::sourceSelected() const
{
    // Not when the user selected another surface, cleared the selection or reopened the segment
    // (e.g. CWindow reopens it after another client saved it).
    return _state && _sourceSurface && !_sourceSurfaceId.empty() &&
           _state->activeSurfaceId() == _sourceSurfaceId && _state->activeSurface().lock() == _sourceSurface &&
           _state->surface("segmentation") == _shownSurface;
}

bool SurfaceRotationOverlayController::sourceCurrent() const
{
    // E.g. the surface panel's Flip, an autosave or another client saved the segment meanwhile.
    // A revision that is not known (not stored, or the store did not answer) never matches.
    if (!_revisionReader) {
        return true;
    }
    const auto stored = _revisionReader(_sourceSurfaceId);
    return _sourceRevision && stored && *stored == *_sourceRevision;
}

bool SurfaceRotationOverlayController::newerRevisionStored() const
{
    if (!_revisionReader || !_sourceRevision) {
        return false;
    }
    const auto stored = _revisionReader(_sourceSurfaceId);
    return stored && *stored != *_sourceRevision;
}

void SurfaceRotationOverlayController::releaseSource()
{
    const bool newer = newerRevisionStored();
    if (!newer && sourceSelected()) {
        endPreview(_sourceSurface);
        return;
    }
    // A newer revision was saved (the surface panel's Flip, an autosave, another client): the
    // geometry the tool started from must not stay selected. Open the saved one, so that the next
    // Rotate, the surface panel and the agent bridge continue from it; failing that, deselect.
    if (newer && _state->activeSurfaceId() == _sourceSurfaceId &&
        _state->activeSurface().lock() == _sourceSurface) {
        if (_reopener && _reopener(_sourceSurfaceId)) {
            return;
        }
        _state->clearActiveSurface();
    }
    // Take the preview down where it is still shown; another selection stays as it is.
    if (_state->surface("segmentation") == _shownSurface) {
        showActive(_sourceSurface);
    }
}

void SurfaceRotationOverlayController::attachViewer(VolumeViewerBase* viewer)
{
    if (!viewer) {
        return;
    }
    auto existing = std::find_if(_viewers.begin(), _viewers.end(), [viewer](const ViewerEntry& entry) {
        return entry.viewer == viewer;
    });
    if (existing != _viewers.end()) {
        return;
    }

    ViewerEntry entry;
    entry.viewer = viewer;
    entry.overlaysUpdatedConn = viewer->connectOverlaysUpdated(
        this, [this, viewer]() {
            auto it = std::find_if(_viewers.begin(), _viewers.end(), [viewer](const ViewerEntry& entry) {
                return entry.viewer == viewer;
            });
            if (it != _viewers.end()) {
                positionWidget(*it);
            }
        });
    entry.destroyedConn = QObject::connect(viewer->asQObject(), &QObject::destroyed,
                                           this, [this, viewer]() { detachViewer(viewer); });
    _viewers.push_back(entry);
}

void SurfaceRotationOverlayController::detachViewer(VolumeViewerBase* viewer)
{
    for (auto it = _viewers.begin(); it != _viewers.end();) {
        if (it->viewer != viewer) {
            ++it;
            continue;
        }
        QObject::disconnect(it->overlaysUpdatedConn);
        QObject::disconnect(it->destroyedConn);
        if (it->viewer) {
            it->viewer->clearOverlayGroup(kOverlayGroupKey);
        }
        it->proxy = nullptr;
        it = _viewers.erase(it);
    }
}

VolumeViewerBase* SurfaceRotationOverlayController::targetViewer() const
{
    if (!_viewerManager) {
        return nullptr;
    }

    VolumeViewerBase* fallback = nullptr;
    for (auto* viewer : _viewerManager->baseViewers()) {
        if (!viewer) {
            continue;
        }
        if (viewer->surfName() == "segmentation") {
            return viewer;
        }
        if (!fallback) {
            fallback = viewer;
        }
    }
    return fallback;
}

std::shared_ptr<QuadSurface> SurfaceRotationOverlayController::currentSourceSurface() const
{
    if (!_state) {
        return nullptr;
    }

    // The surface opened from the store (the editing preview shares its id).
    return _state->activeSurface().lock();
}

void SurfaceRotationOverlayController::ensureWidgetForTarget()
{
    clearWidgets();
    if (!_rotateActive) {
        return;
    }

    auto* viewer = targetViewer();
    if (!viewer || !viewer->graphicsView() || !viewer->graphicsView()->scene()) {
        return;
    }

    auto it = std::find_if(_viewers.begin(), _viewers.end(), [viewer](const ViewerEntry& entry) {
        return entry.viewer == viewer;
    });
    if (it == _viewers.end()) {
        attachViewer(viewer);
        it = std::find_if(_viewers.begin(), _viewers.end(), [viewer](const ViewerEntry& entry) {
            return entry.viewer == viewer;
        });
    }
    if (it == _viewers.end()) {
        return;
    }

    auto* widget = new RotationWidget();
    widget->setAngle(_angleDeg);
    widget->angleChanged = [this](double angle) { setAngle(angle); };
    widget->horizontalFlipChanged = [this](bool enabled) { setHorizontalFlip(enabled); };
    widget->applyRequested = [this]() { applyRotation(); };

    auto* proxy = new QGraphicsProxyWidget();
    proxy->setWidget(widget);
    proxy->setZValue(10000.0);
    viewer->graphicsView()->scene()->addItem(proxy);
    it->proxy = proxy;
    viewer->setOverlayGroup(kOverlayGroupKey, {proxy});
    synchronizeTransformPersistenceCompatibility();
    positionWidget(*it);
}

void SurfaceRotationOverlayController::clearWidgets()
{
    for (auto& entry : _viewers) {
        if (entry.viewer) {
            entry.viewer->clearOverlayGroup(kOverlayGroupKey);
        }
        entry.proxy = nullptr;
    }
}

void SurfaceRotationOverlayController::positionWidget(ViewerEntry& entry) const
{
    if (!entry.proxy || !entry.viewer || !entry.viewer->graphicsView()) {
        return;
    }
    auto* view = entry.viewer->graphicsView();
    const QPoint sceneAnchor = view->mapToScene(QPoint(14, 38)).toPoint();
    entry.proxy->setPos(sceneAnchor);
}

void SurfaceRotationOverlayController::setAngle(double angleDeg)
{
    const double clamped = std::clamp(angleDeg, -360.0, 360.0);
    if (std::abs(_angleDeg - clamped) < 1e-4) {
        return;
    }
    _angleDeg = clamped;
    updatePreview();
}

bool SurfaceRotationOverlayController::setHorizontalFlip(bool enabled)
{
    const bool previouslySelected = _horizontalFlip;
    _horizontalFlip = enabled;
    synchronizeTransformPersistenceCompatibility();
    if (_horizontalFlip != previouslySelected) {
        updatePreview();
    }
    return true;
}

bool SurfaceRotationOverlayController::synchronizeTransformPersistenceCompatibility()
{
    const auto compatibility = surfaceTransformPersistenceCompatibility(_sourceSurface);
    const bool available = compatibility.allowed();
    const QString previewTooltip = available
        ? QString{}
        : QObject::tr("Preview remains available. %1")
              .arg(transformPersistenceUnavailableTooltip(compatibility));
    for (const auto& entry : _viewers) {
        if (!entry.proxy || !entry.proxy->widget()) {
            continue;
        }
        if (auto* checkbox = entry.proxy->widget()->findChild<QCheckBox*>(QString::fromLatin1(kHorizontalFlipObjectName))) {
            const QSignalBlocker blocker(checkbox);
            checkbox->setChecked(_horizontalFlip);
            checkbox->setEnabled(true);
            checkbox->setToolTip(previewTooltip);
        }

        if (auto* apply = entry.proxy->widget()->findChild<QPushButton*>(QString::fromLatin1(kApplyTransformObjectName))) {
            apply->setEnabled(available);
            apply->setToolTip(available ? QString{} : transformPersistenceUnavailableTooltip(compatibility));
        }
    }
    return available;
}

void SurfaceRotationOverlayController::updatePreview()
{
    if (!_rotateActive || !_state || !_sourceSurface) {
        return;
    }
    // While a save worker is running, _sourceSurface is being mutated
    // (the transform) on the worker thread. Cloning it here on
    // the UI thread would race the worker's writes — read while another
    // thread mutates the same cv::Mat data. Skip the preview update; the
    // worker will tear the controller down via its finished slot anyway.
    if (_saveInFlight) {
        return;
    }
    // Another surface was selected, the selection cleared or the segment reopened: leave it
    // (Apply then closes the tool).
    if (!sourceSelected()) {
        return;
    }

    synchronizeTransformPersistenceCompatibility();
    const vc3d::surface_rotation::Transform transform{static_cast<float>(_angleDeg), _horizontalFlip};
    if (transform.isNoOp()) {
        _previewSurface.reset();
        showActive(_sourceSurface);
        return;
    }

    _previewSurface = cloneSurface(_sourceSurface);
    if (!_previewSurface) {
        return;
    }
    transform.applyInMemory(*_previewSurface);
    showActive(_previewSurface);
}

void SurfaceRotationOverlayController::showActive(const std::shared_ptr<QuadSurface>& surface)
{
    // Showing a preview replaces the selected surface, so CWindow drops it as the active one
    // (onSurfaceWillBeDeleted): the segment stays selected while the tool shows it.
    const bool selected = sourceSelected();
    // Include the transformed exclusions and channels in preview checkpoints.
    _state->setSurface("segmentation", surface, false, true, true);
    _shownSurface = surface;
    if (selected) {
        _state->setActiveSurface(_sourceSurfaceId, _sourceSurface);
    }
}

void SurfaceRotationOverlayController::endPreview(const std::shared_ptr<QuadSurface>& surface)
{
    // `surface` is the selected segment now; made active before it replaces the preview.
    _state->setActiveSurface(_sourceSurfaceId, surface);
    // A saved result is already shown. Avoid republishing it as an unsaved draft.
    if (_state->surface("segmentation") != surface)
        _state->setSurface("segmentation", surface, false, true, true);
    _shownSurface = surface;
}

void SurfaceRotationOverlayController::applyRotation()
{
    if (!_rotateActive || !_state || !_sourceSurface) {
        cancelRotate();
        return;
    }

    // Reentrancy guard: another save is already running for this
    // surface. A second transform racing the first would mutate
    // _points concurrently.
    // Just ignore the duplicate Apply.
    if (_saveInFlight) {
        return;
    }

    const vc3d::surface_rotation::Transform transform{static_cast<float>(_angleDeg), _horizontalFlip};
    // Save only over the selected segment as the tool opened it: never select it again after the
    // user moved on, and never overwrite a newer save.
    if (!sourceSelected() || !sourceCurrent()) {
        cancelRotate();
        if (!transform.isNoOp()) {
            warnTransformNotApplied();
        }
        return;
    }

    const bool persistenceAvailable = synchronizeTransformPersistenceCompatibility();
    if (!persistenceAvailable) {
        QMessageBox::warning(
            nullptr,
            tr("Surface Transform Unavailable"),
            transformPersistenceUnavailableTooltip(surfaceTransformPersistenceCompatibility(_sourceSurface)));
        return;
    }

    // No transform: no I/O, nothing to do off-thread.
    if (transform.isNoOp()) {
        endPreview(_sourceSurface);
        closeRotate();
        return;
    }

    // Clone the source before dispatching the worker. The worker can then
    // transform/save without racing the live surface used by the preview, and a
    // failed save leaves the in-memory source unchanged.
    auto surface = cloneSurface(_sourceSurface);
    if (!surface) {
        QMessageBox::warning(nullptr, tr("Surface Transform Failed"), tr("Failed to save the transformed surface: could not copy the selected surface."));
        return;
    }

    const int session = ++_saveSessionId;
    _saveInFlight = true;
    QPointer<SurfaceRotationOverlayController> self(this);

    auto* watcher = new QFutureWatcher<RotationSaveResult>(this);
    connect(watcher, &QFutureWatcher<RotationSaveResult>::finished, this,
            [self, watcher, session]() {
                watcher->deleteLater();
                if (!self) {
                    return;
                }
                // Always clear the in-flight flag once the worker is
                // done, even on stale/cancelled completion. The
                // reentrancy guard relies on this — leaving it set
                // would lock out future Applies after a cancel.
                self->_saveInFlight = false;
                if (!self->_state) {
                    return;
                }
                // Stale completion: cancelRotate() or another Apply
                // bumped the session id. The user's current rotation
                // session (if any) belongs to a different surface or
                // was cancelled — applying our results would
                // overwrite live state with state from an old session.
                if (session != self->_saveSessionId) {
                    return;
                }

                RotationSaveResult result;
                try {
                    result = watcher->result();
                } catch (const std::exception& e) {
                    result.error = QString::fromUtf8(e.what());
                } catch (...) {
                    result.error = tr("Unknown error");
                }

                if (!result.error.isEmpty() || !result.surface) {
                    self->releaseSource();
                    self->closeRotate();
                    QMessageBox::warning(
                        nullptr,
                        tr("Surface Transform Failed"),
                        tr("Failed to save the transformed surface: %1").arg(result.error.isEmpty() ? tr("no surface was written") : result.error));
                    return;
                }

                // The selection or the stored segment changed while the transform ran.
                if (!self->sourceSelected() || !self->sourceCurrent()) {
                    self->cancelRotate();
                    warnTransformNotApplied();
                    return;
                }

                // The durable save must capture the complete transformed draft.
                self->showActive(result.surface);

                // The store edit refuses the save unless it still holds the geometry revision the
                // transform started from (a save that finished or was resolved meanwhile wins).
                if (!self->_saver || !self->_saver(result.surface.get(), self->_sourceRevision)) {
                    self->releaseSource();
                    // The failed save can stay pending in the segment's store edit, and the next save
                    // through it commits it first: continue only from the segment reopened.
                    if (self->sourceSelected() && !(self->_reopener && self->_reopener(self->_sourceSurfaceId))) {
                        self->_state->clearActiveSurface();
                    }
                    self->closeRotate();
                    QMessageBox::warning(nullptr, tr("Surface Transform Failed"),
                                         tr("Failed to save the transformed surface to the object store."));
                    return;
                }
                // The saved surface is the segment now: the next Rotate/Flip and other
                // readers of the active surface continue from it, not from the source.
                self->endPreview(result.surface);
                self->closeRotate();
            });

    auto future = QtConcurrent::run([surface, transform]() -> RotationSaveResult {
        // The transform runs here; publishing happens on the UI thread.
        try {
            transform.applyInMemory(*surface);
        } catch (const std::exception& e) {
            return RotationSaveResult{nullptr, QString::fromUtf8(e.what())};
        } catch (...) {
            return RotationSaveResult{nullptr, QObject::tr("Unknown error")};
        }
        return RotationSaveResult{surface, {}};
    });
    watcher->setFuture(future);
}

std::shared_ptr<QuadSurface> SurfaceRotationOverlayController::cloneSurface(const std::shared_ptr<QuadSurface>& surface)
{
    if (!surface) {
        return nullptr;
    }

    auto clone = std::make_shared<QuadSurface>(surface->rawPoints(), surface->scale());
    for (const auto& name : surface->channelNames()) {
        cv::Mat channel = surface->channel(name, SURF_CHANNEL_NORESIZE);
        if (!channel.empty()) {
            clone->setChannel(name, channel.clone());
        }
    }
    clone->id = surface->id;
    clone->meta = surface->meta;
    clone->maskedVertices = surface->maskedVertices;
    return clone;
}
