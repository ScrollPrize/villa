#include "TransformOverlayController.hpp"

#include "../adaptive/CAdaptiveVolumeViewer.hpp"
#include "../CState.hpp"
#include "../CVolumeViewerView.hpp"
#include "../ViewerManager.hpp"
#include "../VolumeViewerBase.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <QApplication>
#include <QDoubleSpinBox>
#include <QGraphicsProxyWidget>
#include <QGraphicsScene>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QPushButton>
#include <QSignalBlocker>
#include <QStatusBar>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <cmath>
#include <exception>

namespace
{
constexpr const char* kOverlayGroupKey = "surface_transform_rotate";
constexpr double kPi = 3.14159265358979323846;

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

        auto* apply = new QPushButton(tr("Apply"), this);
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
};
} // namespace

TransformOverlayController::TransformOverlayController(CState* state, QObject* parent)
    : QObject(parent)
    , _state(state)
{
}

TransformOverlayController::~TransformOverlayController()
{
    clearWidgets();
    if (_viewerManager) {
        QObject::disconnect(_viewerCreatedConn);
        QObject::disconnect(_managerDestroyedConn);
    }
}

void TransformOverlayController::setViewerManager(ViewerManager* manager)
{
    if (_viewerManager == manager) {
        return;
    }
    clearWidgets();
    if (_viewerManager) {
        QObject::disconnect(_viewerCreatedConn);
        QObject::disconnect(_managerDestroyedConn);
    }

    _viewerManager = manager;
    if (!_viewerManager) {
        return;
    }

    _viewerCreatedConn = QObject::connect(_viewerManager, &ViewerManager::viewerCreated,
                                          this, [this](auto* viewer) { attachViewer(viewer); });
    _managerDestroyedConn = QObject::connect(_viewerManager, &QObject::destroyed,
                                             this, [this]() {
                                                 clearWidgets();
                                                 _viewerManager = nullptr;
                                             });
    _viewerManager->forEachViewer([this](auto* viewer) { attachViewer(viewer); });
}

void TransformOverlayController::beginRotate()
{
    auto source = currentSourceSurface();
    if (!source) {
        return;
    }

    if (_rotateActive && _sourceSurface == source) {
        ensureWidgetForTarget();
        return;
    }

    cancelRotate();
    _sourceSurface = std::move(source);
    _angleDeg = 0.0;
    _rotateActive = true;
    ensureWidgetForTarget();
}

void TransformOverlayController::cancelRotate()
{
    if (_rotateActive && _state && _sourceSurface) {
        _state->setSurface("segmentation", _sourceSurface, false, true);
    }
    _rotateActive = false;
    _angleDeg = 0.0;
    _sourceSurface.reset();
    _previewSurface.reset();
    clearWidgets();
}

void TransformOverlayController::attachViewer(VolumeViewerBase* viewer)
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

void TransformOverlayController::detachViewer(VolumeViewerBase* viewer)
{
    auto it = std::remove_if(_viewers.begin(), _viewers.end(), [viewer](ViewerEntry& entry) {
        if (entry.viewer != viewer) {
            return false;
        }
        QObject::disconnect(entry.overlaysUpdatedConn);
        QObject::disconnect(entry.destroyedConn);
        if (entry.viewer) {
            entry.viewer->clearOverlayGroup(kOverlayGroupKey);
        }
        entry.proxy = nullptr;
        return true;
    });
    _viewers.erase(it, _viewers.end());
}

VolumeViewerBase* TransformOverlayController::targetViewer() const
{
    if (!_viewerManager) {
        return nullptr;
    }

    VolumeViewerBase* fallback = nullptr;
    for (auto* viewer : _viewerManager->viewers()) {
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

std::shared_ptr<QuadSurface> TransformOverlayController::currentSourceSurface() const
{
    if (!_state) {
        return nullptr;
    }
    auto active = _state->activeSurface().lock();
    if (active) {
        return active;
    }
    return std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));
}

void TransformOverlayController::ensureWidgetForTarget()
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
    widget->applyRequested = [this]() { applyRotation(); };

    auto* proxy = new QGraphicsProxyWidget();
    proxy->setWidget(widget);
    proxy->setZValue(10000.0);
    viewer->graphicsView()->scene()->addItem(proxy);
    it->proxy = proxy;
    viewer->setOverlayGroup(kOverlayGroupKey, {proxy});
    positionWidget(*it);
}

void TransformOverlayController::clearWidgets()
{
    for (auto& entry : _viewers) {
        if (entry.viewer) {
            entry.viewer->clearOverlayGroup(kOverlayGroupKey);
        }
        entry.proxy = nullptr;
    }
}

void TransformOverlayController::positionWidget(ViewerEntry& entry) const
{
    if (!entry.proxy || !entry.viewer || !entry.viewer->graphicsView()) {
        return;
    }
    auto* view = entry.viewer->graphicsView();
    const QPoint sceneAnchor = view->mapToScene(QPoint(14, 38)).toPoint();
    entry.proxy->setPos(sceneAnchor);
}

void TransformOverlayController::setAngle(double angleDeg)
{
    const double clamped = std::clamp(angleDeg, -360.0, 360.0);
    if (std::abs(_angleDeg - clamped) < 1e-4) {
        return;
    }
    _angleDeg = clamped;
    updatePreview();
}

void TransformOverlayController::updatePreview()
{
    if (!_rotateActive || !_state || !_sourceSurface) {
        return;
    }

    if (std::abs(_angleDeg) < 0.01) {
        _previewSurface.reset();
        _state->setSurface("segmentation", _sourceSurface, false, true);
        return;
    }

    _previewSurface = cloneSurface(_sourceSurface);
    if (!_previewSurface) {
        return;
    }
    _previewSurface->rotate(static_cast<float>(_angleDeg));
    _state->setSurface("segmentation", _previewSurface, false, true);
}

void TransformOverlayController::applyRotation()
{
    if (!_rotateActive || !_state || !_sourceSurface) {
        cancelRotate();
        return;
    }

    try {
        if (std::abs(_angleDeg) >= 0.01) {
            _sourceSurface->saveSnapshot();
            _sourceSurface->rotate(static_cast<float>(_angleDeg));
            _sourceSurface->saveOverwrite();
            if (_viewerManager) {
                _viewerManager->refreshSurfacePatchIndex(_sourceSurface);
            }
        }
        _state->setSurface("segmentation", _sourceSurface, false, true);
    } catch (const std::exception&) {
        _state->setSurface("segmentation", _sourceSurface, false, true);
        clearWidgets();
        _rotateActive = false;
        _previewSurface.reset();
        _sourceSurface.reset();
        QMessageBox::warning(nullptr,
                             tr("Rotation Failed"),
                             tr("Failed to save the rotated surface."));
        return;
    }

    _rotateActive = false;
    _angleDeg = 0.0;
    _previewSurface.reset();
    _sourceSurface.reset();
    clearWidgets();
}

std::shared_ptr<QuadSurface> TransformOverlayController::cloneSurface(const std::shared_ptr<QuadSurface>& surface)
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
    clone->path = surface->path;
    clone->id = surface->id;
    clone->meta = surface->meta;
    return clone;
}
