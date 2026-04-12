#include "CAdaptiveVolumeViewer.hpp"

#include "ViewerManager.hpp"
#include "VCSettings.hpp"
#include "../CState.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/render/PostProcess.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/types/SampleParams.hpp"

#include <QSettings>
#include <QTimer>
#include <QVBoxLayout>
#include <QLabel>
#include <QGraphicsScene>
#include <QMdiSubWindow>
#include <QWindowStateChangeEvent>
#include <QApplication>
#include <QPointer>

#include <algorithm>
#include <cmath>
#include <limits>

// ============================================================================
// Construction
// ============================================================================

CAdaptiveVolumeViewer::CAdaptiveVolumeViewer(CState* state,
                                              ViewerManager* manager,
                                              QWidget* parent)
    : QWidget(parent)
    , _state(state)
    , _viewerManager(manager)
{
    _view = new CVolumeViewerView(this);
    fGraphicsView = _view;
    _view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    _view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    _view->setTransformationAnchor(QGraphicsView::NoAnchor);
    _view->setRenderHint(QPainter::Antialiasing, false);
    _view->setScrollPanDisabled(true);
    _view->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);

    connect(_view, &CVolumeViewerView::sendScrolled, this, &CAdaptiveVolumeViewer::onScrolled);
    connect(_view, &CVolumeViewerView::sendVolumeClicked, this, &CAdaptiveVolumeViewer::onVolumeClicked);
    connect(_view, &CVolumeViewerView::sendZoom, this, &CAdaptiveVolumeViewer::onZoom);
    connect(_view, &CVolumeViewerView::sendResized, this, &CAdaptiveVolumeViewer::onResized);
    connect(_view, &CVolumeViewerView::sendCursorMove, this, &CAdaptiveVolumeViewer::onCursorMove);
    connect(_view, &CVolumeViewerView::sendPanRelease, this, &CAdaptiveVolumeViewer::onPanRelease);
    connect(_view, &CVolumeViewerView::sendPanStart, this, &CAdaptiveVolumeViewer::onPanStart);
    connect(_view, &CVolumeViewerView::sendMousePress, this, &CAdaptiveVolumeViewer::onMousePress);
    connect(_view, &CVolumeViewerView::sendMouseMove, this, &CAdaptiveVolumeViewer::onMouseMove);
    connect(_view, &CVolumeViewerView::sendMouseRelease, this, &CAdaptiveVolumeViewer::onMouseRelease);
    connect(_view, &CVolumeViewerView::sendKeyPress, this, &CAdaptiveVolumeViewer::onKeyPress);
    connect(_view, &CVolumeViewerView::sendKeyRelease, this, &CAdaptiveVolumeViewer::onKeyRelease);

    _scene = new QGraphicsScene(this);
    _scene->setItemIndexMethod(QGraphicsScene::NoIndex);
    _view->setScene(_scene);
    _view->setDirectFramebuffer(&_framebuffer);

    _renderTimer = new QTimer(this);
    _renderTimer->setSingleShot(true);
    _renderTimer->setInterval(16);
    connect(_renderTimer, &QTimer::timeout, this, [this]() {
        if (_renderPending) {
            _renderPending = false;
            submitRender();
            updateStatusLabel();
        }
    });


    using namespace vc3d::settings;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    _camera.downscaleOverride = settings.value(perf::DOWNSCALE_OVERRIDE, perf::DOWNSCALE_OVERRIDE_DEFAULT).toInt();
    _navSpeed = settings.value(viewer::NAV_SPEED, viewer::NAV_SPEED_DEFAULT).toFloat();
    if (_navSpeed <= 0.0f) _navSpeed = 1.0f;
    _panSensitivity = settings.value(viewer::PAN_SENSITIVITY, viewer::PAN_SENSITIVITY_DEFAULT).toFloat();
    if (_panSensitivity <= 0.0f) _panSensitivity = 1.0f;
    _zoomSensitivity = settings.value(viewer::ZOOM_SENSITIVITY, viewer::ZOOM_SENSITIVITY_DEFAULT).toFloat();
    if (_zoomSensitivity <= 0.0f) _zoomSensitivity = 1.0f;
    _zScrollSensitivity = settings.value(viewer::ZSCROLL_SENSITIVITY, viewer::ZSCROLL_SENSITIVITY_DEFAULT).toFloat();
    if (_zScrollSensitivity <= 0.0f) _zScrollSensitivity = 1.0f;

    auto* layout = new QVBoxLayout;
    layout->addWidget(_view);
    setLayout(layout);

    _lbl = new QLabel(this);
    _lbl->setStyleSheet("QLabel { color : #00FF00; background-color: rgba(0,0,0,128); padding: 2px 4px; }");
    _lbl->setMinimumWidth(700);
    _lbl->adjustSize();
    _lbl->move(10, 5);
}

CAdaptiveVolumeViewer::~CAdaptiveVolumeViewer()
{
    if (_chunkCbId != 0 && _volume && _volume->tieredCache()) {
        _volume->tieredCache()->removeChunkReadyListener(_chunkCbId);
        _chunkCbId = 0;
    }
}

// ============================================================================
// Data setup
// ============================================================================

void CAdaptiveVolumeViewer::setSurface(const std::string& name)
{
    _surfName = name;
    if (_state)
        onSurfaceChanged(name, _state->surface(name));
}

Surface* CAdaptiveVolumeViewer::currentSurface() const
{
    if (!_state) {
        auto shared = _surfWeak.lock();
        return shared ? shared.get() : nullptr;
    }
    return _state->surfaceRaw(_surfName);
}

// ============================================================================
// Volume / surface change handlers
// ============================================================================

void CAdaptiveVolumeViewer::OnVolumeChanged(std::shared_ptr<Volume> vol)
{
    if (_chunkCbId != 0 && _volume && _volume->tieredCache()) {
        _volume->tieredCache()->removeChunkReadyListener(_chunkCbId);
        _chunkCbId = 0;
    }

    _volume = std::move(vol);
    _hadValidDataBounds = false;

    if (_volume && _volume->numScales() >= 1) {
        auto* cache = _volume->tieredCache();
        QPointer<CAdaptiveVolumeViewer> guard(this);
        _chunkCbId = cache->addChunkReadyListener(
            [guard, cache](const vc::cache::ChunkKey&) {
                QMetaObject::invokeMethod(qApp, [guard, cache]() {
                    if (guard) {
                        cache->clearChunkArrivedFlag();
                        guard->scheduleRender();
                    }
                }, Qt::QueuedConnection);
            });
    }

    // Create default PlaneSurface for axis-aligned views
    if (!_surfWeak.lock() && _volume && isAxisAlignedView()) {
        auto shape = _volume->shape();
        cv::Vec3f center(static_cast<float>(shape[0]) * 0.5f,
                         static_cast<float>(shape[1]) * 0.5f,
                         static_cast<float>(shape[2]) * 0.5f);
        cv::Vec3f normal;
        if (_surfName == "xy plane") normal = cv::Vec3f(0, 0, 1);
        else if (_surfName == "xz plane" || _surfName == "seg xz") normal = cv::Vec3f(0, 1, 0);
        else normal = cv::Vec3f(1, 0, 0);
        _defaultSurface = std::make_shared<PlaneSurface>(center, normal);
        _surfWeak = _defaultSurface;
    }

    if (_volume) {
        int nScales = static_cast<int>(_volume->numScales());
        _camera.recalcPyramidLevel(nScales);
        double vs = _volume->voxelSize() / static_cast<double>(_camera.dsScale);
        _view->setVoxelSize(vs, vs);
    }

    // Recompute content bounds
    auto surf = _surfWeak.lock();
    if (_volume && surf) {
        if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
            auto [w, h, d] = _volume->shape();
            float corners[][3] = {
                {0,0,0}, {(float)w,0,0}, {0,(float)h,0}, {(float)w,(float)h,0},
                {0,0,(float)d}, {(float)w,0,(float)d}, {0,(float)h,(float)d}, {(float)w,(float)h,(float)d}
            };
            _contentMinU = _contentMinV = std::numeric_limits<float>::max();
            _contentMaxU = _contentMaxV = std::numeric_limits<float>::lowest();
            for (auto& c : corners) {
                cv::Vec3f proj = plane->project(cv::Vec3f(c[0], c[1], c[2]), 1.0, 1.0);
                _contentMinU = std::min(_contentMinU, proj[0]);
                _contentMinV = std::min(_contentMinV, proj[1]);
                _contentMaxU = std::max(_contentMaxU, proj[0]);
                _contentMaxV = std::max(_contentMaxV, proj[1]);
            }
        }
    }

    // Resize framebuffer
    QSize vpSize = _view->viewport()->size();
    int vpW = std::max(1, vpSize.width());
    int vpH = std::max(1, vpSize.height());
    if (_framebuffer.isNull() || _framebuffer.width() != vpW || _framebuffer.height() != vpH) {
        _framebuffer = QImage(vpW, vpH, QImage::Format_RGB32);
        _framebuffer.fill(QColor(64, 64, 64));
    }
    _scene->setSceneRect(0, 0, vpW, vpH);

    submitRender();
    updateStatusLabel();
}

void CAdaptiveVolumeViewer::onSurfaceChanged(const std::string& name,
                                              const std::shared_ptr<Surface>& surf,
                                              bool /*isEditUpdate*/)
{
    if (_surfName != name) return;
    _surfWeak = surf;

    if (!surf) {
        _scene->clear();
        _overlayGroups.clear();
        return;
    }

    // Recompute content bounds for PlaneSurface
    if (_volume) {
        if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
            auto [w, h, d] = _volume->shape();
            float corners[][3] = {
                {0,0,0}, {(float)w,0,0}, {0,(float)h,0}, {(float)w,(float)h,0},
                {0,0,(float)d}, {(float)w,0,(float)d}, {0,(float)h,(float)d}, {(float)w,(float)h,(float)d}
            };
            _contentMinU = _contentMinV = std::numeric_limits<float>::max();
            _contentMaxU = _contentMaxV = std::numeric_limits<float>::lowest();
            for (auto& c : corners) {
                cv::Vec3f proj = plane->project(cv::Vec3f(c[0], c[1], c[2]), 1.0, 1.0);
                _contentMinU = std::min(_contentMinU, proj[0]);
                _contentMinV = std::min(_contentMinV, proj[1]);
                _contentMaxU = std::max(_contentMaxU, proj[0]);
                _contentMaxV = std::max(_contentMaxV, proj[1]);
            }
        }
    }

    submitRender();
}

void CAdaptiveVolumeViewer::onSurfaceWillBeDeleted(const std::string& /*name*/,
                                                    const std::shared_ptr<Surface>& surf)
{
    auto current = _surfWeak.lock();
    if (current && current == surf)
        _surfWeak.reset();
}

void CAdaptiveVolumeViewer::onVolumeClosing()
{
    onSurfaceChanged(_surfName, nullptr);
}

void CAdaptiveVolumeViewer::onPOIChanged(const std::string& name, POI* poi)
{
    if (name != "focus" || !poi) return;

    auto surf = _surfWeak.lock();
    auto* plane = dynamic_cast<PlaneSurface*>(surf.get());
    if (!plane) return;

    plane->setOrigin(poi->p);
    if (cv::norm(poi->n) > 0.5)
        plane->setNormal(poi->n);

    // Check if data bounds just became valid
    if (!_hadValidDataBounds && _volume) {
        const auto& db = _volume->dataBounds();
        if (db.valid) {
            _hadValidDataBounds = true;
            // Recompute content bounds with new data
            OnVolumeChanged(_volume);
        }
    }

    // Center on focus
    cv::Vec3f proj = plane->project(poi->p, 1.0, 1.0);
    _camera.surfacePtr[0] = proj[0];
    _camera.surfacePtr[1] = proj[1];

    submitRender();
}

// ============================================================================
// Rendering
// ============================================================================

void CAdaptiveVolumeViewer::scheduleRender()
{
    _renderPending = true;
    if (!_renderTimer->isActive())
        _renderTimer->start();
}

void CAdaptiveVolumeViewer::submitRender()
{
    // Re-read sensitivity settings (changed live via Viewer Controls panel)
    {
        QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
        using namespace vc3d::settings;
        _panSensitivity = std::max(0.01f, s.value(viewer::PAN_SENSITIVITY, viewer::PAN_SENSITIVITY_DEFAULT).toFloat());
        _zoomSensitivity = std::max(0.01f, s.value(viewer::ZOOM_SENSITIVITY, viewer::ZOOM_SENSITIVITY_DEFAULT).toFloat());
        _zScrollSensitivity = std::max(0.01f, s.value(viewer::ZSCROLL_SENSITIVITY, viewer::ZSCROLL_SENSITIVITY_DEFAULT).toFloat());
        int interpIdx = s.value(perf::INTERPOLATION_METHOD, 1).toInt();
        _samplingMethod = static_cast<vc::Sampling>(std::clamp(interpIdx, 0, 3));
    }

    auto surf = _surfWeak.lock();
    if (!surf || !_volume || !_volume->zarrDataset()) return;

    int fbW = _framebuffer.width();
    int fbH = _framebuffer.height();
    if (fbW <= 0 || fbH <= 0) return;

    auto* fbBits = reinterpret_cast<uint32_t*>(_framebuffer.bits());
    int fbStride = _framebuffer.bytesPerLine() / 4;

    std::array<uint32_t, 256> lut;
    vc::buildWindowLevelColormapLut(lut, _windowLow, _windowHigh, _baseColormapId);

    vc::SampleParams sp;
    sp.level = _camera.dsScaleIdx;
    sp.method = _samplingMethod;

    const int numLevels = static_cast<int>(_volume->numScales());

    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        cv::Vec3f vx = plane->basisX();
        cv::Vec3f vy = plane->basisY();
        cv::Vec3f n = plane->normal(cv::Vec3f(0, 0, 0));

        float halfW = static_cast<float>(fbW) * 0.5f / _camera.scale;
        float halfH = static_cast<float>(fbH) * 0.5f / _camera.scale;

        cv::Vec3f origin = vx * (_camera.surfacePtr[0] - halfW)
                         + vy * (_camera.surfacePtr[1] - halfH)
                         + plane->origin() + n * _camera.zOff;
        cv::Vec3f vx_step = vx / _camera.scale;
        cv::Vec3f vy_step = vy / _camera.scale;

        int numLayers = 1, zStart = 0;
        float zStep = 1.0f;
        const cv::Vec3f* pNormal = nullptr;
        std::string method;
        if (_compositeSettings.planeEnabled) {
            const int front = _compositeSettings.planeLayersFront;
            const int behind = _compositeSettings.planeLayersBehind;
            numLayers = front + behind + 1;
            zStart = -behind;
            zStep = _compositeSettings.reverseDirection ? -1.0f : 1.0f;
            pNormal = &n;
            method = _compositeSettings.params.method;
        } else {
            pNormal = &n; // ignored for numLayers=1
        }

        // Composite averages ~11 layers — the averaging is itself a low-pass,
        // so per-layer Nearest matches Trilinear visually at ~8x the speed.
        vc::Sampling sampleMethod = (numLayers > 1) ? vc::Sampling::Nearest
                                                    : _samplingMethod;
        sampleAdaptiveARGB32(
            fbBits, fbStride, _volume->tieredCache(),
            sp.level, numLevels,
            nullptr, &origin, &vx_step, &vy_step,
            nullptr, pNormal,
            numLayers, zStart, zStep,
            fbW, fbH, method, lut.data(), sampleMethod);
    } else {
        cv::Mat_<cv::Vec3f> coords;
        cv::Mat_<cv::Vec3f> normals;
        cv::Vec3f offset(_camera.surfacePtr[0] * _camera.scale,
                         _camera.surfacePtr[1] * _camera.scale,
                         _camera.zOff);
        const bool wantComposite = _compositeSettings.enabled;
        surf->gen(&coords, wantComposite ? &normals : nullptr,
                  cv::Size(fbW, fbH), cv::Vec3f(0, 0, 0),
                  _camera.scale, offset);

        if (!coords.empty()) {
            int numLayers = 1, zStart = 0;
            float zStep = 1.0f;
            const cv::Mat_<cv::Vec3f>* pNormals = nullptr;
            std::string method;
            if (wantComposite && !normals.empty()) {
                const int front = _compositeSettings.layersFront;
                const int behind = _compositeSettings.layersBehind;
                numLayers = front + behind + 1;
                zStart = -behind;
                zStep = _compositeSettings.reverseDirection ? -1.0f : 1.0f;
                pNormals = &normals;
                method = _compositeSettings.params.method;
            }
            vc::Sampling sampleMethod = (numLayers > 1) ? vc::Sampling::Nearest
                                                        : _samplingMethod;
            sampleAdaptiveARGB32(
                fbBits, fbStride, _volume->tieredCache(),
                sp.level, numLevels,
                &coords, nullptr, nullptr, nullptr,
                pNormals, nullptr,
                numLayers, zStart, zStep,
                fbW, fbH, method, lut.data(), sampleMethod);
        }
    }

    // Update camera tracking for coordinate conversions
    _camSurfX = _camera.surfacePtr[0];
    _camSurfY = _camera.surfacePtr[1];
    _camScale = _camera.scale;

    _view->viewport()->repaint();
    updateStatusLabel();
}

void CAdaptiveVolumeViewer::renderVisible(bool force)
{
    if (force)
        submitRender();
    else
        scheduleRender();
}

// ============================================================================
// Navigation
// ============================================================================

void CAdaptiveVolumeViewer::panByF(float dx, float dy)
{
    const float invScale = _panSensitivity / _camera.scale;
    _camera.surfacePtr[0] -= dx * invScale;
    _camera.surfacePtr[1] -= dy * invScale;

    if (_contentMaxU > _contentMinU) {
        _camera.surfacePtr[0] = std::clamp(_camera.surfacePtr[0], _contentMinU, _contentMaxU);
        _camera.surfacePtr[1] = std::clamp(_camera.surfacePtr[1], _contentMinV, _contentMaxV);
    }

    submitRender();
}

void CAdaptiveVolumeViewer::zoomStepsAt(int steps, const QPointF& scenePos)
{
    if (steps == 0) return;

    float factor = std::pow(1.05f, static_cast<float>(steps) * _zoomSensitivity);
    float newScale = std::clamp(_camera.scale * factor,
                                AdaptiveCamera::MIN_SCALE,
                                AdaptiveCamera::MAX_SCALE);
    if (std::abs(newScale - _camera.scale) < _camera.scale * 1e-6f) return;

    // Zoom-at-point: surface position under cursor stays fixed
    float vpW = static_cast<float>(_view->viewport()->width());
    float vpH = static_cast<float>(_view->viewport()->height());
    float mx = static_cast<float>(scenePos.x());
    float my = static_cast<float>(scenePos.y());
    float dx = 0, dy = 0;
    if (mx >= 0 && mx < vpW && my >= 0 && my < vpH) {
        dx = mx - vpW * 0.5f;
        dy = my - vpH * 0.5f;
    }

    _camera.surfacePtr[0] += dx * (1.0f / _camera.scale - 1.0f / newScale);
    _camera.surfacePtr[1] += dy * (1.0f / _camera.scale - 1.0f / newScale);
    _camera.scale = newScale;

    if (_volume) {
        float oldDs = _camera.dsScale;
        _camera.recalcPyramidLevel(static_cast<int>(_volume->numScales()));
        if (std::abs(_camera.dsScale - oldDs) > 1e-6f) {
            double vs = _volume->voxelSize() / static_cast<double>(_camera.dsScale);
            _view->setVoxelSize(vs, vs);
        }
    }

    _view->resetTransform();

    // Resize framebuffer if viewport changed
    QSize vpSize = _view->viewport()->size();
    int w = std::max(1, vpSize.width());
    int h = std::max(1, vpSize.height());
    if (_framebuffer.isNull() || _framebuffer.width() != w || _framebuffer.height() != h) {
        _framebuffer = QImage(w, h, QImage::Format_RGB32);
        _framebuffer.fill(QColor(64, 64, 64));
    }
    _scene->setSceneRect(0, 0, w, h);

    submitRender();
}

void CAdaptiveVolumeViewer::adjustZoomByFactor(float factor)
{
    int steps = (factor > 1.0f) ? 1 : (factor < 1.0f) ? -1 : 0;
    if (steps == 0) return;
    QPointF center(static_cast<float>(_view->viewport()->width()) * 0.5f,
                   static_cast<float>(_view->viewport()->height()) * 0.5f);
    zoomStepsAt(steps, center);
}

// ============================================================================
// Event handlers
// ============================================================================

void CAdaptiveVolumeViewer::onZoom(int steps, QPointF scenePoint, Qt::KeyboardModifiers modifiers)
{
    auto surf = _surfWeak.lock();
    if (!surf) return;

    if (modifiers & Qt::ShiftModifier) {
        // Z-scroll
        float dz = static_cast<float>(steps) * _zScrollSensitivity;

        PlaneSurface* plane = dynamic_cast<PlaneSurface*>(surf.get());
        if (_surfName != "segmentation" && plane && _state) {
            POI* focus = _state->poi("focus");
            if (!focus) {
                focus = new POI;
                focus->p = plane->origin();
                focus->n = plane->normal(cv::Vec3f(0, 0, 0), {});
            }

            cv::Vec3f normal = plane->normal(cv::Vec3f(0, 0, 0), {});
            double length = cv::norm(normal);
            if (length > 0.0) normal *= static_cast<float>(1.0 / length);

            cv::Vec3f newPos = focus->p + normal * dz;

            if (_volume) {
                auto [w, h, d] = _volume->shape();
                newPos[0] = std::clamp(newPos[0], 0.0f, static_cast<float>(w - 1));
                newPos[1] = std::clamp(newPos[1], 0.0f, static_cast<float>(h - 1));
                newPos[2] = std::clamp(newPos[2], 0.0f, static_cast<float>(d - 1));
            }

            focus->p = newPos;
            if (length > 0.0) focus->n = normal;
            focus->surfaceId = _surfName;
            _state->setPOI("focus", focus);
        } else {
            // Direct z-offset
            float maxZ = 10000.0f;
            if (_volume) {
                auto [w, h, d] = _volume->shape();
                maxZ = static_cast<float>(std::max({w, h, d}));
            }
            _camera.zOff = std::clamp(_camera.zOff + dz, -maxZ, maxZ);
            submitRender();
            updateStatusLabel();
        }
    } else {
        int zoomDir = (steps > 0) ? 1 : (steps < 0) ? -1 : 0;
        if (zoomDir != 0)
            zoomStepsAt(zoomDir, scenePoint);
    }
}

void CAdaptiveVolumeViewer::onResized()
{
    QSize vpSize = _view->viewport()->size();
    int w = std::max(1, vpSize.width());
    int h = std::max(1, vpSize.height());
    if (_framebuffer.isNull() || _framebuffer.width() != w || _framebuffer.height() != h) {
        _framebuffer = QImage(w, h, QImage::Format_RGB32);
        _framebuffer.fill(QColor(64, 64, 64));
    }
    _scene->setSceneRect(0, 0, w, h);
    submitRender();
}

void CAdaptiveVolumeViewer::onCursorMove(QPointF scenePos)
{
    _lastScenePos = scenePos;
    if (_isPanning) {
        float dx = static_cast<float>(scenePos.x() - _lastPanSceneF.x());
        float dy = static_cast<float>(scenePos.y() - _lastPanSceneF.y());
        _lastPanSceneF = scenePos;
        if (std::abs(dx) > 0.001f || std::abs(dy) > 0.001f) {
            panByF(dx, dy);
        }
    }
}

void CAdaptiveVolumeViewer::onPanStart(Qt::MouseButton, Qt::KeyboardModifiers)
{
    _isPanning = true;
    _lastPanSceneF = _view->mapToScene(_view->mapFromGlobal(QCursor::pos()));
}

void CAdaptiveVolumeViewer::onPanRelease(Qt::MouseButton, Qt::KeyboardModifiers)
{
    _isPanning = false;
    submitRender();
}

void CAdaptiveVolumeViewer::onVolumeClicked(QPointF scenePos, Qt::MouseButton buttons,
                                             Qt::KeyboardModifiers modifiers)
{
    cv::Vec3f p = sceneToVolume(scenePos);
    cv::Vec3f n(0, 0, 1);
    auto surf = _surfWeak.lock();
    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get()))
        n = plane->normal(cv::Vec3f(0, 0, 0));
    emit sendVolumeClicked(p, n, surf.get(), buttons, modifiers);
}

void CAdaptiveVolumeViewer::onMousePress(QPointF scenePos, Qt::MouseButton button,
                                          Qt::KeyboardModifiers modifiers)
{
    cv::Vec3f p = sceneToVolume(scenePos);
    cv::Vec3f n(0, 0, 1);
    emit sendMousePressVolume(p, n, button, modifiers);
}

void CAdaptiveVolumeViewer::onMouseMove(QPointF scenePos, Qt::MouseButtons buttons,
                                         Qt::KeyboardModifiers modifiers)
{
    cv::Vec3f p = sceneToVolume(scenePos);
    emit sendMouseMoveVolume(p, buttons, modifiers);
}

void CAdaptiveVolumeViewer::onMouseRelease(QPointF scenePos, Qt::MouseButton button,
                                            Qt::KeyboardModifiers modifiers)
{
    cv::Vec3f p = sceneToVolume(scenePos);
    emit sendMouseReleaseVolume(p, button, modifiers);
}

void CAdaptiveVolumeViewer::onKeyPress(int key, Qt::KeyboardModifiers)
{
    constexpr float PAN_PX = 64.0f;
    switch (key) {
        case Qt::Key_Left:  panByF(PAN_PX, 0); break;
        case Qt::Key_Right: panByF(-PAN_PX, 0); break;
        case Qt::Key_Up:    panByF(0, PAN_PX); break;
        case Qt::Key_Down:  panByF(0, -PAN_PX); break;
    }
}

// ============================================================================
// Coordinate transforms
// ============================================================================

QPointF CAdaptiveVolumeViewer::surfaceToScene(float surfX, float surfY) const
{
    float vpCx = static_cast<float>(_framebuffer.width()) * 0.5f;
    float vpCy = static_cast<float>(_framebuffer.height()) * 0.5f;
    return {static_cast<qreal>((surfX - _camSurfX) * _camScale + vpCx),
            static_cast<qreal>((surfY - _camSurfY) * _camScale + vpCy)};
}

cv::Vec2f CAdaptiveVolumeViewer::sceneToSurface(const QPointF& scenePos) const
{
    if (_framebuffer.isNull() || _camScale <= 0) return {0, 0};
    float vpCx = static_cast<float>(_framebuffer.width()) * 0.5f;
    float vpCy = static_cast<float>(_framebuffer.height()) * 0.5f;
    return {(static_cast<float>(scenePos.x()) - vpCx) / _camScale + _camSurfX,
            (static_cast<float>(scenePos.y()) - vpCy) / _camScale + _camSurfY};
}

QPointF CAdaptiveVolumeViewer::volumeToScene(const cv::Vec3f& volPoint)
{
    auto surf = _surfWeak.lock();
    if (!surf) return {};
    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        cv::Vec3f proj = plane->project(volPoint, 1.0, 1.0);
        return surfaceToScene(proj[0], proj[1]);
    }
    return {};
}

cv::Vec3f CAdaptiveVolumeViewer::sceneToVolume(const QPointF& scenePoint) const
{
    auto surf = _surfWeak.lock();
    if (!surf) return {0, 0, 0};
    cv::Vec2f sp = sceneToSurface(scenePoint);
    cv::Vec3f surfLoc = {sp[0], sp[1], 0};
    cv::Vec3f ptr(0, 0, 0);
    return surf->coord(ptr, surfLoc);
}

// ============================================================================
// Window/level
// ============================================================================

cv::Vec2f CAdaptiveVolumeViewer::sceneToSurfaceCoords(const QPointF& scenePos) const
{
    return sceneToSurface(scenePos);
}

bool CAdaptiveVolumeViewer::sceneToVolumePN(cv::Vec3f& p, cv::Vec3f& n,
                                             const QPointF& scenePos) const
{
    auto surf = _surfWeak.lock();
    if (!surf) {
        p = cv::Vec3f(0, 0, 0);
        n = cv::Vec3f(0, 0, 1);
        return false;
    }
    cv::Vec2f sp = sceneToSurface(scenePos);
    cv::Vec3f surfLoc = {sp[0], sp[1], 0};
    cv::Vec3f ptr(0, 0, 0);
    n = surf->normal(ptr, surfLoc);
    p = surf->coord(ptr, surfLoc);
    return true;
}

void CAdaptiveVolumeViewer::adjustSurfaceOffset(float dn)
{
    float maxZ = 10000.0f;
    if (_volume) {
        auto [w, h, d] = _volume->shape();
        maxZ = static_cast<float>(std::max({w, h, d}));
    }
    _camera.zOff = std::clamp(_camera.zOff + dn, -maxZ, maxZ);
    submitRender();
    updateStatusLabel();
}

void CAdaptiveVolumeViewer::resetSurfaceOffsets()
{
    _camera.zOff = 0.0f;
    submitRender();
}

void CAdaptiveVolumeViewer::setVolumeWindow(float low, float high)
{
    _windowLow = std::clamp(low, 0.0f, 65535.0f);
    _windowHigh = std::clamp(high, 0.0f, 65535.0f);
    if (_volume) submitRender();
}

// ============================================================================
// Overlay group management (for VolumeViewerBase)
// ============================================================================

void CAdaptiveVolumeViewer::setOverlayGroup(const std::string& key,
                                             const std::vector<QGraphicsItem*>& items)
{
    clearOverlayGroup(key);
    _overlayGroups[key] = items;
}

void CAdaptiveVolumeViewer::clearOverlayGroup(const std::string& key)
{
    auto it = _overlayGroups.find(key);
    if (it == _overlayGroups.end()) return;
    for (auto* item : it->second) {
        _scene->removeItem(item);
        delete item;
    }
    _overlayGroups.erase(it);
}

void CAdaptiveVolumeViewer::clearAllOverlayGroups()
{
    for (auto& [key, items] : _overlayGroups) {
        for (auto* item : items) {
            _scene->removeItem(item);
            delete item;
        }
    }
    _overlayGroups.clear();
}

// ============================================================================
// Status / misc
// ============================================================================

void CAdaptiveVolumeViewer::updateStatusLabel()
{
    if (!_lbl || !_volume) return;

    QString status = QString("%1x 1:%2 z=%3")
        .arg(static_cast<double>(_camera.scale), 0, 'f', 2)
        .arg(1 << _camera.dsScaleIdx)
        .arg(static_cast<double>(_camera.zOff), 0, 'f', 1);

    if (_volume->tieredCache()) {
        auto s = _volume->tieredCache()->stats();

        uint64_t total = s.blockHits + s.coldHits + s.iceFetches + s.misses;
        if (total > 0) {
            auto pct = [&](uint64_t n) { return static_cast<int>(100 * n / total); };
            status += QString(" | blk %1% cold %2%").arg(pct(s.blockHits)).arg(pct(s.coldHits));
        }
        // Block cache: MB resident out of budget (4 KiB per block).
        double blockMb = static_cast<double>(s.blocks) * 4.0 / 1024.0;
        status += QString(" | blk %1M").arg(blockMb, 0, 'f', 0);

        double diskMb = static_cast<double>(s.diskBytes) / (1024.0 * 1024.0);
        const char* unit = s.sharded ? "shard" : "chunk";
        status += QString(" | disk %1M %2%3")
            .arg(diskMb, 0, 'f', 1)
            .arg(s.diskShards)
            .arg(unit);

        if (s.sharded) {
            status += QString(" | dl %1sh w %2sh io %3 neg %4")
                .arg(s.iceFetches)
                .arg(s.diskWrites)
                .arg(s.ioPending)
                .arg(s.negativeCount);
        } else {
            status += QString(" | dl %1 w %2 io %3 neg %4")
                .arg(s.iceFetches)
                .arg(s.diskWrites)
                .arg(s.ioPending)
                .arg(s.negativeCount);
        }
    }

    status += " [adaptive]";
    _lbl->setText(status);
    _lbl->adjustSize();
    _lbl->show();
}

void CAdaptiveVolumeViewer::fitSurfaceInView()
{
    _camera.surfacePtr = cv::Vec3f(0, 0, 0);
    _camera.scale = 0.5f;
    if (_volume)
        _camera.recalcPyramidLevel(static_cast<int>(_volume->numScales()));
    submitRender();
}

bool CAdaptiveVolumeViewer::isAxisAlignedView() const
{
    return _surfName == "xy plane" || _surfName == "xz plane" ||
           _surfName == "yz plane" || _surfName == "seg xz" || _surfName == "seg yz";
}

bool CAdaptiveVolumeViewer::isWindowMinimized() const
{
    auto* sub = qobject_cast<QMdiSubWindow*>(parentWidget());
    return sub && sub->isMinimized();
}

bool CAdaptiveVolumeViewer::eventFilter(QObject* watched, QEvent* event)
{
    if (event->type() == QEvent::WindowStateChange) {
        auto* sub = qobject_cast<QMdiSubWindow*>(watched);
        if (sub && !sub->isMinimized() && _dirtyWhileMinimized) {
            _dirtyWhileMinimized = false;
            submitRender();
        }
    }
    return QWidget::eventFilter(watched, event);
}
