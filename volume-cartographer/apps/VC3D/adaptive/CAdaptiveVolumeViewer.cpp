#include "CAdaptiveVolumeViewer.hpp"

#include "ViewerManager.hpp"
#include "VCSettings.hpp"
#include "../CState.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/core/render/PostProcess.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/types/SampleParams.hpp"
#include "vc/core/cache/BlockPipeline.hpp"
#include "vc/core/cache/ChunkKey.hpp"
#include <cstring>
#include <unordered_set>

#include <opencv2/imgproc.hpp>

#include <QSettings>
#include <QTimer>
#include <QVBoxLayout>
#include <QLabel>
#include <QGraphicsScene>
#include <QGraphicsPathItem>
#include <QGraphicsEllipseItem>
#include <QMdiSubWindow>
#include <QPainterPath>
#include <QPen>
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
    // Anchor the scene at the viewport top-left, not the default AlignCenter.
    // The framebuffer draws at viewport (0,0) in drawBackground, so if the
    // scene were centered (which happens whenever sceneRect < viewport, e.g.
    // during a resize lag) mapToScene would shift mouse coords by half the
    // size delta and every edit would land up-left of the cursor.
    _view->setAlignment(Qt::AlignLeft | Qt::AlignTop);
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
    _panSensitivity = settings.value(viewer::PAN_SENSITIVITY, viewer::PAN_SENSITIVITY_DEFAULT).toFloat();
    if (_panSensitivity <= 0.0f) _panSensitivity = 1.0f;
    _zoomSensitivity = settings.value(viewer::ZOOM_SENSITIVITY, viewer::ZOOM_SENSITIVITY_DEFAULT).toFloat();
    if (_zoomSensitivity <= 0.0f) _zoomSensitivity = 1.0f;
    _zScrollSensitivity = settings.value(viewer::ZSCROLL_SENSITIVITY, viewer::ZSCROLL_SENSITIVITY_DEFAULT).toFloat();
    if (_zScrollSensitivity <= 0.0f) _zScrollSensitivity = 1.0f;
    {
        int interpIdx = settings.value(perf::INTERPOLATION_METHOD, 1).toInt();
        _samplingMethod = static_cast<vc::Sampling>(std::clamp(interpIdx, 0, 3));
    }
    _highlightDownscaled = settings.value("viewer_controls/highlight_downscaled", false).toBool();

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
    unregisterOverlayChunkListener();
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

void CAdaptiveVolumeViewer::unregisterOverlayChunkListener()
{
    if (_overlayChunkCbId != 0 && _overlayVolume && _overlayVolume->tieredCache()) {
        _overlayVolume->tieredCache()->removeChunkReadyListener(_overlayChunkCbId);
    }
    _overlayChunkCbId = 0;
}

void CAdaptiveVolumeViewer::setOverlayVolume(std::shared_ptr<Volume> volume)
{
    if (_overlayVolume == volume) {
        return;
    }

    unregisterOverlayChunkListener();
    _overlayVolume = std::move(volume);

    if (_overlayVolume && _overlayVolume->numScales() >= 1) {
        auto* cache = _overlayVolume->tieredCache();
        if (cache) {
            QPointer<CAdaptiveVolumeViewer> guard(this);
            std::weak_ptr<Volume> overlayWeak = _overlayVolume;
            _overlayChunkCbId = cache->addChunkReadyListener(
                [guard, overlayWeak](const vc::cache::ChunkKey&) {
                    QMetaObject::invokeMethod(qApp, [guard, overlayWeak]() {
                        if (!guard) return;
                        auto vol = overlayWeak.lock();
                        if (!vol || guard->_overlayVolume != vol) return;
                        if (auto* c = vol->tieredCache()) {
                            c->clearChunkArrivedFlag();
                        }
                        guard->scheduleRender();
                    }, Qt::QueuedConnection);
                });
        }
    }

    scheduleRender();
}

void CAdaptiveVolumeViewer::setOverlayOpacity(float opacity)
{
    const float clamped = std::clamp(opacity, 0.0f, 1.0f);
    if (std::abs(clamped - _overlayOpacity) < 1e-6f) {
        return;
    }
    _overlayOpacity = clamped;
    if (_overlayVolume) {
        scheduleRender();
    }
}

void CAdaptiveVolumeViewer::setOverlayColormap(const std::string& colormapId)
{
    if (_overlayColormapId == colormapId) {
        return;
    }
    _overlayColormapId = colormapId;
    if (_overlayVolume) {
        scheduleRender();
    }
}

void CAdaptiveVolumeViewer::setOverlayThreshold(float threshold)
{
    setOverlayWindow(std::max(threshold, 0.0f), _overlayWindowHigh);
}

void CAdaptiveVolumeViewer::setOverlayWindow(float low, float high)
{
    constexpr float kMaxOverlayValue = 255.0f;
    const float clampedLow = std::clamp(low, 0.0f, kMaxOverlayValue);
    float clampedHigh = std::clamp(high, 0.0f, kMaxOverlayValue);
    if (clampedHigh <= clampedLow) {
        clampedHigh = std::min(kMaxOverlayValue, clampedLow + 1.0f);
    }

    const bool unchanged = std::abs(clampedLow - _overlayWindowLow) < 1e-6f &&
                           std::abs(clampedHigh - _overlayWindowHigh) < 1e-6f;
    if (unchanged) {
        return;
    }

    _overlayWindowLow = clampedLow;
    _overlayWindowHigh = clampedHigh;
    if (_overlayVolume) {
        scheduleRender();
    }
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
        // Capture a weak_ptr to the Volume, not a raw cache pointer, so the
        // queued UI callback can't dereference a BlockPipeline that was
        // destroyed by a volume swap. removeChunkReadyListener prevents
        // NEW notifications from firing, but doesn't cancel Qt-queued
        // events that were already in flight — so we have to re-check at
        // dispatch time whether the cache we were notified about is still
        // the viewer's current cache, and only then touch it.
        std::weak_ptr<Volume> volumeWeak = _volume;
        _chunkCbId = cache->addChunkReadyListener(
            [guard, volumeWeak](const vc::cache::ChunkKey&) {
                QMetaObject::invokeMethod(qApp, [guard, volumeWeak]() {
                    if (!guard) return;
                    auto vol = volumeWeak.lock();
                    if (!vol || guard->_volume != vol) return;
                    if (auto* c = vol->tieredCache()) {
                        c->clearChunkArrivedFlag();
                    }
                    guard->scheduleRender();
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

    scheduleRender();
    updateStatusLabel();
}

void CAdaptiveVolumeViewer::onSurfaceChanged(const std::string& name,
                                              const std::shared_ptr<Surface>& surf,
                                              bool /*isEditUpdate*/)
{
    const bool isCurrentSurface = (_surfName == name);
    const bool isIntersectionTarget =
        _intersectTgts.count(name) != 0 ||
        (_intersectTgts.count("visible_segmentation") != 0 &&
         (name == "segmentation" || _highlightedSurfaceIds.count(name) != 0));

    if (!isCurrentSurface) {
        if (isIntersectionTarget) {
            invalidateIntersect(name);
            _lastIntersectFp = {};
            renderIntersections();
        }
        return;
    }

    _surfWeak = surf;
    // Any surface change (swap or in-place edit) invalidates the cached
    // gen() output — geometry may have shifted.
    _genCacheDirty = true;

    if (!surf) {
        _scene->clear();
        _overlayGroups.clear();
        // QGraphicsScene::clear() delete'd every QGraphicsItem we had cached
        // pointers to. Drop the dangling handles so the next
        // invalidateIntersect() doesn't double-free.
        _intersectionItems.clear();
        _focusMarker = nullptr;
        _lastIntersectFp = {};
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

    // Coalesce rapid surface updates (e.g. paint strokes emit surfaceChanged
    // on every edit) through the 60fps render timer. A burst of 30 edits
    // used to fire 30 full submitRender calls; now it fires one per tick.
    scheduleRender();
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
    // Unregister the chunk-ready listener on the BlockPipeline that is
    // about to be destroyed, and drop our own reference to the Volume so
    // we don't keep the cache resident.
    if (_chunkCbId != 0 && _volume && _volume->tieredCache()) {
        _volume->tieredCache()->removeChunkReadyListener(_chunkCbId);
    }
    _chunkCbId = 0;
    onSurfaceChanged(_surfName, nullptr);
    _volume.reset();
}

void CAdaptiveVolumeViewer::onPOIChanged(const std::string& name, POI* poi)
{
    if (name != "focus" || !poi) return;

    auto surf = _surfWeak.lock();
    auto* plane = dynamic_cast<PlaneSurface*>(surf.get());

    if (plane) {
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
    }

    updateFocusMarker(poi);
    emit overlaysUpdated();
    scheduleRender();
}

// ============================================================================
// Rendering
// ============================================================================

void CAdaptiveVolumeViewer::scheduleRender()
{
    syncCameraTransform();
    _renderPending = true;
    if (!_renderTimer->isActive())
        _renderTimer->start();
}

void CAdaptiveVolumeViewer::syncCameraTransform()
{
    _camSurfX = _camera.surfacePtr[0];
    _camSurfY = _camera.surfacePtr[1];
    _camScale = _camera.scale;
}


void CAdaptiveVolumeViewer::reloadPerfSettings()
{
    QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
    using namespace vc3d::settings;
    _panSensitivity = std::max(0.01f, s.value(viewer::PAN_SENSITIVITY, viewer::PAN_SENSITIVITY_DEFAULT).toFloat());
    _zoomSensitivity = std::max(0.01f, s.value(viewer::ZOOM_SENSITIVITY, viewer::ZOOM_SENSITIVITY_DEFAULT).toFloat());
    _zScrollSensitivity = std::max(0.01f, s.value(viewer::ZSCROLL_SENSITIVITY, viewer::ZSCROLL_SENSITIVITY_DEFAULT).toFloat());
    int interpIdx = s.value(perf::INTERPOLATION_METHOD, 1).toInt();
    _samplingMethod = static_cast<vc::Sampling>(std::clamp(interpIdx, 0, 3));
    _highlightDownscaled = s.value("viewer_controls/highlight_downscaled", false).toBool();
}

void CAdaptiveVolumeViewer::submitRender()
{
    const CompositeParams& lightP = _compositeSettings.params;
    const bool rakingEnabled = _compositeSettings.postRakingEnabled;
    const float rakingAz = _compositeSettings.postRakingAzimuth;
    const float rakingEl = _compositeSettings.postRakingElevation;
    const float rakingStrength = std::clamp(_compositeSettings.postRakingStrength, 0.0f, 1.0f);
    const float rakingDepth = std::max(0.01f, _compositeSettings.postRakingDepthScale);

    // Debug overlay: paint a per-pixel gradient based on fallback-level depth.
    // Cached in reloadPerfSettings() instead of re-read from disk each frame.
    const bool highlightDownscaled = _highlightDownscaled;

    auto surf = _surfWeak.lock();
    if (!surf || !_volume || !_volume->zarrDataset()) return;

    int fbW = _framebuffer.width();
    int fbH = _framebuffer.height();
    if (fbW <= 0 || fbH <= 0) return;

    // Always populate the level buffer — the debug overlay is optional, but
    // we need the per-pixel fallback depth to enqueue the missing high-res
    // chunks below regardless of whether the overlay is shown.
    if (_levelBuffer.rows != fbH || _levelBuffer.cols != fbW) {
        _levelBuffer.create(fbH, fbW);
    }
    _levelBuffer.setTo(0);
    uint8_t* lvlOutPtr = _levelBuffer.ptr<uint8_t>(0);
    const int lvlOutStride = int(_levelBuffer.step1());

    auto* fbBits = reinterpret_cast<uint32_t*>(_framebuffer.bits());
    int fbStride = _framebuffer.bytesPerLine() / 4;

    // Build the render LUT. For stretch mode we use last frame's min/max
    // so the render is a single pass; we refresh the cached range by
    // scanning the framebuffer after. A camera change invalidates the
    // cache and forces a 2-pass on the first frame after motion.
    const bool stretch = _compositeSettings.postStretchValues;
    const uint8_t isoCutoff = _compositeSettings.params.isoCutoff;
    auto applyIsoCutoff = [&](std::array<uint32_t, 256>& l, uint8_t cutoff) {
        if (cutoff == 0) return;
        const uint32_t zero = l[0];
        for (int i = 0; i < cutoff; i++) l[i] = zero;
    };

    std::array<uint32_t, 256> lut;
    const bool stretchFirstPass = stretch && !_cachedStretchValid;
    // CLAHE and raking both operate on gray, so the colormap is deferred
    // until after those passes. The sampling LUT in that case is gray-only.
    const bool postGrayDomain = _compositeSettings.postClaheEnabled || rakingEnabled;
    const bool deferColormap = postGrayDomain && !_baseColormapId.empty();
    const std::string& sampleColormapId = deferColormap
        ? std::string() : _baseColormapId;
    if (stretchFirstPass) {
        // Identity gray LUT so we can extract the raw sample after sampling.
        for (int i = 0; i < 256; i++) {
            uint32_t v = uint32_t(i);
            lut[i] = 0xFF000000u | (v << 16) | (v << 8) | v;
        }
    } else {
        float wlo = stretch ? float(_cachedStretchLo) : _windowLow;
        float whi = stretch ? float(_cachedStretchHi) : _windowHigh;
        // Reuse previous LUT if the inputs haven't changed.
        if (_cachedWindowLow == wlo && _cachedWindowHigh == whi
            && _cachedColormapId == sampleColormapId
            && _cachedIsoCutoff == isoCutoff) {
            lut = _cachedLut;
        } else {
            vc::buildWindowLevelColormapLut(lut, wlo, whi, sampleColormapId);
            applyIsoCutoff(lut, isoCutoff);
            _cachedLut = lut;
            _cachedWindowLow = wlo;
            _cachedWindowHigh = whi;
            _cachedColormapId = sampleColormapId;
            _cachedIsoCutoff = isoCutoff;
        }
    }

    vc::SampleParams sp;
    const int numLevels = static_cast<int>(_volume->numScales());
    // Always render at the user-requested level. The sampler's per-pixel
    // adaptive fallback handles regions that aren't ready yet by dropping
    // those pixels to whichever coarser level is resident — no whole-frame
    // resolution cycling, and cached fine chunks are used immediately.
    sp.level = _camera.dsScaleIdx;
    sp.method = _samplingMethod;

    const cv::Mat_<cv::Vec3f>* overlayCoords = nullptr;
    cv::Vec3f overlayOrigin(0.0f, 0.0f, 0.0f);
    cv::Vec3f overlayVxStep(0.0f, 0.0f, 0.0f);
    cv::Vec3f overlayVyStep(0.0f, 0.0f, 0.0f);
    cv::Vec3f overlayPlaneNormal(0.0f, 0.0f, 1.0f);
    bool haveOverlayPlane = false;

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
        overlayOrigin = origin;
        overlayVxStep = vx_step;
        overlayVyStep = vy_step;
        overlayPlaneNormal = n;
        haveOverlayPlane = true;

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
            fbW, fbH, method, lut.data(), sampleMethod,
            &lightP,  // sampler uses lightP for volumetric and lighting paths
            lvlOutPtr, lvlOutStride);
    } else {
        // surf->gen treats offset as the TOP-LEFT of the rendered region in
        // scaled surface units, but camera.surfacePtr is the CENTRE of the
        // view (matching sceneToSurface). Shift by half the viewport so the
        // rendered pixels and the mouse→surface math agree — without this
        // every edit lands up-left of the cursor by half the viewport.
        // Don't pass zOff into surf->gen — its per-pixel-normal offset makes
        // zoom expose curvature drift on a non-planar surface. Instead build
        // the base (unoffset) coords here and apply zOff as a single rigid
        // world-space translation in _zOffWorldDir below.
        cv::Vec3f offset(_camera.surfacePtr[0] * _camera.scale - float(fbW) * 0.5f,
                         _camera.surfacePtr[1] * _camera.scale - float(fbH) * 0.5f,
                         0.0f);
        const bool wantComposite = _compositeSettings.enabled;
        // Always request normals so shift+scroll can sample the view-center
        // normal without a separate gen pass.
        const bool cacheHit =
            !_genCacheDirty
            && _genCacheSurfKey == surf.get()
            && _genCacheFbW == fbW
            && _genCacheFbH == fbH
            && _genCacheScale == _camera.scale
            && _genCacheOffset == offset
            && _genCacheWantComposite == wantComposite
            && _genCacheZOff == _camera.zOff
            && _genCacheZOffDir == _zOffWorldDir
            && !_genCoords.empty();
        if (!cacheHit) {
            surf->gen(&_genCoords, &_genNormals,
                      cv::Size(fbW, fbH), cv::Vec3f(0, 0, 0),
                      _camera.scale, offset);
            // Lazy-capture the translation direction when zOff was set by a
            // path that didn't populate _zOffWorldDir (adjustSurfaceOffset
            // via Ctrl+./Ctrl+, shortcuts, or any other non-Shift-scroll
            // source). Without this, those offsets would be silent no-ops
            // until the user first Shift-scrolled.
            if (_camera.zOff != 0.0f &&
                _zOffWorldDir[0] == 0.0f && _zOffWorldDir[1] == 0.0f && _zOffWorldDir[2] == 0.0f &&
                !_genNormals.empty()) {
                const int cy = _genNormals.rows / 2;
                const int cx = _genNormals.cols / 2;
                const cv::Vec3f n = _genNormals(cy, cx);
                if (std::isfinite(n[0]) && std::isfinite(n[1]) && std::isfinite(n[2])) {
                    const float len = static_cast<float>(cv::norm(n));
                    if (len > 1e-6f) {
                        _zOffWorldDir = n / len;
                    }
                }
            }
            // Apply z-offset as a rigid world-space translation using the
            // cached direction. On cache hits _genCoords is already shifted
            // — avoid double-applying by only running this on cache miss.
            if (_camera.zOff != 0.0f &&
                (_zOffWorldDir[0] != 0.0f || _zOffWorldDir[1] != 0.0f || _zOffWorldDir[2] != 0.0f) &&
                !_genCoords.empty()) {
                const cv::Vec3f tr = _zOffWorldDir * _camera.zOff;
                const int rows = _genCoords.rows;
                const int cols = _genCoords.cols;
                for (int y = 0; y < rows; ++y) {
                    cv::Vec3f* row = _genCoords.ptr<cv::Vec3f>(y);
                    for (int x = 0; x < cols; ++x) {
                        cv::Vec3f& p = row[x];
                        // Skip invalid sentinels (NaN or -1 marker).
                        if (p[0] != p[0] || p[0] == -1.0f) continue;
                        p += tr;
                    }
                }
            }
            _genCacheSurfKey = surf.get();
            _genCacheFbW = fbW;
            _genCacheFbH = fbH;
            _genCacheScale = _camera.scale;
            _genCacheOffset = offset;
            _genCacheWantComposite = wantComposite;
            _genCacheZOff = _camera.zOff;
            _genCacheZOffDir = _zOffWorldDir;
            _genCacheDirty = false;
        }
        cv::Mat_<cv::Vec3f>& coords = _genCoords;
        cv::Mat_<cv::Vec3f>& normals = _genNormals;

        if (!coords.empty()) {
            overlayCoords = &coords;
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
                fbW, fbH, method, lut.data(), sampleMethod,
            &lightP,  // sampler uses lightP for volumetric and lighting paths
            lvlOutPtr, lvlOutStride);
        }
    }

    // Stretch handling:
    //   - First pass after a camera change: the identity LUT is active;
    //     scan min/max, build the stretched LUT, re-blit. Cache the range
    //     so subsequent frames render in a single pass.
    //   - Subsequent frames (cached valid): the stretched LUT already
    //     ran inside sampleAdaptiveARGB32. Refresh the cached min/max in
    //     the background by sampling the framebuffer so the stretch
    //     tracks drifting scene content.
    if (stretch) {
        // Scanning w*h pixels for min/max every frame is wasteful during
        // pans where content drifts only slowly. On the first pass we
        // always scan (the 2-pass LUT rebuild depends on it); otherwise
        // throttle to ~every 150 ms so the stretch still tracks drifting
        // content without blocking every frame.
        int lo = _cachedStretchLo, hi = _cachedStretchHi;
        const auto now = std::chrono::steady_clock::now();
        // Always scan when the cached bounds haven't been initialized,
        // otherwise the first frame after a fresh invalidation would use
        // the stale default 0/255 bounds for the whole 150 ms hysteresis
        // window.
        const bool doScan = stretchFirstPass
            || !_cachedStretchValid
            || (now - _lastStretchScan) > std::chrono::milliseconds(150);
        if (doScan) {
            lo = 255; hi = 0;
            for (int y = 0; y < fbH; y++) {
                const uint32_t* row = fbBits + size_t(y) * size_t(fbStride);
                for (int x = 0; x < fbW; x++) {
                    int v = int(row[x] & 0xFFu);
                    if (v < lo) lo = v;
                    if (v > hi) hi = v;
                }
            }
            _lastStretchScan = now;
        }
        if (stretchFirstPass) {
            // One-time 2-pass: re-LUT the identity output we just rendered.
            std::array<uint32_t, 256> stretchedLut;
            if (hi > lo) {
                vc::buildWindowLevelColormapLut(stretchedLut,
                    float(lo), float(hi), _baseColormapId);
            } else {
                vc::buildWindowLevelColormapLut(stretchedLut,
                    _windowLow, _windowHigh, _baseColormapId);
            }
            applyIsoCutoff(stretchedLut, isoCutoff);
            for (int y = 0; y < fbH; y++) {
                uint32_t* row = fbBits + size_t(y) * size_t(fbStride);
                for (int x = 0; x < fbW; x++) {
                    row[x] = stretchedLut[row[x] & 0xFFu];
                }
            }
            // Populate cached LUT for the next render.
            _cachedLut = stretchedLut;
            _cachedWindowLow = float(lo);
            _cachedWindowHigh = float(hi);
            _cachedColormapId = _baseColormapId;
            _cachedIsoCutoff = isoCutoff;
        }
        if (hi > lo) {
            _cachedStretchLo = lo;
            _cachedStretchHi = hi;
            _cachedStretchValid = true;
        }
    }

    // CLAHE post-pass — runs on grayscale before any colormap is applied.
    // When a colormap is selected we also run the colormap LUT here.
    if (postGrayDomain) {
        // Reuse the member buffer; create() is a no-op when size matches.
        _grayBuf.create(fbH, fbW);
        cv::Mat_<uint8_t>& gray = _grayBuf;
        for (int y = 0; y < fbH; y++) {
            const uint32_t* row = fbBits + size_t(y) * size_t(fbStride);
            uint8_t* dst = gray.ptr<uint8_t>(y);
            for (int x = 0; x < fbW; x++) dst[x] = uint8_t(row[x] & 0xFFu);
        }
        if (_compositeSettings.postClaheEnabled) {
            const int tile = std::max(1, _compositeSettings.postClaheTileSize);
            const double clip = std::max(0.01, double(_compositeSettings.postClaheClipLimit));
            // Cache the CLAHE instance: it allocates internal histogram
            // buffers on construction. Rebuild only when the parameters
            // actually change.
            if (!_claheCache || tile != _claheCacheTile || clip != _claheCacheClip) {
                _claheCache = cv::createCLAHE(clip, cv::Size(tile, tile));
                _claheCacheTile = tile;
                _claheCacheClip = clip;
            }
            _claheCache->apply(gray, gray);
        }
        if (rakingEnabled) {
            // Treat gray as a heightfield. Scharr gives a screen-space
            // gradient; compose a surface normal (depth scales vertical
            // slope vs. unit-height image plane) and Lambert-shade it.
            // Lazy-allocate the gradient matrices — raking is off by
            // default, so unconditionally reserving ~16 MB/viewer at 1080p
            // for this path would be wasteful. Once raking turns on we
            // reuse the matrices across frames.
            if (_rakingGx.rows != fbH || _rakingGx.cols != fbW
                || _rakingGx.type() != CV_32F) {
                _rakingGx = cv::Mat(fbH, fbW, CV_32F);
                _rakingGy = cv::Mat(fbH, fbW, CV_32F);
            }
            cv::Scharr(gray, _rakingGx, CV_32F, 1, 0, 1.0 / 32.0);
            cv::Scharr(gray, _rakingGy, CV_32F, 0, 1, 1.0 / 32.0);
            cv::Mat& gx = _rakingGx;
            cv::Mat& gy = _rakingGy;
            const float azRad = rakingAz * float(M_PI) / 180.0f;
            const float elRad = rakingEl * float(M_PI) / 180.0f;
            const float ce = std::cos(elRad);
            const float Lx = ce * std::cos(azRad);
            const float Ly = ce * std::sin(azRad);
            const float Lz = std::sin(elRad);
            const float strength = rakingStrength;
            const float ambient = 1.0f - strength;
            const float depth = rakingDepth;
            #pragma omp parallel for
            for (int y = 0; y < fbH; y++) {
                uint8_t* dst = gray.ptr<uint8_t>(y);
                const float* gxr = gx.ptr<float>(y);
                const float* gyr = gy.ptr<float>(y);
                for (int x = 0; x < fbW; x++) {
                    const float nx = -gxr[x] * depth;
                    const float ny = -gyr[x] * depth;
                    const float nz = 1.0f;
                    const float invLen = 1.0f
                        / std::sqrt(nx * nx + ny * ny + nz * nz);
                    const float nDotL = (nx * Lx + ny * Ly + nz * Lz) * invLen;
                    float lit = ambient + strength * std::max(0.0f, nDotL);
                    if (lit > 1.0f) lit = 1.0f;
                    float v = float(dst[x]) * lit;
                    if (v < 0.0f) v = 0.0f;
                    if (v > 255.0f) v = 255.0f;
                    dst[x] = uint8_t(v);
                }
            }
        }

        if (deferColormap) {
            // Identity window/level (gray is already windowed) + user colormap.
            // Cache the built LUT: the identity-window LUT depends only on
            // _baseColormapId, so we rebuild only when the colormap actually
            // changes instead of once per frame.
            if (!_deferredCmapValid || _deferredCmapId != _baseColormapId) {
                vc::buildWindowLevelColormapLut(_deferredCmapLut, 0.0f, 255.0f, _baseColormapId);
                _deferredCmapId = _baseColormapId;
                _deferredCmapValid = true;
            }
            const auto& cmapLut = _deferredCmapLut;
            for (int y = 0; y < fbH; y++) {
                uint32_t* row = fbBits + size_t(y) * size_t(fbStride);
                const uint8_t* src = gray.ptr<uint8_t>(y);
                for (int x = 0; x < fbW; x++) row[x] = cmapLut[src[x]];
            }
        } else {
            for (int y = 0; y < fbH; y++) {
                uint32_t* row = fbBits + size_t(y) * size_t(fbStride);
                const uint8_t* src = gray.ptr<uint8_t>(y);
                for (int x = 0; x < fbW; x++) {
                    uint32_t v = src[x];
                    row[x] = 0xFF000000u | (v << 16) | (v << 8) | v;
                }
            }
        }
    }

    renderOverlayVolume(
        fbBits, fbStride, fbW, fbH,
        overlayCoords,
        haveOverlayPlane ? &overlayOrigin : nullptr,
        haveOverlayPlane ? &overlayVxStep : nullptr,
        haveOverlayPlane ? &overlayVyStep : nullptr,
        haveOverlayPlane ? &overlayPlaneNormal : nullptr);

    if (highlightDownscaled && lvlOutPtr
        && _levelBuffer.rows == fbH && _levelBuffer.cols == fbW) {
        // Fallback-depth gradient: green (1 level coarser) → red (5+ coarser).
        // Encoded as ARGB with 0xFF alpha so the blend below can mix against
        // the framebuffer without per-channel premultiplication logic.
        static const uint32_t kLevelColors[6] = {
            0u,                      // 0: desired level — no overlay
            0xFF34D399u,             // 1: green
            0xFF4ADE80u,             // 2: lime
            0xFFFACC15u,             // 3: yellow
            0xFFFB923Cu,             // 4: orange
            0xFFEF4444u,             // 5+: red
        };
        // Linear blend `out = (src * (256-a) + color * a) / 256` per channel.
        // a = 96 ≈ 37% overlay — enough to read the tint without hiding data.
        constexpr uint32_t kA = 96;
        constexpr uint32_t kIA = 256 - kA;
        for (int y = 0; y < fbH; y++) {
            uint32_t* row = fbBits + size_t(y) * size_t(fbStride);
            const uint8_t* lvls = _levelBuffer.ptr<uint8_t>(y);
            for (int x = 0; x < fbW; x++) {
                const uint8_t lvl = lvls[x];
                if (lvl == 0) continue;
                const uint32_t c = kLevelColors[std::min<uint8_t>(lvl, 5)];
                const uint32_t src = row[x];
                const uint32_t sr = (src >> 16) & 0xFFu;
                const uint32_t sg = (src >> 8) & 0xFFu;
                const uint32_t sb = src & 0xFFu;
                const uint32_t cr = (c >> 16) & 0xFFu;
                const uint32_t cg = (c >> 8) & 0xFFu;
                const uint32_t cb = c & 0xFFu;
                const uint32_t r = (sr * kIA + cr * kA) >> 8;
                const uint32_t g = (sg * kIA + cg * kA) >> 8;
                const uint32_t b = (sb * kIA + cb * kA) >> 8;
                row[x] = 0xFF000000u | (r << 16) | (g << 8) | b;
            }
        }
    }

    // Update camera tracking for coordinate conversions
    syncCameraTransform();

    updateFocusMarker();
    renderIntersections();
    emit overlaysUpdated();
    // update() schedules a deferred repaint via the event loop; repaint()
    // blocks the UI thread synchronously until paintEvent returns, which
    // stalls every frame during pans/zooms.
    _view->viewport()->update();
    updateStatusLabel();
}

void CAdaptiveVolumeViewer::renderVisible(bool force)
{
    if (!force) {
        scheduleRender();
        return;
    }

    if (_renderTimer && _renderTimer->isActive()) {
        _renderTimer->stop();
    }
    _renderPending = false;
    submitRender();
    updateStatusLabel();
}

void CAdaptiveVolumeViewer::centerOnVolumePoint(const cv::Vec3f& point, bool forceRender)
{
    auto surf = _surfWeak.lock();
    if (!surf) {
        return;
    }

    cv::Vec2f surfacePoint(0.0f, 0.0f);
    bool haveSurfacePoint = false;
    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        const cv::Vec3f projected = plane->project(point, 1.0, 1.0);
        surfacePoint = {projected[0], projected[1]};
        haveSurfacePoint = true;
    } else if (auto* quad = dynamic_cast<QuadSurface*>(surf.get())) {
        cv::Vec3f ptr = quad->pointer();
        auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
        if (quad->pointTo(ptr, point, 4.0f, 100, patchIndex) >= 0.0f) {
            const cv::Vec3f loc = quad->loc(ptr);
            surfacePoint = {loc[0], loc[1]};
            haveSurfacePoint = true;
        }
    }

    if (!haveSurfacePoint ||
        !std::isfinite(surfacePoint[0]) ||
        !std::isfinite(surfacePoint[1])) {
        return;
    }

    _camera.surfacePtr[0] = surfacePoint[0];
    _camera.surfacePtr[1] = surfacePoint[1];
    _cachedStretchValid = false;
    syncCameraTransform();

    if (forceRender) {
        renderVisible(true);
    } else {
        scheduleRender();
    }
    emit overlaysUpdated();
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

    _cachedStretchValid = false;
    scheduleRender();
    emit overlaysUpdated();
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

    _cachedStretchValid = false;
    // Zoom can fire as fast as the keyboard repeats. scheduleRender()
    // coalesces bursts into the 60 fps render timer so we don't render
    // dozens of intermediate frames the user never sees.
    scheduleRender();
    emit overlaysUpdated();
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
            // Direct z-offset — rigid translation along the surface normal at
            // view center (captured fresh each shift+scroll).
            float maxZ = 10000.0f;
            if (_volume) {
                auto [w, h, d] = _volume->shape();
                maxZ = static_cast<float>(std::max({w, h, d}));
            }
            // Capture the translation direction from the normal at view
            // center. _genNormals is populated every render (we always
            // request normals on the flattened path).
            if (!_genNormals.empty()) {
                const int cy = _genNormals.rows / 2;
                const int cx = _genNormals.cols / 2;
                const cv::Vec3f n = _genNormals(cy, cx);
                if (std::isfinite(n[0]) && std::isfinite(n[1]) && std::isfinite(n[2])) {
                    const float len = static_cast<float>(cv::norm(n));
                    if (len > 1e-6f) {
                        _zOffWorldDir = n / len;
                    }
                }
            }
            _camera.zOff = std::clamp(_camera.zOff + dz, -maxZ, maxZ);
            scheduleRender();
            updateStatusLabel();
        }
    } else if (modifiers & Qt::ControlModifier) {
        // Ctrl+wheel is bound to segmentation brush-radius changes
        // (SegmentationModule connects sendSegmentationRadiusWheel). Without
        // this branch the event falls into the zoom path and the in-editor
        // radius shortcut silently stops working.
        emit sendSegmentationRadiusWheel(steps, scenePoint, sceneToVolume(scenePoint));
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
    scheduleRender();
    emit overlaysUpdated();
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
    scheduleRender();
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
    if (_bboxMode && _surfName == "segmentation") {
        if (button == Qt::LeftButton) {
            auto surf = _surfWeak.lock();
            if (!dynamic_cast<QuadSurface*>(surf.get())) {
                return;
            }
            const cv::Vec2f sp = sceneToSurface(scenePos);
            if (!std::isfinite(sp[0]) || !std::isfinite(sp[1])) {
                return;
            }
            _bboxStartSurf = QPointF(sp[0], sp[1]);
            _activeBBoxSurfRect = QRectF(_bboxStartSurf, _bboxStartSurf).normalized();
            emit overlaysUpdated();
        }
        return;
    }

    cv::Vec3f p = sceneToVolume(scenePos);
    cv::Vec3f n(0, 0, 1);
    emit sendMousePressVolume(p, n, button, modifiers);
}

void CAdaptiveVolumeViewer::onMouseMove(QPointF scenePos, Qt::MouseButtons buttons,
                                         Qt::KeyboardModifiers modifiers)
{
    if (_bboxMode && _surfName == "segmentation") {
        _lastScenePos = scenePos;
        if (_activeBBoxSurfRect && (buttons & Qt::LeftButton)) {
            auto surf = _surfWeak.lock();
            if (!dynamic_cast<QuadSurface*>(surf.get())) {
                return;
            }
            const cv::Vec2f sp = sceneToSurface(scenePos);
            if (!std::isfinite(sp[0]) || !std::isfinite(sp[1])) {
                return;
            }
            const QPointF cur(sp[0], sp[1]);
            _activeBBoxSurfRect = QRectF(_bboxStartSurf, cur).normalized();
            emit overlaysUpdated();
        }
        return;
    }

    cv::Vec3f p = sceneToVolume(scenePos);
    emit sendMouseMoveVolume(p, buttons, modifiers);
}

void CAdaptiveVolumeViewer::onMouseRelease(QPointF scenePos, Qt::MouseButton button,
                                            Qt::KeyboardModifiers modifiers)
{
    if (_bboxMode && _surfName == "segmentation") {
        if (button == Qt::LeftButton && _activeBBoxSurfRect) {
            const QRectF rSurf = _activeBBoxSurfRect->normalized();
            const int idx = static_cast<int>(_selections.size());
            const QColor color = QColor::fromHsv((idx * 53) % 360, 200, 255);
            _selections.push_back({rSurf, color});
            _activeBBoxSurfRect.reset();
            emit overlaysUpdated();
        }
        return;
    }

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
    // Framebuffer is drawn at viewport (0,0) in CVolumeViewerView::drawBackground.
    // Do the surface→viewport math first, then map viewport→scene through the
    // view so any transform/alignment/scroll state is accounted for.
    float vpCx = static_cast<float>(_framebuffer.width()) * 0.5f;
    float vpCy = static_cast<float>(_framebuffer.height()) * 0.5f;
    const qreal vx = (surfX - _camSurfX) * _camScale + vpCx;
    const qreal vy = (surfY - _camSurfY) * _camScale + vpCy;
    return _view->mapToScene(QPointF(vx, vy).toPoint());
}

cv::Vec2f CAdaptiveVolumeViewer::sceneToSurface(const QPointF& scenePos) const
{
    if (_framebuffer.isNull() || _camScale <= 0) return {0, 0};
    // Reverse: scene→viewport first (undoes any view transform), then the
    // framebuffer-centered surface math. Matches surfaceToScene exactly.
    QPoint vp = _view->mapFromScene(scenePos);
    float vpCx = static_cast<float>(_framebuffer.width()) * 0.5f;
    float vpCy = static_cast<float>(_framebuffer.height()) * 0.5f;
    return {(static_cast<float>(vp.x()) - vpCx) / _camScale + _camSurfX,
            (static_cast<float>(vp.y()) - vpCy) / _camScale + _camSurfY};
}

QRectF CAdaptiveVolumeViewer::surfaceRectToSceneRect(const QRectF& surfRect) const
{
    const QRectF r = surfRect.normalized();
    return QRectF(surfaceToScene(static_cast<float>(r.left()), static_cast<float>(r.top())),
                  surfaceToScene(static_cast<float>(r.right()), static_cast<float>(r.bottom())))
        .normalized();
}

QRectF CAdaptiveVolumeViewer::sceneRectToSurfaceRect(const QRectF& sceneRect) const
{
    const QRectF r = sceneRect.normalized();
    const std::array<QPointF, 4> corners = {
        r.topLeft(),
        r.topRight(),
        r.bottomLeft(),
        r.bottomRight(),
    };

    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();
    for (const auto& corner : corners) {
        const cv::Vec2f sp = sceneToSurface(corner);
        minX = std::min(minX, sp[0]);
        minY = std::min(minY, sp[1]);
        maxX = std::max(maxX, sp[0]);
        maxY = std::max(maxY, sp[1]);
    }

    return QRectF(QPointF(minX, minY), QPointF(maxX, maxY)).normalized();
}

QPointF CAdaptiveVolumeViewer::volumeToScene(const cv::Vec3f& volPoint)
{
    auto surf = _surfWeak.lock();
    if (!surf) return {};
    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        cv::Vec3f proj = plane->project(volPoint, 1.0, 1.0);
        return surfaceToScene(proj[0], proj[1]);
    }
    if (auto* quad = dynamic_cast<QuadSurface*>(surf.get())) {
        cv::Vec3f ptr = quad->pointer();
        auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
        float dist = quad->pointTo(ptr, volPoint, 4.0f, 100, patchIndex);
        if (dist < 0.0f) return {};
        cv::Vec3f loc = quad->loc(ptr);
        return surfaceToScene(loc[0], loc[1]);
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

void CAdaptiveVolumeViewer::updateFocusMarker(POI* poi)
{
    if (!_scene) return;
    if (!poi && _state) {
        poi = _state->poi("focus");
    }
    if (!poi || !_surfWeak.lock()) {
        if (_focusMarker) _focusMarker->hide();
        return;
    }

    if (!_focusMarker || !_focusMarker->scene()) {
        auto* marker = new QGraphicsEllipseItem(-10.0, -10.0, 20.0, 20.0);
        QPen pen(QColor(50, 255, 215), 3.0, Qt::DashDotLine, Qt::RoundCap, Qt::RoundJoin);
        pen.setCosmetic(true);
        marker->setPen(pen);
        marker->setBrush(Qt::NoBrush);
        marker->setZValue(110.0);
        _scene->addItem(marker);
        _focusMarker = marker;
    }

    const QPointF scenePos = volumeToScene(poi->p);
    if (!std::isfinite(scenePos.x()) || !std::isfinite(scenePos.y())) {
        _focusMarker->hide();
        return;
    }

    _focusMarker->setPos(scenePos);
    _focusMarker->show();
}

void CAdaptiveVolumeViewer::renderOverlayVolume(uint32_t* fbBits,
                                                int fbStride,
                                                int fbW,
                                                int fbH,
                                                const cv::Mat_<cv::Vec3f>* coords,
                                                const cv::Vec3f* origin,
                                                const cv::Vec3f* vxStep,
                                                const cv::Vec3f* vyStep,
                                                const cv::Vec3f* planeNormal)
{
    if (!fbBits || fbW <= 0 || fbH <= 0 || !_overlayVolume || _overlayOpacity <= 0.0f) {
        return;
    }

    const bool hasCoords = coords && !coords->empty();
    const bool hasPlane = origin && vxStep && vyStep && planeNormal;
    if (!hasCoords && !hasPlane) {
        return;
    }

    const int overlayLevels = static_cast<int>(_overlayVolume->numScales());
    if (overlayLevels <= 0) {
        return;
    }
    auto* overlayCache = _overlayVolume->tieredCache();
    if (!overlayCache) {
        return;
    }

    const int desiredLevel = std::clamp(_camera.dsScaleIdx, 0, overlayLevels - 1);

    if (_cachedOverlayWindowLow != _overlayWindowLow ||
        _cachedOverlayWindowHigh != _overlayWindowHigh ||
        _cachedOverlayColormapId != _overlayColormapId) {
        vc::buildWindowLevelColormapLut(
            _cachedOverlayLut, _overlayWindowLow, _overlayWindowHigh, _overlayColormapId);

        const int low = static_cast<int>(std::clamp(_overlayWindowLow, 0.0f, 255.0f));
        for (int i = 0; i < low; ++i) {
            _cachedOverlayLut[i] = 0u;
        }
        // The adaptive sampler returns lut[0] for missing chunks. Keeping
        // zero transparent prevents an unloaded overlay from darkening the
        // streamed base volume while its chunks are still arriving.
        _cachedOverlayLut[0] = 0u;

        _cachedOverlayWindowLow = _overlayWindowLow;
        _cachedOverlayWindowHigh = _overlayWindowHigh;
        _cachedOverlayColormapId = _overlayColormapId;
    }

    if (_overlayBufferW != fbW || _overlayBufferH != fbH) {
        _overlayBuffer.assign(size_t(fbW) * size_t(fbH), 0u);
        _overlayBufferW = fbW;
        _overlayBufferH = fbH;
    }

    sampleAdaptiveARGB32(
        _overlayBuffer.data(), fbW,
        overlayCache, desiredLevel, overlayLevels,
        hasCoords ? coords : nullptr,
        hasPlane ? origin : nullptr,
        hasPlane ? vxStep : nullptr,
        hasPlane ? vyStep : nullptr,
        nullptr,
        hasPlane ? planeNormal : nullptr,
        1, 0, 1.0f,
        fbW, fbH, std::string(), _cachedOverlayLut.data(), vc::Sampling::Nearest,
        nullptr, nullptr, 0);

    const int alpha = static_cast<int>(std::lround(std::clamp(_overlayOpacity, 0.0f, 1.0f) * 256.0f));
    if (alpha <= 0) {
        return;
    }
    const int invAlpha = 256 - std::min(alpha, 256);

    #pragma omp parallel for
    for (int y = 0; y < fbH; ++y) {
        uint32_t* dst = fbBits + size_t(y) * size_t(fbStride);
        const uint32_t* src = _overlayBuffer.data() + size_t(y) * size_t(fbW);
        for (int x = 0; x < fbW; ++x) {
            const uint32_t ov = src[x];
            if (ov == 0u) {
                continue;
            }

            const uint32_t base = dst[x];
            const uint32_t br = (base >> 16) & 0xFFu;
            const uint32_t bg = (base >> 8) & 0xFFu;
            const uint32_t bb = base & 0xFFu;
            const uint32_t or_ = (ov >> 16) & 0xFFu;
            const uint32_t og = (ov >> 8) & 0xFFu;
            const uint32_t ob = ov & 0xFFu;

            const uint32_t r = (br * uint32_t(invAlpha) + or_ * uint32_t(alpha)) >> 8;
            const uint32_t g = (bg * uint32_t(invAlpha) + og * uint32_t(alpha)) >> 8;
            const uint32_t b = (bb * uint32_t(invAlpha) + ob * uint32_t(alpha)) >> 8;
            dst[x] = 0xFF000000u | (r << 16) | (g << 8) | b;
        }
    }
}

// ============================================================================
// Window/level
// ============================================================================

cv::Vec2f CAdaptiveVolumeViewer::sceneToSurfaceCoords(const QPointF& scenePos) const
{
    return sceneToSurface(scenePos);
}

void CAdaptiveVolumeViewer::setBBoxMode(bool enabled)
{
    _bboxMode = enabled;
    if (!enabled && _activeBBoxSurfRect) {
        _activeBBoxSurfRect.reset();
        emit overlaysUpdated();
    }
}

std::optional<QRectF> CAdaptiveVolumeViewer::activeBBoxSceneRect() const
{
    if (!_activeBBoxSurfRect) {
        return std::nullopt;
    }
    return surfaceRectToSceneRect(*_activeBBoxSurfRect);
}

auto CAdaptiveVolumeViewer::selections() const -> std::vector<std::pair<QRectF, QColor>>
{
    std::vector<std::pair<QRectF, QColor>> out;
    out.reserve(_selections.size());
    for (const auto& selection : _selections) {
        out.emplace_back(surfaceRectToSceneRect(selection.surfRect), selection.color);
    }
    return out;
}

void CAdaptiveVolumeViewer::clearSelections()
{
    if (_selections.empty() && !_activeBBoxSurfRect) {
        return;
    }
    _selections.clear();
    _activeBBoxSurfRect.reset();
    emit overlaysUpdated();
}

QuadSurface* CAdaptiveVolumeViewer::makeBBoxFilteredSurfaceFromSceneRect(const QRectF& sceneRect)
{
    if (_surfName != "segmentation") {
        return nullptr;
    }

    auto surf = _surfWeak.lock();
    auto* quad = dynamic_cast<QuadSurface*>(surf.get());
    if (!quad) {
        return nullptr;
    }

    const cv::Mat_<cv::Vec3f>* srcPtr = quad->rawPointsPtr();
    if (!srcPtr || srcPtr->empty()) {
        return nullptr;
    }
    const cv::Mat_<cv::Vec3f>& src = *srcPtr;
    const int height = src.rows;
    const int width = src.cols;

    QRectF rSurf = sceneRectToSurfaceRect(sceneRect).normalized();
    if (!std::isfinite(rSurf.left()) || !std::isfinite(rSurf.top()) ||
        !std::isfinite(rSurf.right()) || !std::isfinite(rSurf.bottom())) {
        return nullptr;
    }

    const double cx = width * 0.5;
    const double cy = height * 0.5;
    const cv::Vec2f scale = quad->scale();
    if (scale[0] == 0.0f || scale[1] == 0.0f) {
        return nullptr;
    }

    int i0 = std::max(0, static_cast<int>(std::floor(cx + rSurf.left() * scale[0])));
    int i1 = std::min(width - 1, static_cast<int>(std::ceil(cx + rSurf.right() * scale[0])));
    int j0 = std::max(0, static_cast<int>(std::floor(cy + rSurf.top() * scale[1])));
    int j1 = std::min(height - 1, static_cast<int>(std::ceil(cy + rSurf.bottom() * scale[1])));
    if (i0 > i1 || j0 > j1) {
        return nullptr;
    }

    const int outW = i1 - i0 + 1;
    const int outH = j1 - j0 + 1;
    cv::Mat_<cv::Vec3f> cropped(outH, outW, cv::Vec3f(-1.0f, -1.0f, -1.0f));

    for (int j = j0; j <= j1; ++j) {
        for (int i = i0; i <= i1; ++i) {
            const cv::Vec3f& p = src(j, i);
            if (p[0] == -1.0f && p[1] == -1.0f && p[2] == -1.0f) {
                continue;
            }
            const double u = (i - cx) / scale[0];
            const double v = (j - cy) / scale[1];
            if (u >= rSurf.left() && u <= rSurf.right() &&
                v >= rSurf.top() && v <= rSurf.bottom()) {
                cropped(j - j0, i - i0) = p;
            }
        }
    }

    cv::Mat_<cv::Vec3f> cleaned = clean_surface_outliers(cropped);

    auto countValidInCol = [&](int c) {
        int count = 0;
        for (int r = 0; r < cleaned.rows; ++r) {
            if (cleaned(r, c)[0] != -1.0f) {
                ++count;
            }
        }
        return count;
    };
    auto countValidInRow = [&](int r) {
        int count = 0;
        for (int c = 0; c < cleaned.cols; ++c) {
            if (cleaned(r, c)[0] != -1.0f) {
                ++count;
            }
        }
        return count;
    };

    const int minValidCol = std::max(1, std::min(3, cleaned.rows));
    const int minValidRow = std::max(1, std::min(3, cleaned.cols));
    int left = 0;
    int right = cleaned.cols - 1;
    int top = 0;
    int bottom = cleaned.rows - 1;
    while (left <= right && countValidInCol(left) < minValidCol) {
        ++left;
    }
    while (right >= left && countValidInCol(right) < minValidCol) {
        --right;
    }
    while (top <= bottom && countValidInRow(top) < minValidRow) {
        ++top;
    }
    while (bottom >= top && countValidInRow(bottom) < minValidRow) {
        --bottom;
    }

    if (left > right || top > bottom) {
        left = cleaned.cols;
        right = -1;
        top = cleaned.rows;
        bottom = -1;
        for (int j = 0; j < cleaned.rows; ++j) {
            for (int i = 0; i < cleaned.cols; ++i) {
                if (cleaned(j, i)[0] != -1.0f) {
                    left = std::min(left, i);
                    right = std::max(right, i);
                    top = std::min(top, j);
                    bottom = std::max(bottom, j);
                }
            }
        }
        if (right < 0 || bottom < 0) {
            return nullptr;
        }
    }

    const int finalW = right - left + 1;
    const int finalH = bottom - top + 1;
    cv::Mat_<cv::Vec3f> finalPts(finalH, finalW, cv::Vec3f(-1.0f, -1.0f, -1.0f));
    for (int j = top; j <= bottom; ++j) {
        for (int i = left; i <= right; ++i) {
            finalPts(j - top, i - left) = cleaned(j, i);
        }
    }

    return new QuadSurface(finalPts, quad->scale());
}

// ============================================================================
// Intersection overlay
// ============================================================================

namespace {
constexpr std::array<QRgb, 12> kIntersectionPalette = {
    qRgb(255, 120, 120), qRgb(120, 200, 255), qRgb(120, 255, 140),
    qRgb(255, 220, 100), qRgb(220, 140, 255), qRgb(255, 160, 200),
    qRgb(140, 255, 220), qRgb(200, 255, 140), qRgb(255, 180, 120),
    qRgb(180, 200, 255), qRgb(255, 140, 180), qRgb(160, 255, 180),
};
constexpr int kIntersectionZ = 100;
constexpr int kHighlightedIntersectionZ = 110;
constexpr int kActiveIntersectionZ = 120;
constexpr float kActiveIntersectionOpacityScale = 1.2f;
constexpr float kActiveIntersectionWidthScale = 1.3f;
constexpr float kActiveIntersectionMinWidthDelta = 0.75f;

struct IntersectionStyle {
    QRgb color = 0;
    int z = kIntersectionZ;
    int widthQ = 0;

    bool operator==(const IntersectionStyle& other) const
    {
        return color == other.color && z == other.z && widthQ == other.widthQ;
    }
};

struct IntersectionStyleHash {
    size_t operator()(const IntersectionStyle& style) const
    {
        size_t h = std::hash<QRgb>{}(style.color);
        h ^= std::hash<int>{}(style.z) + 0x9e3779b9u + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(style.widthQ) + 0x9e3779b9u + (h << 6) + (h >> 2);
        return h;
    }
};

QColor activeSegmentationColorForView(const std::string& surfName)
{
    if (surfName == "seg yz" || surfName == "yz plane") {
        return QColor(Qt::yellow);
    }
    if (surfName == "seg xz" || surfName == "xz plane") {
        return QColor(Qt::red);
    }
    return QColor(255, 140, 0);
}

float activeSegmentationIntersectionWidth(float baseWidth)
{
    return std::max(baseWidth * kActiveIntersectionWidthScale,
                    baseWidth + kActiveIntersectionMinWidthDelta);
}
}

void CAdaptiveVolumeViewer::invalidateIntersect(const std::string&)
{
    for (auto* item : _intersectionItems) {
        if (item && item->scene()) _scene->removeItem(item);
        delete item;
    }
    _intersectionItems.clear();
}

void CAdaptiveVolumeViewer::renderIntersections()
{
    auto surf = _surfWeak.lock();
    auto* plane = dynamic_cast<PlaneSurface*>(surf.get());
    if (!plane || !_state || !_viewerManager || !_scene || !_view) {
        invalidateIntersect();
        _lastIntersectFp = {};
        return;
    }

    auto* patchIndex = _viewerManager->surfacePatchIndex();
    if (!patchIndex || patchIndex->empty()) {
        invalidateIntersect();
        _lastIntersectFp = {};
        return;
    }

    // Resolve target surface names to shared_ptrs.
    std::unordered_set<SurfacePatchIndex::SurfacePtr> targets;
    auto addTarget = [&](const std::string& name) {
        if (auto quad = std::dynamic_pointer_cast<QuadSurface>(_state->surface(name)))
            targets.insert(std::move(quad));
    };
    for (const auto& name : _intersectTgts) {
        if (name == "visible_segmentation") {
            if (_highlightedSurfaceIds.empty()) addTarget("segmentation");
            else for (const auto& id : _highlightedSurfaceIds) addTarget(id);
        } else {
            addTarget(name);
        }
    }
    if (targets.empty()) { invalidateIntersect(); _lastIntersectFp = {}; return; }

    // Compute the plane ROI in surface-param space covering the current viewport.
    QRectF sceneRect = _view->mapToScene(_view->viewport()->rect()).boundingRect();
    if (!sceneRect.isValid()) { invalidateIntersect(); _lastIntersectFp = {}; return; }

    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();
    const std::array<QPointF, 4> corners = {
        sceneRect.topLeft(), sceneRect.topRight(),
        sceneRect.bottomLeft(), sceneRect.bottomRight(),
    };
    for (const auto& c : corners) {
        cv::Vec2f sp = sceneToSurfaceCoords(c);
        minX = std::min(minX, sp[0]);
        minY = std::min(minY, sp[1]);
        maxX = std::max(maxX, sp[0]);
        maxY = std::max(maxY, sp[1]);
    }
    cv::Rect planeRoi{int(std::floor(minX)), int(std::floor(minY)),
                      std::max(1, int(std::ceil(maxX - minX))),
                      std::max(1, int(std::ceil(maxY - minY)))};

    auto activeSeg = std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));

    // Skip the rebuild entirely if nothing material has changed.
    IntersectFingerprint fp;
    fp.roiX = planeRoi.x;
    fp.roiY = planeRoi.y;
    fp.roiW = planeRoi.width;
    fp.roiH = planeRoi.height;
    auto quantizeVec = [](const cv::Vec3f& v) {
        return std::array<int, 3>{
            int(std::lround(v[0] * 1000.0f)),
            int(std::lround(v[1] * 1000.0f)),
            int(std::lround(v[2] * 1000.0f)),
        };
    };
    fp.planeOriginQ = quantizeVec(plane->origin());
    fp.planeNormalQ = quantizeVec(plane->normal({}, {}));
    fp.planeBasisXQ = quantizeVec(plane->basisX());
    fp.planeBasisYQ = quantizeVec(plane->basisY());
    fp.opacityQ = int(std::lround(_intersectionOpacity * 1000.0f));
    fp.thicknessQ = int(std::lround(_intersectionThickness * 1000.0f));
    fp.patchCount = patchIndex->patchCount();
    fp.surfaceCount = patchIndex->surfaceCount();
    size_t th = 0;
    size_t gh = 0;
    for (const auto& t : targets) {
        th ^= std::hash<const void*>{}(t.get()) + 0x9e3779b9u + (th << 6) + (th >> 2);
        gh ^= std::hash<const void*>{}(t.get()) ^
              (std::hash<uint64_t>{}(patchIndex->generation(t)) + 0x9e3779b9u);
    }
    fp.targetHash = th;
    fp.targetGenerationHash = gh;
    fp.activeSegHash = activeSeg ? std::hash<const void*>{}(activeSeg.get()) : 0;
    size_t hh = 0;
    for (const auto& id : _highlightedSurfaceIds) {
        hh ^= std::hash<std::string>{}(id) + 0x9e3779b9u + (hh << 6) + (hh >> 2);
    }
    fp.highlightedSurfaceHash = hh;
    fp.valid = true;
    if (_lastIntersectFp == fp && !_intersectionItems.empty()) {
        return;
    }
    invalidateIntersect();
    _lastIntersectFp = fp;

    auto intersections = patchIndex->computePlaneIntersections(*plane, planeRoi, targets);
    if (intersections.empty()) { /* kept cleared by invalidate above */ return; }

    // Group path segments by draw style for batched drawing. Active
    // segmentation gets its own z so it always renders above other
    // intersection overlays.
    std::unordered_map<IntersectionStyle, QPainterPath, IntersectionStyleHash> groupedPaths;
    std::unordered_map<IntersectionStyle, QColor, IntersectionStyleHash> groupedColors;
    // Bitwise finite check: matches repo convention (see feedback_ffast_math).
    // Finite iff exponent bits are not all 1s.
    auto isFiniteScalar = [](double v) {
        uint64_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
    };
    auto isFinitePoint = [&](const QPointF& p) {
        return isFiniteScalar(p.x()) && isFiniteScalar(p.y());
    };
    // volumeToScene() locks _surfWeak and dynamic_casts on every call. We
    // already have `plane` cached from the top of this function, so inline
    // the projection once per segment endpoint here.
    auto planeToScene = [&](const cv::Vec3f& volPoint) {
        cv::Vec3f proj = plane->project(volPoint, 1.0, 1.0);
        return surfaceToScene(proj[0], proj[1]);
    };

    for (const auto& [target, segments] : intersections) {
        if (!target || segments.empty()) continue;

        QColor baseColor;
        int zValue = kIntersectionZ;
        float opacity = _intersectionOpacity;
        float penWidth = _intersectionThickness;
        if (target == activeSeg) {
            baseColor = activeSegmentationColorForView(_surfName);
            zValue = kActiveIntersectionZ;
            opacity *= kActiveIntersectionOpacityScale;
            penWidth = activeSegmentationIntersectionWidth(penWidth);
        } else if (_highlightedSurfaceIds.count(target->id)) {
            baseColor = QColor(0, 220, 255);
            zValue = kHighlightedIntersectionZ;
        } else {
            const auto& id = target->id;
            auto it = _surfaceColorAssignments.find(id);
            size_t idx;
            if (it != _surfaceColorAssignments.end()) {
                idx = it->second;
            } else if (_surfaceColorAssignments.size() < 500) {
                idx = _nextColorIndex++;
                _surfaceColorAssignments[id] = idx;
            } else {
                idx = std::hash<std::string>{}(id);
            }
            baseColor = QColor::fromRgba(kIntersectionPalette[idx % kIntersectionPalette.size()]);
        }
        baseColor.setAlphaF(std::clamp(opacity, 0.0f, 1.0f));
        if (baseColor.alpha() <= 0) continue;

        for (const auto& seg : segments) {
            QPointF a = planeToScene(seg.world[0]);
            QPointF b = planeToScene(seg.world[1]);
            if (!isFinitePoint(a) || !isFinitePoint(b)) continue;
            const IntersectionStyle style{
                baseColor.rgba(),
                zValue,
                int(std::lround(std::max(0.0f, penWidth) * 1000.0f)),
            };
            QPainterPath& path = groupedPaths[style];
            path.moveTo(a);
            path.lineTo(b);
            groupedColors[style] = baseColor;
        }
    }

    _intersectionItems.reserve(groupedPaths.size());
    for (const auto& [style, path] : groupedPaths) {
        if (path.isEmpty()) continue;
        auto* item = new QGraphicsPathItem(path);
        QPen pen(groupedColors[style]);
        pen.setWidthF(static_cast<qreal>(style.widthQ) / 1000.0);
        pen.setCapStyle(Qt::RoundCap);
        pen.setJoinStyle(Qt::RoundJoin);
        pen.setCosmetic(true);
        item->setPen(pen);
        item->setBrush(Qt::NoBrush);
        item->setZValue(style.z);
        _scene->addItem(item);
        _intersectionItems.push_back(item);
    }

    _view->viewport()->update();
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
    scheduleRender();
    updateStatusLabel();
}

void CAdaptiveVolumeViewer::resetSurfaceOffsets()
{
    _camera.zOff = 0.0f;
    _zOffWorldDir = cv::Vec3f(0.0f, 0.0f, 0.0f);
    scheduleRender();
}

void CAdaptiveVolumeViewer::setVolumeWindow(float low, float high)
{
    _windowLow = std::clamp(low, 0.0f, 65535.0f);
    _windowHigh = std::clamp(high, 0.0f, 65535.0f);
    if (_volume) scheduleRender();
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

    // Throttle to ~10 Hz. Building the status string (arg/concat) and
    // pulling cache stats locks several mutexes; doing it every frame at
    // 60+ fps shows up in profiles. Interactive feedback is unchanged at
    // human timescales.
    auto now = std::chrono::steady_clock::now();
    if (now - _lastStatusUpdate < std::chrono::milliseconds(100)) return;
    _lastStatusUpdate = now;

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
            status += QString(" | dl %1sh w %2sh dlq %3 enq %4 ldq %5 dcq %6 neg %7")
                .arg(s.iceFetches)
                .arg(s.diskWrites)
                .arg(s.downloadPending)
                .arg(s.encodePending)
                .arg(s.loadPending)
                .arg(s.decodePending)
                .arg(s.negativeCount);
        } else {
            status += QString(" | dl %1 w %2 dlq %3 enq %4 ldq %5 dcq %6 neg %7")
                .arg(s.iceFetches)
                .arg(s.diskWrites)
                .arg(s.downloadPending)
                .arg(s.encodePending)
                .arg(s.loadPending)
                .arg(s.decodePending)
                .arg(s.negativeCount);
        }
    }

    status += " [adaptive]";
    // Skip the setText+adjustSize+show round-trip when the string is
    // identical to the last one we pushed. The label is rebuilt every
    // frame, but stats only tick a few times a second — most frames would
    // churn identical text through Qt's text layout engine.
    if (status != _lastStatusText) {
        _lastStatusText = status;
        _lbl->setText(status);
        _lbl->adjustSize();
        _lbl->show();
    }
}

void CAdaptiveVolumeViewer::fitSurfaceInView()
{
    _camera.surfacePtr = cv::Vec3f(0, 0, 0);
    _camera.scale = 0.5f;
    if (_volume)
        _camera.recalcPyramidLevel(static_cast<int>(_volume->numScales()));
    scheduleRender();
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
            scheduleRender();
        }
    }
    return QWidget::eventFilter(watched, event);
}
