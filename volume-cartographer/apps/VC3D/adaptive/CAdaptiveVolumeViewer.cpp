#include "CAdaptiveVolumeViewer.hpp"

#include "ViewerManager.hpp"
#include "VCSettings.hpp"
#include "../CState.hpp"

#include "vc/core/util/Surface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/core/render/PostProcess.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/types/SampleParams.hpp"
#include "vc/core/cache/BlockPipeline.hpp"
#include "vc/core/cache/ChunkKey.hpp"
#include "vc/core/cache/TickCoordinator.hpp"
#include <cstring>
#include <unordered_set>

#include <opencv2/imgproc.hpp>

#include <QSettings>
#include <QThreadPool>
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
#include <QOpenGLWidget>
#include <QSurfaceFormat>
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
    // GPU-backed paint device for the scene: all painter ops (QImage blit,
    // overlay items, intersections) go through an OpenGL surface instead
    // of Qt's CPU raster engine. The QImage framebuffer we populate in
    // drawBackground becomes a GL texture upload + textured quad, and
    // QGraphicsView compositing runs on the GPU rasterizer. Drops the
    // CPU blit cost (was ~5% of frame per the perf map) and frees main-
    // thread cycles for the sampling kernel.
    {
        QSurfaceFormat fmt;
        fmt.setSwapInterval(0);  // disable vsync — we already coalesce at 16 ms
        auto* gl = new QOpenGLWidget(_view);
        gl->setFormat(fmt);
        _view->setViewport(gl);
    }
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
    connect(_view, &CVolumeViewerView::sendMouseDoubleClick, this, [this](QPointF scenePos, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
        cv::Vec3f p = sceneToVolume(scenePos);
        emit sendMouseDoubleClickVolume(p, button, modifiers);
    });
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
        if (_intersectionsDirty) {
            _intersectionsDirty = false;
            renderIntersectionsNow();
        }
    });

    // When the user stops actively panning / zooming, kick a full-res
    // re-render to replace the progressive-level frame with a crisp one.
    // 180 ms chosen to be comfortably past the typical debounce of wheel
    // events + tail of a pan drag, so we don't flip to full-res in the
    // middle of continued motion.
    _interactionIdleTimer = new QTimer(this);
    _interactionIdleTimer->setSingleShot(true);
    _interactionIdleTimer->setInterval(180);
    connect(_interactionIdleTimer, &QTimer::timeout, this, [this]() {
        if (_interactive) {
            _interactive = false;
            scheduleRender();
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
    // Wait for any async workers (render, intersection compute) that hold
    // `this`. Dropping to member destruction while one is in flight causes
    // SIGSEGV on exit — observed on Qt app shutdown after a warm-cache
    // session. Simple spin is fine; the common case is counter==0 on entry.
    while (_backgroundWorkers.load(std::memory_order_acquire) > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (_chunkCbId != 0 && _volume && _volume->tieredCache()) {
        _volume->tieredCache()->removeChunkReadyListener(_chunkCbId);
        _chunkCbId = 0;
    }
    if (_tickViewportSlot >= 0) {
        vc::cache::TickCoordinator::releaseViewportSlotGlobal(_tickViewportSlot);
        _tickViewportSlot = -1;
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
    fprintf(stderr, "[Viewer:%s] OnVolumeChanged: old=%p new=%p _surfWeak=%p axisAligned=%d\n",
            _surfName.c_str(), (void*)_volume.get(), (void*)vol.get(),
            (void*)_surfWeak.lock().get(), isAxisAlignedView() ? 1 : 0);

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
                // The pipeline's chunkArrivedFlag_ is edge-triggered via
                // atomic exchange — this callback only fires when the flag
                // flips false→true. Deferring the clear to submitRender (on
                // the 16ms render timer) ensures every subsequent arrival in
                // the same tick window finds the flag already set and takes
                // the exchange=true/return-early branch — no listener fire,
                // no cross-thread event post. One wake per 16ms tick max,
                // instead of one per chunk burst.
                QMetaObject::invokeMethod(qApp, [guard, volumeWeak]() {
                    if (!guard) return;
                    auto vol = volumeWeak.lock();
                    if (!vol || guard->_volume != vol) return;
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

    fprintf(stderr, "[Viewer:%s] OnVolumeChanged: after surface setup: _surfWeak=%p _volume=%p\n",
            _surfName.c_str(), (void*)_surfWeak.lock().get(), (void*)_volume.get());

    if (_volume) {
        int nScales = static_cast<int>(_volume->numScales());
        std::vector<float> sfs(nScales);
        for (int i = 0; i < nScales; i++) sfs[i] = static_cast<float>(size_t{1} << i);
        _camera.recalcPyramidLevel(nScales, sfs.data());
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
    if (!_renderTimer->isActive()) {
        // 16 ms (~60 fps) for crisp idle frames; 33 ms (~30 fps) while the
        // user is actively panning/zooming. At progressive +1 pyramid the
        // preview is already blurrier than a settled frame, so 30 fps
        // matches what the eye perceives during motion and halves the
        // render work the kernel has to do under load.
        _renderTimer->setInterval(_interactive ? 33 : 16);
        _renderTimer->start();
    }
}

// Toggle to re-enable progressive rendering during pan/zoom. The motion-
// time resolution drop felt visually jarring in practice, so it's off by
// default — but the plumbing (submitRender level bump, idle-timer catch-
// up, 30 fps interactive coalesce) stays in place so it can be switched
// back on with a single recompile.
static constexpr bool kProgressiveRenderingEnabled = false;

void CAdaptiveVolumeViewer::beginInteraction()
{
    // Called from any event path that represents live user motion (pan
    // drag, zoom wheel). Marks _interactive so the next submitRender
    // picks the progressive pyramid level, and arms the idle timer so a
    // full-res render fires once motion stops.
    if (!kProgressiveRenderingEnabled) return;
    _interactive = true;
    _interactionIdleTimer->start();
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

void CAdaptiveVolumeViewer::recordRenderDuration(double seconds)
{
    if (seconds <= 0.0) return;
    _renderDurationsSec[_renderDurationHead] = seconds;
    _renderDurationHead = (_renderDurationHead + 1) % kFpsRingSize;
    if (_renderDurationCount < kFpsRingSize) ++_renderDurationCount;
}

float CAdaptiveVolumeViewer::measuredFps() const
{
    if (_renderDurationCount == 0) return 0.0f;
    double sum = 0.0;
    for (int i = 0; i < _renderDurationCount; ++i) sum += _renderDurationsSec[i];
    const double avg = sum / double(_renderDurationCount);
    if (avg <= 1e-6) return 0.0f;
    return float(1.0 / avg);
}

void CAdaptiveVolumeViewer::submitRender()
{
    // Re-arm the chunk-arrival edge detector for the next tick window.
    // Any chunk that decodes during this render will set the flag again and
    // fire exactly one post-event to trigger the next render. See the
    // addChunkReadyListener callback above for why the clear lives here.
    if (_volume) {
        if (auto* c = _volume->tieredCache()) c->clearChunkArrivedFlag();
    }

    // Quick main-thread checks before dispatch. The worker can't check
    // these safely — the shared_ptr from weak_ptr::lock() and the volume
    // pointers need main-thread-stable reads.
    auto surf = _surfWeak.lock();
    if (!surf || !_volume || !_volume->zarrDataset()) return;
    const int fbW = _framebuffer.width();
    const int fbH = _framebuffer.height();
    if (fbW <= 0 || fbH <= 0) return;

    // Serialise: if a worker is already rendering, just note that another
    // frame is pending. finishRenderOnMainThread() will reschedule once
    // the in-flight worker commits. Must precede any _framebufferWork
    // mutation — reassigning the QImage while the worker is writing to it
    // would corrupt both the new and in-flight frames.
    if (_renderWorkerBusy.exchange(true, std::memory_order_acq_rel)) {
        _renderPendingAfterWorker = true;
        return;
    }

    // Now that we own the busy gate, it's safe to resize the work buffer
    // — no worker is touching it. create() / assignment only happen on
    // size/format change.
    if (_framebufferWork.isNull() || _framebufferWork.width() != fbW
        || _framebufferWork.height() != fbH
        || _framebufferWork.format() != _framebuffer.format()) {
        _framebufferWork = QImage(fbW, fbH, _framebuffer.format());
    }
    _renderT0 = std::chrono::steady_clock::now();

    // Snapshot everything the worker will read that main-thread setters
    // can write. Done under the busy gate so settings setters queue
    // behind us on the event loop — the snapshot always reflects a
    // self-consistent frame of state rather than a torn mid-setter view.
    // Tick coordinator slot is also allocated here on main (avoids a
    // teardown race with the destructor) before we publish inside the
    // worker.
    if (_tickViewportSlot < 0) {
        _tickViewportSlot = vc::cache::TickCoordinator::acquireViewportSlotGlobal();
    }
    RenderContext ctx;
    ctx.camera = _camera;
    ctx.compositeSettings = _compositeSettings;
    ctx.samplingMethod = _samplingMethod;
    ctx.interactive = _interactive;
    ctx.windowLow = _windowLow;
    ctx.windowHigh = _windowHigh;
    ctx.baseColormapId = _baseColormapId;
    ctx.highlightDownscaled = _highlightDownscaled;
    ctx.zOffWorldDir = _zOffWorldDir;
    ctx.surf = std::move(surf);
    ctx.volume = _volume;

    // Dispatch the whole render body to QThreadPool. The main thread
    // returns immediately — Qt input events are no longer stalled behind
    // the tile sample loop, the CLAHE pass, or the stretch scan.
    _backgroundWorkers.fetch_add(1, std::memory_order_acq_rel);
    QThreadPool::globalInstance()->start([this, ctx = std::move(ctx)]() {
        try {
            renderIntoFramebuffer(_framebufferWork, ctx);
        } catch (const std::exception& ex) {
            fprintf(stderr, "[Viewer:%s] RENDER EXCEPTION: %s\n",
                    _surfName.c_str(), ex.what());
        } catch (...) {
            fprintf(stderr, "[Viewer:%s] RENDER EXCEPTION (unknown)\n",
                    _surfName.c_str());
        }
        QMetaObject::invokeMethod(this,
            "finishRenderOnMainThread", Qt::QueuedConnection);
        // Decrement last — the destructor spins on this reaching 0, so
        // all `this` access (including the invokeMethod post above) must
        // happen before we signal "worker done".
        _backgroundWorkers.fetch_sub(1, std::memory_order_release);
    });
}

void CAdaptiveVolumeViewer::renderIntoFramebuffer(QImage& fb,
                                                   const RenderContext& ctx)
{
    // All reads of mutable viewer state go through `ctx` — `_camera`,
    // `_compositeSettings`, `_windowLow`, `_baseColormapId`,
    // `_highlightDownscaled`, `_samplingMethod`, `_interactive`,
    // `_zOffWorldDir`, `_surfWeak`, `_volume` are all mutated by main-
    // thread handlers and must not be touched from this worker thread.
    // Per-render scratch caches (_genCoords, _cachedLut, _claheCache,
    // etc.) are only touched here and are serialised by the
    // _renderWorkerBusy gate so they remain safe without the snapshot.
    const CompositeParams& lightP = ctx.compositeSettings.params;
    const bool rakingEnabled = ctx.compositeSettings.postRakingEnabled;
    const float rakingAz = ctx.compositeSettings.postRakingAzimuth;
    const float rakingEl = ctx.compositeSettings.postRakingElevation;
    const float rakingStrength = std::clamp(ctx.compositeSettings.postRakingStrength, 0.0f, 1.0f);
    const float rakingDepth = std::max(0.01f, ctx.compositeSettings.postRakingDepthScale);

    // Debug overlay: paint a per-pixel gradient based on fallback-level depth.
    // Cached in reloadPerfSettings() instead of re-read from disk each frame.
    const bool highlightDownscaled = ctx.highlightDownscaled;

    const auto& surf = ctx.surf;
    if (!surf || !ctx.volume || !ctx.volume->zarrDataset()) return;

    // fb size was validated on main thread before dispatch.
    const int fbW = fb.width();
    const int fbH = fb.height();
    if (fbW <= 0 || fbH <= 0) return;

    // Publish viewport snapshot for the tick coordinator. Used by
    // prefetch coalescing (so the tick drain knows which pipelines/levels
    // are in use) and future slice scoping. Slot allocation itself moved
    // to submitRender (main thread) so the destructor's release doesn't
    // race with a lazy-allocation here.
    if (_tickViewportSlot >= 0) {
        vc::cache::ViewportSnapshot vs;
        vs.active = true;
        vs.level = ctx.camera.dsScaleIdx;
        vs.pipeline = ctx.volume->tieredCache();
        vc::cache::TickCoordinator::publishViewportGlobal(_tickViewportSlot, vs);
    }

    // Level buffer is only consumed by the downscale-highlight debug
    // overlay below; when the overlay is off, pass a null pointer to the
    // kernel so it skips per-pixel level writes entirely, and skip the
    // full-framebuffer setTo(0) memset here. At 1080p that memset is
    // ~8 MB/frame of unnecessary bandwidth.
    uint8_t* lvlOutPtr = nullptr;
    int lvlOutStride = 0;
    if (highlightDownscaled) {
        if (_levelBuffer.rows != fbH || _levelBuffer.cols != fbW) {
            _levelBuffer.create(fbH, fbW);
        }
        _levelBuffer.setTo(0);
        lvlOutPtr = _levelBuffer.ptr<uint8_t>(0);
        lvlOutStride = int(_levelBuffer.step1());
    }

    auto* fbBits = reinterpret_cast<uint32_t*>(fb.bits());
    int fbStride = fb.bytesPerLine() / 4;

    // Build the render LUT. For stretch mode we use last frame's min/max
    // so the render is a single pass; we refresh the cached range by
    // scanning the framebuffer after. A camera change invalidates the
    // cache and forces a 2-pass on the first frame after motion.
    const bool stretch = ctx.compositeSettings.postStretchValues;
    const uint8_t isoCutoff = ctx.compositeSettings.params.isoCutoff;
    auto applyIsoCutoff = [&](std::array<uint32_t, 256>& l, uint8_t cutoff) {
        if (cutoff == 0) return;
        const uint32_t zero = l[0];
        for (int i = 0; i < cutoff; i++) l[i] = zero;
    };

    std::array<uint32_t, 256> lut;
    const bool stretchFirstPass = stretch && !_cachedStretchValid;
    // CLAHE and raking both operate on gray, so the colormap is deferred
    // until after those passes. The sampling LUT in that case is gray-only.
    const bool postGrayDomain = ctx.compositeSettings.postClaheEnabled || rakingEnabled;
    const bool deferColormap = postGrayDomain && !ctx.baseColormapId.empty();
    const std::string sampleColormapId = deferColormap
        ? std::string() : ctx.baseColormapId;
    if (stretchFirstPass) {
        // Identity gray LUT so we can extract the raw sample after sampling.
        for (int i = 0; i < 256; i++) {
            uint32_t v = uint32_t(i);
            lut[i] = 0xFF000000u | (v << 16) | (v << 8) | v;
        }
    } else {
        float wlo = stretch ? float(_cachedStretchLo) : ctx.windowLow;
        float whi = stretch ? float(_cachedStretchHi) : ctx.windowHigh;
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
    const int numLevels = static_cast<int>(ctx.volume->numScales());
    // Always render at the user-requested level. The sampler's per-pixel
    // adaptive fallback handles regions that aren't ready yet by dropping
    // those pixels to whichever coarser level is resident — no whole-frame
    // resolution cycling, and cached fine chunks are used immediately.
    sp.level = ctx.camera.dsScaleIdx;
    // During live interaction (pan drag, zoom wheel) bump the pyramid
    // level one step coarser. Each step halves the voxels read per
    // sample, which cuts the ray-march cost ~2-4x for a still-coherent
    // preview frame. The interaction-idle timer triggers a full-res
    // render ~180 ms after motion stops.
    if (ctx.interactive) {
        sp.level = std::min(sp.level + 1, std::max(0, numLevels - 1));
    }
    sp.method = ctx.samplingMethod;

    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        cv::Vec3f vx = plane->basisX();
        cv::Vec3f vy = plane->basisY();
        cv::Vec3f n = plane->normal(cv::Vec3f(0, 0, 0));

        float halfW = static_cast<float>(fbW) * 0.5f / ctx.camera.scale;
        float halfH = static_cast<float>(fbH) * 0.5f / ctx.camera.scale;

        cv::Vec3f origin = vx * (ctx.camera.surfacePtr[0] - halfW)
                         + vy * (ctx.camera.surfacePtr[1] - halfH)
                         + plane->origin() + n * ctx.camera.zOff;
        cv::Vec3f vx_step = vx / ctx.camera.scale;
        cv::Vec3f vy_step = vy / ctx.camera.scale;

        int numLayers = 1, zStart = 0;
        float zStep = 1.0f;
        const cv::Vec3f* pNormal = nullptr;
        std::string method;
        if (ctx.compositeSettings.planeEnabled) {
            const int front = ctx.compositeSettings.planeLayersFront;
            const int behind = ctx.compositeSettings.planeLayersBehind;
            numLayers = front + behind + 1;
            zStart = -behind;
            zStep = ctx.compositeSettings.reverseDirection ? -1.0f : 1.0f;
            pNormal = &n;
            method = ctx.compositeSettings.params.method;
        } else {
            pNormal = &n; // ignored for numLayers=1
        }

        // Composite averages ~11 layers — the averaging is itself a low-pass,
        // so per-layer Nearest matches Trilinear visually at ~8x the speed.
        vc::Sampling sampleMethod = (numLayers > 1) ? vc::Sampling::Nearest
                                                    : ctx.samplingMethod;
        sampleAdaptiveARGB32(
            fbBits, fbStride, ctx.volume->tieredCache(),
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
        cv::Vec3f offset(ctx.camera.surfacePtr[0] * ctx.camera.scale - float(fbW) * 0.5f,
                         ctx.camera.surfacePtr[1] * ctx.camera.scale - float(fbH) * 0.5f,
                         0.0f);
        const bool wantComposite = ctx.compositeSettings.enabled;
        // Use the snapshot of _zOffWorldDir taken on the main thread.
        // Updates discovered here are posted back to main via
        // invokeMethod (see below) so the write never crosses threads.
        cv::Vec3f zOffWorldDir = ctx.zOffWorldDir;
        // Always request normals so shift+scroll can sample the view-center
        // normal without a separate gen pass.
        const bool cacheHit =
            !_genCacheDirty
            && _genCacheSurfKey == surf.get()
            && _genCacheFbW == fbW
            && _genCacheFbH == fbH
            && _genCacheScale == ctx.camera.scale
            && _genCacheOffset == offset
            && _genCacheWantComposite == wantComposite
            && _genCacheZOff == ctx.camera.zOff
            && _genCacheZOffDir == zOffWorldDir
            && !_genCoords.empty();
        if (!cacheHit) {
            surf->gen(&_genCoords, &_genNormals,
                      cv::Size(fbW, fbH), cv::Vec3f(0, 0, 0),
                      ctx.camera.scale, offset);
            // Lazy-capture the translation direction when zOff was set by a
            // path that didn't populate _zOffWorldDir (adjustSurfaceOffset
            // via Ctrl+./Ctrl+, shortcuts, or any other non-Shift-scroll
            // source). Without this, those offsets would be silent no-ops
            // until the user first Shift-scrolled.
            if (ctx.camera.zOff != 0.0f &&
                zOffWorldDir[0] == 0.0f && zOffWorldDir[1] == 0.0f && zOffWorldDir[2] == 0.0f &&
                !_genNormals.empty()) {
                const int cy = _genNormals.rows / 2;
                const int cx = _genNormals.cols / 2;
                const cv::Vec3f n = _genNormals(cy, cx);
                if (std::isfinite(n[0]) && std::isfinite(n[1]) && std::isfinite(n[2])) {
                    const float len = static_cast<float>(cv::norm(n));
                    if (len > 1e-6f) {
                        zOffWorldDir = n / len;
                        // Publish the discovered direction back to main so
                        // subsequent renders (and onZoom's own capture
                        // branch) see it. Guarded "still-zero" check on
                        // main avoids stomping a user-driven update that
                        // raced with this worker.
                        cv::Vec3f dir = zOffWorldDir;
                        QMetaObject::invokeMethod(this, [this, dir]() {
                            if (_zOffWorldDir[0] == 0.0f
                             && _zOffWorldDir[1] == 0.0f
                             && _zOffWorldDir[2] == 0.0f) {
                                _zOffWorldDir = dir;
                            }
                        }, Qt::QueuedConnection);
                    }
                }
            }
            // Apply z-offset as a rigid world-space translation using the
            // cached direction. On cache hits _genCoords is already shifted
            // — avoid double-applying by only running this on cache miss.
            if (ctx.camera.zOff != 0.0f &&
                (zOffWorldDir[0] != 0.0f || zOffWorldDir[1] != 0.0f || zOffWorldDir[2] != 0.0f) &&
                !_genCoords.empty()) {
                const cv::Vec3f tr = zOffWorldDir * ctx.camera.zOff;
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
            _genCacheScale = ctx.camera.scale;
            _genCacheOffset = offset;
            _genCacheWantComposite = wantComposite;
            _genCacheZOff = ctx.camera.zOff;
            _genCacheZOffDir = zOffWorldDir;
            _genCacheDirty = false;
        }
        cv::Mat_<cv::Vec3f>& coords = _genCoords;
        cv::Mat_<cv::Vec3f>& normals = _genNormals;

        if (!coords.empty()) {
            int numLayers = 1, zStart = 0;
            float zStep = 1.0f;
            const cv::Mat_<cv::Vec3f>* pNormals = nullptr;
            std::string method;
            if (wantComposite && !normals.empty()) {
                const int front = ctx.compositeSettings.layersFront;
                const int behind = ctx.compositeSettings.layersBehind;
                numLayers = front + behind + 1;
                zStart = -behind;
                zStep = ctx.compositeSettings.reverseDirection ? -1.0f : 1.0f;
                pNormals = &normals;
                method = ctx.compositeSettings.params.method;
            }
            vc::Sampling sampleMethod = (numLayers > 1) ? vc::Sampling::Nearest
                                                        : ctx.samplingMethod;
            sampleAdaptiveARGB32(
                fbBits, fbStride, ctx.volume->tieredCache(),
                sp.level, numLevels,
                &coords, nullptr, nullptr, nullptr,
                pNormals, nullptr,
                numLayers, zStart, zStep,
                fbW, fbH, method, lut.data(), sampleMethod,
                &lightP,  // sampler uses lightP for volumetric and lighting paths
                lvlOutPtr, lvlOutStride,
                // Coords cached → prior frame already did the chunk
                // enumeration + fetchInteractive for this exact geometry.
                // The per-sample adaptive-fallback path still handles any
                // block not yet resident, so correctness is preserved.
                cacheHit);
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
                    float(lo), float(hi), ctx.baseColormapId);
            } else {
                vc::buildWindowLevelColormapLut(stretchedLut,
                    ctx.windowLow, ctx.windowHigh, ctx.baseColormapId);
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
            _cachedColormapId = ctx.baseColormapId;
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
        if (ctx.compositeSettings.postClaheEnabled) {
            const int tile = std::max(1, ctx.compositeSettings.postClaheTileSize);
            const double clip = std::max(0.01, double(ctx.compositeSettings.postClaheClipLimit));
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
            if (!_deferredCmapValid || _deferredCmapId != ctx.baseColormapId) {
                vc::buildWindowLevelColormapLut(_deferredCmapLut, 0.0f, 255.0f, ctx.baseColormapId);
                _deferredCmapId = ctx.baseColormapId;
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

}

void CAdaptiveVolumeViewer::finishRenderOnMainThread()
{
    const bool sizesMatch = (_framebuffer.size() == _framebufferWork.size());
    if (sizesMatch) {
        std::swap(_framebuffer, _framebufferWork);
    }

    // Main-thread-only tail. syncCameraTransform writes Qt view state,
    // updateFocusMarker / renderIntersections / overlaysUpdated all touch
    // the scene graph, viewport()->update schedules a paint event.
    syncCameraTransform();
    updateFocusMarker();
    renderIntersections();
    emit overlaysUpdated();
    _view->viewport()->update();

    const auto renderDt = std::chrono::steady_clock::now() - _renderT0;
    recordRenderDuration(std::chrono::duration<double>(renderDt).count());
    updateStatusLabel();

    _renderWorkerBusy.store(false, std::memory_order_release);
    // Re-schedule if a pending frame was queued OR if we discarded a
    // stale-sized frame above — either way we owe the view a current
    // render.
    if (_renderPendingAfterWorker || !sizesMatch) {
        _renderPendingAfterWorker = false;
        // Queue the next frame on the render timer instead of calling
        // submitRender recursively — keeps the dispatch on the Qt event
        // loop and lets any batched setters that fired while the worker
        // ran settle into state before we read it.
        scheduleRender();
    }
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
    beginInteraction();
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
        int ns = static_cast<int>(_volume->numScales());
        std::vector<float> sfs(ns);
        for (int i = 0; i < ns; i++) sfs[i] = static_cast<float>(size_t{1} << i);
        _camera.recalcPyramidLevel(ns, sfs.data());
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
    beginInteraction();
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
            beginInteraction();
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
    cv::Vec3f surfLoc = {sp[0], sp[1], _camera.zOff};
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

// ============================================================================
// Window/level
// ============================================================================

cv::Vec2f CAdaptiveVolumeViewer::sceneToSurfaceCoords(const QPointF& scenePos) const
{
    return sceneToSurface(scenePos);
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
    // Coalesce rapid-fire UI setters (opacity / thickness / surface-
    // change / slider drags) into one rtree + triangle-clip pass per
    // render tick. The _renderTimer callback drains this flag at the
    // same 16 ms boundary it kicks submitRender() on, so we reuse the
    // existing tick instead of running a second timer.
    _intersectionsDirty = true;
    if (_renderTimer && !_renderTimer->isActive()) {
        _renderTimer->start();
    }
}

void CAdaptiveVolumeViewer::renderIntersectionsNow()
{
    auto surf = _surfWeak.lock();
    if (!surf || !_state || !_viewerManager || !_scene || !_view) {
        invalidateIntersect();
        _lastIntersectFp = {};
        return;
    }

    auto* plane = dynamic_cast<PlaneSurface*>(surf.get());
    if (!plane) {
        renderFlattenedIntersections(surf);
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

    // Serialise: one plane-intersection worker at a time. If another is
    // still running, remember dirty and let the next render tick restart
    // us. Don't clear the scene items here — leave the old overlay
    // visible rather than blanking while we recompute.
    if (_planeWorkerBusy.exchange(true, std::memory_order_acq_rel)) {
        _intersectionsDirty = true;
        _lastIntersectFp = {};
        return;
    }
    invalidateIntersect();
    _lastIntersectFp = fp;

    // Snapshot camera state for the worker's plane-to-scene projections.
    struct PlaneCamSnapshot {
        float camSurfX, camSurfY, camScale;
        float vpCx, vpCy;
        QTransform viewToScene;
    };
    PlaneCamSnapshot cam{
        _camSurfX, _camSurfY, _camScale,
        static_cast<float>(_framebuffer.width()) * 0.5f,
        static_cast<float>(_framebuffer.height()) * 0.5f,
        _view ? _view->transform().inverted() : QTransform(),
    };

    // Pre-resolve per-target styling on the main thread — it reads
    // _surfaceColorAssignments / _nextColorIndex and touches the palette
    // LUT, all of which we want to keep single-threaded. The worker's
    // output is a flat list of styled segments; the scene rebuild then
    // groups them by style into QPainterPaths.
    struct TargetStyle {
        QColor color;
        int z;
        float penWidth;
    };
    std::unordered_map<const void*, TargetStyle> targetStyles;
    targetStyles.reserve(targets.size());
    for (const auto& target : targets) {
        if (!target) continue;
        TargetStyle ts;
        ts.z = kIntersectionZ;
        float opacity = _intersectionOpacity;
        ts.penWidth = _intersectionThickness;
        QColor baseColor;
        if (target == activeSeg) {
            baseColor = activeSegmentationColorForView(_surfName);
            ts.z = kActiveIntersectionZ;
            opacity *= kActiveIntersectionOpacityScale;
            ts.penWidth = activeSegmentationIntersectionWidth(ts.penWidth);
        } else if (_highlightedSurfaceIds.count(target->id)) {
            baseColor = QColor(0, 220, 255);
            ts.z = kHighlightedIntersectionZ;
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
        ts.color = baseColor;
        targetStyles.emplace(target.get(), ts);
    }

    // Capture the plane by shared_ptr (reuse via _state is fine — shared
    // ownership keeps it alive across the worker). patchIndex is owned by
    // ViewerManager (app-lifetime).
    auto planeSp = std::dynamic_pointer_cast<PlaneSurface>(
        _state->surface(_surfName));
    if (!planeSp) {
        _planeWorkerBusy.store(false, std::memory_order_release);
        return;
    }

    _backgroundWorkers.fetch_add(1, std::memory_order_acq_rel);
    QThreadPool::globalInstance()->start(
        [this, fp, planeSp, planeRoi, targets = std::move(targets),
         targetStyles = std::move(targetStyles), cam, patchIndex]() mutable {
            auto isFiniteScalar = [](double v) {
                uint64_t bits;
                std::memcpy(&bits, &v, sizeof(bits));
                return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
            };
            auto surfToScene = [&](float sx, float sy) -> QPointF {
                const qreal vx = (sx - cam.camSurfX) * cam.camScale + cam.vpCx;
                const qreal vy = (sy - cam.camSurfY) * cam.camScale + cam.vpCy;
                return cam.viewToScene.map(QPointF(vx, vy));
            };
            auto planeToScene = [&](const cv::Vec3f& volPoint) {
                cv::Vec3f proj = planeSp->project(volPoint, 1.0, 1.0);
                return surfToScene(proj[0], proj[1]);
            };

            // Heavy compute: rtree walk + per-patch triangle clip.
            auto intersections = patchIndex->computePlaneIntersections(
                *planeSp, planeRoi, targets);

            // Group by draw style — same output shape the main-thread apply
            // step needs for batched QGraphicsPathItem creation.
            std::unordered_map<IntersectionStyle, QPainterPath,
                               IntersectionStyleHash> groupedPaths;
            std::unordered_map<IntersectionStyle, QColor,
                               IntersectionStyleHash> groupedColors;
            for (const auto& [target, segments] : intersections) {
                if (!target || segments.empty()) continue;
                auto it = targetStyles.find(target.get());
                if (it == targetStyles.end()) continue;
                const auto& ts = it->second;
                if (ts.color.alpha() <= 0) continue;
                for (const auto& seg : segments) {
                    QPointF a = planeToScene(seg.world[0]);
                    QPointF b = planeToScene(seg.world[1]);
                    if (!isFiniteScalar(a.x()) || !isFiniteScalar(a.y())
                     || !isFiniteScalar(b.x()) || !isFiniteScalar(b.y()))
                        continue;
                    const IntersectionStyle style{
                        ts.color.rgba(),
                        ts.z,
                        int(std::lround(std::max(0.0f, ts.penWidth) * 1000.0f)),
                    };
                    groupedPaths[style].moveTo(a);
                    groupedPaths[style].lineTo(b);
                    groupedColors[style] = ts.color;
                }
            }

            // Post-back builds QGraphicsPathItems on the main thread.
            QMetaObject::invokeMethod(this,
                [this, fp,
                 groupedPaths = std::move(groupedPaths),
                 groupedColors = std::move(groupedColors)]() mutable {
                    if (_lastIntersectFp == fp) {
                        invalidateIntersect();
                        _intersectionItems.reserve(groupedPaths.size());
                        for (const auto& [style, path] : groupedPaths) {
                            if (path.isEmpty()) continue;
                            auto* item = new QGraphicsPathItem(path);
                            QPen pen(groupedColors[style]);
                            pen.setWidthF(
                                static_cast<qreal>(style.widthQ) / 1000.0);
                            pen.setCapStyle(Qt::RoundCap);
                            pen.setJoinStyle(Qt::RoundJoin);
                            pen.setCosmetic(true);
                            item->setPen(pen);
                            item->setBrush(Qt::NoBrush);
                            item->setZValue(style.z);
                            _scene->addItem(item);
                            _intersectionItems.push_back(item);
                        }
                        if (_view) _view->viewport()->update();
                    }
                    _planeWorkerBusy.store(false,
                                           std::memory_order_release);
                },
                Qt::QueuedConnection);
            _backgroundWorkers.fetch_sub(1, std::memory_order_release);
        });
}

void CAdaptiveVolumeViewer::renderFlattenedIntersections(const std::shared_ptr<Surface>& surf)
{
    auto activeSeg = std::dynamic_pointer_cast<QuadSurface>(surf);
    // Only the active segmentation viewer gets plane intersections drawn in UV.
    // If the view hosts some other QuadSurface there's nothing to render.
    if (!activeSeg || _state->surface("segmentation") != activeSeg) {
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

    struct PlaneEntry {
        std::shared_ptr<PlaneSurface> plane;
        QColor color;
    };
    std::array<std::pair<const char*, QColor>, 3> kPlaneSpecs = {{
        {"seg xy", QColor(255, 140, 0)}, // orange
        {"seg xz", QColor(Qt::red)},
        {"seg yz", QColor(Qt::yellow)},
    }};
    std::vector<PlaneEntry> planes;
    planes.reserve(3);
    for (const auto& [name, color] : kPlaneSpecs) {
        if (!_intersectTgts.count(name)) continue;
        if (auto p = std::dynamic_pointer_cast<PlaneSurface>(_state->surface(name))) {
            planes.push_back({std::move(p), color});
        }
    }
    if (planes.empty()) {
        invalidateIntersect();
        _lastIntersectFp = {};
        return;
    }

    // Fingerprint covers the only things the triangle clip depends on:
    // the 3 plane poses, the active segmentation identity + generation,
    // opacity, thickness. Patch count is folded in so a segment edit
    // that adds/removes patches still triggers a rebuild. When nothing
    // material changed, the rtree + triangle-clip pass is skipped —
    // that's the expensive work profiled at ~5% of main-thread CPU.
    IntersectFingerprint fp;
    auto mix = [](std::size_t s, std::size_t v) {
        return s ^ (v + 0x9e3779b9u + (s << 6) + (s >> 2));
    };
    auto hashVec = [&](std::size_t s, const cv::Vec3f& v) {
        for (int i = 0; i < 3; ++i)
            s = mix(s, std::hash<int>{}(int(std::lround(v[i] * 1000.0f))));
        return s;
    };
    std::size_t planesHash = 0;
    for (const auto& e : planes) {
        planesHash = hashVec(planesHash, e.plane->origin());
        planesHash = hashVec(planesHash, e.plane->normal({}, {}));
        planesHash = hashVec(planesHash, e.plane->basisX());
        planesHash = hashVec(planesHash, e.plane->basisY());
        planesHash = mix(planesHash,
            std::hash<uint32_t>{}(uint32_t(e.color.rgba())));
    }
    fp.flattenedPlanesHash = planesHash;
    fp.opacityQ = int(std::lround(_intersectionOpacity * 1000.0f));
    fp.thicknessQ = int(std::lround(_intersectionThickness * 1000.0f));
    fp.patchCount = patchIndex->patchCount();
    fp.surfaceCount = patchIndex->surfaceCount();
    fp.activeSegHash = std::hash<const void*>{}(activeSeg.get());
    fp.targetGenerationHash = std::hash<uint64_t>{}(
        patchIndex->generation(activeSeg));
    // Fold the camera state into the fingerprint so pan/zoom invalidates
    // the cache — surfaceToScene() below consumes all of this.
    std::size_t cameraHash = 0;
    auto hashInt = [&](std::size_t s, int v) {
        return mix(s, std::hash<int>{}(v));
    };
    cameraHash = hashInt(cameraHash, int(std::lround(_camSurfX * 1000.0f)));
    cameraHash = hashInt(cameraHash, int(std::lround(_camSurfY * 1000.0f)));
    cameraHash = hashInt(cameraHash, int(std::lround(_camScale * 1000.0f)));
    cameraHash = hashInt(cameraHash, _framebuffer.width());
    cameraHash = hashInt(cameraHash, _framebuffer.height());
    if (_view) {
        const QTransform t = _view->transform();
        auto q = [](qreal v) { return int(std::lround(v * 1000.0)); };
        cameraHash = hashInt(cameraHash, q(t.m11()));
        cameraHash = hashInt(cameraHash, q(t.m12()));
        cameraHash = hashInt(cameraHash, q(t.m21()));
        cameraHash = hashInt(cameraHash, q(t.m22()));
        cameraHash = hashInt(cameraHash, q(t.dx()));
        cameraHash = hashInt(cameraHash, q(t.dy()));
    }
    fp.cameraHash = cameraHash;
    fp.valid = true;
    if (_lastIntersectFp == fp && !_intersectionItems.empty()) {
        return;
    }
    invalidateIntersect();
    _lastIntersectFp = fp;

    // Everything past this point is pure computation over `activeSeg`,
    // `planes`, `patchIndex`, and a snapshot of the camera transform. We
    // dispatch it to QThreadPool so the rtree walk + per-triangle plane
    // clip don't block input processing — on the heavy workload
    // (~1.97M patches) this pass was measurable stalls on the main
    // thread. The result is posted back via invokeMethod(Queued) and
    // applied to the scene on the main thread.
    if (_flattenedWorkerBusy.exchange(true, std::memory_order_acq_rel)) {
        // A previous compute is still running. _intersectionsDirty will
        // stay true; _renderTimer will re-enter here once the worker
        // finishes and resets the flag.
        _intersectionsDirty = true;
        _lastIntersectFp = {};
        return;
    }

    Rect3D allBounds{cv::Vec3f(0, 0, 0), cv::Vec3f(1, 1, 1)};
    if (_volume) {
        auto [w, h, d] = _volume->shape();
        allBounds.high = {static_cast<float>(w),
                          static_cast<float>(h),
                          static_cast<float>(d)};
    }

    const float clipTol = std::max(_intersectionThickness, 1e-4f);

    // Snapshot camera state so the worker can call a pure surfaceToScene
    // equivalent without touching Qt objects. _view->transform() /
    // _framebuffer / _camSurfX/Y/Scale are all read here on the main
    // thread.
    struct CamSnapshot {
        float camSurfX, camSurfY, camScale;
        float vpCx, vpCy;
        QTransform viewToScene;
    };
    CamSnapshot cam{
        _camSurfX, _camSurfY, _camScale,
        static_cast<float>(_framebuffer.width()) * 0.5f,
        static_cast<float>(_framebuffer.height()) * 0.5f,
        _view ? _view->transform().inverted() : QTransform(),
    };

    const float penWidth = std::max(_intersectionThickness,
                                    kActiveIntersectionMinWidthDelta);
    const float opacity = std::clamp(
        _intersectionOpacity * kActiveIntersectionOpacityScale, 0.0f, 1.0f);

    std::vector<QColor> colors;
    colors.reserve(planes.size());
    std::vector<std::shared_ptr<PlaneSurface>> planeSurfs;
    planeSurfs.reserve(planes.size());
    for (const auto& e : planes) {
        colors.push_back(e.color);
        planeSurfs.push_back(e.plane);
    }

    _backgroundWorkers.fetch_add(1, std::memory_order_acq_rel);
    QThreadPool::globalInstance()->start(
        [this, fp, allBounds, clipTol, cam,
         activeSeg, patchIndex, planeSurfs = std::move(planeSurfs),
         colors = std::move(colors), penWidth, opacity]() mutable {
            auto isFiniteScalar = [](double v) {
                uint64_t bits;
                std::memcpy(&bits, &v, sizeof(bits));
                return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
            };
            auto surfToScene = [&](float sx, float sy) -> QPointF {
                const qreal vx = (sx - cam.camSurfX) * cam.camScale + cam.vpCx;
                const qreal vy = (sy - cam.camSurfY) * cam.camScale + cam.vpCy;
                // QGraphicsView::mapToScene(QPoint) == viewportTransform-inverse
                // applied to the point. inverted() was captured on the main
                // thread; apply it here without touching QGraphicsView.
                return cam.viewToScene.map(QPointF(vx, vy));
            };

            std::vector<QPainterPath> paths(planeSurfs.size());
            patchIndex->forEachTriangle(allBounds, activeSeg,
                [&](const SurfacePatchIndex::TriangleCandidate& tri) {
                    for (size_t idx = 0; idx < planeSurfs.size(); ++idx) {
                        auto seg = SurfacePatchIndex::clipTriangleToPlane(
                            tri, *planeSurfs[idx], clipTol);
                        if (!seg) continue;
                        cv::Vec3f a = activeSeg->loc(seg->surfaceParams[0]);
                        cv::Vec3f b = activeSeg->loc(seg->surfaceParams[1]);
                        QPointF pa = surfToScene(a[0], a[1]);
                        QPointF pb = surfToScene(b[0], b[1]);
                        if (!isFiniteScalar(pa.x()) || !isFiniteScalar(pa.y())
                         || !isFiniteScalar(pb.x()) || !isFiniteScalar(pb.y()))
                            continue;
                        paths[idx].moveTo(pa);
                        paths[idx].lineTo(pb);
                    }
                });

            // Post back to main thread. Queued connection — if `this` is
            // destroyed before the event fires, Qt drops it silently.
            QMetaObject::invokeMethod(this,
                [this, fp, paths = std::move(paths),
                 colors = std::move(colors), penWidth, opacity]() mutable {
                    // Bail if fingerprint changed while we were computing.
                    // _renderTimer will notice _intersectionsDirty and
                    // reschedule.
                    if (_lastIntersectFp == fp) {
                        invalidateIntersect();
                        _intersectionItems.reserve(colors.size());
                        for (size_t idx = 0; idx < paths.size(); ++idx) {
                            if (paths[idx].isEmpty()) continue;
                            QColor c = colors[idx];
                            c.setAlphaF(opacity);
                            auto* item = new QGraphicsPathItem(paths[idx]);
                            QPen pen(c);
                            pen.setWidthF(static_cast<qreal>(penWidth));
                            pen.setCapStyle(Qt::RoundCap);
                            pen.setJoinStyle(Qt::RoundJoin);
                            pen.setCosmetic(true);
                            item->setPen(pen);
                            item->setBrush(Qt::NoBrush);
                            item->setZValue(kActiveIntersectionZ);
                            _scene->addItem(item);
                            _intersectionItems.push_back(item);
                        }
                        if (_view) _view->viewport()->update();
                    }
                    _flattenedWorkerBusy.store(false,
                                               std::memory_order_release);
                },
                Qt::QueuedConnection);
            _backgroundWorkers.fetch_sub(1, std::memory_order_release);
        });
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
    cv::Vec3f surfLoc = {sp[0], sp[1], _camera.zOff};
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
    const float lo = std::clamp(low, 0.0f, 65535.0f);
    const float hi = std::clamp(high, 0.0f, 65535.0f);
    if (lo == _windowLow && hi == _windowHigh) return;
    _windowLow = lo;
    _windowHigh = hi;
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

    // Z display: for plane viewers, shift+scroll moves the 'focus' POI
    // rather than writing _camera.zOff, so we show the plane's signed
    // offset along its own normal from the world origin. For the
    // segmentation (flattened) view, shift+scroll writes _camera.zOff
    // directly, so show that.
    float zDisplay = _camera.zOff;
    if (auto surf = _surfWeak.lock()) {
        if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
            const cv::Vec3f n = plane->normal(cv::Vec3f(0, 0, 0), {});
            const double len = cv::norm(n);
            if (len > 1e-6) {
                const cv::Vec3f nHat = n * static_cast<float>(1.0 / len);
                zDisplay = plane->origin().dot(nHat);
            }
        }
    }
    QString status = QString("%1x 1:%2 z=%3")
        .arg(static_cast<double>(_camera.scale), 0, 'f', 2)
        .arg(1 << _camera.dsScaleIdx)
        .arg(static_cast<double>(zDisplay), 0, 'f', 1);

    const float fps = measuredFps();
    if (fps > 0.0f) {
        status += QString(" | %1 fps").arg(static_cast<double>(fps), 0, 'f', 1);
    }

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
    if (_volume) {
        int ns = static_cast<int>(_volume->numScales());
        std::vector<float> sfs(ns);
        for (int i = 0; i < ns; i++) sfs[i] = static_cast<float>(size_t{1} << i);
        _camera.recalcPyramidLevel(ns, sfs.data());
    }
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
