#include "TileRenderController.hpp"

#include <QTimer>
#include <QImage>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>

#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Surface.hpp"

// --- Tile pipeline profiling ---
// Tracks submit→render→display latency per tile and prints percentile stats.
struct TileLatencyStats {
    struct Sample {
        double submitToRenderMs;  // time in render pool (queue + render)
        double renderToDisplayMs; // time from render done to QPixmap set
        double totalMs;           // submit to display
    };
    std::vector<Sample> samples;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();

    void add(std::chrono::steady_clock::time_point submit,
             std::chrono::steady_clock::time_point renderDone,
             std::chrono::steady_clock::time_point displayed) {
        if (submit == std::chrono::steady_clock::time_point{}) return;
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        samples.push_back({ms(submit, renderDone), ms(renderDone, displayed), ms(submit, displayed)});
    }

    void maybePrint() {
        auto now = std::chrono::steady_clock::now();
        if (samples.size() < 5) return;
        if (std::chrono::duration<double>(now - lastReport).count() < 2.0) return;

        auto percentile = [](std::vector<double>& v, double p) -> double {
            if (v.empty()) return 0;
            std::sort(v.begin(), v.end());
            size_t idx = std::min(static_cast<size_t>(v.size() * p), v.size() - 1);
            return v[idx];
        };

        std::vector<double> render, drain, total;
        for (auto& s : samples) {
            render.push_back(s.submitToRenderMs);
            drain.push_back(s.renderToDisplayMs);
            total.push_back(s.totalMs);
        }

        auto avg = [](const std::vector<double>& v) {
            return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        };

        fprintf(stderr,
            "\n=== Tile Pipeline Latency (%zu tiles) ===\n"
            "                  avg     min     p50     p90     p95     p99     max\n"
            "  render:    %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f ms\n"
            "  drain:     %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f ms\n"
            "  total:     %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f ms\n",
            samples.size(),
            avg(render), percentile(render, 0.0), percentile(render, 0.5),
            percentile(render, 0.9), percentile(render, 0.95), percentile(render, 0.99),
            percentile(render, 1.0),
            avg(drain), percentile(drain, 0.0), percentile(drain, 0.5),
            percentile(drain, 0.9), percentile(drain, 0.95), percentile(drain, 0.99),
            percentile(drain, 1.0),
            avg(total), percentile(total, 0.0), percentile(total, 0.5),
            percentile(total, 0.9), percentile(total, 0.95), percentile(total, 0.99),
            percentile(total, 1.0));

        samples.clear();
        lastReport = now;
    }
};

static TileLatencyStats g_tileStats;

TileRenderController::TileRenderController(TileScene* tileScene, RenderPool* sharedPool, QObject* parent)
    : QObject(parent)
    , _tileScene(tileScene)
    , _renderPool(sharedPool)
    , _viewportRenderer(&sharedPool->corePool())
{
    // 60fps vsync timer — runs only while there's work to do.
    _vsyncTimer = new QTimer(this);
    _vsyncTimer->setTimerType(Qt::PreciseTimer);
    _vsyncTimer->setInterval(16);  // ~60 Hz
    connect(_vsyncTimer, &QTimer::timeout, this, &TileRenderController::tick);

    // Wake the timer when workers complete tiles.
    connect(_renderPool, &RenderPool::tileReady, this, &TileRenderController::ensureTickRunning,
            Qt::QueuedConnection);

    // Direct pixel→QPixmap conversion during drain — skips TileGrid pixel storage.
    // One memcpy into QImage (which owns its buffer), then QPixmap::fromImage.
    _viewportRenderer.setResultCallback([this](TileRenderResult&& result) {
        if (result.pixels.empty()) return;

        QImage img(result.width, result.height, QImage::Format_RGB32);
        const int srcStride = result.width * 4;
        const int dstStride = img.bytesPerLine();
        if (srcStride == dstStride) {
            std::memcpy(img.bits(), result.pixels.data(),
                        static_cast<size_t>(srcStride) * result.height);
        } else {
            for (int y = 0; y < result.height; y++) {
                std::memcpy(img.scanLine(y),
                            reinterpret_cast<const uchar*>(result.pixels.data()) + y * srcStride,
                            srcStride);
            }
        }
        QPixmap pixmap = QPixmap::fromImage(std::move(img), Qt::NoFormatConversion);

        _tileScene->setTilePixmapOnly(result.worldKey, pixmap);
        _anyUpdatedThisTick = true;
        ++_tilesDisplayedThisTick;

        g_tileStats.add(result.submitTime, result.renderDone,
                        std::chrono::steady_clock::now());
    });
}

TileRenderController::~TileRenderController()
{
}

void TileRenderController::ensureTickRunning()
{
    if (!_vsyncTimer->isActive()) {
        // Fire immediately on first wake, then repeat at 16ms
        tick();
        _vsyncTimer->start();
    }
}

void TileRenderController::onCameraChanged(
    const TiledViewerCamera& camera,
    const std::shared_ptr<Surface>& surface,
    const std::shared_ptr<Volume>& volume,
    const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
    const QRectF& viewportRect)
{
    ensureTickRunning();
    _viewportRenderer.onCameraChanged(
        camera, surface, volume, buildParams,
        static_cast<float>(viewportRect.left()),
        static_cast<float>(viewportRect.top()),
        static_cast<float>(viewportRect.right()),
        static_cast<float>(viewportRect.bottom()));
}

void TileRenderController::onParamsChanged(
    const TiledViewerCamera& camera,
    const std::shared_ptr<Surface>& surface,
    const std::shared_ptr<Volume>& volume,
    const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
    const QRectF& viewportRect)
{
    ensureTickRunning();
    _viewportRenderer.onParamsChanged(
        camera, surface, volume, buildParams,
        static_cast<float>(viewportRect.left()),
        static_cast<float>(viewportRect.top()),
        static_cast<float>(viewportRect.right()),
        static_cast<float>(viewportRect.bottom()));
}

void TileRenderController::scheduleRender(
    const TiledViewerCamera& camera,
    const std::shared_ptr<Surface>& surface,
    const std::shared_ptr<Volume>& volume,
    const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
    const QRectF& viewportRect)
{
    _viewportRenderer.scheduleRender(
        camera, surface, volume, buildParams,
        static_cast<float>(viewportRect.left()),
        static_cast<float>(viewportRect.top()),
        static_cast<float>(viewportRect.right()),
        static_cast<float>(viewportRect.bottom()));
    ensureTickRunning();
}

void TileRenderController::cancelAll()
{
    _viewportRenderer.cancelAll();
}

void TileRenderController::clearState()
{
    _viewportRenderer.clearState();
}

void TileRenderController::markOverlaysDirty()
{
    _viewportRenderer.markOverlaysDirty();
    ensureTickRunning();
}

void TileRenderController::markChunkArrived()
{
    _viewportRenderer.markChunkArrived();
    ensureTickRunning();
}

void TileRenderController::setOverlayCallback(std::function<void()> cb)
{
    _viewportRenderer.setOverlayCallback(std::move(cb));
}

void TileRenderController::syncGridBounds(const ContentBounds& bounds, int viewportW, int viewportH)
{
    _viewportRenderer.tileGrid().rebuildGrid(bounds, viewportW, viewportH);
}

void TileRenderController::tick()
{
    _anyUpdatedThisTick = false;
    _tilesDisplayedThisTick = 0;

    auto tickStart = std::chrono::steady_clock::now();

    // drain + progressive refinement + overlays.
    // The result callback (set in constructor) converts pixels→QPixmap inline.
    bool moreWork = _viewportRenderer.tick();

    auto tickEnd = std::chrono::steady_clock::now();
    double tickMs = std::chrono::duration<double, std::milli>(tickEnd - tickStart).count();
    if (tickMs > 5.0 || _tilesDisplayedThisTick > 0) {
        fprintf(stderr, "[tick] %.1fms tiles=%d\n", tickMs, _tilesDisplayedThisTick);
    }

    if (_anyUpdatedThisTick) {
        g_tileStats.maybePrint();
        emit sceneNeedsUpdate();
    }

    if (!moreWork)
        _vsyncTimer->stop();
}
