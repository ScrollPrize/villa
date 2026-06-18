#include "RenderBenchReplay.hpp"

#include <QApplication>
#include <QCoreApplication>
#include <QElapsedTimer>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QObject>
#include <QWidget>

#include <algorithm>
#include <cstdint>

#include "CState.hpp"
#include "CWindow.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Surface.hpp"

namespace
{
constexpr int kReadyTimeoutMs = 60000;     // wait for volume/surface to come up
constexpr int kMaxFrameMsLocal = 30000;    // per-keyframe settle ceiling, local data
constexpr int kMaxFrameMsRemote = 180000;  // per-keyframe ceiling for S3 (slow/flaky net)
constexpr int kQuietWindowMs = 150;        // continuous quiescence before "settled"
constexpr int kPumpSliceMs = 5;            // event-loop poll granularity
constexpr int kOffscreen4kW = 3840;
constexpr int kOffscreen4kH = 2160;

struct FrameTiming {
    qint64 repaintMs = -1;
    qint64 fullRenderMs = -1;
    qint64 settledMs = -1;
    qint64 workerMs = -1;
    bool settled = false;
};

struct PhaseSummary {
    qint64 minMs = 0;
    qint64 maxMs = 0;
    qint64 totalMs = 0;
    int count = 0;

    void add(qint64 value)
    {
        if (value < 0)
            return;
        if (count == 0) {
            minMs = value;
            maxMs = value;
        } else {
            minMs = std::min(minMs, value);
            maxMs = std::max(maxMs, value);
        }
        totalMs += value;
        ++count;
    }

    double avg() const
    {
        return count > 0 ? static_cast<double>(totalMs) / static_cast<double>(count) : 0.0;
    }
};
}  // namespace

bool RenderBenchReplay::load(const QString& path)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) {
        Logger()->error("[vc3d-replay] cannot open {}", path.toStdString());
        return false;
    }
    QJsonParseError err;
    const auto doc = QJsonDocument::fromJson(f.readAll(), &err);
    f.close();
    if (err.error != QJsonParseError::NoError || !doc.isObject()) {
        Logger()->error("[vc3d-replay] parse error in {}: {}",
                        path.toStdString(), err.errorString().toStdString());
        return false;
    }
    const auto root = doc.object();
    const auto h = root["header"].toObject();
    _header.volpkgPath = h["volpkgPath"].toString();
    _header.volumeId = h["volumeId"].toString();
    _header.segmentId = h["segmentId"].toString();
    _header.volpkgIsRemote = h["volpkgIsRemote"].toBool();
    const auto vp = h["viewport"].toObject();
    _header.viewportW = vp["width"].toInt();
    _header.viewportH = vp["height"].toInt();

    _keyframes.clear();
    for (const auto v : root["keyframes"].toArray()) {
        const auto o = v.toObject();
        Keyframe kf;
        kf.surface = o["surface"].toString();
        kf.surfacePtrX = static_cast<float>(o["surfacePtrX"].toDouble());
        kf.surfacePtrY = static_cast<float>(o["surfacePtrY"].toDouble());
        kf.scale = static_cast<float>(o["scale"].toDouble());
        kf.zOffset = static_cast<float>(o["zOffset"].toDouble());
        const auto dir = o["zOffsetWorldDir"].toArray();
        if (dir.size() == 3) {
            kf.zDirX = static_cast<float>(dir[0].toDouble());
            kf.zDirY = static_cast<float>(dir[1].toDouble());
            kf.zDirZ = static_cast<float>(dir[2].toDouble());
        }
        kf.dsScaleIdx = o["dsScaleIdx"].toInt();
        _keyframes.push_back(kf);
    }
    Logger()->info("[vc3d-replay] loaded {} keyframes volume='{}' segment='{}'",
                   _keyframes.size(), _header.volumeId.toStdString(),
                   _header.segmentId.toStdString());
    return true;
}

bool RenderBenchReplay::waitForCondition(const std::function<bool()>& pred, int timeoutMs)
{
    QElapsedTimer t;
    t.start();
    while (!pred()) {
        if (t.elapsed() >= timeoutMs)
            return pred();
        QCoreApplication::processEvents(QEventLoop::AllEvents, kPumpSliceMs);
        QCoreApplication::sendPostedEvents(nullptr, QEvent::MetaCall);
    }
    return true;
}

bool RenderBenchReplay::settleFrame(QPointer<CChunkedVolumeViewer> viewer, int maxFrameMs, int quietWindowMs)
{
    QElapsedTimer total;
    total.start();
    QElapsedTimer quiet;
    bool quietRunning = false;
    forever {
        QCoreApplication::processEvents(QEventLoop::AllEvents, kPumpSliceMs);
        QCoreApplication::sendPostedEvents(nullptr, QEvent::MetaCall);

        if (!viewer)
            return false;  // viewer torn down mid-settle
        const bool quiescent = viewer->isRenderQuiescent()
                            && viewer->chunkFetchesInFlight() == 0;
        if (quiescent) {
            if (!quietRunning) {
                quiet.start();
                quietRunning = true;
            }
            if (quiet.elapsed() >= quietWindowMs)
                return true;
        } else {
            quietRunning = false;
        }
        if (total.elapsed() >= maxFrameMs)
            return false;
    }
}

void RenderBenchReplay::run(CWindow& window)
{
    auto fail = [](const QString& msg) {
        Logger()->error("[vc3d-replay] {}", msg.toStdString());
        QApplication::exit(1);
    };

    // 1. Open the recorded project.
    window.OpenVolume(_header.volpkgPath);
    if (!window._state || !window._state->vpkg()) {
        fail("failed to open volpkg " + _header.volpkgPath);
        return;
    }

    // 2. Select the recorded volume.
    auto vpkg = window._state->vpkg();
    const std::string volId = _header.volumeId.toStdString();
    if (!volId.empty() && vpkg->hasVolume(volId)) {
        window.setVolume(vpkg->volume(volId));
    }

    QPointer<CChunkedVolumeViewer> viewer = window.segmentationViewer();
    if (!viewer) {
        fail("no segmentation viewer");
        return;
    }

    // 3. Wait for the volume to be live on the viewer.
    const bool volReady = waitForCondition([&] {
        if (!viewer)
            return false;
        auto v = viewer->currentVolume();
        return v && (volId.empty() || v->id() == volId);
    }, kReadyTimeoutMs);
    if (!volReady) {
        fail("timed out waiting for volume " + _header.volumeId);
        return;
    }

    // 4. Activate the recorded segment (ensure it's loaded first).
    const std::string segId = _header.segmentId.toStdString();
    if (!segId.empty()) {
        auto surf = vpkg->getSurface(segId);
        if (!surf)
            surf = vpkg->loadSurface(segId);
        if (!surf) {
            fail("segment not found in volpkg: " + _header.segmentId);
            return;
        }
        window.onSurfaceActivated(_header.segmentId, surf.get());
        // onSurfaceActivated marks the active surface; the segmentation viewer
        // shows whatever is bound to the "segmentation" slot, so drive that too.
        window._state->setSurface("segmentation", surf, false, false);
    }

    // 5. Wait for the surface to be set on the viewer.
    const bool surfReady = waitForCondition([&] {
        return viewer && viewer->currentSurface() != nullptr
            && (segId.empty() || viewer->surfName() == "segmentation");
    }, kReadyTimeoutMs);
    if (!surfReady) {
        fail("timed out waiting for segment " + _header.segmentId);
        return;
    }

    // 6. Pin the viewport size for deterministic framebuffer dims / pyramid level.
    // The framebuffer is sized from graphicsView()->viewport()->size(), and the
    // view lives inside an MDI subwindow + layout, so resizing the viewport alone
    // gets overridden on the next layout pass. setFixedSize on the view itself
    // forces the viewport to match and survives relayout; we leave it fixed for
    // the whole replay since we never need it resizable here.
    const int targetViewportW = _offscreen4k ? kOffscreen4kW : _header.viewportW;
    const int targetViewportH = _offscreen4k ? kOffscreen4kH : _header.viewportH;
    if (viewer && targetViewportW > 0 && targetViewportH > 0) {
        if (auto* gv = viewer->graphicsView()) {
            auto pinViewport = [&](int targetW, int targetH) {
                gv->setFixedSize(targetW, targetH);
                QCoreApplication::processEvents(QEventLoop::AllEvents, kPumpSliceMs);
                QSize got = gv->viewport()->size();
                if (got.width() != targetW || got.height() != targetH) {
                    const QSize viewSize = gv->size();
                    gv->setFixedSize(std::max(1, viewSize.width() + targetW - got.width()),
                                     std::max(1, viewSize.height() + targetH - got.height()));
                    QCoreApplication::processEvents(QEventLoop::AllEvents, kPumpSliceMs);
                }
            };
            pinViewport(targetViewportW, targetViewportH);
            QCoreApplication::processEvents(QEventLoop::AllEvents, kPumpSliceMs);
            const QSize got = gv->viewport()->size();
            if (got.width() != targetViewportW || got.height() != targetViewportH) {
                Logger()->warn("[vc3d-replay] viewport pinned to {}x{} but got {}x{} "
                               "(framebuffer follows the actual viewport size)",
                               targetViewportW, targetViewportH,
                               got.width(), got.height());
            }
        }
    }

    // Remote (S3) data streams chunks in over the network and can stall on a
    // flaky connection; give it a much longer settle ceiling than local data.
    const int maxFrameMs = _header.volpkgIsRemote ? kMaxFrameMsRemote : kMaxFrameMsLocal;
    Logger()->info("[vc3d-replay] remote={} per-frame ceiling={}ms quiet_window={}ms "
                   "offscreen4k={} skip_chunk_complete={} skip_fast_render={}",
                   _header.volpkgIsRemote, maxFrameMs, kQuietWindowMs,
                   _offscreen4k, _skipChunkComplete, _skipFastRender);
    Logger()->info("[vc3d-replay] current phases: repaint=current framebuffer paint, "
                   "full_render=first accepted worker render, settled=render quiet plus zero chunk fetches");
    Logger()->info("[vc3d-replay] current fast_render phase: absent on this branch "
                   "(camera changes submit the normal worker render directly)");

    std::vector<FrameTiming> timedFrames;
    timedFrames.reserve(_keyframes.size());

    auto driveFrame = [&](int i, const Keyframe& kf, bool timed, bool forceSettle) -> FrameTiming {
        FrameTiming timing;
        if (!viewer)
            return timing;
        CChunkedVolumeViewer::CameraState cs;
        cs.surfacePtrX = kf.surfacePtrX;
        cs.surfacePtrY = kf.surfacePtrY;
        cs.scale = kf.scale;
        cs.zOffset = kf.zOffset;
        cs.zOffsetWorldDir = {kf.zDirX, kf.zDirY, kf.zDirZ};
        if (timed) {
            Logger()->info("[vc3d-replay] frame={} begin scale={:.4f} zOff={:.3f}",
                           i, kf.scale, kf.zOffset);
        }
        (void)waitForCondition([&] {
            return !viewer || viewer->isRenderQuiescent();
        }, maxFrameMs);

        QElapsedTimer frameTimer;
        frameTimer.start();

        bool repainted = false;
        QMetaObject::Connection paintConn;
        if (auto* gv = viewer->graphicsView()) {
            paintConn = QObject::connect(gv, &CVolumeViewerView::paintCompleted,
                                         gv, [&] { repainted = true; });
        }

        viewer->applyCameraStateForReplayRepaint(cs);
        if (auto* gv = viewer->graphicsView()) {
            if (auto* viewport = gv->viewport())
                viewport->repaint();
        }
        if (!repainted) {
            (void)waitForCondition([&] { return repainted || !viewer; }, maxFrameMs);
        }
        if (paintConn)
            QObject::disconnect(paintConn);
        if (repainted)
            timing.repaintMs = frameTimer.elapsed();

        bool fullRenderDone = false;
        qint64 fullRenderWorkerMs = -1;
        auto renderConn = QObject::connect(
            viewer, &CChunkedVolumeViewer::renderFrameCompleted,
            viewer, [&](std::uint64_t, qint64 workerElapsedMs) {
                if (!fullRenderDone) {
                    fullRenderDone = true;
                    fullRenderWorkerMs = workerElapsedMs;
                }
            });

        viewer->renderVisible(true, "replay full render");
        const bool fullRenderSeen = waitForCondition([&] {
            return fullRenderDone || !viewer;
        }, maxFrameMs);
        QObject::disconnect(renderConn);
        if (fullRenderSeen && fullRenderDone) {
            timing.fullRenderMs = frameTimer.elapsed();
            timing.workerMs = fullRenderWorkerMs;
        }

        if (forceSettle || !_skipChunkComplete) {
            timing.settled = settleFrame(viewer, maxFrameMs, kQuietWindowMs);
            timing.settledMs = frameTimer.elapsed();
        }

        if (timed) {
            const QSize fb = viewer->graphicsView()
                ? viewer->graphicsView()->viewport()->size() : QSize(0, 0);
            Logger()->info("[vc3d-replay] frame={} phases repaint_ms={} full_render_ms={} "
                           "settled_ms={} full_worker_ms={} dsLevel={} recordedDs={} fb={}x{} settled={}",
                           i, timing.repaintMs, timing.fullRenderMs,
                           timing.settledMs, timing.workerMs,
                           viewer->datasetScaleIndex(), kf.dsScaleIdx,
                           fb.width(), fb.height(), timing.settled);
        }
        return timing;
    };

    // 7. Optional warm pass (discard timings).
    if (_warm) {
        Logger()->info("[vc3d-replay] warm pass over {} keyframes", _keyframes.size());
        for (std::size_t i = 0; i < _keyframes.size(); ++i)
            (void)driveFrame(static_cast<int>(i), _keyframes[i], /*timed=*/false, /*forceSettle=*/false);
    }

    // 8. Prime the first recorded view to a fully settled state, but do not count
    // it. This removes first-frame setup/cache effects from the timed summaries.
    std::size_t timedStart = 0;
    if (!_keyframes.empty()) {
        Logger()->info("[vc3d-replay] priming frame=0 until settled (not timed)");
        (void)driveFrame(0, _keyframes[0], /*timed=*/false, /*forceSettle=*/true);
        timedStart = 1;
    }

    // 9. Timed pass.
    Logger()->info("[vc3d-replay] timed pass over {} keyframes (start_frame={})",
                   _keyframes.size() - timedStart, timedStart);
    for (std::size_t i = timedStart; i < _keyframes.size(); ++i) {
        timedFrames.push_back(driveFrame(static_cast<int>(i), _keyframes[i],
                                         /*timed=*/true, /*forceSettle=*/false));
    }

    PhaseSummary repaintSummary;
    PhaseSummary fullRenderSummary;
    PhaseSummary settledSummary;
    PhaseSummary workerSummary;
    for (const auto& frame : timedFrames) {
        repaintSummary.add(frame.repaintMs);
        fullRenderSummary.add(frame.fullRenderMs);
        settledSummary.add(frame.settledMs);
        workerSummary.add(frame.workerMs);
    }
    Logger()->info("[vc3d-replay] summary_table |phase|count|min_ms|mean_ms|max_ms|status|");
    Logger()->info("[vc3d-replay] summary_table |---|---:|---:|---:|---:|---|");
    auto logSummary = [](const char* name, const PhaseSummary& s) {
        if (s.count == 0) {
            Logger()->info("[vc3d-replay] summary_table |{}|0|-|-|-|no samples|", name);
            return;
        }
        Logger()->info("[vc3d-replay] summary_table |{}|{}|{}|{:.2f}|{}|ok|",
                       name, s.count, s.minMs, s.avg(), s.maxMs);
    };
    logSummary("repaint", repaintSummary);
    logSummary("full_render", fullRenderSummary);
    logSummary("full_worker", workerSummary);
    if (_skipChunkComplete) {
        Logger()->info("[vc3d-replay] summary_table |settled|0|-|-|-|skipped by option|");
    } else {
        logSummary("settled", settledSummary);
    }
    Logger()->info("[vc3d-replay] summary_table |fast_render|0|-|-|-|present=false skipped_by_option={}|",
                   _skipFastRender);

    Logger()->info("[vc3d-replay] done");
    QApplication::quit();
}
