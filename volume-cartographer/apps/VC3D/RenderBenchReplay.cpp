#include "RenderBenchReplay.hpp"

#include <QApplication>
#include <QCoreApplication>
#include <QElapsedTimer>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include "CState.hpp"
#include "CWindow.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"

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
    if (viewer && _header.viewportW > 0 && _header.viewportH > 0) {
        if (auto* gv = viewer->graphicsView()) {
            gv->setFixedSize(_header.viewportW, _header.viewportH);
            QCoreApplication::processEvents(QEventLoop::AllEvents, kPumpSliceMs);
            const QSize got = gv->viewport()->size();
            if (got.width() != _header.viewportW || got.height() != _header.viewportH) {
                Logger()->warn("[vc3d-replay] viewport pinned to {}x{} but got {}x{} "
                               "(framebuffer follows the actual viewport size)",
                               _header.viewportW, _header.viewportH,
                               got.width(), got.height());
            }
        }
    }

    // Remote (S3) data streams chunks in over the network and can stall on a
    // flaky connection; give it a much longer settle ceiling than local data.
    const int maxFrameMs = _header.volpkgIsRemote ? kMaxFrameMsRemote : kMaxFrameMsLocal;
    Logger()->info("[vc3d-replay] remote={} per-frame settle ceiling={}ms",
                   _header.volpkgIsRemote, maxFrameMs);

    auto driveFrame = [&](int i, const Keyframe& kf, bool timed) {
        if (!viewer)
            return;
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
        QElapsedTimer frameTimer;
        frameTimer.start();
        viewer->applyCameraState(cs, /*forceRender=*/true);
        const bool settled = settleFrame(viewer, maxFrameMs, kQuietWindowMs);
        if (timed) {
            const QSize fb = viewer->graphicsView()
                ? viewer->graphicsView()->viewport()->size() : QSize(0, 0);
            Logger()->info("[vc3d-replay] frame={} end wall_ms={} dsLevel={} "
                           "recordedDs={} fb={}x{} settled={}",
                           i, frameTimer.elapsed(), viewer->datasetScaleIndex(),
                           kf.dsScaleIdx, fb.width(), fb.height(), settled);
        }
    };

    // 7. Optional warm pass (discard timings).
    if (_warm) {
        Logger()->info("[vc3d-replay] warm pass over {} keyframes", _keyframes.size());
        for (std::size_t i = 0; i < _keyframes.size(); ++i)
            driveFrame(static_cast<int>(i), _keyframes[i], /*timed=*/false);
    }

    // 8. Timed pass.
    Logger()->info("[vc3d-replay] timed pass over {} keyframes", _keyframes.size());
    for (std::size_t i = 0; i < _keyframes.size(); ++i)
        driveFrame(static_cast<int>(i), _keyframes[i], /*timed=*/true);

    Logger()->info("[vc3d-replay] done");
    QApplication::quit();
}
