#include "RenderBenchReplay.hpp"

#include <QApplication>
#include <QCoreApplication>
#include <QElapsedTimer>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QObject>
#include <QPainter>
#include <QThread>
#include <QWidget>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

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
    double repaintMs = -1.0;
    double fullRenderMs = -1.0;
    double settledMs = -1.0;
    double workerMs = -1.0;
    bool settled = false;

    double qtOverheadMs() const
    {
        if (fullRenderMs < 0.0 || workerMs < 0.0)
            return -1.0;
        return fullRenderMs - workerMs;
    }
};

struct PhaseSummary {
    double minMs = 0.0;
    double maxMs = 0.0;
    double totalMs = 0.0;
    int count = 0;

    void add(double value)
    {
        if (value < 0.0)
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
        return count > 0 ? totalMs / static_cast<double>(count) : 0.0;
    }
};

struct TimedPaintSample {
    int paint = 0;
    int view = -1;
    int renderedView = -1;
    double viewAgeMs = -1.0;
    double staleFrameMs = -1.0;
    double paintCostMs = -1.0;
};

double elapsedMs(const QElapsedTimer& timer)
{
    return static_cast<double>(timer.nsecsElapsed()) / 1000000.0;
}

double nsToMs(qint64 ns)
{
    return static_cast<double>(ns) / 1000000.0;
}

std::string msString(double value)
{
    if (value < 0.0)
        return "-";
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << value;
    return out.str();
}

CChunkedVolumeViewer::CameraState cameraStateFromKeyframe(const RenderBenchReplay::Keyframe& kf)
{
    CChunkedVolumeViewer::CameraState cs;
    cs.surfacePtrX = kf.surfacePtrX;
    cs.surfacePtrY = kf.surfacePtrY;
    cs.scale = kf.scale;
    cs.zOffset = kf.zOffset;
    cs.zOffsetWorldDir = {kf.zDirX, kf.zDirY, kf.zDirZ};
    return cs;
}
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
        if (_replayLimit > 0 &&
            static_cast<int>(_keyframes.size()) >= _replayLimit) {
            break;
        }
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
    std::vector<FrameTiming> timedFrames;
    timedFrames.reserve(_keyframes.size());
    QImage offscreenPaintTarget;

    auto paintViewToTarget = [&](double* paintCostMs = nullptr) {
        auto* gv = viewer ? viewer->graphicsView() : nullptr;
        if (!gv)
            return false;
        if (_offscreen4k) {
            const QSize viewportSize = gv->viewport() ? gv->viewport()->size() : QSize();
            if (!viewportSize.isValid() || viewportSize.isEmpty())
                return false;
            if (offscreenPaintTarget.size() != viewportSize ||
                offscreenPaintTarget.format() != QImage::Format_RGB32) {
                offscreenPaintTarget = QImage(viewportSize, QImage::Format_RGB32);
            }
            QElapsedTimer paintTimer;
            paintTimer.start();
            QPainter painter(&offscreenPaintTarget);
            gv->render(&painter,
                       QRectF(QPointF(0.0, 0.0), QSizeF(viewportSize)),
                       QRect(QPoint(0, 0), viewportSize),
                       Qt::IgnoreAspectRatio);
            painter.end();
            if (paintCostMs)
                *paintCostMs = elapsedMs(paintTimer);
        } else if (paintCostMs) {
            *paintCostMs = 0.0;
        }
        return true;
    };

    if (_timedProfile) {
        if (_keyframes.empty()) {
            QApplication::quit();
            return;
        }

        viewer->applyCameraStateForReplayRepaint(cameraStateFromKeyframe(_keyframes[0]));
        viewer->renderVisible(true, "replay timed profile prime");
        (void)settleFrame(viewer, maxFrameMs, kQuietWindowMs);

        QElapsedTimer profileTimer;
        profileTimer.start();
        const qint64 periodNs = static_cast<qint64>(std::max(1, _timedProfilePeriodMs)) * 1000000LL;
        const std::size_t firstTimedFrame = _keyframes.size() > 1 ? 1 : 0;
        const qint64 runUntilNs =
            static_cast<qint64>(_keyframes.size() - firstTimedFrame + 1) * periodNs;

        int currentView = 0;
        int displayedView = 0;
        int paintCount = 0;
        std::vector<qint64> viewChangeNs(_keyframes.size(), 0);
        std::unordered_map<std::uint64_t, int> serialToView;
        std::vector<TimedPaintSample> paintSamples;
        paintSamples.reserve(_keyframes.size() * 2);

        std::cout << std::setw(6) << "paint"
                  << " " << std::setw(6) << "view"
                  << " " << std::setw(8) << "rendered"
                  << " " << std::setw(10) << "view_age"
                  << " " << std::setw(12) << "stale_frame"
                  << " " << std::setw(10) << "paint_cost"
                  << '\n';
        std::cout << std::setw(6) << "------"
                  << " " << std::setw(6) << "------"
                  << " " << std::setw(8) << "--------"
                  << " " << std::setw(10) << "----------"
                  << " " << std::setw(12) << "------------"
                  << " " << std::setw(10) << "----------"
                  << '\n';

        auto samplePaint = [&]() {
            double paintCostMs = 0.0;
            if (!paintViewToTarget(&paintCostMs))
                return;
            const qint64 nowNs = profileTimer.nsecsElapsed();
            TimedPaintSample sample;
            sample.paint = ++paintCount;
            sample.view = currentView;
            sample.renderedView = displayedView;
            if (currentView >= 0 && static_cast<std::size_t>(currentView) < viewChangeNs.size())
                sample.viewAgeMs = nsToMs(nowNs - viewChangeNs[static_cast<std::size_t>(currentView)]);
            if (displayedView >= 0) {
                if (displayedView < currentView &&
                    static_cast<std::size_t>(displayedView + 1) < viewChangeNs.size()) {
                    sample.staleFrameMs =
                        nsToMs(nowNs - viewChangeNs[static_cast<std::size_t>(displayedView + 1)]);
                } else {
                    sample.staleFrameMs = 0.0;
                }
            }
            sample.paintCostMs = paintCostMs;
            paintSamples.push_back(sample);
            std::cout << std::setw(6) << sample.paint
                      << " " << std::setw(6) << sample.view
                      << " " << std::setw(8) << sample.renderedView
                      << " " << std::setw(10) << msString(sample.viewAgeMs)
                      << " " << std::setw(12) << msString(sample.staleFrameMs)
                      << " " << std::setw(10) << msString(sample.paintCostMs)
                      << '\n';
        };

        const auto submittedConn = QObject::connect(
            viewer, &CChunkedVolumeViewer::renderFrameSubmitted,
            viewer, [&](std::uint64_t serial) {
                serialToView[serial] = currentView;
            });
        const auto completedConn = QObject::connect(
            viewer, &CChunkedVolumeViewer::renderFrameCompleted,
            viewer, [&](std::uint64_t serial, double) {
                if (auto it = serialToView.find(serial); it != serialToView.end())
                    displayedView = it->second;
            });
        const auto paintConn = QObject::connect(
            viewer->graphicsView(), &CVolumeViewerView::paintCompleted,
            viewer->graphicsView(), [&] { samplePaint(); });

        std::size_t nextFrame = firstTimedFrame;
        qint64 nextSwitchNs = 0;
        while (profileTimer.nsecsElapsed() < runUntilNs) {
            const qint64 nowNs = profileTimer.nsecsElapsed();
            if (nextFrame < _keyframes.size() && nowNs >= nextSwitchNs) {
                currentView = static_cast<int>(nextFrame);
                viewChangeNs[nextFrame] = nowNs;
                viewer->applyCameraStateForReplayRepaint(cameraStateFromKeyframe(_keyframes[nextFrame]));
                if (auto* gv = viewer->graphicsView()) {
                    if (auto* viewport = gv->viewport())
                        viewport->update();
                }
                viewer->renderVisible(true, "replay timed profile");
                ++nextFrame;
                nextSwitchNs += periodNs;
            }
            QCoreApplication::processEvents(QEventLoop::AllEvents, kPumpSliceMs);
            QCoreApplication::sendPostedEvents(nullptr, QEvent::MetaCall);
            QThread::msleep(1);
        }
        QCoreApplication::processEvents(QEventLoop::AllEvents, kPumpSliceMs);
        QCoreApplication::sendPostedEvents(nullptr, QEvent::MetaCall);

        QObject::disconnect(paintConn);
        QObject::disconnect(completedConn);
        QObject::disconnect(submittedConn);

        PhaseSummary viewAgeSummary;
        PhaseSummary staleFrameSummary;
        PhaseSummary paintCostSummary;
        for (const auto& sample : paintSamples) {
            viewAgeSummary.add(sample.viewAgeMs);
            staleFrameSummary.add(sample.staleFrameMs);
            paintCostSummary.add(sample.paintCostMs);
        }
        std::cout << '\n'
                  << std::left << std::setw(14) << "phase"
                  << " " << std::right << std::setw(8) << "count"
                  << " " << std::setw(10) << "min"
                  << " " << std::setw(10) << "mean"
                  << " " << std::setw(10) << "max"
                  << " " << std::left << std::setw(28) << "status"
                  << '\n';
        std::cout << std::left << std::setw(14) << "--------------"
                  << " " << std::right << std::setw(8) << "--------"
                  << " " << std::setw(10) << "----------"
                  << " " << std::setw(10) << "----------"
                  << " " << std::setw(10) << "----------"
                  << " " << std::left << std::setw(28) << "----------------------------"
                  << '\n';
        auto logSummary = [](const char* name, const PhaseSummary& s) {
            if (s.count == 0) {
                std::cout << std::left << std::setw(14) << name
                          << " " << std::right << std::setw(8) << 0
                          << " " << std::setw(10) << "-"
                          << " " << std::setw(10) << "-"
                          << " " << std::setw(10) << "-"
                          << " " << std::left << std::setw(28) << "no samples"
                          << '\n';
                return;
            }
            std::cout << std::left << std::setw(14) << name
                      << " " << std::right << std::setw(8) << s.count
                      << " " << std::setw(10) << msString(s.minMs)
                      << " " << std::setw(10) << msString(s.avg())
                      << " " << std::setw(10) << msString(s.maxMs)
                      << " " << std::left << std::setw(28) << "ok"
                      << '\n';
        };
        logSummary("view_age", viewAgeSummary);
        logSummary("stale_frame", staleFrameSummary);
        logSummary("paint_cost", paintCostSummary);
        QApplication::quit();
        return;
    }

    auto driveFrame = [&](int i, const Keyframe& kf, bool timed, bool forceSettle) -> FrameTiming {
        FrameTiming timing;
        if (!viewer)
            return timing;
        CChunkedVolumeViewer::CameraState cs = cameraStateFromKeyframe(kf);
        (void)waitForCondition([&] {
            return !viewer || viewer->isRenderQuiescent();
        }, maxFrameMs);

        QElapsedTimer frameTimer;
        frameTimer.start();

        auto paintViewportNow = [&]() {
            auto* gv = viewer->graphicsView();
            if (!gv)
                return false;

            if (_offscreen4k)
                return paintViewToTarget();

            bool painted = false;
            QMetaObject::Connection paintConn;
            {
                paintConn = QObject::connect(gv, &CVolumeViewerView::paintCompleted,
                                             gv, [&] { painted = true; });
                if (auto* viewport = gv->viewport())
                    viewport->repaint();
            }
            if (!painted) {
                (void)waitForCondition([&] { return painted || !viewer; }, maxFrameMs);
            }
            if (paintConn)
                QObject::disconnect(paintConn);
            return painted;
        };

        viewer->applyCameraStateForReplayRepaint(cs);
        if (paintViewportNow())
            timing.repaintMs = elapsedMs(frameTimer);

        bool fullRenderDone = false;
        double fullRenderWorkerMs = -1.0;
        auto renderConn = QObject::connect(
            viewer, &CChunkedVolumeViewer::renderFrameCompleted,
            viewer, [&](std::uint64_t, double workerElapsedMs) {
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
            if (paintViewportNow())
                timing.fullRenderMs = elapsedMs(frameTimer);
            timing.workerMs = fullRenderWorkerMs;
        }

        if (forceSettle || !_skipChunkComplete) {
            timing.settled = settleFrame(viewer, maxFrameMs, kQuietWindowMs);
            if (timing.settled && paintViewportNow())
                timing.settledMs = elapsedMs(frameTimer);
        }

        if (timed) {
            std::cout << std::setw(6) << i
                      << " " << std::setw(10) << msString(timing.repaintMs)
                      << " " << std::setw(14) << msString(timing.fullRenderMs)
                      << " " << std::setw(10) << msString(timing.settledMs)
                      << " " << std::setw(14) << msString(timing.qtOverheadMs())
                      << " " << std::setw(7) << viewer->datasetScaleIndex()
                      << " " << std::setw(10) << kf.dsScaleIdx
                      << '\n';
        }
        return timing;
    };

    // 7. Optional warm pass (discard timings).
    if (_warm) {
        for (std::size_t i = 0; i < _keyframes.size(); ++i)
            (void)driveFrame(static_cast<int>(i), _keyframes[i], /*timed=*/false, /*forceSettle=*/false);
    }

    // 8. Prime the first recorded view to a fully settled state, but do not count
    // it. This removes first-frame setup/cache effects from the timed summaries.
    std::size_t timedStart = 0;
    if (!_keyframes.empty()) {
        (void)driveFrame(0, _keyframes[0], /*timed=*/false, /*forceSettle=*/true);
        timedStart = 1;
    }

    // 9. Timed pass.
    std::cout << std::setw(6) << "frame"
              << " " << std::setw(10) << "repaint"
              << " " << std::setw(14) << "full_render"
              << " " << std::setw(10) << "settled"
              << " " << std::setw(14) << "qt_overhead"
              << " " << std::setw(7) << "ds"
              << " " << std::setw(10) << "recorded"
              << '\n';
    std::cout << std::setw(6) << "------"
              << " " << std::setw(10) << "----------"
              << " " << std::setw(14) << "--------------"
              << " " << std::setw(10) << "----------"
              << " " << std::setw(14) << "--------------"
              << " " << std::setw(7) << "-------"
              << " " << std::setw(10) << "----------"
              << '\n';
    for (std::size_t i = timedStart; i < _keyframes.size(); ++i) {
        timedFrames.push_back(driveFrame(static_cast<int>(i), _keyframes[i],
                                         /*timed=*/true, /*forceSettle=*/false));
    }

    PhaseSummary repaintSummary;
    PhaseSummary fullRenderSummary;
    PhaseSummary settledSummary;
    PhaseSummary qtOverheadSummary;
    for (const auto& frame : timedFrames) {
        repaintSummary.add(frame.repaintMs);
        fullRenderSummary.add(frame.fullRenderMs);
        settledSummary.add(frame.settledMs);
        qtOverheadSummary.add(frame.qtOverheadMs());
    }
    std::cout << '\n'
              << std::left << std::setw(14) << "phase"
              << " " << std::right << std::setw(8) << "count"
              << " " << std::setw(10) << "min"
              << " " << std::setw(10) << "mean"
              << " " << std::setw(10) << "max"
              << " " << std::left << std::setw(28) << "status"
              << '\n';
    std::cout << std::left << std::setw(14) << "--------------"
              << " " << std::right << std::setw(8) << "--------"
              << " " << std::setw(10) << "----------"
              << " " << std::setw(10) << "----------"
              << " " << std::setw(10) << "----------"
              << " " << std::left << std::setw(28) << "----------------------------"
              << '\n';
    auto logSummary = [](const char* name, const PhaseSummary& s) {
        if (s.count == 0) {
            std::cout << std::left << std::setw(14) << name
                      << " " << std::right << std::setw(8) << 0
                      << " " << std::setw(10) << "-"
                      << " " << std::setw(10) << "-"
                      << " " << std::setw(10) << "-"
                      << " " << std::left << std::setw(28) << "no samples"
                      << '\n';
            return;
        }
        std::cout << std::left << std::setw(14) << name
                  << " " << std::right << std::setw(8) << s.count
                  << " " << std::setw(10) << msString(s.minMs)
                  << " " << std::setw(10) << msString(s.avg())
                  << " " << std::setw(10) << msString(s.maxMs)
                  << " " << std::left << std::setw(28) << "ok"
                  << '\n';
    };
    logSummary("repaint", repaintSummary);
    logSummary("full_render", fullRenderSummary);
    logSummary("qt_overhead", qtOverheadSummary);
    if (_skipChunkComplete) {
        std::cout << std::left << std::setw(14) << "settled"
                  << " " << std::right << std::setw(8) << 0
                  << " " << std::setw(10) << "-"
                  << " " << std::setw(10) << "-"
                  << " " << std::setw(10) << "-"
                  << " " << std::left << std::setw(28) << "skipped by option"
                  << '\n';
    } else {
        logSummary("settled", settledSummary);
    }
    std::cout << std::left << std::setw(14) << "fast_render"
              << " " << std::right << std::setw(8) << 0
              << " " << std::setw(10) << "-"
              << " " << std::setw(10) << "-"
              << " " << std::setw(10) << "-"
              << " " << std::left << std::setw(28)
              << (_skipFastRender ? "absent, skip requested" : "absent")
              << '\n';
    std::cout.flush();
    QApplication::quit();
}
