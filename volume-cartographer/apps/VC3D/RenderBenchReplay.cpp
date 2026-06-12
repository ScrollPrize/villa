#include "RenderBenchReplay.hpp"

#include <QApplication>
#include <QCoreApplication>
#include <QElapsedTimer>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QSizePolicy>

#include "CState.hpp"
#include "CWindow.hpp"
#include "ViewerManager.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Surface.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

namespace
{
constexpr int kReadyTimeoutMs = 60000;     // wait for volume/surface to come up
constexpr int kMaxFrameMsLocal = 30000;    // per-keyframe settle ceiling, local data
constexpr int kMaxFrameMsRemote = 180000;  // per-keyframe ceiling for S3 (slow/flaky net)
constexpr int kQuietWindowMs = 150;        // continuous quiescence before "settled"
constexpr int kPumpSliceMs = 0;            // replay poll should not wait for more platform events

struct StageTiming {
    qint64 wallMs = -1;
    double cpuMs = -1.0;
};

struct FrameTiming {
    int index = 0;
    bool settled = false;
    StageTiming cachedPreview;
    StageTiming interactiveRender;
    StageTiming stableRender;
    StageTiming fullSettle;
    CChunkedVolumeViewer::ReplayRenderState::InterfaceTiming interactiveInterface;
    CChunkedVolumeViewer::ReplayRenderState::InterfaceTiming stableInterface;
    qint64 processEventsTotalMs = 0;
    qint64 processEventsMaxMs = 0;
    int processEventsCalls = 0;
    int datasetLevel = 0;
    int recordedDatasetLevel = 0;
    int framebufferW = 0;
    int framebufferH = 0;
    float scale = 1.0f;
    float zOffset = 0.0f;
};

double processCpuMs()
{
    return 1000.0 * static_cast<double>(std::clock()) / static_cast<double>(CLOCKS_PER_SEC);
}

std::string formatWall(const StageTiming& timing)
{
    if (timing.wallMs < 0) {
        return "-";
    }
    return std::to_string(timing.wallMs);
}

std::string formatCpu(const StageTiming& timing)
{
    if (timing.cpuMs < 0.0) {
        return "-";
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(0) << timing.cpuMs;
    return out.str();
}

std::string formatThreadingFactor(const StageTiming& timing)
{
    if (timing.wallMs < 0 || timing.cpuMs < 0.0) {
        return "-";
    }
    std::ostringstream out;
    const double factor = timing.wallMs > 0
        ? timing.cpuMs / static_cast<double>(timing.wallMs)
        : 0.0;
    out << std::fixed << std::setprecision(2) << factor;
    return out.str();
}

void appendStageHeader(std::ostringstream& out, const char* prefix)
{
    out << std::setw(5) << (std::string(prefix) + "w")
        << std::setw(6) << (std::string(prefix) + "c")
        << std::setw(6) << (std::string(prefix) + "x");
}

void appendStageRow(std::ostringstream& out, const StageTiming& timing)
{
    out << std::setw(5) << formatWall(timing)
        << std::setw(6) << formatCpu(timing)
        << std::setw(6) << formatThreadingFactor(timing);
}

std::string formatMs(qint64 value)
{
    if (value < 0) {
        return "-";
    }
    return std::to_string(value);
}

void appendInterfaceHeader(std::ostringstream& out)
{
    out << std::setw(5) << "irw"
        << std::setw(5) << "iqd"
        << std::setw(5) << "iup"
        << std::setw(5) << "rq"
        << std::setw(5) << "se"
        << std::setw(5) << "rt"
        << std::setw(5) << "sb"
        << std::setw(5) << "wk"
        << std::setw(4) << "ca"
        << std::setw(5) << "la"
        << std::setw(4) << "ci"
        << std::setw(5) << "li"
        << std::setw(5) << "rw"
        << std::setw(5) << "qd"
        << std::setw(5) << "gw"
        << std::setw(5) << "up"
        << std::setw(5) << "pt"
        << std::setw(5) << "it"
        << std::setw(5) << "dw";
}

void appendInterfaceRow(
    std::ostringstream& out,
    const CChunkedVolumeViewer::ReplayRenderState::InterfaceTiming& interactive,
    const CChunkedVolumeViewer::ReplayRenderState::InterfaceTiming& timing)
{
    out << std::setw(5) << formatMs(interactive.workerMs)
        << std::setw(5) << formatMs(interactive.queueMs)
        << std::setw(5) << formatMs(interactive.updateToPaintMs)
        << std::setw(5) << formatMs(timing.requestMs)
        << std::setw(5) << formatMs(timing.settleMs)
        << std::setw(5) << formatMs(timing.renderTimerMs)
        << std::setw(5) << formatMs(timing.submitMs)
        << std::setw(5) << formatMs(timing.workerStartMs)
        << std::setw(4) << timing.activeChunkReadyResets
        << std::setw(5) << formatMs(timing.lastActiveChunkReadyMs)
        << std::setw(4) << timing.idleChunkReadyResets
        << std::setw(5) << formatMs(timing.lastIdleChunkReadyMs)
        << std::setw(5) << formatMs(timing.workerMs)
        << std::setw(5) << formatMs(timing.queueMs)
        << std::setw(5) << formatMs(timing.guiMs)
        << std::setw(5) << formatMs(timing.updateToPaintMs)
        << std::setw(5) << formatMs(timing.paintTotalMs)
        << std::setw(5) << formatMs(timing.paintSceneItemsMs)
        << std::setw(5) << formatMs(timing.paintDrawMs);
}

std::string timingTableHeader()
{
    std::ostringstream out;
    out << std::right
        << std::setw(4) << "fr"
        << std::setw(3) << "ok";
    appendStageHeader(out, "c");
    appendStageHeader(out, "i");
    appendStageHeader(out, "s");
    appendStageHeader(out, "q");
    appendInterfaceHeader(out);
    out
        << std::setw(5) << "ep"
        << std::setw(5) << "em"
        << std::setw(5) << "ec"
        << std::setw(4) << "ds"
        << std::setw(4) << "rd"
        << std::setw(10) << "fb"
        << std::setw(6) << "scale"
        << std::setw(6) << "zoff";
    return out.str();
}

std::string timingTableLegend()
{
    return "legend: fr=frame ok=settled c=cached i=interactive s=stable q=quiet; w=wall c=cpu x=cpu/wall; irw=interactive renderer iqd=interactive queued cb iup=interactive update->paint rq=stable requested se=settled rt=render timer sb=submitted wk=worker start ca=active chunk resets la=last active chunk ci=idle chunk resets li=last idle chunk rw=stable renderer qd=queued cb gw=gui handoff up=update->paint pt=paint total it=scene/items dw=fb draw; ep=processEvents total em=max ec=calls; rd=recorded_ds";
}

std::string timingTableSeparator()
{
    return std::string(timingTableHeader().size(), '-');
}

std::string timingTableRow(const FrameTiming& timing)
{
    std::ostringstream fb;
    fb << timing.framebufferW << 'x' << timing.framebufferH;

    std::ostringstream out;
    out << std::right
        << std::setw(4) << timing.index
        << std::setw(3) << (timing.settled ? "y" : "n");
    appendStageRow(out, timing.cachedPreview);
    appendStageRow(out, timing.interactiveRender);
    appendStageRow(out, timing.stableRender);
    appendStageRow(out, timing.fullSettle);
    appendInterfaceRow(out, timing.interactiveInterface, timing.stableInterface);
    out
        << std::setw(5) << timing.processEventsTotalMs
        << std::setw(5) << timing.processEventsMaxMs
        << std::setw(5) << timing.processEventsCalls
        << std::setw(4) << timing.datasetLevel
        << std::setw(4) << timing.recordedDatasetLevel
        << std::setw(10) << fb.str()
        << std::setw(6) << std::fixed << std::setprecision(3) << timing.scale
        << std::setw(6) << std::fixed << std::setprecision(2) << timing.zOffset;
    return out.str();
}

double percentile(std::vector<qint64> sortedValues, double p)
{
    if (sortedValues.empty()) {
        return 0.0;
    }
    std::sort(sortedValues.begin(), sortedValues.end());
    if (sortedValues.size() == 1) {
        return static_cast<double>(sortedValues.front());
    }

    const double clamped = std::clamp(p, 0.0, 1.0);
    const double pos = clamped * static_cast<double>(sortedValues.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(pos));
    const auto hi = static_cast<std::size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lo);
    return static_cast<double>(sortedValues[lo]) * (1.0 - frac)
         + static_cast<double>(sortedValues[hi]) * frac;
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

void RenderBenchReplay::prepareBenchmarkView(CWindow& window, QPointer<CChunkedVolumeViewer> viewer)
{
    if (!viewer) {
        return;
    }

    window.switchToMainWorkspace();
    if (_targetWindowSize.isValid()) {
        window.showNormal();
        window.setGeometry(0, 0, _targetWindowSize.width(), _targetWindowSize.height());
        window.resize(_targetWindowSize);
        Logger()->info("[vc3d-replay] replay window size forced to {}x{}",
                       _targetWindowSize.width(), _targetWindowSize.height());
    } else {
        window.showMaximized();
    }
    window.raise();
    window.activateWindow();

    if (!window._focusedViewActive) {
        window.toggleFocusedView();
    }

    QMdiSubWindow* replaySubWindow = qobject_cast<QMdiSubWindow*>(viewer->parentWidget());
    QMdiArea* replayMdiArea = replaySubWindow ? replaySubWindow->mdiArea() : nullptr;

    int disabledViewers = 0;
    if (window._viewerManager) {
        window._viewerManager->setSegmentationEditActive(false);
        window._viewerManager->setOverlayVolume(nullptr, std::string{});
        window._viewerManager->setOverlayOpacity(0.0f);
        window._viewerManager->setHighlightedSurfaceIds({});

        for (auto* baseViewer : window._viewerManager->baseViewers()) {
            if (!baseViewer) {
                continue;
            }
            const bool targetViewer = baseViewer == viewer.data();
            baseViewer->setOverlayVolume(nullptr);
            baseViewer->setOverlayOpacity(0.0f);
            baseViewer->setPlaneIntersectionLinesVisible(false);
            baseViewer->setIntersects({});
            baseViewer->setIntersectionOpacity(0.0f);
            baseViewer->setIntersectionThickness(0.0f);
            baseViewer->setHighlightedSurfaceIds({});
            baseViewer->setSurfaceOverlayEnabled(false);
            baseViewer->setSurfaceOverlays({});
            baseViewer->setShowDirectionHints(false);
            baseViewer->setShowSurfaceNormals(false);
            baseViewer->setSegmentationEditActive(false);
            baseViewer->setSegmentationIntersectionDeferral(false);
            baseViewer->setLinkedCursorVolumePoint(std::nullopt);
            baseViewer->setBBoxMode(false);
            baseViewer->clearSelections();
            baseViewer->clearAllOverlayGroups();

            if (auto* chunked = dynamic_cast<CChunkedVolumeViewer*>(baseViewer)) {
                chunked->setReplaySuppressViewportUpdates(!targetViewer);
            }
            ++disabledViewers;
        }
    } else {
        viewer->setOverlayVolume(nullptr);
        viewer->setPlaneIntersectionLinesVisible(false);
        viewer->setIntersects({});
        viewer->setSurfaceOverlayEnabled(false);
        viewer->setSurfaceOverlays({});
        viewer->setShowDirectionHints(false);
        viewer->setShowSurfaceNormals(false);
        viewer->clearAllOverlayGroups();
        disabledViewers = 1;
    }

    int hiddenSubWindows = 0;
    auto hideNonReplaySubWindows = [&](QMdiArea* mdiArea) {
        if (!mdiArea) {
            return;
        }
        const bool replayArea = mdiArea == replayMdiArea;
        for (auto* subWindow : mdiArea->subWindowList()) {
            if (!subWindow || subWindow == replaySubWindow) {
                continue;
            }
            subWindow->hide();
            subWindow->setUpdatesEnabled(false);
            if (auto* child = subWindow->widget()) {
                child->setUpdatesEnabled(false);
            }
            ++hiddenSubWindows;
        }
        if (!replayArea) {
            mdiArea->setUpdatesEnabled(false);
            if (auto* viewport = mdiArea->viewport()) {
                viewport->setUpdatesEnabled(false);
            }
        }
    };
    hideNonReplaySubWindows(window.mdiArea);
    hideNonReplaySubWindows(window._fiberSliceMdiArea);
    hideNonReplaySubWindows(window._intersectionsMdiArea);

    Logger()->info("[vc3d-replay] disabled replay drawing overlays on {} viewers; hid {} non-target subwindows",
                   disabledViewers, hiddenSubWindows);

    if (auto* subWindow = replaySubWindow) {
        if (auto* mdi = subWindow->mdiArea()) {
            mdi->setActiveSubWindow(subWindow);
        }
        subWindow->setUpdatesEnabled(true);
        if (auto* child = subWindow->widget()) {
            child->setUpdatesEnabled(true);
        }
        subWindow->show();
        subWindow->showMaximized();
        subWindow->raise();
        subWindow->activateWindow();
    }
    if (auto* gv = viewer->graphicsView()) {
        gv->setMinimumSize(0, 0);
        gv->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
        gv->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        gv->updateGeometry();
        gv->setFocus();
    }

    for (int i = 0; i < 5; ++i) {
        QCoreApplication::processEvents(QEventLoop::AllEvents, kPumpSliceMs);
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

    // 6. Strip away replay-only UI/overlay overhead and maximize the replayed
    // slice/surface view. The viewport pin below intentionally happens after
    // this so the final graphics-view framebuffer follows the trace header.
    prepareBenchmarkView(window, viewer);

    // 7. Let the maximized window/subwindow layout determine the render area.
    // The framebuffer is sized from graphicsView()->viewport()->size(), so do
    // not pin the inner graphics view: doing so leaves unused space inside the
    // maximized subwindow. The timing table records the actual framebuffer.
    if (viewer && _header.viewportW > 0 && _header.viewportH > 0) {
        if (auto* gv = viewer->graphicsView()) {
            QCoreApplication::processEvents(QEventLoop::AllEvents, kPumpSliceMs);
            const QSize got = gv->viewport()->size();
            Logger()->info("[vc3d-replay] trace viewport={}x{} actual framebuffer={}x{}",
                           _header.viewportW, _header.viewportH,
                           got.width(), got.height());
        }
        viewer->setReplaySuppressViewportUpdates(false);
        viewer->setReplayPollRenderResults(true);
    }

    // Remote (S3) data streams chunks in over the network and can stall on a
    // flaky connection; give it a much longer settle ceiling than local data.
    const int maxFrameMs = _header.volpkgIsRemote ? kMaxFrameMsRemote : kMaxFrameMsLocal;
    Logger()->info("[vc3d-replay] remote={} per-frame settle ceiling={}ms",
                   _header.volpkgIsRemote, maxFrameMs);

    auto driveFrame = [&](int i, const Keyframe& kf) -> FrameTiming {
        FrameTiming timing;
        timing.index = i;
        timing.recordedDatasetLevel = kf.dsScaleIdx;
        timing.scale = kf.scale;
        timing.zOffset = kf.zOffset;
        if (!viewer)
            return timing;
        const auto startState = viewer->replayRenderState();
        CChunkedVolumeViewer::CameraState cs;
        cs.surfacePtrX = kf.surfacePtrX;
        cs.surfacePtrY = kf.surfacePtrY;
        cs.scale = kf.scale;
        cs.zOffset = kf.zOffset;
        cs.zOffsetWorldDir = {kf.zDirX, kf.zDirY, kf.zDirZ};

        QElapsedTimer total;
        total.start();
        QElapsedTimer quiet;
        bool quietRunning = false;
        const double startCpuMs = processCpuMs();
        auto mark = [&](StageTiming& stage) {
            if (stage.wallMs >= 0) {
                return;
            }
            stage.wallMs = total.elapsed();
            stage.cpuMs = std::max(0.0, processCpuMs() - startCpuMs);
        };
        auto sampleMilestones = [&]() {
            if (!viewer) {
                return;
            }
            const auto state = viewer->replayRenderState();
            if (state.cachedPreviewSerial != startState.cachedPreviewSerial) {
                mark(timing.cachedPreview);
            }
            if (state.interactiveRenderSerial != startState.interactiveRenderSerial) {
                mark(timing.interactiveRender);
            }
            if (state.interactiveTiming.serial != 0 &&
                state.interactiveTiming.serial != startState.interactiveTiming.serial) {
                timing.interactiveInterface = state.interactiveTiming;
            }
            if (state.stableRenderSerial != startState.stableRenderSerial) {
                mark(timing.stableRender);
            }
            if (state.stableTiming.serial != 0 &&
                state.stableTiming.serial != startState.stableTiming.serial) {
                timing.stableInterface = state.stableTiming;
            }
        };

        viewer->applyInteractiveCameraState(cs);
        sampleMilestones();

        forever {
            QElapsedTimer pumpTimer;
            pumpTimer.start();
            QCoreApplication::processEvents(QEventLoop::AllEvents, kPumpSliceMs);
            const qint64 pumpMs = pumpTimer.elapsed();
            timing.processEventsTotalMs += pumpMs;
            timing.processEventsMaxMs = std::max(timing.processEventsMaxMs, pumpMs);
            ++timing.processEventsCalls;

            if (!viewer) {
                mark(timing.fullSettle);
                break;
            }
            viewer->drainReplayRenderResults();
            sampleMilestones();

            const bool quiescent = viewer->isRenderQuiescent()
                                && viewer->chunkFetchesInFlight() == 0;
            if (quiescent) {
                if (!quietRunning) {
                    quiet.start();
                    quietRunning = true;
                }
                if (quiet.elapsed() >= kQuietWindowMs) {
                    timing.settled = true;
                    mark(timing.fullSettle);
                    break;
                }
            } else {
                quietRunning = false;
            }
            if (total.elapsed() >= maxFrameMs) {
                mark(timing.fullSettle);
                break;
            }
        }

        timing.datasetLevel = viewer->datasetScaleIndex();
        const QSize fb = viewer->graphicsView()
            ? viewer->graphicsView()->viewport()->size() : QSize(0, 0);
        timing.framebufferW = fb.width();
        timing.framebufferH = fb.height();
        return timing;
    };

    // 8. Optional warm pass (discard timings).
    if (_warm) {
        Logger()->info("[vc3d-replay] warm pass over {} keyframes", _keyframes.size());
        for (std::size_t i = 0; i < _keyframes.size(); ++i)
            (void)driveFrame(static_cast<int>(i), _keyframes[i]);
    }

    // 9. Timed pass.
    Logger()->info("[vc3d-replay] timed pass over {} keyframes", _keyframes.size());
    Logger()->info("[vc3d-replay] timing table follows (plain stdout, no per-row log prefix)");
    std::cout << timingTableLegend() << '\n'
              << timingTableHeader() << '\n'
              << timingTableSeparator() << '\n';

    std::vector<FrameTiming> timings;
    timings.reserve(_keyframes.size());
    for (std::size_t i = 0; i < _keyframes.size(); ++i) {
        FrameTiming timing = driveFrame(static_cast<int>(i), _keyframes[i]);
        timings.push_back(timing);
        std::cout << timingTableRow(timing) << std::endl;
    }

    std::vector<qint64> wallMs;
    std::vector<double> cpuMs;
    wallMs.reserve(timings.size());
    cpuMs.reserve(timings.size());
    int settledCount = 0;
    for (const auto& timing : timings) {
        if (timing.fullSettle.wallMs >= 0) {
            wallMs.push_back(timing.fullSettle.wallMs);
        }
        if (timing.fullSettle.cpuMs >= 0.0) {
            cpuMs.push_back(timing.fullSettle.cpuMs);
        }
        if (timing.settled) {
            ++settledCount;
        }
    }
    if (!wallMs.empty()) {
        const auto [minIt, maxIt] = std::minmax_element(wallMs.begin(), wallMs.end());
        const qint64 sum = std::accumulate(wallMs.begin(), wallMs.end(), qint64{0});
        const double mean = static_cast<double>(sum) / static_cast<double>(wallMs.size());
        const double cpuSum = std::accumulate(cpuMs.begin(), cpuMs.end(), 0.0);
        const double cpuMean = cpuMs.empty() ? 0.0 : cpuSum / static_cast<double>(cpuMs.size());
        const double aggregateThreading = sum > 0
            ? cpuSum / static_cast<double>(sum)
            : 0.0;
        Logger()->info("[vc3d-replay] timing summary count={} settled={} min_ms={} "
                       "mean_ms={:.2f} max_ms={} p50_ms={:.2f} p95_ms={:.2f} "
                       "mean_cpu_ms={:.2f} threading_factor={:.2f}",
                       wallMs.size(), settledCount, *minIt, mean, *maxIt,
                       percentile(wallMs, 0.50), percentile(wallMs, 0.95),
                       cpuMean, aggregateThreading);
    } else {
        Logger()->info("[vc3d-replay] timing summary count=0 settled=0");
    }

    Logger()->info("[vc3d-replay] done");
    QApplication::quit();
}
