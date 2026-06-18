#pragma once

#include <QPointer>
#include <QString>

#include <functional>
#include <vector>

class CWindow;
class CChunkedVolumeViewer;

// Drives the viewer through a recorded camera-state timeline (see
// RenderBenchRecorder) so the existing [vc3d-profile] render logs measure an
// identical workload every run. Each frame is settled to a warm, stable render
// before advancing. Emits [vc3d-replay] frame markers and quits when done.
class RenderBenchReplay
{
public:
    struct Header {
        QString volpkgPath;
        QString volumeId;
        QString segmentId;
        int viewportW = 0;
        int viewportH = 0;
        bool volpkgIsRemote = false;
    };

    struct Keyframe {
        QString surface;
        float surfacePtrX = 0.0f, surfacePtrY = 0.0f;
        float scale = 1.0f, zOffset = 0.0f;
        float zDirX = 0.0f, zDirY = 0.0f, zDirZ = 0.0f;
        int dsScaleIdx = 0;
    };

    // Loads a recording from JSON. Returns false on parse/IO error.
    bool load(const QString& path);

    void setWarmPass(bool warm) { _warm = warm; }

    // Opens the recorded volume+segment in the window, then replays every
    // keyframe. Calls QApplication::quit() on completion. Blocking (pumps the
    // event loop internally); call after the event loop is running.
    void run(CWindow& window);

private:
    bool _warm = false;
    Header _header;
    std::vector<Keyframe> _keyframes;

    // Pump the event loop until pred() is true or timeoutMs elapses. Returns
    // pred()'s final value.
    static bool waitForCondition(const std::function<bool()>& pred, int timeoutMs);
    // Pump until the viewer is quiescent for quietWindowMs continuously, or
    // maxFrameMs elapses. Returns true if it settled, false on timeout or if the
    // viewer is destroyed mid-settle.
    static bool settleFrame(QPointer<CChunkedVolumeViewer> viewer, int maxFrameMs, int quietWindowMs);
};
