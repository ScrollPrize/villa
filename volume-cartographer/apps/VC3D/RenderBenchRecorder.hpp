#pragma once

#include <QElapsedTimer>
#include <QObject>
#include <QPointer>
#include <QString>

#include <vector>

#include "volume_viewers/CChunkedVolumeViewer.hpp"

// Records a timeline of viewer camera states while the user navigates, for
// deterministic replay under --profile. One keyframe per completed stable
// render; identical consecutive camera states (chunk-stream re-renders) are
// deduped. See RenderBenchReplay for playback.
class RenderBenchRecorder : public QObject
{
    Q_OBJECT

public:
    struct Header {
        QString volpkgPath;
        bool volpkgIsRemote = false;
        QString volumeId;
        QString segmentId;
        int viewportW = 0;
        int viewportH = 0;
        std::size_t cacheSizeGB = 0;
        QString samplingMethod = "trilinear";
        QString vc3dCommit;
    };

    struct Keyframe {
        qint64 tMs = 0;
        QString surface;
        float surfacePtrX = 0.0f;
        float surfacePtrY = 0.0f;
        float scale = 1.0f;
        float zOffset = 0.0f;
        float zDirX = 0.0f, zDirY = 0.0f, zDirZ = 0.0f;
        int surfaceViewRotationQuarterTurns = 0;
        bool surfaceViewFlippedHorizontally = false;
        int zSlice = 0;
        int dsScaleIdx = 0;
    };

    explicit RenderBenchRecorder(QString outPath, QObject* parent = nullptr);

    bool attached() const { return _attached; }
    // Connects to the viewer's render signal and stamps the session header.
    void attach(CChunkedVolumeViewer* viewer, Header header);
    // Writes the recorded session to the output path as JSON. Returns false on
    // I/O error. Safe to call with no keyframes.
    bool save() const;

private slots:
    void onRender();
    void onZSliceChanged(int z);

private:
    QString _outPath;
    bool _attached = false;
    QPointer<CChunkedVolumeViewer> _viewer;
    Header _header;
    QElapsedTimer _clock;
    std::vector<Keyframe> _keyframes;
    int _lastZSlice = 0;
    bool _haveLast = false;
    Keyframe _last;
};
