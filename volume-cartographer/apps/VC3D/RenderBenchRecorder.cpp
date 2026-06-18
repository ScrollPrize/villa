#include "RenderBenchRecorder.hpp"

#include <cmath>

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include "vc/core/util/Logging.hpp"

namespace
{
// Quantized equality so chunk-stream re-renders (same viewpoint) collapse to one
// keyframe. Tolerances mirror the JSON precision we care about for replay.
bool sameCamera(const RenderBenchRecorder::Keyframe& a,
                const RenderBenchRecorder::Keyframe& b)
{
    auto close = [](float x, float y, float tol) { return std::abs(x - y) <= tol; };
    return a.surface == b.surface
        && close(a.surfacePtrX, b.surfacePtrX, 1e-3f)
        && close(a.surfacePtrY, b.surfacePtrY, 1e-3f)
        && close(a.scale, b.scale, 1e-4f)
        && close(a.zOffset, b.zOffset, 1e-3f)
        && close(a.zDirX, b.zDirX, 1e-4f)
        && close(a.zDirY, b.zDirY, 1e-4f)
        && close(a.zDirZ, b.zDirZ, 1e-4f);
}
}  // namespace

RenderBenchRecorder::RenderBenchRecorder(QString outPath, QObject* parent)
    : QObject(parent), _outPath(std::move(outPath))
{
}

void RenderBenchRecorder::attach(CChunkedVolumeViewer* viewer, Header header)
{
    if (_attached || !viewer)
        return;
    _viewer = viewer;
    _header = std::move(header);
    _clock.start();
    connect(viewer, &CChunkedVolumeViewer::overlaysUpdated,
            this, &RenderBenchRecorder::onRender);
    connect(viewer, &CChunkedVolumeViewer::sendZSliceChanged,
            this, &RenderBenchRecorder::onZSliceChanged);
    _attached = true;
    Logger()->info("[vc3d-record] attached volume='{}' segment='{}' -> {}",
                   _header.volumeId.toStdString(), _header.segmentId.toStdString(),
                   _outPath.toStdString());
}

void RenderBenchRecorder::onZSliceChanged(int z)
{
    _lastZSlice = z;
}

void RenderBenchRecorder::onRender()
{
    if (!_viewer)
        return;
    const auto cam = _viewer->cameraState();
    Keyframe kf;
    kf.tMs = _clock.elapsed();
    kf.surface = QString::fromStdString(_viewer->surfName());
    kf.surfacePtrX = cam.surfacePtrX;
    kf.surfacePtrY = cam.surfacePtrY;
    kf.scale = cam.scale;
    kf.zOffset = cam.zOffset;
    kf.zDirX = cam.zOffsetWorldDir[0];
    kf.zDirY = cam.zOffsetWorldDir[1];
    kf.zDirZ = cam.zOffsetWorldDir[2];
    kf.zSlice = _lastZSlice;
    kf.dsScaleIdx = _viewer->datasetScaleIndex();

    if (_haveLast && sameCamera(kf, _last))
        return;
    _keyframes.push_back(kf);
    _last = kf;
    _haveLast = true;
}

bool RenderBenchRecorder::save() const
{
    QJsonObject header;
    header["volpkgPath"] = _header.volpkgPath;
    header["volpkgIsRemote"] = _header.volpkgIsRemote;
    header["volumeId"] = _header.volumeId;
    header["segmentId"] = _header.segmentId;
    QJsonObject viewport;
    viewport["width"] = _header.viewportW;
    viewport["height"] = _header.viewportH;
    header["viewport"] = viewport;
    header["cacheSizeGB"] = static_cast<double>(_header.cacheSizeGB);
    header["samplingMethod"] = _header.samplingMethod;
    header["vc3dCommit"] = _header.vc3dCommit;

    QJsonArray keyframes;
    int i = 0;
    for (const auto& kf : _keyframes) {
        QJsonObject o;
        o["i"] = i++;
        o["tMs"] = static_cast<double>(kf.tMs);
        o["surface"] = kf.surface;
        o["surfacePtrX"] = kf.surfacePtrX;
        o["surfacePtrY"] = kf.surfacePtrY;
        o["scale"] = kf.scale;
        o["zOffset"] = kf.zOffset;
        QJsonArray dir;
        dir.append(kf.zDirX);
        dir.append(kf.zDirY);
        dir.append(kf.zDirZ);
        o["zOffsetWorldDir"] = dir;
        o["zSlice"] = kf.zSlice;
        o["dsScaleIdx"] = kf.dsScaleIdx;
        keyframes.append(o);
    }

    QJsonObject root;
    root["version"] = 1;
    root["header"] = header;
    root["keyframes"] = keyframes;

    QFile f(_outPath);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        Logger()->error("[vc3d-record] failed to open {} for writing",
                        _outPath.toStdString());
        return false;
    }
    f.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    f.close();
    Logger()->info("[vc3d-record] wrote {} keyframes to {}",
                   _keyframes.size(), _outPath.toStdString());
    return true;
}
