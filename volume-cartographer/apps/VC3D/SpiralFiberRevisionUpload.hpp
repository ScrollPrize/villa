#pragma once

#include <QFile>
#include <QFileInfo>
#include <cstdint>
#include <QJsonObject>
#include <QString>

namespace vc3d {

struct SpiralTrackedFiber {
    QString path;
    QString revision;
    QString snapshotPath;
    uint64_t latestGeneration = 0;
    uint64_t sentGeneration = 0;
    uint64_t inFlightGeneration = 0;
    bool added = false;
    bool uploadInFlight = false;
    bool retryAfterReconnect = false;

    void abandonUpload()
    {
        retryAfterReconnect = retryAfterReconnect || uploadInFlight;
        uploadInFlight = false;
        inFlightGeneration = 0;
        if (!snapshotPath.isEmpty()) QFile::remove(snapshotPath);
        snapshotPath.clear();
    }

    bool needsUpload(bool synchronized) const
    {
        return synchronized && !uploadInFlight
            && (retryAfterReconnect
                || (added && latestGeneration > sentGeneration));
    }
};

inline QString spiralFiberConflictRevision(const QJsonObject& failure)
{
    return failure.value(QStringLiteral("current_revision")).toString();
}

inline bool spiralFiberUploadNeedsCasRetry(const QString& revision,
                                           const QString& error)
{
    return !error.isEmpty() && !revision.isEmpty();
}

// The Spiral input id of a fiber JSON file. The fitter identifies dataset
// fibers by file stem and the service commits an uploaded fiber to
// paths.fibers/<id>.json, so the stem is the id under which a re-upload
// replaces the fiber instead of adding a second copy.
inline QString spiralFiberInputId(const QString& fiberJsonPath)
{
    return QFileInfo(fiberJsonPath).completeBaseName();
}

}  // namespace vc3d
