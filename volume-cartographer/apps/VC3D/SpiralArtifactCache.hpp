#pragma once

#include <QJsonObject>
#include <QObject>
#include <QString>
#include <QUrl>

#include <functional>
#include <memory>

class QNetworkAccessManager;
class QNetworkRequest;

// Downloads immutable service artifacts (previews, geometry snapshots,
// checkpoints) into a local cache and publishes each completed artifact
// directory atomically. The same path is used for loopback and remote
// services so local use exercises the network implementation.
//
// Layout: <CacheLocation>/spiral/<endpoint-fingerprint>/<session-id>/<artifact-id>/
class SpiralArtifactCache : public QObject
{
    Q_OBJECT
public:
    using RequestFactory = std::function<QNetworkRequest(const QString& path, int timeoutMs)>;
    using FetchCallback = std::function<void(const QString& localEntryPath, const QString& error, bool gone)>;

    explicit SpiralArtifactCache(QObject* parent = nullptr);

    // The fingerprint identifies the endpoint on disk and must not contain the
    // API key; requests are built by the owning service manager so artifact
    // transfers share its authentication and base URL.
    void setEndpoint(const QString& fingerprint, QNetworkAccessManager* network,
                     RequestFactory requestFactory);
    void clearEndpoint();

    // Fetch (or reuse) an artifact; calls back with the local path of its
    // entry-point file. `gone` reports 410: skip to the newest artifact.
    void fetchArtifact(const QString& sessionId, const QString& artifactId,
                       FetchCallback done);

    // Remove old artifact directories of one session, keeping `keep` newest
    // and never removing paths in `pinned` (currently displayed data).
    void pruneSession(const QString& sessionId, int keep, const QStringList& pinned);

    QString cacheRoot() const;

private:
    struct FetchJob;
    void startNextFile(const std::shared_ptr<FetchJob>& job);
    void finishJob(const std::shared_ptr<FetchJob>& job, const QString& error, bool gone = false);
    static bool validateManifest(const QJsonObject& manifest, QString* error);

    QString _fingerprint;
    QNetworkAccessManager* _network = nullptr;
    RequestFactory _requestFactory;
    quint64 _generation = 0; // invalidates in-flight jobs when the endpoint changes
};
