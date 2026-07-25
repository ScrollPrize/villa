#include "SpiralArtifactCache.hpp"

#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QJsonArray>
#include <QJsonDocument>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QRegularExpression>
#include <QStandardPaths>
#include <QtConcurrent/QtConcurrent>

namespace {
constexpr int kManifestTimeoutMs = 30000;
constexpr int kFileTimeoutMs = 60 * 60 * 1000; // bounded by transfer, effectively
constexpr qint64 kMaxDeclaredTotalBytes = 64LL * 1024 * 1024 * 1024;
constexpr int kMaxFiles = 4096;
const QString kManifestCopyName = QStringLiteral(".artifact-manifest.json");

bool isSafeRelativeName(const QString& name)
{
    if (name.isEmpty() || name.size() > 1024 || name.startsWith(QLatin1Char('/'))
        || name.contains(QLatin1Char('\\')))
        return false;
    const QStringList parts = name.split(QLatin1Char('/'));
    if (parts.size() > 8) return false;
    for (const QString& part : parts) {
        if (part.isEmpty() || part == QStringLiteral(".") || part == QStringLiteral(".."))
            return false;
        if (part == kManifestCopyName) return false;
    }
    return true;
}

QString hashFileSha256(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return {};
    QCryptographicHash hash(QCryptographicHash::Sha256);
    if (!hash.addData(&file)) return {};
    return QString::fromLatin1(hash.result().toHex());
}
} // namespace

struct SpiralArtifactCache::FetchJob
{
    quint64 generation = 0;
    QString sessionId;
    QString artifactId;
    QJsonObject manifest;
    QString entryPoint;
    QList<QJsonObject> pendingFiles;
    QString partialDir;
    QString finalDir;
    FetchCallback done;
};

SpiralArtifactCache::SpiralArtifactCache(QObject* parent) : QObject(parent) {}

QString SpiralArtifactCache::cacheRoot() const
{
    return QDir(QStandardPaths::writableLocation(QStandardPaths::CacheLocation))
        .filePath(QStringLiteral("spiral"));
}

void SpiralArtifactCache::setEndpoint(const QString& fingerprint,
                                      QNetworkAccessManager* network,
                                      RequestFactory requestFactory)
{
    _fingerprint = fingerprint;
    _network = network;
    _requestFactory = std::move(requestFactory);
    ++_generation;
}

void SpiralArtifactCache::clearEndpoint()
{
    _network = nullptr;
    _requestFactory = {};
    ++_generation;
}

bool SpiralArtifactCache::validateManifest(const QJsonObject& manifest, QString* error)
{
    if (manifest.value(QStringLiteral("schema_version")).toInt() != 1) {
        *error = QObject::tr("Unsupported artifact manifest schema");
        return false;
    }
    const QJsonArray files = manifest.value(QStringLiteral("files")).toArray();
    if (files.isEmpty() || files.size() > kMaxFiles) {
        *error = QObject::tr("Artifact manifest has an unreasonable file count");
        return false;
    }
    qint64 total = 0;
    for (const QJsonValue& value : files) {
        const QJsonObject entry = value.toObject();
        const QString name = entry.value(QStringLiteral("name")).toString();
        const qint64 size = entry.value(QStringLiteral("size")).toInteger(-1);
        const QString sha = entry.value(QStringLiteral("sha256")).toString();
        if (!isSafeRelativeName(name) || size < 0 || sha.size() != 64) {
            *error = QObject::tr("Artifact manifest contains an unsafe or malformed entry (%1)").arg(name);
            return false;
        }
        total += size;
    }
    if (total > kMaxDeclaredTotalBytes) {
        *error = QObject::tr("Artifact manifest declares an unreasonable total size");
        return false;
    }
    const QString entryPoint = manifest.value(QStringLiteral("entry_point")).toString();
    if (!isSafeRelativeName(entryPoint)) {
        *error = QObject::tr("Artifact manifest has an unsafe entry point");
        return false;
    }
    return true;
}

void SpiralArtifactCache::fetchArtifact(const QString& sessionId, const QString& artifactId,
                                        FetchCallback done)
{
    if (!_network || !_requestFactory) {
        done({}, tr("No Spiral service connection"), false);
        return;
    }
    static const QRegularExpression safeId(QStringLiteral("^[A-Za-z0-9._-]+$"));
    if (!safeId.match(artifactId).hasMatch() || !safeId.match(sessionId).hasMatch()) {
        done({}, tr("Malformed artifact reference"), false);
        return;
    }
    auto job = std::make_shared<FetchJob>();
    job->generation = _generation;
    job->sessionId = sessionId;
    job->artifactId = artifactId;
    job->finalDir = QDir(cacheRoot()).filePath(_fingerprint + QLatin1Char('/')
                                               + sessionId + QLatin1Char('/') + artifactId);
    job->partialDir = job->finalDir + QStringLiteral(".partial");
    job->done = std::move(done);

    // A published cache directory is immutable and complete: reuse it.
    const QString publishedManifest = QDir(job->finalDir).filePath(kManifestCopyName);
    if (QFileInfo::exists(publishedManifest)) {
        QFile file(publishedManifest);
        if (file.open(QIODevice::ReadOnly)) {
            const QJsonObject manifest = QJsonDocument::fromJson(file.readAll()).object();
            const QString entry = manifest.value(QStringLiteral("entry_point")).toString();
            if (isSafeRelativeName(entry)
                && QFileInfo::exists(QDir(job->finalDir).filePath(entry))) {
                job->done(QDir(job->finalDir).filePath(entry), {}, false);
                return;
            }
        }
        QDir(job->finalDir).removeRecursively();
    }

    QNetworkRequest request = _requestFactory(
        QStringLiteral("/artifacts/%1/manifest").arg(artifactId), kManifestTimeoutMs);
    auto* reply = _network->get(request);
    connect(reply, &QNetworkReply::finished, this, [this, reply, job]() {
        const int http = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        const QByteArray bytes = reply->readAll();
        const QString networkError =
            reply->error() == QNetworkReply::NoError ? QString() : reply->errorString();
        reply->deleteLater();
        if (job->generation != _generation) return;
        if (http == 410) { finishJob(job, tr("Artifact was pruned by the service"), true); return; }
        const QJsonObject manifest = QJsonDocument::fromJson(bytes).object();
        if (http != 200 || manifest.isEmpty()) {
            finishJob(job, tr("Could not fetch artifact manifest: %1")
                               .arg(!networkError.isEmpty()
                                        ? networkError
                                        : manifest.value(QStringLiteral("error"))
                                              .toString(tr("HTTP %1").arg(http))));
            return;
        }
        QString error;
        if (!validateManifest(manifest, &error)) { finishJob(job, error); return; }
        job->manifest = manifest;
        job->entryPoint = manifest.value(QStringLiteral("entry_point")).toString();
        for (const QJsonValue& value : manifest.value(QStringLiteral("files")).toArray())
            job->pendingFiles.push_back(value.toObject());
        if (!QDir().mkpath(job->partialDir)) {
            finishJob(job, tr("Could not create the artifact cache directory"));
            return;
        }
        startNextFile(job);
    });
}

void SpiralArtifactCache::startNextFile(const std::shared_ptr<FetchJob>& job)
{
    if (job->generation != _generation) return;
    if (job->pendingFiles.isEmpty()) {
        // Publish atomically: write the manifest copy, then rename the
        // completed directory into place.
        QFile manifestFile(QDir(job->partialDir).filePath(kManifestCopyName));
        if (!manifestFile.open(QIODevice::WriteOnly)
            || manifestFile.write(QJsonDocument(job->manifest).toJson(QJsonDocument::Compact)) < 0) {
            finishJob(job, tr("Could not write the artifact manifest copy"));
            return;
        }
        manifestFile.close();
        QDir().mkpath(QFileInfo(job->finalDir).absolutePath());
        QDir dir;
        if (QFileInfo::exists(job->finalDir)) QDir(job->finalDir).removeRecursively();
        if (!dir.rename(job->partialDir, job->finalDir)) {
            finishJob(job, tr("Could not publish the artifact cache directory"));
            return;
        }
        job->done(QDir(job->finalDir).filePath(job->entryPoint), {}, false);
        return;
    }

    const QJsonObject entry = job->pendingFiles.takeFirst();
    const QString name = entry.value(QStringLiteral("name")).toString();
    const qint64 declaredSize = entry.value(QStringLiteral("size")).toInteger();
    const QString declaredSha = entry.value(QStringLiteral("sha256")).toString().toLower();

    const QString targetPath = QDir(job->partialDir).filePath(name);
    QDir().mkpath(QFileInfo(targetPath).absolutePath());

    // A complete file from an interrupted earlier attempt is reused after
    // verification; a shorter one resumes with a Range request.
    qint64 existing = 0;
    if (QFileInfo::exists(targetPath)) {
        existing = QFileInfo(targetPath).size();
        if (existing > declaredSize) {
            QFile::remove(targetPath);
            existing = 0;
        }
    }

    auto file = std::make_shared<QFile>(targetPath);
    if (!file->open(existing > 0 ? (QIODevice::WriteOnly | QIODevice::Append)
                                 : QIODevice::WriteOnly)) {
        finishJob(job, tr("Could not open cache file %1").arg(name));
        return;
    }

    QNetworkRequest request = _requestFactory(
        QStringLiteral("/artifacts/%1/files/%2").arg(job->artifactId, name), kFileTimeoutMs);
    if (existing > 0 && existing < declaredSize)
        request.setRawHeader("Range", QStringLiteral("bytes=%1-").arg(existing).toUtf8());
    QNetworkReply* reply = existing == declaredSize ? nullptr : _network->get(request);

    auto verifyAndContinue = [this, job, name, declaredSize, declaredSha, targetPath]() {
        auto* watcher = new QFutureWatcher<QString>(this);
        connect(watcher, &QFutureWatcher<QString>::finished, this, [this, watcher, job]() {
            const QString error = watcher->result();
            watcher->deleteLater();
            if (job->generation != _generation) return;
            if (!error.isEmpty()) { finishJob(job, error); return; }
            startNextFile(job);
        });
        watcher->setFuture(QtConcurrent::run([name, declaredSize, declaredSha, targetPath]() -> QString {
            const qint64 actualSize = QFileInfo(targetPath).size();
            if (actualSize != declaredSize)
                return tr("Artifact file %1 has %2 bytes; the manifest declares %3")
                    .arg(name).arg(actualSize).arg(declaredSize);
            const QString actualSha = hashFileSha256(targetPath);
            if (actualSha != declaredSha)
                return tr("Artifact file %1 failed its SHA-256 digest check").arg(name);
            return {};
        }));
    };

    if (!reply) { verifyAndContinue(); return; }

    connect(reply, &QNetworkReply::readyRead, this, [reply, file]() {
        file->write(reply->readAll());
    });
    connect(reply, &QNetworkReply::finished, this,
            [this, reply, file, job, name, verifyAndContinue]() {
                file->write(reply->readAll());
                file->close();
                const int http = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
                const bool ok = reply->error() == QNetworkReply::NoError
                    && (http == 200 || http == 206);
                const QString errorText = reply->errorString();
                const bool gone = http == 410;
                reply->deleteLater();
                if (job->generation != _generation) return;
                if (gone) { finishJob(job, tr("Artifact was pruned by the service"), true); return; }
                if (!ok) {
                    finishJob(job, tr("Downloading artifact file %1 failed: %2").arg(name, errorText));
                    return;
                }
                verifyAndContinue();
            });
}

void SpiralArtifactCache::finishJob(const std::shared_ptr<FetchJob>& job,
                                    const QString& error, bool gone)
{
    // Keep the partial directory: verified complete files resume a later fetch.
    job->done({}, error, gone);
}

void SpiralArtifactCache::pruneSession(const QString& sessionId, int keep,
                                       const QStringList& pinned)
{
    if (_fingerprint.isEmpty()) return;
    const QDir sessionDir(QDir(cacheRoot()).filePath(_fingerprint + QLatin1Char('/') + sessionId));
    if (!sessionDir.exists()) return;
    QFileInfoList entries = sessionDir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot,
                                                     QDir::Time); // newest first
    int kept = 0;
    for (const QFileInfo& entry : entries) {
        const QString path = entry.absoluteFilePath();
        const bool isPartial = path.endsWith(QStringLiteral(".partial"));
        bool isPinned = false;
        for (const QString& pin : pinned)
            if (!pin.isEmpty() && (pin == path || pin.startsWith(path + QLatin1Char('/'))))
                isPinned = true;
        if (isPinned) continue;
        if (!isPartial && kept < keep) { ++kept; continue; }
        // Partial directories older than a day are abandoned resumes.
        if (isPartial && entry.lastModified().secsTo(QDateTime::currentDateTime()) < 86400)
            continue;
        QDir(path).removeRecursively();
    }
}
