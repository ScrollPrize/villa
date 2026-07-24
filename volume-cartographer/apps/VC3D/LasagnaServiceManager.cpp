#include "LasagnaServiceManager.hpp"

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QHostInfo>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QRegularExpression>
#include <QSet>
#include <QSysInfo>
#include <QTemporaryDir>
#include <QUrl>
#include <QtConcurrent/QtConcurrent>

#include <algorithm>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <vector>

#ifdef Q_OS_UNIX
#include <signal.h>
#include <unistd.h>
#endif

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// Avahi client library for mDNS service discovery
#if defined(Q_OS_LINUX) && defined(VC_HAVE_AVAHI) && VC_HAVE_AVAHI
#include <avahi-client/client.h>
#include <avahi-client/lookup.h>
#include <avahi-common/error.h>
#include <avahi-common/malloc.h>
#include <avahi-common/simple-watch.h>
#endif

namespace
{
constexpr int kServiceStartTimeoutMs = 60000;  // 1 minute (no torch compile)
constexpr int kServiceStopTimeoutMs = 500;
constexpr int kPollIntervalMs = 500;
constexpr const char* kFitServiceApiVersion = "2";
constexpr const char* kFitServiceApiVersionHeader = "X-Fit-Service-API-Version";
constexpr const char* kVc3dSourceHeader = "X-VC3D-Source";

double bytesToMiB(qint64 bytes)
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

double elapsedSeconds(const QElapsedTimer& timer)
{
    const qint64 ms = timer.elapsed();
    return ms > 0 ? static_cast<double>(ms) / 1000.0 : 0.001;
}

void logTransferTiming(const char* label, qint64 bytes, double seconds)
{
    const double mib = bytesToMiB(bytes);
    const double rate = seconds > 0.0 ? mib / seconds : 0.0;
    std::cout << "[lasagna] " << label
              << ": " << mib << " MiB"
              << " in " << seconds << "s"
              << " (" << rate << " MiB/s)" << std::endl;
}

QString stripTifxyzSuffix(const QString& name)
{
    return name.endsWith(QStringLiteral(".tifxyz"))
        ? name.left(name.size() - 7)
        : name;
}

QString stripVersionSuffix(const QString& base)
{
    const int pos = base.lastIndexOf(QStringLiteral("_v"));
    if (pos < 0 || pos + 2 >= base.size()) {
        return base;
    }
    for (int i = pos + 2; i < base.size(); ++i) {
        if (!base[i].isDigit()) {
            return base;
        }
    }
    return base.left(pos);
}

QString objectRefKey(const QJsonObject& ref)
{
    return ref[QStringLiteral("type")].toString() + QStringLiteral("\n")
        + ref[QStringLiteral("name")].toString() + QStringLiteral("\n")
        + ref[QStringLiteral("hash")].toString();
}

QString md5Ref(const QByteArray& bytes)
{
    return QStringLiteral("md5:%1").arg(QString::fromLatin1(
        QCryptographicHash::hash(bytes, QCryptographicHash::Md5).toHex()));
}

QJsonObject materializeLocalArtifactUpload(const QJsonObject& upload)
{
    if (upload.contains(QStringLiteral("data")) || upload.contains(QStringLiteral("files"))) {
        return upload;
    }
    const QJsonObject ref = upload[QStringLiteral("object")].toObject();
    const QString payload = upload[QStringLiteral("_local_payload")].toString();
    const QString localPath = upload[QStringLiteral("_local_path")].toString();
    if (ref.isEmpty() || payload.isEmpty() || localPath.isEmpty()) {
        return upload;
    }

    QJsonObject out;
    out[QStringLiteral("object")] = ref;
    if (payload == QStringLiteral("file")) {
        QFile f(localPath);
        if (!f.open(QIODevice::ReadOnly)) {
            throw std::runtime_error(QStringLiteral("Cannot read artifact file: %1").arg(localPath).toStdString());
        }
        const QByteArray bytes = f.readAll();
        const QString actualHash = md5Ref(bytes);
        if (actualHash != ref[QStringLiteral("hash")].toString()) {
            throw std::runtime_error(QStringLiteral("Artifact file hash mismatch for %1: declared %2 actual %3")
                                         .arg(ref[QStringLiteral("name")].toString(), ref[QStringLiteral("hash")].toString(), actualHash)
                                         .toStdString());
        }
        out[QStringLiteral("data")] = QString::fromLatin1(bytes.toBase64());
        return out;
    }

    if (payload == QStringLiteral("directory")) {
        namespace fs = std::filesystem;
        const fs::path root(localPath.toStdString());
        std::vector<fs::path> files;
        std::error_code ec;
        for (const auto& entry : fs::recursive_directory_iterator(root, ec)) {
            if (entry.is_regular_file()) {
                files.push_back(entry.path());
            }
        }
        if (ec) {
            throw std::runtime_error(QStringLiteral("Cannot scan artifact directory: %1").arg(localPath).toStdString());
        }
        std::sort(files.begin(), files.end());

        QByteArray manifest;
        QJsonObject encodedFiles;
        for (const auto& path : files) {
            QFile f(QString::fromStdString(path.string()));
            if (!f.open(QIODevice::ReadOnly)) {
                throw std::runtime_error(QStringLiteral("Cannot read artifact file: %1")
                                             .arg(QString::fromStdString(path.string()))
                                             .toStdString());
            }
            const QByteArray bytes = f.readAll();
            const auto relPath = fs::relative(path, root, ec);
            if (ec) {
                throw std::runtime_error(QStringLiteral("Cannot compute artifact relative path: %1")
                                             .arg(QString::fromStdString(path.string()))
                                             .toStdString());
            }
            const QString rel = QString::fromStdString(relPath.generic_string());
            manifest.append(rel.toUtf8());
            manifest.append('\t');
            manifest.append(md5Ref(bytes).toUtf8());
            manifest.append('\n');
            encodedFiles[rel] = QString::fromLatin1(bytes.toBase64());
        }
        const QString actualHash = md5Ref(manifest);
        if (actualHash != ref[QStringLiteral("hash")].toString()) {
            throw std::runtime_error(QStringLiteral("Artifact directory hash mismatch for %1: declared %2 actual %3")
                                         .arg(ref[QStringLiteral("name")].toString(), ref[QStringLiteral("hash")].toString(), actualHash)
                                         .toStdString());
        }
        out[QStringLiteral("files")] = encodedFiles;
        return out;
    }

    throw std::runtime_error(QStringLiteral("Unknown local artifact payload kind: %1").arg(payload).toStdString());
}

void appendObjectRefIfNew(QJsonArray& refs, QSet<QString>& refKeys, const QJsonObject& ref)
{
    if (ref.isEmpty() ||
        !ref.contains(QStringLiteral("type")) ||
        !ref.contains(QStringLiteral("name")) ||
        !ref.contains(QStringLiteral("hash"))) {
        return;
    }
    const QString key = objectRefKey(ref);
    if (refKeys.contains(key)) {
        return;
    }
    refs.append(ref);
    refKeys.insert(key);
}

void collectObjectRefs(QJsonArray& refs, QSet<QString>& refKeys, const QJsonValue& value)
{
    if (value.isArray()) {
        for (const QJsonValue& child : value.toArray()) {
            collectObjectRefs(refs, refKeys, child);
        }
        return;
    }
    if (!value.isObject()) {
        return;
    }
    const QJsonObject obj = value.toObject();
    appendObjectRefIfNew(refs, refKeys, obj);
    for (const QJsonValue& child : obj) {
        collectObjectRefs(refs, refKeys, child);
    }
}

QString uniqueSegmentName(const QString& targetDir, const QString& requestedName)
{
    const QString suffix = QStringLiteral(".tifxyz");
    const QString requested = requestedName.endsWith(suffix)
        ? requestedName
        : requestedName + suffix;
    const QDir dir(targetDir);
    if (!dir.exists(requested)) {
        return requested;
    }

    const QString root = stripVersionSuffix(stripTifxyzSuffix(requested));
    const QString prefix = root + QStringLiteral("_v");
    int maxVersion = 0;
    const QFileInfoList entries = dir.entryInfoList(QStringList{QStringLiteral("*.tifxyz")},
                                                    QDir::Dirs | QDir::NoDotAndDotDot);
    for (const QFileInfo& entry : entries) {
        const QString name = entry.fileName();
        if (!name.startsWith(prefix) || !name.endsWith(suffix)) {
            continue;
        }
        const QString digits = name.mid(prefix.size(), name.size() - prefix.size() - suffix.size());
        if (digits.isEmpty()) {
            continue;
        }
        bool ok = false;
        const int version = digits.toInt(&ok);
        if (ok) {
            maxVersion = std::max(maxVersion, version);
        }
    }

    QString candidate;
    int version = maxVersion + 1;
    do {
        candidate = QStringLiteral("%1_v%2%3")
            .arg(root)
            .arg(version++, 3, 10, QLatin1Char('0'))
            .arg(suffix);
    } while (dir.exists(candidate));
    return candidate;
}

void updateTifxyzMetadataIdentity(const QString& tifxyzDir, const QString& name)
{
    QFile metaFile(QDir(tifxyzDir).filePath(QStringLiteral("meta.json")));
    if (!metaFile.open(QIODevice::ReadOnly)) {
        return;
    }
    QJsonDocument doc = QJsonDocument::fromJson(metaFile.readAll());
    metaFile.close();
    if (!doc.isObject()) {
        return;
    }
    QJsonObject root = doc.object();
    root[QStringLiteral("uuid")] = name;
    root[QStringLiteral("name")] = name;
    if (!metaFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        return;
    }
    metaFile.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
}

QNetworkRequest fitServiceRequest(const QUrl& url)
{
    QNetworkRequest req(url);
    req.setRawHeader(kFitServiceApiVersionHeader, kFitServiceApiVersion);
    return req;
}

bool isTransportError(const QNetworkReply* reply)
{
    return reply->error() != QNetworkReply::NoError
        && !reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).isValid();
}

struct ResultsPlacementResult
{
    bool ok{false};
    QString error;
    QString targetDir;
    QStringList placedNames;
    QStringList warnings;
};

ResultsPlacementResult placeResultsArchive(const QByteArray& data,
                                           const QString& targetDir,
                                           const QString& expectedOutputName = QString())
{
    ResultsPlacementResult result;
    result.targetDir = targetDir;

    QDir().mkpath(targetDir);
    QTemporaryDir unpackDir(QDir(targetDir).filePath(QStringLiteral(".lasagna_unpack_XXXXXX")));
    if (!unpackDir.isValid()) {
        result.error = QObject::tr("Cannot create temporary unpack directory in %1").arg(targetDir);
        return result;
    }

    QString tarPath = QDir(unpackDir.path()).filePath(QStringLiteral(".lasagna_results.tar.gz"));
    QFile tarFile(tarPath);
    if (!tarFile.open(QIODevice::WriteOnly)) {
        result.error = QObject::tr("Cannot write temp file: %1").arg(tarPath);
        return result;
    }
    tarFile.write(data);
    tarFile.close();

    QProcess tar;
    tar.setWorkingDirectory(unpackDir.path());
    tar.start(QStringLiteral("tar"), {QStringLiteral("xzf"), tarPath});
    if (!tar.waitForFinished(30000)) {
        QFile::remove(tarPath);
        result.error = QObject::tr("tar extraction timed out");
        return result;
    }
    QFile::remove(tarPath);

    if (tar.exitCode() != 0) {
        QString err = QString::fromUtf8(tar.readAllStandardError());
        result.error = QObject::tr("tar extraction failed: %1").arg(err);
        return result;
    }

    QDir unpackRoot(unpackDir.path());
    const QFileInfoList children = unpackRoot.entryInfoList(
        QDir::Dirs | QDir::Files | QDir::NoDotAndDotDot);
    const QString expectedName = expectedOutputName.trimmed();
    const bool useExpectedName = !expectedName.isEmpty() && children.size() == 1;
    for (const QFileInfo& child : children) {
        if (child.fileName() == QStringLiteral(".lasagna_results.tar.gz")) {
            continue;
        }
        const QString finalName = uniqueSegmentName(
            targetDir,
            useExpectedName ? expectedName : child.fileName());
        const QString finalPath = QDir(targetDir).filePath(finalName);
        if (QFileInfo::exists(finalPath)) {
            result.error = QObject::tr("Refusing to overwrite existing segment: %1").arg(finalPath);
            return result;
        }
        if (child.isDir() && finalName.endsWith(QStringLiteral(".tifxyz"))) {
            updateTifxyzMetadataIdentity(child.absoluteFilePath(), finalName);
            if (finalName != child.fileName()) {
                result.warnings << QObject::tr(
                    "Lasagna result name collision: requested %1, placed as %2.")
                    .arg(child.fileName(), finalName);
            }
        }
        if (!QDir().rename(child.absoluteFilePath(), finalPath)) {
            result.error = QObject::tr("Cannot place downloaded result: %1").arg(finalPath);
            return result;
        }
        result.placedNames << finalName;
    }

    result.ok = true;
    return result;
}

QString findPythonExecutable()
{
    QStringList candidates;

    QString envPython = qEnvironmentVariable("PYTHON_EXECUTABLE");
    if (!envPython.isEmpty()) {
        candidates.append(envPython);
    }

    QString condaPrefix = qEnvironmentVariable("CONDA_PREFIX");
    if (!condaPrefix.isEmpty()) {
        candidates.append(QDir(condaPrefix).filePath("bin/python"));
        candidates.append(QDir(condaPrefix).filePath("bin/python3"));
    }

    QString home = QDir::homePath();
    candidates.append(QDir(home).filePath("miniconda3/bin/python"));
    candidates.append(QDir(home).filePath("miniconda3/bin/python3"));
    candidates.append(QDir(home).filePath("anaconda3/bin/python"));
    candidates.append(QDir(home).filePath("anaconda3/bin/python3"));

    candidates.append("python3");
    candidates.append("python");
    candidates.append("/usr/bin/python3");
    candidates.append("/usr/local/bin/python3");

    for (const QString& candidate : candidates) {
        QProcess test;
        test.start(candidate, {"--version"});
        if (test.waitForFinished(1000) && test.exitCode() == 0) {
            return candidate;
        }
    }
    return "python3";
}

QString findLasagnaServiceScript()
{
    QString appDir = QCoreApplication::applicationDirPath();
    QStringList searchPaths = {
        // Current monorepo layout: build/bin -> villa/lasagna.
        QDir(appDir).filePath("../../../lasagna/fit_service.py"),
        QDir::home().filePath("villa/lasagna/fit_service.py"),
        // Development: build dir is volume-cartographer/build/bin/
        QDir(appDir).filePath("../../vesuvius/src/vesuvius/exps_2d_model/fit_service.py"),
        QDir(appDir).filePath("../../../vesuvius/src/vesuvius/exps_2d_model/fit_service.py"),
        // Installed
        QDir(appDir).filePath("../share/vesuvius/exps_2d_model/fit_service.py"),
        // Environment variable
        qEnvironmentVariable("LASAGNA_SERVICE_PATH"),
    };

    for (const QString& path : searchPaths) {
        if (!path.isEmpty() && QFile::exists(path)) {
            return QFileInfo(path).absoluteFilePath();
        }
    }
    return {};
}
}  // namespace

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------

LasagnaServiceManager& LasagnaServiceManager::instance()
{
    static LasagnaServiceManager inst;
    return inst;
}

LasagnaServiceManager* LasagnaServiceManager::createTransient(QObject* parent)
{
    return new LasagnaServiceManager(parent, true);
}

LasagnaServiceManager::LasagnaServiceManager(QObject* parent,
                                             bool containProcessTree)
    : QObject(parent)
    , _containProcessTree(containProcessTree)
{
    _nam = new QNetworkAccessManager(this);
    _pollTimer = new QTimer(this);
    _pollTimer->setInterval(kPollIntervalMs);
    connect(_pollTimer, &QTimer::timeout, this, &LasagnaServiceManager::pollStatus);
}

LasagnaServiceManager::~LasagnaServiceManager()
{
    stopService();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

QString LasagnaServiceManager::baseUrl() const
{
    return QStringLiteral("http://%1:%2").arg(_host).arg(_port);
}

QString LasagnaServiceManager::localSourceName() const
{
    QString user = qEnvironmentVariable("USER");
    if (user.isEmpty()) {
        user = qEnvironmentVariable("USERNAME");
    }
    if (user.isEmpty()) {
        user = QStringLiteral("vc3d");
    }
    QString host = qEnvironmentVariable("HOSTNAME").trimmed();
    if (host.isEmpty() || host == QStringLiteral("localhost")) {
        host = QSysInfo::machineHostName().trimmed();
    }
    if (host.isEmpty() || host == QStringLiteral("localhost")) {
        host = QHostInfo::localHostName().trimmed();
    }
    return host.isEmpty() ? user : QStringLiteral("%1@%2").arg(user, host);
}

// ---------------------------------------------------------------------------
// Service lifecycle
// ---------------------------------------------------------------------------

bool LasagnaServiceManager::ensureServiceRunning(const QString& pythonPath,
                                                 const QString& dataDirectory)
{
    if (_isExternal && _serviceReady) {
        return true;
    }
    const QString requestedDataDirectory =
        dataDirectory.trimmed().isEmpty()
            ? QString{}
            : QFileInfo(dataDirectory).absoluteFilePath();
    if (_process && _process->state() == QProcess::Running && _serviceReady) {
        if (requestedDataDirectory.isEmpty() ||
            requestedDataDirectory == _dataDirectory) {
            return true;
        }
        // The fit service discovers datasets only at startup. Switch its data
        // directory when the active project/volume changes.
        stopService();
    }
    return startService(pythonPath, requestedDataDirectory);
}

void LasagnaServiceManager::connectToExternal(const QString& host, int port)
{
    // Stop any existing internal service first
    if (_process) {
        stopService();
    }

    ++_requestGeneration;
    clearLocalUploadJobs();
    _isExternal = true;
    _host = host;
    _port = port;
    _lastError.clear();
    _serviceReady = false;
    _lastQueueGeneration = -1;
    _fetchedQueueGeneration = -1;
    _statusRequestInFlight = false;
    _jobsRequestInFlight = false;
    _jobsRequestPending = false;

    emit statusMessage(tr("Connecting to external service at %1:%2...").arg(host).arg(port));

    // Ping GET /health
    QUrl url(QStringLiteral("%1/health").arg(baseUrl()));
    QNetworkRequest req = fitServiceRequest(url);

    const quint64 generation = _requestGeneration;
    QNetworkReply* reply = _nam->get(req);
    connect(reply, &QNetworkReply::finished, this, [this, reply, generation]() {
        if (generation != _requestGeneration) {
            reply->deleteLater();
            return;
        }
        reply->deleteLater();

        if (isTransportError(reply)) {
            _lastError = tr("Cannot reach external service: %1").arg(reply->errorString());
            _serviceReady = false;
            _isExternal = false;
            emit serviceError(_lastError);
            return;
        }
        if (!validateApiVersion(reply, tr("Service health check"))) {
            return;
        }
        if (reply->error() != QNetworkReply::NoError) {
            _lastError = tr("External service health check failed: %1").arg(reply->errorString());
            _serviceReady = false;
            _isExternal = false;
            emit serviceError(_lastError);
            return;
        }

        _serviceReady = true;
        _pollTimer->start();
        emit statusMessage(tr("Connected to external service on %1:%2").arg(_host).arg(_port));
        emit serviceStarted();
        fetchJobs();
    });
}

bool LasagnaServiceManager::startService(const QString& pythonPath,
                                         const QString& dataDirectory)
{
    ++_requestGeneration;
    clearLocalUploadJobs();
    _lastError.clear();
    _serviceReady = false;
    _port = 0;
    _lastQueueGeneration = -1;
    _fetchedQueueGeneration = -1;
    _statusRequestInFlight = false;
    _jobsRequestInFlight = false;
    _jobsRequestPending = false;

    QString scriptPath = findLasagnaServiceScript();
    if (scriptPath.isEmpty()) {
        _lastError = tr("Could not find fit_service.py. Set LASAGNA_SERVICE_PATH environment variable.");
        emit serviceError(_lastError);
        return false;
    }

    const QFileInfo dataDirectoryInfo(dataDirectory);
    const QDir dataDir(dataDirectoryInfo.absoluteFilePath());
    if (dataDirectory.trimmed().isEmpty() || !dataDirectoryInfo.isDir() ||
        dataDir.entryList(QStringList{QStringLiteral("*.lasagna.json")},
                          QDir::Files | QDir::Readable | QDir::NoDotAndDotDot)
            .isEmpty()) {
        _lastError = dataDirectory.trimmed().isEmpty()
            ? tr("No Lasagna data directory was provided")
            : tr("Lasagna data directory contains no .lasagna.json datasets: %1")
                  .arg(dataDirectory);
        emit serviceError(_lastError);
        return false;
    }
    _dataDirectory = dataDirectoryInfo.absoluteFilePath();

    _process = std::make_unique<QProcess>();
    _process->setProcessChannelMode(QProcess::SeparateChannels);

#ifdef Q_OS_UNIX
    if (_containProcessTree) {
        // Give this service a private process group. Its Python workers inherit
        // the group, allowing the transient Spiral workflow to reap the whole
        // tree even if the HTTP service has to be killed.
        _process->setChildProcessModifier([] {
            ::setpgid(0, 0);
        });
    }
#endif

#ifdef Q_OS_WIN
    if (_containProcessTree) {
        auto* job = CreateJobObjectW(nullptr, nullptr);
        if (job) {
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
            limits.BasicLimitInformation.LimitFlags =
                JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if (SetInformationJobObject(
                    job, JobObjectExtendedLimitInformation, &limits,
                    sizeof(limits))) {
                _processJob = job;
            } else {
                CloseHandle(job);
            }
        }
    }
#endif

    connect(_process.get(), QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &LasagnaServiceManager::handleProcessFinished);
    connect(_process.get(), &QProcess::errorOccurred,
            this, &LasagnaServiceManager::handleProcessError);
    connect(_process.get(), &QProcess::readyReadStandardOutput,
            this, &LasagnaServiceManager::handleReadyReadStdout);
    connect(_process.get(), &QProcess::readyReadStandardError,
            this, &LasagnaServiceManager::handleReadyReadStderr);

    // Set PYTHONPATH so lasagna_service.py can import sibling modules (fit, optimizer, etc.)
    QDir scriptDir(QFileInfo(scriptPath).absolutePath());
    QString exps2dPath = scriptDir.absolutePath();

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    QString existing = env.value("PYTHONPATH");
    if (existing.isEmpty()) {
        env.insert("PYTHONPATH", exps2dPath);
    } else {
        env.insert("PYTHONPATH", exps2dPath + ":" + existing);
    }
    _process->setProcessEnvironment(env);

    QString python = pythonPath.isEmpty() ? findPythonExecutable() : pythonPath;

    // Port 0 = auto-select
    QStringList args = {
        scriptPath, QStringLiteral("--port"), QStringLiteral("0"),
        QStringLiteral("--data-dir"), _dataDirectory
    };

    emit statusMessage(tr("Starting lasagna service..."));
    std::cout << "Starting lasagna service: " << python.toStdString();
    for (const QString& arg : args) {
        std::cout << " " << arg.toStdString();
    }
    std::cout << std::endl;

    _process->start(python, args);

    if (!_process->waitForStarted(5000)) {
        _lastError = tr("Failed to start lasagna service process");
        emit serviceError(_lastError);
        _process.reset();
#ifdef Q_OS_WIN
        if (_processJob) {
            CloseHandle(static_cast<HANDLE>(_processJob));
            _processJob = nullptr;
        }
#endif
        _dataDirectory.clear();
        return false;
    }

#ifdef Q_OS_WIN
    if (_containProcessTree && _processJob) {
        HANDLE process = OpenProcess(
            PROCESS_SET_QUOTA | PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION,
            FALSE, static_cast<DWORD>(_process->processId()));
        if (!process ||
            !AssignProcessToJobObject(static_cast<HANDLE>(_processJob), process)) {
            CloseHandle(static_cast<HANDLE>(_processJob));
            _processJob = nullptr;
        }
        if (process) CloseHandle(process);
    }
#endif

    emit statusMessage(tr("Waiting for lasagna service to initialize..."));

    QElapsedTimer timer;
    timer.start();
    while (timer.elapsed() < kServiceStartTimeoutMs) {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 100);

        if (_serviceReady) {
            _pollTimer->start();
            emit statusMessage(tr("Lasagna service ready on port %1").arg(_port));
            emit serviceStarted();
            fetchJobs();
            return true;
        }

        if (!_process || _process->state() != QProcess::Running) {
            if (_lastError.isEmpty()) {
                _lastError = tr("Lasagna service terminated unexpectedly");
            }
            emit serviceError(_lastError);
            return false;
        }
    }

    _lastError = tr("Lasagna service startup timed out");
    emit serviceError(_lastError);
    stopService();
    return false;
}

void LasagnaServiceManager::stopService()
{
    ++_requestGeneration;
    _pollTimer->stop();
    _statusRequestInFlight = false;
    clearLocalUploadJobs();

    if (_isExternal) {
        // External mode: just reset state, don't terminate any process
        _serviceReady = false;
        _optimizationRunning = false;
        _isExternal = false;
        _dataDirectory.clear();
        _host = QStringLiteral("127.0.0.1");
        _port = 0;
        _activeJobId.clear();
        _submittedJobIds.clear();
        _startedJobIds.clear();
        _completedJobIds.clear();
        _jobOutputDirs.clear();
        _jobOutputNames.clear();
        _lastJobs = QJsonArray();
        _lastQueueGeneration = -1;
        _fetchedQueueGeneration = -1;
        _statusRequestInFlight = false;
        _jobsRequestInFlight = false;
        _jobsRequestPending = false;
        emit serviceStopped();
        return;
    }

    if (!_process) {
        return;
    }

    std::cout << "Stopping lasagna service..." << std::endl;

    if (_process->state() == QProcess::Running) {
#ifdef Q_OS_UNIX
        if (_containProcessTree && _process->processId() > 0) {
            const auto processGroup = static_cast<pid_t>(_process->processId());
            ::kill(-processGroup, SIGTERM);
            _process->waitForFinished(kServiceStopTimeoutMs);
            // The group signal is intentional even when the leader exited:
            // workers can outlive it and must not survive this transient job.
            ::kill(-processGroup, SIGKILL);
            if (_process->state() != QProcess::NotRunning) {
                _process->kill();
                _process->waitForFinished(1000);
            }
        } else
#endif
        {
            _process->terminate();
            if (!_process->waitForFinished(kServiceStopTimeoutMs)) {
                _process->kill();
                _process->waitForFinished(1000);
            }
        }
    }

#ifdef Q_OS_WIN
    if (_processJob) {
        // Graceful termination above gives the service a chance to flush. The
        // job is the backstop for any worker processes that remain.
        TerminateJobObject(static_cast<HANDLE>(_processJob), 1);
        CloseHandle(static_cast<HANDLE>(_processJob));
        _processJob = nullptr;
    }
#endif

    _process.reset();
    _serviceReady = false;
    _dataDirectory.clear();
    _port = 0;
    _optimizationRunning = false;
    _activeJobId.clear();
    _submittedJobIds.clear();
    _startedJobIds.clear();
    _completedJobIds.clear();
    _jobOutputDirs.clear();
    _jobOutputNames.clear();
    _lastJobs = QJsonArray();
    _lastQueueGeneration = -1;
    _fetchedQueueGeneration = -1;
    _statusRequestInFlight = false;
    _jobsRequestInFlight = false;
    _jobsRequestPending = false;

    emit serviceStopped();
}

bool LasagnaServiceManager::isRunning() const
{
    if (_isExternal) {
        return _serviceReady;
    }
    return _process && _process->state() == QProcess::Running && _serviceReady;
}

QString LasagnaServiceManager::findConfigFile(const QString& fileName)
{
    const QString serviceScript = findLasagnaServiceScript();
    if (serviceScript.isEmpty() || fileName.trimmed().isEmpty()) {
        return {};
    }
    const QString candidate = QDir(QFileInfo(serviceScript).absolutePath())
                                  .filePath(QStringLiteral("configs/%1").arg(fileName));
    return QFileInfo(candidate).isFile() ? QFileInfo(candidate).absoluteFilePath()
                                         : QString{};
}

QJsonObject LasagnaServiceManager::makeTifxyzArtifactUpload(
    const QString& tifxyzDirectory)
{
    namespace fs = std::filesystem;
    const fs::path root(tifxyzDirectory.toStdString());
    std::error_code ec;
    if (!fs::is_directory(root, ec)) {
        return {};
    }

    std::vector<fs::path> files;
    for (const auto& entry : fs::recursive_directory_iterator(root, ec)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path());
        }
    }
    if (ec || files.empty()) {
        return {};
    }
    std::sort(files.begin(), files.end());

    QByteArray manifest;
    for (const auto& path : files) {
        QFile file(QString::fromStdString(path.string()));
        if (!file.open(QIODevice::ReadOnly)) {
            return {};
        }
        const QByteArray bytes = file.readAll();
        const fs::path relative = fs::relative(path, root, ec);
        if (ec) {
            return {};
        }
        manifest.append(QString::fromStdString(relative.generic_string()).toUtf8());
        manifest.append('\t');
        manifest.append(md5Ref(bytes).toUtf8());
        manifest.append('\n');
    }

    QJsonObject ref;
    ref[QStringLiteral("type")] = QStringLiteral("tifxyz_segment");
    ref[QStringLiteral("name")] = QFileInfo(tifxyzDirectory).fileName();
    ref[QStringLiteral("hash")] = md5Ref(manifest);

    QJsonObject upload;
    upload[QStringLiteral("object")] = ref;
    upload[QStringLiteral("_local_payload")] = QStringLiteral("directory");
    upload[QStringLiteral("_local_path")] = QFileInfo(tifxyzDirectory).absoluteFilePath();
    return upload;
}

void LasagnaServiceManager::rankLaplaceSnapPairs(
    const QJsonObject& request,
    std::function<void(const QJsonObject&)> onSuccess,
    std::function<void(const QString&)> onError,
    std::function<void(int, const QJsonObject&)> onPartialResult)
{
    if (!isRunning()) {
        const QString msg = tr("Lasagna service is not running.");
        if (onError) {
            onError(msg);
        }
        return;
    }

    QUrl url(QStringLiteral("%1/jobs").arg(baseUrl()));
    QNetworkRequest req = fitServiceRequest(url);
    req.setRawHeader(kVc3dSourceHeader, localSourceName().toUtf8());
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    const quint64 generation = _requestGeneration;
    QJsonObject jobRequest = request;
    jobRequest[QStringLiteral("job_type")] = QStringLiteral("laplace_rank");
    jobRequest[QStringLiteral("config_name")] = QStringLiteral("Atlas snap rank");
    const QByteArray body = QJsonDocument(jobRequest).toJson(QJsonDocument::Compact);
    const int jobCount = request.value(QStringLiteral("jobs")).toArray().size();
    auto timer = std::make_shared<QElapsedTimer>();
    timer->start();
    std::cout << "[lasagna] queued laplace rank request:"
              << " jobs=" << jobCount
              << " bytes=" << body.size()
              << " url=" << url.toString().toStdString()
              << std::endl;
    QNetworkReply* reply = _nam->post(req, body);
    connect(reply, &QNetworkReply::finished, this,
            [this,
             reply,
             generation,
             timer,
             jobCount,
             onSuccess = std::move(onSuccess),
             onError = std::move(onError),
             onPartialResult = std::move(onPartialResult)]() mutable {
        if (generation != _requestGeneration) {
            reply->deleteLater();
            return;
        }
        const QByteArray bytes = reply->readAll();
        const QVariant status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute);
        std::cout << "[lasagna] queued laplace rank submit response:"
                  << " elapsed=" << elapsedSeconds(*timer) << "s";
        if (status.isValid()) {
            std::cout << " http=" << status.toInt();
        } else {
            std::cout << " http=<none>";
        }
        std::cout << " error=" << reply->error()
                  << " bytes=" << bytes.size()
                  << std::endl;
        reply->deleteLater();

        auto fail = [this, onError](const QString& msg) {
            _lastError = msg;
            std::cerr << "[lasagna] queued laplace rank error: "
                      << msg.toStdString() << std::endl;
            if (onError) {
                onError(msg);
            }
        };

        if (isTransportError(reply)) {
            fail(tr("Laplace rank submit failed: %1").arg(reply->errorString()));
            return;
        }
        if (!validateApiVersion(reply, tr("Laplace rank submit"))) {
            if (onError) {
                onError(_lastError);
            }
            return;
        }

        QJsonParseError parseError;
        const QJsonDocument doc = QJsonDocument::fromJson(bytes, &parseError);
        if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
            fail(tr("Laplace rank submit returned malformed JSON at byte %1: %2")
                     .arg(parseError.offset)
                     .arg(parseError.errorString()));
            return;
        }
        const QJsonObject obj = doc.object();
        if (reply->error() != QNetworkReply::NoError ||
            obj.contains(QStringLiteral("error"))) {
            const QString serviceError = obj[QStringLiteral("error")].toString(reply->errorString());
            const QString code = obj[QStringLiteral("code")].toString();
            fail(code.isEmpty()
                     ? tr("Laplace rank submit failed: %1").arg(serviceError)
                     : tr("Laplace rank submit failed (%1): %2").arg(code, serviceError));
            return;
        }
        const QString jobId = obj[QStringLiteral("job_id")].toString();
        if (jobId.isEmpty()) {
            fail(tr("Laplace rank submit returned no job id"));
            return;
        }

        _submittedJobIds.insert(jobId);
        _activeJobId = jobId;
        _optimizationRunning = true;
        _pollTimer->start();
        emit jobStarted(jobId);
        emit optimizationStarted();
        fetchJobs();

        struct RankPollState
        {
            QString jobId;
            int afterSeq{0};
            int expectedResults{0};
            int receivedResults{0};
            QJsonArray results;
            QSet<int> seenIndices;
            int finishedPollsWaitingForEvents{0};
            bool done{false};
        };
        auto state = std::make_shared<RankPollState>();
        state->jobId = jobId;
        state->expectedResults = jobCount;
        for (int i = 0; i < jobCount; ++i) {
            state->results.append(QJsonValue());
        }

        auto* pollTimer = new QTimer(this);
        pollTimer->setInterval(kPollIntervalMs);
        auto poll = std::make_shared<std::function<void()>>();
        auto finishWithError = std::make_shared<std::function<void(const QString&)>>();
        *finishWithError = [this, pollTimer, state, fail](const QString& msg) {
            if (state->done) {
                return;
            }
            state->done = true;
            pollTimer->stop();
            pollTimer->deleteLater();
            _completedJobIds.insert(state->jobId);
            fail(msg);
        };
        *poll = [this,
                 generation,
                 state,
                 pollTimer,
                 poll,
                 finishWithError,
                 onSuccess,
                 onPartialResult]() {
            if (state->done || generation != _requestGeneration) {
                return;
            }
            QUrl eventsUrl(QStringLiteral("%1/jobs/%2/events?after=%3")
                               .arg(baseUrl(), state->jobId)
                               .arg(state->afterSeq));
            QNetworkRequest eventsReq = fitServiceRequest(eventsUrl);
            QNetworkReply* eventsReply = _nam->get(eventsReq);
            connect(eventsReply, &QNetworkReply::finished, this,
                    [this,
                     eventsReply,
                     generation,
                     state,
                     pollTimer,
                     finishWithError,
                     onSuccess,
                     onPartialResult]() {
                if (state->done || generation != _requestGeneration) {
                    eventsReply->deleteLater();
                    return;
                }
                const QByteArray eventBytes = eventsReply->readAll();
                if (isTransportError(eventsReply) ||
                    !validateApiVersion(eventsReply, tr("Laplace rank events")) ||
                    eventsReply->error() != QNetworkReply::NoError) {
                    const QString msg = tr("Laplace rank event poll failed: %1")
                                            .arg(eventsReply->errorString());
                    eventsReply->deleteLater();
                    (*finishWithError)(msg);
                    return;
                }
                QJsonParseError eventParseError;
                const QJsonDocument eventDoc = QJsonDocument::fromJson(eventBytes, &eventParseError);
                eventsReply->deleteLater();
                if (eventParseError.error != QJsonParseError::NoError || !eventDoc.isObject()) {
                    (*finishWithError)(tr("Laplace rank events returned malformed JSON"));
                    return;
                }
                const QJsonObject eventRoot = eventDoc.object();
                const QJsonArray events = eventRoot[QStringLiteral("events")].toArray();
                for (const QJsonValue& value : events) {
                    const QJsonObject event = value.toObject();
                    state->afterSeq = std::max(state->afterSeq, event[QStringLiteral("seq")].toInt());
                    if (event[QStringLiteral("type")].toString() != QStringLiteral("laplace_rank_result")) {
                        continue;
                    }
                    const int index = event[QStringLiteral("index")].toInt(-1);
                    if (index < 0 || index >= state->expectedResults) {
                        continue;
                    }
                    if (!state->seenIndices.contains(index)) {
                        state->seenIndices.insert(index);
                        ++state->receivedResults;
                    }
                    const QJsonValue resultValue = event[QStringLiteral("result")];
                    state->results.replace(index, resultValue);
                    if (onPartialResult) {
                        onPartialResult(index, resultValue.toObject());
                    }
                }

                QUrl statusUrl(QStringLiteral("%1/jobs/%2").arg(baseUrl(), state->jobId));
                QNetworkRequest statusReq = fitServiceRequest(statusUrl);
                QNetworkReply* statusReply = _nam->get(statusReq);
                connect(statusReply, &QNetworkReply::finished, this,
                        [this, statusReply, generation, state, pollTimer, finishWithError, onSuccess]() {
                    if (state->done || generation != _requestGeneration) {
                        statusReply->deleteLater();
                        return;
                    }
                    const QByteArray statusBytes = statusReply->readAll();
                    if (isTransportError(statusReply) ||
                        !validateApiVersion(statusReply, tr("Laplace rank status")) ||
                        statusReply->error() != QNetworkReply::NoError) {
                        const QString msg = tr("Laplace rank status poll failed: %1")
                                                .arg(statusReply->errorString());
                        statusReply->deleteLater();
                        (*finishWithError)(msg);
                        return;
                    }
                    QJsonParseError statusParseError;
                    const QJsonDocument statusDoc = QJsonDocument::fromJson(statusBytes, &statusParseError);
                    statusReply->deleteLater();
                    if (statusParseError.error != QJsonParseError::NoError || !statusDoc.isObject()) {
                        (*finishWithError)(tr("Laplace rank status returned malformed JSON"));
                        return;
                    }
                    const QJsonObject statusObj = statusDoc.object();
                    updateCachedJobFromStatus(statusObj);
                    const QString stateText = statusObj[QStringLiteral("state")].toString();
                    if (stateText == QStringLiteral("finished")) {
                        if (state->receivedResults != state->expectedResults) {
                            ++state->finishedPollsWaitingForEvents;
                            std::cerr << "[lasagna] queued laplace rank finished but waiting for events:"
                                      << " received=" << state->receivedResults
                                      << " expected=" << state->expectedResults
                                      << " poll=" << state->finishedPollsWaitingForEvents
                                      << std::endl;
                            if (state->finishedPollsWaitingForEvents < 10) {
                                return;
                            }
                            (*finishWithError)(
                                tr("Laplace rank finished with %1/%2 result events after waiting")
                                    .arg(state->receivedResults)
                                    .arg(state->expectedResults));
                            return;
                        }
                        state->done = true;
                        pollTimer->stop();
                        pollTimer->deleteLater();
                        _completedJobIds.insert(state->jobId);
                        QJsonObject response;
                        response[QStringLiteral("results")] = state->results;
                        if (onSuccess) {
                            onSuccess(response);
                        }
                    } else if (stateText == QStringLiteral("error") ||
                               stateText == QStringLiteral("cancelled")) {
                        QString msg = statusObj[QStringLiteral("error")].toString();
                        if (msg.isEmpty()) {
                            msg = stateText == QStringLiteral("cancelled")
                                ? tr("Laplace rank cancelled")
                                : tr("Laplace rank failed");
                        }
                        (*finishWithError)(msg);
                    }
                });
            });
        };
        connect(pollTimer, &QTimer::timeout, this, [poll]() { (*poll)(); });
        (*poll)();
        pollTimer->start();
    });
}

bool LasagnaServiceManager::validateApiVersion(QNetworkReply* reply, const QString& context)
{
    const QByteArray got = reply->rawHeader(kFitServiceApiVersionHeader);
    if (got == QByteArray(kFitServiceApiVersion)) {
        return true;
    }

    const QString gotText = got.isEmpty()
        ? tr("<missing>")
        : QString::fromLatin1(got);
    const QString msg = tr("%1 failed: fit-service API version mismatch "
                           "(expected %2=%3, got %4)")
        .arg(context, QString::fromLatin1(kFitServiceApiVersionHeader),
             QString::fromLatin1(kFitServiceApiVersion), gotText);
    _lastError = msg;
    emit serviceError(msg);
    stopService();
    return false;
}

// ---------------------------------------------------------------------------
// Process I/O handlers
// ---------------------------------------------------------------------------

void LasagnaServiceManager::handleProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    ++_requestGeneration;
    std::cout << "Lasagna service finished with exit code " << exitCode << std::endl;

    if (exitStatus == QProcess::CrashExit) {
        _lastError = tr("Lasagna service crashed");
    } else if (exitCode != 0) {
        _lastError = tr("Lasagna service exited with code %1").arg(exitCode);
    }

    _serviceReady = false;
    _pollTimer->stop();
    _statusRequestInFlight = false;
    _jobsRequestInFlight = false;
    _jobsRequestPending = false;
    _activeJobId.clear();
    clearLocalUploadJobs();
    emit serviceStopped();
}

void LasagnaServiceManager::handleProcessError(QProcess::ProcessError error)
{
    QString errorStr;
    switch (error) {
    case QProcess::FailedToStart:
        errorStr = tr("Failed to start lasagna service");
        break;
    case QProcess::Crashed:
        errorStr = tr("Lasagna service crashed");
        break;
    default:
        errorStr = tr("Lasagna service error");
        break;
    }

    _lastError = errorStr;
    std::cerr << "[lasagna] service error: " << errorStr.toStdString() << std::endl;
    emit serviceError(errorStr);
}

void LasagnaServiceManager::handleReadyReadStdout()
{
    if (!_process) return;

    QString output = QString::fromUtf8(_process->readAllStandardOutput());
    std::cout << "[lasagna] " << output.toStdString();

    // Parse "listening on http://127.0.0.1:PORT"
    if (!_serviceReady) {
        static const QRegularExpression re(R"(listening on http://[\w.]+:(\d+))");
        auto match = re.match(output);
        if (match.hasMatch()) {
            _port = match.captured(1).toInt();
            _serviceReady = true;
        }
    }
}

void LasagnaServiceManager::handleReadyReadStderr()
{
    if (!_process) return;

    QString error = QString::fromUtf8(_process->readAllStandardError());
    std::cerr << "[lasagna] " << error.toStdString();

    if (!error.trimmed().isEmpty() && !_serviceReady) {
        if (_lastError.isEmpty() && error.contains("error", Qt::CaseInsensitive)) {
            _lastError = error.trimmed();
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP communication
// ---------------------------------------------------------------------------

void LasagnaServiceManager::upsertLocalUploadJob(const QJsonObject& job)
{
    const QString jobId = job[QStringLiteral("job_id")].toString();
    if (jobId.isEmpty()) {
        return;
    }
    if (!_localUploadJobs.contains(jobId)) {
        _localUploadOrder.append(jobId);
    }
    _localUploadJobs.insert(jobId, job);
    emitJobsUpdatedOverlay();
}

void LasagnaServiceManager::updateLocalUploadJob(const QString& jobId, const QJsonObject& updates)
{
    if (!_localUploadJobs.contains(jobId)) {
        return;
    }
    QJsonObject job = _localUploadJobs.value(jobId);
    for (auto it = updates.constBegin(); it != updates.constEnd(); ++it) {
        job[it.key()] = it.value();
    }
    _localUploadJobs.insert(jobId, job);
    emitJobsUpdatedOverlay();

    if (job[QStringLiteral("state")].toString() == QStringLiteral("upload")) {
        emit artifactUploadProgress(jobId,
                                    job[QStringLiteral("upload_current")].toInt(),
                                    job[QStringLiteral("upload_total")].toInt(),
                                    job[QStringLiteral("upload_progress")].toDouble(),
                                    job[QStringLiteral("upload_label")].toString());
    }
}

void LasagnaServiceManager::removeLocalUploadJob(const QString& jobId)
{
    if (!_localUploadJobs.remove(jobId)) {
        return;
    }
    _localUploadOrder.removeAll(jobId);
    _cancelledLocalUploadJobs.remove(jobId);
    emitJobsUpdatedOverlay();
}

void LasagnaServiceManager::updateCachedJobFromStatus(const QJsonObject& status)
{
    const QString state = status[QStringLiteral("state")].toString();
    if (state.isEmpty() || state == QStringLiteral("idle")) {
        return;
    }

    QString jobId = status[QStringLiteral("job_id")].toString();
    if (jobId.isEmpty()) {
        jobId = _activeJobId;
    }
    if (jobId.isEmpty()) {
        return;
    }

    bool changed = false;
    bool found = false;
    QJsonArray updated;
    for (const QJsonValue& value : _lastJobs) {
        QJsonObject job = value.toObject();
        if (job[QStringLiteral("job_id")].toString() == jobId) {
            found = true;
            for (auto it = status.constBegin(); it != status.constEnd(); ++it) {
                if (it.key() == QStringLiteral("api_version")
                    || it.key() == QStringLiteral("queue_generation")) {
                    continue;
                }
                if (job.value(it.key()) != it.value()) {
                    job[it.key()] = it.value();
                    changed = true;
                }
            }
        }
        updated.append(job);
    }

    if (!found && _submittedJobIds.contains(jobId)) {
        QJsonObject job = status;
        job.remove(QStringLiteral("api_version"));
        job.remove(QStringLiteral("queue_generation"));
        job[QStringLiteral("job_id")] = jobId;
        updated.append(job);
        changed = true;
    }

    if (!changed) {
        return;
    }
    _lastJobs = updated;
    emitJobsUpdatedOverlay();
}

QJsonArray LasagnaServiceManager::jobsWithLocalUploads(const QJsonArray& serviceJobs) const
{
    QJsonArray jobs;
    for (const QString& jobId : _localUploadOrder) {
        if (_localUploadJobs.contains(jobId)) {
            jobs.append(_localUploadJobs.value(jobId));
        }
    }
    for (const QJsonValue& value : serviceJobs) {
        jobs.append(value);
    }
    return jobs;
}

void LasagnaServiceManager::emitJobsUpdatedOverlay()
{
    emit jobsUpdated(jobsWithLocalUploads(_lastJobs));
}

void LasagnaServiceManager::clearLocalUploadJobs()
{
    if (_activeArtifactReply) {
        _activeArtifactReply->abort();
        _activeArtifactReply = nullptr;
    }
    _activeArtifactReplyJobId.clear();
    _artifactUploadQueue.clear();
    _hasActiveArtifactUpload = false;
    _activeArtifactUpload = ArtifactUploadJob{};
    _cancelledLocalUploadJobs.clear();
    _localUploadJobs.clear();
    _localUploadOrder.clear();
    emitJobsUpdatedOverlay();
}

void LasagnaServiceManager::finishActiveArtifactUpload()
{
    if (!_hasActiveArtifactUpload) {
        return;
    }
    _activeArtifactReply = nullptr;
    _activeArtifactReplyJobId.clear();
    _cancelledLocalUploadJobs.remove(_activeArtifactUpload.jobId);
    _activeArtifactUpload = ArtifactUploadJob{};
    _hasActiveArtifactUpload = false;
}

bool LasagnaServiceManager::cancelLocalUploadJob(const QString& jobId)
{
    if (!_localUploadJobs.contains(jobId)) {
        return false;
    }

    for (int i = 0; i < _artifactUploadQueue.size(); ++i) {
        if (_artifactUploadQueue[i].jobId == jobId) {
            _artifactUploadQueue.removeAt(i);
            updateLocalUploadJob(jobId, {
                {QStringLiteral("state"), QStringLiteral("cancelled")},
                {QStringLiteral("error"), tr("Artifact upload cancelled")},
            });
            return true;
        }
    }

    if (_hasActiveArtifactUpload && _activeArtifactUpload.jobId == jobId) {
        _cancelledLocalUploadJobs.insert(jobId);
        updateLocalUploadJob(jobId, {
            {QStringLiteral("state"), QStringLiteral("cancelled")},
            {QStringLiteral("error"), tr("Artifact upload cancelled")},
        });
        if (_activeArtifactReply) {
            _activeArtifactReply->abort();
        } else {
            finishActiveArtifactUpload();
            processNextArtifactUpload();
        }
        return true;
    }

    updateLocalUploadJob(jobId, {
        {QStringLiteral("state"), QStringLiteral("cancelled")},
        {QStringLiteral("error"), tr("Artifact upload cancelled")},
    });
    return true;
}

void LasagnaServiceManager::enqueueArtifactUpload(const QJsonObject& requestConfig,
                                                  const QJsonArray& objects,
                                                  const QString& localOutputDir,
                                                  const QString& source)
{
    const QString localJobId = QStringLiteral("vc3d-upload-%1").arg(_nextLocalUploadId++);
    QJsonObject localJob;
    localJob[QStringLiteral("job_id")] = localJobId;
    localJob[QStringLiteral("source")] = source;
    localJob[QStringLiteral("config_name")] = requestConfig[QStringLiteral("config_name")].toString();
    localJob[QStringLiteral("output_name")] = requestConfig[QStringLiteral("output_name")].toString();
    localJob[QStringLiteral("state")] = QStringLiteral("upload");
    localJob[QStringLiteral("upload_state")] = QStringLiteral("queued");
    localJob[QStringLiteral("upload_label")] = tr("Waiting to upload artifacts");
    localJob[QStringLiteral("upload_current")] = 0;
    localJob[QStringLiteral("upload_total")] = 0;
    localJob[QStringLiteral("upload_progress")] = 0.0;
    localJob[QStringLiteral("submitted_at")] = static_cast<double>(QDateTime::currentSecsSinceEpoch());
    upsertLocalUploadJob(localJob);

    _artifactUploadQueue.append(ArtifactUploadJob{
        localJobId,
        requestConfig,
        objects,
        localOutputDir,
        source,
    });
    emit statusMessage(tr("Queued Lasagna artifact upload for %1")
                           .arg(localJob[QStringLiteral("output_name")].toString()));
    processNextArtifactUpload();
}

void LasagnaServiceManager::processNextArtifactUpload()
{
    if (_hasActiveArtifactUpload || _artifactUploadQueue.isEmpty()) {
        return;
    }
    _activeArtifactUpload = _artifactUploadQueue.takeFirst();
    _hasActiveArtifactUpload = true;
    const QString localJobId = _activeArtifactUpload.jobId;
    const QJsonObject requestConfig = _activeArtifactUpload.requestConfig;
    const QJsonArray objects = _activeArtifactUpload.objects;
    const QString localOutputDir = _activeArtifactUpload.localOutputDir;
    const QString source = _activeArtifactUpload.source;

    updateLocalUploadJob(localJobId, {
        {QStringLiteral("upload_state"), QStringLiteral("checking")},
        {QStringLiteral("upload_label"), tr("Checking remote artifacts")},
        {QStringLiteral("upload_current"), 0},
        {QStringLiteral("upload_total"), 0},
        {QStringLiteral("upload_progress"), 0.0},
    });
    emit statusMessage(tr("Checking Lasagna artifacts..."));

    QJsonArray refs;
    QSet<QString> refKeys;
    const QJsonObject jobSpec = requestConfig[QStringLiteral("job_spec")].toObject();
    appendObjectRefIfNew(refs, refKeys, jobSpec[QStringLiteral("model")].toObject());
    appendObjectRefIfNew(refs, refKeys, jobSpec[QStringLiteral("atlas")].toObject());
    collectObjectRefs(refs, refKeys, jobSpec[QStringLiteral("object_refs")]);
    collectObjectRefs(refs, refKeys,
                      jobSpec[QStringLiteral("config")].toObject()[QStringLiteral("atlas")]);
    for (const QJsonValue& value : jobSpec[QStringLiteral("linked_surfaces")].toArray()) {
        appendObjectRefIfNew(refs, refKeys, value.toObject());
    }
    for (const QJsonValue& value : objects) {
        const QJsonObject upload = value.toObject();
        appendObjectRefIfNew(refs, refKeys, upload[QStringLiteral("object")].toObject());
    }

    QJsonObject queryBody;
    queryBody[QStringLiteral("objects")] = refs;
    QUrl queryUrl(QStringLiteral("%1/objects/query").arg(baseUrl()));
    QNetworkRequest queryReq = fitServiceRequest(queryUrl);
    queryReq.setRawHeader(kVc3dSourceHeader, source.toUtf8());
    queryReq.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    const quint64 generation = _requestGeneration;
    QNetworkReply* queryReply = _nam->post(
        queryReq, QJsonDocument(queryBody).toJson(QJsonDocument::Compact));
    _activeArtifactReply = queryReply;
    _activeArtifactReplyJobId = localJobId;
    connect(queryReply, &QNetworkReply::finished, this,
            [this, queryReply, localJobId, objects, requestConfig, localOutputDir, source, generation]() {
        if (generation != _requestGeneration) {
            queryReply->deleteLater();
            return;
        }
        _activeArtifactReply = nullptr;
        _activeArtifactReplyJobId.clear();
        queryReply->deleteLater();

        if (_cancelledLocalUploadJobs.contains(localJobId)) {
            finishActiveArtifactUpload();
            processNextArtifactUpload();
            return;
        }
        if (isTransportError(queryReply)) {
            const QString msg = tr("Artifact query failed: %1").arg(queryReply->errorString());
            updateLocalUploadJob(localJobId, {
                {QStringLiteral("state"), QStringLiteral("error")},
                {QStringLiteral("error"), msg},
            });
            emit optimizationError(msg);
            finishActiveArtifactUpload();
            processNextArtifactUpload();
            return;
        }
        if (!validateApiVersion(queryReply, tr("Artifact query"))) {
            updateLocalUploadJob(localJobId, {
                {QStringLiteral("state"), QStringLiteral("error")},
                {QStringLiteral("error"), _lastError},
            });
            emit optimizationError(_lastError);
            finishActiveArtifactUpload();
            processNextArtifactUpload();
            return;
        }
        if (queryReply->error() != QNetworkReply::NoError) {
            const QString msg = tr("Artifact query failed: %1").arg(queryReply->errorString());
            updateLocalUploadJob(localJobId, {
                {QStringLiteral("state"), QStringLiteral("error")},
                {QStringLiteral("error"), msg},
            });
            emit optimizationError(msg);
            finishActiveArtifactUpload();
            processNextArtifactUpload();
            return;
        }
        const QJsonObject response = QJsonDocument::fromJson(queryReply->readAll()).object();
        if (response.contains(QStringLiteral("error"))) {
            const QString msg = response[QStringLiteral("error")].toString();
            updateLocalUploadJob(localJobId, {
                {QStringLiteral("state"), QStringLiteral("error")},
                {QStringLiteral("error"), msg},
            });
            emit optimizationError(msg);
            finishActiveArtifactUpload();
            processNextArtifactUpload();
            return;
        }
        QSet<QString> missing;
        for (const QJsonValue& value : response[QStringLiteral("missing")].toArray()) {
            missing.insert(objectRefKey(value.toObject()));
        }
        auto uploads = std::make_shared<QJsonArray>();
        QSet<QString> uploadable;
        for (const QJsonValue& value : objects) {
            const QJsonObject ref = value.toObject()[QStringLiteral("object")].toObject();
            if (!ref.isEmpty()) {
                uploadable.insert(objectRefKey(ref));
            }
        }
        QStringList unresolved;
        for (const QString& key : std::as_const(missing)) {
            if (!uploadable.contains(key)) {
                const QStringList parts = key.split(QLatin1Char('\n'));
                unresolved.append(parts.size() >= 3
                    ? QStringLiteral("%1 %2 %3").arg(parts[0], parts[1], parts[2])
                    : key);
            }
        }
        if (!unresolved.isEmpty()) {
            const QString msg = tr("Missing Lasagna artifact(s) with no local upload source: %1")
                .arg(unresolved.join(QStringLiteral("; ")));
            updateLocalUploadJob(localJobId, {
                {QStringLiteral("state"), QStringLiteral("error")},
                {QStringLiteral("error"), msg},
            });
            emit optimizationError(msg);
            finishActiveArtifactUpload();
            processNextArtifactUpload();
            return;
        }
        for (const QJsonValue& value : objects) {
            const QJsonObject upload = value.toObject();
            if (missing.contains(objectRefKey(upload[QStringLiteral("object")].toObject()))) {
                uploads->append(upload);
            }
        }

        updateLocalUploadJob(localJobId, {
            {QStringLiteral("upload_state"), QStringLiteral("uploading")},
            {QStringLiteral("upload_label"),
             uploads->isEmpty() ? tr("Artifacts already available") : tr("Uploading artifacts")},
            {QStringLiteral("upload_current"), 0},
            {QStringLiteral("upload_total"), uploads->size()},
            {QStringLiteral("upload_progress"), uploads->isEmpty() ? 1.0 : 0.0},
        });

        auto index = std::make_shared<int>(0);
        auto uploadNext = std::make_shared<std::function<void()>>();
        *uploadNext = [this, uploads, index, localJobId, requestConfig, localOutputDir, source,
                       generation, uploadNext]() {
            if (generation != _requestGeneration) {
                return;
            }
            if (_cancelledLocalUploadJobs.contains(localJobId)) {
                finishActiveArtifactUpload();
                processNextArtifactUpload();
                return;
            }
            if (*index >= uploads->size()) {
                updateLocalUploadJob(localJobId, {
                    {QStringLiteral("upload_state"), QStringLiteral("submitting")},
                    {QStringLiteral("upload_label"), tr("Submitting job")},
                    {QStringLiteral("upload_current"), uploads->size()},
                    {QStringLiteral("upload_total"), uploads->size()},
                    {QStringLiteral("upload_progress"), 1.0},
                });
                postOptimizationRequest(requestConfig, localOutputDir, localJobId);
                return;
            }

            const QJsonObject uploadSource = uploads->at(*index).toObject();
            const QJsonObject ref = uploadSource[QStringLiteral("object")].toObject();
            const QString label = ref[QStringLiteral("name")].toString().isEmpty()
                ? tr("Uploading artifact %1 of %2").arg(*index + 1).arg(uploads->size())
                : tr("Uploading %1").arg(ref[QStringLiteral("name")].toString());
            updateLocalUploadJob(localJobId, {
                {QStringLiteral("upload_label"), label},
                {QStringLiteral("upload_current"), *index + 1},
                {QStringLiteral("upload_total"), uploads->size()},
                {QStringLiteral("upload_progress"),
                 uploads->isEmpty() ? 1.0 : static_cast<double>(*index) / uploads->size()},
            });

            QUrl url(QStringLiteral("%1/objects").arg(baseUrl()));
            QNetworkRequest req = fitServiceRequest(url);
            req.setRawHeader(kVc3dSourceHeader, source.toUtf8());
            req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
            QJsonObject upload;
            try {
                upload = materializeLocalArtifactUpload(uploadSource);
            } catch (const std::exception& ex) {
                const QString msg = tr("Artifact packing failed: %1").arg(QString::fromStdString(ex.what()));
                updateLocalUploadJob(localJobId, {
                    {QStringLiteral("state"), QStringLiteral("error")},
                    {QStringLiteral("error"), msg},
                });
                emit optimizationError(msg);
                finishActiveArtifactUpload();
                processNextArtifactUpload();
                return;
            }
            const QByteArray body = QJsonDocument(upload).toJson(QJsonDocument::Compact);
            QNetworkReply* reply = _nam->post(req, body);
            _activeArtifactReply = reply;
            _activeArtifactReplyJobId = localJobId;
            connect(reply, &QNetworkReply::uploadProgress,
                    this,
                    [this, localJobId, uploads, index](qint64 bytesSent, qint64 bytesTotal) {
                if (bytesTotal <= 0 || uploads->isEmpty()
                    || _cancelledLocalUploadJobs.contains(localJobId)) {
                    return;
                }
                const double objectProgress = std::clamp(
                    static_cast<double>(bytesSent) / static_cast<double>(bytesTotal), 0.0, 1.0);
                const double progress = (static_cast<double>(*index) + objectProgress)
                    / static_cast<double>(uploads->size());
                updateLocalUploadJob(localJobId, {
                    {QStringLiteral("upload_progress"), progress},
                });
            });
            connect(reply, &QNetworkReply::finished, this,
                    [this, reply, uploads, index, localJobId, generation, uploadNext]() {
                if (generation != _requestGeneration) {
                    reply->deleteLater();
                    return;
                }
                _activeArtifactReply = nullptr;
                _activeArtifactReplyJobId.clear();
                reply->deleteLater();

                if (_cancelledLocalUploadJobs.contains(localJobId)) {
                    finishActiveArtifactUpload();
                    processNextArtifactUpload();
                    return;
                }
                if (isTransportError(reply)) {
                    const QString msg = tr("Artifact upload failed: %1").arg(reply->errorString());
                    updateLocalUploadJob(localJobId, {
                        {QStringLiteral("state"), QStringLiteral("error")},
                        {QStringLiteral("error"), msg},
                    });
                    emit optimizationError(msg);
                    finishActiveArtifactUpload();
                    processNextArtifactUpload();
                    return;
                }
                if (!validateApiVersion(reply, tr("Artifact upload"))) {
                    updateLocalUploadJob(localJobId, {
                        {QStringLiteral("state"), QStringLiteral("error")},
                        {QStringLiteral("error"), _lastError},
                    });
                    emit optimizationError(_lastError);
                    finishActiveArtifactUpload();
                    processNextArtifactUpload();
                    return;
                }
                if (reply->error() != QNetworkReply::NoError) {
                    const QString msg = tr("Artifact upload failed: %1").arg(reply->errorString());
                    updateLocalUploadJob(localJobId, {
                        {QStringLiteral("state"), QStringLiteral("error")},
                        {QStringLiteral("error"), msg},
                    });
                    emit optimizationError(msg);
                    finishActiveArtifactUpload();
                    processNextArtifactUpload();
                    return;
                }
                const QJsonObject response = QJsonDocument::fromJson(reply->readAll()).object();
                if (response.contains(QStringLiteral("error"))) {
                    const QString msg = response[QStringLiteral("error")].toString();
                    updateLocalUploadJob(localJobId, {
                        {QStringLiteral("state"), QStringLiteral("error")},
                        {QStringLiteral("error"), msg},
                    });
                    emit optimizationError(msg);
                    finishActiveArtifactUpload();
                    processNextArtifactUpload();
                    return;
                }
                ++(*index);
                (*uploadNext)();
            });
        };
        (*uploadNext)();
    });
}

void LasagnaServiceManager::postOptimizationRequest(const QJsonObject& requestConfig,
                                                    const QString& localOutputDir,
                                                    const QString& localUploadJobId)
{
    const QString source = requestConfig[QStringLiteral("source")].toString().trimmed().isEmpty()
        ? localSourceName()
        : requestConfig[QStringLiteral("source")].toString();
    _localOutputDir = localOutputDir;

    QUrl url(QStringLiteral("%1/jobs").arg(baseUrl()));
    QNetworkRequest req = fitServiceRequest(url);
    req.setRawHeader(kVc3dSourceHeader, source.toUtf8());
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    QByteArray body = QJsonDocument(requestConfig).toJson(QJsonDocument::Compact);
    const qint64 bodyBytes = body.size();
    const QString requestedOutputName = requestConfig[QStringLiteral("output_name")].toString().trimmed();

    std::cout << "[lasagna] sending queued optimize request: "
              << bytesToMiB(bodyBytes) << " MiB" << std::endl;

    const quint64 generation = _requestGeneration;
    QNetworkReply* reply = _nam->post(req, body);
    if (!localUploadJobId.isEmpty()) {
        _activeArtifactReply = reply;
        _activeArtifactReplyJobId = localUploadJobId;
    }
    auto sendTimer = std::make_shared<QElapsedTimer>();
    auto uploadLogged = std::make_shared<bool>(false);
    sendTimer->start();
    connect(reply, &QNetworkReply::uploadProgress,
            this,
            [sendTimer, uploadLogged, bodyBytes](qint64 bytesSent, qint64 bytesTotal) {
        if (*uploadLogged) {
            return;
        }
        const qint64 expected = bytesTotal > 0 ? bytesTotal : bodyBytes;
        if (expected > 0 && bytesSent >= expected) {
            *uploadLogged = true;
            logTransferTiming("optimize upload", expected, elapsedSeconds(*sendTimer));
        }
    });
    connect(reply, &QNetworkReply::finished, this,
            [this, reply, sendTimer, uploadLogged, bodyBytes, generation, localUploadJobId, requestedOutputName]() {
        if (generation != _requestGeneration) {
            reply->deleteLater();
            return;
        }
        if (!localUploadJobId.isEmpty()) {
            _activeArtifactReply = nullptr;
            _activeArtifactReplyJobId.clear();
            if (_cancelledLocalUploadJobs.contains(localUploadJobId)) {
                reply->deleteLater();
                finishActiveArtifactUpload();
                processNextArtifactUpload();
                return;
            }
        }
        if (!*uploadLogged) {
            *uploadLogged = true;
            logTransferTiming("optimize upload", bodyBytes, elapsedSeconds(*sendTimer));
        }
        const QVariant status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute);
        std::cout << "[lasagna] optimize response:"
                  << " elapsed=" << elapsedSeconds(*sendTimer) << "s";
        if (status.isValid()) {
            std::cout << " http=" << status.toInt();
        } else {
            std::cout << " http=<none>";
        }
        std::cout << " error=" << reply->error()
                  << " bytes_available=" << reply->bytesAvailable()
                  << std::endl;
        if (!requestedOutputName.isEmpty()) {
            reply->setProperty("vc3d_requested_output_name", requestedOutputName);
        }
        handleOptimizeReply(reply, localUploadJobId);
        if (!localUploadJobId.isEmpty()) {
            finishActiveArtifactUpload();
            processNextArtifactUpload();
        }
    });
}

void LasagnaServiceManager::startOptimization(const QJsonObject& config,
                                              const QString& localOutputDir)
{
    if (!isRunning()) {
        emit optimizationError(tr("Lasagna service is not running"));
        return;
    }

    _localOutputDir = localOutputDir;
    QJsonObject requestConfig = config;
    QString source = requestConfig.contains(QStringLiteral("source"))
        ? requestConfig[QStringLiteral("source")].toString()
        : localSourceName();
    if (source.trimmed().isEmpty()) {
        source = localSourceName();
    }
    if (!requestConfig.contains(QStringLiteral("source"))
        || requestConfig[QStringLiteral("source")].toString().trimmed().isEmpty()) {
        requestConfig[QStringLiteral("source")] = source;
    }

    const QJsonArray objects = requestConfig[QStringLiteral("_objects")].toArray();
    requestConfig.remove(QStringLiteral("_objects"));
    if (!objects.isEmpty()) {
        enqueueArtifactUpload(requestConfig, objects, localOutputDir, source);
        return;
    }

    postOptimizationRequest(requestConfig, localOutputDir);
}

void LasagnaServiceManager::stopOptimization()
{
    if (!isRunning()) return;

    if (_hasActiveArtifactUpload) {
        cancelLocalUploadJob(_activeArtifactUpload.jobId);
        return;
    }

    if (!_activeJobId.isEmpty()) {
        cancelJob(_activeJobId);
        return;
    }

    QUrl url(QStringLiteral("%1/stop").arg(baseUrl()));
    QNetworkRequest req = fitServiceRequest(url);
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    QNetworkReply* reply = _nam->post(req, QByteArray("{}"));
    const quint64 generation = _requestGeneration;
    connect(reply, &QNetworkReply::finished, this, [this, reply, generation]() {
        if (generation != _requestGeneration) {
            reply->deleteLater();
            return;
        }
        reply->deleteLater();
        if (isTransportError(reply)) {
            return;
        }
        validateApiVersion(reply, tr("Stop optimization"));
    });
}

void LasagnaServiceManager::cancelJob(const QString& jobId)
{
    if (cancelLocalUploadJob(jobId)) {
        return;
    }
    if (!isRunning() || jobId.isEmpty()) return;
    QUrl url(QStringLiteral("%1/jobs/%2/cancel").arg(baseUrl(), jobId));
    QNetworkRequest req = fitServiceRequest(url);
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    QNetworkReply* reply = _nam->post(req, QByteArray("{}"));
    const quint64 generation = _requestGeneration;
    connect(reply, &QNetworkReply::finished, this, [this, reply, generation]() {
        if (generation != _requestGeneration) {
            reply->deleteLater();
            return;
        }
        reply->deleteLater();
        if (isTransportError(reply)) {
            return;
        }
        validateApiVersion(reply, tr("Cancel job"));
    });
}

void LasagnaServiceManager::moveJobBefore(const QString& jobId, const QString& beforeJobId)
{
    if (!isRunning() || jobId.isEmpty()) return;
    QJsonObject body;
    body[QStringLiteral("job_id")] = jobId;
    if (!beforeJobId.isEmpty()) {
        body[QStringLiteral("before_job_id")] = beforeJobId;
    }
    QUrl url(QStringLiteral("%1/jobs/reorder").arg(baseUrl()));
    QNetworkRequest req = fitServiceRequest(url);
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    QNetworkReply* reply = _nam->post(req, QJsonDocument(body).toJson(QJsonDocument::Compact));
    const quint64 generation = _requestGeneration;
    connect(reply, &QNetworkReply::finished, this, [this, reply, generation]() {
        if (generation != _requestGeneration) {
            reply->deleteLater();
            return;
        }
        reply->deleteLater();
        if (isTransportError(reply)) {
            return;
        }
        validateApiVersion(reply, tr("Reorder jobs"));
    });
}

void LasagnaServiceManager::moveJobToEnd(const QString& jobId)
{
    moveJobBefore(jobId, QString());
}

void LasagnaServiceManager::exportLasagnaVis(const QJsonObject& config)
{
    if (!isRunning()) {
        emit visExportError(tr("Lasagna service is not running"));
        return;
    }

    // Extract output_dir from config — it's a client-side path, not sent to server
    QJsonObject serverConfig = config;
    _visOutputDir = serverConfig[QStringLiteral("output_dir")].toString();
    serverConfig.remove(QStringLiteral("output_dir"));

    QUrl url(QStringLiteral("%1/export_vis").arg(baseUrl()));
    QNetworkRequest req = fitServiceRequest(url);
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    req.setTransferTimeout(120000); // 2 min timeout for export

    QByteArray body = QJsonDocument(serverConfig).toJson(QJsonDocument::Compact);

    const quint64 generation = _requestGeneration;
    QNetworkReply* reply = _nam->post(req, body);
    connect(reply, &QNetworkReply::finished, this, [this, reply, generation]() {
        if (generation != _requestGeneration) {
            reply->deleteLater();
            return;
        }
        reply->deleteLater();

        if (isTransportError(reply)) {
            emit visExportError(tr("Export failed: %1").arg(reply->errorString()));
            return;
        }
        if (!validateApiVersion(reply, tr("Export visualization"))) {
            emit visExportError(_lastError);
            return;
        }
        if (reply->error() != QNetworkReply::NoError) {
            emit visExportError(tr("Export failed: %1").arg(reply->errorString()));
            return;
        }

        // Check Content-Type: JSON means error, gzip means success
        QString contentType = reply->header(QNetworkRequest::ContentTypeHeader).toString();
        QByteArray data = reply->readAll();

        if (contentType.contains(QStringLiteral("json"))) {
            QJsonDocument doc = QJsonDocument::fromJson(data);
            QJsonObject obj = doc.object();
            emit visExportError(obj[QStringLiteral("error")].toString());
            return;
        }

        // Extract tar.gz into _visOutputDir
        QDir().mkpath(_visOutputDir);
        QString tarPath = _visOutputDir + QStringLiteral("/.lasagna_vis.tar.gz");
        QFile tarFile(tarPath);
        if (!tarFile.open(QIODevice::WriteOnly)) {
            emit visExportError(tr("Cannot write temp file: %1").arg(tarPath));
            return;
        }
        tarFile.write(data);
        tarFile.close();

        QProcess tar;
        tar.setWorkingDirectory(_visOutputDir);
        tar.start(QStringLiteral("tar"),
                  {QStringLiteral("xzf"), tarPath});
        if (!tar.waitForFinished(30000)) {
            QFile::remove(tarPath);
            emit visExportError(tr("tar extraction timed out"));
            return;
        }
        QFile::remove(tarPath);

        if (tar.exitCode() != 0) {
            QString err = QString::fromUtf8(tar.readAllStandardError());
            emit visExportError(tr("tar extraction failed: %1").arg(err));
            return;
        }

        std::cout << "[lasagna] vis export unpacked to "
                  << _visOutputDir.toStdString() << " ("
                  << data.size() << " bytes)" << std::endl;
        emit visExportFinished(_visOutputDir);
    });
}

void LasagnaServiceManager::handleOptimizeReply(QNetworkReply* reply,
                                                const QString& localUploadJobId)
{
    reply->deleteLater();
    auto failLocalUpload = [this, &localUploadJobId](const QString& msg) {
        if (!localUploadJobId.isEmpty() && _localUploadJobs.contains(localUploadJobId)) {
            updateLocalUploadJob(localUploadJobId, {
                {QStringLiteral("state"), QStringLiteral("error")},
                {QStringLiteral("error"), msg},
            });
        }
    };

    if (isTransportError(reply)) {
        QString msg = tr("Failed to start optimization: %1").arg(reply->errorString());
        failLocalUpload(msg);
        emit optimizationError(msg);
        return;
    }
    if (!validateApiVersion(reply, tr("Submit optimization"))) {
        failLocalUpload(_lastError);
        emit optimizationError(_lastError);
        return;
    }
    if (reply->error() != QNetworkReply::NoError) {
        QString msg = tr("Failed to start optimization: %1").arg(reply->errorString());
        failLocalUpload(msg);
        emit optimizationError(msg);
        return;
    }

    QJsonDocument doc = QJsonDocument::fromJson(reply->readAll());
    QJsonObject obj = doc.object();

    if (obj.contains("error")) {
        const QString msg = obj["error"].toString();
        failLocalUpload(msg);
        emit optimizationError(msg);
        return;
    }

    const QString jobId = obj[QStringLiteral("job_id")].toString();
    if (obj.contains(QStringLiteral("queue_generation"))) {
        _lastQueueGeneration = static_cast<qint64>(obj[QStringLiteral("queue_generation")].toDouble());
    }
    if (!jobId.isEmpty()) {
        if (!localUploadJobId.isEmpty()) {
            removeLocalUploadJob(localUploadJobId);
        }
        _activeJobId = jobId;
        _submittedJobIds.insert(jobId);
        _startedJobIds.insert(jobId);
        _jobOutputDirs.insert(jobId, _localOutputDir);
        QString outputName = reply->property("vc3d_requested_output_name").toString().trimmed();
        if (outputName.isEmpty()) {
            outputName = obj[QStringLiteral("output_name")].toString().trimmed();
        }
        _jobOutputNames.insert(jobId, outputName);
        QJsonObject optimisticJob;
        optimisticJob[QStringLiteral("job_id")] = jobId;
        optimisticJob[QStringLiteral("sequence")] = obj[QStringLiteral("sequence")].toInt();
        optimisticJob[QStringLiteral("source")] = obj[QStringLiteral("source")].toString();
        optimisticJob[QStringLiteral("config_name")] = obj[QStringLiteral("config_name")].toString();
        optimisticJob[QStringLiteral("output_name")] = obj[QStringLiteral("output_name")].toString();
        optimisticJob[QStringLiteral("state")] = QStringLiteral("waiting");
        optimisticJob[QStringLiteral("queue_position")] = obj[QStringLiteral("queue_position")].toInt();
        optimisticJob[QStringLiteral("submitted_at")] = static_cast<double>(QDateTime::currentSecsSinceEpoch());
        QJsonArray optimisticJobs = _lastJobs;
        optimisticJobs.append(optimisticJob);
        _lastJobs = optimisticJobs;
        emitJobsUpdatedOverlay();
        emit jobStarted(jobId);
        emit statusMessage(tr("Lasagna job %1 queued at position %2")
                               .arg(jobId)
                               .arg(obj[QStringLiteral("queue_position")].toInt()));
    } else if (!localUploadJobId.isEmpty()) {
        const QString msg = tr("Failed to start optimization: missing job id in response");
        failLocalUpload(msg);
        emit optimizationError(msg);
    }
    _optimizationRunning = true;
    _pollTimer->start();
    emit optimizationStarted();
    fetchJobs();
}

void LasagnaServiceManager::pollStatus()
{
    if (!isRunning()) {
        _pollTimer->stop();
        return;
    }
    if (_statusRequestInFlight) {
        return;
    }
    _statusRequestInFlight = true;

    QUrl url(QStringLiteral("%1/status").arg(baseUrl()));
    QNetworkRequest req = fitServiceRequest(url);

    const quint64 generation = _requestGeneration;
    QNetworkReply* reply = _nam->get(req);
    connect(reply, &QNetworkReply::finished, this, [this, reply, generation]() {
        if (generation != _requestGeneration) {
            reply->deleteLater();
            return;
        }
        handleStatusReply(reply);
    });
}

void LasagnaServiceManager::fetchJobs()
{
    if (!isRunning()) {
        return;
    }
    if (_jobsRequestInFlight) {
        _jobsRequestPending = true;
        return;
    }
    _jobsRequestInFlight = true;
    _jobsRequestPending = false;

    QUrl url(QStringLiteral("%1/jobs").arg(baseUrl()));
    QNetworkRequest req = fitServiceRequest(url);

    const quint64 generation = _requestGeneration;
    QNetworkReply* reply = _nam->get(req);
    connect(reply, &QNetworkReply::finished, this, [this, reply, generation]() {
        if (generation != _requestGeneration) {
            reply->deleteLater();
            return;
        }
        handleJobsReply(reply);
    });
}

void LasagnaServiceManager::handleJobsReply(QNetworkReply* reply)
{
    _jobsRequestInFlight = false;
    reply->deleteLater();

    if (isTransportError(reply)) {
        if (_jobsRequestPending) {
            fetchJobs();
        }
        return;
    }
    if (!validateApiVersion(reply, tr("Fetch jobs"))) {
        return;
    }
    if (reply->error() != QNetworkReply::NoError) {
        if (_jobsRequestPending) {
            fetchJobs();
        }
        return;
    }

    QJsonDocument doc = QJsonDocument::fromJson(reply->readAll());
    QJsonObject root = doc.object();
    QJsonArray jobs = root[QStringLiteral("jobs")].toArray();
    if (root.contains(QStringLiteral("queue_generation"))) {
        _fetchedQueueGeneration = static_cast<qint64>(root[QStringLiteral("queue_generation")].toDouble());
        _lastQueueGeneration = _fetchedQueueGeneration;
    }
    _lastJobs = jobs;
    emitJobsUpdatedOverlay();

    bool anyTrackedActive = false;
    for (const QJsonValue& value : jobs) {
        QJsonObject obj = value.toObject();
        const QString jobId = obj[QStringLiteral("job_id")].toString();
        if (!_submittedJobIds.contains(jobId)) {
            continue;
        }

        const QString state = obj[QStringLiteral("state")].toString();
        if (state == QStringLiteral("running")) {
            anyTrackedActive = true;
            _activeJobId = jobId;
            if (!_startedJobIds.contains(jobId)) {
                _startedJobIds.insert(jobId);
                emit jobStarted(jobId);
            }
            emit optimizationProgress(obj[QStringLiteral("stage")].toString(),
                                      obj[QStringLiteral("step")].toInt(),
                                      obj[QStringLiteral("total_steps")].toInt(),
                                      obj[QStringLiteral("loss")].toDouble(),
                                      obj[QStringLiteral("stage_progress")].toDouble(),
                                      obj[QStringLiteral("overall_progress")].toDouble(),
                                      obj[QStringLiteral("stage_name")].toString());
        } else if ((state == QStringLiteral("upload") || state == QStringLiteral("waiting"))
                   && !_completedJobIds.contains(jobId)) {
            anyTrackedActive = true;
        } else if (state == QStringLiteral("finished") && !_completedJobIds.contains(jobId)) {
            _completedJobIds.insert(jobId);
            const QString localOutputDir = _jobOutputDirs.value(jobId);
            if (!localOutputDir.isEmpty()) {
                downloadResults(jobId, localOutputDir);
            } else {
                const QString outputDir = obj[QStringLiteral("output_dir")].toString();
                emit jobFinished(jobId, outputDir);
                emit optimizationFinished(outputDir);
            }
        } else if ((state == QStringLiteral("error") || state == QStringLiteral("cancelled"))
                   && !_completedJobIds.contains(jobId)) {
            _completedJobIds.insert(jobId);
            QString errorMsg = obj[QStringLiteral("error")].toString();
            if (errorMsg.isEmpty()) {
                errorMsg = state == QStringLiteral("cancelled") ? tr("Cancelled") : tr("Unknown error");
            }
            emit jobError(jobId, errorMsg);
            emit optimizationError(errorMsg);
        }
    }

    _optimizationRunning = anyTrackedActive;
    if (!anyTrackedActive) {
        _activeJobId.clear();
    }
    if (_jobsRequestPending) {
        fetchJobs();
    }
}

void LasagnaServiceManager::handleStatusReply(QNetworkReply* reply)
{
    _statusRequestInFlight = false;
    reply->deleteLater();

    if (isTransportError(reply)) {
        return;  // Transient network error, will retry next poll
    }
    if (!validateApiVersion(reply, tr("Poll status"))) {
        return;
    }
    if (reply->error() != QNetworkReply::NoError) {
        return;
    }

    QJsonDocument doc = QJsonDocument::fromJson(reply->readAll());
    QJsonObject obj = doc.object();
    const bool hasQueueGeneration = obj.contains(QStringLiteral("queue_generation"));
    const qint64 queueGeneration = hasQueueGeneration
        ? static_cast<qint64>(obj[QStringLiteral("queue_generation")].toDouble())
        : -1;

    if (!hasQueueGeneration || queueGeneration != _fetchedQueueGeneration) {
        _lastQueueGeneration = queueGeneration;
        fetchJobs();
    }

    QString state = obj["state"].toString();
    QString stage = obj["stage"].toString();
    int step = obj["step"].toInt();
    int totalSteps = obj["total_steps"].toInt();
    double loss = obj["loss"].toDouble();
    double stageProgress = obj["stage_progress"].toDouble();
    double overallProgress = obj["overall_progress"].toDouble();
    QString stageName = obj["stage_name"].toString();

    if (state == "running") {
        const QString jobId = obj[QStringLiteral("job_id")].toString();
        if (!jobId.isEmpty()) {
            _activeJobId = jobId;
        }
        updateCachedJobFromStatus(obj);
        emit optimizationProgress(stage, step, totalSteps, loss,
                                  stageProgress, overallProgress, stageName);
    }
}

// ---------------------------------------------------------------------------
// Results download
// ---------------------------------------------------------------------------

void LasagnaServiceManager::downloadResults(const QString& jobId,
                                            const QString& outputDir)
{
    emit statusMessage(tr("Downloading results from external service..."));

    const QString targetDir = outputDir.isEmpty() ? _localOutputDir : outputDir;
    const QString expectedOutputName = _jobOutputNames.value(jobId);
    QUrl url(jobId.isEmpty()
        ? QStringLiteral("%1/results").arg(baseUrl())
        : QStringLiteral("%1/jobs/%2/results").arg(baseUrl(), jobId));
    QNetworkRequest req = fitServiceRequest(url);

    const quint64 generation = _requestGeneration;
    QNetworkReply* reply = _nam->get(req);
    connect(reply, &QNetworkReply::finished, this, [this, reply, jobId, targetDir, expectedOutputName, generation]() {
        if (generation != _requestGeneration) {
            reply->deleteLater();
            return;
        }
        reply->deleteLater();

        if (isTransportError(reply)) {
            const QString msg = tr("Failed to download results: %1").arg(reply->errorString());
            if (!jobId.isEmpty()) {
                emit jobError(jobId, msg);
            }
            emit optimizationError(msg);
            return;
        }
        if (!validateApiVersion(reply, tr("Download results"))) {
            if (!jobId.isEmpty()) {
                emit jobError(jobId, _lastError);
            }
            emit optimizationError(_lastError);
            return;
        }
        if (reply->error() != QNetworkReply::NoError) {
            const QString msg = tr("Failed to download results: %1").arg(reply->errorString());
            if (!jobId.isEmpty()) {
                emit jobError(jobId, msg);
            }
            emit optimizationError(msg);
            return;
        }

        QByteArray data = reply->readAll();
        std::cout << "[lasagna] downloaded results archive ("
                  << data.size() << " bytes)" << std::endl;

        emit statusMessage(tr("Unpacking results from external service..."));

        auto* watcher = new QFutureWatcher<ResultsPlacementResult>(this);
        connect(watcher, &QFutureWatcher<ResultsPlacementResult>::finished,
                this, [this, watcher, jobId]() {
            watcher->deleteLater();
            const ResultsPlacementResult result = watcher->result();
            if (!result.ok) {
                if (!jobId.isEmpty()) {
                    emit jobError(jobId, result.error);
                }
                emit optimizationError(result.error);
                return;
            }

            std::cout << "[lasagna] results unpacked to "
                      << result.targetDir.toStdString();
            if (!result.placedNames.isEmpty()) {
                std::cout << " as " << result.placedNames.join(QStringLiteral(", ")).toStdString();
            }
            std::cout << std::endl;
            for (const QString& warning : result.warnings) {
                std::cerr << "[lasagna] " << warning.toStdString() << std::endl;
                emit statusMessage(warning);
            }
            emit resultsPlaced(result.targetDir, result.placedNames);
            if (!jobId.isEmpty()) {
                emit jobFinished(jobId, result.targetDir);
            }
            emit optimizationFinished(result.targetDir);
        });
        watcher->setFuture(QtConcurrent::run([data = std::move(data), targetDir, expectedOutputName]() {
            return placeResultsArchive(data, targetDir, expectedOutputName);
        }));
    });
}

// ---------------------------------------------------------------------------
// Service discovery
// ---------------------------------------------------------------------------

QJsonArray LasagnaServiceManager::discoverServices()
{
    QJsonArray result;

    // Track seen host:port to avoid duplicates between file and mDNS discovery
    QSet<QString> seen;

    // --- File-based discovery (local) ---
    QString dirPath = QDir::homePath() + QStringLiteral("/.fit_services");
    QDir dir(dirPath);
    if (dir.exists()) {
        const auto entries = dir.entryInfoList(
            QStringList{QStringLiteral("*.json")}, QDir::Files);
        for (const QFileInfo& fi : entries) {
            QFile f(fi.absoluteFilePath());
            if (!f.open(QIODevice::ReadOnly)) {
                continue;
            }
            QJsonDocument doc = QJsonDocument::fromJson(f.readAll());
            f.close();
            if (!doc.isObject()) {
                QFile::remove(fi.absoluteFilePath());
                continue;
            }
            QJsonObject obj = doc.object();
            int pid = obj[QStringLiteral("pid")].toInt(-1);
            if (pid <= 0) {
                QFile::remove(fi.absoluteFilePath());
                continue;
            }

#ifdef Q_OS_UNIX
            if (kill(pid, 0) != 0) {
                QFile::remove(fi.absoluteFilePath());
                continue;
            }
#endif

            QString key = QStringLiteral("%1:%2")
                .arg(obj[QStringLiteral("host")].toString())
                .arg(obj[QStringLiteral("port")].toInt());
            seen.insert(key);
            result.append(obj);
        }
    }

    // --- mDNS discovery via avahi-client library ---
#if defined(Q_OS_LINUX) && defined(VC_HAVE_AVAHI) && VC_HAVE_AVAHI
    {
        struct AvahiDiscovery {
            QJsonArray* result;
            QSet<QString>* seen;
            AvahiSimplePoll* poll;
            AvahiClient* client;
            int pendingResolves{0};
            bool browseComplete{false};

            void maybeQuit() {
                if (browseComplete && pendingResolves <= 0)
                    avahi_simple_poll_quit(poll);
            }

            void addResolved(const char* name, const AvahiAddress* addr,
                             uint16_t port, AvahiStringList* txt) {
                char addrBuf[AVAHI_ADDRESS_STR_MAX];
                avahi_address_snprint(addrBuf, sizeof(addrBuf), addr);
                QString host = QString::fromUtf8(addrBuf);
                QString key = QStringLiteral("%1:%2").arg(host).arg(port);
                if (seen->contains(key)) return;
                seen->insert(key);

                QJsonObject obj;
                obj[QStringLiteral("host")] = host;
                obj[QStringLiteral("port")] = static_cast<int>(port);
                obj[QStringLiteral("name")] = QString::fromUtf8(name);

                for (auto* t = txt; t; t = avahi_string_list_get_next(t)) {
                    char* k = nullptr;
                    char* v = nullptr;
                    if (avahi_string_list_get_pair(t, &k, &v, nullptr) == 0 && k) {
                        QString tk = QString::fromUtf8(k);
                        QString tv = v ? QString::fromUtf8(v) : QString();
                        if (tk == QStringLiteral("data_dir"))
                            obj[QStringLiteral("data_dir")] = tv;
                        else if (tk == QStringLiteral("datasets")) {
                            QJsonArray ds;
                            for (const QString& d : tv.split(','))
                                if (!d.isEmpty()) ds.append(d);
                            obj[QStringLiteral("datasets")] = ds;
                        }
                        avahi_free(k);
                        avahi_free(v);
                    }
                }
                result->append(obj);
            }

            static void resolveCallback(
                    AvahiServiceResolver* r, AvahiIfIndex, AvahiProtocol,
                    AvahiResolverEvent event, const char* name, const char*,
                    const char*, const char*, const AvahiAddress* addr,
                    uint16_t port, AvahiStringList* txt,
                    AvahiLookupResultFlags, void* userdata) {
                auto* self = static_cast<AvahiDiscovery*>(userdata);
                if (event == AVAHI_RESOLVER_FOUND && addr)
                    self->addResolved(name, addr, port, txt);
                self->pendingResolves--;
                avahi_service_resolver_free(r);
                self->maybeQuit();
            }

            static void browseCallback(
                    AvahiServiceBrowser*, AvahiIfIndex iface,
                    AvahiProtocol proto, AvahiBrowserEvent event,
                    const char* name, const char* type, const char* domain,
                    AvahiLookupResultFlags, void* userdata) {
                auto* self = static_cast<AvahiDiscovery*>(userdata);
                if (event == AVAHI_BROWSER_NEW) {
                    self->pendingResolves++;
                    avahi_service_resolver_new(
                        self->client, iface, proto, name, type, domain,
                        AVAHI_PROTO_UNSPEC, static_cast<AvahiLookupFlags>(0),
                        resolveCallback, userdata);
                } else if (event == AVAHI_BROWSER_ALL_FOR_NOW ||
                           event == AVAHI_BROWSER_FAILURE) {
                    self->browseComplete = true;
                    self->maybeQuit();
                }
            }
        };

        auto* poll = avahi_simple_poll_new();
        if (poll) {
            int error = 0;
            auto* client = avahi_client_new(
                avahi_simple_poll_get(poll), static_cast<AvahiClientFlags>(0),
                [](AvahiClient*, AvahiClientState, void*) {}, nullptr, &error);

            if (client) {
                AvahiDiscovery ctx{&result, &seen, poll, client, 0, false};
                auto* browser = avahi_service_browser_new(
                    client, AVAHI_IF_UNSPEC, AVAHI_PROTO_UNSPEC,
                    "_fitoptimizer._tcp", nullptr,
                    static_cast<AvahiLookupFlags>(0),
                    AvahiDiscovery::browseCallback, &ctx);

                if (browser) {
                    QElapsedTimer timer;
                    timer.start();
                    while (!timer.hasExpired(5000)) {
                        if (avahi_simple_poll_iterate(poll, 200) != 0)
                            break;
                    }
                    avahi_service_browser_free(browser);
                }
                avahi_client_free(client);
            } else {
                std::cerr << "[lasagna] avahi client error: "
                          << avahi_strerror(error) << std::endl;
            }
            avahi_simple_poll_free(poll);
        }
    }
#elif defined(Q_OS_LINUX)
    std::cerr << "[lasagna] avahi-client headers not available; skipping mDNS discovery"
              << std::endl;
#endif

    return result;
}

void LasagnaServiceManager::fetchDatasets()
{
    if (!isRunning()) {
        return;
    }

    QUrl url(QStringLiteral("%1/datasets").arg(baseUrl()));
    QNetworkRequest req = fitServiceRequest(url);

    const quint64 generation = _requestGeneration;
    QNetworkReply* reply = _nam->get(req);
    connect(reply, &QNetworkReply::finished, this, [this, reply, generation]() {
        if (generation != _requestGeneration) {
            reply->deleteLater();
            return;
        }
        reply->deleteLater();
        if (isTransportError(reply)) {
            return;
        }
        if (!validateApiVersion(reply, tr("Fetch datasets"))) {
            return;
        }
        if (reply->error() != QNetworkReply::NoError) {
            return;
        }
        QJsonDocument doc = QJsonDocument::fromJson(reply->readAll());
        QJsonObject obj = doc.object();
        QJsonArray datasets = obj[QStringLiteral("datasets")].toArray();
        emit datasetsReceived(datasets);
    });
}
