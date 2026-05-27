#include "LasagnaServiceManager.hpp"

#include <QCoreApplication>
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
#include <functional>
#include <iostream>
#include <memory>

#ifdef Q_OS_UNIX
#include <signal.h>
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

void updateTifxyzUuid(const QString& tifxyzDir, const QString& name)
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
};

ResultsPlacementResult placeResultsArchive(const QByteArray& data, const QString& targetDir)
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
    for (const QFileInfo& child : children) {
        if (child.fileName() == QStringLiteral(".lasagna_results.tar.gz")) {
            continue;
        }
        const QString finalName = uniqueSegmentName(targetDir, child.fileName());
        const QString finalPath = QDir(targetDir).filePath(finalName);
        if (QFileInfo::exists(finalPath)) {
            result.error = QObject::tr("Refusing to overwrite existing segment: %1").arg(finalPath);
            return result;
        }
        if (child.isDir() && finalName != child.fileName()) {
            updateTifxyzUuid(child.absoluteFilePath(), finalName);
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

LasagnaServiceManager::LasagnaServiceManager(QObject* parent)
    : QObject(parent)
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

bool LasagnaServiceManager::ensureServiceRunning(const QString& pythonPath)
{
    if (_isExternal && _serviceReady) {
        return true;
    }
    if (_process && _process->state() == QProcess::Running && _serviceReady) {
        return true;
    }
    return startService(pythonPath);
}

void LasagnaServiceManager::connectToExternal(const QString& host, int port)
{
    // Stop any existing internal service first
    if (_process) {
        stopService();
    }

    ++_requestGeneration;
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

bool LasagnaServiceManager::startService(const QString& pythonPath)
{
    ++_requestGeneration;
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

    _process = std::make_unique<QProcess>();
    _process->setProcessChannelMode(QProcess::SeparateChannels);

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
    QStringList args = {scriptPath, "--port", "0"};

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
        return false;
    }

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

    if (_isExternal) {
        // External mode: just reset state, don't terminate any process
        _serviceReady = false;
        _optimizationRunning = false;
        _isExternal = false;
        _host = QStringLiteral("127.0.0.1");
        _port = 0;
        _activeJobId.clear();
        _submittedJobIds.clear();
        _startedJobIds.clear();
        _completedJobIds.clear();
        _jobOutputDirs.clear();
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
        _process->terminate();
        if (!_process->waitForFinished(kServiceStopTimeoutMs)) {
            _process->kill();
            _process->waitForFinished(1000);
        }
    }

    _process.reset();
    _serviceReady = false;
    _port = 0;
    _optimizationRunning = false;
    _activeJobId.clear();
    _submittedJobIds.clear();
    _startedJobIds.clear();
    _completedJobIds.clear();
    _jobOutputDirs.clear();
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
        QJsonArray refs;
        QSet<QString> refKeys;
        const QJsonObject jobSpec = requestConfig[QStringLiteral("job_spec")].toObject();
        const QJsonObject modelRef = jobSpec[QStringLiteral("model")].toObject();
        if (!modelRef.isEmpty()) {
            refs.append(modelRef);
            refKeys.insert(objectRefKey(modelRef));
        }
        for (const QJsonValue& value : jobSpec[QStringLiteral("linked_surfaces")].toArray()) {
            const QJsonObject ref = value.toObject();
            if (!ref.isEmpty() && !refKeys.contains(objectRefKey(ref))) {
                refs.append(ref);
                refKeys.insert(objectRefKey(ref));
            }
        }
        for (const QJsonValue& value : objects) {
            const QJsonObject upload = value.toObject();
            const QJsonObject ref = upload[QStringLiteral("object")].toObject();
            if (!ref.isEmpty() && !refKeys.contains(objectRefKey(ref))) {
                refs.append(ref);
                refKeys.insert(objectRefKey(ref));
            }
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
        connect(queryReply, &QNetworkReply::finished, this,
                [this, queryReply, objects, requestConfig, localOutputDir, source, generation]() {
            if (generation != _requestGeneration) {
                queryReply->deleteLater();
                return;
            }
            queryReply->deleteLater();
            if (isTransportError(queryReply)) {
                emit optimizationError(tr("Artifact query failed: %1").arg(queryReply->errorString()));
                return;
            }
            if (!validateApiVersion(queryReply, tr("Artifact query"))) {
                emit optimizationError(_lastError);
                return;
            }
            if (queryReply->error() != QNetworkReply::NoError) {
                emit optimizationError(tr("Artifact query failed: %1").arg(queryReply->errorString()));
                return;
            }
            const QJsonObject response = QJsonDocument::fromJson(queryReply->readAll()).object();
            if (response.contains(QStringLiteral("error"))) {
                emit optimizationError(response[QStringLiteral("error")].toString());
                return;
            }
            QSet<QString> missing;
            for (const QJsonValue& value : response[QStringLiteral("missing")].toArray()) {
                missing.insert(objectRefKey(value.toObject()));
            }
            auto uploads = std::make_shared<QJsonArray>();
            for (const QJsonValue& value : objects) {
                const QJsonObject upload = value.toObject();
                if (missing.contains(objectRefKey(upload[QStringLiteral("object")].toObject()))) {
                    uploads->append(upload);
                }
            }

            auto index = std::make_shared<int>(0);
            auto uploadNext = std::make_shared<std::function<void()>>();
            *uploadNext = [this, uploads, index, requestConfig, localOutputDir, source, generation, uploadNext]() {
                if (generation != _requestGeneration) {
                    return;
                }
                if (*index >= uploads->size()) {
                    startOptimization(requestConfig, localOutputDir);
                    return;
                }
                const QJsonObject upload = uploads->at(*index).toObject();
                QUrl url(QStringLiteral("%1/objects").arg(baseUrl()));
                QNetworkRequest req = fitServiceRequest(url);
                req.setRawHeader(kVc3dSourceHeader, source.toUtf8());
                req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
                QNetworkReply* reply = _nam->post(req, QJsonDocument(upload).toJson(QJsonDocument::Compact));
                connect(reply, &QNetworkReply::finished, this,
                        [this, reply, uploads, index, requestConfig, localOutputDir, generation, uploadNext]() {
                    if (generation != _requestGeneration) {
                        reply->deleteLater();
                        return;
                    }
                    reply->deleteLater();
                    if (isTransportError(reply)) {
                        emit optimizationError(tr("Artifact upload failed: %1").arg(reply->errorString()));
                        return;
                    }
                    if (!validateApiVersion(reply, tr("Artifact upload"))) {
                        emit optimizationError(_lastError);
                        return;
                    }
                    if (reply->error() != QNetworkReply::NoError) {
                        emit optimizationError(tr("Artifact upload failed: %1").arg(reply->errorString()));
                        return;
                    }
                    const QJsonObject response = QJsonDocument::fromJson(reply->readAll()).object();
                    if (response.contains(QStringLiteral("error"))) {
                        emit optimizationError(response[QStringLiteral("error")].toString());
                        return;
                    }
                    ++(*index);
                    (*uploadNext)();
                });
            };
            (*uploadNext)();
        });
        emit statusMessage(tr("Syncing Lasagna artifacts..."));
        return;
    }

    QUrl url(QStringLiteral("%1/jobs").arg(baseUrl()));
    QNetworkRequest req = fitServiceRequest(url);
    req.setRawHeader(kVc3dSourceHeader, source.toUtf8());
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    QByteArray body = QJsonDocument(requestConfig).toJson(QJsonDocument::Compact);
    const qint64 bodyBytes = body.size();

    std::cout << "[lasagna] sending queued optimize request: "
              << bytesToMiB(bodyBytes) << " MiB" << std::endl;

    const quint64 generation = _requestGeneration;
    QNetworkReply* reply = _nam->post(req, body);
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
            [this, reply, sendTimer, uploadLogged, bodyBytes, generation]() {
        if (generation != _requestGeneration) {
            reply->deleteLater();
            return;
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
        handleOptimizeReply(reply);
    });
}

void LasagnaServiceManager::stopOptimization()
{
    if (!isRunning()) return;

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

void LasagnaServiceManager::handleOptimizeReply(QNetworkReply* reply)
{
    reply->deleteLater();

    if (isTransportError(reply)) {
        QString msg = tr("Failed to start optimization: %1").arg(reply->errorString());
        emit optimizationError(msg);
        return;
    }
    if (!validateApiVersion(reply, tr("Submit optimization"))) {
        emit optimizationError(_lastError);
        return;
    }
    if (reply->error() != QNetworkReply::NoError) {
        QString msg = tr("Failed to start optimization: %1").arg(reply->errorString());
        emit optimizationError(msg);
        return;
    }

    QJsonDocument doc = QJsonDocument::fromJson(reply->readAll());
    QJsonObject obj = doc.object();

    if (obj.contains("error")) {
        emit optimizationError(obj["error"].toString());
        return;
    }

    const QString jobId = obj[QStringLiteral("job_id")].toString();
    if (obj.contains(QStringLiteral("queue_generation"))) {
        _lastQueueGeneration = static_cast<qint64>(obj[QStringLiteral("queue_generation")].toDouble());
    }
    if (!jobId.isEmpty()) {
        _activeJobId = jobId;
        _submittedJobIds.insert(jobId);
        _startedJobIds.insert(jobId);
        _jobOutputDirs.insert(jobId, _localOutputDir);
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
        emit jobsUpdated(optimisticJobs);
        emit jobStarted(jobId);
        emit statusMessage(tr("Lasagna job %1 queued at position %2")
                               .arg(jobId)
                               .arg(obj[QStringLiteral("queue_position")].toInt()));
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
    emit jobsUpdated(jobs);

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
    QUrl url(jobId.isEmpty()
        ? QStringLiteral("%1/results").arg(baseUrl())
        : QStringLiteral("%1/jobs/%2/results").arg(baseUrl(), jobId));
    QNetworkRequest req = fitServiceRequest(url);

    const quint64 generation = _requestGeneration;
    QNetworkReply* reply = _nam->get(req);
    connect(reply, &QNetworkReply::finished, this, [this, reply, jobId, targetDir, generation]() {
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
            emit resultsPlaced(result.targetDir, result.placedNames);
            if (!jobId.isEmpty()) {
                emit jobFinished(jobId, result.targetDir);
            }
            emit optimizationFinished(result.targetDir);
        });
        watcher->setFuture(QtConcurrent::run([data = std::move(data), targetDir]() {
            return placeResultsArchive(data, targetDir);
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
