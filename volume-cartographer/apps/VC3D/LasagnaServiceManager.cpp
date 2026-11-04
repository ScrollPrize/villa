#include "LasagnaServiceManager.hpp"
#include "ObjectStoreRegions.hpp"
#include <vc_object_store/Uuid.hpp>

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
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
#include <QUrl>

#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <system_error>

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

        _serviceStoreUuid = QJsonDocument::fromJson(reply->readAll()).object()
                                .value(QStringLiteral("store_uuid")).toString();
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
    if (_storePath.isEmpty()) {
        _lastError = tr("Lasagna needs an attached object store.");
        emit serviceError(_lastError);
        return false;
    }
    _serviceStorePath = _storePath;

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
        QStringLiteral("--data-dir"), _dataDirectory,
        // The service writes into this process's store file; its session syncs it.
        QStringLiteral("--store"), _serviceStorePath, QStringLiteral("--shared-store")
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

    if (_isExternal) {
        // External mode: just reset state, don't terminate any process
        _serviceReady = false;
        _optimizationRunning = false;
        _isExternal = false;
        _serviceStoreUuid.clear();
        _dataDirectory.clear();
        _host = QStringLiteral("127.0.0.1");
        _port = 0;
        _activeJobId.clear();
        _submittedJobIds.clear();
        _startedJobIds.clear();
        _completedJobIds.clear();
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
    emit jobsUpdated(_lastJobs);
}

void LasagnaServiceManager::postOptimizationRequest(const QJsonObject& requestConfig)
{
    const QString source = requestConfig[QStringLiteral("source")].toString().trimmed().isEmpty()
        ? localSourceName()
        : requestConfig[QStringLiteral("source")].toString();

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

void LasagnaServiceManager::startOptimization(const QJsonObject& config)
{
    if (!isRunning()) {
        emit optimizationError(tr("Lasagna service is not running"));
        return;
    }
    if (const QString error = storeError(); !error.isEmpty()) {
        emit optimizationError(error);
        return;
    }

    QJsonObject requestConfig = config;
    requestConfig[QStringLiteral("store_uuid")] = QString::fromStdString(_storeAttachment->storeUuid);
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

    // The job reads the surfaces' current revisions from the store: an
    // external service sees them once this replica has pushed its edits.
    afterStoreSync([this, requestConfig]() { postOptimizationRequest(requestConfig); });
}

QString LasagnaServiceManager::storeError() const
{
    if (_storeSession.expired() || !_storeAttachment) {
        return tr("Lasagna needs an attached object store.");
    }
    if (_isExternal && _serviceStoreUuid != QString::fromStdString(_storeAttachment->storeUuid)) {
        return _serviceStoreUuid.isEmpty()
            ? tr("The Lasagna service at %1:%2 has no object store; start it with --store on a replica of "
                 "this hub.").arg(_host).arg(_port)
            : tr("The Lasagna service at %1:%2 works on store %3, not the attached store; attach the same "
                 "hosted store.").arg(_host).arg(_port).arg(_serviceStoreUuid);
    }
    return {};
}

void LasagnaServiceManager::afterStoreSync(std::function<void()> then)
{
    const auto session = _storeSession.lock();
    if (!_isExternal || !session || !session->hubStatus(*_storeAttachment)) {
        then();
        return;
    }
    // The remote service reads the hub: wait off the GUI thread for a sync started now.
    const quint64 generation = _requestGeneration;
    vc3d::store::background<std::optional<vc::objects::HubSessionStatus>>(this,
        [session, attachment = *_storeAttachment] { return session->syncHubAndWait(attachment, std::chrono::seconds(60)); },
        [this, generation, then](vc3d::store::Outcome<std::optional<vc::objects::HubSessionStatus>> result) {
            if (generation != _requestGeneration || !_storeAttachment) return;
            const auto& status = result.value;
            QString error;
            if (!result.error.isEmpty())
                error = tr("The object store did not sync with its hub: %1").arg(result.error);
            else if (!status)
                error = tr("The object store was detached.");
            else if (!status->conflicts.empty())
                error = tr("Resolve the object store's hub conflicts before running Lasagna.");
            else if (!status->error.empty())
                error = tr("The object store did not sync with its hub: %1").arg(QString::fromStdString(status->error));
            if (error.isEmpty())
                then();
            else
                emit optimizationError(error);
        });
}

void LasagnaServiceManager::setObjectStore(std::shared_ptr<vc::objects::ObjectStoreSession> session,
                                           std::optional<vc::objects::AttachmentIdentity> attachment)
{
    _storeSession = session;
    _storeAttachment = attachment;
    _storePath.clear();
    if (session && _storeAttachment) _storePath = QString::fromStdString(session->path(*_storeAttachment).string());
    // The internal service works on one store file; restart it on the next job.
    if (_process && !_isExternal && _serviceStorePath != _storePath) {
        stopService();
    }
}

std::optional<std::filesystem::path> LasagnaServiceManager::surfaceModel(const std::string& surfaceUuid) const
{
    const auto session = _storeSession.lock();
    if (!session || !_storeAttachment) {
        return std::nullopt;
    }
    const auto model = session->artifactDirectory(*_storeAttachment, surfaceUuid, false) / "model.pt";
    std::error_code ec;
    return std::filesystem::is_regular_file(model, ec) ? std::optional(model) : std::nullopt;
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
    // The service reads the model from the surface's artifact directory.
    if (const QString error = storeError(); !error.isEmpty()) {
        emit visExportError(error);
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
        const QString msg = obj["error"].toString();
        emit optimizationError(msg);
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
        emit jobsUpdated(_lastJobs);
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
    emit jobsUpdated(_lastJobs);

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
            QStringList surfaces;
            for (const QJsonValue& surface : obj[QStringLiteral("output_surfaces")].toArray()) {
                surfaces.append(surface.toString());
            }
            // An external service pushed them to the hub: pull them now.
            if (const auto session = _storeSession.lock(); _isExternal && session && _storeAttachment) {
                session->syncHub(*_storeAttachment);
            }
            emit jobFinished(jobId, surfaces);
            emit optimizationFinished(surfaces);
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
