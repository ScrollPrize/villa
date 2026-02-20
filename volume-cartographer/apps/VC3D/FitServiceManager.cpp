#include "FitServiceManager.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QRegularExpression>
#include <QSet>
#include <QUrl>

#include <iostream>

#ifdef Q_OS_UNIX
#include <signal.h>
#endif

namespace
{
constexpr int kServiceStartTimeoutMs = 60000;  // 1 minute (no torch compile)
constexpr int kServiceStopTimeoutMs = 5000;
constexpr int kPollIntervalMs = 500;

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

QString findFitServiceScript()
{
    QString appDir = QCoreApplication::applicationDirPath();
    QStringList searchPaths = {
        // Development: build dir is volume-cartographer/build/bin/
        QDir(appDir).filePath("../../vesuvius/src/vesuvius/exps_2d_model/fit_service.py"),
        QDir(appDir).filePath("../../../vesuvius/src/vesuvius/exps_2d_model/fit_service.py"),
        // Installed
        QDir(appDir).filePath("../share/vesuvius/exps_2d_model/fit_service.py"),
        // Environment variable
        qEnvironmentVariable("FIT_SERVICE_PATH"),
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

FitServiceManager& FitServiceManager::instance()
{
    static FitServiceManager inst;
    return inst;
}

FitServiceManager::FitServiceManager(QObject* parent)
    : QObject(parent)
{
    _nam = new QNetworkAccessManager(this);
    _pollTimer = new QTimer(this);
    _pollTimer->setInterval(kPollIntervalMs);
    connect(_pollTimer, &QTimer::timeout, this, &FitServiceManager::pollStatus);
}

FitServiceManager::~FitServiceManager()
{
    stopService();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

QString FitServiceManager::baseUrl() const
{
    return QStringLiteral("http://%1:%2").arg(_host).arg(_port);
}

// ---------------------------------------------------------------------------
// Service lifecycle
// ---------------------------------------------------------------------------

bool FitServiceManager::ensureServiceRunning(const QString& pythonPath)
{
    if (_isExternal && _serviceReady) {
        return true;
    }
    if (_process && _process->state() == QProcess::Running && _serviceReady) {
        return true;
    }
    return startService(pythonPath);
}

void FitServiceManager::connectToExternal(const QString& host, int port)
{
    // Stop any existing internal service first
    if (_process) {
        stopService();
    }

    _isExternal = true;
    _host = host;
    _port = port;
    _lastError.clear();
    _serviceReady = false;

    emit statusMessage(tr("Connecting to external service at %1:%2...").arg(host).arg(port));

    // Ping GET /health
    QUrl url(QStringLiteral("%1/health").arg(baseUrl()));
    QNetworkRequest req(url);

    QNetworkReply* reply = _nam->get(req);
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();

        if (reply->error() != QNetworkReply::NoError) {
            _lastError = tr("Cannot reach external service: %1").arg(reply->errorString());
            _serviceReady = false;
            _isExternal = false;
            emit serviceError(_lastError);
            return;
        }

        _serviceReady = true;
        emit statusMessage(tr("Connected to external service on %1:%2").arg(_host).arg(_port));
        emit serviceStarted();
    });
}

bool FitServiceManager::startService(const QString& pythonPath)
{
    _lastError.clear();
    _serviceReady = false;
    _port = 0;

    QString scriptPath = findFitServiceScript();
    if (scriptPath.isEmpty()) {
        _lastError = tr("Could not find fit_service.py. Set FIT_SERVICE_PATH environment variable.");
        emit serviceError(_lastError);
        return false;
    }

    _process = std::make_unique<QProcess>();
    _process->setProcessChannelMode(QProcess::SeparateChannels);

    connect(_process.get(), QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &FitServiceManager::handleProcessFinished);
    connect(_process.get(), &QProcess::errorOccurred,
            this, &FitServiceManager::handleProcessError);
    connect(_process.get(), &QProcess::readyReadStandardOutput,
            this, &FitServiceManager::handleReadyReadStdout);
    connect(_process.get(), &QProcess::readyReadStandardError,
            this, &FitServiceManager::handleReadyReadStderr);

    // Set PYTHONPATH so fit_service.py can import sibling modules (fit, optimizer, etc.)
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

    emit statusMessage(tr("Starting fit optimizer service..."));
    std::cout << "Starting fit service: " << python.toStdString();
    for (const QString& arg : args) {
        std::cout << " " << arg.toStdString();
    }
    std::cout << std::endl;

    _process->start(python, args);

    if (!_process->waitForStarted(5000)) {
        _lastError = tr("Failed to start fit optimizer service process");
        emit serviceError(_lastError);
        _process.reset();
        return false;
    }

    emit statusMessage(tr("Waiting for fit optimizer service to initialize..."));

    QElapsedTimer timer;
    timer.start();
    while (timer.elapsed() < kServiceStartTimeoutMs) {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 100);

        if (_serviceReady) {
            emit statusMessage(tr("Fit optimizer service ready on port %1").arg(_port));
            emit serviceStarted();
            return true;
        }

        if (!_process || _process->state() != QProcess::Running) {
            if (_lastError.isEmpty()) {
                _lastError = tr("Fit optimizer service terminated unexpectedly");
            }
            emit serviceError(_lastError);
            return false;
        }
    }

    _lastError = tr("Fit optimizer service startup timed out");
    emit serviceError(_lastError);
    stopService();
    return false;
}

void FitServiceManager::stopService()
{
    _pollTimer->stop();

    if (_isExternal) {
        // External mode: just reset state, don't terminate any process
        _serviceReady = false;
        _optimizationRunning = false;
        _isExternal = false;
        _host = QStringLiteral("127.0.0.1");
        _port = 0;
        emit serviceStopped();
        return;
    }

    if (!_process) {
        return;
    }

    std::cout << "Stopping fit optimizer service..." << std::endl;

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

    emit serviceStopped();
}

bool FitServiceManager::isRunning() const
{
    if (_isExternal) {
        return _serviceReady;
    }
    return _process && _process->state() == QProcess::Running && _serviceReady;
}

// ---------------------------------------------------------------------------
// Process I/O handlers
// ---------------------------------------------------------------------------

void FitServiceManager::handleProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    std::cout << "Fit optimizer service finished with exit code " << exitCode << std::endl;

    if (exitStatus == QProcess::CrashExit) {
        _lastError = tr("Fit optimizer service crashed");
    } else if (exitCode != 0) {
        _lastError = tr("Fit optimizer service exited with code %1").arg(exitCode);
    }

    _serviceReady = false;
    _pollTimer->stop();
    emit serviceStopped();
}

void FitServiceManager::handleProcessError(QProcess::ProcessError error)
{
    QString errorStr;
    switch (error) {
    case QProcess::FailedToStart:
        errorStr = tr("Failed to start fit optimizer service");
        break;
    case QProcess::Crashed:
        errorStr = tr("Fit optimizer service crashed");
        break;
    default:
        errorStr = tr("Fit optimizer service error");
        break;
    }

    _lastError = errorStr;
    std::cerr << "Fit service error: " << errorStr.toStdString() << std::endl;
    emit serviceError(errorStr);
}

void FitServiceManager::handleReadyReadStdout()
{
    if (!_process) return;

    QString output = QString::fromUtf8(_process->readAllStandardOutput());
    std::cout << "[fit-service] " << output.toStdString();

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

void FitServiceManager::handleReadyReadStderr()
{
    if (!_process) return;

    QString error = QString::fromUtf8(_process->readAllStandardError());
    std::cerr << "[fit-service] " << error.toStdString();

    if (!error.trimmed().isEmpty() && !_serviceReady) {
        if (_lastError.isEmpty() && error.contains("error", Qt::CaseInsensitive)) {
            _lastError = error.trimmed();
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP communication
// ---------------------------------------------------------------------------

void FitServiceManager::startOptimization(const QJsonObject& config,
                                           const QString& localOutputDir)
{
    if (!isRunning()) {
        emit optimizationError(tr("Fit optimizer service is not running"));
        return;
    }

    _localOutputDir = localOutputDir;

    QUrl url(QStringLiteral("%1/optimize").arg(baseUrl()));
    QNetworkRequest req(url);
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    QByteArray body = QJsonDocument(config).toJson(QJsonDocument::Compact);

    QNetworkReply* reply = _nam->post(req, body);
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        handleOptimizeReply(reply);
    });
}

void FitServiceManager::stopOptimization()
{
    if (!isRunning()) return;

    QUrl url(QStringLiteral("%1/stop").arg(baseUrl()));
    QNetworkRequest req(url);
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    _nam->post(req, QByteArray("{}"));
}

void FitServiceManager::handleOptimizeReply(QNetworkReply* reply)
{
    reply->deleteLater();

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

    _optimizationRunning = true;
    _pollTimer->start();
    emit optimizationStarted();
}

void FitServiceManager::pollStatus()
{
    if (!isRunning()) {
        _pollTimer->stop();
        return;
    }

    QUrl url(QStringLiteral("%1/status").arg(baseUrl()));
    QNetworkRequest req(url);

    QNetworkReply* reply = _nam->get(req);
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        handleStatusReply(reply);
    });
}

void FitServiceManager::handleStatusReply(QNetworkReply* reply)
{
    reply->deleteLater();

    if (reply->error() != QNetworkReply::NoError) {
        return;  // Transient network error, will retry next poll
    }

    QJsonDocument doc = QJsonDocument::fromJson(reply->readAll());
    QJsonObject obj = doc.object();

    QString state = obj["state"].toString();
    QString stage = obj["stage"].toString();
    int step = obj["step"].toInt();
    int totalSteps = obj["total_steps"].toInt();
    double loss = obj["loss"].toDouble();

    if (state == "running") {
        emit optimizationProgress(stage, step, totalSteps, loss);
    } else if (state == "finished") {
        _optimizationRunning = false;
        _pollTimer->stop();
        if (_isExternal && !_localOutputDir.isEmpty()) {
            // External mode: download results archive and unpack locally
            downloadResults();
        } else {
            QString outputDir = obj["output_dir"].toString();
            emit optimizationFinished(outputDir);
        }
    } else if (state == "error") {
        _optimizationRunning = false;
        _pollTimer->stop();
        QString errorMsg = obj["error"].toString();
        emit optimizationError(errorMsg.isEmpty() ? tr("Unknown error") : errorMsg);
    } else if (state == "idle" && _optimizationRunning) {
        // Unexpected idle while we think it's running — stop polling
        _optimizationRunning = false;
        _pollTimer->stop();
    }
}

// ---------------------------------------------------------------------------
// Results download (external mode)
// ---------------------------------------------------------------------------

void FitServiceManager::downloadResults()
{
    emit statusMessage(tr("Downloading results from external service..."));

    QUrl url(QStringLiteral("%1/results").arg(baseUrl()));
    QNetworkRequest req(url);

    QNetworkReply* reply = _nam->get(req);
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();

        if (reply->error() != QNetworkReply::NoError) {
            emit optimizationError(
                tr("Failed to download results: %1").arg(reply->errorString()));
            return;
        }

        QByteArray data = reply->readAll();
        std::cout << "[fit-optimizer] downloaded results archive ("
                  << data.size() << " bytes)" << std::endl;

        // Write tar.gz to a temp file, then extract into _localOutputDir
        QString tarPath = _localOutputDir + QStringLiteral("/.fit_results.tar.gz");
        QFile tarFile(tarPath);
        if (!tarFile.open(QIODevice::WriteOnly)) {
            emit optimizationError(tr("Cannot write temp file: %1").arg(tarPath));
            return;
        }
        tarFile.write(data);
        tarFile.close();

        // Extract using tar
        QProcess tar;
        tar.setWorkingDirectory(_localOutputDir);
        tar.start(QStringLiteral("tar"),
                  {QStringLiteral("xzf"), tarPath});
        if (!tar.waitForFinished(30000)) {
            QFile::remove(tarPath);
            emit optimizationError(tr("tar extraction timed out"));
            return;
        }
        QFile::remove(tarPath);

        if (tar.exitCode() != 0) {
            QString err = QString::fromUtf8(tar.readAllStandardError());
            emit optimizationError(tr("tar extraction failed: %1").arg(err));
            return;
        }

        std::cout << "[fit-optimizer] results unpacked to "
                  << _localOutputDir.toStdString() << std::endl;
        emit optimizationFinished(_localOutputDir);
    });
}

// ---------------------------------------------------------------------------
// Service discovery
// ---------------------------------------------------------------------------

QJsonArray FitServiceManager::discoverServices()
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

    // --- mDNS discovery via avahi-browse ---
    QProcess proc;
    proc.start(QStringLiteral("avahi-browse"),
               {QStringLiteral("-r"), QStringLiteral("-t"),
                QStringLiteral("--parsable"), QStringLiteral("_fitoptimizer._tcp")});
    if (proc.waitForFinished(5000)) {
        // Parsable output format (resolved lines start with '='):
        // =;interface;protocol;name;type;domain;hostname;address;port;txt1 txt2 ...
        QString output = QString::fromUtf8(proc.readAllStandardOutput());
        for (const QString& line : output.split('\n')) {
            if (!line.startsWith('=')) {
                continue;
            }
            QStringList fields = line.split(';');
            if (fields.size() < 9) {
                continue;
            }

            QString host = fields[7];
            int port = fields[8].toInt();
            QString key = QStringLiteral("%1:%2").arg(host).arg(port);
            if (seen.contains(key)) {
                continue;
            }
            seen.insert(key);

            QJsonObject obj;
            obj[QStringLiteral("host")] = host;
            obj[QStringLiteral("port")] = port;
            obj[QStringLiteral("name")] = fields[3];

            // Parse TXT records (field 9+, space-separated, each quoted)
            if (fields.size() > 9) {
                QString txtRaw = fields[9];
                for (const QString& record : txtRaw.split(' ')) {
                    QString r = record.trimmed();
                    if (r.startsWith('"') && r.endsWith('"')) {
                        r = r.mid(1, r.size() - 2);
                    }
                    if (r.startsWith(QStringLiteral("data_dir="))) {
                        obj[QStringLiteral("data_dir")] = r.mid(9);
                    } else if (r.startsWith(QStringLiteral("datasets="))) {
                        QJsonArray ds;
                        for (const QString& d : r.mid(9).split(',')) {
                            if (!d.isEmpty()) {
                                ds.append(d);
                            }
                        }
                        obj[QStringLiteral("datasets")] = ds;
                    }
                }
            }

            result.append(obj);
        }
    }

    return result;
}

void FitServiceManager::fetchDatasets()
{
    if (!isRunning()) {
        return;
    }

    QUrl url(QStringLiteral("%1/datasets").arg(baseUrl()));
    QNetworkRequest req(url);

    QNetworkReply* reply = _nam->get(req);
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();
        if (reply->error() != QNetworkReply::NoError) {
            return;
        }
        QJsonDocument doc = QJsonDocument::fromJson(reply->readAll());
        QJsonObject obj = doc.object();
        QJsonArray datasets = obj[QStringLiteral("datasets")].toArray();
        emit datasetsReceived(datasets);
    });
}
