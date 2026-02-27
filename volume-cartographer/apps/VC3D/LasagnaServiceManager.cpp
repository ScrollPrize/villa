#include "LasagnaServiceManager.hpp"

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

// Avahi client library for mDNS service discovery
#ifdef Q_OS_LINUX
#include <avahi-client/client.h>
#include <avahi-client/lookup.h>
#include <avahi-common/error.h>
#include <avahi-common/malloc.h>
#include <avahi-common/simple-watch.h>
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

bool LasagnaServiceManager::startService(const QString& pythonPath)
{
    _lastError.clear();
    _serviceReady = false;
    _port = 0;

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
            emit statusMessage(tr("Lasagna service ready on port %1").arg(_port));
            emit serviceStarted();
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

    emit serviceStopped();
}

bool LasagnaServiceManager::isRunning() const
{
    if (_isExternal) {
        return _serviceReady;
    }
    return _process && _process->state() == QProcess::Running && _serviceReady;
}

// ---------------------------------------------------------------------------
// Process I/O handlers
// ---------------------------------------------------------------------------

void LasagnaServiceManager::handleProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    std::cout << "Lasagna service finished with exit code " << exitCode << std::endl;

    if (exitStatus == QProcess::CrashExit) {
        _lastError = tr("Lasagna service crashed");
    } else if (exitCode != 0) {
        _lastError = tr("Lasagna service exited with code %1").arg(exitCode);
    }

    _serviceReady = false;
    _pollTimer->stop();
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

    QUrl url(QStringLiteral("%1/optimize").arg(baseUrl()));
    QNetworkRequest req(url);
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    QByteArray body = QJsonDocument(config).toJson(QJsonDocument::Compact);

    QNetworkReply* reply = _nam->post(req, body);
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        handleOptimizeReply(reply);
    });
}

void LasagnaServiceManager::stopOptimization()
{
    if (!isRunning()) return;

    QUrl url(QStringLiteral("%1/stop").arg(baseUrl()));
    QNetworkRequest req(url);
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    _nam->post(req, QByteArray("{}"));
}

void LasagnaServiceManager::handleOptimizeReply(QNetworkReply* reply)
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

void LasagnaServiceManager::pollStatus()
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

void LasagnaServiceManager::handleStatusReply(QNetworkReply* reply)
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
    double stageProgress = obj["stage_progress"].toDouble();
    double overallProgress = obj["overall_progress"].toDouble();
    QString stageName = obj["stage_name"].toString();

    if (state == "running") {
        emit optimizationProgress(stage, step, totalSteps, loss,
                                  stageProgress, overallProgress, stageName);
    } else if (state == "finished") {
        _optimizationRunning = false;
        _pollTimer->stop();
        if (!_localOutputDir.isEmpty()) {
            // Download results archive and unpack locally
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
// Results download
// ---------------------------------------------------------------------------

void LasagnaServiceManager::downloadResults()
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
        std::cout << "[lasagna] downloaded results archive ("
                  << data.size() << " bytes)" << std::endl;

        // Write tar.gz to a temp file, then extract into _localOutputDir
        QString tarPath = _localOutputDir + QStringLiteral("/.lasagna_results.tar.gz");
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

        std::cout << "[lasagna] results unpacked to "
                  << _localOutputDir.toStdString() << std::endl;
        emit optimizationFinished(_localOutputDir);
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
#ifdef Q_OS_LINUX
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
#endif

    return result;
}

void LasagnaServiceManager::fetchDatasets()
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
