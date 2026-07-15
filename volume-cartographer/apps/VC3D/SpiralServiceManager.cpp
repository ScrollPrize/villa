#include "SpiralServiceManager.hpp"

#include "VCSettings.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QProcessEnvironment>
#include <QRandomGenerator>
#include <QRegularExpression>
#include <QSettings>
#include <QStandardPaths>
#include <QUrl>

namespace {
constexpr int kPollMs = 500;
}

SpiralServiceManager::SpiralServiceManager(QObject* parent) : QObject(parent)
{
    _process = new QProcess(this);
    _process->setProcessChannelMode(QProcess::SeparateChannels);
    _network = new QNetworkAccessManager(this);
    _poll = new QTimer(this);
    _poll->setInterval(kPollMs);
    connect(_poll, &QTimer::timeout, this, &SpiralServiceManager::pollStatus);
    connect(_process, &QProcess::readyReadStandardOutput, this, [this]() {
        const QString output = QString::fromUtf8(_process->readAllStandardOutput());
        for (const QString& line : output.split('\n', Qt::SkipEmptyParts)) {
            emit logMessage(line);
            const QRegularExpressionMatch match = QRegularExpression(
                QStringLiteral("^SPIRAL_SERVICE_READY port=(\\d+) api_version=(\\d+)$")).match(line.trimmed());
            if (match.hasMatch() && match.captured(2).toInt() == 1) {
                _port = match.captured(1).toInt();
                _ready = true;
                _poll->start();
                emit serviceStateChanged(tr("Ready"));
                pollStatus();
            }
        }
    });
    connect(_process, &QProcess::readyReadStandardError, this, [this]() {
        const QString output = QString::fromUtf8(_process->readAllStandardError());
        for (const QString& line : output.split('\n', Qt::SkipEmptyParts)) emit logMessage(line);
    });
    connect(_process, &QProcess::errorOccurred, this, [this](QProcess::ProcessError) {
        emit errorOccurred(_process->errorString());
    });
    connect(_process, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            [this](int code, QProcess::ExitStatus) {
                _ready = false;
                _port = 0;
                _poll->stop();
                emit serviceStateChanged(tr("Stopped"));
                if (code != 0) emit errorOccurred(tr("Spiral service exited with code %1").arg(code));
            });
}

SpiralServiceManager::~SpiralServiceManager() { stopService(); }

QString SpiralServiceManager::findPython() const
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const QStringList candidates{
        settings.value(QStringLiteral("spiral/python")).toString(),
        qEnvironmentVariable("SPIRAL_PYTHON"),
        qEnvironmentVariable("PYTHON_EXECUTABLE"),
        QDir(qEnvironmentVariable("CONDA_PREFIX")).filePath(QStringLiteral("bin/python")),
        QStandardPaths::findExecutable(QStringLiteral("python3")),
        QStandardPaths::findExecutable(QStringLiteral("python")),
    };
    for (const QString& candidate : candidates)
        if (!candidate.isEmpty() && QFileInfo(candidate).isExecutable()) return QFileInfo(candidate).absoluteFilePath();
    return {};
}

QString SpiralServiceManager::findService() const
{
    const QString app = QCoreApplication::applicationDirPath();
    const QStringList candidates{
        qEnvironmentVariable("SPIRAL_SERVICE_PATH"),
        QDir::current().filePath(QStringLiteral("scripts/spiral/spiral_service.py")),
        QDir(app).filePath(QStringLiteral("../../scripts/spiral/spiral_service.py")),
        QDir(app).filePath(QStringLiteral("../../../scripts/spiral/spiral_service.py")),
        QDir(app).filePath(QStringLiteral("../share/volume-cartographer/spiral/spiral_service.py")),
    };
    for (const QString& candidate : candidates)
        if (!candidate.isEmpty() && QFileInfo(candidate).isFile()) return QFileInfo(candidate).absoluteFilePath();
    return {};
}

void SpiralServiceManager::ensureStarted()
{
    if (_ready || _process->state() != QProcess::NotRunning) return;
    const QString python = findPython();
    const QString service = findService();
    if (python.isEmpty() || service.isEmpty()) {
        emit errorOccurred(tr("Cannot find the Spiral Python interpreter or spiral_service.py. Set SPIRAL_PYTHON and SPIRAL_SERVICE_PATH."));
        return;
    }
    _nonce = QString::number(QRandomGenerator::global()->generate64(), 16)
        + QString::number(QRandomGenerator::global()->generate64(), 16);
    _lastStatusGeneration = -1;
    _lastPreviewGeneration = -1;
    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    const QString moduleDir = QFileInfo(service).absolutePath();
    const QString oldPythonPath = environment.value(QStringLiteral("PYTHONPATH"));
    environment.insert(QStringLiteral("PYTHONPATH"), oldPythonPath.isEmpty() ? moduleDir : moduleDir + QDir::listSeparator() + oldPythonPath);
    environment.insert(QStringLiteral("PYTHONUNBUFFERED"), QStringLiteral("1"));
    _process->setProcessEnvironment(environment);
    emit serviceStateChanged(tr("Starting"));
    _process->start(python, {service, QStringLiteral("--nonce"), _nonce,
                             QStringLiteral("--parent-pid"), QString::number(QCoreApplication::applicationPid())});
}

void SpiralServiceManager::stopService()
{
    _poll->stop();
    _ready = false;
    if (_process->state() == QProcess::NotRunning) return;
    _process->terminate();
    if (!_process->waitForFinished(2000)) {
        _process->kill();
        _process->waitForFinished(1000);
    }
}

QString SpiralServiceManager::commandId()
{
    return QStringLiteral("vc3d-%1-%2").arg(QCoreApplication::applicationPid()).arg(++_commandCounter);
}

void SpiralServiceManager::resolveDataset(const QString& root)
{
    if (!_ready) { ensureStarted(); emit errorOccurred(tr("Spiral service is starting; retry dataset resolution when Ready.")); return; }
    post(QStringLiteral("/dataset/resolve"), {{QStringLiteral("dataset_root"), root}},
         [this](const QJsonObject& value) { emit datasetResolved(value); });
}

void SpiralServiceManager::loadSession(QJsonObject request)
{
    const QJsonObject inputPaths = request.value(QStringLiteral("paths")).toObject();
    request[QStringLiteral("command_id")] = commandId();
    post(QStringLiteral("/session/load"), request,
         [this, inputPaths](const QJsonObject& response) {
             emit sessionAccepted(inputPaths,
                                  response.value(QStringLiteral("session_generation")).toInteger());
         });
}

void SpiralServiceManager::runIterations(int iterations)
{
    post(QStringLiteral("/session/run"), {{QStringLiteral("command_id"), commandId()},
                                           {QStringLiteral("iterations"), iterations}});
}

void SpiralServiceManager::stopAfterIteration()
{
    post(QStringLiteral("/session/stop"), {{QStringLiteral("command_id"), commandId()}});
}

void SpiralServiceManager::saveCheckpoint(const QString& path)
{
    post(QStringLiteral("/session/save-checkpoint"), {{QStringLiteral("command_id"), commandId()},
                                                       {QStringLiteral("path"), path}});
}

void SpiralServiceManager::deleteSession()
{
    // DELETE with a JSON body is not provided by the small helper yet; session
    // teardown is guaranteed when the process is stopped with the workspace.
    stopService();
}

void SpiralServiceManager::post(const QString& path, QJsonObject body,
                                std::function<void(const QJsonObject&)> success)
{
    if (!_ready) { ensureStarted(); emit errorOccurred(tr("Spiral service is not ready")); return; }
    QNetworkRequest request(QUrl(QStringLiteral("http://127.0.0.1:%1%2").arg(_port).arg(path)));
    request.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
    request.setRawHeader("X-Spiral-Nonce", _nonce.toUtf8());
    auto* reply = _network->post(request, QJsonDocument(body).toJson(QJsonDocument::Compact));
    connect(reply, &QNetworkReply::finished, this, [this, reply, success]() { handleReply(reply, success); });
}

void SpiralServiceManager::get(const QString& path, std::function<void(const QJsonObject&)> success)
{
    if (!_ready) return;
    QNetworkRequest request(QUrl(QStringLiteral("http://127.0.0.1:%1%2").arg(_port).arg(path)));
    request.setRawHeader("X-Spiral-Nonce", _nonce.toUtf8());
    auto* reply = _network->get(request);
    connect(reply, &QNetworkReply::finished, this, [this, reply, success]() { handleReply(reply, success); });
}

void SpiralServiceManager::handleReply(QNetworkReply* reply, std::function<void(const QJsonObject&)> success)
{
    const QByteArray bytes = reply->readAll();
    const int http = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
    const QString networkError = reply->errorString();
    const QJsonDocument document = QJsonDocument::fromJson(bytes);
    reply->deleteLater();
    if (!document.isObject() || http >= 400) {
        QString message = document.object().value(QStringLiteral("error")).toString(
            networkError.isEmpty() ? tr("Invalid Spiral service response") : networkError);
        const QJsonArray details = document.object().value(QStringLiteral("details")).toArray();
        QStringList detailLines;
        for (const QJsonValue& value : details) {
            const QJsonObject detail = value.toObject();
            const QString field = detail.value(QStringLiteral("field")).toString();
            const QString description = detail.value(QStringLiteral("message")).toString();
            if (!description.isEmpty())
                detailLines.push_back(field.isEmpty() ? description : QStringLiteral("%1: %2").arg(field, description));
        }
        if (!detailLines.isEmpty()) message += QStringLiteral("\n") + detailLines.join(QStringLiteral("\n"));
        emit errorOccurred(message);
        return;
    }
    if (success) success(document.object());
}

void SpiralServiceManager::pollStatus()
{
    get(QStringLiteral("/session/status"), [this](const QJsonObject& status) {
        const qint64 generation = status.value(QStringLiteral("generation")).toInteger(-1);
        if (generation < _lastStatusGeneration) return;
        _lastStatusGeneration = generation;
        emit sessionStatusChanged(status);
        const qint64 previewGeneration = status.value(QStringLiteral("preview_generation")).toInteger(-1);
        const qint64 sessionGeneration = status.value(QStringLiteral("session_generation")).toInteger(-1);
        const qint64 handoffGeneration = sessionGeneration < 0 || previewGeneration < 0
            ? -1
            : (sessionGeneration << 32) | (previewGeneration & 0xffffffffLL);
        const QString manifest = status.value(QStringLiteral("preview_manifest_path")).toString();
        if (handoffGeneration > _lastPreviewGeneration && !manifest.isEmpty()) {
            _lastPreviewGeneration = handoffGeneration;
            emit previewAvailable(manifest, handoffGeneration);
        }
    });
}
