#include "FiberFollowService.hpp"

#include "PythonExecutable.hpp"
#include "SocketRequest.hpp"
#include "VCSettings.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QStandardPaths>
#include <QProcessEnvironment>
#include <QThread>
#include <QUuid>

#include <stdexcept>
#include <cmath>

namespace {
constexpr int kStartTimeoutMs = 10 * 60 * 1000;  // first start compiles kernels
constexpr int kRequestTimeoutSeconds = 600;
}

FiberFollowService& FiberFollowService::instance()
{
    static FiberFollowService service;
    return service;
}

FiberFollowService::FiberFollowService(QObject* parent)
    : QObject(parent)
{
}

FiberFollowService::~FiberFollowService()
{
    stop();
}

bool FiberFollowService::isRunning() const
{
    return _process && _process->state() == QProcess::Running && !socketPath().empty();
}

std::string FiberFollowService::socketPath() const
{
    std::lock_guard lock(_mutex);
    return _socketPath;
}

void FiberFollowService::stop()
{
    {
        std::lock_guard lock(_mutex);
        _socketPath.clear();
    }
    if (_process) {
        _process->disconnect(this);
        if (_process->state() != QProcess::NotRunning) {
            _process->terminate();
            if (!_process->waitForFinished(5000)) {
                _process->kill();
                _process->waitForFinished(2000);
            }
        }
        _process.reset();
    }
}

bool FiberFollowService::ensureRunning(const QString& checkpoint, const QString& python, QString* error,
                                      const QString& volumeSource, const QString& volumeLocator)
{
    const auto fail = [error](const QString& message) {
        if (error) {
            *error = message;
        }
        return false;
    };
    const QString checkpointPath = checkpoint.trimmed().isEmpty()
        ? QString() : QFileInfo(checkpoint).absoluteFilePath();
    if (!checkpointPath.isEmpty() && !QFileInfo(checkpointPath).isFile()) {
        return fail(tr("The fiber-follow checkpoint does not exist: %1. Choose a checkpoint or use the default "
                       "Hugging Face model (Line Annotation menu → Fiber follow model).")
                        .arg(checkpointPath));
    }
    const QString interpreter = vc3d::pythonExecutableForServices(python);
    if (isRunning() && _checkpoint == checkpointPath && _python == interpreter &&
        _volumeSource == volumeSource && _volumeLocator == volumeLocator) {
        return true;
    }
    stop();

    const QString socket = QDir(QStandardPaths::writableLocation(QStandardPaths::TempLocation))
        .filePath(QStringLiteral("vc3d_fiber_follow_%1.sock")
                      .arg(QUuid::createUuid().toString(QUuid::WithoutBraces).left(8)));
    _process = std::make_unique<QProcess>();
    auto environment = QProcessEnvironment::systemEnvironment();
    // The native Python chunk reader uses the same configuration, codecs and
    // persistent cache as the viewer. Keep this opt-in scoped to the service.
    environment.insert(QStringLiteral("VC3D_CONFIG_DIR"), QFileInfo(vc3d::settingsFilePath()).absolutePath());
    environment.insert(QStringLiteral("VESUVIUS_CT_READER"), QStringLiteral("vc3d"));
    environment.insert(QStringLiteral("VESUVIUS_VC3D_VOLUME_SOURCE"), volumeSource);
    environment.insert(QStringLiteral("VESUVIUS_VC3D_VOLUME_LOCATOR"), volumeLocator);
    // Source builds ship matching bindings next to bin/. Prefer them over an
    // older copy installed in the selected Python environment.
    const QString bindings = QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(QStringLiteral("../python"));
    if (QFileInfo::exists(QDir(bindings).filePath(QStringLiteral("vc/__init__.py")))) {
        const QString existing = environment.value(QStringLiteral("PYTHONPATH"));
        environment.insert(QStringLiteral("PYTHONPATH"), existing.isEmpty()
            ? bindings : bindings + QDir::listSeparator() + existing);
    }
    environment.insert(QStringLiteral("PYTHONUNBUFFERED"), QStringLiteral("1"));
    _process->setProcessEnvironment(environment);
    _process->setProcessChannelMode(QProcess::MergedChannels);
    _output.clear();
    connect(_process.get(), &QProcess::readyReadStandardOutput, this, [this]() {
        const QString text = QString::fromUtf8(_process->readAllStandardOutput());
        _output += text;
        for (const auto& line : text.split('\n', Qt::SkipEmptyParts)) {
            emit statusMessage(line.trimmed());
        }
    });
    QStringList arguments{
        QStringLiteral("-m"), QStringLiteral("vesuvius.neural_tracing.tracers.fibers.tracing.service"),
        QStringLiteral("--socket"), socket,
        QStringLiteral("--parent-pid"), QString::number(QCoreApplication::applicationPid())};
    if (!checkpointPath.isEmpty()) {
        arguments << QStringLiteral("--checkpoint") << checkpointPath;
    }
    emit statusMessage(tr("Starting the fiber-follow model with %1…").arg(interpreter));
    _process->start(interpreter, arguments);
    if (!_process->waitForStarted(30000)) {
        const QString message = tr("Could not start %1: %2").arg(interpreter, _process->errorString());
        _process.reset();
        return fail(message);
    }
    QElapsedTimer timer;
    timer.start();
    while (!_output.contains(QStringLiteral("listening on"))) {
        if (_process->state() == QProcess::NotRunning) {
            const QString log = _output.right(2000);
            _process.reset();
            return fail(tr("The fiber-follow service exited during startup:\n%1").arg(log));
        }
        if (timer.elapsed() > kStartTimeoutMs) {
            stop();
            return fail(tr("The fiber-follow service did not start within 10 minutes."));
        }
        _process->waitForReadyRead(100);
        QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents, 50);
    }
    connect(_process.get(), &QProcess::finished, this, [this](int code, QProcess::ExitStatus) {
        {
            std::lock_guard lock(_mutex);
            _socketPath.clear();
        }
        emit statusMessage(tr("The fiber-follow service stopped (exit %1).").arg(code));
    });
    _checkpoint = checkpointPath;
    _python = interpreter;
    _volumeSource = volumeSource;
    _volumeLocator = volumeLocator;
    {
        std::lock_guard lock(_mutex);
        _socketPath = socket.toStdString();
    }
    emit statusMessage(tr("Fiber-follow model ready: %1").arg(checkpointPath.isEmpty()
        ? tr("default (Hugging Face)") : QFileInfo(checkpointPath).fileName()));
    return true;
}

nlohmann::json FiberFollowService::request(const nlohmann::json& request) const
{
    const std::string socket = socketPath();
    if (socket.empty()) {
        throw std::runtime_error("the fiber-follow service is not running");
    }
    auto reply = nlohmann::json::parse(
        vc3d::socketRequestLine(socket, request.dump(), kRequestTimeoutSeconds, "fiber-follow"));
    if (!reply.value("ok", false)) {
        throw std::runtime_error("fiber-follow: " + reply.value("error", std::string("request failed")));
    }
    return reply;
}

vc::lasagna::LineModel FiberFollowService::lineModel(
    const std::vector<cv::Vec3d>& points, double baseUm) const
{
    if (points.size() < 2 || !std::isfinite(baseUm) || baseUm <= 0)
        throw std::runtime_error("Model normals need a line and a valid volume voxel size.");
    auto positions = nlohmann::json::array();
    for (const auto& p : points) positions.push_back({p[0], p[1], p[2]});
    const auto reply = request({{"op", "normals"}, {"points", positions}, {"base_um", baseUm}});
    const auto& normals = reply.at("normals");
    if (normals.size() != points.size())
        throw std::runtime_error("Fiber follow returned the wrong number of normals.");
    vc::lasagna::LineModel line;
    for (size_t i = 0; i < points.size(); ++i) {
        const auto& n = normals.at(i);
        cv::Vec3d normal{n.at(0).get<double>(), n.at(1).get<double>(), n.at(2).get<double>()};
        const double length = cv::norm(normal);
        if (!std::isfinite(length) || length < 1e-8)
            throw std::runtime_error("Fiber follow returned an invalid sheet normal.");
        vc::lasagna::LinePoint point;
        point.position = points[i];
        point.sampledNormal.normal = normal / length;
        point.sampledNormal.valid = point.valid = true;
        line.points.push_back(point);
    }
    line.displayFrameAnchorIndex = static_cast<int>(points.size() / 2);
    return line;
}
