#include "NeuralTraceServiceManager.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QStandardPaths>
#include <QThread>
#include <QUuid>
#include <QElapsedTimer>

#include <iostream>

namespace
{
constexpr int kServiceStartTimeoutMs = 300000; // 5 minutes for torch compilation
constexpr int kServiceStopTimeoutMs = 5000;    // 5 seconds to gracefully stop

QString findPythonExecutable()
{
    QStringList candidates;

    // Check for explicit PYTHON_EXECUTABLE override first
    QString envPython = qEnvironmentVariable("PYTHON_EXECUTABLE");
    if (!envPython.isEmpty()) {
        candidates.append(envPython);
    }

    // Check for active conda environment (CONDA_PREFIX is set when env is active)
    QString condaPrefix = qEnvironmentVariable("CONDA_PREFIX");
    if (!condaPrefix.isEmpty()) {
        candidates.append(QDir(condaPrefix).filePath("bin/python"));
        candidates.append(QDir(condaPrefix).filePath("bin/python3"));
    }

    // Check for miniconda in home directory
    QString home = QDir::homePath();
    candidates.append(QDir(home).filePath("miniconda3/bin/python"));
    candidates.append(QDir(home).filePath("miniconda3/bin/python3"));
    candidates.append(QDir(home).filePath("anaconda3/bin/python"));
    candidates.append(QDir(home).filePath("anaconda3/bin/python3"));

    // System Python as fallback
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

    return "python3"; // Default fallback
}

QString generateSocketPath()
{
    QString tempDir = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
    QString uuid = QUuid::createUuid().toString(QUuid::WithoutBraces).left(8);
    return QDir(tempDir).filePath(QString("neural_tracer_%1.sock").arg(uuid));
}
}

NeuralTraceServiceManager& NeuralTraceServiceManager::instance()
{
    static NeuralTraceServiceManager instance;
    return instance;
}

NeuralTraceServiceManager::NeuralTraceServiceManager(QObject* parent)
    : QObject(parent)
{
}

NeuralTraceServiceManager::~NeuralTraceServiceManager()
{
    stopService();
}

bool NeuralTraceServiceManager::ensureServiceRunning(const QString& checkpointPath,
                                                      const QString& volumeZarr,
                                                      int volumeScale,
                                                      const QString& pythonPath)
{
    // Check if already running with same config
    if (_process && _process->state() == QProcess::Running && _serviceReady) {
        if (_currentCheckpointPath == checkpointPath &&
            _currentVolumeZarr == volumeZarr &&
            _currentVolumeScale == volumeScale) {
            return true;
        }
        // Configuration changed, need to restart
        emit statusMessage(tr("Restarting neural trace service with new configuration..."));
        stopService();
    }

    return startService(checkpointPath, volumeZarr, volumeScale, pythonPath);
}

bool NeuralTraceServiceManager::startService(const QString& checkpointPath,
                                              const QString& volumeZarr,
                                              int volumeScale,
                                              const QString& pythonPath)
{
    _lastError.clear();
    _serviceReady = false;

    // Validate inputs
    if (checkpointPath.isEmpty()) {
        _lastError = tr("Checkpoint path is required");
        emit serviceError(_lastError);
        return false;
    }
    if (!QFile::exists(checkpointPath)) {
        _lastError = tr("Checkpoint file does not exist: %1").arg(checkpointPath);
        emit serviceError(_lastError);
        return false;
    }
    if (volumeZarr.isEmpty()) {
        _lastError = tr("Volume zarr path is required");
        emit serviceError(_lastError);
        return false;
    }
    if (!QDir(volumeZarr).exists()) {
        _lastError = tr("Volume zarr directory does not exist: %1").arg(volumeZarr);
        emit serviceError(_lastError);
        return false;
    }

    // Generate socket path
    _socketPath = generateSocketPath();

    // Remove any existing socket file
    if (QFile::exists(_socketPath)) {
        QFile::remove(_socketPath);
    }

    // Create process
    _process = std::make_unique<QProcess>();
    _process->setProcessChannelMode(QProcess::SeparateChannels);

    connect(_process.get(), QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &NeuralTraceServiceManager::handleProcessFinished);
    connect(_process.get(), &QProcess::errorOccurred,
            this, &NeuralTraceServiceManager::handleProcessError);
    connect(_process.get(), &QProcess::readyReadStandardOutput,
            this, &NeuralTraceServiceManager::handleReadyReadStandardOutput);
    connect(_process.get(), &QProcess::readyReadStandardError,
            this, &NeuralTraceServiceManager::handleReadyReadStandardError);

    // Find trace_service.py - look relative to the application
    QString appDir = QCoreApplication::applicationDirPath();
    QStringList searchPaths = {
        // Development paths
        QDir(appDir).filePath("../../vesuvius/src/vesuvius/neural_tracing/trace_service.py"),
        QDir(appDir).filePath("../../../vesuvius/src/vesuvius/neural_tracing/trace_service.py"),
        // Installed paths
        QDir(appDir).filePath("../share/vesuvius/neural_tracing/trace_service.py"),
        // Environment variable
        qEnvironmentVariable("NEURAL_TRACE_SERVICE_PATH"),
    };

    QString traceServicePath;
    for (const QString& path : searchPaths) {
        if (!path.isEmpty() && QFile::exists(path)) {
            traceServicePath = QFileInfo(path).absoluteFilePath();
            break;
        }
    }

    if (traceServicePath.isEmpty()) {
        _lastError = tr("Could not find trace_service.py. Set NEURAL_TRACE_SERVICE_PATH environment variable.");
        emit serviceError(_lastError);
        return false;
    }

    // Use provided Python path if specified, otherwise auto-detect
    QString python = pythonPath.isEmpty() ? findPythonExecutable() : pythonPath;

    // Set up environment with vesuvius/src in PYTHONPATH
    // trace_service.py is at vesuvius/src/vesuvius/neural_tracing/trace_service.py
    // so vesuvius/src is two directories up from the script's parent
    QDir traceServiceDir(QFileInfo(traceServicePath).absolutePath());
    traceServiceDir.cdUp();  // neural_tracing -> vesuvius
    traceServiceDir.cdUp();  // vesuvius -> src
    QString vesuviusSrcPath = traceServiceDir.absolutePath();

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    QString existingPythonPath = env.value("PYTHONPATH");
    if (existingPythonPath.isEmpty()) {
        env.insert("PYTHONPATH", vesuviusSrcPath);
    } else {
        env.insert("PYTHONPATH", vesuviusSrcPath + ":" + existingPythonPath);
    }
    _process->setProcessEnvironment(env);

    QStringList args = {
        traceServicePath,
        "--checkpoint_path", checkpointPath,
        "--volume_zarr", volumeZarr,
        "--volume_scale", QString::number(volumeScale),
        "--socket_path", _socketPath
    };

    emit statusMessage(tr("Starting neural trace service..."));
    std::cout << "Starting neural trace service: " << python.toStdString();
    for (const QString& arg : args) {
        std::cout << " " << arg.toStdString();
    }
    std::cout << std::endl;

    _process->start(python, args);

    if (!_process->waitForStarted(5000)) {
        _lastError = tr("Failed to start neural trace service process");
        emit serviceError(_lastError);
        _process.reset();
        return false;
    }

    // Wait for the service to be ready (socket file to appear and "listening" message)
    emit statusMessage(tr("Waiting for neural trace service to initialize (this may take a minute for torch compilation)..."));

    QElapsedTimer timer;
    timer.start();
    while (timer.elapsed() < kServiceStartTimeoutMs) {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 100);

        if (_serviceReady) {
            _currentCheckpointPath = checkpointPath;
            _currentVolumeZarr = volumeZarr;
            _currentVolumeScale = volumeScale;
            emit statusMessage(tr("Neural trace service ready"));
            emit serviceStarted();
            return true;
        }

        if (!_process || _process->state() != QProcess::Running) {
            if (_lastError.isEmpty()) {
                _lastError = tr("Neural trace service process terminated unexpectedly");
            }
            emit serviceError(_lastError);
            return false;
        }
    }

    _lastError = tr("Neural trace service startup timed out after %1 seconds").arg(kServiceStartTimeoutMs / 1000);
    emit serviceError(_lastError);
    stopService();
    return false;
}

void NeuralTraceServiceManager::stopService()
{
    if (!_process) {
        return;
    }

    std::cout << "Stopping neural trace service..." << std::endl;

    if (_process->state() == QProcess::Running) {
        _process->terminate();
        if (!_process->waitForFinished(kServiceStopTimeoutMs)) {
            _process->kill();
            _process->waitForFinished(1000);
        }
    }

    _process.reset();
    _serviceReady = false;

    // Clean up socket file
    if (!_socketPath.isEmpty() && QFile::exists(_socketPath)) {
        QFile::remove(_socketPath);
    }
    _socketPath.clear();

    _currentCheckpointPath.clear();
    _currentVolumeZarr.clear();
    _currentVolumeScale = 0;

    emit serviceStopped();
}

bool NeuralTraceServiceManager::isRunning() const
{
    return _process && _process->state() == QProcess::Running && _serviceReady;
}

QString NeuralTraceServiceManager::socketPath() const
{
    return _serviceReady ? _socketPath : QString();
}

void NeuralTraceServiceManager::handleProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    std::cout << "Neural trace service finished with exit code " << exitCode << std::endl;

    if (exitStatus == QProcess::CrashExit) {
        _lastError = tr("Neural trace service crashed");
    } else if (exitCode != 0) {
        _lastError = tr("Neural trace service exited with code %1").arg(exitCode);
    }

    _serviceReady = false;
    emit serviceStopped();
}

void NeuralTraceServiceManager::handleProcessError(QProcess::ProcessError error)
{
    QString errorStr;
    switch (error) {
    case QProcess::FailedToStart:
        errorStr = tr("Failed to start neural trace service");
        break;
    case QProcess::Crashed:
        errorStr = tr("Neural trace service crashed");
        break;
    case QProcess::Timedout:
        errorStr = tr("Neural trace service timed out");
        break;
    case QProcess::WriteError:
        errorStr = tr("Failed to write to neural trace service");
        break;
    case QProcess::ReadError:
        errorStr = tr("Failed to read from neural trace service");
        break;
    default:
        errorStr = tr("Unknown neural trace service error");
        break;
    }

    _lastError = errorStr;
    std::cerr << "Neural trace service error: " << errorStr.toStdString() << std::endl;
    emit serviceError(errorStr);
}

void NeuralTraceServiceManager::handleReadyReadStandardOutput()
{
    if (!_process) return;

    QString output = QString::fromUtf8(_process->readAllStandardOutput());
    std::cout << "[neural-tracer] " << output.toStdString();

    // Check for the "listening" message that indicates service is ready
    if (output.contains("listening on")) {
        _serviceReady = true;
    }
}

void NeuralTraceServiceManager::handleReadyReadStandardError()
{
    if (!_process) return;

    QString error = QString::fromUtf8(_process->readAllStandardError());
    std::cerr << "[neural-tracer] " << error.toStdString();

    // Capture error output for diagnostics
    if (!error.trimmed().isEmpty() && !_serviceReady) {
        // Don't overwrite with every stderr line, just keep first significant error
        if (_lastError.isEmpty() && error.contains("error", Qt::CaseInsensitive)) {
            _lastError = error.trimmed();
        }
    }
}
