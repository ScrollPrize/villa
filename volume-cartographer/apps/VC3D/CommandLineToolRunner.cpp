#include "CommandLineToolRunner.hpp"
#include "CWindow.hpp"
#include <QDir>
#include <QFileInfo>
#include <QStatusBar>
#include <QVBoxLayout>
#include <QCoreApplication>
#include <QDateTime>
#include <QTextStream>



CommandLineToolRunner::CommandLineToolRunner(QStatusBar* statusBar, CWindow* mainWindow, QObject* parent)
    : QObject(parent)
    , _mainWindow(mainWindow)
    , _progressUtil(new ProgressUtil(nullptr, this))
    , _process(nullptr)
    , _consoleOutput(new ConsoleOutputWidget())
    , _consoleDialog(new QDialog(nullptr, Qt::Window))
    , _autoShowConsole(true)
    , _scale(1.0f)
    , _resolution(0)
    , _layers(31)
    , _seed_x(0)
    , _seed_y(0)
    , _seed_z(0)
    , _parallelProcesses(8)
    , _iterationCount(1000)
    , _logFile(nullptr)
    , _logStream(nullptr)
{
    _consoleDialog->setWindowTitle(tr("Command Output"));
    _consoleDialog->resize(700, 500);

    QVBoxLayout* layout = new QVBoxLayout(_consoleDialog);
    layout->addWidget(_consoleOutput);
    _consoleDialog->setLayout(layout);
}

CommandLineToolRunner::~CommandLineToolRunner()
{
    if (_process) {
        if (_process->state() != QProcess::NotRunning) {
            _process->terminate();
            _process->waitForFinished(3000);
        }
        delete _process;
    }

    if (_logStream) {
        delete _logStream;
        _logStream = nullptr;
    }
    if (_logFile) {
        _logFile->close();
        delete _logFile;
        _logFile = nullptr;
    }

    delete _consoleDialog;
    // _consoleOutput is deleted by _consoleDialog
}

void CommandLineToolRunner::setVolumePath(const QString& path)
{
    _volumePath = path;
}

void CommandLineToolRunner::setSegmentPath(const QString& path)
{
    _segmentPath = path;
}

void CommandLineToolRunner::setOutputPattern(const QString& pattern)
{
    _outputPattern = pattern;
}

void CommandLineToolRunner::setRenderParams(float scale, int resolution, int layers)
{
    _scale = scale;
    _resolution = resolution;
    _layers = layers;
}

void CommandLineToolRunner::setGrowParams(QString volumePath, QString tgtDir, QString jsonParams, int seed_x, int seed_y, int seed_z, bool useExpandMode, bool useRandomSeed)
{
    _volumePath = volumePath;
    _tgtDir = tgtDir;
    _jsonParams = jsonParams;
    _seed_x = seed_x;
    _seed_y = seed_y;
    _seed_z = seed_z;
    _useExpandMode = useExpandMode;
    _useRandomSeed = useRandomSeed;

    QFile file(_jsonParams);
    if (file.open(QIODevice::ReadOnly)) {
        QByteArray jsonData = file.readAll();
        file.close();

        QJsonDocument doc = QJsonDocument::fromJson(jsonData);
        QJsonObject jsonObj = doc.object();

        if (useExpandMode) {
            jsonObj["mode"] = "expansion";
        } else if (useRandomSeed) {
            jsonObj["mode"] = "random_seed";
        } else {
            jsonObj["mode"] = "explicit_seed";
        }

        doc.setObject(jsonObj);
        jsonData = doc.toJson();

        if (file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
            file.write(jsonData);
            file.close();
        }
    }
}

void CommandLineToolRunner::setTraceParams(QString volumePath, QString srcDir, QString tgtDir, QString jsonParams, QString srcSegment)
{
    _volumePath = volumePath;
    _srcDir = srcDir;
    _tgtDir = tgtDir;
    _jsonParams = jsonParams;
    _srcSegment = srcSegment;
}

void CommandLineToolRunner::setAddOverlapParams(QString tgtDir, QString tifxyzPath)
{
    _tgtDir = tgtDir;
    _tifxyzPath = tifxyzPath;
}

void CommandLineToolRunner::setToObjParams(QString tifxyzPath, QString objPath)
{
    _tifxyzPath = tifxyzPath;
    _objPath = objPath;
}

void CommandLineToolRunner::setIncludeTifs(bool include)
{
    _includeTifs = include;
}

void CommandLineToolRunner::setOmpThreads(int threads)
{
    _ompThreads = threads;
}

void CommandLineToolRunner::setToObjOptions(bool normalizeUV, bool alignGrid, int decimateIterations, bool cleanSurface, float cleanK)
{
    _optNormalizeUV = normalizeUV;
    _optAlignGrid = alignGrid;
    _optDecimateIter = decimateIterations;
    _optCleanSurface = cleanSurface;
    _optCleanK = cleanK;
}

void CommandLineToolRunner::setRenderAdvanced(
    int cropX,
    int cropY,
    int cropWidth,
    int cropHeight,
    const QString& affinePath,
    bool invertAffine,
    float scaleSegmentation,
    double rotateDegrees,
    int flipAxis)
{
    _cropX = cropX;
    _cropY = cropY;
    _cropWidth = cropWidth;
    _cropHeight = cropHeight;
    _affinePath = affinePath;
    _invertAffine = invertAffine;
    _scaleSeg = scaleSegmentation;
    _rotateDeg = rotateDegrees;
    _flipAxis = flipAxis;
}

bool CommandLineToolRunner::execute(Tool tool)
{
    if (_process && _process->state() != QProcess::NotRunning) {
        QMessageBox::warning(nullptr, tr("Warning"), tr("A tool is already running."));
        return false;
    }

    _consoleOutput->clear();

    QString toolCmd = toolName(tool);
    QFileInfo toolInfo(toolCmd);
    if (!toolInfo.exists() || !toolInfo.isExecutable()) {
        QString errorMsg = tr("Tool executable not found or not executable: %1").arg(toolCmd);
        _consoleOutput->appendOutput(errorMsg);
        showConsoleOutput();
        QMessageBox::warning(nullptr, tr("Error"), errorMsg);
        return false;
    }

    if (_mainWindow) {
        QString currentVolumePath = _mainWindow->getCurrentVolumePath();
        if (currentVolumePath.isEmpty()) {
            QMessageBox::warning(nullptr, tr("Error"), tr("No volume selected."));
            return false;
        }
        _volumePath = currentVolumePath;
    } else if (_volumePath.isEmpty()) {
        QMessageBox::warning(nullptr, tr("Error"), tr("Volume path not specified and no main window available."));
        return false;
    }

    if (tool == Tool::RenderTifXYZ && _segmentPath.isEmpty()) {
        QMessageBox::warning(nullptr, tr("Error"), tr("Segment path not specified."));
        return false;
    }

    if (tool == Tool::GrowSegFromSegment && _srcSegment.isEmpty()) {
        QMessageBox::warning(nullptr, tr("Error"), tr("Source segment not specified."));
        return false;
    }

    if (tool == Tool::RenderTifXYZ) {
        if (_outputPattern.isEmpty()) {
            QMessageBox::warning(nullptr, tr("Error"), tr("Output pattern not specified."));
            return false;
        }

        QFileInfo outputInfo(_outputPattern);
        QDir outputDir = outputInfo.dir();
        if (!outputDir.exists()) {
            if (!outputDir.mkpath(".")) {
                QMessageBox::warning(nullptr, tr("Error"), tr("Failed to create output directory: %1").arg(outputDir.path()));
                return false;
            }
        }
    }

    _currentTool = tool;

    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
    QString toolBaseName = QFileInfo(toolCmd).baseName();
    QString logFilePath = QString("/tmp/%1_%2.txt").arg(toolBaseName).arg(timestamp);

    if (_logStream) {
        delete _logStream;
        _logStream = nullptr;
    }
    if (_logFile) {
        _logFile->close();
        delete _logFile;
        _logFile = nullptr;
    }

    _logFile = new QFile(logFilePath);
    if (_logFile->open(QIODevice::WriteOnly | QIODevice::Text)) {
        _logStream = new QTextStream(_logFile);
        _logStream->setAutoDetectUnicode(true);

        *_logStream << "Tool: " << toolCmd << Qt::endl;
        *_logStream << "Started: " << QDateTime::currentDateTime().toString(Qt::ISODate) << Qt::endl;
        *_logStream << "Arguments: " << buildArguments(tool).join(" ") << Qt::endl;
        *_logStream << "===================================" << Qt::endl << Qt::endl;
        _logStream->flush();

        _consoleOutput->appendOutput(tr("Logging output to: %1\n").arg(logFilePath));
    } else {
        _consoleOutput->appendOutput(tr("Warning: Failed to create log file: %1\n").arg(logFilePath));
    }

    if (!_process) {
        _process = new QProcess(this);
        _process->setProcessChannelMode(QProcess::MergedChannels);

        connect(_process, &QProcess::started, this, &CommandLineToolRunner::onProcessStarted);
        connect(_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
                this, &CommandLineToolRunner::onProcessFinished);
        connect(_process, &QProcess::errorOccurred, this, &CommandLineToolRunner::onProcessError);
        connect(_process, &QProcess::readyRead, this, &CommandLineToolRunner::onProcessReadyRead);
    }

    // Apply per-run environment variables (e.g., OMP_NUM_THREADS)
    {
        QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
        if (_ompThreads > 0) {
            env.insert("OMP_NUM_THREADS", QString::number(_ompThreads));
            if (_logStream) {
                *_logStream << "ENV: OMP_NUM_THREADS=" << _ompThreads << Qt::endl;
                _logStream->flush();
            }
        }
        _process->setProcessEnvironment(env);
    }

    QStringList args = buildArguments(tool);
    QString toolCommand = toolName(tool);

    QString startMessage;

    if (tool == Tool::GrowSegFromSeeds) {
        // vc_grow_seg_from_seeds needs to use xargs for parallell processes
        startMessage = tr("Starting %1 with %2 parallel processes for: %3")
                          .arg(toolCommand)
                          .arg(_parallelProcesses)
                          .arg(QFileInfo(_segmentPath).fileName());

        QString baseCmd = QString("%1 %2").arg(toolCommand).arg(args.join(" "));
        QString shellCmd = QString("time seq 1 %1 | xargs -i -P %2 bash -c 'nice ionice %3 || true'")
                              .arg(_iterationCount)
                              .arg(_parallelProcesses)
                              .arg(baseCmd.replace("'", "\\'"));

        emit toolStarted(_currentTool, startMessage);

        _consoleOutput->setTitle(tr("Running: %1 (with xargs)").arg(toolCommand));

        if (_autoShowConsole) {
            showConsoleOutput();
        }

        _process->setProgram("/bin/bash");
        _process->setArguments(QStringList() << "-c" << shellCmd);
        _process->start();
    } else {
        startMessage = tr("Starting %1 for: %2").arg(toolCommand).arg(QFileInfo(_segmentPath).fileName());
        emit toolStarted(_currentTool, startMessage);

        _consoleOutput->setTitle(tr("Running: %1").arg(toolCommand));

        if (_autoShowConsole) {
            showConsoleOutput();
        }

        _process->start(toolCommand, args);
    }

    return true;
}

void CommandLineToolRunner::cancel()
{
    if (_process && _process->state() != QProcess::NotRunning) {
        _process->terminate();
    }
}

bool CommandLineToolRunner::isRunning() const
{
    return (_process && _process->state() != QProcess::NotRunning);
}

void CommandLineToolRunner::showConsoleOutput()
{
    if (_consoleDialog) {
        _consoleDialog->show();
        _consoleDialog->raise();
        _consoleDialog->activateWindow();
    }
}

void CommandLineToolRunner::hideConsoleOutput()
{
    if (_consoleDialog) {
        _consoleDialog->hide();
    }
}

void CommandLineToolRunner::setAutoShowConsoleOutput(bool autoShow)
{
    _autoShowConsole = autoShow;
}

void CommandLineToolRunner::setParallelProcesses(int count)
{
    _parallelProcesses = count;
}

void CommandLineToolRunner::setIterationCount(int count)
{
    _iterationCount = count;
}

void CommandLineToolRunner::onProcessReadyRead()
{
    if (_process) {
        QByteArray output = _process->readAll();
        QString outputText = QString::fromUtf8(output);

        _consoleOutput->appendOutput(outputText);

        if (_logStream) {
            *_logStream << outputText;
            _logStream->flush();
        }

        emit consoleOutputReceived(outputText);
    }
}

void CommandLineToolRunner::onProcessStarted()
{
    QString message = tr("Running %1...").arg(toolName(_currentTool));
    if (_progressUtil) _progressUtil->startAnimation(message);
}

void CommandLineToolRunner::onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    if (_logStream) {
        *_logStream << Qt::endl << "===================================" << Qt::endl;
        *_logStream << "Process finished at: " << QDateTime::currentDateTime().toString(Qt::ISODate) << Qt::endl;
        *_logStream << "Exit code: " << exitCode << Qt::endl;
        *_logStream << "Exit status: " << (exitStatus == QProcess::NormalExit ? "Normal" : "Crashed") << Qt::endl;
        _logStream->flush();
    }

    if (_logStream) {
        delete _logStream;
        _logStream = nullptr;
    }
    if (_logFile) {
        _logFile->close();
        delete _logFile;
        _logFile = nullptr;
    }

    if (exitCode == 0 && exitStatus == QProcess::NormalExit) {
        QString message = tr("%1 completed successfully").arg(toolName(_currentTool));
        QString outputPath = getOutputPath();

        // the runner can copy the output of a process to the clipboard, currently this only makes sense for rendering
        // so the user can quickly open the output dir
        bool copyToClipboard = (_currentTool == Tool::RenderTifXYZ);

        if (copyToClipboard) {
            QApplication::clipboard()->setText(outputPath);
            if (_progressUtil) _progressUtil->stopAnimation(message + tr(" - Path copied to clipboard"));
        } else {
            if (_progressUtil) _progressUtil->stopAnimation(message);
        }

        emit toolFinished(_currentTool, true, message, outputPath, copyToClipboard);
    } else {
        QString errorMessage = tr("%1 failed with exit code: %2")
                                .arg(toolName(_currentTool))
                                .arg(exitCode);

        if (_progressUtil) _progressUtil->stopAnimation(tr("Process failed"));

        emit toolFinished(_currentTool, false, errorMessage, QString(), false);
    }
}

void CommandLineToolRunner::onProcessError(QProcess::ProcessError error)
{
    QString errorMessage = tr("Error running %1: ").arg(toolName(_currentTool));

    switch (error) {
        case QProcess::FailedToStart:
            errorMessage += tr("failed to start - Tool executable not found or not executable. Command: %1").arg(toolName(_currentTool));
            break;
        case QProcess::Crashed: errorMessage += tr("crashed"); break;
        case QProcess::Timedout: errorMessage += tr("timed out"); break;
        case QProcess::WriteError: errorMessage += tr("write error"); break;
        case QProcess::ReadError: errorMessage += tr("read error"); break;
        default: errorMessage += tr("unknown error"); break;
    }

    QStringList args = buildArguments(_currentTool);
    errorMessage += tr("\nArguments: %1").arg(args.join(" "));

    if (_logStream) {
        *_logStream << Qt::endl << "ERROR: " << errorMessage << Qt::endl;
        _logStream->flush();
    }

    if (_logStream) {
        delete _logStream;
        _logStream = nullptr;
    }
    if (_logFile) {
        _logFile->close();
        delete _logFile;
        _logFile = nullptr;
    }

    if (_progressUtil) _progressUtil->stopAnimation(tr("Process failed"));

    emit toolFinished(_currentTool, false, errorMessage, QString(), false);

    if (_consoleOutput) {
        _consoleOutput->appendOutput(errorMessage);
    }

    showConsoleOutput();
}

QStringList CommandLineToolRunner::buildArguments(Tool tool)
{
    QStringList args;

    switch (tool) {
        case Tool::RenderTifXYZ:
            args << "--volume" << _volumePath
                 << "--output" << _outputPattern
                 << "--segmentation" << _segmentPath
                 << "--scale" << QString::number(_scale)
                 << "--group-idx" << QString::number(_resolution)
                 << "--num-slices" << QString::number(_layers);
            // Advanced / optional args
            if (_cropWidth > 0 && _cropHeight > 0) {
                args << "--crop-x" << QString::number(_cropX)
                     << "--crop-y" << QString::number(_cropY)
                     << "--crop-width" << QString::number(_cropWidth)
                     << "--crop-height" << QString::number(_cropHeight);
            }
            if (!_affinePath.isEmpty()) {
                args << "--affine-transform" << _affinePath;
                if (_invertAffine) args << "--invert-affine";
            }
            if (std::abs(_scaleSeg - 1.0f) > 1e-6f) {
                args << "--scale-segmentation" << QString::number(_scaleSeg);
            }
            if (std::abs(_rotateDeg) > 1e-6) {
                args << "--rotate" << QString::number(_rotateDeg);
            }
            if (_flipAxis >= 0) {
                args << "--flip" << QString::number(_flipAxis);
            }
            if (_includeTifs) {
                args << "--include-tifs";
            }
            break;

        case Tool::GrowSegFromSegment:
            args << _volumePath
                 << _srcDir
                 << _tgtDir
                 << _jsonParams
                 << _srcSegment;
            break;


        case Tool::GrowSegFromSeeds:
            args << _volumePath
                 << _tgtDir
                 << _jsonParams;
            // Only add coordinates if not in expand mode and not using random seeding
            if (!_useExpandMode && !_useRandomSeed) {
                args << QString::number(_seed_x)
                     << QString::number(_seed_y)
                     << QString::number(_seed_z);
            }
            break;

        case Tool::SegAddOverlap:
            args << _tgtDir
                 << _tifxyzPath;
            break;

        case Tool::tifxyz2obj:
            args << _tifxyzPath
                 << _objPath;
            if (_optNormalizeUV) args << "--normalize-uv";
            if (_optAlignGrid)   args << "--align-grid";
            if (_optDecimateIter > 0) {
                args << "--decimate" << QString::number(_optDecimateIter);
            }
            if (_optCleanSurface) {
                args << "--clean" << QString::number(_optCleanK);
            }
            break;
        case Tool::obj2tifxyz:
            args << _objPath
                 << _objOutputDir
                 << QString::number(_objStretchFactor)
                 << QString::number(_objMeshUnits)
                 << QString::number(_objStepSize);
            break;
    }

    return args;
}

QString CommandLineToolRunner::toolName(Tool tool) const
{
    QString basePath = QCoreApplication::applicationDirPath() + "/";

    switch (tool) {
        case Tool::RenderTifXYZ:
            return basePath + "vc_render_tifxyz";

        case Tool::GrowSegFromSegment:
            return basePath + "vc_grow_seg_from_segments";

        case Tool::GrowSegFromSeeds:
            return basePath + "vc_grow_seg_from_seed";

        case Tool::SegAddOverlap:
            return basePath + "vc_seg_add_overlap";

        case Tool::tifxyz2obj:
            return basePath + "vc_tifxyz2obj";

        case Tool::obj2tifxyz:
            return basePath + "vc_obj2tifxyz_legacy";

        default:
            return "unknown_tool";
    }
}

QString CommandLineToolRunner::getOutputPath() const
{
    QFileInfo outputInfo(_outputPattern);
    return outputInfo.dir().path();
}

void CommandLineToolRunner::setObj2TifxyzParams(const QString& objPath, const QString& outputDir,
                                                float stretchFactor, float meshUnits, int stepSize)
{
    _objPath = objPath;
    _objOutputDir = outputDir;
    _objStretchFactor = stretchFactor;
    _objMeshUnits = meshUnits;
    _objStepSize = stepSize;
}