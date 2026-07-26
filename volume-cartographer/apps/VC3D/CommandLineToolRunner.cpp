#include "CommandLineToolRunner.hpp"
#include <QDir>
#include <QFileInfo>
#include <QStatusBar>
#include <QVBoxLayout>
#include <QCoreApplication>
#include <QDateTime>
#include <QTextStream>
#include <QMessageBox>
#include <QClipboard>
#include <QApplication>

#include <cmath>
#include <utility>


namespace {

QString quoteArg(const QString& arg)
{
    if (arg.isEmpty())
        return "\"\"";

    bool needsQuotes = false;
    for (const QChar& ch : arg) {
        if (ch.isSpace() || ch == '"' || ch == '\'' || ch == '\\') {
            needsQuotes = true;
            break;
        }
    }

    if (!needsQuotes) {
        return arg;
    }

    QString escaped = arg;
    escaped.replace('\\', "\\\\");
    escaped.replace('"', "\\\"");
    return "\"" + escaped + "\"";
}

QString formatCommand(const QString& program, const QStringList& args, int ompThreads)
{
    QStringList quotedArgs;
    quotedArgs.reserve(args.size());
    for (const auto& arg : args) {
        quotedArgs << quoteArg(arg);
    }

    QString base = QString("%1 %2").arg(program, quotedArgs.join(' '));
    if (ompThreads > 0) {
        return QString("OMP_NUM_THREADS=%1 %2").arg(ompThreads).arg(base);
    }
    return base;
}

} // namespace

CommandLineToolRunner::CommandLineToolRunner(
    QStatusBar* statusBar,
    std::function<QString()> currentVolumePath,
    QObject* parent)
    : QObject(parent)
    , _currentVolumePath(std::move(currentVolumePath))
    , _progressUtil(new ProgressUtil(nullptr, this))
    , _process(nullptr)
    , _consoleOutput(new ConsoleOutputWidget())
    , _consoleDialog(new QDialog(nullptr, Qt::Window))
    , _scale(1.0f)
    , _resolution(0)
    , _layers(31)
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
    _explicitVolumePath = !_volumePath.isEmpty();
}

void CommandLineToolRunner::setRemoteVolumeUrl(const QString& url)
{
    _remoteVolumeUrl = url.trimmed();
}

void CommandLineToolRunner::setRemoteVolumeAuth(const QString& accessKey,
                                                const QString& secretKey,
                                                const QString& sessionToken,
                                                const QString& region)
{
    _remoteAccessKey = accessKey;
    _remoteSecretKey = secretKey;
    _remoteSessionToken = sessionToken;
    _remoteRegion = region;
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

void CommandLineToolRunner::setRenderVoxelSize(double voxelSizeUm, bool enabled)
{
    _renderVoxelSizeUm = voxelSizeUm;
    _useRenderVoxelSize = enabled && std::isfinite(voxelSizeUm) && voxelSizeUm > 0.0;
}

void CommandLineToolRunner::setRenderOutputFormat(RenderOutputFormat format)
{
    _renderOutputFormat = format;
}

void CommandLineToolRunner::setTraceParams(QString volumePath, QString srcDir, QString tgtDir, QString jsonParams, QString srcSegment)
{
    setVolumePath(volumePath);
    _srcDir = srcDir;
    _tgtDir = tgtDir;
    _jsonParams = jsonParams;
    _srcSegment = srcSegment;
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

void CommandLineToolRunner::setNextOmpThreads(int threads)
{
    _nextOmpThreads = threads;
}

void CommandLineToolRunner::setFlattenOptions(bool flatten, int iterations, int downsample)
{
    _flatten = flatten;
    _flattenIters = iterations;
    _flattenDownsample = downsample;
}

void CommandLineToolRunner::setToObjOptions(bool normalizeUV, bool alignGrid)
{
    _optNormalizeUV = normalizeUV;
    _optAlignGrid = alignGrid;
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

bool CommandLineToolRunner::execute(Tool tool, ExecutionOptions options)
{
    const int ompThreads = std::exchange(_nextOmpThreads, -1);
    const bool isCustom = (tool == Tool::CustomCommand);
    const auto warn = [&](const QString& title, const QString& message) {
        if (options.presentation == Presentation::Interactive) {
            QMessageBox::warning(nullptr, title, message);
        }
    };
    if (_process && _process->state() != QProcess::NotRunning) {
        warn(tr("Warning"), tr("A tool is already running."));
        return false;
    }

    if (_preserveConsoleOutput) {
        _consoleOutput->appendOutput(tr("\n========== Next Pass ==========\n\n"));
        _preserveConsoleOutput = false;
    } else {
        _consoleOutput->clear();
    }

    if (isCustom && _customCommand.isEmpty()) {
        warn(tr("Error"), tr("Custom command not specified."));
        return false;
    }

    QString toolCmd = isCustom ? _customCommand : toolName(tool);
    QFileInfo toolInfo(toolCmd);
    if (!toolInfo.exists() || !toolInfo.isExecutable()) {
        QString errorMsg = tr("Tool executable not found or not executable: %1").arg(toolCmd);
        _consoleOutput->appendOutput(errorMsg);
        if (options.showConsole) {
            showConsoleOutput();
        }
        warn(tr("Error"), errorMsg);
        return false;
    }

    // vc_merge_tifxyz and vc_merge_patch are path-only tools (segment dirs);
    // no volume required.
    const bool needsVolume = !isCustom
                             && tool != Tool::MergeTifxyz
                             && tool != Tool::MergePatch;
    if (needsVolume) {
        if (_explicitVolumePath) {
            if (_volumePath.isEmpty()) {
                warn(tr("Error"), tr("Volume path not specified."));
                return false;
            }
        } else {
            QString resolvedVolumePath = _volumePath;
            if (_currentVolumePath) {
                resolvedVolumePath = _currentVolumePath();
                if (resolvedVolumePath.isEmpty()) {
                    warn(tr("Error"), tr("No volume selected."));
                    return false;
                }
            } else if (resolvedVolumePath.isEmpty()) {
                warn(tr("Error"), tr("Volume path not specified and no main window available."));
                return false;
            }
            _volumePath = resolvedVolumePath;
        }
    }

    if (tool == Tool::GrowSegFromSegment || tool == Tool::AlphaCompRefine) {
        const auto lowered = _volumePath.trimmed().toLower();
        if (lowered.startsWith("http://") || lowered.startsWith("https://") ||
            lowered.startsWith("s3://") || lowered.startsWith("s3+")) {
            warn(
                tr("Unsupported Remote Volume"),
                tr("This command accepts only a local volume path. "
                   "Remote locators are rejected without stripping selectors."));
            return false;
        }
    }

    if (tool == Tool::MergeTifxyz && _mergeJsonPath.isEmpty()) {
        warn(tr("Error"), tr("merge.json path not specified."));
        return false;
    }

    if (tool == Tool::RenderTifXYZ && _segmentPath.isEmpty()) {
        warn(tr("Error"), tr("Segment path not specified."));
        return false;
    }

    if (tool == Tool::GrowSegFromSegment && _srcSegment.isEmpty()) {
        warn(tr("Error"), tr("Source segment not specified."));
        return false;
    }

    if (tool == Tool::NeighborCopy) {
        if (_jsonParams.isEmpty() || _resumeSurfacePath.isEmpty() || _tgtDir.isEmpty()) {
            warn(tr("Error"), tr("Neighbor copy parameters incomplete."));
            return false;
        }
    }

    if (tool == Tool::RenderTifXYZ) {
        if (_outputPattern.isEmpty()) {
            warn(tr("Error"), tr("Output pattern not specified."));
            return false;
        }

        QFileInfo outputInfo(_outputPattern);
        QDir outputDir = outputInfo.dir();
        if (!outputDir.exists()) {
            if (!outputDir.mkpath(".")) {
                warn(tr("Error"),
                     tr("Failed to create output directory: %1").arg(outputDir.path()));
                return false;
            }
        }
    }

    _currentTool = tool;
    _executionOptions = options;
    _terminalReported = false;
    _pendingProcessError.clear();

    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
    QString toolBaseName = QFileInfo(toolCmd).baseName();
    QString logFilePath = QString("%1/%2_%3.txt")
        .arg(QDir::tempPath()).arg(toolBaseName).arg(timestamp);

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
        QStringList argsForLog = isCustom ? _customArgs : buildArguments(tool);
        *_logStream << "Arguments: " << argsForLog.join(" ") << Qt::endl;
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
        if (ompThreads > 0) {
            env.insert("OMP_NUM_THREADS", QString::number(ompThreads));
            if (_logStream) {
                *_logStream << "ENV: OMP_NUM_THREADS=" << ompThreads << Qt::endl;
                _logStream->flush();
            }
        }
        if (!_remoteVolumeUrl.isEmpty() && !_remoteAccessKey.isEmpty() && !_remoteSecretKey.isEmpty()) {
            env.insert("AWS_ACCESS_KEY_ID", _remoteAccessKey);
            env.insert("AWS_SECRET_ACCESS_KEY", _remoteSecretKey);
            if (!_remoteSessionToken.isEmpty()) {
                env.insert("AWS_SESSION_TOKEN", _remoteSessionToken);
            }
            if (!_remoteRegion.isEmpty()) {
                env.insert("AWS_DEFAULT_REGION", _remoteRegion);
            }
            if (_logStream) {
                *_logStream << "ENV: AWS_ACCESS_KEY_ID=<set>" << Qt::endl;
                *_logStream << "ENV: AWS_SECRET_ACCESS_KEY=<set>" << Qt::endl;
                if (!_remoteSessionToken.isEmpty()) {
                    *_logStream << "ENV: AWS_SESSION_TOKEN=<set>" << Qt::endl;
                }
                if (!_remoteRegion.isEmpty()) {
                    *_logStream << "ENV: AWS_DEFAULT_REGION=" << _remoteRegion << Qt::endl;
                }
                _logStream->flush();
            }
        }
        _process->setProcessEnvironment(env);
    }

    QStringList args = isCustom ? _customArgs : buildArguments(tool);
    QString toolCommand = toolCmd;
    QString formattedCommand = formatCommand(toolCommand, args, ompThreads);

    QString startMessage;
    const QString toolLabel = isCustom
                              ? (_customLabel.isEmpty() ? QFileInfo(toolCommand).baseName()
                                                       : _customLabel)
                              : QFileInfo(toolCommand).baseName();

    startMessage = isCustom ? tr("Starting %1").arg(toolLabel)
                           : tr("Starting %1 for: %2")
                                 .arg(toolLabel)
                                 .arg(QFileInfo(_segmentPath).fileName());
    emit toolStarted(_currentTool, startMessage);

    _consoleOutput->setTitle(tr("Running: %1").arg(toolLabel));
    _consoleOutput->appendOutput(tr("Command: %1\n").arg(formattedCommand));
    if (_logStream) {
        *_logStream << "Command: " << formattedCommand << Qt::endl;
        _logStream->flush();
    }

    if (options.showConsole) {
        showConsoleOutput();
    }

    _process->start(toolCommand, args);

    return true;
}

bool CommandLineToolRunner::executeCustomCommand(const QString& command,
                                                 const QStringList& args,
                                                 const QString& label,
                                                 ExecutionOptions options)
{
    _customCommand = command;
    _customArgs = args;
    _customLabel = label;
    return execute(Tool::CustomCommand, options);
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

void CommandLineToolRunner::setPreserveConsoleOutput(bool preserve)
{
    _preserveConsoleOutput = preserve;
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

void CommandLineToolRunner::closeLog()
{
    if (_logStream) {
        delete _logStream;
        _logStream = nullptr;
    }
    if (_logFile) {
        _logFile->close();
        delete _logFile;
        _logFile = nullptr;
    }
}

void CommandLineToolRunner::finishExecution(bool success,
                                            const QString& message,
                                            const QString& outputPath,
                                            bool copyToClipboard)
{
    if (_terminalReported) {
        return;
    }
    _terminalReported = true;
    _explicitVolumePath = false;

    // Move the completed run out of the active slot before notifying observers.
    // A toolFinished handler may synchronously start another process.
    _completionOptions = _executionOptions;
    _executionOptions = {};
    _deliveringCompletion = true;
    emit toolFinished(
        _currentTool, success, message, outputPath, copyToClipboard);
    _deliveringCompletion = false;
}

void CommandLineToolRunner::onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    if (_terminalReported) {
        return;
    }
    if (_logStream) {
        *_logStream << Qt::endl << "===================================" << Qt::endl;
        *_logStream << "Process finished at: " << QDateTime::currentDateTime().toString(Qt::ISODate) << Qt::endl;
        *_logStream << "Exit code: " << exitCode << Qt::endl;
        *_logStream << "Exit status: " << (exitStatus == QProcess::NormalExit ? "Normal" : "Crashed") << Qt::endl;
        _logStream->flush();
    }

    closeLog();

    if (_pendingProcessError.isEmpty() &&
        exitCode == 0 &&
        exitStatus == QProcess::NormalExit) {
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

        finishExecution(true, message, outputPath, copyToClipboard);
    } else {
        const QString errorMessage = _pendingProcessError.isEmpty()
            ? tr("%1 failed with exit code: %2")
                  .arg(toolName(_currentTool))
                  .arg(exitCode)
            : _pendingProcessError;

        if (_progressUtil) _progressUtil->stopAnimation(tr("Process failed"));

        finishExecution(false, errorMessage);
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
    _pendingProcessError = errorMessage;

    if (_logStream) {
        *_logStream << Qt::endl << "ERROR: " << errorMessage << Qt::endl;
        _logStream->flush();
    }

    const bool showConsole = _executionOptions.showConsole;

    if (_consoleOutput) {
        _consoleOutput->appendOutput(errorMessage);
    }

    if (showConsole) {
        showConsoleOutput();
    }

    // Crashes and other runtime errors are followed by finished(); let that
    // signal own terminal delivery. FailedToStart is the exception: Qt does not
    // emit finished() when the process never launched.
    if (error != QProcess::FailedToStart) {
        return;
    }

    closeLog();
    if (_progressUtil) {
        _progressUtil->stopAnimation(tr("Process failed"));
    }
    finishExecution(false, errorMessage);
}

QStringList CommandLineToolRunner::buildArguments(Tool tool)
{
    QStringList args;

    switch (tool) {
        case Tool::RenderTifXYZ:
            args << "--volume" << _volumePath
                 << (_renderOutputFormat == RenderOutputFormat::Zarr
                         ? QStringLiteral("--zarr-output")
                         : QStringLiteral("--tif-output"))
                 << _outputPattern
                 << "--segmentation" << _segmentPath
                 << "--scale" << QString::number(_scale)
                 << "--group-idx" << QString::number(_resolution)
                 << "--num-slices" << QString::number(_layers);
            if (!_remoteVolumeUrl.isEmpty()) {
                args << "--remote-url" << _remoteVolumeUrl;
            }
            if (_useRenderVoxelSize) {
                args << "--voxel-size" << QString::number(_renderVoxelSizeUm, 'g', 17)
                     << "--voxel-unit" << "micrometer";
            }
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
            if (_flatten) {
                args << "--flatten";
                args << "--flatten-iterations" << QString::number(_flattenIters);
                args << "--flatten-downsample" << QString::number(_flattenDownsample);
            }
            break;

        case Tool::GrowSegFromSegment:
            args << _volumePath
                 << _srcDir
                 << _tgtDir
                 << _jsonParams
                 << _srcSegment;
            break;


        case Tool::tifxyz2obj:
            args << _tifxyzPath
                 << _objPath;
            if (_optNormalizeUV) args << "--normalize-uv";
            if (_optAlignGrid)   args << "--align-grid";
            break;
        case Tool::obj2tifxyz:
            args << _objPath
                 << _objOutputDir
                 << QString::number(_objStretchFactor)
                 << QString::number(_objMeshUnits)
                 << QString::number(_objStepSize);
            break;
        case Tool::AlphaCompRefine:
            args << _volumePath
                 << _segmentPath
                 << _refineDst
                 << _jsonParams;
            break;
        case Tool::NeighborCopy:
            args << "-v" << _volumePath
                 << "-p" << _jsonParams
                 << "--resume" << _resumeSurfacePath
                 << "-t" << _tgtDir;
            if (!_resumeOpt.isEmpty()) {
                args << "--resume-opt" << _resumeOpt;
            }
            break;
        case Tool::MergeTifxyz:
            args << "--merge" << _mergeJsonPath;
            // The tool defaults paths-dir to <merge.parent>/paths; pass
            // the resolved segments dir (e.g. paths_2um_ds2/) explicitly
            // so this works regardless of the volpkg's directory layout.
            if (!_mergePathsDir.isEmpty()) {
                args << "--paths-dir" << _mergePathsDir;
            }
            // Empty refSurface -> let vc_merge_tifxyz auto-pick the
            // surface with the largest valid-cell count. Otherwise pass
            // the user's pick straight through.
            if (!_mergeRefSurface.isEmpty()) {
                args << "--ref" << _mergeRefSurface;
            }
            args << "--ransac-iters"      << QString::number(_mergeRansacIters)
                 << "--ransac-min-thresh" << QString::number(_mergeRansacMinThresh, 'g', 10)
                 << "--ransac-max-thresh" << QString::number(_mergeRansacMaxThresh, 'g', 10)
                 << "--ransac-mad-k"      << QString::number(_mergeRansacMadK,      'g', 10)
                 << "--ransac-seed"       << QString::number(_mergeRansacSeed)
                 << "--anchor-cap"        << QString::number(_mergeAnchorCap)
                 << "--strip-cols"        << QString::number(_mergeStripCols);
            break;
        case Tool::MergePatch:
            // Either positional (auto-detect roles by valid-cell count) or
            // explicit --parent / --child when the user swapped via the
            // dialog. The binary requires exactly one of these forms.
            if (_patchExplicitRoles) {
                args << "--parent" << _patchParentPath
                     << "--child"  << _patchChildPath;
            } else {
                args << _patchParentPath << _patchChildPath;
            }
            args << "--border-cells"      << QString::number(_patchBorderCells)
                 << "--blend-cells"       << QString::number(_patchBlendCells)
                 << "--idw-k"             << QString::number(_patchIdwK)
                 << "--ransac-iters"      << QString::number(_patchRansacIters)
                 << "--ransac-min-thresh" << QString::number(_patchRansacMinThresh, 'g', 10)
                 << "--ransac-max-thresh" << QString::number(_patchRansacMaxThresh, 'g', 10)
                 << "--ransac-mad-k"      << QString::number(_patchRansacMadK,      'g', 10)
                 << "--ransac-seed"       << QString::number(_patchRansacSeed)
                 << "--anchor-cap"        << QString::number(_patchAnchorCap);
            break;
        case Tool::CustomCommand:
            args = _customArgs;
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

        case Tool::tifxyz2obj:
            return basePath + "vc_tifxyz2obj";

        case Tool::obj2tifxyz:
            return basePath + "vc_obj2tifxyz_legacy";

        case Tool::AlphaCompRefine:
            return basePath + "vc_objrefine";

        case Tool::NeighborCopy:
            return basePath + "vc_grow_seg_from_seed";

        case Tool::MergeTifxyz:
            return basePath + "vc_merge_tifxyz";

        case Tool::MergePatch:
            return basePath + "vc_merge_patch";

        case Tool::CustomCommand:
            return _customCommand.isEmpty() ? "custom_command" : _customCommand;

        default:
            return "unknown_tool";
    }
}

QString CommandLineToolRunner::getOutputPath() const
{
    if (_currentTool == Tool::AlphaCompRefine) {
        return _refineDst;
    }
    if (_currentTool == Tool::NeighborCopy) {
        return _tgtDir;
    }
    if (_currentTool == Tool::MergeTifxyz) {
        // The tool auto-names the output dir under <volpkg>/paths/; we
        // surface the merge.json instead so the user can find the run.
        return _mergeJsonPath;
    }
    if (_currentTool == Tool::MergePatch) {
        // The binary overwrites the parent in place; surface that path so
        // the post-run notification can point users at what changed.
        return _patchParentPath;
    }
    if (_currentTool == Tool::CustomCommand) {
        return QString();
    }

    QFileInfo outputInfo(_outputPattern);
    return outputInfo.dir().path();
}

void CommandLineToolRunner::setMergeParams(const QString& mergeJsonPath,
                                           const QString& pathsDir,
                                           const QString& refSurface,
                                           int ransacIters,
                                           double ransacMinThresh,
                                           double ransacMaxThresh,
                                           double ransacMadK,
                                           int ransacSeed,
                                           int anchorCap,
                                           int stripCols)
{
    _mergeJsonPath        = mergeJsonPath;
    _mergePathsDir        = pathsDir;
    _mergeRefSurface      = refSurface;
    _mergeRansacIters     = ransacIters;
    _mergeRansacMinThresh = ransacMinThresh;
    _mergeRansacMaxThresh = ransacMaxThresh;
    _mergeRansacMadK      = ransacMadK;
    _mergeRansacSeed      = ransacSeed;
    _mergeAnchorCap       = anchorCap;
    _mergeStripCols       = stripCols;
}

void CommandLineToolRunner::setMergePatchParams(const QString& parentPath,
                                                const QString& childPath,
                                                bool explicitRoles,
                                                int borderCells,
                                                int blendCells,
                                                int idwK,
                                                int ransacIters,
                                                double ransacMinThresh,
                                                double ransacMaxThresh,
                                                double ransacMadK,
                                                int ransacSeed,
                                                int anchorCap)
{
    _patchParentPath      = parentPath;
    _patchChildPath       = childPath;
    _patchExplicitRoles   = explicitRoles;
    _patchBorderCells     = borderCells;
    _patchBlendCells      = blendCells;
    _patchIdwK            = idwK;
    _patchRansacIters     = ransacIters;
    _patchRansacMinThresh = ransacMinThresh;
    _patchRansacMaxThresh = ransacMaxThresh;
    _patchRansacMadK      = ransacMadK;
    _patchRansacSeed      = ransacSeed;
    _patchAnchorCap       = anchorCap;
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

void CommandLineToolRunner::setObjRefineParams(const QString& volumePath,
                                               const QString& srcSurface,
                                               const QString& dstSurface,
                                               const QString& jsonParams)
{
    setVolumePath(volumePath);
    _segmentPath = srcSurface;
    _refineDst = dstSurface;
    _jsonParams = jsonParams;
}

void CommandLineToolRunner::setNeighborCopyParams(const QString& volumePath,
                                                  const QString& paramsJson,
                                                  const QString& resumeSurface,
                                                  const QString& outputDir,
                                                  const QString& resumeOpt)
{
    setVolumePath(volumePath);
    _jsonParams = paramsJson;
    _resumeSurfacePath = resumeSurface;
    _tgtDir = outputDir;
    _resumeOpt = resumeOpt;
}
