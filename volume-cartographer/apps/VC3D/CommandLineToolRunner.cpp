#include "CommandLineToolRunner.hpp"

#include <vc_reader/zarr.hpp>
#include "ToolExecutable.hpp"
#include "ToolOutput.hpp"
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



CommandLineToolRunner::CommandLineToolRunner(
    QStatusBar* statusBar,
    std::function<QString()> currentVolumePath,
    QObject* parent)
    : QObject(parent)
    , _currentVolumePath(std::move(currentVolumePath))
    , _progressUtil(new ProgressUtil(nullptr, this))
    , _consoleOutput(new ConsoleOutputWidget())
    , _consoleDialog(new QDialog(nullptr, Qt::Window))
    , _scale(1.0f)
    , _resolution(0)
    , _layers(31)
{
    _consoleDialog->setWindowTitle(tr("Command Output"));
    _consoleDialog->resize(700, 500);

    QVBoxLayout* layout = new QVBoxLayout(_consoleDialog);
    layout->addWidget(_consoleOutput);
    _consoleDialog->setLayout(layout);
}

CommandLineToolRunner::~CommandLineToolRunner()
{
    delete _run;
    delete _consoleDialog;
    // _consoleOutput is deleted by _consoleDialog
}

void CommandLineToolRunner::setVolumePath(const QString& path)
{
    _volumePath = path;
    _explicitVolumePath = !_volumePath.isEmpty();
}

void CommandLineToolRunner::setStore(const QString& storePath, double voxelScale)
{
    _storePath = storePath;
    _voxelScale = voxelScale;
}

void CommandLineToolRunner::setSurface(const QString& uuid)
{
    _surfaceUuid = uuid;
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

void CommandLineToolRunner::setTraceParams(QString volumePath, QString volumeUuid, QString sourceTag, QString outputTag,
                                           QString jsonParams, QString srcSurface)
{
    setVolumePath(volumePath);
    _volumeUuid = volumeUuid;
    _sourceTag = sourceTag;
    _outputTag = outputTag;
    _jsonParams = jsonParams;
    _sourceSurfaceUuid = srcSurface;
}

void CommandLineToolRunner::setToObjParams(QString surfaceUuid, QString objName, QString volumeUuid)
{
    _surfaceUuid = surfaceUuid;
    _volumeUuid = volumeUuid;
    _objPath = objName;
}

void CommandLineToolRunner::setIncludeTifs(bool include)
{
    _includeTifs = include;
}

void CommandLineToolRunner::setNextOmpThreads(int threads)
{
    _nextOmpThreads = threads;
}

void CommandLineToolRunner::setToObjOptions(bool normalizeUV, bool alignGrid)
{
    _optNormalizeUV = normalizeUV;
    _optAlignGrid = alignGrid;
}

void CommandLineToolRunner::setRenderVolume(const QString& volumeUuid, double voxelScale)
{
    _renderVolumeUuid = volumeUuid;
    _renderVoxelScale = voxelScale;
}

void CommandLineToolRunner::setRenderAdvanced(
    int cropX,
    int cropY,
    int cropWidth,
    int cropHeight,
    double rotateDegrees,
    int flipAxis)
{
    _cropX = cropX;
    _cropY = cropY;
    _cropWidth = cropWidth;
    _cropHeight = cropHeight;
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
    if (isRunning()) {
        warn(tr("Warning"), tr("A tool is already running."));
        return false;
    }

    _consoleOutput->clear();

    if (isCustom && _customCommand.isEmpty()) {
        warn(tr("Error"), tr("Custom command not specified."));
        return false;
    }

    const QString toolCmd = isCustom ? _customCommand : toolName(tool);

    if (!isCustom) {
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
        if (vcr::zarr::isRemote(_volumePath.trimmed().toStdString())) {
            warn(
                tr("Unsupported Remote Volume"),
                tr("This command accepts only a local volume path. "
                   "Remote locators are rejected without stripping selectors."));
            return false;
        }
    }

    if (!isCustom && _storePath.isEmpty()) {
        warn(tr("Error"), tr("No object store attached."));
        return false;
    }

    if ((tool == Tool::RenderTifXYZ || tool == Tool::tifxyz2obj) && _surfaceUuid.isEmpty()) {
        warn(tr("Error"), tr("Surface not specified."));
        return false;
    }

    if (tool == Tool::GrowSegFromSegment && _sourceSurfaceUuid.isEmpty()) {
        warn(tr("Error"), tr("Source segment not specified."));
        return false;
    }

    if (tool == Tool::NeighborCopy) {
        if (_jsonParams.isEmpty() || _resumeSurfaceUuid.isEmpty()) {
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

    auto launch = launchFor(tool);
    launch.tool = toolCmd;
    launch.ompThreads = ompThreads;
    const QString toolLabel = isCustom && !_customLabel.isEmpty() ? _customLabel : QFileInfo(toolCmd).baseName();
    launch.logPath = QStringLiteral("%1/%2_%3.txt")
        .arg(QDir::tempPath(), QFileInfo(toolCmd).baseName(),
             QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss")));

    delete _run;
    _run = new vc3d::ToolProcess(this);
    connect(_run, &vc3d::ToolProcess::started, this, [this, toolLabel] {
        if (_progressUtil) _progressUtil->startAnimation(tr("Running %1...").arg(toolLabel));
    });
    connect(_run, &vc3d::ToolProcess::output, this, [this](const QString& chunk) {
        _consoleOutput->appendOutput(chunk);
        emit consoleOutputReceived(chunk);
    });
    connect(_run, &vc3d::ToolProcess::line, this, [this](const QString& line) {
        if (line.trimmed().startsWith(QStringLiteral("obj:"))) _exportedObj = line.trimmed().mid(4).trimmed();
    });
    connect(_run, &vc3d::ToolProcess::finished, this, [this, toolLabel](const vc3d::ToolProcess::Result& result) {
        _createdSurfaces = result.surfaces;
        if (result.ok) {
            const QString message = tr("%1 completed successfully").arg(toolLabel);
            const QString outputPath = getOutputPath();
            // A render's output directory goes to the clipboard to open it quickly.
            const bool copyToClipboard = _currentTool == Tool::RenderTifXYZ;
            if (copyToClipboard) QApplication::clipboard()->setText(outputPath);
            if (_progressUtil)
                _progressUtil->stopAnimation(copyToClipboard ? message + tr(" - Path copied to clipboard") : message);
            finishExecution(true, message, outputPath, copyToClipboard);
            return;
        }
        const QString message = result.cancelled ? tr("%1 cancelled").arg(toolLabel) : result.error;
        _consoleOutput->appendOutput(message);
        if (_executionOptions.showConsole) showConsoleOutput();
        if (_progressUtil) _progressUtil->stopAnimation(tr("Process failed"));
        finishExecution(false, message);
    });

    _currentTool = tool;
    _createdSurfaces.clear();
    _exportedObj.clear();
    _executionOptions = options;
    _terminalReported = false;
    if (!_run->start(launch)) {
        _consoleOutput->appendOutput(_run->error());
        if (options.showConsole) showConsoleOutput();
        warn(tr("Error"), _run->error());
        _terminalReported = true;
        _executionOptions = {};
        return false;
    }
    emit toolStarted(_currentTool, isCustom ? tr("Starting %1").arg(toolLabel)
                                            : tr("Starting %1 for: %2")
                                                  .arg(toolLabel, _surfaceUuid.isEmpty() ? _sourceSurfaceUuid : _surfaceUuid));
    _consoleOutput->setTitle(tr("Running: %1").arg(toolLabel));
    _consoleOutput->appendOutput(tr("Logging output to: %1\n").arg(launch.logPath));
    _consoleOutput->appendOutput(tr("Command: %1\n").arg(_run->command()));
    if (options.showConsole) showConsoleOutput();
    return true;
}

bool CommandLineToolRunner::executeCustomCommand(const QString& command,
                                                 const QStringList& args,
                                                 const QString& label,
                                                 ExecutionOptions options,
                                                 const QString& store,
                                                 const QString& volume,
                                                 double voxelScale)
{
    _customCommand = command;
    _customArgs = args;
    _customLabel = label;
    _customStore = store;
    _customVolume = volume;
    _customVoxelScale = voxelScale;
    return execute(Tool::CustomCommand, options);
}

void CommandLineToolRunner::cancel()
{
    if (_run) _run->cancel();
}

bool CommandLineToolRunner::isRunning() const
{
    return _run && _run->running();
}

void CommandLineToolRunner::showConsoleOutput()
{
    if (_consoleDialog) {
        _consoleDialog->show();
        _consoleDialog->raise();
        _consoleDialog->activateWindow();
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

vc3d::ToolProcess::Launch CommandLineToolRunner::launchFor(Tool tool)
{
    vc3d::ToolProcess::Launch launch;
    if (tool == Tool::CustomCommand) {
        launch.args = _customArgs;
        launch.store = _customStore;
        launch.volume = _customVolume;
        launch.voxelScale = _customVoxelScale;
        return launch;
    }
    launch.args = buildArguments(tool);
    launch.store = _storePath;
    launch.volume = tool == Tool::RenderTifXYZ ? _renderVolumeUuid
                                               : _volumeUuid;
    launch.voxelScale = tool == Tool::RenderTifXYZ ? _renderVoxelScale : _voxelScale;
    return launch;
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
                 << "--surface" << _surfaceUuid
                 << "--scale" << QString::number(_scale)
                 << "--group-idx" << QString::number(_resolution)
                 << "--num-slices" << QString::number(_layers);
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
            if (std::abs(_rotateDeg) > 1e-6) {
                args << "--rotate" << QString::number(_rotateDeg);
            }
            if (_flipAxis >= 0) {
                args << "--flip" << QString::number(_flipAxis);
            }
            // "Also write TIFF slices" only adds anything when the primary
            // output is the zarr -- a TifStack render already writes them.
            // vc_render_tifxyz accepts --zarr-output and --tif-output together,
            // so ask for the slices in a sibling directory of the .zarr rather
            // than through a flag of its own.
            if (_includeTifs && _renderOutputFormat == RenderOutputFormat::Zarr) {
                const QFileInfo zarrInfo(_outputPattern);
                args << "--tif-output"
                     << zarrInfo.dir().filePath(zarrInfo.completeBaseName()
                                                + QStringLiteral("_layers"));
            }
            break;

        case Tool::GrowSegFromSegment:
            args << "--volume" << _volumePath
                 << "--params" << _jsonParams
                 << "--surface" << _sourceSurfaceUuid
                 << "--source-tag" << _sourceTag
                 << "--tag" << _outputTag;
            break;

        case Tool::tifxyz2obj:
            args << "--surface" << _surfaceUuid
                 << "--name" << _objPath;
            if (_optNormalizeUV) args << "--normalize-uv";
            if (_optAlignGrid)   args << "--align-grid";
            break;
        case Tool::AlphaCompRefine:
            // Without a new name the surface is refined in place.
            args << "--surface" << _surfaceUuid;
            if (!_refineName.isEmpty()) args << "--name" << _refineName;
            args << _volumePath << _jsonParams;
            break;
        case Tool::NeighborCopy:
            args << "-v" << _volumePath
                 << "-p" << _jsonParams
                 << "--surface" << _resumeSurfaceUuid
                 << "--tag" << _outputTag;
            if (!_resumeOpt.isEmpty()) {
                args << "--resume-opt" << _resumeOpt;
            }
            break;
        case Tool::CustomCommand:
            args = _customArgs;
            break;
    }

    return args;
}

QString CommandLineToolRunner::toolName(Tool tool) const
{
    // The resolved executable, or its bare name (the launch then reports it missing).
    const auto resolved = [](const QString& name) {
        const QString path = vc3d::findToolExecutable(name);
        return path.isEmpty() ? name : path;
    };
    switch (tool) {
        case Tool::RenderTifXYZ:
            return resolved(QStringLiteral("vc_render_tifxyz"));

        case Tool::GrowSegFromSegment:
            return resolved(QStringLiteral("vc_grow_seg_from_segments"));

        case Tool::tifxyz2obj:
            return resolved(QStringLiteral("vc_tifxyz2obj"));

        case Tool::AlphaCompRefine:
            return resolved(QStringLiteral("vc_objrefine"));

        case Tool::NeighborCopy:
            return resolved(QStringLiteral("vc_grow_seg_from_seed"));

        case Tool::CustomCommand:
            return _customCommand.isEmpty() ? "custom_command" : _customCommand;

        default:
            return "unknown_tool";
    }
}

QString CommandLineToolRunner::getOutputPath() const
{
    switch (_currentTool) {
        case Tool::RenderTifXYZ: return QFileInfo(_outputPattern).dir().path();
        case Tool::tifxyz2obj: return _exportedObj;
        // An in-place refinement creates nothing: the source changed.
        case Tool::AlphaCompRefine: return _createdSurfaces.isEmpty() ? _surfaceUuid : _createdSurfaces.join('\n');
        case Tool::CustomCommand: return QString();
        default: return _createdSurfaces.join('\n');
    }
}

void CommandLineToolRunner::setObjRefineParams(const QString& volumePath,
                                               const QString& srcSurface,
                                               const QString& newName,
                                               const QString& jsonParams,
                                               const QString& volumeUuid)
{
    setVolumePath(volumePath);
    _volumeUuid = volumeUuid;
    _surfaceUuid = srcSurface;
    _refineName = newName;
    _jsonParams = jsonParams;
}

void CommandLineToolRunner::setNeighborCopyParams(const QString& volumePath,
                                                  const QString& paramsJson,
                                                  const QString& resumeSurface,
                                                  const QString& volumeUuid,
                                                  const QString& outputTag,
                                                  const QString& resumeOpt)
{
    setVolumePath(volumePath);
    _jsonParams = paramsJson;
    _resumeSurfaceUuid = resumeSurface;
    _volumeUuid = volumeUuid;
    _outputTag = outputTag;
    _resumeOpt = resumeOpt;
}
