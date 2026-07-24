#pragma once

#include <functional>

#include <QObject>
#include <QString>
#include <QStringList>
#include <QProcess>
#include <QDialog>
#include <QFile>

#include "elements/ProgressUtil.hpp"
#include "ConsoleOutputWidget.hpp"

/**
 * @brief Class to manage execution of command-line tools
 */
class CommandLineToolRunner : public QObject
{
    Q_OBJECT

public:

    explicit CommandLineToolRunner(
        QStatusBar* statusBar,
        std::function<QString()> currentVolumePath,
        QObject* parent = nullptr);
    
    ~CommandLineToolRunner();

    enum class Tool {
        RenderTifXYZ,
        GrowSegFromSegment,
        tifxyz2obj,
        obj2tifxyz,
        AlphaCompRefine,
        NeighborCopy,
        MergeTifxyz,
        MergePatch,
        CustomCommand
    };

    enum class Presentation {
        Interactive,
        Silent
    };

    struct ExecutionOptions {
        constexpr ExecutionOptions(
            Presentation presentation = Presentation::Interactive,
            bool showConsole = true)
            : presentation(presentation)
            , showConsole(showConsole)
        {
        }

        static constexpr ExecutionOptions silent()
        {
            return {Presentation::Silent, false};
        }

        Presentation presentation;
        bool showConsole;
    };

    // Output format for Tool::RenderTifXYZ: a per-slice TIFF stack or an
    // OME-Zarr store. Interactive launches reset this to TifStack.
    enum class RenderOutputFormat {
        TifStack,   // --tif-output <pattern>
        Zarr        // --zarr-output <path.zarr>
    };

    void setVolumePath(const QString& path);
    void setRemoteVolumeUrl(const QString& url);
    void setRemoteVolumeAuth(const QString& accessKey,
                             const QString& secretKey,
                             const QString& sessionToken,
                             const QString& region);
    void setSegmentPath(const QString& path);
    void setOutputPattern(const QString& pattern);

    bool executeCustomCommand(const QString& command,
                             const QStringList& args,
                             const QString& label = QString(),
                             ExecutionOptions options = {});

    // tool specific params 
    void setRenderParams(float scale, int resolution, int layers);
    void setRenderVoxelSize(double voxelSizeUm, bool enabled);
    // Selects --tif-output vs --zarr-output for the next RenderTifXYZ run (default
    // TifStack; interactive re-asserts TifStack to avoid leaking a prior Zarr choice).
    void setRenderOutputFormat(RenderOutputFormat format);
    void setRenderAdvanced(int cropX, int cropY, int cropWidth, int cropHeight,
                           const QString& affinePath, bool invertAffine,
                           float scaleSegmentation, double rotateDegrees, int flipAxis);
    void setTraceParams(QString volumePath, QString srcDir, QString tgtDir, QString jsonParams, QString srcSegment);
    void setToObjParams(QString tifxyzPath, QString objPath);
    void setToObjOptions(bool normalizeUV, bool alignGrid);
    void setObj2TifxyzParams(const QString& objPath, const QString& outputDir,
                             float stretchFactor = 1000.0f,
                             float meshUnits = 1.0f,
                             int stepSize = 20);
    void setObjRefineParams(const QString& volumePath,
                            const QString& srcSurface,
                            const QString& dstSurface,
                            const QString& jsonParams);
    void setNeighborCopyParams(const QString& volumePath,
                               const QString& paramsJson,
                               const QString& resumeSurface,
                               const QString& outputDir,
                               const QString& resumeOpt);
    // vc_merge_tifxyz: only --merge is required; the remaining args are
    // RANSAC + blend tunables that all have working defaults. pathsDir
    // is forwarded as --paths-dir so the volpkg's actual segments
    // directory (e.g. paths_2um_ds2/, traces/) is used instead of the
    // tool's `<merge.parent>/paths` default.
    void setMergeParams(const QString& mergeJsonPath,
                        const QString& pathsDir,
                        const QString& refSurface,
                        int ransacIters,
                        double ransacMinThresh,
                        double ransacMaxThresh,
                        double ransacMadK,
                        int ransacSeed,
                        int anchorCap,
                        int stripCols);
    // vc_merge_patch: two tifxyz dirs; the binary auto-detects parent vs
    // child by valid-cell count unless explicitRoles is set, in which case
    // we pass --parent / --child explicitly. The parent tifxyz is
    // overwritten in place (the binary snapshots the pre-patch state to
    // <parent_dir>/backups/<parent_name>/{0..7}/ first).
    void setMergePatchParams(const QString& parentPath,
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
                             int anchorCap);
    bool execute(Tool tool, ExecutionOptions options = {});
    void cancel();
    bool isRunning() const;
    
    void showConsoleOutput();
    void hideConsoleOutput();
    void setIncludeTifs(bool include);
    // Applies to the next execute() attempt only.
    void setNextOmpThreads(int threads);
    void setFlattenOptions(bool flatten, int iterations, int downsample = 1);
    void setPreserveConsoleOutput(bool preserve);

    // Valid while toolFinished is being delivered for the completed process.
    bool currentExecutionIsSilent() const
    {
        const auto& options =
            _deliveringCompletion ? _completionOptions : _executionOptions;
        return options.presentation == Presentation::Silent;
    }

signals:
    void toolStarted(Tool tool, const QString& message);
    void toolFinished(Tool tool, bool success, const QString& message, const QString& outputPath, bool copyToClipboard = false);
    void consoleOutputReceived(const QString& output);

private slots:
    void onProcessStarted();
    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void onProcessError(QProcess::ProcessError error);
    void onProcessReadyRead();

private:
    QStringList buildArguments(Tool tool);
    QString toolName(Tool tool) const;
    QString getOutputPath() const;
    void closeLog();
    void finishExecution(bool success,
                         const QString& message,
                         const QString& outputPath = {},
                         bool copyToClipboard = false);

    std::function<QString()> _currentVolumePath;
    ProgressUtil* _progressUtil;
    
    QProcess* _process;
    ConsoleOutputWidget* _consoleOutput;
    QDialog* _consoleDialog;
    
    QString _volumePath;
    QString _remoteVolumeUrl;
    QString _remoteAccessKey;
    QString _remoteSecretKey;
    QString _remoteSessionToken;
    QString _remoteRegion;
    QString _segmentPath;
    QString _outputPattern;
    QString _tgtDir;
    QString _srcDir;
    QString _srcSegment;
    QString _tifxyzPath;
    QString _objPath;
    QString _jsonParams;
    QString _resumeSurfacePath;
    QString _resumeOpt;
    
    float _scale;
    int _resolution;
    int _layers;
    double _renderVoxelSizeUm{0.0};
    bool _useRenderVoxelSize{false};
    RenderOutputFormat _renderOutputFormat{RenderOutputFormat::TifStack};
    // Advanced render options
    int _cropX{0};
    int _cropY{0};
    int _cropWidth{0};
    int _cropHeight{0};
    QString _affinePath;
    bool _invertAffine{false};
    float _scaleSeg{1.0f};
    double _rotateDeg{0.0};
    int _flipAxis{-1};
    bool _includeTifs{false};

    // ABF++ flattening options
    bool _flatten{false};
    int _flattenIters{10};
    int _flattenDownsample{1};

    // vc_tifxyz2obj options
    bool _optNormalizeUV{false};
    bool _optAlignGrid{false};

    Tool _currentTool;
    QString _customCommand;
    QStringList _customArgs;
    QString _customLabel;

    QFile* _logFile;
    QTextStream* _logStream;

    int _nextOmpThreads{-1};
    bool _explicitVolumePath{false};
    ExecutionOptions _executionOptions;
    ExecutionOptions _completionOptions;
    bool _deliveringCompletion{false};
    bool _terminalReported{true};
    QString _pendingProcessError;

    QString _objOutputDir;
    float _objStretchFactor = 1000.0f;
    float _objMeshUnits = 1.0f;
    int _objStepSize = 20;
    QString _refineDst;
    bool _preserveConsoleOutput{false};

    // vc_merge_tifxyz parameters
    QString _mergeJsonPath;
    QString _mergePathsDir;
    QString _mergeRefSurface;
    int     _mergeRansacIters{3000};
    double  _mergeRansacMinThresh{5.0};
    double  _mergeRansacMaxThresh{10.0};
    double  _mergeRansacMadK{3.0};
    int     _mergeRansacSeed{0};
    int     _mergeAnchorCap{0};
    int     _mergeStripCols{0};

    // vc_merge_patch parameters
    QString _patchParentPath;
    QString _patchChildPath;
    bool    _patchExplicitRoles{false};
    int     _patchBorderCells{16};
    int     _patchBlendCells{6};
    int     _patchIdwK{4};
    int     _patchRansacIters{3000};
    double  _patchRansacMinThresh{5.0};
    double  _patchRansacMaxThresh{10.0};
    double  _patchRansacMadK{3.0};
    int     _patchRansacSeed{0};
    int     _patchAnchorCap{0};
};
