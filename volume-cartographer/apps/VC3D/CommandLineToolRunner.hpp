#pragma once

#include "ToolProcess.hpp"
#include <functional>

#include <QObject>
#include <QString>
#include <QStringList>
#include <QDialog>

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
    // Tools read and write surfaces in this object store (the attached store's
    // file; a hosted replica's local file, which tools sync themselves), in the
    // voxels of their volume at `voxelScale` of its own (--voxel-scale).
    void setStore(const QString& storePath, double voxelScale);
    // The stored surface a render or OBJ export reads.
    void setSurface(const QString& uuid);
    void setOutputPattern(const QString& pattern);

    // `store` and `volume`, when given, are the run's store contract (ToolProcess::Launch).
    bool executeCustomCommand(const QString& command,
                             const QStringList& args,
                             const QString& label = QString(),
                             ExecutionOptions options = {},
                             const QString& store = {},
                             const QString& volume = {},
                             double voxelScale = 1);

    // tool specific params 
    void setRenderParams(float scale, int resolution, int layers);
    void setRenderVoxelSize(double voxelSizeUm, bool enabled);
    // Selects --tif-output vs --zarr-output for the next RenderTifXYZ run (default
    // TifStack; interactive re-asserts TifStack to avoid leaking a prior Zarr choice).
    void setRenderOutputFormat(RenderOutputFormat format);
    // The volume a render reads the surface in, through the store's view of it,
    // at `voxelScale` of its voxels.
    void setRenderVolume(const QString& volumeUuid, double voxelScale);
    void setRenderAdvanced(int cropX, int cropY, int cropWidth, int cropHeight,
                           double rotateDegrees, int flipAxis);
    // vc_grow_seg_from_segments: grow `srcSurface` along the surfaces of `volumeUuid`
    // tagged `sourceTag` into new surfaces tagged `outputTag` and bound to the volume.
    void setTraceParams(QString volumePath, QString volumeUuid, QString sourceTag, QString outputTag,
                        QString jsonParams, QString srcSurface);
    // vc_tifxyz2obj: export a stored surface to `objName` in its artifact directory.
    // vc_tifxyz2obj: the surface exported in `volumeUuid`'s voxels.
    void setToObjParams(QString surfaceUuid, QString objName, QString volumeUuid);
    void setToObjOptions(bool normalizeUV, bool alignGrid);
    // vc_objrefine: `dstSurface` is a new surface's name, or `srcSurface` to refine in place;
    // the surface is read in the voxels of `volumeUuid`, the volume at `volumePath`.
    // `newName` names the refined surface; empty refines `srcSurface` in place.
    void setObjRefineParams(const QString& volumePath,
                            const QString& srcSurface,
                            const QString& newName,
                            const QString& jsonParams,
                            const QString& volumeUuid);
    // vc_grow_seg_from_seed: grow from (resume) `resumeSurface` into a new surface
    // tagged `outputTag` and bound to `volumeUuid`.
    void setNeighborCopyParams(const QString& volumePath,
                               const QString& paramsJson,
                               const QString& resumeSurface,
                               const QString& volumeUuid,
                               const QString& outputTag,
                               const QString& resumeOpt);
    // vc_merge_tifxyz: --merge lists the input surfaces by UUID; the merge is a
    // new surface tagged `outputTag` and bound to `volumeUuid`. The remaining
    // args are RANSAC + blend tunables that all have working defaults.
    void setMergeParams(const QString& mergeJsonPath,
                        const QString& volumeUuid,
                        const QString& outputTag,
                        const QString& refSurface,
                        int ransacIters,
                        double ransacMinThresh,
                        double ransacMaxThresh,
                        double ransacMadK,
                        int ransacSeed,
                        int anchorCap,
                        int stripCols);
    // vc_merge_patch: two stored surfaces; the binary auto-detects parent vs
    // child by valid-cell count unless explicitRoles is set, in which case
    // we pass --parent / --child explicitly. The parent is edited in place
    // (its revision history keeps the pre-patch state).
    void setMergePatchParams(const QString& parentSurface,
                             const QString& childSurface,
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
    // Surfaces the current (or last) run created, from its `surface_uuid:` lines.
    const QStringList& createdSurfaces() const { return _createdSurfaces; }
    
    void showConsoleOutput();
    void setIncludeTifs(bool include);
    // Applies to the next execute() attempt only.
    void setNextOmpThreads(int threads);

#ifdef VC_TESTING
    QStringList argumentsForTesting(Tool tool) { return vc3d::ToolProcess::arguments(launchFor(tool)); }
#endif

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

private:
    QStringList buildArguments(Tool tool);
    // The tool's arguments and store contract (all but the executable and environment).
    vc3d::ToolProcess::Launch launchFor(Tool tool);
    QString toolName(Tool tool) const;
    QString getOutputPath() const;
    void finishExecution(bool success,
                         const QString& message,
                         const QString& outputPath = {},
                         bool copyToClipboard = false);

    std::function<QString()> _currentVolumePath;
    ProgressUtil* _progressUtil;
    
    vc3d::ToolProcess* _run = nullptr;
    ConsoleOutputWidget* _consoleOutput;
    QDialog* _consoleDialog;
    
    QString _volumePath;
    QString _remoteVolumeUrl;
    QString _remoteAccessKey;
    QString _remoteSecretKey;
    QString _remoteSessionToken;
    QString _remoteRegion;
    QString _storePath;
    double _voxelScale = 1;
    QString _surfaceUuid;
    QString _outputPattern;
    QString _sourceTag;
    QString _outputTag;
    QString _volumeUuid;
    QString _sourceSurfaceUuid;
    QString _objPath;
    QString _jsonParams;
    QString _resumeSurfaceUuid;
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
    QString _renderVolumeUuid;
    double _renderVoxelScale = 1;
    double _rotateDeg{0.0};
    int _flipAxis{-1};
    bool _includeTifs{false};

    // vc_tifxyz2obj options
    bool _optNormalizeUV{false};
    bool _optAlignGrid{false};

    Tool _currentTool;
    QString _customCommand;
    QStringList _customArgs;
    QString _customStore;
    QString _customVolume;
    double _customVoxelScale = 1;
    QString _customLabel;

    int _nextOmpThreads{-1};
    bool _explicitVolumePath{false};
    ExecutionOptions _executionOptions;
    ExecutionOptions _completionOptions;
    bool _deliveringCompletion{false};
    bool _terminalReported{true};

    QString _refineName;  // the refined surface's name; empty: refined in place
    QStringList _createdSurfaces;
    QString _exportedObj;

    // vc_merge_tifxyz parameters
    QString _mergeJsonPath;
    QString _mergeRefSurface;
    int     _mergeRansacIters{3000};
    double  _mergeRansacMinThresh{5.0};
    double  _mergeRansacMaxThresh{10.0};
    double  _mergeRansacMadK{3.0};
    int     _mergeRansacSeed{0};
    int     _mergeAnchorCap{0};
    int     _mergeStripCols{0};

    // vc_merge_patch parameters
    QString _patchParentSurface;
    QString _patchChildSurface;
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
