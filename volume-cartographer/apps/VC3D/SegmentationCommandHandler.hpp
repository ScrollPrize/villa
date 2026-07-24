#pragma once

#include <optional>
#include <memory>
#include <functional>
#include <filesystem>
#include <string>

#include <QObject>
#include <QString>
#include <QStringList>
#include <QSet>
#include <QJsonObject>
#include <QTemporaryFile>
#include <QVector3D>

#include "elements/VolumeSelector.hpp"
#include "AlphaCompRefineRequest.hpp"
#include "CommandLaunchError.hpp"
// Full definition needed: RenderSegmentParams stores a
// CommandLineToolRunner::RenderOutputFormat (nested enum) by value.
#include "CommandLineToolRunner.hpp"

class CState;
class CommandLineToolRunner;
class SurfacePanelController;
class SegmentationGrower;
class QuadSurface;
class QWidget;
class QFileInfo;

/**
 * SegmentationCommandHandler
 *
 * Handles all segment operation slots previously defined in CWindowContextMenu.cpp.
 * Designed to be created by CWindow and wired up to the context menu signals.
 */
class SegmentationCommandHandler : public QObject
{
    Q_OBJECT

public:
    // --- Job structs (moved from CWindow.hpp) ---

    struct NeighborCopyJob {
        enum class Stage { None, FirstPass, SecondPass };
        Stage stage{Stage::None};
        QString segmentId;
        QString volumePath;
        QString resumeSurfacePath;
        QString outputDir;
        QString generatedSurfacePath;
        QString pass1JsonPath;
        QString pass2JsonPath;
        QString directoryPrefix;
        QString resumeOptMode{QStringLiteral("local")};
        int pass2OmpThreads{1};
        bool copyOut{true};
        QSet<QString> baselineEntries;
        std::unique_ptr<QTemporaryFile> pass1JsonFile;
        std::unique_ptr<QTemporaryFile> pass2JsonFile;
    };

    struct ResumeLocalJob {
        QString segmentId;
        QString outputDir;
        QString paramsPath;
        std::unique_ptr<QTemporaryFile> paramsFile;
    };

    struct AlphaCompJob {
        std::unique_ptr<QTemporaryFile> paramsFile;
    };

    struct GrowPatchSeedJob {
        QString outputDir;
        QString paramsPath;
        std::unique_ptr<QTemporaryFile> paramsFile;
    };

    // Arguments accepted by startGrowPatchFromSeed.
    struct GrowPatchSeedParams {
        QString volumeId;          // vpkg volume id; empty => current volume
        int     iterations{200};   // clamped/validated to [1, 100000]
        double  minAreaCm{0.002};  // >= 0
        QString outputDir;         // absolute or relative to volpkg root; empty =>
                                   // default head of the dialog's choice list
    };

    // Arguments accepted by startRunTrace.
    struct RunTraceParams {
        QJsonObject paramOverrides;   // merged over <volpkg>/trace_params.json
        int         ompThreads{-1};   // -1 => runner default
        QString     tgtDir;           // empty => <volpkg>/traces (created if missing)
    };

    // Arguments accepted by startRenderSegment. Options that belong to the
    // interactive dialog are deliberately omitted.
    struct RenderSegmentParams {
        QString volumeId;             // vpkg volume id; empty => current volume
        CommandLineToolRunner::RenderOutputFormat outputFormat{
            CommandLineToolRunner::RenderOutputFormat::TifStack};
        QString outputDir;            // absolute or relative to volpkg root; empty
                                      // => the segment folder (matching the dialog
                                      // default of <segment>/layers)
        float   scale{1.0f};          // pixels per level-g voxel; > 0
        int     groupIdx{0};          // OME-Zarr group index; >= 0
        int     numSlices{1};         // number of slices to render; >= 1
        bool    hasVoxelSize{false};  // true => override voxel size with voxelSizeUm
        double  voxelSizeUm{0.0};     // physical voxel size (micrometers), > 0
    };

    // Arguments accepted by startSlimFlatten. keepPercent defaults to full
    // resolution rather than the dialog's session default.
    struct SlimFlattenParams {
        int     iterations{50};      // flatboi iterations; > 0 (SlimJob clamps <=0 to 20)
        double  tolerance{0.0};      // 0.0 => no --tol (run all iterations)
        QString energyType{QStringLiteral("symmetric_dirichlet")}; // or "conformal"
        double  keepPercent{100.0};  // 1..100; 100 => full-res (no decimation)
        bool    inpaintHoles{false};
        QString outputDir;           // absolute or relative to volpkg root; empty =>
                                     // <segment>_flatboi (matching the dialog default)
    };

    // Arguments accepted by startStraighten.
    struct StraightenParams {
        bool    unbend{true};
        double  unbendSmoothCols{300.0};  // only emitted when unbend is true
        int     overlapPasses{2};         // always emitted as --overlap-pairs
        bool    orthogonalize{true};
        bool    trim{true};
        double  trimMaxEdge{100.0};       // only emitted when trim is true
        QString outputDir;                // absolute or relative to volpkg root; empty
                                          // => <segment>_straightened (dialog default)
    };

    // Inputs shared by the interactive dialog and dialog-free caller. Base
    // tracer params are fixed; paramOverrides merge over them last.
    struct ResumeLocalGrowParams {
        QString     volumeId;         // vpkg volume id; empty => current volume
        int         ompThreads{1};    // OMP_NUM_THREADS for the run; matches the dialog
                                      // default (VCSettings RESUME_LOCAL_OMP_THREADS_DEFAULT).
        QJsonObject paramOverrides;   // merged over the fixed resume-local tracer params
    };

    using AlphaCompRefineParams = AlphaCompRefineRequest;

    // --- Construction ---

    explicit SegmentationCommandHandler(QWidget* parentWidget,
                                        CState* state,
                                        QObject* parent = nullptr);
    ~SegmentationCommandHandler() override = default;

    // --- Dependency setters ---

    void setCmdRunner(CommandLineToolRunner* runner) { _cmdRunner = runner; }
    void setSurfacePanel(SurfacePanelController* panel) { _surfacePanel = panel; }
    void setSegmentationGrower(SegmentationGrower* grower) { _segmentationGrower = grower; }

    /**
     * Callback that returns the normal3d zarr path from the segmentation widget.
     * If not set, normal3d zarr path will be empty.
     */
    void setNormal3dZarrPathGetter(std::function<QString()> fn) { _normal3dZarrPathGetter = std::move(fn); }

    /**
     * Callback that returns the normal grid path paired with the current
     * volume. If not set, falls back to the project's first normal grid entry.
     */
    void setNormalGridPathGetter(std::function<QString()> fn) { _normalGridPathGetter = std::move(fn); }

    /**
     * Callback for checking if editing is in progress (from SegmentationModule).
     * Used by onRenameSurface and onCopySurfaceRequested to block during edits.
     */
    void setIsEditingCheck(std::function<bool()> fn) { _isEditingCheck = std::move(fn); }

    /**
     * Callback for clearing the surface selection in the main window.
     * Used by onMoveSegmentToPaths, onRenameSurface, etc.
     */
    void setClearSelectionCallback(std::function<void()> fn) { _clearSelectionCallback = std::move(fn); }

    /**
     * Callback for restoring selection to a renamed surface by new ID.
     * Used by onRenameSurface after the folder rename.
     */
    void setRestoreSelectionCallback(std::function<void(const std::string&)> fn) { _restoreSelectionCallback = std::move(fn); }

    // --- Access to job state ---

    std::optional<NeighborCopyJob>& neighborCopyJob() { return _neighborCopyJob; }
    const std::optional<NeighborCopyJob>& neighborCopyJob() const { return _neighborCopyJob; }
    std::optional<ResumeLocalJob>& resumeLocalJob() { return _resumeLocalJob; }
    const std::optional<ResumeLocalJob>& resumeLocalJob() const { return _resumeLocalJob; }

signals:
    /** Replaces statusBar()->showMessage() */
    void statusMessage(QString text, int timeout);

    /** Replaces QMessageBox::warning() for non-blocking warnings */
    void showWarning(QString title, QString text);

    /**
     * Emitted when a flattening job (SlimJob / ABFJob / StraightenJob) begins.
     * Emitted from both interactive slots and direct start* methods. `kind` is
     * "flatten.slim" / "flatten.abf" / "flatten.straighten"; `label` is a short
     * human string.
     */
    void flattenJobStarted(QString kind, QString label);

    /**
     * Emitted once per flattening job at its terminal state. `message` holds the
     * captured error text on failure when dialogs are suppressed;
     * `outputPath` is the artifact dir on success, else empty. `success=false`
     * with an empty message denotes a user cancel.
     */
    void flattenJobFinished(bool success, QString message, QString outputPath);

public slots:
    void onRenderSegment(const std::string& segmentId);
    void onGrowSegmentFromSegment(const std::string& segmentId);
    void onConvertToObj(const std::string& segmentId);
    void onCropSurfaceToValidRegion(const std::string& segmentId);
    void onFlipSurface(const std::string& segmentId, bool flipU);
    void onRotateSurface(const std::string& segmentId);
    void onAlphaCompRefine(const std::string& segmentId);
    void onSlimFlatten(const std::string& segmentId);
    void onStraighten(const std::string& segmentId);
    void onABFFlatten(const std::string& segmentId);
    void onExportWidthChunks(const std::string& segmentId);
    void onRasterizeSegments(const QStringList& segmentIds);
    // Open the MergeTifxyzDialog seeded with `segmentIds` (empty = open
    // dialog with empty grid for the user to populate via "Add segments").
    void onMergeTifxyz(const QStringList& segmentIds);
    // Open the MergePatchDialog. When invoked from the surface context
    // menu we receive exactly two segment IDs; from the top menu the list
    // is empty and the user picks both via the dialog combos.
    void onMergePatch(const QStringList& segmentIds);
    void onAddIgnoreLabel();
    void onNeighborCopyRequested(const QString& segmentId, bool copyOut);
    void onResumeLocalGrowPatchRequested(const QString& segmentId);
    void onCreateSegmentGrowPatchFromSeed(const QVector3D& seedPoint);
    void onReloadFromBackup(const QString& segmentId, int backupIndex);
    void onCopySurfaceRequested(const QString& segmentId);
    void onMoveSegmentToPaths(const QString& segmentId);
    void onRenameSurface(const QString& segmentId);

    void handleNeighborCopyToolFinished(bool success);

    // Internal helpers exposed as slots for QTimer::singleShot
    void launchNeighborCopySecondPass();

public:
    /// Starts GrowPatch from a seed without opening a dialog. Returns true once
    /// vc_grow_seg_from_seed starts; completion uses toolFinished.
    bool startGrowPatchFromSeed(const QVector3D& seedPoint,
                                const GrowPatchSeedParams& params,
                                CommandLaunchError* error = nullptr);

    /// Output dir from the most recent successful startGrowPatchFromSeed while its
    /// job is active; empty when none pending.
    QString activeGrowPatchOutputDir() const
    {
        return _growPatchSeedJob ? _growPatchSeedJob->outputDir : QString();
    }

    /// Starts vc_grow_seg_from_segments without opening a dialog. On success,
    /// `resolvedOutputDir` receives the target directory when provided.
    bool startRunTrace(const std::string& segmentId,
                       const RunTraceParams& params,
                       CommandLaunchError* error = nullptr,
                       QString* resolvedOutputDir = nullptr);

    /// Starts vc_render_tifxyz without opening a dialog. On success,
    /// `resolvedOutputDir` receives the layers directory or Zarr store.
    bool startRenderSegment(const std::string& segmentId,
                            const RenderSegmentParams& params,
                            CommandLaunchError* error = nullptr,
                            QString* resolvedOutputDir = nullptr);

    /// Starts SLIM/flatboi asynchronously with dialogs suppressed. Preconditions
    /// and tools are resolved before launch; completion uses flattenJobFinished.
    bool startSlimFlatten(const std::string& segmentId,
                          const SlimFlattenParams& params,
                          CommandLaunchError* error = nullptr,
                          QString* resolvedOutputDir = nullptr);

    /// Starts ABF++ asynchronously on a worker thread with dialogs suppressed.
    /// Completion uses flattenJobFinished.
    bool startAbfFlatten(const std::string& segmentId,
                         int iterations,
                         int downsampleFactor,
                         CommandLaunchError* error = nullptr,
                         QString* resolvedOutputDir = nullptr);

    /// Starts vc_straighten asynchronously with dialogs suppressed. The tool and
    /// output directory are validated before launch; completion uses
    /// flattenJobFinished.
    bool startStraighten(const std::string& segmentId,
                         const StraightenParams& params,
                         CommandLaunchError* error = nullptr,
                         QString* resolvedOutputDir = nullptr);

    /// Dialog-free Resume-opt Local (GrowPatch) launch. The interactive slot
    /// gathers the same ResumeLocalGrowParams and uses the same launch core.
    /// Completion is observable via CommandLineToolRunner::toolFinished. On
    /// success `resolvedOutputDir` receives the target directory when provided.
    bool startResumeLocalGrowPatch(const std::string& segmentId,
                                   const ResumeLocalGrowParams& params,
                                   CommandLaunchError* error = nullptr,
                                   QString* resolvedOutputDir = nullptr);

    /// Dialog-free alpha-comp refinement launch. The interactive slot gathers
    /// the same request and uses the same preparation and launch core.
    /// Completion is observable via CommandLineToolRunner::toolFinished. On
    /// success `resolvedOutputDir` (when non-null) receives the refined output
    /// directory.
    bool startAlphaCompRefine(const std::string& segmentId,
                              const AlphaCompRefineParams& params,
                              CommandLaunchError* error = nullptr,
                              QString* resolvedOutputDir = nullptr);

    /// Dialog-free core of onCropSurfaceToValidRegion (the interactive slot wraps
    /// this). Crops the surface grid to its tightest valid bounds, writes it in
    /// place, and refreshes metrics; failures via `errorMessage` (never a dialog)
    /// distinguish failure from the already-tightest no-op. Failure sentences:
    /// "No volume package or volume loaded", "Invalid segment or segment not
    /// loaded", "Missing coordinate grid", "does not contain any valid vertices",
    /// channel-size mismatch, and save failures.
    bool cropSurfaceToValidRegion(const std::string& segmentId,
                                  QString* errorMessage = nullptr);

    /// Dialog-free core of onRenameSurface: renames the segment folder + meta.json
    /// UUID from `oldId` to `newName`, keeping the editing guard, name validation
    /// (^[a-zA-Z0-9_-]+$), collision check, CState/vpkg cleanup, and rollback.
    /// Never opens QInputDialog / QMessageBox. Returns true on success; on failure
    /// sets `err` (when non-null) to a classifiable sentence: "editing in
    /// progress", "invalid name", "name unchanged", "segment not found", "name
    /// exists", "no volume package", or a metadata/rename error string.
    bool renameSurfaceHeadless(const QString& oldId, const QString& newName,
                               QString* err = nullptr);

    QString findNewNeighborSurface(const NeighborCopyJob& job) const;
    bool startNeighborCopyPass(const QString& paramsPath,
                               const QString& resumeSurface,
                               const QString& resumeOpt,
                               int ompThreads);
    bool appendRasterizationMetadata(const QString& outputZarrPath,
                                   const QStringList& segmentIds,
                                   const QStringList& segmentPaths) const;

private:
    /** Helper: get current volume path from state */
    QString getCurrentVolumePath() const;
    QString getCurrentRenderVolumePath(QString* remoteUrlOut = nullptr) const;

    /**
     * Validate that a volume package is loaded and the surface exists.
     * Shows appropriate warning dialogs on failure.
     * If \p checkRunner is true, also verifies _cmdRunner is set and idle.
     * Returns the QuadSurface* on success, or nullptr on failure.
     */
    QuadSurface* requireSurfaceAndRunner(const std::string& segmentId,
                                          bool checkRunner = true);

    /**
     * Build the list of available volumes from the current vpkg, with the
     * currently-loaded volume selected as default.  Returns an empty vector
     * (and shows a warning) if no volumes are available.
     * If \p defaultOut is non-null, receives the default volume ID.
     */
    QVector<VolumeSelector::VolumeOption> buildVolumeOptionList(
        QString* defaultOut = nullptr);
    void configureCommandRunnerRemoteAuthForVolumePath(const QString& volumePath);

    /**
     * Fixed resume-local tracer params before caller overrides.
     */
    QJsonObject buildResumeLocalBaseParamsJson() const;
    QJsonObject buildResumeLocalParamsJson(const QJsonObject& overrides) const;

    bool startResumeLocalGrowPatchImpl(const std::string& segmentId,
                                       const ResumeLocalGrowParams& params,
                                       bool interactive,
                                       CommandLaunchError* error,
                                       QString* resolvedOutputDir);
    bool startRunTraceImpl(const std::string& segmentId,
                           const RunTraceParams& params,
                           bool interactive,
                           CommandLaunchError* error,
                           QString* resolvedOutputDir);
    bool startGrowPatchFromSeedImpl(const QVector3D& seedPoint,
                                    const GrowPatchSeedParams& params,
                                    bool interactive,
                                    CommandLaunchError* error);
    bool startAlphaCompRefineImpl(const std::string& segmentId,
                                  const AlphaCompRefineParams& params,
                                  bool interactive,
                                  CommandLaunchError* error,
                                  QString* resolvedOutputDir);
    static QJsonObject alphaCompRefineParamsJson(
        const AlphaCompRefineParams& params);

    /**
     * Resolve a segment launch's output directory: the surface's own parent
     * directory, falling back to <volpkg>/paths, created if missing. Returns the
     * absolute path, or an empty string with \p errorMessage set on a directory
     * creation failure.
     */
    QString resolveSegmentOutputDir(const std::filesystem::path& surfacePath,
                                    QString* errorMessage) const;

    /**
     * The default alpha-comp refinement output path (<src>_refined, preserving
     * any file suffix), shared by the dialog and direct-launch defaults.
     */
    static QString defaultRefinedOutputPath(const QFileInfo& srcInfo);

    QWidget* _parentWidget{nullptr};
    CState* _state{nullptr};
    CommandLineToolRunner* _cmdRunner{nullptr};
    SurfacePanelController* _surfacePanel{nullptr};
    SegmentationGrower* _segmentationGrower{nullptr};

    std::optional<NeighborCopyJob> _neighborCopyJob;
    std::optional<ResumeLocalJob> _resumeLocalJob;
    std::optional<AlphaCompJob> _alphaCompJob;
    std::optional<GrowPatchSeedJob> _growPatchSeedJob;

    // Callbacks for CWindow-specific operations
    std::function<QString()> _normal3dZarrPathGetter;
    std::function<QString()> _normalGridPathGetter;
    std::function<bool()> _isEditingCheck;
    std::function<void()> _clearSelectionCallback;
    std::function<void(const std::string&)> _restoreSelectionCallback;
};
