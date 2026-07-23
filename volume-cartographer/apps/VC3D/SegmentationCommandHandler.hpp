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

struct CommandLaunchError {
    enum Kind {
        Other,
        InvalidState,
        SegmentNotFound,
        VolumeNotFound,
        InputNotFound,
        RemoteVolume,
        ToolUnavailable,
        Busy,
    };

    Kind kind{Other};
    QString message;
};

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

    struct GrowPatchSeedJob {
        QString outputDir;
        QString paramsPath;
        std::unique_ptr<QTemporaryFile> paramsFile;
    };

    // Params for the headless startGrowPatchFromSeed (mirror of the interactive
    // GrowPatch dialog; SPEC §4).
    struct GrowPatchSeedParams {
        QString volumeId;          // vpkg volume id; empty => current volume
        int     iterations{200};   // clamped/validated to [1, 100000]
        double  minAreaCm{0.002};  // >= 0
        QString outputDir;         // absolute or relative to volpkg root; empty =>
                                   // default head of the dialog's choice list
    };

    // Params for the headless startRunTrace (mirror of TraceParamsDialog;
    // SPEC §14.4 / §15.4).
    struct RunTraceParams {
        QJsonObject paramOverrides;   // merged over <volpkg>/trace_params.json
        int         ompThreads{-1};   // -1 => runner default
        QString     tgtDir;           // empty => <volpkg>/traces (created if missing)
    };

    // Params for the headless startRenderSegment (reduced from RenderParamsDialog;
    // SPEC §19). Advanced dialog-only options (crop/affine/rotate/flip/flatten/
    // include-tifs/composite/alpha) are deliberately omitted.
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

    // Params for the headless startSlimFlatten (mirror of SlimFlattenDialog;
    // SPEC §20). Headless keepPercent defaults to 100 (full-res, no decimation),
    // not the dialog's 1.5% session default.
    struct SlimFlattenParams {
        int     iterations{50};      // flatboi iterations; > 0 (SlimJob clamps <=0 to 20)
        double  tolerance{0.0};      // 0.0 => no --tol (run all iterations)
        QString energyType{QStringLiteral("symmetric_dirichlet")}; // or "conformal"
        double  keepPercent{100.0};  // 1..100; 100 => full-res (no decimation)
        bool    inpaintHoles{false};
        QString outputDir;           // absolute or relative to volpkg root; empty =>
                                     // <segment>_flatboi (matching the dialog default)
    };

    // Params for the headless startStraighten (mirror of StraightenDialog;
    // SPEC §20). Defaults mirror the dialog's initial state.
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

    // Params for the headless startResumeLocalGrowPatch. Base tracer params are
    // fixed (matching the interactive path); paramOverrides merge over them last,
    // like the dialog's JsonProfileEditor "extra params".
    struct ResumeLocalGrowParams {
        QString     volumeId;         // vpkg volume id; empty => current volume
        int         ompThreads{1};    // OMP_NUM_THREADS for the run; matches the dialog
                                      // default (VCSettings RESUME_LOCAL_OMP_THREADS_DEFAULT).
        QJsonObject paramOverrides;   // merged over the fixed resume-local tracer params
    };

    // Params for the headless startAlphaCompRefine (mirror of AlphaCompRefineDialog;
    // defaults = its session defaults, ToolDialogs.cpp AlphaCompRefineDialog::s_*).
    struct AlphaCompRefineParams {
        bool    refine{true};
        double  start{-6.0};
        double  stop{30.0};
        double  step{2.0};
        int     low{26};
        int     high{255};
        double  borderOff{1.0};
        int     radius{3};
        bool    genVertexColor{false};
        bool    overwrite{true};
        double  readerScale{0.5};
        QString scaleGroup{QStringLiteral("1")};
        int     ompThreads{-1};       // -1 => runner default (no OMP override)
        QString outputDir;            // absolute or relative to volpkg root; empty =>
                                      // <segment>_refined (matching the dialog default)
    };

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
     * Emitted when a flattening job (SlimJob / ABFJob / StraightenJob) begins so
     * the agent bridge can track it as a source:"flatten" job (SPEC §8.3, §20).
     * Emitted from BOTH interactive slots and headless start* methods. `kind` is
     * "flatten.slim" / "flatten.abf" / "flatten.straighten"; `label` is a short
     * human string.
     */
    void flattenJobStarted(QString kind, QString label);

    /**
     * Emitted once per flattening job at its terminal state (SPEC §8.3, §20).
     * `message` holds the captured error text on failure (suppressed dialog);
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
    /// Headless GrowPatch-from-seed launch (mirror of
    /// onCreateSegmentGrowPatchFromSeed): failures return through `error`, never a
    /// dialog; returns true when vc_grow_seg_from_seed started. Completion
    /// observable via CommandLineToolRunner::toolFinished as in the interactive
    /// path.
    bool startGrowPatchFromSeed(const QVector3D& seedPoint,
                                const GrowPatchSeedParams& params,
                                CommandLaunchError* error = nullptr);

    /// Output dir from the most recent successful startGrowPatchFromSeed while its
    /// job is active; empty when none pending.
    QString activeGrowPatchOutputDir() const
    {
        return _growPatchSeedJob ? _growPatchSeedJob->outputDir : QString();
    }

    /// Headless Run-Trace launch (vc_grow_seg_from_segments; mirror of
    /// onGrowSegmentFromSegment): preconditions, params-JSON merge/write, launch;
    /// failures via `error` (never a dialog). Completion
    /// is observable via CommandLineToolRunner::toolFinished.
    /// On success, `resolvedOutputDir` (when non-null) receives the target dir.
    bool startRunTrace(const std::string& segmentId,
                       const RunTraceParams& params,
                       CommandLaunchError* error = nullptr,
                       QString* resolvedOutputDir = nullptr);

    /// Headless Render launch (vc_render_tifxyz; mirror of onRenderSegment):
    /// preconditions, volume/output resolution, launch; failures via
    /// `error` (never a dialog). Completion observable via
    /// CommandLineToolRunner::toolFinished.
    /// `error.kind` provides a stable machine-readable failure category. On success,
    /// `resolvedOutputDir` (when non-null) receives the output artifact path
    /// (the layers dir for a TIFF stack, or the .zarr store).
    bool startRenderSegment(const std::string& segmentId,
                            const RenderSegmentParams& params,
                            CommandLaunchError* error = nullptr,
                            QString* resolvedOutputDir = nullptr);

    /// Headless SLIM/flatboi flatten launch. Resolves all preconditions and tools
    /// up front (so the SlimJob never hits a synchronous dialog), then constructs a
    /// SlimJob with dialogs suppressed; runs async, completion via
    /// flattenJobFinished (SPEC §20). Failures return through `error`, never a dialog.
    /// `error.kind` provides a stable machine-readable failure category. On
    /// success `resolvedOutputDir` (when non-null) receives the
    /// flattened tifxyz directory.
    bool startSlimFlatten(const std::string& segmentId,
                          const SlimFlattenParams& params,
                          CommandLaunchError* error = nullptr,
                          QString* resolvedOutputDir = nullptr);

    /// Headless ABF++ flatten launch (in-process, QtConcurrent). Constructs an
    /// ABFJob with dialogs suppressed; runs on a worker thread, completion via
    /// flattenJobFinished (SPEC §20). Failure sentences: "No volume package
    /// loaded", "Invalid segment" (-32007 segment). On success `resolvedOutputDir`
    /// receives the <segment>_abf directory.
    bool startAbfFlatten(const std::string& segmentId,
                         int iterations,
                         int downsampleFactor,
                         CommandLaunchError* error = nullptr,
                         QString* resolvedOutputDir = nullptr);

    /// Headless vc_straighten launch. Resolves the tool and validates the output
    /// dir up front, constructs a StraightenJob with dialogs suppressed; async,
    /// completion via flattenJobFinished (SPEC §20). Failure sentences: "No volume
    /// package or volume loaded", "Invalid segment" (-32007 segment),
    /// "vc_straighten not found" (-32006), "Output directory already exists"
    /// (-32005). On success `resolvedOutputDir` receives the straightened dir.
    bool startStraighten(const std::string& segmentId,
                         const StraightenParams& params,
                         CommandLaunchError* error = nullptr,
                         QString* resolvedOutputDir = nullptr);

    /// Headless Resume-opt Local (GrowPatch) launch (mirror of
    /// onResumeLocalGrowPatchRequested): preconditions, params-JSON, and
    /// vc_grow_seg_from_segments launch (Tool::NeighborCopy, resumeOpt "local");
    /// failures return through `error` (never a dialog). Completion is observable
    /// via CommandLineToolRunner::toolFinished (the CWindow slot reloads surfaces
    /// and clears resumeLocalJob()). On success `resolvedOutputDir` (when
    /// non-null) receives the target segment directory.
    bool startResumeLocalGrowPatch(const std::string& segmentId,
                                   const ResumeLocalGrowParams& params,
                                   CommandLaunchError* error = nullptr,
                                   QString* resolvedOutputDir = nullptr);

    /// Headless alpha-comp refinement launch (vc_objrefine, Tool::AlphaCompRefine;
    /// mirror of onAlphaCompRefine): preconditions, output-path derivation,
    /// params-JSON write, launch; failures return through `error` (never a dialog).
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
    /// so the bridge returns a real error, not a false success. Returns true on
    /// success, including the already-tightest no-op. Failure sentences: "No volume
    /// package or volume loaded", "Invalid segment or segment not loaded", "Missing
    /// coordinate grid", "does not contain any valid vertices", channel-size
    /// mismatch, and save failures.
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
     * Fixed resume-local tracer params (normal-grid/normal3d paths + constant
     * opt-step/radius/iteration settings) before overrides; shared by the
     * interactive slot and startResumeLocalGrowPatch so the two can't drift.
     */
    QJsonObject buildResumeLocalBaseParamsJson() const;

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
     * any file suffix), shared by the interactive dialog default and the
     * headless empty-outputDir default.
     */
    static QString defaultRefinedOutputPath(const QFileInfo& srcInfo);

    QWidget* _parentWidget{nullptr};
    CState* _state{nullptr};
    CommandLineToolRunner* _cmdRunner{nullptr};
    SurfacePanelController* _surfacePanel{nullptr};
    SegmentationGrower* _segmentationGrower{nullptr};

    std::optional<NeighborCopyJob> _neighborCopyJob;
    std::optional<ResumeLocalJob> _resumeLocalJob;
    std::optional<GrowPatchSeedJob> _growPatchSeedJob;

    // Callbacks for CWindow-specific operations
    std::function<QString()> _normal3dZarrPathGetter;
    std::function<QString()> _normalGridPathGetter;
    std::function<bool()> _isEditingCheck;
    std::function<void()> _clearSelectionCallback;
    std::function<void(const std::string&)> _restoreSelectionCallback;
};
