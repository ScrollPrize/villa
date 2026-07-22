#pragma once

#include <optional>
#include <memory>
#include <functional>
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

    // Parameters the interactive GrowPatch dialog collects, packaged so the
    // non-interactive (agent bridge) path can supply them directly. See
    // startGrowPatchFromSeed and apps/VC3D/agent_bridge/SPEC.md §4.
    struct GrowPatchSeedParams {
        QString volumeId;          // vpkg volume id; empty => current volume
        int     iterations{200};   // clamped/validated to [1, 100000]
        double  minAreaCm{0.002};  // >= 0
        QString outputDir;         // absolute or relative to volpkg root; empty =>
                                   // default head of the dialog's choice list
    };

    // Parameters the interactive Run-Trace (TraceParamsDialog) collects, packaged
    // so the non-interactive (agent bridge) path can supply them directly. See
    // startRunTrace and apps/VC3D/agent_bridge/SPEC.md §14.4 / §15.4.
    struct RunTraceParams {
        QJsonObject paramOverrides;   // merged over <volpkg>/trace_params.json
        int         ompThreads{-1};   // -1 => runner default
        QString     tgtDir;           // empty => <volpkg>/traces (created if missing)
    };

    // Parameters the interactive Render (RenderParamsDialog) collects, reduced to
    // the core surface the non-interactive (agent bridge) path threads through.
    // See startRenderSegment and apps/VC3D/agent_bridge/SPEC.md §19. Advanced
    // dialog-only options (crop/affine/rotate/flip/flatten/include-tifs/composite
    // /alpha) are deliberately omitted from the headless surface.
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

    // Parameters the interactive SlimFlattenDialog collects, packaged so the
    // non-interactive (agent bridge) path can supply them directly. See
    // startSlimFlatten and apps/VC3D/agent_bridge/SPEC.md §20. The headless
    // default keepPercent is 100 (full-resolution SLIM, no decimation) -- the
    // production-recommended path -- rather than the dialog's 1.5% session
    // default.
    struct SlimFlattenParams {
        int     iterations{50};      // flatboi iterations; > 0 (SlimJob clamps <=0 to 20)
        double  tolerance{0.0};      // 0.0 => no --tol (run all iterations)
        QString energyType{QStringLiteral("symmetric_dirichlet")}; // or "conformal"
        double  keepPercent{100.0};  // 1..100; 100 => full-res (no decimation)
        bool    inpaintHoles{false};
        QString outputDir;           // absolute or relative to volpkg root; empty =>
                                     // <segment>_flatboi (matching the dialog default)
    };

    // Parameters the interactive StraightenDialog collects, packaged so the
    // non-interactive (agent bridge) path can supply them directly. See
    // startStraighten and apps/VC3D/agent_bridge/SPEC.md §20. Defaults mirror
    // the dialog's initial state.
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
     * Emitted by the flattening job classes (SlimJob / ABFJob / StraightenJob)
     * when a job begins. Lets the agent bridge track flattening as a
     * source:"flatten" job (SPEC §8.3, §20). Emitted from BOTH the interactive
     * slots and the headless start* methods, so the bridge registers
     * human-initiated flattens as external jobs too. `kind` is one of
     * "flatten.slim" / "flatten.abf" / "flatten.straighten"; `label` is a short
     * human string. No-op when nothing is connected (bridge disabled).
     */
    void flattenJobStarted(QString kind, QString label);

    /**
     * Emitted by the flattening job classes at every terminal state
     * (success/failure/cancel). Carries the outcome for the bridge to surface
     * as the source:"flatten" job's completion (SPEC §8.3, §20). On failure
     * `message` holds the captured error text that WOULD have been shown in a
     * QMessageBox (suppressed on headless runs). `outputPath` is the flattened
     * artifact directory on success, empty otherwise. Emitted exactly once per
     * job. `success=false` with an empty message denotes a user cancel.
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
    /// Headless GrowPatch-from-seed launch. Performs ALL validation and
    /// execution that onCreateSegmentGrowPatchFromSeed performs today, but
    /// reports failures through `errorMessage` (never via QMessageBox) and never
    /// opens a dialog. Returns true when the vc_grow_seg_from_seed process was
    /// started (job accepted), false otherwise. On success, completion is
    /// observable via CommandLineToolRunner::toolFinished exactly as in the
    /// interactive path (same one-shot handler: meta.json coordinate identity
    /// fixup, VolumePkg::refreshSegmentations, surface panel reload).
    bool startGrowPatchFromSeed(const QVector3D& seedPoint,
                                const GrowPatchSeedParams& params,
                                QString* errorMessage = nullptr);

    /// Output directory resolved by the most recent successful
    /// startGrowPatchFromSeed call while its job is still active. Empty when no
    /// GrowPatch seed job is pending.
    QString activeGrowPatchOutputDir() const
    {
        return _growPatchSeedJob ? _growPatchSeedJob->outputDir : QString();
    }

    /// Headless Run-Trace launch (vc_grow_seg_from_segments). Performs ALL the
    /// preconditions, params-JSON merge/write, and launch that
    /// onGrowSegmentFromSegment performs today, but reports failures through
    /// `errorMessage` (never via QMessageBox) and never opens a dialog. Returns
    /// true when the tool process was launched (job accepted), false otherwise.
    /// Completion is observable via CommandLineToolRunner::toolFinished exactly
    /// as in the interactive path. Distinct failure sentences let the bridge map
    /// to JSON-RPC codes: "No volume package or volume loaded", "Invalid segment"
    /// (-32007 segment), "remote" (-32009), "trace_params.json not found"
    /// (-32007 file), "already running" (-32004), "Command line tools not
    /// available" (-32006), and generic write/create failures (-32005).
    /// On success, `resolvedOutputDir` (when non-null) receives the target dir.
    bool startRunTrace(const std::string& segmentId,
                       const RunTraceParams& params,
                       QString* errorMessage = nullptr,
                       QString* resolvedOutputDir = nullptr);

    /// Headless Render launch (vc_render_tifxyz). Performs ALL preconditions,
    /// volume/output resolution, and launch that onRenderSegment performs today,
    /// but reports failures through `errorMessage` (never via QMessageBox) and
    /// never opens the RenderParamsDialog. Returns true when the tool process was
    /// launched (job accepted), false otherwise. Completion is observable via
    /// CommandLineToolRunner::toolFinished exactly as in the interactive path.
    /// Distinct failure sentences let the bridge map to JSON-RPC codes:
    /// "No volume package or volume loaded" / "No volume loaded", "Invalid
    /// segment" (-32007 segment), "Unknown volume id" (-32007 volume),
    /// "vc_render_tifxyz not found" (-32006), "already running" (-32004), and
    /// generic create/validation failures (-32005). On success,
    /// `resolvedOutputDir` (when non-null) receives the output artifact path
    /// (the layers dir for a TIFF stack, or the .zarr store).
    bool startRenderSegment(const std::string& segmentId,
                            const RenderSegmentParams& params,
                            QString* errorMessage = nullptr,
                            QString* resolvedOutputDir = nullptr);

    /// Headless SLIM/flatboi flatten launch. Performs ALL preconditions and
    /// tool resolution up front (so the constructed SlimJob never hits a
    /// synchronous dialog), constructs a SlimJob with dialogs suppressed, and
    /// returns true when the job was accepted (its multi-stage QProcess
    /// pipeline runs asynchronously). Never opens the SlimFlattenDialog and
    /// never pops a QMessageBox -- all failures report through `errorMessage`
    /// and completion is observable via flattenJobFinished (SPEC §20). Distinct
    /// failure sentences let the bridge map to JSON-RPC codes: "No volume
    /// package or volume loaded", "Invalid segment" (-32007 segment), "flatboi
    /// not found" / "vc_tifxyz2obj not found" / ... (-32006), and generic
    /// output-dir failures (-32005). On success `resolvedOutputDir` (when
    /// non-null) receives the flattened tifxyz directory.
    bool startSlimFlatten(const std::string& segmentId,
                          const SlimFlattenParams& params,
                          QString* errorMessage = nullptr,
                          QString* resolvedOutputDir = nullptr);

    /// Headless ABF++ flatten launch (in-process, QtConcurrent). Constructs an
    /// ABFJob with dialogs suppressed; the flatten runs on a worker thread and
    /// completion is observable via flattenJobFinished (SPEC §20). Never opens
    /// the ABFFlattenDialog or a QMessageBox. Failure sentences: "No volume
    /// package loaded", "Invalid segment" (-32007 segment). On success
    /// `resolvedOutputDir` receives the <segment>_abf directory.
    bool startAbfFlatten(const std::string& segmentId,
                         int iterations,
                         int downsampleFactor,
                         QString* errorMessage = nullptr,
                         QString* resolvedOutputDir = nullptr);

    /// Headless vc_straighten launch. Resolves the tool and validates the
    /// output dir up front, constructs a StraightenJob with dialogs suppressed,
    /// and returns true when the subprocess was launched. Never opens the
    /// StraightenDialog or a QMessageBox; completion observable via
    /// flattenJobFinished (SPEC §20). Failure sentences: "No volume package or
    /// volume loaded", "Invalid segment" (-32007 segment), "vc_straighten not
    /// found" (-32006), "Output directory already exists" (-32005). On success
    /// `resolvedOutputDir` receives the straightened tifxyz directory.
    bool startStraighten(const std::string& segmentId,
                         const StraightenParams& params,
                         QString* errorMessage = nullptr,
                         QString* resolvedOutputDir = nullptr);

    /// Dialog-free core of onRenameSurface: renames the segment folder + meta.json
    /// UUID from `oldId` to `newName`, keeping the editing guard, the
    /// ^[a-zA-Z0-9_-]+$ name validation, the collision check, the CState/vpkg
    /// cleanup, and the rollback paths. Never opens QInputDialog / QMessageBox.
    /// Returns true on success. On failure returns false and, when `err` is
    /// non-null, sets it to a classifiable sentence: "editing in progress",
    /// "invalid name", "name unchanged", "segment not found", "name exists",
    /// "no volume package", or a metadata/rename error string. The interactive
    /// onRenameSurface calls this after its dialog; the agent bridge calls it
    /// directly (ADDITIONS_SPEC item 5).
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
