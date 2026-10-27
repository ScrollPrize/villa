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
#include "StraightenRequest.hpp"
#include "CommandLaunchError.hpp"
// Full definition needed: RenderSegmentParams stores a
// CommandLineToolRunner::RenderOutputFormat (nested enum) by value.
#include "CommandLineToolRunner.hpp"
#include "SurfaceCatalog.hpp"

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
 * Runs the segment operations (command-line tools, their dialogs and jobs)
 * requested from the surface panel, menus and agent bridge. Created by CWindow.
 */
class SegmentationCommandHandler : public QObject
{
    Q_OBJECT

public:
    void setEditingDestinationResolver(std::function<bool(const std::shared_ptr<QuadSurface>&)> resolver)
    { _editingDestinationResolver = std::move(resolver); }
    // --- Job structs (moved from CWindow.hpp) ---

    // One vc_grow_seg_from_seed run projects the surface and refines the copy.
    struct NeighborCopyJob {
        bool copyOut{true};
        std::unique_ptr<QTemporaryFile> paramsFile;
    };

    struct ResumeLocalJob {
        QString segmentId;
        QString paramsPath;
        std::unique_ptr<QTemporaryFile> paramsFile;
    };

    struct AlphaCompJob {
        std::unique_ptr<QTemporaryFile> paramsFile;
    };

    struct GrowPatchSeedJob {
        QString paramsPath;
        std::unique_ptr<QTemporaryFile> paramsFile;
    };

    // Arguments accepted by startGrowPatchFromSeed.
    struct GrowPatchSeedParams {
        QString volumeId;          // vpkg volume id; empty => current volume
        int     iterations{200};   // clamped/validated to [1, 100000]
        double  minAreaCm{0.002};  // >= 0
        QString outputTag{QStringLiteral("paths")};  // tag of the new surface
    };

    // Arguments accepted by startRunTrace.
    struct RunTraceParams {
        QJsonObject paramOverrides;   // merged over <volpkg>/trace_params.json
        int         ompThreads{-1};   // -1 => runner default
        QString     sourceTag{QStringLiteral("paths")};  // the surfaces the trace grows along
        QString     outputTag{QStringLiteral("traces")}; // tag of the traced surface
    };

    // Arguments accepted by startRenderSegment. Options that belong to the
    // interactive dialog are deliberately omitted.
    struct RenderSegmentParams {
        QString volumeId;             // vpkg volume id; empty => current volume
        CommandLineToolRunner::RenderOutputFormat outputFormat{
            CommandLineToolRunner::RenderOutputFormat::TifStack};
        QString outputDir;            // absolute or relative to volpkg root; empty
                                      // => the surface's artifact directory (matching
                                      // the dialog default of <artifacts>/layers)
        float   scale{1.0f};          // pixels per level-g voxel; > 0
        int     groupIdx{0};          // OME-Zarr group index; >= 0
        int     numSlices{1};         // number of slices to render; >= 1
        bool    hasVoxelSize{false};  // true => override voxel size with voxelSizeUm
        double  voxelSizeUm{0.0};     // physical voxel size (micrometers), > 0
        QString outputPattern;        // explicit output (the dialog's); overrides outputDir
        int     ompThreads{-1};       // -1 => the tool's default
        int     cropX{0}, cropY{0}, cropWidth{0}, cropHeight{0};
        double  rotateDegrees{0.0};
        int     flipAxis{-1};         // -1 => no flip
        bool    includeTifs{false};   // also write TIFF slices (Zarr output)
        bool    flatten{false};
        int     flattenIterations{10};
        int     flattenDownsample{1};
    };

    // Arguments accepted by startSlimFlatten. keepPercent defaults to full
    // resolution rather than the dialog's session default.
    struct SlimFlattenParams {
        int     iterations{50};      // flatboi iterations; > 0 (SlimJob clamps <=0 to 20)
        double  tolerance{0.0};      // 0.0 => no --tol (run all iterations)
        QString energyType{QStringLiteral("symmetric_dirichlet")}; // or "conformal"
        double  keepPercent{100.0};  // 1..100; 100 => full-res (no decimation)
        bool    inpaintHoles{false};
        QString outputName;          // the flattened surface's name; empty =>
                                     // <name>_flatboi (matching the dialog default)
    };

    // Arguments accepted by startStraighten.
    using StraightenParams = StraightenRequest;

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

    // --- Access to job state ---

    // Publishes a changed active surface to the object store (every vertex and loaded channel).
    void setSurfaceSaver(std::function<bool(QuadSurface*)> saver) { _surfaceSaver = std::move(saver); }

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
     * `outputPath` is the new surface's UUID on success, else empty. `success=false`
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
    void onRasterizeSegments(const QStringList& segmentIds);
    // Open the MergeSurfacesDialog seeded with `segmentIds` (empty = open
    // dialog with empty grid for the user to populate via "Add segments").
    void onMergeSurfaces(const QStringList& segmentIds);
    // Open the MergePatchDialog. When invoked from the surface context
    // menu we receive exactly two segment IDs; from the top menu the list
    // is empty and the user picks both via the dialog combos.
    void onMergePatch(const QStringList& segmentIds);
    void onGrowTrackPatches();
    void onAddIgnoreLabel();
    void onNeighborCopyRequested(const QString& segmentId, bool copyOut);
    void onResumeLocalGrowPatchRequested(const QString& segmentId);
    void onCreateSegmentGrowPatchFromSeed(const QVector3D& seedPoint);

    void handleNeighborCopyToolFinished(bool success);

public:
    /// Starts GrowPatch from a seed without opening a dialog. Returns true once
    /// vc_grow_seg_from_seed starts; completion uses toolFinished.
    bool startGrowPatchFromSeed(const QVector3D& seedPoint,
                                const GrowPatchSeedParams& params,
                                CommandLaunchError* error = nullptr);

    /// Starts vc_grow_seg_from_segments without opening a dialog. The traced
    /// surfaces are reported by toolFinished (their UUIDs as the output).
    bool startRunTrace(const std::string& segmentId,
                       const RunTraceParams& params,
                       CommandLaunchError* error = nullptr);

    /// Starts vc_render_tifxyz without opening a dialog. On success,
    /// `resolvedOutputDir` receives the layers directory or Zarr store.
    bool startRenderSegment(const std::string& segmentId,
                            const RenderSegmentParams& params,
                            CommandLaunchError* error = nullptr,
                            QString* resolvedOutputDir = nullptr);
    // The one render launch, from the dialog (interactive) or the bridge.
    bool startRenderImpl(const std::string& segmentId,
                         const RenderSegmentParams& params,
                         bool interactive,
                         CommandLaunchError* error,
                         QString* resolvedOutputDir);

    /// Starts SLIM/flatboi asynchronously with dialogs suppressed. Preconditions
    /// and tools are resolved before launch; completion uses flattenJobFinished.
    /// `resolvedOutputName` receives the new surface's name.
    bool startSlimFlatten(const std::string& segmentId,
                          const SlimFlattenParams& params,
                          CommandLaunchError* error = nullptr,
                          QString* resolvedOutputName = nullptr);

    /// Starts ABF++ asynchronously on a worker thread with dialogs suppressed.
    /// Completion uses flattenJobFinished.
    bool startAbfFlatten(const std::string& segmentId,
                         int iterations,
                         int downsampleFactor,
                         CommandLaunchError* error = nullptr,
                         QString* resolvedOutputName = nullptr);
    // The one ABF launch, from the dialog (interactive) or the bridge.
    bool startAbfFlattenImpl(const std::string& segmentId,
                             int iterations,
                             int downsampleFactor,
                             bool interactive,
                             CommandLaunchError* error,
                             QString* resolvedOutputName);

    /// Starts vc_straighten asynchronously with dialogs suppressed. The tool is
    /// validated before launch; completion uses flattenJobFinished.
    bool startStraighten(const std::string& segmentId,
                         const StraightenParams& params,
                         CommandLaunchError* error = nullptr,
                         QString* resolvedOutputName = nullptr);
    // The one straighten launch, from the dialog (interactive) or the bridge.
    bool startStraightenImpl(const std::string& segmentId,
                             const StraightenParams& params,
                             bool interactive,
                             CommandLaunchError* error,
                             QString* resolvedOutputName);

    /// Dialog-free Resume-opt Local (GrowPatch) launch. The interactive slot
    /// gathers the same ResumeLocalGrowParams and uses the same launch core.
    /// Completion is observable via CommandLineToolRunner::toolFinished (the
    /// resumed surface's UUID as the output).
    bool startResumeLocalGrowPatch(const std::string& segmentId,
                                   const ResumeLocalGrowParams& params,
                                   CommandLaunchError* error = nullptr);

    /// Dialog-free alpha-comp refinement launch. The interactive slot gathers
    /// the same request and uses the same preparation and launch core.
    /// Completion is observable via CommandLineToolRunner::toolFinished (the
    /// refined surface's UUID as the output).
    bool startAlphaCompRefine(const std::string& segmentId,
                              const AlphaCompRefineParams& params,
                              CommandLaunchError* error = nullptr);

    /// Dialog-free core of onCropSurfaceToValidRegion (the interactive slot wraps
    /// this). Crops the surface grid to its tightest valid bounds, writes it in
    /// place, and refreshes metrics; failures via `errorMessage` (never a dialog)
    /// distinguish failure from the already-tightest no-op. Failure sentences:
    /// "No volume package or volume loaded", "Invalid segment or segment not
    /// loaded", "Missing coordinate grid", "does not contain any valid vertices",
    /// channel-size mismatch, and save failures.
    bool cropSurfaceToValidRegion(const std::string& segmentId,
                                  QString* errorMessage = nullptr);


    // Marks a rasterized label volume and records its source surfaces (UUIDs).

private:
    /** Helper: get current volume path from state */
    QString getCurrentVolumePath() const;

    /**
     * Validate that a volume package is loaded and the surface exists.
     * Shows appropriate warning dialogs on failure.
     * If \p checkRunner is true, also verifies _cmdRunner is set and idle.
     * Returns the QuadSurface* on success, or nullptr on failure.
     */
    // A surface tools read from the store and the store they open.
    struct StoredSurface {
        std::string uuid;
        QString name;        // record name; the UUID when unnamed
        QString store;       // the store file (a hosted replica's local file)
        QString volumeUuid;  // the current volume, which outputs are bound to
    };
    // The listed surface `id` in a writable store, stored in the current volume's
    // voxels (tools read stored coordinates) or, for a tool that reads through
    // the store's view, just opening in the current volume; nullopt with
    // `error` set otherwise.
    std::optional<StoredSurface> storedSurface(const std::string& id, CommandLaunchError* error,
                                               bool throughView = false) const;
    // Same, reporting failures in a dialog; `checkRunner` also needs an idle runner.
    std::optional<StoredSurface> requireStoredSurface(const std::string& id, bool checkRunner,
                                                      bool throughView = false);
    // The listed surfaces of the user's store stored in the current volume's
    // voxels, or (throughView) that open in it, as tools that read through
    // the store's view take them.
    std::vector<const SurfaceCatalog::Row*> toolSurfaces(bool throughView = false) const;
    // The attached store's file and the current volume, for tools that read no surface.
    std::optional<StoredSurface> storeForTools(CommandLaunchError* error) const;
    std::function<bool(const std::shared_ptr<QuadSurface>&)> _editingDestinationResolver;
    std::function<bool(QuadSurface*)> _surfaceSaver;
    // The active surface when it is `segmentId`: tools change the surface on screen.
    std::shared_ptr<QuadSurface> activeSurface(const std::string& segmentId) const;
    bool publishSurface(QuadSurface* surface, const QString& action);
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
    // The tool reads volume `volumeId`: a remote one with its locator and auth.
    void configureCommandRunnerForVolume(const std::string& volumeId);

    /**
     * Fixed resume-local tracer params before caller overrides.
     */
    QJsonObject buildResumeLocalBaseParamsJson() const;
    QJsonObject buildResumeLocalParamsJson(const QJsonObject& overrides) const;

    bool startResumeLocalGrowPatchImpl(const std::string& segmentId,
                                       const ResumeLocalGrowParams& params,
                                       bool interactive,
                                       CommandLaunchError* error);
    bool startRunTraceImpl(const std::string& segmentId,
                           const RunTraceParams& params,
                           bool interactive,
                           CommandLaunchError* error);
    bool startSlimFlattenImpl(const std::string& segmentId,
                              const SlimFlattenParams& params,
                              bool interactive,
                              CommandLaunchError* error,
                              QString* resolvedOutputName);
    bool startGrowPatchFromSeedImpl(const QVector3D& seedPoint,
                                    const GrowPatchSeedParams& params,
                                    bool interactive,
                                    CommandLaunchError* error);
    bool startAlphaCompRefineImpl(const std::string& segmentId,
                                  const AlphaCompRefineParams& params,
                                  bool interactive,
                                  CommandLaunchError* error);
    static QJsonObject alphaCompRefineParamsJson(
        const AlphaCompRefineParams& params);


    QWidget* _parentWidget{nullptr};
    CState* _state{nullptr};
    CommandLineToolRunner* _cmdRunner{nullptr};
    SurfacePanelController* _surfacePanel{nullptr};
    SegmentationGrower* _segmentationGrower{nullptr};

    std::optional<NeighborCopyJob> _neighborCopyJob;
    std::optional<ResumeLocalJob> _resumeLocalJob;
    std::optional<AlphaCompJob> _alphaCompJob;
    std::optional<GrowPatchSeedJob> _growPatchSeedJob;
    // The merged trace params of the running Run Trace.
    std::unique_ptr<QTemporaryFile> _traceParamsFile;

    // Callbacks for CWindow-specific operations
    std::function<QString()> _normal3dZarrPathGetter;
    std::function<QString()> _normalGridPathGetter;
};
