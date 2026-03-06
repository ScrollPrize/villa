#pragma once

#include <optional>
#include <memory>
#include <functional>
#include <string>

#include <QObject>
#include <QString>
#include <QStringList>
#include <QSet>
#include <QTemporaryFile>

#include "elements/VolumeSelector.hpp"

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
     * Callback for waiting on pending index rebuilds.
     * Used by onRenameSurface.
     */
    void setWaitForIndexRebuildCallback(std::function<void()> fn) { _waitForIndexRebuildCallback = std::move(fn); }

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

public slots:
    void onRenderSegment(const std::string& segmentId);
    void onGrowSegmentFromSegment(const std::string& segmentId);
    void onAddOverlap(const std::string& segmentId);
    void onConvertToObj(const std::string& segmentId);
    void onCropSurfaceToValidRegion(const std::string& segmentId);
    void onFlipSurface(const std::string& segmentId, bool flipU);
    void onRotateSurface(const std::string& segmentId);
    void onAlphaCompRefine(const std::string& segmentId);
    void onSlimFlatten(const std::string& segmentId);
    void onABFFlatten(const std::string& segmentId);
    void onAWSUpload(const std::string& segmentId);
    void onExportWidthChunks(const std::string& segmentId);
    void onRasterizeSegments(const QStringList& segmentIds);
    void onAddIgnoreLabel();
    void onGrowSeeds(const std::string& segmentId, bool isExpand, bool isRandomSeed = false);
    void onNeighborCopyRequested(const QString& segmentId, bool copyOut);
    void onResumeLocalGrowPatchRequested(const QString& segmentId);
    void onReloadFromBackup(const QString& segmentId, int backupIndex);
    void onCopySurfaceRequested(const QString& segmentId);
    void onMoveSegmentToPaths(const QString& segmentId);
    void onRenameSurface(const QString& segmentId);

    void handleNeighborCopyToolFinished(bool success);

    // Internal helpers exposed as slots for QTimer::singleShot
    void launchNeighborCopySecondPass();

public:
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

    QWidget* _parentWidget{nullptr};
    CState* _state{nullptr};
    CommandLineToolRunner* _cmdRunner{nullptr};
    SurfacePanelController* _surfacePanel{nullptr};
    SegmentationGrower* _segmentationGrower{nullptr};

    std::optional<NeighborCopyJob> _neighborCopyJob;
    std::optional<ResumeLocalJob> _resumeLocalJob;

    // Callbacks for CWindow-specific operations
    std::function<QString()> _normal3dZarrPathGetter;
    std::function<bool()> _isEditingCheck;
    std::function<void()> _clearSelectionCallback;
    std::function<void()> _waitForIndexRebuildCallback;
    std::function<void(const std::string&)> _restoreSelectionCallback;
};
