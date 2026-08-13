#pragma once

#include <QObject>
#include <QColor>
#include <QPointF>
#include <QString>
#include <QStringList>
#include <QPointer>

#include "OpenDataSegmentCache.hpp"

#include <functional>
#include <filesystem>
#include <map>
#include <memory>
#include <unordered_set>
#include <vector>

class CState;
class Segmentation;
class QLineEdit;
class ViewerManager;
class SurfaceTreeWidgetItem;
class VolumePkg;
class CChunkedVolumeViewer;
class VCCollection;
class QTreeWidget;
class QCheckBox;
class QComboBox;
class QDialog;
class QDoubleSpinBox;
class QPushButton;
class QStandardItemModel;
class QProgressDialog;
template <typename T> class QFutureWatcher;
class QuadSurface;
class DropdownChecklistButton;

// Forward declare for QuadSurface pointer usage
#include "vc/core/util/QuadSurface.hpp"

class SurfacePanelController : public QObject
{
    Q_OBJECT

public:
    struct UiRefs {
        QTreeWidget* treeWidget{nullptr};
        QPushButton* reloadButton{nullptr};
    };

    struct FilterUiRefs {
        DropdownChecklistButton* dropdown{nullptr};
        QCheckBox* focusPoints{nullptr};
        QDoubleSpinBox* focusPointDistance{nullptr};
        QComboBox* pointSet{nullptr};
        QPushButton* pointSetAll{nullptr};
        QPushButton* pointSetNone{nullptr};
        QComboBox* pointSetMode{nullptr};
        QCheckBox* unreviewed{nullptr};
        QCheckBox* noExpansion{nullptr};
        QCheckBox* noDefective{nullptr};
        QCheckBox* partialReview{nullptr};
        QCheckBox* showPartialReview{nullptr};
        QCheckBox* hideUnapproved{nullptr};
        QCheckBox* inspectOnly{nullptr};
        QCheckBox* currentOnly{nullptr};
        QLineEdit* surfaceIdFilter{nullptr};
        QDoubleSpinBox* zLowerBound{nullptr};
        QDoubleSpinBox* zUpperBound{nullptr};
    };

    struct TagUiRefs {
        DropdownChecklistButton* dropdown{nullptr};
        QCheckBox* approved{nullptr};
        QCheckBox* defective{nullptr};
        QCheckBox* reviewed{nullptr};
        QCheckBox* inspect{nullptr};
    };

    enum class Tag {
        Approved,
        Defective,
        Reviewed,
        Inspect,
    };

    struct SegmentFolderSelection {
        std::string dirName;
        std::filesystem::path path;
        bool currentFolder{false};
        bool defaultPalette{false};
        QColor color;
    };

    SurfacePanelController(const UiRefs& ui,
                           CState* state,
                           ViewerManager* viewerManager,
                           std::function<CChunkedVolumeViewer*()> segmentationViewerProvider,
                           std::function<void()> filtersUpdated,
                           QObject* parent = nullptr);

    void setVolumePkg(const std::shared_ptr<VolumePkg>& pkg);
    void clear();
    bool hasSurfaces() const;

    void loadSurfaces(bool reload);
    void loadSurfacesIncremental();
    void refreshSurfaceList();
    void updateTreeItemIcon(SurfaceTreeWidgetItem* item);
    void refreshSurfaceMetrics(const std::string& surfaceId);

    void configureFilters(const FilterUiRefs& filters, VCCollection* pointCollection);
    void configureTags(const TagUiRefs& tags);

    void refreshPointSetFilterOptions();
    void applyFilters();

    void syncSelectionUi(const std::string& surfaceId, QuadSurface* surface);
    bool selectSurfaceById(const std::string& surfaceId);
    /// Programmatically activate a segment: selects it in the tree (signals blocked,
    /// via selectSurfaceById) then emits surfaceActivated exactly as a live click would
    /// (reaching CWindow::onSurfaceActivated). Never shows UI. Returns false with a
    /// reason in *errorMessage when the id is unknown, unloadable, an unmaterialized
    /// open-data placeholder, or selection is locked while growth runs.
    bool activateSurfaceById(const std::string& surfaceId,
                             QString* errorMessage = nullptr);
    void resetTagUi();

    bool isCurrentOnlyFilterEnabled() const;
    bool toggleTag(Tag tag);
    bool setTagChecked(Tag tag, bool checked);
    void reloadSurfacesFromDisk();
    void refreshFiltersOnly();
    void setSelectionLocked(bool locked);
    void setTransformWarning(const QString& warningText);
    void setVisibleSegmentFolders(std::vector<SegmentFolderSelection> folders);
    void addSingleSegmentation(const std::string& segId);
    void removeSingleSegmentation(const std::string& segId, bool suppressSignals = false);
    // Irreversibly deletes each id from disk and refreshes the panel. The
    // interactive caller is responsible for confirmation.
    bool deleteSegmentsHeadless(const QStringList& segmentIds, QString* err = nullptr,
                                int* deletedCount = nullptr);
    // Replace the whole highlighted-surface set at once and push it to the viewers,
    // keeping _highlightedSurfaceIds, the source of truth behind the
    // "Highlight in slice views" checkmarks, in sync.
    void setHighlightedSurfaceIds(const std::vector<std::string>& ids);
    std::vector<std::string> highlightedSurfaceIds() const;
    bool cycleToNextVisibleSegment();
    bool cycleToPreviousVisibleSegment();
    void materializeCurrentOpenDataFolder();
    // Enables the "Add to current spiral fit" context action while a Spiral
    // session is active on the connected service.
    void setSpiralFitAvailable(bool available) { _spiralFitAvailable = available; }

    /// Outcome of an asynchronous single-segment fetch request. Only `Started`
    /// implies the completion callback will fire later (on the main thread);
    /// the other outcomes are resolved synchronously by the caller with no
    /// callback.
    enum class OpenDataFetchOutcome {
        Started,             ///< async materialize kicked off; onDone will fire
        AlreadyMaterialized, ///< not a placeholder; nothing to fetch
        NotFound,            ///< no such segment / surface could not be loaded
        Busy,                ///< another single-segment materialize is running
    };

    /// Materialize one open-data placeholder without a progress dialog or
    /// forced activation. Shares the GUI path's single-flight guard. On a
    /// `Started` outcome, reloads the surface list on success and invokes
    /// onDone(success, message) on the main thread.
    OpenDataFetchOutcome fetchOpenDataSegmentAsync(
        const std::string& id,
        std::function<void(bool success, const QString& message)> onDone);

    /// True while a single-segment materialization holds the shared
    /// single-flight guard.
    bool isOpenDataMaterializationRunning() const;

signals:
    void surfacesLoaded();
    void surfaceSelectionCleared();
    void filtersApplied(int hiddenCount);
    void surfaceActivated(const QString& id, QuadSurface* surface);
    void copySegmentPathRequested(const QString& segmentId);
    void addSurfaceToSpiralFitRequested(const QString& segmentId);
    void renderSegmentRequested(const QString& segmentId);
    void growSegmentRequested(const QString& segmentId);
    void convertToObjRequested(const QString& segmentId);
    void visLasagnaObjRequested(const QString& segmentId);
    void cropBoundsRequested(const QString& segmentId);
    void slimFlattenRequested(const QString& segmentId);
    void straightenRequested(const QString& segmentId);
    void abfFlattenRequested(const QString& segmentId);
    void recalcAreaRequested(const QStringList& segmentIds);
    void exportTifxyzChunksRequested(const QString& segmentId);
    void alphaCompRefineRequested(const QString& segmentId);
    void rasterizeSegmentsRequested(const QStringList& segmentIds);
    void generateSegmentMaskRequested(const QString& segmentId);
    void appendSegmentMaskRequested(const QString& segmentId);
    void mergeTifxyzRequested(const QStringList& segmentIds);
    // Emitted when the user right-clicks two selected segments and picks
    // "Patch tifxyz...". CWindow wires this to
    // SegmentationCommandHandler::onMergePatch.
    void mergePatchRequested(const QStringList& segmentIds);
    void addIgnoreLabelRequested();
    void statusMessageRequested(const QString& message, int timeoutMs);
    void moveToPathsRequested(const QString& segmentId);
    void renameSurfaceRequested(const QString& segmentId);
    void copySurfaceRequested(const QString& segmentId);
    void resumeLocalGrowPatchRequested(const QString& segmentId);
    void neighborCopyRequested(const QString& segmentId, bool copyOut);
    void reloadFromBackupRequested(const QString& segmentId, int backupIndex);
    void flipURequested(const QString& segmentId);
    void flipVRequested(const QString& segmentId);
    void rotateSurfaceRequested(const QString& segmentId);
    void focusSurfaceRequested(const QString& segmentId);
    void surfaceActivatedPreserveEditing(const QString& id, QuadSurface* surface);


private:
    struct SurfaceChanges {
        std::vector<std::string> toAdd;
        std::vector<std::string> toRemove;
        std::vector<std::string> toReload;
    };

    SurfaceChanges detectSurfaceChanges() const;
    void populateSurfaceTree();

    void connectFilterSignals();
    void connectTagSignals();
    void setupSurfaceColumnMenu();
    void restoreSurfaceColumnVisibility();
    void showSurfaceColumnMenu(const QPoint& pos);
    void buildFilterDialog();
    void showFilterDialog();
    void rebuildPointSetFilterModel();
    void handleTreeSelectionChanged();
    void showContextMenu(const QPoint& pos);
    void handleDeleteSegments(const QStringList& segmentIds);
    void onTagCheckboxToggled();
    void applyFiltersInternal();
    void updateFilterSummary();
    void updateTagSummary();
    void updateTagCheckboxStatesForSurface(QuadSurface* surface);
    void setTagCheckboxEnabled(bool enabledApproved,
                               bool enabledDefective,
                               bool enabledReviewed,
                               bool enabledInspect);
    void logSurfaceLoadSummary() const;
    QString folderSelectionCacheKey() const;
    void applyHighlightSelection(const std::string& id, bool enabled);
    void applyTransformWarningStyle(SurfaceTreeWidgetItem* item);
    bool cycleVisibleSegment(int direction);
    std::shared_ptr<QuadSurface> getSurfaceById(const std::string& id) const;
    bool startOpenDataMaterialization(const std::string& id,
                                      const std::shared_ptr<QuadSurface>& surface);
    void activateMaterializedSurface(const std::string& id,
                                     const std::filesystem::path& path);

    UiRefs _ui;
    CState* _state{nullptr};
    ViewerManager* _viewerManager{nullptr};
    std::shared_ptr<VolumePkg> _volumePkg;
    std::function<CChunkedVolumeViewer*()> _segmentationViewerProvider;
    std::function<void()> _filtersUpdated;
    FilterUiRefs _filters;
    TagUiRefs _tags;
    QDialog* _filterDialog{nullptr};
    VCCollection* _pointCollection{nullptr};
    std::string _currentSurfaceId;
    QMetaObject::Connection _pointSetModelConnection;
    bool _configuringFilters{false};
    bool _spiralFitAvailable{false};
    bool _selectionLocked{false};
    QStringList _lockedSelectionIds;
    bool _selectionLockNotified{false};
    QString _transformWarningText;
    std::unordered_set<std::string> _highlightedSurfaceIds;
    std::vector<SegmentFolderSelection> _visibleSegmentFolders;
    std::unordered_set<std::string> _multiFolderSurfaceIds;
    // Segmentations loaded for non-current overlay folders, keyed by segment
    // path. Retained so re-checking a folder (or switching back to a selection
    // that includes it) reuses the already-loaded surfaces.
    std::map<std::string, std::shared_ptr<Segmentation>> _overlaySegmentations;
    QFutureWatcher<vc3d::opendata::OpenDataSegmentMaterializationResult>*
        _segmentMaterializationWatcher{nullptr};
    QFutureWatcher<vc3d::opendata::OpenDataSegmentMaterializationResult>*
        _folderMaterializationWatcher{nullptr};
    std::string _pendingMaterializationId;
    QPointer<QProgressDialog> _segmentMaterializationProgress;
    QPointer<QProgressDialog> _folderMaterializationProgress;
};
