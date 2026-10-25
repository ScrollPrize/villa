#pragma once

#include <QObject>
#include <QString>
#include <QStringList>

#include "SurfaceCatalog.hpp"

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

class CState;
class QLineEdit;
class ViewerManager;
class SurfaceTreeWidgetItem;
class CChunkedVolumeViewer;
class VCCollection;
class QTreeWidget;
class QCheckBox;
class QComboBox;
class QDialog;
class QDoubleSpinBox;
class QPushButton;
class QuadSurface;
class DropdownChecklistButton;

// The surface list: the current volume's surfaces from the SurfaceCatalog,
// filtered, tagged and opened from the object store.
class SurfacePanelController : public QObject
{
    Q_OBJECT

public:
    struct UiRefs {
        QTreeWidget* treeWidget{nullptr};
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

    SurfacePanelController(const UiRefs& ui,
                           CState* state,
                           SurfaceCatalog* catalog,
                           ViewerManager* viewerManager,
                           std::function<CChunkedVolumeViewer*()> segmentationViewerProvider,
                           QObject* parent = nullptr);

    bool hasSurfaces() const;

    void configureFilters(const FilterUiRefs& filters, VCCollection* pointCollection);
    void configureTags(const TagUiRefs& tags);

    void refreshPointSetFilterOptions();
    void applyFilters();

    bool selectSurfaceById(const std::string& surfaceId);
    /// Programmatically activate a surface: selects it in the tree (signals
    /// blocked, via selectSurfaceById), opens it from the store and emits
    /// surfaceActivated exactly as a live click would. Never shows UI. Returns
    /// false with a reason in *errorMessage when the id is unknown, cannot be
    /// opened, or selection is locked while growth runs.
    bool activateSurfaceById(const std::string& surfaceId,
                             QString* errorMessage = nullptr);
    void resetTagUi();

    bool isCurrentOnlyFilterEnabled() const;
    bool toggleTag(Tag tag);
    bool setTagChecked(Tag tag, bool checked);
    void setSelectionLocked(bool locked);
    void setTransformWarning(const QString& warningText);
    // Retires each surface in the store. The interactive caller is responsible
    // for confirmation.
    bool retireSurfaces(const QStringList& surfaceIds, QString* err = nullptr);
    // Replace the whole highlighted-surface set at once and push it to the viewers,
    // keeping _highlightedSurfaceIds, the source of truth behind the
    // "Highlight in slice views" checkmarks, in sync.
    void setHighlightedSurfaceIds(const std::vector<std::string>& ids);
    std::vector<std::string> highlightedSurfaceIds() const;
    bool cycleToNextVisibleSegment();
    bool cycleToPreviousVisibleSegment();
    // Enables the "Add to current spiral fit" context action while a Spiral
    // session is active on the connected service.
    void setSpiralFitAvailable(bool available) { _spiralFitAvailable = available; }

    SurfaceCatalog* catalog() const { return _catalog; }
    // The surface last opened by activation, pinned at the revisions it was read from.
    const std::optional<SurfaceCatalog::Opened>& opened() const { return _opened; }
    // Surfaces the filters show that can be drawn in the current volume.
    const std::vector<std::string>& visibleSurfaceIds() const { return _visibleSurfaceIds; }

signals:
    void surfaceSelectionCleared();
    void filtersApplied(int hiddenCount);
    void visibleSurfacesChanged(const std::vector<std::string>& ids);
    void highlightedSurfacesChanged(const std::vector<std::string>& ids);
    void surfaceActivated(const QString& id, const std::shared_ptr<QuadSurface>& surface);
    void surfaceActivatedPreserveEditing(const QString& id, const std::shared_ptr<QuadSurface>& surface);
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
    void alphaCompRefineRequested(const QString& segmentId);
    void rasterizeSegmentsRequested(const QStringList& segmentIds);
    void mergeTifxyzRequested(const QStringList& segmentIds);
    // Emitted when the user right-clicks two selected segments and picks
    // "Patch tifxyz...". CWindow wires this to
    // SegmentationCommandHandler::onMergePatch.
    void mergePatchRequested(const QStringList& segmentIds);
    void addIgnoreLabelRequested();
    void statusMessageRequested(const QString& message, int timeoutMs);
    void resumeLocalGrowPatchRequested(const QString& segmentId);
    void neighborCopyRequested(const QString& segmentId, bool copyOut);
    void flipURequested(const QString& segmentId);
    void flipVRequested(const QString& segmentId);
    void rotateSurfaceRequested(const QString& segmentId);
    void focusSurfaceRequested(const QString& segmentId);

private:
    void syncTree();
    void setItemRow(SurfaceTreeWidgetItem* item, const SurfaceCatalog::Row& row);
    SurfaceTreeWidgetItem* itemFor(const std::string& id) const;
    // Open the selected surface and announce it.
    void activate(const std::string& id, bool preserveEditing);
    void fetchThenActivate(const std::string& id, bool preserveEditing);

    void connectFilterSignals();
    void connectTagSignals();
    void setupSurfaceColumnMenu();
    void restoreSurfaceColumnVisibility();
    void showSurfaceColumnMenu(const QPoint& pos);
    void buildFilterDialog();
    void showFilterDialog();
    void rebuildPointSetFilterModel();
    void rebuildRecordTagFilterModel();
    void handleTreeSelectionChanged();
    void showContextMenu(const QPoint& pos);
    void handleRetireSurfaces(const QStringList& surfaceIds);
    void renameSurface(const std::string& id);
    void copySurface(const std::string& id);
    void bindSurface(const std::string& id);
    void onTagCheckboxToggled();
    void applyFiltersInternal();
    // Geometric filters (focus point, point sets) read surfaces near points
    // from the store in the background; applyFilters() uses the last answer.
    void refreshProximityFilters();
    void updateFilterSummary();
    void updateTagSummary();
    void updateTagCheckboxStates(const SurfaceCatalog::Row* row);
    void setTagCheckboxEnabled(bool enabled);
    void applyHighlightSelection(const std::string& id, bool enabled);
    void applyTransformWarningStyle(SurfaceTreeWidgetItem* item);
    bool cycleVisibleSegment(int direction);

    UiRefs _ui;
    CState* _state{nullptr};
    SurfaceCatalog* _catalog{nullptr};
    ViewerManager* _viewerManager{nullptr};
    std::function<CChunkedVolumeViewer*()> _segmentationViewerProvider;
    FilterUiRefs _filters;
    TagUiRefs _tags;
    QComboBox* _recordTagFilter{nullptr};
    QDialog* _filterDialog{nullptr};
    VCCollection* _pointCollection{nullptr};
    std::string _currentSurfaceId;
    // The catalog surface being fetched for activation (the last one asked for).
    std::string _fetchingSurfaceId;
    std::optional<SurfaceCatalog::Opened> _opened;
    std::vector<std::string> _visibleSurfaceIds;
    QMetaObject::Connection _pointSetModelConnection;
    bool _configuringFilters{false};
    bool _spiralFitAvailable{false};
    bool _selectionLocked{false};
    QStringList _lockedSelectionIds;
    bool _selectionLockNotified{false};
    QString _transformWarningText;
    std::unordered_set<std::string> _highlightedSurfaceIds;

    // Proximity filter answers: the query they answer and the surfaces it found.
    struct ProximityAnswer {
        std::vector<std::vector<std::array<double, 3>>> pointSets;
        double tolerance = 0;
        // Per point set: surfaces near any point, and near every point.
        std::vector<std::set<std::string>> any, all;
    };
    std::optional<ProximityAnswer> _focusAnswer, _pointSetAnswer;
    uint64_t _proximityGeneration{0};
};
