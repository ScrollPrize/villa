#pragma once

#include <QObject>
#include <QPointF>

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

class CSurfaceCollection;
class ViewerManager;
class SurfaceTreeWidgetItem;
class OpChain;
class VolumePkg;
class CVolumeViewer;
class VCCollection;
class QTreeWidget;
class QCheckBox;
class QComboBox;
class QPushButton;
class QStandardItemModel;
class QuadSurface;

class SurfacePanelController : public QObject
{
    Q_OBJECT

public:
    struct UiRefs {
        QTreeWidget* treeWidget{nullptr};
        QPushButton* reloadButton{nullptr};
    };

    struct FilterUiRefs {
        QCheckBox* focusPoints{nullptr};
        QComboBox* pointSet{nullptr};
        QPushButton* pointSetAll{nullptr};
        QPushButton* pointSetNone{nullptr};
        QComboBox* pointSetMode{nullptr};
        QCheckBox* unreviewed{nullptr};
        QCheckBox* revisit{nullptr};
        QCheckBox* noExpansion{nullptr};
        QCheckBox* noDefective{nullptr};
        QCheckBox* partialReview{nullptr};
        QCheckBox* hideUnapproved{nullptr};
        QCheckBox* inspectOnly{nullptr};
        QCheckBox* currentOnly{nullptr};
    };

    struct TagUiRefs {
        QCheckBox* approved{nullptr};
        QCheckBox* defective{nullptr};
        QCheckBox* reviewed{nullptr};
        QCheckBox* revisit{nullptr};
        QCheckBox* inspect{nullptr};
    };

    enum class Tag {
        Approved,
        Defective,
        Reviewed,
        Revisit,
        Inspect,
    };

    SurfacePanelController(const UiRefs& ui,
                           CSurfaceCollection* surfaces,
                           ViewerManager* viewerManager,
                           std::unordered_map<std::string, OpChain*>* opchains,
                           std::function<CVolumeViewer*()> segmentationViewerProvider,
                           std::function<void()> filtersUpdated,
                           QObject* parent = nullptr);

    void setVolumePkg(const std::shared_ptr<VolumePkg>& pkg);
    void clear();

    void loadSurfaces(bool reload);
    void loadSurfacesIncremental();
    void updateTreeItemIcon(SurfaceTreeWidgetItem* item);

    void configureFilters(const FilterUiRefs& filters, VCCollection* pointCollection);
    void configureTags(const TagUiRefs& tags);

    void refreshPointSetFilterOptions();
    void applyFilters();

    void syncSelectionUi(const std::string& surfaceId, QuadSurface* surface);
    void resetTagUi();

    bool isCurrentOnlyFilterEnabled() const;
    bool toggleTag(Tag tag);

signals:
    void surfacesLoaded();
    void surfaceSelectionCleared();
    void filtersApplied(int hiddenCount);

private:
    struct SurfaceChanges {
        std::vector<std::string> toAdd;
        std::vector<std::string> toRemove;
    };

    SurfaceChanges detectSurfaceChanges() const;
    void populateSurfaceTree();
    void addSingleSegmentation(const std::string& segId);
    void removeSingleSegmentation(const std::string& segId);

    void connectFilterSignals();
    void connectTagSignals();
    void rebuildPointSetFilterModel();
    void onTagCheckboxToggled();
    void applyFiltersInternal();
    void updateTagCheckboxStatesForSurface(QuadSurface* surface);
    void setTagCheckboxEnabled(bool enabledApproved,
                               bool enabledDefective,
                               bool enabledReviewed,
                               bool enabledRevisit,
                               bool enabledInspect);

    UiRefs _ui;
    CSurfaceCollection* _surfaces{nullptr};
    ViewerManager* _viewerManager{nullptr};
    std::unordered_map<std::string, OpChain*>* _opchains{nullptr};
    std::shared_ptr<VolumePkg> _volumePkg;
    std::function<CVolumeViewer*()> _segmentationViewerProvider;
    std::function<void()> _filtersUpdated;
    FilterUiRefs _filters;
    TagUiRefs _tags;
    VCCollection* _pointCollection{nullptr};
    std::string _currentSurfaceId;
    QMetaObject::Connection _pointSetModelConnection;
    bool _configuringFilters{false};
};
