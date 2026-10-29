#pragma once

#include "FiberDisplayOptions.hpp"

#include <QDockWidget>
#include <QStandardItemModel>
#include <QPushButton>
#include <QString>
#include <QStringList>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

class QLabel;
class QButtonGroup;
class QSlider;
class QSpinBox;
class QAction;
class QCheckBox;
class QDoubleSpinBox;
class QLineEdit;
class QTreeView;
class QVBoxLayout;

class CFiberWidget : public QDockWidget
{
    Q_OBJECT

public:
    struct FiberEntry {
        struct AlignmentMetrics {
            bool available = false;
            bool pending = false;
            int sampleCount = 0;
            double meanErrorDegrees = 0.0;
            double maxErrorDegrees = 0.0;
            std::string error;
        };

        // Mirrors vc3d::line_annotation::FiberTraceState without pulling
        // the annotation headers into this standalone widget.
        enum class TraceState {
            Legacy,       // no prediction-traced spans
            Predictions,  // traced spans, fiber mode native_fiber_trace3d
            Mixed,        // traced spans under a lasagna-global fiber
        };

        struct SpanEntry {
            int spanIndex = 0;
            int firstControlIndex = 0;
            int secondControlIndex = 0;
            int controlPointCount = 0;
            int linePointCount = 0;
            double lengthVx = 0.0;
            AlignmentMetrics alignment;
            // 'C' (cspline), 'L' (lasagna), or 'T' (prediction trace).
            char interpMarker = 'L';
            // Fiber-inference manifest the trace ran with; empty for
            // non-trace spans.
            std::string fiberManifest;
        };

        uint64_t id = 0;
        std::string fileName;
        int controlPointCount = 0;
        int linePointCount = 0;
        double lengthVx = 0.0;
        AlignmentMetrics alignment;
        std::vector<SpanEntry> spans;
        double hvZDistance = 0.0;
        double hvFiberLength = 0.0;
        double horizontalScore = 0.0;
        double verticalScore = 0.0;
        double automaticCertainty = 0.0;
        std::string automaticHvTag;
        std::string manualHvTag;
        std::vector<std::string> tags;
        int linkedFiberCount = 0;
        int pendingLinkCount = 0;
        TraceState traceState = TraceState::Legacy;
        bool detailsLoaded = true;
    };

    explicit CFiberWidget(QWidget* parent = nullptr);
    ~CFiberWidget();

    uint64_t selectedFiberId() const { return _selectedFiberId; }
    std::vector<uint64_t> selectedFiberIds() const;
    bool canDeleteSelection() const;
    bool canRenameFiberFile() const;
    std::vector<uint64_t> orderedFiberIds() const;
    QAction* createRenameFiberFileAction(QObject* parent);
    // The listed fibers: one store catalog page. Entries without details load
    // them through fiberDetailsRequested when selected or expanded.
    void setFibers(const std::vector<FiberEntry>& fibers);
    // Replaces one listed fiber's entry with its full details.
    void updateFiberDetails(const FiberEntry& fiber);
    // Lists a fiber that is not on the current page (a selection made elsewhere).
    void revealFiber(const FiberEntry& fiber);
    // Physical size of one annotation-frame voxel in µm, the frame the
    // entries' lengthVx are measured in. With it, lengths (the len column,
    // the selected fiber's header line, the total at the top) read in
    // centimetres; unset, they stay in voxels and the column header says so.
    void setAnnotationVoxelSizeUm(std::optional<double> voxelSizeUm);
    // Sum of every listed fiber's lengthVx (finite values only).
    [[nodiscard]] double totalLengthVx() const;
    void updateAlignmentMetrics(uint64_t fiberId,
                                const FiberEntry::AlignmentMetrics& alignment,
                                const std::vector<FiberEntry::AlignmentMetrics>& spanAlignments);
    void setKnownTags(const std::vector<std::string>& tags);
    void selectFiber(uint64_t fiberId);
    void selectFibers(const std::vector<uint64_t>& fiberIds);
    void setDeleteConfirmationForTesting(std::function<bool(const std::vector<uint64_t>&)> confirmer);
    void setDisplayOptions(const FiberDisplayOptions& options);
    [[nodiscard]] FiberDisplayOptions displayOptions() const;
    void setStoreStatus(bool attached, const QString& text);
    void setNextPageAvailable(bool available);
    // The stored trace followed by the slider; pointCount 0 disables it.
    void setTraceSelection(const QString& text, const QString& toolTip, int64_t pointCount, int index, bool openable);
    // Progress of an Open in Line Annotation request; empty hides it.
    void setAnnotationStatus(const QString& text);
    // While the Line Annotation workspace is active a click opens the fiber there.
    void setLineAnnotationActive(bool active);

signals:
    void fiberDetailsRequested(uint64_t fiberId);
    void fiberOpenRequested(uint64_t fiberId);
    void deleteFibersRequested(std::vector<uint64_t> fiberIds);
    void manualHvTagChanged(uint64_t fiberId, QString tag);
    void fiberTagChanged(uint64_t fiberId, QString tag, bool enabled);
    void hvScoreRecalculationRequested(uint64_t fiberId);
    void fiberSpanOpenRequested(uint64_t fiberId, int firstControlIndex, int secondControlIndex);
    void addFibersToPointCollectionsRequested(std::vector<uint64_t> fiberIds);
    void addFibersToSpiralFitRequested(std::vector<uint64_t> fiberIds);
    void renameFiberFileRequested(uint64_t fiberId);
    void importFibersRequested();
    void exportFibersRequested();
    void metricsCalculationRequested(std::vector<uint64_t> orderedFiberIds);
    void displayOptionsChanged(FiberDisplayOptions options);
    void attachStoreRequested();
    void detachStoreRequested();
    void firstPageRequested();
    void nextPageRequested();
    // A listed fiber was clicked, or its ID entered.
    void fiberActivated(uint64_t fiberId);
    void traceIndexChanged(int index);
    void openInLineAnnotationRequested(uint64_t fiberId);

private slots:
    void onSelectionChanged();
    void onDoubleClicked(const QModelIndex& index);
    void onDeleteClicked();
    void onManualHvButtonClicked(int id);
    void onManualHvResetClicked();
    void onRecalculateHvScoreClicked();
    void onAddTagClicked();
    void onHeaderSectionClicked(int section);
    void showContextMenu(const QPoint& pos);

public:
    // Enables the "Add to current spiral fit" context action while a Spiral
    // session is active on the connected service.
    void setSpiralFitAvailable(bool available) { _spiralFitAvailable = available; }

private:
    // Lazy span children: collapsed fibers carry one placeholder child; the
    // real span rows are built on first expansion (see kSpanPlaceholderRole).
    void buildSpanChildRows(QStandardItem* root, const FiberEntry& fiber);
    bool spanChildrenArePlaceholder(QStandardItem* root) const;
    void ensureSpanChildrenBuilt(const QModelIndex& index);

    bool _spiralFitAvailable = false;
    void setupUi();
    void rebuildModel();
    // A listed fiber's row cells, and its span rows (or the placeholder that
    // keeps the expand arrow until they are built).
    void fillFiberRow(const QList<QStandardItem*>& row, const FiberEntry& fiber);
    void fillSpanChildren(QStandardItem* root, const FiberEntry& fiber, bool expanded);
    // The column headers, with the len column naming its current unit.
    [[nodiscard]] QStringList headerLabels() const;
    // Re-texts every length cell and the header after the unit changed; the
    // model's rows, selection and expansion stay as they are.
    void refreshLengthDisplays();
    void updateTotalLengthLabel();
    void emitDisplayOptions();
    void sortFibers();
    QStandardItem* findFiberItem(uint64_t fiberId);
    QList<QStandardItem*> rowItemsForNameItem(QStandardItem* nameItem) const;
    void updateMetricDisplayForRow(QStandardItem* nameItem,
                                   const FiberEntry::AlignmentMetrics& alignment);
    void refreshMetricDisplays();
    const FiberEntry* selectedFiber() const;
    void updateClassificationUi();
    void rebuildTagList();
    void applyTagLocally(uint64_t fiberId, const std::string& tag, bool enabled);
    void requestFiberTagChange(const QString& tag, bool enabled);
    void requestDeleteSelectedFibers();
    void requestRenameFiberFile();
    bool confirmDeleteFibers(const std::vector<uint64_t>& fiberIds);
    static QString displayNameForFiber(const FiberEntry& fiber);
    static QString directionForFiber(const FiberEntry& fiber);
    static QString statusTextForFiber(const FiberEntry& fiber);

    uint64_t _selectedFiberId = 0;
    std::vector<FiberEntry> _listedFibers;
    std::vector<std::string> _knownTags;
    std::function<bool(const std::vector<uint64_t>&)> _deleteConfirmationForTesting;
    int _sortColumn = 0;
    Qt::SortOrder _sortOrder = Qt::AscendingOrder;
    std::optional<double> _voxelSizeUm;

    QLabel* _storeStatusLabel = nullptr;
    QPushButton* _attachButton = nullptr;
    QPushButton* _detachButton = nullptr;
    QDoubleSpinBox* _minLengthSpinBox = nullptr;
    QCheckBox* _horizontalCheckBox = nullptr;
    QCheckBox* _verticalCheckBox = nullptr;
    QCheckBox* _showFibersCheckBox = nullptr;
    QCheckBox* _showLinkedCheckBox = nullptr;
    QCheckBox* _rainbowCheckBox = nullptr;
    QDoubleSpinBox* _sliceDistanceSpinBox = nullptr;
    QSpinBox* _maxDisplayedSpinBox = nullptr;
    QDoubleSpinBox* _lineWidthSpinBox = nullptr;
    QPushButton* _firstPageButton = nullptr;
    QPushButton* _nextPageButton = nullptr;
    QLineEdit* _fiberIdEdit = nullptr;
    QLabel* _traceLabel = nullptr;
    QSlider* _traceSlider = nullptr;
    QPushButton* _openInLineAnnotationButton = nullptr;
    QLabel* _annotationStatusLabel = nullptr;
    QLabel* _totalLengthLabel = nullptr;
    QCheckBox* _calcMetricsCheckBox;
    QTreeView* _treeView;
    QStandardItemModel* _model;
    QLabel* _nameLabel;
    QLabel* _scoreLabel;
    QLabel* _autoLabel;
    QWidget* _tagListWidget;
    QVBoxLayout* _tagListLayout;
    QLineEdit* _newTagEdit;
    QPushButton* _addTagButton;
    QButtonGroup* _manualHvGroup;
    QPushButton* _manualHButton;
    QPushButton* _manualVButton;
    QPushButton* _manualResetButton;
    QPushButton* _recalculateScoreButton;
    QPushButton* _importButton;
    QPushButton* _exportButton;
    QPushButton* _deleteButton;
};
