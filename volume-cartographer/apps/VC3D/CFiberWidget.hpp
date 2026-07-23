#pragma once

#include <QDockWidget>
#include <QStandardItemModel>
#include <QPushButton>
#include <QString>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

class QLabel;
class QButtonGroup;
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

        struct SpanEntry {
            int spanIndex = 0;
            int firstControlIndex = 0;
            int secondControlIndex = 0;
            int controlPointCount = 0;
            int linePointCount = 0;
            double lengthVx = 0.0;
            AlignmentMetrics alignment;
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
    };

    explicit CFiberWidget(QWidget* parent = nullptr);
    ~CFiberWidget();

    uint64_t selectedFiberId() const { return _selectedFiberId; }
    std::vector<uint64_t> selectedFiberIds() const;
    bool canDeleteSelection() const;
    bool canCreateAtlasFromSelection() const;
    bool canShowFiberSlice() const;
    bool canRenameFiberFile() const;
    std::vector<uint64_t> orderedFiberIds() const;
    QAction* createShowFiberSliceAction(QObject* parent);
    QAction* createRenameFiberFileAction(QObject* parent);
    void setFibers(const std::vector<FiberEntry>& fibers);
    void setAlignmentMetricsPending(bool pending);
    void updateAlignmentMetrics(uint64_t fiberId,
                                const FiberEntry::AlignmentMetrics& alignment,
                                const std::vector<FiberEntry::AlignmentMetrics>& spanAlignments);
    void setKnownTags(const std::vector<std::string>& tags);
    void selectFiber(uint64_t fiberId);
    void selectFibers(const std::vector<uint64_t>& fiberIds);
    void setDeleteConfirmationForTesting(std::function<bool(const std::vector<uint64_t>&)> confirmer);
    void setShowFibersAvailable(bool available);
    void setShowFibersChecked(bool checked);
    [[nodiscard]] bool showFibersChecked() const;
    void setFiberViewDistance(double distance);
    [[nodiscard]] double fiberViewDistance() const;

signals:
    void fiberOpenRequested(uint64_t fiberId);
    void deleteFibersRequested(std::vector<uint64_t> fiberIds);
    void manualHvTagChanged(uint64_t fiberId, QString tag);
    void fiberTagChanged(uint64_t fiberId, QString tag, bool enabled);
    void hvScoreRecalculationRequested(uint64_t fiberId);
    void fiberSpanOpenRequested(uint64_t fiberId, int firstControlIndex, int secondControlIndex);
    void newAtlasFromFiberRequested(uint64_t fiberId);
    void addFibersToPointCollectionsRequested(std::vector<uint64_t> fiberIds);
    void addFibersToSpiralFitRequested(std::vector<uint64_t> fiberIds);
    void fiberSliceRequested(uint64_t fiberId);
    void renameFiberFileRequested(uint64_t fiberId);
    void importFibersRequested();
    void exportFibersRequested();
    void metricsCalculationRequested(std::vector<uint64_t> orderedFiberIds);
    void showFibersToggled(bool checked);
    void fiberViewDistanceChanged(double distance);

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
    bool _spiralFitAvailable = false;
    void setupUi();
    void rebuildModel();
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
    void requestShowFiberSlice();
    void requestRenameFiberFile();
    bool confirmDeleteFibers(const std::vector<uint64_t>& fiberIds);
    static QString displayNameForFiber(const FiberEntry& fiber);
    static QString directionForFiber(const FiberEntry& fiber);

    uint64_t _selectedFiberId = 0;
    std::vector<FiberEntry> _fibers;
    std::vector<std::string> _knownTags;
    std::function<bool(const std::vector<uint64_t>&)> _deleteConfirmationForTesting;
    int _sortColumn = 0;
    Qt::SortOrder _sortOrder = Qt::AscendingOrder;

    QCheckBox* _calcMetricsCheckBox;
    QCheckBox* _showFibersCheckBox;
    QDoubleSpinBox* _fiberViewDistanceSpinBox;
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
