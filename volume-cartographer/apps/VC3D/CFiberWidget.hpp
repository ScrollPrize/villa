#pragma once

#include <QDockWidget>
#include <QListView>
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

class CFiberWidget : public QDockWidget
{
    Q_OBJECT

public:
    struct FiberEntry {
        uint64_t id = 0;
        std::string fileName;
        int controlPointCount = 0;
        int linePointCount = 0;
        double lengthVx = 0.0;
        double hvZDistance = 0.0;
        double hvFiberLength = 0.0;
        double horizontalScore = 0.0;
        double verticalScore = 0.0;
        double automaticCertainty = 0.0;
        std::string automaticHvTag;
        std::string manualHvTag;
    };

    explicit CFiberWidget(QWidget* parent = nullptr);
    ~CFiberWidget();

    uint64_t selectedFiberId() const { return _selectedFiberId; }
    std::vector<uint64_t> selectedFiberIds() const;
    bool canDeleteSelection() const;
    bool canCreateAtlasFromSelection() const;
    bool canShowFiberSlice() const;
    bool canRenameFiberFile() const;
    QAction* createShowFiberSliceAction(QObject* parent);
    QAction* createRenameFiberFileAction(QObject* parent);
    void setFibers(const std::vector<FiberEntry>& fibers);
    void selectFiber(uint64_t fiberId);
    void selectFibers(const std::vector<uint64_t>& fiberIds);
    void setDeleteConfirmationForTesting(std::function<bool(const std::vector<uint64_t>&)> confirmer);

signals:
    void fiberOpenRequested(uint64_t fiberId);
    void deleteFibersRequested(std::vector<uint64_t> fiberIds);
    void manualHvTagChanged(uint64_t fiberId, QString tag);
    void hvScoreRecalculationRequested(uint64_t fiberId);
    void newAtlasFromFiberRequested(uint64_t fiberId);
    void fiberSliceRequested(uint64_t fiberId);
    void renameFiberFileRequested(uint64_t fiberId);

private slots:
    void onSelectionChanged();
    void onDoubleClicked(const QModelIndex& index);
    void onDeleteClicked();
    void onManualHvButtonClicked(int id);
    void onManualHvResetClicked();
    void onRecalculateHvScoreClicked();
    void showContextMenu(const QPoint& pos);

private:
    void setupUi();
    QStandardItem* findFiberItem(uint64_t fiberId);
    const FiberEntry* selectedFiber() const;
    void updateClassificationUi();
    void requestDeleteSelectedFibers();
    void requestShowFiberSlice();
    void requestRenameFiberFile();
    bool confirmDeleteFibers(const std::vector<uint64_t>& fiberIds);
    static QString labelForFiber(const FiberEntry& fiber);

    uint64_t _selectedFiberId = 0;
    std::vector<FiberEntry> _fibers;
    std::function<bool(const std::vector<uint64_t>&)> _deleteConfirmationForTesting;

    QListView* _listView;
    QStandardItemModel* _model;
    QLabel* _scoreLabel;
    QLabel* _autoLabel;
    QButtonGroup* _manualHvGroup;
    QPushButton* _manualHButton;
    QPushButton* _manualVButton;
    QPushButton* _manualResetButton;
    QPushButton* _recalculateScoreButton;
    QPushButton* _deleteButton;
};
