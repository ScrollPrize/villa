#pragma once

#include <QDockWidget>
#include <QListView>
#include <QStandardItemModel>
#include <QPushButton>

#include <cstdint>
#include <vector>

class CFiberWidget : public QDockWidget
{
    Q_OBJECT

public:
    struct FiberEntry {
        uint64_t id = 0;
        int controlPointCount = 0;
        int linePointCount = 0;
        double lengthVx = 0.0;
    };

    explicit CFiberWidget(QWidget* parent = nullptr);
    ~CFiberWidget();

    uint64_t selectedFiberId() const { return _selectedFiberId; }
    void setFibers(const std::vector<FiberEntry>& fibers);
    void selectFiber(uint64_t fiberId);

signals:
    void fiberOpenRequested(uint64_t fiberId);
    void deleteFiberRequested(uint64_t fiberId);

private slots:
    void onSelectionChanged();
    void onDoubleClicked(const QModelIndex& index);
    void onDeleteClicked();

private:
    void setupUi();
    QStandardItem* findFiberItem(uint64_t fiberId);
    static QString labelForFiber(const FiberEntry& fiber);

    uint64_t _selectedFiberId = 0;
    std::vector<FiberEntry> _fibers;

    QListView* _listView;
    QStandardItemModel* _model;
    QPushButton* _deleteButton;
};
