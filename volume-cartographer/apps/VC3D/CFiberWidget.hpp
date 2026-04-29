#pragma once

#include <QDockWidget>
#include <QListView>
#include <QStandardItemModel>
#include <QPushButton>
#include <QButtonGroup>
#include "vc/ui/VCCollection.hpp"

class CFiberWidget : public QDockWidget
{
    Q_OBJECT

public:
    explicit CFiberWidget(VCCollection* collection, QWidget* parent = nullptr);
    ~CFiberWidget();

    uint64_t selectedFiberId() const { return _selectedFiberId; }
    int currentStep() const { return _currentStep; }

signals:
    void newFiberRequested();
    void fiberSelected(uint64_t fiberId);
    void stepChanged(int step);
    void invertDirectionRequested();

public slots:
    void selectFiber(uint64_t fiberId);

private slots:
    void onNewFiberClicked();
    void onInvertDirClicked();
    void onStepButtonClicked(int id);
    void onSelectionChanged();

    void onCollectionsAdded(const std::vector<uint64_t>& collectionIds);
    void onCollectionChanged(uint64_t collectionId);
    void onCollectionRemoved(uint64_t collectionId);
    void onPointAdded(const ColPoint& point);
    void onPointRemoved(uint64_t pointId);

private:
    void setupUi();
    void refreshList();
    bool isFiber(uint64_t collectionId) const;
    QStandardItem* findFiberItem(uint64_t fiberId);

    VCCollection* _collection;
    uint64_t _selectedFiberId = 0;
    int _currentStep = 50;

    QListView* _listView;
    QStandardItemModel* _model;
    QPushButton* _newFiberButton;
    QPushButton* _invertDirButton;
    QButtonGroup* _stepGroup;
};
