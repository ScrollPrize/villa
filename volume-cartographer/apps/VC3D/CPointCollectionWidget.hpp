#pragma once

#include <QDockWidget>
#include "vc/ui/VCCollection.hpp"
#include <QTreeView>
#include <QStandardItemModel>
#include <QPushButton>
#include <QWidget>
#include <QGroupBox>
#include <QLineEdit>
#include <QCheckBox>
#include <QItemSelection>
#include <QDoubleSpinBox>






class CPointCollectionWidget : public QDockWidget
{
    Q_OBJECT

public:
    explicit CPointCollectionWidget(VCCollection *collection, QWidget *parent = nullptr);
    ~CPointCollectionWidget();

signals:
    void collectionSelected(uint64_t collectionId);
    void pointSelected(uint64_t pointId);
    void pointDoubleClicked(uint64_t pointId);

public slots:
    void selectCollection(uint64_t collectionId);
    void selectPoint(uint64_t pointId);

private slots:
    void refreshTree();
    void onCollectionsAdded(const std::vector<uint64_t>& collectionIds);
    void onCollectionChanged(uint64_t collectionId);
    void onCollectionRemoved(uint64_t collectionId);
    void onPointAdded(const ColPoint& point);
    void onPointChanged(const ColPoint& point);
    void onPointRemoved(uint64_t pointId);

    void onResetClicked();
    void onSelectionChanged(const QItemSelection &selected, const QItemSelection &deselected);
    void onNewNameClicked();
    void onNameEdited(const QString &name);
    void onAbsoluteWindingChanged(int state);
    void onColorButtonClicked();
    void onWindingEdited(double value);
    void onWindingEnabledChanged(int state);
    void onFillWindingPlusClicked();
    void onFillWindingMinusClicked();
    void onFillWindingEqualsClicked();
    void onSaveClicked();
    void onLoadClicked();
  
 private:
    void keyPressEvent(QKeyEvent *event) override;
    void setupUi();
    void updateMetadataWidgets();
    QStandardItem* findCollectionItem(uint64_t collectionId);

    VCCollection *_point_collection = nullptr;
    uint64_t _selected_collection_id = 0;
    uint64_t _selected_point_id = 0;

    QTreeView *_tree_view;
    QStandardItemModel *_model;
 
    QPushButton *_load_button;
    QPushButton *_save_button;
    QPushButton *_reset_button;
 
    QGroupBox *_collection_metadata_group;
    QLineEdit *_collection_name_edit;
    QPushButton *_new_name_button;
    QCheckBox *_absolute_winding_checkbox;
    QPushButton *_color_button;
    QPushButton *_fill_winding_plus_button;
    QPushButton *_fill_winding_minus_button;
    QPushButton *_fill_winding_equals_button;

    QGroupBox *_point_metadata_group;
    QCheckBox *_winding_enabled_checkbox;
    QDoubleSpinBox* _winding_spinbox;
};

