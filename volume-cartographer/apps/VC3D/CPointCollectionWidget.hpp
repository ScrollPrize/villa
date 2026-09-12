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
#include <QLabel>

#include <cmath>
#include <filesystem>
#include <unordered_map>
#include <optional>






struct CorrPointResult {
    float winding_obs = NAN;
    float winding_err = NAN;
    float p[3] = {NAN, NAN, NAN};  // fullres position used during optimization
};

class CPointCollectionWidget : public QDockWidget
{
    Q_OBJECT

public:
    explicit CPointCollectionWidget(VCCollection *collection, QWidget *parent = nullptr);
    ~CPointCollectionWidget();

    void loadCorrPointsResults(const std::filesystem::path& jsonPath);
    void clearCorrPointsResults();

    void setAnnotateChecked(bool checked);
    double pointViewTolerance() const;

signals:
    void annotateToggled(bool enabled);
    void pointViewToleranceChanged(double tolerance);
    void collectionSelected(uint64_t collectionId);
    void collectionSelectionCleared();
    void pointSelected(vc::PointRef point);
    void pointSelectionCleared();
    void pointDoubleClicked(vc::PointRef point);
    void convertPointToAnchorRequested(vc::PointRef point);
    void focusCollectionRequested(uint64_t collectionId);
    void focusPointRequested(vc::PointRef point);

public slots:
    void selectCollection(uint64_t collectionId);
    void clearSelection();
    void selectPoint(vc::PointRef point);

private slots:
    void refreshTree();
    void onCollectionsAdded(const std::vector<uint64_t>& collectionIds);
    void onCollectionChanged(uint64_t collectionId);
    void onCollectionRemoved(uint64_t collectionId);
    void onPointAdded(const ColPoint& point);
    void onPointsAdded(const std::vector<ColPoint>& points);
    void onPointChanged(const ColPoint& point);
    void onPointRemoved(vc::PointRef point);

    void onResetClicked();
    void onResetWindingClicked();
    void onSelectionChanged(const QItemSelection &selected, const QItemSelection &deselected);
    void onNewNameClicked();
    void onNameEdited(const QString &name);
    void onAbsoluteWindingChanged(Qt::CheckState state);
    void onColorButtonClicked();
    void onWindingEdited(double value);
    void onWindingEnabledChanged(Qt::CheckState state);
    void onFillWindingPlusClicked();
    void onFillWindingMinusClicked();
    void onFillWindingEqualsClicked();
    void onSaveClicked();
    void onLoadClicked();
    void onConvertToAnchorClicked();
    void onClearAnchorClicked();
    void showContextMenu(const QPoint& pos);

 private:
    void keyPressEvent(QKeyEvent *event) override;
    void setupUi();
    void clearTreeModel();
    void updateMetadataWidgets();
    QStandardItem* findCollectionItem(uint64_t collectionId);

    VCCollection *_point_collection = nullptr;
    std::optional<uint64_t> _selected_collection_id;
    std::optional<vc::PointRef> _selected_point;

    QCheckBox *_chkAnnotate{nullptr};
    QDoubleSpinBox *_pointViewToleranceSpinbox{nullptr};

    QTreeView *_tree_view;
    QStandardItemModel *_model;
 
    QPushButton *_load_button;
    QPushButton *_save_button;
    QPushButton *_reset_button;
    QPushButton *_reset_winding_button;

    QGroupBox *_collection_metadata_group;
    QLineEdit *_collection_name_edit;
    QPushButton *_new_name_button;
    QCheckBox *_absolute_winding_checkbox;
    QPushButton *_color_button;
    QPushButton *_fill_winding_plus_button;
    QPushButton *_fill_winding_minus_button;
    QPushButton *_fill_winding_equals_button;
    QDoubleSpinBox *_fill_constant_spinbox;
    QLabel *_anchor_status_label;
    QPushButton *_clear_anchor_button;
    QLabel *_tags_label;

    QGroupBox *_point_metadata_group;
    QCheckBox *_winding_enabled_checkbox;
    QDoubleSpinBox* _winding_spinbox;
    QPushButton *_convert_to_anchor_button;

    std::unordered_map<uint64_t, CorrPointResult> _corr_point_results;
    std::unordered_map<uint64_t, float> _corr_collection_avgs;
};
