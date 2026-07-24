#pragma once

#include <QDockWidget>
#include <QPersistentModelIndex>

#include <unordered_map>
#include <vector>

#include "vc/ui/VCCollection.hpp"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QItemSelection;
class QPoint;
class QPushButton;
class QStandardItem;
class QStandardItemModel;
class QSpinBox;
class QTreeView;

class WrapAnnotationWidget : public QDockWidget
{
    Q_OBJECT

public:
    explicit WrapAnnotationWidget(VCCollection* collection, QWidget* parent = nullptr);

    bool sameWrapAnnotationEnabled() const;
    double sameWrapAnnotationSpacing() const;
    double sameWrapAnnotationPolylineOpacity() const;
    bool sameWrapAnnotationMergeEnabled() const;
    int sameWrapAnnotationPathType() const;
    int sameWrapAnnotationFilterType() const;
    int sameWrapAnnotationFilterKernelSize() const;
    int relWindingIntersectionSource() const;
    double relWindingPatchTolerance() const;

public slots:
    // Programmatic equivalent of toggling the "Same-wrap annotation mode" checkbox;
    // drives _chkSameWrapAnnotation's checked state so its toggled() signal fires the
    // same CWindow -> CChunkedVolumeViewer::setSameWrapAnnotationMode wiring as a click.
    // No-op if the checkbox is absent.
    void setSameWrapAnnotationEnabled(bool enabled);
    void setRelWindingAnnotationChecked(bool checked);
    void selectCollection(uint64_t collectionId);
    void selectPoint(uint64_t pointId);

private slots:
    void refreshSameWrapTree();
    void onCollectionsAdded(const std::vector<uint64_t>& collectionIds);
    void onCollectionChanged(uint64_t collectionId);
    void onCollectionRemoved(uint64_t collectionId);
    void onPointAdded(const ColPoint& point);
    void onPointsAdded(const std::vector<ColPoint>& points);
    void onPointChanged(const ColPoint& point);
    void onPointRemoved(uint64_t pointId);
    void onSelectionChanged(const QItemSelection& selected, const QItemSelection& deselected);
    void showContextMenu(const QPoint& pos);

signals:
    void collectionSelected(uint64_t collectionId);
    void pointSelected(uint64_t pointId);
    void pointDoubleClicked(uint64_t pointId);
    void focusViewsRequested(uint64_t collectionId, uint64_t pointId);
    void sameWrapAnnotationToggled(bool enabled);
    void sameWrapAnnotationSpacingChanged(double spacing);
    void sameWrapAnnotationPolylineOpacityChanged(double opacity);
    void sameWrapAnnotationMergeToggled(bool enabled);
    void sameWrapAnnotationPathTypeChanged(int pathType);
    void sameWrapAnnotationFilterTypeChanged(int filterType);
    void sameWrapAnnotationFilterKernelSizeChanged(int kernelSize);
    void sameWrapAnnotationClearRequested();
    void relWindingAnnotationToggled(bool enabled);
    void relWindingIntersectionSourceChanged(int source);
    void relWindingPatchToleranceChanged(double tolerance);

private:
    void setupUi();
    QStandardItem* findCollectionItem(uint64_t collectionId) const;
    QStandardItem* findPointItem(uint64_t pointId) const;
    void appendCollectionRow(const VCCollection::Collection& collection);
    void appendPointRow(QStandardItem* collectionItem,
                        const VCCollection::Collection& collection,
                        const ColPoint& point);
    void updateCollectionCount(QStandardItem* collectionItem);

    VCCollection* _pointCollection{nullptr};
    uint64_t _selectedCollectionId{0};
    uint64_t _selectedPointId{0};
    QCheckBox* _chkSameWrapAnnotation{nullptr};
    QCheckBox* _chkSameWrapMerge{nullptr};
    QComboBox* _sameWrapPathTypeCombo{nullptr};
    QComboBox* _sameWrapFilterTypeCombo{nullptr};
    QSpinBox* _sameWrapFilterKernelSpinbox{nullptr};
    QDoubleSpinBox* _sameWrapSpacingSpinbox{nullptr};
    QDoubleSpinBox* _sameWrapPolylineOpacitySpinbox{nullptr};
    QPushButton* _clearSameWrapAnnotationButton{nullptr};
    QComboBox* _relWindingIntersectionSourceCombo{nullptr};
    QDoubleSpinBox* _relWindingPatchToleranceSpinbox{nullptr};
    QPushButton* _relWindingAnnotationButton{nullptr};
    QTreeView* _sameWrapTreeView{nullptr};
    QStandardItemModel* _sameWrapModel{nullptr};
    std::unordered_map<uint64_t, QPersistentModelIndex> _pointItems;
};
