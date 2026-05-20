#pragma once

#include <QDockWidget>

#include "vc/ui/VCCollection.hpp"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
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
    double sameWrapAnnotationMergeTolerance() const;
    double sameWrapAnnotationPolylineOpacity() const;
    bool sameWrapAnnotationMergeEnabled() const;
    int sameWrapAnnotationPathType() const;
    int sameWrapAnnotationFilterType() const;
    int sameWrapAnnotationFilterKernelSize() const;

public slots:
    void setRelWindingAnnotationChecked(bool checked);

private slots:
    void refreshSameWrapTree();

signals:
    void sameWrapAnnotationToggled(bool enabled);
    void sameWrapAnnotationSpacingChanged(double spacing);
    void sameWrapAnnotationMergeToleranceChanged(double tolerance);
    void sameWrapAnnotationPolylineOpacityChanged(double opacity);
    void sameWrapAnnotationMergeToggled(bool enabled);
    void sameWrapAnnotationPathTypeChanged(int pathType);
    void sameWrapAnnotationFilterTypeChanged(int filterType);
    void sameWrapAnnotationFilterKernelSizeChanged(int kernelSize);
    void sameWrapAnnotationClearRequested();
    void relWindingAnnotationToggled(bool enabled);

private:
    void setupUi();

    VCCollection* _pointCollection{nullptr};
    QCheckBox* _chkSameWrapAnnotation{nullptr};
    QCheckBox* _chkSameWrapMerge{nullptr};
    QComboBox* _sameWrapPathTypeCombo{nullptr};
    QComboBox* _sameWrapFilterTypeCombo{nullptr};
    QSpinBox* _sameWrapFilterKernelSpinbox{nullptr};
    QDoubleSpinBox* _sameWrapSpacingSpinbox{nullptr};
    QDoubleSpinBox* _sameWrapMergeToleranceSpinbox{nullptr};
    QDoubleSpinBox* _sameWrapPolylineOpacitySpinbox{nullptr};
    QPushButton* _clearSameWrapAnnotationButton{nullptr};
    QPushButton* _relWindingAnnotationButton{nullptr};
    QTreeView* _sameWrapTreeView{nullptr};
    QStandardItemModel* _sameWrapModel{nullptr};
};
