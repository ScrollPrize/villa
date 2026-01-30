#pragma once

#include <QWidget>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QPushButton;
class QSpinBox;
class CollapsibleSettingsGroup;

class SegmentationCellReoptPanel : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationCellReoptPanel(QWidget* parent = nullptr);

    CollapsibleSettingsGroup* cellReoptGroup() const { return _groupCellReopt; }
    QCheckBox* modeCheck() const { return _chkCellReoptMode; }
    QSpinBox* maxStepsSpin() const { return _spinCellReoptMaxSteps; }
    QSpinBox* maxPointsSpin() const { return _spinCellReoptMaxPoints; }
    QDoubleSpinBox* minSpacingSpin() const { return _spinCellReoptMinSpacing; }
    QDoubleSpinBox* perimeterOffsetSpin() const { return _spinCellReoptPerimeterOffset; }
    QComboBox* collectionCombo() const { return _comboCellReoptCollection; }
    QPushButton* runButton() const { return _btnCellReoptRun; }

private:
    CollapsibleSettingsGroup* _groupCellReopt{nullptr};
    QCheckBox* _chkCellReoptMode{nullptr};
    QSpinBox* _spinCellReoptMaxSteps{nullptr};
    QSpinBox* _spinCellReoptMaxPoints{nullptr};
    QDoubleSpinBox* _spinCellReoptMinSpacing{nullptr};
    QDoubleSpinBox* _spinCellReoptPerimeterOffset{nullptr};
    QComboBox* _comboCellReoptCollection{nullptr};
    QPushButton* _btnCellReoptRun{nullptr};
};
