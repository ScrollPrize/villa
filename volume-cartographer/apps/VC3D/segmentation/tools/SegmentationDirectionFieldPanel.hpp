#pragma once

#include <QWidget>

class QComboBox;
class QDoubleSpinBox;
class QLineEdit;
class QListWidget;
class QPushButton;
class QToolButton;
class CollapsibleSettingsGroup;

class SegmentationDirectionFieldPanel : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationDirectionFieldPanel(QWidget* parent = nullptr);

    CollapsibleSettingsGroup* directionFieldGroup() const { return _groupDirectionField; }
    QLineEdit* pathEdit() const { return _directionFieldPathEdit; }
    QToolButton* browseButton() const { return _directionFieldBrowseButton; }
    QComboBox* orientationCombo() const { return _comboDirectionFieldOrientation; }
    QComboBox* scaleCombo() const { return _comboDirectionFieldScale; }
    QDoubleSpinBox* weightSpin() const { return _spinDirectionFieldWeight; }
    QPushButton* addButton() const { return _directionFieldAddButton; }
    QPushButton* removeButton() const { return _directionFieldRemoveButton; }
    QPushButton* clearButton() const { return _directionFieldClearButton; }
    QListWidget* listWidget() const { return _directionFieldList; }

private:
    CollapsibleSettingsGroup* _groupDirectionField{nullptr};
    QLineEdit* _directionFieldPathEdit{nullptr};
    QToolButton* _directionFieldBrowseButton{nullptr};
    QComboBox* _comboDirectionFieldOrientation{nullptr};
    QComboBox* _comboDirectionFieldScale{nullptr};
    QDoubleSpinBox* _spinDirectionFieldWeight{nullptr};
    QPushButton* _directionFieldAddButton{nullptr};
    QPushButton* _directionFieldRemoveButton{nullptr};
    QPushButton* _directionFieldClearButton{nullptr};
    QListWidget* _directionFieldList{nullptr};
};
