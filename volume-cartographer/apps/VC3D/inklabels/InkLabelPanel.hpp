#pragma once

#include <QWidget>

class InkLabelController;
class QCheckBox;
class QDoubleSpinBox;
class QLabel;
class QPushButton;
class QSlider;

// Controls for InkLabelController: labelling on/off, tool, brush radius,
// label opacity, undo/redo and the label files.
class InkLabelPanel : public QWidget
{
    Q_OBJECT

public:
    explicit InkLabelPanel(InkLabelController* controller, QWidget* parent = nullptr);

private:
    void updateFromController();
    void saveToFolder();
    void loadFromFile();

    InkLabelController* _controller{nullptr};
    QCheckBox* _enabled{nullptr};
    QPushButton* _ink{nullptr};
    QPushButton* _noInk{nullptr};
    QPushButton* _erase{nullptr};
    QDoubleSpinBox* _radius{nullptr};
    QSlider* _opacity{nullptr};
    QLabel* _opacityValue{nullptr};
    QCheckBox* _visible{nullptr};
    QPushButton* _undo{nullptr};
    QPushButton* _redo{nullptr};
    QPushButton* _save{nullptr};
    QPushButton* _saveTo{nullptr};
    QPushButton* _load{nullptr};
    QLabel* _status{nullptr};
};
