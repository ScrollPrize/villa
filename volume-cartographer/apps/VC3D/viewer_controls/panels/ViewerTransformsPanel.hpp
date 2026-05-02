#pragma once

#include <QWidget>

class QLabel;
class QPushButton;
class QCheckBox;
class QSpinBox;

struct ViewerTransformControls {
    QCheckBox* preview{nullptr};
    QCheckBox* scaleOnly{nullptr};
    QCheckBox* invert{nullptr};
    QSpinBox* scale{nullptr};
    QPushButton* loadAffine{nullptr};
    QPushButton* saveTransformed{nullptr};
    QLabel* status{nullptr};
};

class ViewerTransformsPanel : public QWidget
{
    Q_OBJECT

public:
    explicit ViewerTransformsPanel(QWidget* parent = nullptr);

    [[nodiscard]] const ViewerTransformControls& controls() const { return _controls; }

private:
    ViewerTransformControls _controls;
};
