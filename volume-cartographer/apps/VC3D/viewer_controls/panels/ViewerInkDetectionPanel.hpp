#pragma once

#include <QWidget>

class InkLabelController;
class QComboBox;
class QLabel;
class QPushButton;
class QSlider;
class ViewerManager;

class ViewerInkDetectionPanel : public QWidget
{
    Q_OBJECT

public:
    // With `labels` set, the ink label section is added below the detection rows.
    explicit ViewerInkDetectionPanel(ViewerManager* viewerManager, InkLabelController* labels = nullptr, QWidget* parent = nullptr);

private:
    void populateDetections();
    void populateColormaps();
    void updateControlState();

    ViewerManager* _viewerManager{nullptr};
    QComboBox* _detectionCombo{nullptr};
    QComboBox* _colormapCombo{nullptr};
    QSlider* _opacitySlider{nullptr};
    QLabel* _opacityValue{nullptr};
    QPushButton* _horizontalFlipButton{nullptr};
    QPushButton* _verticalFlipButton{nullptr};
};
