#pragma once

#include <QWidget>

class QComboBox;
class QLabel;
class QPushButton;
class QSlider;
class ViewerManager;

class ViewerInkDetectionPanel : public QWidget
{
    Q_OBJECT

public:
    explicit ViewerInkDetectionPanel(ViewerManager* viewerManager, QWidget* parent = nullptr);

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
