#pragma once

#include <QWidget>

class QLabel;
class QCheckBox;
class QSlider;
class ViewerManager;

class ViewerNormalVisualizationPanel : public QWidget
{
    Q_OBJECT

public:
    struct UiRefs {
        QWidget* contents{nullptr};
        QCheckBox* showSurfaceNormals{nullptr};
        QLabel* normalArrowLengthLabel{nullptr};
        QSlider* normalArrowLengthSlider{nullptr};
        QLabel* normalArrowLengthValueLabel{nullptr};
        QLabel* normalMaxArrowsLabel{nullptr};
        QSlider* normalMaxArrowsSlider{nullptr};
        QLabel* normalMaxArrowsValueLabel{nullptr};
    };

    explicit ViewerNormalVisualizationPanel(const UiRefs& uiRefs,
                                            ViewerManager* viewerManager,
                                            QWidget* parent = nullptr);

signals:
    void statusMessageRequested(QString text, int timeoutMs);

private:
    void setupControls();
    void updateControlsEnabled(bool enabled);

    UiRefs _uiRefs;
    ViewerManager* _viewerManager{nullptr};
};
