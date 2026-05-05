#pragma once

#include <QWidget>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QScrollArea;
class QSpinBox;
class ViewerManager;
class VolumeViewerBase;

class ViewerPostprocessingPanel : public QWidget
{
    Q_OBJECT

public:
    struct UiRefs {
        QScrollArea* scrollArea{nullptr};
        QWidget* contents{nullptr};

        QComboBox* baseColormap{nullptr};
        QCheckBox* stretchValues{nullptr};
        QCheckBox* removeSmallComponents{nullptr};
        QLabel* minComponentSizeLabel{nullptr};
        QSpinBox* minComponentSize{nullptr};
        QCheckBox* claheEnabled{nullptr};
        QLabel* claheClipLimitLabel{nullptr};
        QDoubleSpinBox* claheClipLimit{nullptr};
        QLabel* claheTileSizeLabel{nullptr};
        QSpinBox* claheTileSize{nullptr};
    };

    explicit ViewerPostprocessingPanel(const UiRefs& uiRefs,
                                       ViewerManager* viewerManager,
                                       QWidget* parent = nullptr);

private:
    void setupControls();
    void setupColormapSelector();
    void setSmallComponentControlsEnabled(bool enabled);
    void setClaheControlsEnabled(bool enabled);
    VolumeViewerBase* segmentationBaseViewer() const;

    UiRefs _uiRefs;
    ViewerManager* _viewerManager{nullptr};
};
