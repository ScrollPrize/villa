#pragma once

#include <QWidget>

#include <functional>

class QCheckBox;
class QComboBox;
class QLabel;
class QScrollArea;
class QSpinBox;
class ViewerManager;
class VolumeViewerBase;

class ViewerCompositePanel : public QWidget
{
    Q_OBJECT

public:
    struct UiRefs {
        QScrollArea* scrollArea{nullptr};
        QWidget* contents{nullptr};

        QCheckBox* compositeEnabled{nullptr};
        QComboBox* compositeMode{nullptr};
        QSpinBox* layersInFront{nullptr};
        QSpinBox* layersBehind{nullptr};

        QLabel* alphaMinLabel{nullptr};
        QSpinBox* alphaMin{nullptr};
        QLabel* alphaMaxLabel{nullptr};
        QSpinBox* alphaMax{nullptr};
        QLabel* alphaThresholdLabel{nullptr};
        QSpinBox* alphaThreshold{nullptr};
        QLabel* materialLabel{nullptr};
        QSpinBox* material{nullptr};
        QCheckBox* reverseDirection{nullptr};

        QCheckBox* planeCompositeXY{nullptr};
        QCheckBox* planeCompositeXZ{nullptr};
        QCheckBox* planeCompositeYZ{nullptr};
        QSpinBox* planeLayersFront{nullptr};
        QSpinBox* planeLayersBehind{nullptr};
    };

    explicit ViewerCompositePanel(const UiRefs& uiRefs,
                                  ViewerManager* viewerManager,
                                  QWidget* parent = nullptr);

    void setViewerManager(ViewerManager* viewerManager);
    void toggleSegmentationComposite();
    void setSegmentationCompositeChecked(bool checked);

private:
    void setupControls();
    void applyInitialSettingsToViewer(VolumeViewerBase* viewer);
    void syncUiFromManager();
    void updateCompositeParamsVisibility();
    void applyToSegmentationViewer(const std::function<void(VolumeViewerBase*)>& apply);
    void applyToAllViewers(const std::function<void(VolumeViewerBase*)>& apply);
    void applyToPlaneViewers(const std::function<void(VolumeViewerBase*)>& apply);

    UiRefs _uiRefs;
    ViewerManager* _viewerManager{nullptr};
};
