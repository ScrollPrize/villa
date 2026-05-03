#pragma once

#include <QWidget>

#include <functional>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QScrollArea;
class QSlider;
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

        QLabel* methodScaleLabel{nullptr};
        QSlider* methodScale{nullptr};
        QLabel* methodScaleValue{nullptr};
        QLabel* methodParamLabel{nullptr};
        QSlider* methodParam{nullptr};
        QLabel* methodParamValue{nullptr};

        QLabel* blExtinctionLabel{nullptr};
        QDoubleSpinBox* blExtinction{nullptr};
        QLabel* blEmissionLabel{nullptr};
        QDoubleSpinBox* blEmission{nullptr};
        QLabel* blAmbientLabel{nullptr};
        QDoubleSpinBox* blAmbient{nullptr};

        QCheckBox* lightingEnabled{nullptr};
        QLabel* lightAzimuthLabel{nullptr};
        QSpinBox* lightAzimuth{nullptr};
        QLabel* lightElevationLabel{nullptr};
        QSpinBox* lightElevation{nullptr};
        QLabel* lightDiffuseLabel{nullptr};
        QDoubleSpinBox* lightDiffuse{nullptr};
        QLabel* lightAmbientLabel{nullptr};
        QDoubleSpinBox* lightAmbient{nullptr};
        QCheckBox* useVolumeGradients{nullptr};
        QLabel* shadowStepsLabel{nullptr};
        QSpinBox* shadowSteps{nullptr};

        QCheckBox* rakingEnabled{nullptr};
        QLabel* rakingAzimuthLabel{nullptr};
        QDoubleSpinBox* rakingAzimuth{nullptr};
        QLabel* rakingElevationLabel{nullptr};
        QDoubleSpinBox* rakingElevation{nullptr};
        QLabel* rakingStrengthLabel{nullptr};
        QDoubleSpinBox* rakingStrength{nullptr};
        QLabel* rakingDepthLabel{nullptr};
        QDoubleSpinBox* rakingDepthScale{nullptr};

        QCheckBox* preNormalizeLayers{nullptr};
        QCheckBox* preHistEqLayers{nullptr};
        QCheckBox* preTfEnabled{nullptr};
        QSpinBox* preTfX1{nullptr};
        QSpinBox* preTfY1{nullptr};
        QLabel* preTfKnot2Label{nullptr};
        QSpinBox* preTfX2{nullptr};
        QSpinBox* preTfY2{nullptr};
        QCheckBox* postTfEnabled{nullptr};
        QSpinBox* postTfX1{nullptr};
        QSpinBox* postTfY1{nullptr};
        QLabel* postTfKnot2Label{nullptr};
        QSpinBox* postTfX2{nullptr};
        QSpinBox* postTfY2{nullptr};
        QLabel* dvrAmbientLabel{nullptr};
        QDoubleSpinBox* dvrAmbient{nullptr};
        QLabel* pbrRoughnessLabel{nullptr};
        QDoubleSpinBox* pbrRoughness{nullptr};
        QLabel* pbrMetallicLabel{nullptr};
        QDoubleSpinBox* pbrMetallic{nullptr};

        QCheckBox* planeCompositeXY{nullptr};
        QCheckBox* planeCompositeXZ{nullptr};
        QCheckBox* planeCompositeYZ{nullptr};
        QSpinBox* planeLayersFront{nullptr};
        QSpinBox* planeLayersBehind{nullptr};
    };

    explicit ViewerCompositePanel(const UiRefs& uiRefs,
                                  ViewerManager* viewerManager,
                                  QWidget* parent = nullptr);

    void toggleSegmentationComposite();
    void setSegmentationCompositeChecked(bool checked);

private:
    void setupControls();
    void initializeExistingViewers();
    void applyInitialSettingsToViewer(VolumeViewerBase* viewer);
    void updateCompositeParamsVisibility();
    void updateRakingControlsEnabled(bool enabled);
    void applyToSegmentationViewer(const std::function<void(VolumeViewerBase*)>& apply);
    void applyToAllViewers(const std::function<void(VolumeViewerBase*)>& apply);
    void applyToPlaneViewers(const std::function<void(VolumeViewerBase*)>& apply);

    UiRefs _uiRefs;
    ViewerManager* _viewerManager{nullptr};
};
