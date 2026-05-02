#pragma once

#include <QWidget>

class QDoubleSpinBox;
class QLabel;
class QPushButton;
class QCheckBox;
class QComboBox;
class QScrollArea;
class QSlider;
class QSpinBox;
class ViewerManager;
class WindowRangeWidget;
class ViewerCompositePanel;
class ViewerTransformsPanel;

class ViewerControlsPanel : public QWidget
{
    Q_OBJECT

public:
    struct UiRefs {
        QWidget* contents{nullptr};

        QScrollArea* viewScrollArea{nullptr};
        QWidget* viewContents{nullptr};

        QScrollArea* overlayScrollArea{nullptr};
        QWidget* overlayContents{nullptr};

        QScrollArea* compositeScrollArea{nullptr};
        QWidget* compositeContents{nullptr};
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

        QScrollArea* renderSettingsScrollArea{nullptr};
        QWidget* renderSettingsContents{nullptr};

        QWidget* normalVisualizationContents{nullptr};
        QCheckBox* showSurfaceNormals{nullptr};
        QLabel* normalArrowLengthLabel{nullptr};
        QSlider* normalArrowLengthSlider{nullptr};
        QLabel* normalArrowLengthValueLabel{nullptr};
        QLabel* normalMaxArrowsLabel{nullptr};
        QSlider* normalMaxArrowsSlider{nullptr};
        QLabel* normalMaxArrowsValueLabel{nullptr};

        QScrollArea* preprocessingScrollArea{nullptr};
        QWidget* preprocessingContents{nullptr};

        QScrollArea* postprocessingScrollArea{nullptr};
        QWidget* postprocessingContents{nullptr};

        QPushButton* zoomInButton{nullptr};
        QPushButton* zoomOutButton{nullptr};
        QSpinBox* sliceStepSizeSpin{nullptr};
        QWidget* volumeWindowContainer{nullptr};
        QWidget* overlayWindowContainer{nullptr};
        QSpinBox* intersectionOpacitySpin{nullptr};
        QDoubleSpinBox* intersectionThicknessSpin{nullptr};
    };

    explicit ViewerControlsPanel(const UiRefs& uiRefs,
                                 ViewerManager* viewerManager,
                                 QWidget* parent = nullptr);

    ViewerTransformsPanel* transformsPanel() const { return _transformsPanel; }
    ViewerCompositePanel* compositePanel() const { return _compositePanel; }
    void toggleSegmentationComposite();
    void setViewControlsEnabled(bool enabled);
    void setOverlayWindowAvailable(bool available);
    void setSliceStepSize(int value);

signals:
    void zoomInRequested();
    void zoomOutRequested();
    void sliceStepSizeChanged(int value);
    void statusMessageRequested(QString text, int timeoutMs);

private:
    QWidget* detachScrollContents(QScrollArea* scrollArea, QWidget* contents);
    void addViewerGroups();
    void setupViewerControlWiring();
    void setupWindowRangeControls();
    void setupIntersectionControls();
    void updateOverlayWindowControlsEnabled();
    void rememberGroupState(class CollapsibleSettingsGroup* group, const char* key);
    class CollapsibleSettingsGroup* addViewerGroup(const QString& title,
                                                   QWidget* contents,
                                                   const char* key,
                                                   bool defaultExpanded);

    UiRefs _uiRefs;
    ViewerManager* _viewerManager{nullptr};
    ViewerCompositePanel* _compositePanel{nullptr};
    ViewerTransformsPanel* _transformsPanel{nullptr};
    WindowRangeWidget* _volumeWindowWidget{nullptr};
    WindowRangeWidget* _overlayWindowWidget{nullptr};
    QSpinBox* _sliceStepSizeSpin{nullptr};
    bool _viewControlsEnabled{true};
    bool _overlayWindowAvailable{false};
};
