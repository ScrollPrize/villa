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

        QPushButton* zoomInButton{nullptr};
        QPushButton* zoomOutButton{nullptr};
        QWidget* volumeWindowContainer{nullptr};
        QWidget* overlayWindowContainer{nullptr};
        QDoubleSpinBox* intersectionThicknessSpin{nullptr};
    };

    explicit ViewerControlsPanel(const UiRefs& uiRefs,
                                 ViewerManager* viewerManager,
                                 QWidget* parent = nullptr);

    ViewerTransformsPanel* transformsPanel() const { return _transformsPanel; }
    ViewerCompositePanel* compositePanel() const { return _compositePanel; }
    void setTransformsPanel(ViewerTransformsPanel* panel) { _transformsPanel = panel; }
    void toggleSegmentationComposite();
    void setViewControlsEnabled(bool enabled);
    void setOverlayWindowAvailable(bool available);

signals:
    void zoomInRequested();
    void zoomOutRequested();
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
    bool _viewControlsEnabled{true};
    bool _overlayWindowAvailable{false};
};
