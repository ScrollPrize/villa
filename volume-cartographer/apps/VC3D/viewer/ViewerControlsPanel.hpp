#pragma once

#include <QWidget>

class QLabel;
class QPushButton;
class QCheckBox;
class QDoubleSpinBox;
class QScrollArea;
class QSlider;
class QSpinBox;
class QVBoxLayout;
class QWidget;
class ViewerManager;
class WindowRangeWidget;

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

    struct TransformControls {
        QCheckBox* preview{nullptr};
        QCheckBox* scaleOnly{nullptr};
        QCheckBox* invert{nullptr};
        QSpinBox* scale{nullptr};
        QPushButton* loadAffine{nullptr};
        QPushButton* saveTransformed{nullptr};
        QLabel* status{nullptr};
    };

    explicit ViewerControlsPanel(const UiRefs& uiRefs,
                                 ViewerManager* viewerManager,
                                 QWidget* parent = nullptr);

    const TransformControls& transformControls() const { return _transformControls; }
    void setViewControlsEnabled(bool enabled);
    void setOverlayWindowAvailable(bool available);

signals:
    void zoomInRequested();
    void zoomOutRequested();
    void sliceStepSizeChanged(int value);
    void statusMessageRequested(QString text, int timeoutMs);

private:
    QWidget* detachScrollContents(QScrollArea* scrollArea, QWidget* contents);
    QWidget* createNormalVisualizationContainer(QWidget* sourceContents);
    QWidget* createTransformsPanel();
    void addViewerGroups();
    void addViewExtras(QVBoxLayout* viewExtrasLayout);
    void addNavigationGroup();
    void setupViewerControlWiring();
    void setupNormalVisualizationControls();
    void setupWindowRangeControls();
    void setupIntersectionControls();
    void updateNormalVisualizationControlsEnabled(bool enabled);
    void updateOverlayWindowControlsEnabled();
    void rememberGroupState(class CollapsibleSettingsGroup* group, const char* key);
    class CollapsibleSettingsGroup* addViewerGroup(const QString& title,
                                                   QWidget* contents,
                                                   const char* key,
                                                   bool defaultExpanded);

    UiRefs _uiRefs;
    ViewerManager* _viewerManager{nullptr};
    TransformControls _transformControls;
    WindowRangeWidget* _volumeWindowWidget{nullptr};
    WindowRangeWidget* _overlayWindowWidget{nullptr};
    bool _viewControlsEnabled{true};
    bool _overlayWindowAvailable{false};
};
