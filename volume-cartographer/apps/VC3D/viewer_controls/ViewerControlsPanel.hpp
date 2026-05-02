#pragma once

#include "viewer_controls/panels/ViewerTransformsPanel.hpp"

#include <QWidget>

class QDoubleSpinBox;
class QLabel;
class QPushButton;
class QCheckBox;
class QScrollArea;
class QSlider;
class QSpinBox;
class ViewerManager;
class WindowRangeWidget;
class ViewerTransformsPanel;

class ViewerControlsPanel : public QWidget
{
    Q_OBJECT

public:
    using TransformControls = ViewerTransformControls;

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

    explicit ViewerControlsPanel(const UiRefs& uiRefs,
                                 ViewerManager* viewerManager,
                                 QWidget* parent = nullptr);

    const TransformControls& transformControls() const;
    void setViewControlsEnabled(bool enabled);
    void setOverlayWindowAvailable(bool available);

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
    ViewerTransformsPanel* _transformsPanel{nullptr};
    WindowRangeWidget* _volumeWindowWidget{nullptr};
    WindowRangeWidget* _overlayWindowWidget{nullptr};
    bool _viewControlsEnabled{true};
    bool _overlayWindowAvailable{false};
};
