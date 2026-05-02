#pragma once

#include <QWidget>

class QLabel;
class QPushButton;
class QCheckBox;
class QScrollArea;
class QSpinBox;
class QVBoxLayout;
class QWidget;
class ViewerManager;

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

        QScrollArea* preprocessingScrollArea{nullptr};
        QWidget* preprocessingContents{nullptr};

        QScrollArea* postprocessingScrollArea{nullptr};
        QWidget* postprocessingContents{nullptr};
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

private:
    QWidget* detachScrollContents(QScrollArea* scrollArea, QWidget* contents);
    QWidget* createNormalVisualizationContainer(QWidget* sourceContents);
    QWidget* createTransformsPanel();
    void addViewerGroups();
    void addViewExtras(QVBoxLayout* viewExtrasLayout);
    void addNavigationGroup();
    void rememberGroupState(class CollapsibleSettingsGroup* group, const char* key);
    class CollapsibleSettingsGroup* addViewerGroup(const QString& title,
                                                   QWidget* contents,
                                                   const char* key,
                                                   bool defaultExpanded);

    UiRefs _uiRefs;
    ViewerManager* _viewerManager{nullptr};
    TransformControls _transformControls;
};
