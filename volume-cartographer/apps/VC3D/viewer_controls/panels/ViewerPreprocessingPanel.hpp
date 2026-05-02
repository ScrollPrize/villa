#pragma once

#include <QWidget>

class QLabel;
class QScrollArea;
class QSlider;
class ViewerManager;

class ViewerPreprocessingPanel : public QWidget
{
    Q_OBJECT

public:
    struct UiRefs {
        QScrollArea* scrollArea{nullptr};
        QWidget* contents{nullptr};
        QSlider* isoCutoff{nullptr};
        QLabel* isoCutoffValue{nullptr};
    };

    explicit ViewerPreprocessingPanel(const UiRefs& uiRefs,
                                      ViewerManager* viewerManager,
                                      QWidget* parent = nullptr);

private:
    void setupControls();

    UiRefs _uiRefs;
    ViewerManager* _viewerManager{nullptr};
};
