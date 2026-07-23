#pragma once

#include <QWidget>

class ViewerManager;

class ViewerViewExtrasPanel : public QWidget
{
    Q_OBJECT

public:
    explicit ViewerViewExtrasPanel(ViewerManager* viewerManager, QWidget* parent = nullptr);
    void setViewerManager(ViewerManager* viewerManager) { _viewerManager = viewerManager; }

private:
    ViewerManager* _viewerManager{nullptr};
};
