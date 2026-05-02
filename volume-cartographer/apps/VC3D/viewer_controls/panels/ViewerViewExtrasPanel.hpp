#pragma once

#include <QWidget>

class ViewerManager;

class ViewerViewExtrasPanel : public QWidget
{
    Q_OBJECT

public:
    explicit ViewerViewExtrasPanel(ViewerManager* viewerManager, QWidget* parent = nullptr);

private:
    ViewerManager* _viewerManager{nullptr};
};
