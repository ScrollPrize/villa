#pragma once

#include <QWidget>

class ViewerManager;

class ViewerNavigationPanel : public QWidget
{
    Q_OBJECT

public:
    explicit ViewerNavigationPanel(ViewerManager* viewerManager, QWidget* parent = nullptr);

private:
    void addSensitivityControl(class QVBoxLayout* layout,
                               const QString& label,
                               const char* settingsKey,
                               double defaultValue);

    ViewerManager* _viewerManager{nullptr};
};
