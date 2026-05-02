#include "viewer_controls/panels/ViewerNavigationPanel.hpp"

#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "VolumeViewerBase.hpp"
#include "elements/LabeledControlRow.hpp"

#include <QDoubleSpinBox>
#include <QSettings>
#include <QVBoxLayout>

ViewerNavigationPanel::ViewerNavigationPanel(ViewerManager* viewerManager, QWidget* parent)
    : QWidget(parent)
    , _viewerManager(viewerManager)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    using namespace vc3d::settings;
    addSensitivityControl(layout, tr("Pan sensitivity"), viewer::PAN_SENSITIVITY, viewer::PAN_SENSITIVITY_DEFAULT);
    addSensitivityControl(layout, tr("Zoom sensitivity"), viewer::ZOOM_SENSITIVITY, viewer::ZOOM_SENSITIVITY_DEFAULT);
    addSensitivityControl(layout, tr("Z-scroll sensitivity"), viewer::ZSCROLL_SENSITIVITY, viewer::ZSCROLL_SENSITIVITY_DEFAULT);
}

void ViewerNavigationPanel::addSensitivityControl(QVBoxLayout* layout,
                                                  const QString& label,
                                                  const char* settingsKey,
                                                  double defaultValue)
{
    if (!layout) {
        return;
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    auto* row = new LabeledControlRow(label, this);
    auto* spin = new QDoubleSpinBox(row);
    spin->setRange(0.1, 100.0);
    spin->setSingleStep(0.1);
    spin->setDecimals(1);
    spin->setValue(settings.value(settingsKey, defaultValue).toDouble());
    row->addControl(spin);
    row->addStretch(1);
    layout->addWidget(row);

    connect(spin, &QDoubleSpinBox::valueChanged, this, [this, settingsKey](double value) {
        QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
        s.setValue(settingsKey, value);
        if (_viewerManager) {
            _viewerManager->forEachBaseViewer([](VolumeViewerBase* viewer) {
                if (viewer) {
                    viewer->reloadPerfSettings();
                }
            });
        }
    });
}
