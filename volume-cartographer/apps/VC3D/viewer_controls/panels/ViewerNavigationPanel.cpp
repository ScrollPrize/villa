#include "viewer_controls/panels/ViewerNavigationPanel.hpp"

#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"
#include "elements/LabeledControlRow.hpp"

#include <QDoubleSpinBox>
#include <QSettings>
#include <QSignalBlocker>
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

    // Z-scroll sensitivity is also driven by the Shift+G / Shift+H keybindings, so it is
    // routed through ViewerManager (the single source of truth) rather than writing the
    // setting directly. The spin box both drives and reflects that value.
    auto* zScrollRow = new LabeledControlRow(tr("Z-scroll sensitivity"), this);
    _zScrollSpin = new QDoubleSpinBox(zScrollRow);
    _zScrollSpin->setRange(0.1, 100.0);
    _zScrollSpin->setSingleStep(0.1);
    _zScrollSpin->setDecimals(1);
    _zScrollSpin->setValue(_viewerManager ? _viewerManager->zScrollSensitivity()
                                          : viewer::ZSCROLL_SENSITIVITY_DEFAULT);
    zScrollRow->addControl(_zScrollSpin);
    zScrollRow->addStretch(1);
    layout->addWidget(zScrollRow);

    connect(_zScrollSpin, &QDoubleSpinBox::valueChanged, this, [this](double value) {
        if (_viewerManager) {
            _viewerManager->setZScrollSensitivity(value);
        }
    });
    if (_viewerManager) {
        connect(_viewerManager, &ViewerManager::zScrollSensitivityChanged, this, [this](double value) {
            if (!_zScrollSpin) {
                return;
            }
            QSignalBlocker blocker(_zScrollSpin);
            _zScrollSpin->setValue(value);
        });
    }
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
