#include "viewer_controls/panels/ViewerViewExtrasPanel.hpp"

#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"
#include "elements/LabeledControlRow.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QSettings>
#include <QSpinBox>
#include <QVBoxLayout>

#include <algorithm>

ViewerViewExtrasPanel::ViewerViewExtrasPanel(ViewerManager* viewerManager, QWidget* parent)
    : QWidget(parent)
    , _viewerManager(viewerManager)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    auto* interpRow = new LabeledControlRow(tr("Interpolation"), this);
    auto* cmbInterp = new QComboBox(interpRow);
    cmbInterp->addItem(tr("Nearest"));
    cmbInterp->addItem(tr("Trilinear"));
    cmbInterp->setCurrentIndex(std::clamp(settings.value(vc3d::settings::perf::INTERPOLATION_METHOD, 1).toInt(), 0, 1));
    interpRow->addControl(cmbInterp, 1);
    layout->addWidget(interpRow);

    connect(cmbInterp, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int idx) {
        QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
        s.setValue(vc3d::settings::perf::INTERPOLATION_METHOD, idx);
        if (_viewerManager) {
            _viewerManager->forEachBaseViewer([](VolumeViewerBase* viewer) {
                if (!viewer) {
                    return;
                }
                viewer->reloadPerfSettings();
                viewer->renderVisible(true);
            });
        }
    });

    auto* maxResolutionRow = new LabeledControlRow(tr("Max displayed resolution"), this);
    auto* maxResolution = new QSpinBox(maxResolutionRow);
    maxResolution->setRange(0, 5);
    maxResolution->setValue(std::clamp(
        settings.value(vc3d::settings::viewer::MAX_DISPLAYED_RESOLUTION,
                       vc3d::settings::viewer::MAX_DISPLAYED_RESOLUTION_DEFAULT).toInt(),
        0,
        5));
    maxResolution->setToolTip(
        tr("Clamp rendering and chunk fetches so pyramid levels finer than this scale are not requested."));
    maxResolutionRow->setLabelToolTip(maxResolution->toolTip());
    maxResolutionRow->addControl(maxResolution);
    maxResolutionRow->addStretch(1);
    layout->addWidget(maxResolutionRow);

    connect(maxResolution, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
        s.setValue(vc3d::settings::viewer::MAX_DISPLAYED_RESOLUTION, std::clamp(value, 0, 5));
        if (_viewerManager) {
            _viewerManager->forEachBaseViewer([](VolumeViewerBase* viewer) {
                if (!viewer) {
                    return;
                }
                viewer->reloadPerfSettings();
                viewer->renderVisible(true);
            });
        }
    });

    auto* chkHighlight = new QCheckBox(tr("Highlight downscaled chunks"), this);
    chkHighlight->setToolTip(
        tr("Tint pixels sourced from a coarser pyramid level than the current zoom "
           "target. Green = 1 level coarser; red = 5+ levels coarser. Untinted pixels "
           "rendered at the requested resolution."));
    chkHighlight->setChecked(settings.value("viewer_controls/highlight_downscaled", false).toBool());
    layout->addWidget(chkHighlight);

    connect(chkHighlight, &QCheckBox::toggled, this, [this](bool on) {
        QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
        s.setValue("viewer_controls/highlight_downscaled", on);
        if (_viewerManager) {
            _viewerManager->forEachBaseViewer([](VolumeViewerBase* viewer) {
                if (!viewer) {
                    return;
                }
                viewer->reloadPerfSettings();
                viewer->renderVisible(true);
            });
        }
    });
}
