#include "viewer_controls/panels/ViewerViewExtrasPanel.hpp"

#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"
#include "elements/LabeledControlRow.hpp"

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
}
