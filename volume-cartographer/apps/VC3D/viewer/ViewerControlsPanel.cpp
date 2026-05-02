#include "viewer/ViewerControlsPanel.hpp"

#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "VolumeViewerBase.hpp"
#include "elements/CollapsibleSettingsGroup.hpp"
#include "elements/LabeledControlRow.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QGridLayout>
#include <QLabel>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QVBoxLayout>

#include <algorithm>

namespace
{

void moveGridLayoutItems(QGridLayout* from, QGridLayout* to, QWidget* newParent)
{
    if (!from || !to) {
        return;
    }
    to->setContentsMargins(from->contentsMargins());
    to->setHorizontalSpacing(from->horizontalSpacing());
    to->setVerticalSpacing(from->verticalSpacing());
    for (int column = 0; column < from->columnCount(); ++column) {
        to->setColumnStretch(column, from->columnStretch(column));
        to->setColumnMinimumWidth(column, from->columnMinimumWidth(column));
    }
    for (int row = 0; row < from->rowCount(); ++row) {
        to->setRowStretch(row, from->rowStretch(row));
        to->setRowMinimumHeight(row, from->rowMinimumHeight(row));
    }
    for (int index = from->count() - 1; index >= 0; --index) {
        int row = 0;
        int column = 0;
        int rowSpan = 1;
        int columnSpan = 1;
        from->getItemPosition(index, &row, &column, &rowSpan, &columnSpan);
        if (auto* item = from->takeAt(index)) {
            if (newParent) {
                if (auto* widget = item->widget()) {
                    widget->setParent(newParent);
                } else if (auto* layout = item->layout()) {
                    layout->setParent(newParent);
                }
            }
            to->addItem(item, row, column, rowSpan, columnSpan, item->alignment());
        }
    }
}

} // namespace

ViewerControlsPanel::ViewerControlsPanel(const UiRefs& uiRefs,
                                         ViewerManager* viewerManager,
                                         QWidget* parent)
    : QWidget(parent)
    , _uiRefs(uiRefs)
    , _viewerManager(viewerManager)
{
    if (_uiRefs.contents && _uiRefs.contents != this) {
        auto* existingLayout = qobject_cast<QVBoxLayout*>(_uiRefs.contents->layout());
        if (!existingLayout) {
            existingLayout = new QVBoxLayout(_uiRefs.contents);
            existingLayout->setContentsMargins(4, 4, 4, 4);
            existingLayout->setSpacing(8);
        }
    }

    addViewerGroups();
}

QWidget* ViewerControlsPanel::detachScrollContents(QScrollArea* scrollArea, QWidget* contents)
{
    if (!contents) {
        return nullptr;
    }
    if (scrollArea && scrollArea->widget() == contents) {
        scrollArea->takeWidget();
    }
    contents->setParent(nullptr);
    return contents;
}

QWidget* ViewerControlsPanel::createNormalVisualizationContainer(QWidget* sourceContents)
{
    auto* container = new QWidget(_uiRefs.contents);
    auto* layout = new QGridLayout(container);
    moveGridLayoutItems(qobject_cast<QGridLayout*>(sourceContents ? sourceContents->layout() : nullptr),
                        layout,
                        container);
    return container;
}

QWidget* ViewerControlsPanel::createTransformsPanel()
{
    auto* container = new QWidget(_uiRefs.contents);
    auto* layout = new QVBoxLayout(container);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    _transformControls.preview = new QCheckBox(tr("Preview Result"), container);
    _transformControls.preview->setToolTip(
        tr("Preview the scaled and/or affine-transformed segmentation."));
    layout->addWidget(_transformControls.preview);

    _transformControls.scaleOnly = new QCheckBox(tr("Scale Only"), container);
    _transformControls.scaleOnly->setToolTip(
        tr("Ignore any loaded or volume affine and apply scale only."));
    layout->addWidget(_transformControls.scaleOnly);

    _transformControls.invert = new QCheckBox(tr("Invert Affine"), container);
    _transformControls.invert->setToolTip(
        tr("Invert the loaded affine. Ignored when no affine transform is loaded."));
    layout->addWidget(_transformControls.invert);

    auto* scaleRow = new LabeledControlRow(tr("Scale"), container);
    _transformControls.scale = new QSpinBox(scaleRow);
    _transformControls.scale->setMinimum(1);
    _transformControls.scale->setMaximum(1000);
    _transformControls.scale->setValue(1);
    _transformControls.scale->setToolTip(
        tr("Multiply segmentation points by this integer. Works with or without an affine transform."));
    scaleRow->setLabelToolTip(_transformControls.scale->toolTip());
    scaleRow->addControl(_transformControls.scale);
    scaleRow->addStretch(1);
    layout->addWidget(scaleRow);

    _transformControls.loadAffine = new QPushButton(tr("Load Affine (Optional)"), container);
    _transformControls.loadAffine->setToolTip(
        tr("Load an affine JSON from a local path or URL. Leave the dialog blank to return to the current volume transform."));
    layout->addWidget(_transformControls.loadAffine);

    _transformControls.saveTransformed = new QPushButton(tr("Save Transformed"), container);
    _transformControls.saveTransformed->setToolTip(
        tr("Save a new surface using the current scale and optional affine transform."));
    layout->addWidget(_transformControls.saveTransformed);

    _transformControls.status = new QLabel(container);
    _transformControls.status->setWordWrap(true);
    _transformControls.status->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(_transformControls.status);

    return container;
}

void ViewerControlsPanel::addViewerGroups()
{
    using namespace vc3d::settings;

    auto* viewGroup = addViewerGroup(tr("View"),
                                     detachScrollContents(_uiRefs.viewScrollArea, _uiRefs.viewContents),
                                     viewer::GROUP_VIEW_EXPANDED,
                                     viewer::GROUP_VIEW_EXPANDED_DEFAULT);
    auto* viewExtrasLayout = viewGroup ? viewGroup->contentLayout()
                                       : qobject_cast<QVBoxLayout*>(_uiRefs.contents->layout());
    addViewExtras(viewExtrasLayout);
    addNavigationGroup();

    addViewerGroup(tr("Overlay"),
                   detachScrollContents(_uiRefs.overlayScrollArea, _uiRefs.overlayContents),
                   viewer::GROUP_OVERLAY_EXPANDED,
                   viewer::GROUP_OVERLAY_EXPANDED_DEFAULT);
    addViewerGroup(tr("Composite View"),
                   detachScrollContents(_uiRefs.compositeScrollArea, _uiRefs.compositeContents),
                   viewer::GROUP_COMPOSITE_EXPANDED,
                   viewer::GROUP_COMPOSITE_EXPANDED_DEFAULT);
    addViewerGroup(tr("Render Settings"),
                   detachScrollContents(_uiRefs.renderSettingsScrollArea, _uiRefs.renderSettingsContents),
                   viewer::GROUP_RENDER_SETTINGS_EXPANDED,
                   viewer::GROUP_RENDER_SETTINGS_EXPANDED_DEFAULT);
    addViewerGroup(tr("Normal Visualization"),
                   createNormalVisualizationContainer(_uiRefs.normalVisualizationContents),
                   viewer::GROUP_NORMAL_VIS_EXPANDED,
                   viewer::GROUP_NORMAL_VIS_EXPANDED_DEFAULT);
    addViewerGroup(tr("Preprocessing"),
                   detachScrollContents(_uiRefs.preprocessingScrollArea, _uiRefs.preprocessingContents),
                   viewer::GROUP_PREPROCESSING_EXPANDED,
                   viewer::GROUP_PREPROCESSING_EXPANDED_DEFAULT);
    addViewerGroup(tr("Postprocessing"),
                   detachScrollContents(_uiRefs.postprocessingScrollArea, _uiRefs.postprocessingContents),
                   viewer::GROUP_POSTPROCESSING_EXPANDED,
                   viewer::GROUP_POSTPROCESSING_EXPANDED_DEFAULT);
    addViewerGroup(tr("Transforms"),
                   createTransformsPanel(),
                   viewer::GROUP_TRANSFORMS_EXPANDED,
                   viewer::GROUP_TRANSFORMS_EXPANDED_DEFAULT);

    if (auto* layout = qobject_cast<QVBoxLayout*>(_uiRefs.contents->layout())) {
        layout->addStretch(1);
    }
}

void ViewerControlsPanel::addViewExtras(QVBoxLayout* viewExtrasLayout)
{
    if (!viewExtrasLayout) {
        return;
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    auto* interpRow = new LabeledControlRow(tr("Interpolation"), _uiRefs.contents);
    auto* cmbInterp = new QComboBox(interpRow);
    cmbInterp->addItem(tr("Nearest"));
    cmbInterp->addItem(tr("Trilinear"));
    cmbInterp->setCurrentIndex(std::clamp(settings.value(vc3d::settings::perf::INTERPOLATION_METHOD, 1).toInt(), 0, 1));
    interpRow->addControl(cmbInterp, 1);
    viewExtrasLayout->addWidget(interpRow);

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

    auto* chkHighlight = new QCheckBox(tr("Highlight downscaled chunks"), _uiRefs.contents);
    chkHighlight->setToolTip(
        tr("Tint pixels sourced from a coarser pyramid level than the current zoom "
           "target. Green = 1 level coarser; red = 5+ levels coarser. Untinted pixels "
           "rendered at the requested resolution."));
    chkHighlight->setChecked(settings.value("viewer_controls/highlight_downscaled", false).toBool());
    viewExtrasLayout->addWidget(chkHighlight);

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

void ViewerControlsPanel::addNavigationGroup()
{
    using namespace vc3d::settings;

    auto* layout = qobject_cast<QVBoxLayout*>(_uiRefs.contents ? _uiRefs.contents->layout() : nullptr);
    if (!layout) {
        return;
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    auto* group = new CollapsibleSettingsGroup(tr("Navigation"), _uiRefs.contents);
    layout->addWidget(group);
    group->setExpanded(settings.value("viewer_controls/group_navigation_expanded", true).toBool());
    rememberGroupState(group, "viewer_controls/group_navigation_expanded");

    auto addSpin = [this, &settings, group](const QString& label, const char* settingsKey, float defaultValue) {
        auto* spin = group->addDoubleSpinBox(label, 0.1, 100.0, 0.1, 1);
        spin->setValue(settings.value(settingsKey, defaultValue).toDouble());
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
    };

    addSpin(tr("Pan sensitivity"), viewer::PAN_SENSITIVITY, viewer::PAN_SENSITIVITY_DEFAULT);
    addSpin(tr("Zoom sensitivity"), viewer::ZOOM_SENSITIVITY, viewer::ZOOM_SENSITIVITY_DEFAULT);
    addSpin(tr("Z-scroll sensitivity"), viewer::ZSCROLL_SENSITIVITY, viewer::ZSCROLL_SENSITIVITY_DEFAULT);
}

void ViewerControlsPanel::rememberGroupState(CollapsibleSettingsGroup* group, const char* key)
{
    if (!group) {
        return;
    }
    connect(group, &CollapsibleSettingsGroup::toggled, this, [key](bool expanded) {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.setValue(key, expanded);
    });
}

CollapsibleSettingsGroup* ViewerControlsPanel::addViewerGroup(const QString& title,
                                                              QWidget* contents,
                                                              const char* key,
                                                              bool defaultExpanded)
{
    auto* layout = qobject_cast<QVBoxLayout*>(_uiRefs.contents ? _uiRefs.contents->layout() : nullptr);
    if (!layout || !contents) {
        return nullptr;
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    auto* group = new CollapsibleSettingsGroup(title, _uiRefs.contents);
    group->contentLayout()->addWidget(contents);
    layout->addWidget(group);
    group->setExpanded(settings.value(key, defaultExpanded).toBool());
    rememberGroupState(group, key);
    return group;
}
