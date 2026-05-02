#include "viewer/ViewerControlsPanel.hpp"

#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "VolumeViewerBase.hpp"
#include "WindowRangeWidget.hpp"
#include "elements/CollapsibleSettingsGroup.hpp"
#include "elements/LabeledControlRow.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>

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
    setupViewerControlWiring();
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

void ViewerControlsPanel::setViewControlsEnabled(bool enabled)
{
    _viewControlsEnabled = enabled;
    if (_volumeWindowWidget) {
        _volumeWindowWidget->setControlsEnabled(enabled);
    }
    updateOverlayWindowControlsEnabled();
}

void ViewerControlsPanel::setOverlayWindowAvailable(bool available)
{
    _overlayWindowAvailable = available;
    updateOverlayWindowControlsEnabled();
}

void ViewerControlsPanel::setupViewerControlWiring()
{
    setupNormalVisualizationControls();
    setupWindowRangeControls();
    setupIntersectionControls();

    if (_uiRefs.zoomInButton) {
        connect(_uiRefs.zoomInButton, &QPushButton::clicked, this, &ViewerControlsPanel::zoomInRequested);
    }
    if (_uiRefs.zoomOutButton) {
        connect(_uiRefs.zoomOutButton, &QPushButton::clicked, this, &ViewerControlsPanel::zoomOutRequested);
    }

    if (auto* spinSliceStep = _uiRefs.sliceStepSizeSpin) {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        int savedStep = settings.value(vc3d::settings::viewer::SLICE_STEP_SIZE,
                                       vc3d::settings::viewer::SLICE_STEP_SIZE_DEFAULT).toInt();
        savedStep = std::clamp(savedStep, spinSliceStep->minimum(), spinSliceStep->maximum());
        {
            QSignalBlocker blocker(spinSliceStep);
            spinSliceStep->setValue(savedStep);
        }
        if (_viewerManager) {
            _viewerManager->setSliceStepSize(savedStep);
        }
        connect(spinSliceStep, qOverload<int>(&QSpinBox::valueChanged), this, [this](int value) {
            if (_viewerManager) {
                _viewerManager->setSliceStepSize(value);
            }
            QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
            s.setValue(vc3d::settings::viewer::SLICE_STEP_SIZE, value);
            emit sliceStepSizeChanged(value);
        });
    }
}

void ViewerControlsPanel::setupNormalVisualizationControls()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    if (auto* chkShowNormals = _uiRefs.showSurfaceNormals) {
        bool showNormals = settings.value(vc3d::settings::viewer::SHOW_SURFACE_NORMALS,
                                          vc3d::settings::viewer::SHOW_SURFACE_NORMALS_DEFAULT).toBool();
        {
            QSignalBlocker blocker(chkShowNormals);
            chkShowNormals->setChecked(showNormals);
        }
        updateNormalVisualizationControlsEnabled(showNormals);

        connect(chkShowNormals, &QCheckBox::toggled, this, [this](bool checked) {
            using namespace vc3d::settings;
            QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
            s.setValue(viewer::SHOW_SURFACE_NORMALS, checked ? "1" : "0");
            if (_viewerManager) {
                _viewerManager->forEachBaseViewer([checked](VolumeViewerBase* viewer) {
                    if (viewer) {
                        viewer->setShowSurfaceNormals(checked);
                    }
                });
            }
            updateNormalVisualizationControlsEnabled(checked);
            emit statusMessageRequested(checked ? tr("Surface normals: ON") : tr("Surface normals: OFF"), 2000);
        });
    }

    if (auto* sliderArrowLength = _uiRefs.normalArrowLengthSlider) {
        int savedScale = settings.value(vc3d::settings::viewer::NORMAL_ARROW_LENGTH_SCALE,
                                        vc3d::settings::viewer::NORMAL_ARROW_LENGTH_SCALE_DEFAULT).toInt();
        savedScale = std::clamp(savedScale, sliderArrowLength->minimum(), sliderArrowLength->maximum());
        {
            QSignalBlocker blocker(sliderArrowLength);
            sliderArrowLength->setValue(savedScale);
        }
        if (_uiRefs.normalArrowLengthValueLabel) {
            _uiRefs.normalArrowLengthValueLabel->setText(tr("%1%").arg(savedScale));
        }
        const float scaleFloat = static_cast<float>(savedScale) / 100.0f;
        if (_viewerManager) {
            _viewerManager->forEachBaseViewer([scaleFloat](VolumeViewerBase* viewer) {
                if (viewer) {
                    viewer->setNormalArrowLengthScale(scaleFloat);
                }
            });
        }

        connect(sliderArrowLength, &QSlider::valueChanged, this, [this](int value) {
            using namespace vc3d::settings;
            QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
            s.setValue(viewer::NORMAL_ARROW_LENGTH_SCALE, value);
            if (_uiRefs.normalArrowLengthValueLabel) {
                _uiRefs.normalArrowLengthValueLabel->setText(tr("%1%").arg(value));
            }
            const float scaleFloat = static_cast<float>(value) / 100.0f;
            if (_viewerManager) {
                _viewerManager->forEachBaseViewer([scaleFloat](VolumeViewerBase* viewer) {
                    if (viewer) {
                        viewer->setNormalArrowLengthScale(scaleFloat);
                    }
                });
            }
        });
    }

    if (auto* sliderMaxArrows = _uiRefs.normalMaxArrowsSlider) {
        int savedMaxArrows = settings.value(vc3d::settings::viewer::NORMAL_MAX_ARROWS,
                                            vc3d::settings::viewer::NORMAL_MAX_ARROWS_DEFAULT).toInt();
        savedMaxArrows = std::clamp(savedMaxArrows, sliderMaxArrows->minimum(), sliderMaxArrows->maximum());
        {
            QSignalBlocker blocker(sliderMaxArrows);
            sliderMaxArrows->setValue(savedMaxArrows);
        }
        if (_uiRefs.normalMaxArrowsValueLabel) {
            _uiRefs.normalMaxArrowsValueLabel->setText(QString::number(savedMaxArrows));
        }
        if (_viewerManager) {
            _viewerManager->forEachBaseViewer([savedMaxArrows](VolumeViewerBase* viewer) {
                if (viewer) {
                    viewer->setNormalMaxArrows(savedMaxArrows);
                }
            });
        }

        connect(sliderMaxArrows, &QSlider::valueChanged, this, [this](int value) {
            using namespace vc3d::settings;
            QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
            s.setValue(viewer::NORMAL_MAX_ARROWS, value);
            if (_uiRefs.normalMaxArrowsValueLabel) {
                _uiRefs.normalMaxArrowsValueLabel->setText(QString::number(value));
            }
            if (_viewerManager) {
                _viewerManager->forEachBaseViewer([value](VolumeViewerBase* viewer) {
                    if (viewer) {
                        viewer->setNormalMaxArrows(value);
                    }
                });
            }
        });
    }
}

void ViewerControlsPanel::setupWindowRangeControls()
{
    if (auto* volumeContainer = _uiRefs.volumeWindowContainer) {
        auto* layout = new QHBoxLayout(volumeContainer);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(6);

        _volumeWindowWidget = new WindowRangeWidget(volumeContainer);
        _volumeWindowWidget->setRange(0, 255);
        _volumeWindowWidget->setMinimumSeparation(1);
        _volumeWindowWidget->setControlsEnabled(false);
        layout->addWidget(_volumeWindowWidget);

        connect(_volumeWindowWidget, &WindowRangeWidget::windowValuesChanged,
                this, [this](int low, int high) {
                    if (_viewerManager) {
                        _viewerManager->setVolumeWindow(static_cast<float>(low),
                                                        static_cast<float>(high));
                    }
                });

        if (_viewerManager) {
            connect(_viewerManager, &ViewerManager::volumeWindowChanged,
                    this, [this](float low, float high) {
                        if (!_volumeWindowWidget) {
                            return;
                        }
                        _volumeWindowWidget->setWindowValues(static_cast<int>(std::lround(low)),
                                                             static_cast<int>(std::lround(high)));
                    });

            _volumeWindowWidget->setWindowValues(
                static_cast<int>(std::lround(_viewerManager->volumeWindowLow())),
                static_cast<int>(std::lround(_viewerManager->volumeWindowHigh())));
        }
    }

    if (auto* overlayContainer = _uiRefs.overlayWindowContainer) {
        auto* layout = new QHBoxLayout(overlayContainer);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(6);

        _overlayWindowWidget = new WindowRangeWidget(overlayContainer);
        _overlayWindowWidget->setRange(0, 255);
        _overlayWindowWidget->setMinimumSeparation(1);
        _overlayWindowWidget->setControlsEnabled(false);
        layout->addWidget(_overlayWindowWidget);

        connect(_overlayWindowWidget, &WindowRangeWidget::windowValuesChanged,
                this, [this](int low, int high) {
                    if (_viewerManager) {
                        _viewerManager->setOverlayWindow(static_cast<float>(low),
                                                         static_cast<float>(high));
                    }
                });

        if (_viewerManager) {
            connect(_viewerManager, &ViewerManager::overlayWindowChanged,
                    this, [this](float low, float high) {
                        if (!_overlayWindowWidget) {
                            return;
                        }
                        _overlayWindowWidget->setWindowValues(static_cast<int>(std::lround(low)),
                                                              static_cast<int>(std::lround(high)));
                    });
            connect(_viewerManager, &ViewerManager::overlayVolumeAvailabilityChanged,
                    this, &ViewerControlsPanel::setOverlayWindowAvailable);

            _overlayWindowWidget->setWindowValues(
                static_cast<int>(std::lround(_viewerManager->overlayWindowLow())),
                static_cast<int>(std::lround(_viewerManager->overlayWindowHigh())));
        }
    }
}

void ViewerControlsPanel::setupIntersectionControls()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    if (auto* spinIntersectionOpacity = _uiRefs.intersectionOpacitySpin) {
        const int savedOpacity = settings.value(vc3d::settings::viewer::INTERSECTION_OPACITY,
                                                spinIntersectionOpacity->value()).toInt();
        const int boundedOpacity = std::clamp(savedOpacity,
                                              spinIntersectionOpacity->minimum(),
                                              spinIntersectionOpacity->maximum());
        spinIntersectionOpacity->setValue(boundedOpacity);
        connect(spinIntersectionOpacity, QOverload<int>::of(&QSpinBox::valueChanged),
                this, [this](int value) {
                    if (!_viewerManager) {
                        return;
                    }
                    const float normalized = std::clamp(static_cast<float>(value) / 100.0f, 0.0f, 1.0f);
                    _viewerManager->setIntersectionOpacity(normalized);
                });
        if (_viewerManager) {
            _viewerManager->setIntersectionOpacity(spinIntersectionOpacity->value() / 100.0f);
        }
    }

    if (auto* spinIntersectionThickness = _uiRefs.intersectionThicknessSpin) {
        const double savedThickness = settings.value(vc3d::settings::viewer::INTERSECTION_THICKNESS,
                                                     spinIntersectionThickness->value()).toDouble();
        const double boundedThickness = std::clamp(savedThickness,
                                                   static_cast<double>(spinIntersectionThickness->minimum()),
                                                   static_cast<double>(spinIntersectionThickness->maximum()));
        spinIntersectionThickness->setValue(boundedThickness);
        connect(spinIntersectionThickness,
                QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this,
                [this](double value) {
                    if (_viewerManager) {
                        _viewerManager->setIntersectionThickness(static_cast<float>(value));
                    }
                });
        if (_viewerManager) {
            _viewerManager->setIntersectionThickness(static_cast<float>(spinIntersectionThickness->value()));
        }
    }

    if (_viewerManager) {
        _viewerManager->setSurfacePatchSamplingStride(1, false);
    }
}

void ViewerControlsPanel::updateNormalVisualizationControlsEnabled(bool enabled)
{
    QWidget* controls[] = {
        _uiRefs.normalArrowLengthLabel,
        _uiRefs.normalArrowLengthSlider,
        _uiRefs.normalArrowLengthValueLabel,
        _uiRefs.normalMaxArrowsLabel,
        _uiRefs.normalMaxArrowsSlider,
        _uiRefs.normalMaxArrowsValueLabel,
    };
    for (auto* widget : controls) {
        if (widget) {
            widget->setEnabled(enabled);
        }
    }
}

void ViewerControlsPanel::updateOverlayWindowControlsEnabled()
{
    if (_overlayWindowWidget) {
        _overlayWindowWidget->setControlsEnabled(_viewControlsEnabled && _overlayWindowAvailable);
    }
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
