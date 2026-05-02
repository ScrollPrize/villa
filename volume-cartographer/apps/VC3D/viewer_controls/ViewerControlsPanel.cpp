#include "viewer_controls/ViewerControlsPanel.hpp"

#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "WindowRangeWidget.hpp"
#include "elements/CollapsibleSettingsGroup.hpp"
#include "viewer_controls/panels/ViewerNavigationPanel.hpp"
#include "viewer_controls/panels/ViewerNormalVisualizationPanel.hpp"
#include "viewer_controls/panels/ViewerTransformsPanel.hpp"
#include "viewer_controls/panels/ViewerViewExtrasPanel.hpp"

#include <QDoubleSpinBox>
#include <QHBoxLayout>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>

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

void ViewerControlsPanel::addViewerGroups()
{
    using namespace vc3d::settings;

    auto* viewGroup = addViewerGroup(tr("View"),
                                     detachScrollContents(_uiRefs.viewScrollArea, _uiRefs.viewContents),
                                     viewer::GROUP_VIEW_EXPANDED,
                                     viewer::GROUP_VIEW_EXPANDED_DEFAULT);
    if (viewGroup) {
        viewGroup->contentLayout()->addWidget(new ViewerViewExtrasPanel(_viewerManager, viewGroup));
    }

    addViewerGroup(tr("Navigation"),
                   new ViewerNavigationPanel(_viewerManager, _uiRefs.contents),
                   "viewer_controls/group_navigation_expanded",
                   true);

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

    ViewerNormalVisualizationPanel::UiRefs normalUi{
        .contents = _uiRefs.normalVisualizationContents,
        .showSurfaceNormals = _uiRefs.showSurfaceNormals,
        .normalArrowLengthLabel = _uiRefs.normalArrowLengthLabel,
        .normalArrowLengthSlider = _uiRefs.normalArrowLengthSlider,
        .normalArrowLengthValueLabel = _uiRefs.normalArrowLengthValueLabel,
        .normalMaxArrowsLabel = _uiRefs.normalMaxArrowsLabel,
        .normalMaxArrowsSlider = _uiRefs.normalMaxArrowsSlider,
        .normalMaxArrowsValueLabel = _uiRefs.normalMaxArrowsValueLabel,
    };
    auto* normalPanel = new ViewerNormalVisualizationPanel(normalUi, _viewerManager, _uiRefs.contents);
    connect(normalPanel, &ViewerNormalVisualizationPanel::statusMessageRequested,
            this, &ViewerControlsPanel::statusMessageRequested);
    addViewerGroup(tr("Normal Visualization"),
                   normalPanel,
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

    _transformsPanel = new ViewerTransformsPanel(_uiRefs.contents);
    addViewerGroup(tr("Transforms"),
                   _transformsPanel,
                   viewer::GROUP_TRANSFORMS_EXPANDED,
                   viewer::GROUP_TRANSFORMS_EXPANDED_DEFAULT);

    if (auto* layout = qobject_cast<QVBoxLayout*>(_uiRefs.contents->layout())) {
        layout->addStretch(1);
    }
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

void ViewerControlsPanel::setSliceStepSize(int value)
{
    if (!_sliceStepSizeSpin) {
        return;
    }
    QSignalBlocker blocker(_sliceStepSizeSpin);
    _sliceStepSizeSpin->setValue(std::clamp(value,
                                            _sliceStepSizeSpin->minimum(),
                                            _sliceStepSizeSpin->maximum()));
}

void ViewerControlsPanel::setupViewerControlWiring()
{
    setupWindowRangeControls();
    setupIntersectionControls();

    if (_uiRefs.zoomInButton) {
        connect(_uiRefs.zoomInButton, &QPushButton::clicked, this, &ViewerControlsPanel::zoomInRequested);
    }
    if (_uiRefs.zoomOutButton) {
        connect(_uiRefs.zoomOutButton, &QPushButton::clicked, this, &ViewerControlsPanel::zoomOutRequested);
    }

    if (auto* spinSliceStep = _uiRefs.sliceStepSizeSpin) {
        _sliceStepSizeSpin = spinSliceStep;
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
