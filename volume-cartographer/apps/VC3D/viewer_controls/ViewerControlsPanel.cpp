#include "viewer_controls/ViewerControlsPanel.hpp"

#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "WindowRangeWidget.hpp"
#include "elements/CollapsibleSettingsGroup.hpp"
#include "viewer_controls/panels/ViewerNavigationPanel.hpp"
#include "viewer_controls/panels/ViewerNormalVisualizationPanel.hpp"
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
        _viewExtrasPanel = new ViewerViewExtrasPanel(_viewerManager, viewGroup);
        viewGroup->contentLayout()->addWidget(_viewExtrasPanel);
    }

    _navigationPanel = new ViewerNavigationPanel(_viewerManager, _uiRefs.contents);
    addViewerGroup(tr("Navigation"),
                   _navigationPanel,
                   "viewer_controls/group_navigation_expanded",
                   true);

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
    _normalPanel = new ViewerNormalVisualizationPanel(normalUi, _viewerManager, _uiRefs.contents);
    connect(_normalPanel, &ViewerNormalVisualizationPanel::statusMessageRequested,
            this, &ViewerControlsPanel::statusMessageRequested);
    addViewerGroup(tr("Normal Visualization"),
                   _normalPanel,
                   viewer::GROUP_NORMAL_VIS_EXPANDED,
                   viewer::GROUP_NORMAL_VIS_EXPANDED_DEFAULT);

    if (auto* layout = qobject_cast<QVBoxLayout*>(_uiRefs.contents->layout())) {
        layout->addStretch(1);
    }
}

void ViewerControlsPanel::setViewerManager(ViewerManager* viewerManager)
{
    if (_viewerManager == viewerManager) {
        return;
    }
    if (_viewerManager) {
        disconnect(_viewerManager, nullptr, this, nullptr);
    }
    _viewerManager = viewerManager;
    if (_viewExtrasPanel) _viewExtrasPanel->setViewerManager(_viewerManager);
    if (_navigationPanel) _navigationPanel->setViewerManager(_viewerManager);
    if (_normalPanel) _normalPanel->setViewerManager(_viewerManager);
    connectViewerManagerSignals();

    if (!_viewerManager) {
        setOverlayWindowAvailable(false);
        return;
    }
    if (_volumeWindowWidget) {
        const QSignalBlocker blocker(_volumeWindowWidget);
        _volumeWindowWidget->setWindowValues(
            static_cast<int>(std::lround(_viewerManager->volumeWindowLow())),
            static_cast<int>(std::lround(_viewerManager->volumeWindowHigh())));
    }
    if (_overlayWindowWidget) {
        const QSignalBlocker blocker(_overlayWindowWidget);
        _overlayWindowWidget->setWindowValues(
            static_cast<int>(std::lround(_viewerManager->overlayWindowLow())),
            static_cast<int>(std::lround(_viewerManager->overlayWindowHigh())));
    }
    if (_uiRefs.intersectionThicknessSpin) {
        const QSignalBlocker blocker(_uiRefs.intersectionThicknessSpin);
        _uiRefs.intersectionThicknessSpin->setValue(_viewerManager->intersectionThickness());
    }
    setOverlayWindowAvailable(static_cast<bool>(_viewerManager->overlayVolume()));
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
    setupWindowRangeControls();
    setupIntersectionControls();

    if (_uiRefs.zoomInButton) {
        connect(_uiRefs.zoomInButton, &QPushButton::clicked, this, &ViewerControlsPanel::zoomInRequested);
    }
    if (_uiRefs.zoomOutButton) {
        connect(_uiRefs.zoomOutButton, &QPushButton::clicked, this, &ViewerControlsPanel::zoomOutRequested);
    }

}

void ViewerControlsPanel::connectViewerManagerSignals()
{
    if (!_viewerManager) {
        return;
    }
    connect(_viewerManager, &ViewerManager::volumeWindowChanged,
            this, [this](float low, float high) {
                if (_volumeWindowWidget) {
                    _volumeWindowWidget->setWindowValues(static_cast<int>(std::lround(low)),
                                                         static_cast<int>(std::lround(high)));
                }
            });
    connect(_viewerManager, &ViewerManager::overlayWindowChanged,
            this, [this](float low, float high) {
                if (_overlayWindowWidget) {
                    _overlayWindowWidget->setWindowValues(static_cast<int>(std::lround(low)),
                                                          static_cast<int>(std::lround(high)));
                }
            });
    connect(_viewerManager, &ViewerManager::overlayVolumeAvailabilityChanged,
            this, &ViewerControlsPanel::setOverlayWindowAvailable);
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
            _overlayWindowWidget->setWindowValues(
                static_cast<int>(std::lround(_viewerManager->overlayWindowLow())),
                static_cast<int>(std::lround(_viewerManager->overlayWindowHigh())));
        }
    }

    connectViewerManagerSignals();
}

void ViewerControlsPanel::setupIntersectionControls()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

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
