#include "viewer_controls/panels/ViewerCompositePanel.hpp"

#include "ViewerManager.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QLabel>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <string>

namespace
{

std::string compositeMethodForModeIndex(int index)
{
    switch (index) {
        case 0:  return "max";
        case 1:  return "mean";
        case 2:  return "min";
        case 3:  return "alpha";
        case 4:  return "volumetric";
        default: return "mean";
    }
}

int compositeModeIndexForMethod(const std::string& method)
{
    if (method == "max") return 0;
    if (method == "mean") return 1;
    if (method == "min") return 2;
    if (method == "alpha") return 3;
    if (method == "volumetric") return 4;
    return 1;
}

bool isPlaneViewer(const std::string& name)
{
    return name == "seg xz" || name == "seg yz" || name == "xy plane";
}

void reparentItemWidgets(QLayoutItem* item, QWidget* newParent)
{
    if (!item || !newParent) {
        return;
    }
    if (auto* widget = item->widget()) {
        widget->setParent(newParent);
        return;
    }
    if (auto* layout = item->layout()) {
        for (int i = 0; i < layout->count(); ++i) {
            reparentItemWidgets(layout->itemAt(i), newParent);
        }
    }
}

void moveLayoutItems(QLayout* from, QLayout* to, QWidget* newParent)
{
    if (!from || !to) {
        return;
    }
    to->setContentsMargins(from->contentsMargins());
    to->setSpacing(from->spacing());
    while (auto* item = from->takeAt(0)) {
        reparentItemWidgets(item, newParent);
        if (auto* layout = item->layout()) {
            layout->setParent(to);
        }
        to->addItem(item);
    }
}

void setWidgetVisible(QWidget* widget, bool visible)
{
    if (widget) {
        widget->setVisible(visible);
    }
}

} // namespace

ViewerCompositePanel::ViewerCompositePanel(const UiRefs& uiRefs,
                                           ViewerManager* viewerManager,
                                           QWidget* parent)
    : QWidget(parent)
    , _uiRefs(uiRefs)
{
    if (_uiRefs.scrollArea && _uiRefs.scrollArea->widget() == _uiRefs.contents) {
        _uiRefs.scrollArea->takeWidget();
    }

    auto* layout = new QVBoxLayout(this);
    moveLayoutItems(_uiRefs.contents ? _uiRefs.contents->layout() : nullptr, layout, this);

    if (_uiRefs.compositeMode) {
        QSignalBlocker blocker(_uiRefs.compositeMode);
        _uiRefs.compositeMode->clear();
        _uiRefs.compositeMode->addItem(tr("Maximum"));
        _uiRefs.compositeMode->addItem(tr("Mean"));
        _uiRefs.compositeMode->addItem(tr("Minimum"));
        _uiRefs.compositeMode->addItem(tr("Alpha"));
        _uiRefs.compositeMode->addItem(tr("Volumetric"));
        _uiRefs.compositeMode->setCurrentIndex(compositeModeIndexForMethod("max"));
    }

    setupVolumetricControls(layout);
    setupControls();
    setViewerManagers({viewerManager});
}

void ViewerCompositePanel::setViewerManagers(
    const std::vector<ViewerManager*>& viewerManagers)
{
    std::vector<ViewerManager*> unique;
    for (auto* manager : viewerManagers) {
        if (manager && std::find(unique.begin(), unique.end(), manager) == unique.end())
            unique.push_back(manager);
    }
    if (_viewerManagers == unique) {
        syncUiFromManager();
        return;
    }

    for (const auto& connection : _managerConnections)
        disconnect(connection);
    _managerConnections.clear();
    _viewerManagers = std::move(unique);
    for (auto* manager : _viewerManagers) {
        _managerConnections.push_back(connect(
            manager, &ViewerManager::baseViewerCreated,
            this, &ViewerCompositePanel::applyInitialSettingsToViewer));
    }

    // The first manager supplies the canonical complete settings. Copy them by
    // viewer role so fields not represented by today's controls remain intact.
    if (!_viewerManagers.empty()) {
        for (auto* source : _viewerManagers.front()->baseViewers()) {
            if (!source) continue;
            const auto settings = source->compositeRenderSettings();
            for (std::size_t i = 1; i < _viewerManagers.size(); ++i) {
                _viewerManagers[i]->forEachBaseViewer(
                    [source, &settings](VolumeViewerBase* target) {
                        if (target && target->surfName() == source->surfName())
                            target->setCompositeRenderSettings(settings);
                    });
            }
        }
    }
    syncUiFromManager();
}

void ViewerCompositePanel::toggleSegmentationComposite()
{
    const bool enabled = !(_uiRefs.compositeEnabled &&
                           _uiRefs.compositeEnabled->isChecked());
    applyToSegmentationViewer([enabled](VolumeViewerBase* viewer) {
        auto s = viewer->compositeRenderSettings();
        s.enabled = enabled;
        viewer->setCompositeRenderSettings(s);
    });
    setSegmentationCompositeChecked(enabled);
}

void ViewerCompositePanel::setSegmentationCompositeChecked(bool checked)
{
    if (!_uiRefs.compositeEnabled) {
        return;
    }
    QSignalBlocker blocker(_uiRefs.compositeEnabled);
    _uiRefs.compositeEnabled->setChecked(checked);
}

void ViewerCompositePanel::setupVolumetricControls(QVBoxLayout* layout)
{
    // The volumetric mode is available in the plane (slice) views too. The
    // transfer-function params are shared (they go to every viewer, like the
    // method combo). The camera (azimuth/tilt/perspective) is per-view and
    // edited only via each viewer's on-view gizmo.
    _volumetricGroup = new QWidget(this);
    auto* form = new QFormLayout(_volumetricGroup);
    form->setContentsMargins(0, 2, 0, 2);
    form->setHorizontalSpacing(4);
    form->setVerticalSpacing(2);

    _volumetricGamma = new QDoubleSpinBox(_volumetricGroup);
    _volumetricGamma->setRange(0.1, 5.0);
    _volumetricGamma->setSingleStep(0.1);
    _volumetricGamma->setValue(1.5);
    _volumetricGamma->setToolTip(tr("Opacity transfer function gamma (alpha = opacity · ρ^γ)"));
    form->addRow(tr("Gamma"), _volumetricGamma);

    // Right after the shared composite rows (mode row, params grid).
    layout->insertWidget(2, _volumetricGroup);
    _volumetricGroup->setVisible(false);

    _volumetricFlattenedGroup = new QWidget(this);
    auto* flattenedForm = new QFormLayout(_volumetricFlattenedGroup);
    flattenedForm->setContentsMargins(0, 2, 0, 2);
    flattenedForm->setHorizontalSpacing(4);
    flattenedForm->setVerticalSpacing(2);

    _volumetricWScale = new QDoubleSpinBox(_volumetricFlattenedGroup);
    _volumetricWScale->setRange(0.1, 20.0);
    _volumetricWScale->setSingleStep(0.5);
    _volumetricWScale->setValue(2.5);
    _volumetricWScale->setSuffix(QStringLiteral("×"));
    _volumetricWScale->setToolTip(
        tr("Relief exaggeration: stretches the slab along the surface normal "
           "before the tilted render, making height variation visible on wide "
           "flat segments. Flattened view only — slice views render their "
           "slab unstretched"));
    flattenedForm->addRow(tr("W scale"), _volumetricWScale);

    // At the end of the flattened-view section (header, checkbox, layers
    // grid), just before the plane-view section.
    layout->insertWidget(7, _volumetricFlattenedGroup);
    _volumetricFlattenedGroup->setVisible(false);

    connect(_volumetricGamma, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this](double value) {
        applyToAllViewers([value](VolumeViewerBase* viewer) {
            auto s = viewer->compositeRenderSettings();
            s.params.tfGamma = float(value);
            viewer->setCompositeRenderSettings(s);
        });
    });
    // W scale is flattened-view-only (the slice slab has no relief to
    // exaggerate); the render path ignores it for plane views regardless.
    connect(_volumetricWScale, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this](double value) {
        applyToSegmentationViewer([value](VolumeViewerBase* viewer) {
            auto s = viewer->compositeRenderSettings();
            s.params.wScale = float(value);
            viewer->setCompositeRenderSettings(s);
        });
    });
}

void ViewerCompositePanel::setupControls()
{
    if (_uiRefs.compositeEnabled) {
        connect(_uiRefs.compositeEnabled, &QCheckBox::toggled, this, [this](bool checked) {
            applyToSegmentationViewer([checked](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.enabled = checked;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }

    if (_uiRefs.compositeMode) {
        connect(_uiRefs.compositeMode, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
            const std::string method = compositeMethodForModeIndex(index);
            applyToAllViewers([&method](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.method = method;
                viewer->setCompositeRenderSettings(s);
            });
            updateCompositeParamsVisibility();
        });
    }

    if (_uiRefs.layersInFront) {
        connect(_uiRefs.layersInFront, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            applyToSegmentationViewer([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.layersFront = value;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.layersBehind) {
        connect(_uiRefs.layersBehind, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            applyToSegmentationViewer([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.layersBehind = value;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    // The alpha/opacity params and stack direction feed the plane-view
    // composites too (scalar alpha and volumetric TF), so they go to every
    // viewer. The layer counts stay per-scope (layersFront/Behind vs
    // planeLayersFront/Behind).
    if (_uiRefs.alphaMin) {
        connect(_uiRefs.alphaMin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            applyToAllViewers([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.alphaMin = value / 255.0f;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.alphaMax) {
        connect(_uiRefs.alphaMax, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            applyToAllViewers([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.alphaMax = value / 255.0f;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.alphaThreshold) {
        connect(_uiRefs.alphaThreshold, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            applyToAllViewers([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.alphaCutoff = value / 10000.0f;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.material) {
        connect(_uiRefs.material, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            applyToAllViewers([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.alphaOpacity = value / 255.0f;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.reverseDirection) {
        connect(_uiRefs.reverseDirection, &QCheckBox::toggled, this, [this](bool checked) {
            applyToAllViewers([checked](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.reverseDirection = checked;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }

    if (_uiRefs.planeCompositeXY) {
        connect(_uiRefs.planeCompositeXY, &QCheckBox::toggled, this, [this](bool checked) {
            applyToAllViewers([checked](VolumeViewerBase* viewer) {
                if (viewer->surfName() == "xy plane") {
                    auto s = viewer->compositeRenderSettings();
                    s.planeEnabled = checked;
                    viewer->setCompositeRenderSettings(s);
                }
            });
        });
    }
    if (_uiRefs.planeCompositeXZ) {
        connect(_uiRefs.planeCompositeXZ, &QCheckBox::toggled, this, [this](bool checked) {
            applyToAllViewers([checked](VolumeViewerBase* viewer) {
                if (viewer->surfName() == "seg xz") {
                    auto s = viewer->compositeRenderSettings();
                    s.planeEnabled = checked;
                    viewer->setCompositeRenderSettings(s);
                }
            });
        });
    }
    if (_uiRefs.planeCompositeYZ) {
        connect(_uiRefs.planeCompositeYZ, &QCheckBox::toggled, this, [this](bool checked) {
            applyToAllViewers([checked](VolumeViewerBase* viewer) {
                if (viewer->surfName() == "seg yz") {
                    auto s = viewer->compositeRenderSettings();
                    s.planeEnabled = checked;
                    viewer->setCompositeRenderSettings(s);
                }
            });
        });
    }
    if (_uiRefs.planeLayersFront) {
        connect(_uiRefs.planeLayersFront, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            const int behind = _uiRefs.planeLayersBehind ? _uiRefs.planeLayersBehind->value() : 0;
            applyToPlaneViewers([value, behind](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.planeLayersFront = std::max(0, value);
                s.planeLayersBehind = std::max(0, behind);
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.planeLayersBehind) {
        connect(_uiRefs.planeLayersBehind, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            const int front = _uiRefs.planeLayersFront ? _uiRefs.planeLayersFront->value() : 0;
            applyToPlaneViewers([front, value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.planeLayersFront = std::max(0, front);
                s.planeLayersBehind = std::max(0, value);
                viewer->setCompositeRenderSettings(s);
            });
        });
    }

    updateCompositeParamsVisibility();
}

void ViewerCompositePanel::applyInitialSettingsToViewer(VolumeViewerBase* viewer)
{
    if (!viewer) {
        return;
    }
    // Push the full control state, not just the method: the .ui defaults
    // (e.g. alpha min 170) differ from the CompositeParams defaults, and a
    // control that is never touched would otherwise display a value the
    // viewer isn't using.
    auto s = viewer->compositeRenderSettings();
    // A sibling viewer of the same kind (e.g. another workspace's flattened
    // view) already carries the live state, including its camera; copy it
    // wholesale rather than re-deriving everything from the controls.
    bool foundCanonical = false;
    for (auto* manager : _viewerManagers) {
        for (auto* existing : manager->baseViewers()) {
            if (existing && existing != viewer && existing->surfName() == viewer->surfName()) {
                s = existing->compositeRenderSettings();
                foundCanonical = true;
                break;
            }
        }
        if (foundCanonical) break;
    }
    if (!foundCanonical) {
        if (_uiRefs.layersInFront) {
            s.layersFront = _uiRefs.layersInFront->value();
        }
        if (_uiRefs.layersBehind) {
            s.layersBehind = _uiRefs.layersBehind->value();
        }
        if (_uiRefs.alphaMin) {
            s.params.alphaMin = _uiRefs.alphaMin->value() / 255.0f;
        }
        if (_uiRefs.alphaMax) {
            s.params.alphaMax = _uiRefs.alphaMax->value() / 255.0f;
        }
        if (_uiRefs.alphaThreshold) {
            s.params.alphaCutoff = _uiRefs.alphaThreshold->value() / 10000.0f;
        }
        if (_uiRefs.material) {
            s.params.alphaOpacity = _uiRefs.material->value() / 255.0f;
        }
        if (_uiRefs.reverseDirection) {
            s.reverseDirection = _uiRefs.reverseDirection->isChecked();
        }
        if (_uiRefs.planeLayersFront) {
            s.planeLayersFront = std::max(0, _uiRefs.planeLayersFront->value());
        }
        if (_uiRefs.planeLayersBehind) {
            s.planeLayersBehind = std::max(0, _uiRefs.planeLayersBehind->value());
        }
        if (_volumetricGamma) {
            s.params.tfGamma = float(_volumetricGamma->value());
        }
        if (_volumetricWScale) {
            s.params.wScale = float(_volumetricWScale->value());
        }
        // The camera is per-view and stays at the viewer's own state
        // (default straight-down), edited only via its on-view gizmo.
    }
    s.params.method = compositeMethodForModeIndex(
        _uiRefs.compositeMode ? _uiRefs.compositeMode->currentIndex() : 0);
    viewer->setCompositeRenderSettings(s);
    if (viewer->surfName() == "segmentation") {
        setSegmentationCompositeChecked(s.enabled);
    }
}

void ViewerCompositePanel::syncUiFromManager()
{
    if (_viewerManagers.empty()) {
        return;
    }

    VolumeViewerBase* segmentationViewer = nullptr;
    VolumeViewerBase* firstPlaneViewer = nullptr;
    for (auto* viewer : _viewerManagers.front()->baseViewers()) {
        if (!viewer) {
            continue;
        }
        if (viewer->surfName() == "segmentation") {
            segmentationViewer = viewer;
        } else if (!firstPlaneViewer && isPlaneViewer(viewer->surfName())) {
            firstPlaneViewer = viewer;
        }

        QCheckBox* planeCheck = nullptr;
        if (viewer->surfName() == "xy plane") planeCheck = _uiRefs.planeCompositeXY;
        else if (viewer->surfName() == "seg xz") planeCheck = _uiRefs.planeCompositeXZ;
        else if (viewer->surfName() == "seg yz") planeCheck = _uiRefs.planeCompositeYZ;
        if (planeCheck) {
            const QSignalBlocker blocker(planeCheck);
            planeCheck->setChecked(viewer->compositeRenderSettings().planeEnabled);
        }
    }

    if (segmentationViewer) {
        const auto& settings = segmentationViewer->compositeRenderSettings();
        if (_uiRefs.compositeEnabled) {
            const QSignalBlocker blocker(_uiRefs.compositeEnabled);
            _uiRefs.compositeEnabled->setChecked(settings.enabled);
        }
        if (_uiRefs.compositeMode) {
            const QSignalBlocker blocker(_uiRefs.compositeMode);
            _uiRefs.compositeMode->setCurrentIndex(compositeModeIndexForMethod(settings.params.method));
        }
        if (_uiRefs.layersInFront) {
            const QSignalBlocker blocker(_uiRefs.layersInFront);
            _uiRefs.layersInFront->setValue(settings.layersFront);
        }
        if (_uiRefs.layersBehind) {
            const QSignalBlocker blocker(_uiRefs.layersBehind);
            _uiRefs.layersBehind->setValue(settings.layersBehind);
        }
        if (_uiRefs.alphaMin) {
            const QSignalBlocker blocker(_uiRefs.alphaMin);
            _uiRefs.alphaMin->setValue(static_cast<int>(std::lround(settings.params.alphaMin * 255.0f)));
        }
        if (_uiRefs.alphaMax) {
            const QSignalBlocker blocker(_uiRefs.alphaMax);
            _uiRefs.alphaMax->setValue(static_cast<int>(std::lround(settings.params.alphaMax * 255.0f)));
        }
        if (_uiRefs.alphaThreshold) {
            const QSignalBlocker blocker(_uiRefs.alphaThreshold);
            _uiRefs.alphaThreshold->setValue(static_cast<int>(std::lround(settings.params.alphaCutoff * 10000.0f)));
        }
        if (_uiRefs.material) {
            const QSignalBlocker blocker(_uiRefs.material);
            _uiRefs.material->setValue(static_cast<int>(std::lround(settings.params.alphaOpacity * 255.0f)));
        }
        if (_uiRefs.reverseDirection) {
            const QSignalBlocker blocker(_uiRefs.reverseDirection);
            _uiRefs.reverseDirection->setChecked(settings.reverseDirection);
        }
    }

    if (firstPlaneViewer) {
        const auto& settings = firstPlaneViewer->compositeRenderSettings();
        if (_uiRefs.planeLayersFront) {
            const QSignalBlocker blocker(_uiRefs.planeLayersFront);
            _uiRefs.planeLayersFront->setValue(settings.planeLayersFront);
        }
        if (_uiRefs.planeLayersBehind) {
            const QSignalBlocker blocker(_uiRefs.planeLayersBehind);
            _uiRefs.planeLayersBehind->setValue(settings.planeLayersBehind);
        }
    }
    updateCompositeParamsVisibility();
}

void ViewerCompositePanel::updateCompositeParamsVisibility()
{
    const int methodIndex = _uiRefs.compositeMode ? _uiRefs.compositeMode->currentIndex() : 0;
    const bool isAlpha = methodIndex == 3;
    const bool isVolumetric = methodIndex == 4;

    // The volumetric opacity TF reuses the alpha window and opacity rows;
    // the cutoff threshold is alpha-only.
    setWidgetVisible(_uiRefs.alphaMinLabel, isAlpha || isVolumetric);
    setWidgetVisible(_uiRefs.alphaMin, isAlpha || isVolumetric);
    setWidgetVisible(_uiRefs.alphaMaxLabel, isAlpha || isVolumetric);
    setWidgetVisible(_uiRefs.alphaMax, isAlpha || isVolumetric);
    setWidgetVisible(_uiRefs.alphaThresholdLabel, isAlpha);
    setWidgetVisible(_uiRefs.alphaThreshold, isAlpha);
    setWidgetVisible(_uiRefs.materialLabel, isAlpha || isVolumetric);
    setWidgetVisible(_uiRefs.material, isAlpha || isVolumetric);
    setWidgetVisible(_volumetricGroup, isVolumetric);
    setWidgetVisible(_volumetricFlattenedGroup, isVolumetric);
}

void ViewerCompositePanel::applyToSegmentationViewer(const std::function<void(VolumeViewerBase*)>& apply)
{
    if (_viewerManagers.empty() || !apply) {
        return;
    }
    for (auto* manager : _viewerManagers)
        for (auto* viewer : manager->baseViewers())
            if (viewer && viewer->surfName() == "segmentation") apply(viewer);
}

void ViewerCompositePanel::applyToAllViewers(const std::function<void(VolumeViewerBase*)>& apply)
{
    if (_viewerManagers.empty() || !apply) {
        return;
    }
    for (auto* manager : _viewerManagers)
        manager->forEachBaseViewer([&apply](VolumeViewerBase* viewer) {
            if (viewer) apply(viewer);
        });
}

void ViewerCompositePanel::applyToPlaneViewers(const std::function<void(VolumeViewerBase*)>& apply)
{
    applyToAllViewers([&apply](VolumeViewerBase* viewer) {
        if (isPlaneViewer(viewer->surfName())) {
            apply(viewer);
        }
    });
}
