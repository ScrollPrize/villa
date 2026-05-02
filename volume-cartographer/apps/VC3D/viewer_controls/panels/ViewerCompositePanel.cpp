#include "viewer_controls/panels/ViewerCompositePanel.hpp"

#include "ViewerManager.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QVBoxLayout>

#include <algorithm>
#include <cstdint>
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
        case 4:  return "beerLambert";
        case 5:  return "volumetric";
        case 6:  return "dvr";
        case 7:  return "firstHitIso";
        case 8:  return "devFromMean";
        case 9:  return "emissionDvr";
        case 10: return "maxAboveIso";
        case 11: return "gammaWeighted";
        case 12: return "gradientMag";
        case 13: return "pbrIso";
        case 14: return "shadedDvr";
        default: return "mean";
    }
}

int compositeModeIndexForMethod(const std::string& method)
{
    if (method == "max") return 0;
    if (method == "mean") return 1;
    if (method == "min") return 2;
    if (method == "alpha") return 3;
    if (method == "beerLambert") return 4;
    if (method == "volumetric") return 5;
    if (method == "dvr") return 6;
    if (method == "firstHitIso") return 7;
    if (method == "devFromMean") return 8;
    if (method == "emissionDvr") return 9;
    if (method == "maxAboveIso") return 10;
    if (method == "gammaWeighted") return 11;
    if (method == "gradientMag") return 12;
    if (method == "pbrIso") return 13;
    if (method == "shadedDvr") return 14;
    return 1;
}

bool isPlaneViewer(const std::string& name)
{
    return name == "seg xz" || name == "seg yz" || name == "xy plane";
}

void moveLayoutItems(QLayout* from, QLayout* to, QWidget* newParent)
{
    if (!from || !to) {
        return;
    }
    to->setContentsMargins(from->contentsMargins());
    to->setSpacing(from->spacing());
    while (auto* item = from->takeAt(0)) {
        if (newParent) {
            if (auto* widget = item->widget()) {
                widget->setParent(newParent);
            } else if (auto* layout = item->layout()) {
                layout->setParent(newParent);
            }
        }
        to->addItem(item);
    }
}

void setVisible(QWidget* widget, bool visible)
{
    if (widget) {
        widget->setVisible(visible);
    }
}

void setEnabled(QWidget* widget, bool enabled)
{
    if (widget) {
        widget->setEnabled(enabled);
    }
}

} // namespace

ViewerCompositePanel::ViewerCompositePanel(const UiRefs& uiRefs,
                                           ViewerManager* viewerManager,
                                           QWidget* parent)
    : QWidget(parent)
    , _uiRefs(uiRefs)
    , _viewerManager(viewerManager)
{
    if (_uiRefs.scrollArea && _uiRefs.scrollArea->widget() == _uiRefs.contents) {
        _uiRefs.scrollArea->takeWidget();
    }

    auto* layout = new QVBoxLayout(this);
    moveLayoutItems(_uiRefs.contents ? _uiRefs.contents->layout() : nullptr, layout, this);

    if (_uiRefs.compositeMode) {
        QSignalBlocker blocker(_uiRefs.compositeMode);
        _uiRefs.compositeMode->setCurrentIndex(compositeModeIndexForMethod("max"));
    }

    setupControls();
    initializeExistingViewers();

    if (_viewerManager) {
        connect(_viewerManager, &ViewerManager::baseViewerCreated,
                this, &ViewerCompositePanel::applyInitialSettingsToViewer);
    }
}

void ViewerCompositePanel::toggleSegmentationComposite()
{
    applyToSegmentationViewer([this](VolumeViewerBase* viewer) {
        auto s = viewer->compositeRenderSettings();
        s.enabled = !s.enabled;
        viewer->setCompositeRenderSettings(s);
        setSegmentationCompositeChecked(s.enabled);
    });
}

void ViewerCompositePanel::setSegmentationCompositeChecked(bool checked)
{
    if (!_uiRefs.compositeEnabled) {
        return;
    }
    QSignalBlocker blocker(_uiRefs.compositeEnabled);
    _uiRefs.compositeEnabled->setChecked(checked);
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
    if (_uiRefs.alphaMin) {
        connect(_uiRefs.alphaMin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            applyToSegmentationViewer([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.alphaMin = value / 255.0f;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.alphaMax) {
        connect(_uiRefs.alphaMax, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            applyToSegmentationViewer([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.alphaMax = value / 255.0f;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.alphaThreshold) {
        connect(_uiRefs.alphaThreshold, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            applyToSegmentationViewer([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.alphaCutoff = value / 10000.0f;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.material) {
        connect(_uiRefs.material, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            applyToSegmentationViewer([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.alphaOpacity = value / 255.0f;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.reverseDirection) {
        connect(_uiRefs.reverseDirection, &QCheckBox::toggled, this, [this](bool checked) {
            applyToSegmentationViewer([checked](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.reverseDirection = checked;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }

    if (_uiRefs.blExtinction) {
        connect(_uiRefs.blExtinction, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
            applyToSegmentationViewer([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.blExtinction = static_cast<float>(value);
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.blEmission) {
        connect(_uiRefs.blEmission, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
            applyToSegmentationViewer([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.blEmission = static_cast<float>(value);
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.blAmbient) {
        connect(_uiRefs.blAmbient, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
            applyToSegmentationViewer([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.blAmbient = static_cast<float>(value);
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.lightingEnabled) {
        connect(_uiRefs.lightingEnabled, &QCheckBox::toggled, this, [this](bool checked) {
            applyToSegmentationViewer([checked](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.lightingEnabled = checked;
                viewer->setCompositeRenderSettings(s);
            });
            updateCompositeParamsVisibility();
        });
    }
    if (_uiRefs.lightAzimuth) {
        connect(_uiRefs.lightAzimuth, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            applyToSegmentationViewer([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.lightAzimuth = static_cast<float>(value);
                s.params.updateLightDir();
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.lightElevation) {
        connect(_uiRefs.lightElevation, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            applyToSegmentationViewer([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.lightElevation = static_cast<float>(value);
                s.params.updateLightDir();
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.lightDiffuse) {
        connect(_uiRefs.lightDiffuse, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
            applyToSegmentationViewer([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.lightDiffuse = static_cast<float>(value);
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.lightAmbient) {
        connect(_uiRefs.lightAmbient, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
            applyToSegmentationViewer([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.lightAmbient = static_cast<float>(value);
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.useVolumeGradients) {
        connect(_uiRefs.useVolumeGradients, &QCheckBox::toggled, this, [this](bool checked) {
            applyToSegmentationViewer([checked](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.useVolumeGradients = checked;
                s.params.lightNormalSource = checked ? 1 : 0;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.shadowSteps) {
        connect(_uiRefs.shadowSteps, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            applyToSegmentationViewer([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.shadowSteps = std::clamp(value, 1, 64);
                viewer->setCompositeRenderSettings(s);
            });
        });
    }

    if (_uiRefs.preNormalizeLayers) {
        connect(_uiRefs.preNormalizeLayers, &QCheckBox::toggled, this, [this](bool checked) {
            applyToSegmentationViewer([checked](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.preNormalizeLayers = checked;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.preHistEqLayers) {
        connect(_uiRefs.preHistEqLayers, &QCheckBox::toggled, this, [this](bool checked) {
            applyToSegmentationViewer([checked](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.preHistEqLayers = checked;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }

    auto applyParam = [this](auto&& mutate) {
        applyToSegmentationViewer([&mutate](VolumeViewerBase* viewer) {
            auto s = viewer->compositeRenderSettings();
            mutate(s.params);
            viewer->setCompositeRenderSettings(s);
        });
    };
    if (_uiRefs.preTfEnabled) {
        connect(_uiRefs.preTfEnabled, &QCheckBox::toggled, this, [this, applyParam](bool v) {
            applyParam([v](CompositeParams& p) { p.preTfEnabled = v; });
            updateCompositeParamsVisibility();
        });
    }
    if (_uiRefs.preTfX1) {
        connect(_uiRefs.preTfX1, QOverload<int>::of(&QSpinBox::valueChanged), this,
                [applyParam](int v) { applyParam([v](CompositeParams& p) { p.preTfX1 = uint8_t(v); }); });
    }
    if (_uiRefs.preTfY1) {
        connect(_uiRefs.preTfY1, QOverload<int>::of(&QSpinBox::valueChanged), this,
                [applyParam](int v) { applyParam([v](CompositeParams& p) { p.preTfY1 = uint8_t(v); }); });
    }
    if (_uiRefs.preTfX2) {
        connect(_uiRefs.preTfX2, QOverload<int>::of(&QSpinBox::valueChanged), this,
                [applyParam](int v) { applyParam([v](CompositeParams& p) { p.preTfX2 = uint8_t(v); }); });
    }
    if (_uiRefs.preTfY2) {
        connect(_uiRefs.preTfY2, QOverload<int>::of(&QSpinBox::valueChanged), this,
                [applyParam](int v) { applyParam([v](CompositeParams& p) { p.preTfY2 = uint8_t(v); }); });
    }
    if (_uiRefs.postTfEnabled) {
        connect(_uiRefs.postTfEnabled, &QCheckBox::toggled, this, [this, applyParam](bool v) {
            applyParam([v](CompositeParams& p) { p.postTfEnabled = v; });
            updateCompositeParamsVisibility();
        });
    }
    if (_uiRefs.postTfX1) {
        connect(_uiRefs.postTfX1, QOverload<int>::of(&QSpinBox::valueChanged), this,
                [applyParam](int v) { applyParam([v](CompositeParams& p) { p.postTfX1 = uint8_t(v); }); });
    }
    if (_uiRefs.postTfY1) {
        connect(_uiRefs.postTfY1, QOverload<int>::of(&QSpinBox::valueChanged), this,
                [applyParam](int v) { applyParam([v](CompositeParams& p) { p.postTfY1 = uint8_t(v); }); });
    }
    if (_uiRefs.postTfX2) {
        connect(_uiRefs.postTfX2, QOverload<int>::of(&QSpinBox::valueChanged), this,
                [applyParam](int v) { applyParam([v](CompositeParams& p) { p.postTfX2 = uint8_t(v); }); });
    }
    if (_uiRefs.postTfY2) {
        connect(_uiRefs.postTfY2, QOverload<int>::of(&QSpinBox::valueChanged), this,
                [applyParam](int v) { applyParam([v](CompositeParams& p) { p.postTfY2 = uint8_t(v); }); });
    }
    if (_uiRefs.dvrAmbient) {
        connect(_uiRefs.dvrAmbient, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
                [applyParam](double v) { applyParam([v](CompositeParams& p) { p.dvrAmbient = float(v); }); });
    }
    if (_uiRefs.pbrRoughness) {
        connect(_uiRefs.pbrRoughness, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
                [applyParam](double v) { applyParam([v](CompositeParams& p) { p.pbrRoughness = float(v); }); });
    }
    if (_uiRefs.pbrMetallic) {
        connect(_uiRefs.pbrMetallic, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
                [applyParam](double v) { applyParam([v](CompositeParams& p) { p.pbrMetallic = float(v); }); });
    }

    if (_uiRefs.methodScale) {
        connect(_uiRefs.methodScale, &QSlider::valueChanged, this, [this](int value) {
            if (_uiRefs.methodScaleValue) {
                _uiRefs.methodScaleValue->setText(QString::number(value / 10.0f, 'f', 1));
            }
        });
    }
    if (_uiRefs.methodParam) {
        connect(_uiRefs.methodParam, &QSlider::valueChanged, this, [](int) {});
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

    updateRakingControlsEnabled(_uiRefs.rakingEnabled && _uiRefs.rakingEnabled->isChecked());
    if (_uiRefs.rakingEnabled) {
        connect(_uiRefs.rakingEnabled, &QCheckBox::toggled, this, [this](bool checked) {
            updateRakingControlsEnabled(checked);
            applyToAllViewers([checked](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.postRakingEnabled = checked;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.rakingAzimuth) {
        connect(_uiRefs.rakingAzimuth, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) {
            applyToAllViewers([v](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.postRakingAzimuth = float(v);
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.rakingElevation) {
        connect(_uiRefs.rakingElevation, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) {
            applyToAllViewers([v](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.postRakingElevation = float(v);
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.rakingStrength) {
        connect(_uiRefs.rakingStrength, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) {
            applyToAllViewers([v](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.postRakingStrength = std::clamp(float(v), 0.0f, 1.0f);
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.rakingDepthScale) {
        connect(_uiRefs.rakingDepthScale, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) {
            applyToAllViewers([v](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.postRakingDepthScale = std::max(0.01f, float(v));
                viewer->setCompositeRenderSettings(s);
            });
        });
    }

    updateCompositeParamsVisibility();
}

void ViewerCompositePanel::initializeExistingViewers()
{
    if (!_viewerManager) {
        return;
    }
    for (auto* viewer : _viewerManager->baseViewers()) {
        applyInitialSettingsToViewer(viewer);
    }
}

void ViewerCompositePanel::applyInitialSettingsToViewer(VolumeViewerBase* viewer)
{
    if (!viewer) {
        return;
    }
    auto s = viewer->compositeRenderSettings();
    s.params.method = compositeMethodForModeIndex(_uiRefs.compositeMode ? _uiRefs.compositeMode->currentIndex() : 0);
    viewer->setCompositeRenderSettings(s);
    if (viewer->surfName() == "segmentation") {
        setSegmentationCompositeChecked(s.enabled);
    }
}

void ViewerCompositePanel::updateCompositeParamsVisibility()
{
    const int methodIndex = _uiRefs.compositeMode ? _uiRefs.compositeMode->currentIndex() : 0;
    const bool lightingOn = _uiRefs.lightingEnabled && _uiRefs.lightingEnabled->isChecked();
    const bool preTfOn = _uiRefs.preTfEnabled && _uiRefs.preTfEnabled->isChecked();
    const bool postTfOn = _uiRefs.postTfEnabled && _uiRefs.postTfEnabled->isChecked();

    const bool isAlpha = methodIndex == 3;
    const bool isBL = methodIndex == 4;
    const bool isVolum = methodIndex == 5;
    const bool isDvr = methodIndex == 6;
    const bool isPbr = methodIndex == 13;
    const bool isShadedDvr = methodIndex == 14;

    setVisible(_uiRefs.alphaMinLabel, isAlpha);
    setVisible(_uiRefs.alphaMin, isAlpha);
    setVisible(_uiRefs.alphaMaxLabel, isAlpha);
    setVisible(_uiRefs.alphaMax, isAlpha);
    setVisible(_uiRefs.alphaThresholdLabel, isAlpha);
    setVisible(_uiRefs.alphaThreshold, isAlpha);
    setVisible(_uiRefs.materialLabel, isAlpha);
    setVisible(_uiRefs.material, isAlpha);

    const bool showBL = isBL || isVolum;
    setVisible(_uiRefs.blExtinctionLabel, showBL);
    setVisible(_uiRefs.blExtinction, showBL);
    setVisible(_uiRefs.blEmissionLabel, showBL);
    setVisible(_uiRefs.blEmission, showBL);
    setVisible(_uiRefs.blAmbientLabel, showBL);
    setVisible(_uiRefs.blAmbient, showBL);

    setVisible(_uiRefs.shadowStepsLabel, isVolum);
    setVisible(_uiRefs.shadowSteps, isVolum);

    const bool showDvrAmbient = isDvr || isShadedDvr;
    setVisible(_uiRefs.dvrAmbientLabel, showDvrAmbient);
    setVisible(_uiRefs.dvrAmbient, showDvrAmbient);
    setVisible(_uiRefs.pbrRoughnessLabel, isPbr);
    setVisible(_uiRefs.pbrRoughness, isPbr);
    setVisible(_uiRefs.pbrMetallicLabel, isPbr);
    setVisible(_uiRefs.pbrMetallic, isPbr);

    setVisible(_uiRefs.lightingEnabled, true);
    setVisible(_uiRefs.lightAzimuthLabel, lightingOn);
    setVisible(_uiRefs.lightAzimuth, lightingOn);
    setVisible(_uiRefs.lightElevationLabel, lightingOn);
    setVisible(_uiRefs.lightElevation, lightingOn);
    setVisible(_uiRefs.lightDiffuseLabel, lightingOn);
    setVisible(_uiRefs.lightDiffuse, lightingOn);
    setVisible(_uiRefs.lightAmbientLabel, lightingOn);
    setVisible(_uiRefs.lightAmbient, lightingOn);
    setVisible(_uiRefs.useVolumeGradients, lightingOn);

    setVisible(_uiRefs.preTfX1, preTfOn);
    setVisible(_uiRefs.preTfY1, preTfOn);
    setVisible(_uiRefs.preTfX2, preTfOn);
    setVisible(_uiRefs.preTfY2, preTfOn);
    setVisible(_uiRefs.preTfKnot2Label, preTfOn);
    setVisible(_uiRefs.postTfX1, postTfOn);
    setVisible(_uiRefs.postTfY1, postTfOn);
    setVisible(_uiRefs.postTfX2, postTfOn);
    setVisible(_uiRefs.postTfY2, postTfOn);
    setVisible(_uiRefs.postTfKnot2Label, postTfOn);

    setVisible(_uiRefs.methodScaleLabel, false);
    setVisible(_uiRefs.methodScale, false);
    setVisible(_uiRefs.methodScaleValue, false);
    setVisible(_uiRefs.methodParamLabel, false);
    setVisible(_uiRefs.methodParam, false);
    setVisible(_uiRefs.methodParamValue, false);
}

void ViewerCompositePanel::updateRakingControlsEnabled(bool enabled)
{
    setEnabled(_uiRefs.rakingAzimuth, enabled);
    setEnabled(_uiRefs.rakingElevation, enabled);
    setEnabled(_uiRefs.rakingStrength, enabled);
    setEnabled(_uiRefs.rakingDepthScale, enabled);
    setEnabled(_uiRefs.rakingAzimuthLabel, enabled);
    setEnabled(_uiRefs.rakingElevationLabel, enabled);
    setEnabled(_uiRefs.rakingStrengthLabel, enabled);
    setEnabled(_uiRefs.rakingDepthLabel, enabled);
}

void ViewerCompositePanel::applyToSegmentationViewer(const std::function<void(VolumeViewerBase*)>& apply)
{
    if (!_viewerManager || !apply) {
        return;
    }
    for (auto* viewer : _viewerManager->baseViewers()) {
        if (viewer && viewer->surfName() == "segmentation") {
            apply(viewer);
            return;
        }
    }
}

void ViewerCompositePanel::applyToAllViewers(const std::function<void(VolumeViewerBase*)>& apply)
{
    if (!_viewerManager || !apply) {
        return;
    }
    _viewerManager->forEachBaseViewer([&apply](VolumeViewerBase* viewer) {
        if (viewer) {
            apply(viewer);
        }
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
