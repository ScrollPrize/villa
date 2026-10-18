#pragma once

#include <QMetaObject>
#include <QWidget>

#include <functional>
#include <optional>
#include <string>
#include <vector>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QScrollArea;
class QSlider;
class QSpinBox;
class QVBoxLayout;
class ViewerManager;
class VolumeViewerBase;

class ViewerCompositePanel : public QWidget
{
    Q_OBJECT

public:
    struct UiRefs {
        QScrollArea* scrollArea{nullptr};
        QWidget* contents{nullptr};

        QCheckBox* compositeEnabled{nullptr};
        QComboBox* compositeMode{nullptr};
        QSpinBox* layersInFront{nullptr};
        QSpinBox* layersBehind{nullptr};

        QLabel* alphaMinLabel{nullptr};
        QSpinBox* alphaMin{nullptr};
        QLabel* alphaMaxLabel{nullptr};
        QSpinBox* alphaMax{nullptr};
        QLabel* alphaThresholdLabel{nullptr};
        QSpinBox* alphaThreshold{nullptr};
        QLabel* materialLabel{nullptr};
        QSpinBox* material{nullptr};
        QCheckBox* reverseDirection{nullptr};

        QCheckBox* planeCompositeXY{nullptr};
        QCheckBox* planeCompositeXZ{nullptr};
        QCheckBox* planeCompositeYZ{nullptr};
        QSpinBox* planeLayersFront{nullptr};
        QSpinBox* planeLayersBehind{nullptr};
        QCheckBox* planeReverseDirection{nullptr};
    };

    explicit ViewerCompositePanel(const UiRefs& uiRefs,
                                  ViewerManager* viewerManager,
                                  QWidget* parent = nullptr);

    void setViewerManagers(const std::vector<ViewerManager*>& viewerManagers);
    void toggleSegmentationComposite();
    void setSegmentationCompositeChecked(bool checked);

    // The panel's controls as values. `method` is one of compositeMethods();
    // volumetric lighting is 0-100.
    struct Controls {
        bool enabled{false};
        std::string method;
        int layersFront{0};
        int layersBehind{0};
        bool reverseDirection{false};
        int alphaMin{0};
        int alphaMax{0};
        int alphaThreshold{0};
        int material{0};
        bool planeXY{false};
        bool planeXZ{false};
        bool planeYZ{false};
        int planeLayersFront{0};
        int planeLayersBehind{0};
        bool planeReverseDirection{false};
        double volumetricGamma{1.5};
        int volumetricLighting{0};
        double volumetricWScale{2.5};
    };

    // A change to any subset of the controls.
    struct ControlsUpdate {
        std::optional<bool> enabled;
        std::optional<std::string> method;   // one of compositeMethods()
        std::optional<int> layersFront;
        std::optional<int> layersBehind;
        std::optional<bool> reverseDirection;
        std::optional<int> alphaMin;
        std::optional<int> alphaMax;
        std::optional<int> alphaThreshold;
        std::optional<int> material;
        std::optional<bool> planeXY;
        std::optional<bool> planeXZ;
        std::optional<bool> planeYZ;
        std::optional<int> planeLayersFront;
        std::optional<int> planeLayersBehind;
        std::optional<bool> planeReverseDirection;
        std::optional<double> volumetricGamma;
        std::optional<int> volumetricLighting;
        std::optional<double> volumetricWScale;
    };

    // The composite methods in the order the mode combo lists them.
    static const std::vector<std::string>& compositeMethods();

    Controls controls() const;
    // Sets each given control as a person would, so its own handler applies
    // the value to the viewers, clamps it to the control's range and saves
    // it. A method outside compositeMethods() is ignored.
    void applyControls(const ControlsUpdate& update);

private:
    void setupControls();
    void setupVolumetricControls(QVBoxLayout* layout);
    void initializeExistingViewers();
    void applyInitialSettingsToViewer(VolumeViewerBase* viewer);
    void syncUiFromManager();
    void updateCompositeParamsVisibility();
    void applyToSegmentationViewer(const std::function<void(VolumeViewerBase*)>& apply);
    void applyToAllViewers(const std::function<void(VolumeViewerBase*)>& apply);
    void applyToPlaneViewers(const std::function<void(VolumeViewerBase*)>& apply);

    UiRefs _uiRefs;
    std::vector<ViewerManager*> _viewerManagers;
    std::vector<QMetaObject::Connection> _managerConnections;

    // Volumetric-mode controls (built programmatically; the .ui file only
    // carries the shared composite rows).
    QWidget* _volumetricGroup{nullptr};
    QWidget* _volumetricFlattenedGroup{nullptr};
    QDoubleSpinBox* _volumetricGamma{nullptr};
    QSlider* _volumetricLighting{nullptr};
    QDoubleSpinBox* _volumetricWScale{nullptr};
};
