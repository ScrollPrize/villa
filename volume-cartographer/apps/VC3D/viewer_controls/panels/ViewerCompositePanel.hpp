#pragma once

#include <QMetaObject>
#include <QWidget>

#include <functional>
#include <vector>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QScrollArea;
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
    QDoubleSpinBox* _volumetricWScale{nullptr};
};
