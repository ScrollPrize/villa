#pragma once

#include <QObject>

#include <QMetaObject>
#include <QPointer>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "vc/core/types/Sampling.hpp"

class ViewerManager;
class VolumePkg;
class Volume;
class QCheckBox;
class QComboBox;
class QSpinBox;
class QString;
struct OverlayCompositeSettings;

class VolumeOverlayController : public QObject
{
    Q_OBJECT

public:
    struct UiRefs {
        QPointer<QComboBox> volumeSelect;
        QPointer<QComboBox> colormapSelect;
        QPointer<QComboBox> samplingMethodSelect;
        QPointer<QSpinBox> opacitySpin;
        QPointer<QSpinBox> thresholdSpin;
        QPointer<QSpinBox> maxDisplayedResolutionSpin;
        QPointer<QCheckBox> compositeEnabledCheck;
        QPointer<QComboBox> compositeMethodSelect;
        QPointer<QSpinBox> compositeLayersFrontSpin;
        QPointer<QSpinBox> compositeLayersBehindSpin;
    };

    explicit VolumeOverlayController(ViewerManager* manager, QObject* parent = nullptr);

    void setViewerManager(ViewerManager* manager);
    void setUi(const UiRefs& ui);
    void setVolumePkg(const std::shared_ptr<VolumePkg>& pkg, const QString& path);
    void clearVolumePkg();
    void refreshVolumeOptions();
    void refreshForCurrentVolume();
    void toggleVisibility();
    bool hasOverlaySelection() const;
    [[nodiscard]] const std::string& selectedOverlayId() const noexcept { return _overlayVolumeId; }
    bool selectOverlay(const std::string& volumeId);
    void setSelectedColormap(const std::string& colormapId) { setColormap(colormapId); }
    void setSupplementalVolume(const std::string& volumeId,
                               const QString& label,
                               std::shared_ptr<Volume> volume);
    void removeSupplementalVolume(const std::string& volumeId);
    void syncWindowFromManager(float low, float high);

signals:
    void requestStatusMessage(const QString& message, int timeoutMs);
    void overlaySelectionChanged(std::string volumeId);

private:
    void connectUiSignals();
    void disconnectUiSignals();
    void populateColormapOptions();
    void applyOverlayVolume();
    void updateUiEnabled();
    void loadState();
    void saveState() const;
    void setColormap(const std::string& id);
    void setSamplingMethod(vc::Sampling method);
    void setOpacity(float value);
    void setThreshold(float value);
    void setWindowBounds(float low, float high);
    OverlayCompositeSettings currentCompositeSettings() const;
    void syncCompositeUi();
    void pushCompositeToManager();
    [[nodiscard]] bool isCategoricalOverlay() const;

    void handleVolumeComboChanged(int index);
    void handleColormapChanged(int index);
    void handleSamplingMethodChanged(int index);
    void handleOpacityChanged(int value);
    void handleThresholdChanged(int value);
    void handleMaxDisplayedResolutionChanged(int value);
    void handleCompositeEnabledChanged(bool enabled);
    void handleCompositeMethodChanged(int index);
    void handleCompositeLayersFrontChanged(int value);
    void handleCompositeLayersBehindChanged(int value);

    ViewerManager* _viewerManager{nullptr};
    UiRefs _ui;
    std::shared_ptr<VolumePkg> _volumePkg;
    QString _volpkgPath;
    std::shared_ptr<Volume> _overlayVolume;

    std::string _overlayVolumeId;
    std::string _overlayVolumeIdBeforeToggle;
    std::string _overlayColormapName;
    vc::Sampling _overlaySamplingMethod{vc::Sampling::Nearest};
    float _overlayOpacity{0.5f};
    float _overlayOpacityBeforeToggle{0.5f};
    float _overlayWindowLow{0.0f};
    float _overlayWindowHigh{255.0f};
    int _overlayMaxDisplayedResolution{0};
    bool _compositeEnabled{false};
    std::string _compositeMethod{"max"};
    int _compositeLayersFront{8};
    int _compositeLayersBehind{0};
    bool _overlayVisible{false};

    struct SupplementalVolume {
        QString label;
        std::shared_ptr<Volume> volume;
    };
    std::unordered_map<std::string, SupplementalVolume> _supplementalVolumes;

    std::vector<QMetaObject::Connection> _connections;
    bool _suspendPersistence{false};
};
