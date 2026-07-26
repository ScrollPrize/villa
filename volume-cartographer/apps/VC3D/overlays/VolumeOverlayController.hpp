#pragma once

#include <QObject>

#include <QMetaObject>
#include <QPointer>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vc/core/types/Sampling.hpp"
#include "vc/core/util/Compositing.hpp"

class ViewerManager;
class VolumePkg;
class Volume;
class QCheckBox;
class QComboBox;
class QSpinBox;
class QString;

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

    struct Window {
        float low{0.0f};
        float high{255.0f};
    };

    struct State {
        std::string currentVolumeId;
        std::string volumeId;
        std::string colormap;
        float opacity{0.5f};
        Window window;
        int maxDisplayedResolution{0};
        OverlayCompositeSettings composite;
    };

    struct Update {
        // A present empty id clears the selection; nullopt leaves it unchanged.
        std::optional<std::string> volumeId;
        std::optional<std::string> colormap;
        std::optional<float> opacity;
        std::optional<float> threshold;
        std::optional<Window> window;
        std::optional<int> maxDisplayedResolution;
        std::optional<OverlayCompositeSettings> composite;
    };

    enum class ApplyResult {
        Applied,
        NoVolumePackage,
        UnknownVolume,
    };

    explicit VolumeOverlayController(ViewerManager* manager, QObject* parent = nullptr);

    State state() const;
    ApplyResult apply(const Update& update);
    void setViewerManager(ViewerManager* manager);
    void setUi(const UiRefs& ui);
    void setVolumePkg(const std::shared_ptr<VolumePkg>& pkg, const QString& path);
    void clearVolumePkg();
    void refreshVolumeOptions();
    void refreshForCurrentVolume();
    void toggleVisibility();
    bool hasOverlaySelection() const;
    void syncWindowFromManager(float low, float high);

signals:
    void requestStatusMessage(const QString& message, int timeoutMs);

private:
    void connectUiSignals();
    void disconnectUiSignals();
    void populateColormapOptions();
    void applyOverlayVolume();
    void updateUiEnabled();
    void loadState();
    void saveState() const;
    void setVolumeId(const std::string& id);
    void setColormap(const std::string& id);
    void setSamplingMethod(vc::Sampling method);
    void setOpacity(float value);
    void setThreshold(float value);
    void setWindowBounds(float low, float high);
    void setMaxDisplayedResolution(int value);
    void setComposite(const OverlayCompositeSettings& settings);
    OverlayCompositeSettings currentCompositeSettings() const;
    void syncCompositeUi();
    void pushCompositeToManager();

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

    QMetaObject::Connection _volumeChangedConnection;
    std::vector<QMetaObject::Connection> _connections;
    bool _suspendPersistence{false};
};
