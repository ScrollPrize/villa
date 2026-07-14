#pragma once

#include <QObject>

#include <QMetaObject>
#include <QPointer>

#include <memory>
#include <string>
#include <vector>

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
        QPointer<QSpinBox> opacitySpin;
        QPointer<QSpinBox> thresholdSpin;
        QPointer<QSpinBox> maxDisplayedResolutionSpin;
        QPointer<QCheckBox> compositeEnabledCheck;
        QPointer<QComboBox> compositeMethodSelect;
        QPointer<QSpinBox> compositeLayersFrontSpin;
        QPointer<QSpinBox> compositeLayersBehindSpin;
    };

    explicit VolumeOverlayController(ViewerManager* manager, QObject* parent = nullptr);

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
    void setColormap(const std::string& id);
    void setOpacity(float value);
    void setThreshold(float value);
    void setWindowBounds(float low, float high);
    OverlayCompositeSettings currentCompositeSettings() const;
    void syncCompositeUi();
    void pushCompositeToManager();

    void handleVolumeComboChanged(int index);
    void handleColormapChanged(int index);
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

    std::vector<QMetaObject::Connection> _connections;
    bool _suspendPersistence{false};
};
