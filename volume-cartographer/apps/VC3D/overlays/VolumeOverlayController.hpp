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
class QComboBox;
class QSlider;
class QSpinBox;
class QString;

class VolumeOverlayController : public QObject
{
    Q_OBJECT

public:
    struct UiRefs {
        QPointer<QComboBox> volumeSelect;
        QPointer<QComboBox> colormapSelect;
        QPointer<QSlider> opacitySlider;
        QPointer<QSpinBox> thresholdSpin;
    };

    explicit VolumeOverlayController(ViewerManager* manager, QObject* parent = nullptr);

    void setUi(const UiRefs& ui);
    void setVolumePkg(const std::shared_ptr<VolumePkg>& pkg, const QString& path);
    void clearVolumePkg();
    void refreshVolumeOptions();
    void toggleVisibility();
    bool hasOverlaySelection() const;

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

    void handleVolumeComboChanged(int index);
    void handleColormapChanged(int index);
    void handleOpacityChanged(int value);
    void handleThresholdChanged(int value);

    ViewerManager* _viewerManager{nullptr};
    UiRefs _ui;
    std::shared_ptr<VolumePkg> _volumePkg;
    QString _volpkgPath;
    std::shared_ptr<Volume> _overlayVolume;

    std::string _overlayVolumeId;
    std::string _overlayColormapName;
    float _overlayOpacity{0.5f};
    float _overlayOpacityBeforeToggle{0.5f};
    float _overlayThreshold{1.0f};
    bool _overlayVisible{false};

    std::vector<QMetaObject::Connection> _connections;
    bool _suspendPersistence{false};
};
