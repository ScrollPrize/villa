#include "VolumeOverlayController.hpp"

#include "../ViewerManager.hpp"
#include "../VolumeViewerCmaps.hpp"
#include "../volume_viewers/VolumeViewerBase.hpp"
#include "../VCSettings.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDir>
#include <QFileInfo>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QVariant>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace
{
constexpr const char* kOverlaySettingsGroup = "overlay_state";

QString normalizedVolpkgPath(const QString& path)
{
    if (path.isEmpty()) {
        return QString();
    }

    QFileInfo info(path);
    if (info.exists()) {
        const QString canonical = info.canonicalFilePath();
        if (!canonical.isEmpty()) {
            return canonical;
        }
    }

    return QDir::cleanPath(info.absoluteFilePath());
}

QString overlaySettingsGroupKey(const QString& volpkgPath)
{
    const QString normalized = normalizedVolpkgPath(volpkgPath);
    if (normalized.isEmpty()) {
        return QString();
    }

    const QByteArray hash = QCryptographicHash::hash(normalized.toUtf8(), QCryptographicHash::Sha1).toHex();
    return QString::fromLatin1(hash);
}

QString overlayVolumeLabel(const std::shared_ptr<Volume>& volume, const QString& id)
{
    if (!volume) {
        return id;
    }

    const QString name = QString::fromStdString(volume->name());
    if (name.isEmpty()) {
        return id;
    }

    return QStringLiteral("%1 (%2)").arg(name, id);
}

float normalizedOpacityFromPercent(int percentValue)
{
    return std::clamp(percentValue / 100.0f, 0.0f, 1.0f);
}

int percentValueFromOpacity(float opacity)
{
    return static_cast<int>(std::round(std::clamp(opacity, 0.0f, 1.0f) * 100.0f));
}

float windowValueFromSpin(int spinValue)
{
    return std::clamp(static_cast<float>(spinValue), 0.0f, 255.0f);
}

int spinValueFromWindow(float value)
{
    const float clamped = std::clamp(value, 0.0f, 255.0f);
    return static_cast<int>(std::round(clamped));
}

std::string sanitizedCompositeMethod(const std::string& method)
{
    if (method == "max" || method == "mean" || method == "min") {
        return method;
    }
    return vc3d::settings::volume_overlay::COMPOSITE_METHOD_DEFAULT;
}

std::string coordinateSpaceTag(const VolumePkg& pkg, const std::string& volumeId)
{
    constexpr std::string_view prefix = "vc-open-data-coordinate-space:";
    for (const auto& tag : pkg.volumeTags(volumeId)) {
        if (tag.rfind(prefix, 0) == 0)
            return tag.substr(prefix.size());
    }
    return {};
}

bool overlayCoordinatesCompatible(const VolumePkg& pkg,
                                  const std::string& baseId,
                                  const std::string& overlayId)
{
    const auto base = coordinateSpaceTag(pkg, baseId);
    const auto overlay = coordinateSpaceTag(pkg, overlayId);
    return (base.empty() && overlay.empty()) ||
           (!base.empty() && base == overlay);
}
} // namespace

VolumeOverlayController::VolumeOverlayController(ViewerManager* manager, QObject* parent)
    : QObject(parent)
    , _viewerManager(manager)
{
}

void VolumeOverlayController::setViewerManager(ViewerManager* manager)
{
    if (_viewerManager == manager) {
        return;
    }
    if (_viewerManager) {
        _viewerManager->setVolumeOverlay(nullptr);
    }

    _viewerManager = manager;
    if (!_viewerManager) {
        updateUiEnabled();
        return;
    }

    const bool wasSuspended = _suspendPersistence;
    _suspendPersistence = true;
    _overlayVolume = _viewerManager->overlayVolume();
    _overlayVolumeId = _viewerManager->overlayVolumeId();
    _overlayOpacity = _viewerManager->overlayOpacity();
    _overlayOpacityBeforeToggle = _overlayOpacity;
    _overlayColormapName = _viewerManager->overlayColormap();
    _overlayWindowLow = _viewerManager->overlayWindowLow();
    _overlayWindowHigh = _viewerManager->overlayWindowHigh();
    _overlayMaxDisplayedResolution = _viewerManager->overlayMaxDisplayedResolution();
    const auto& composite = _viewerManager->overlayComposite();
    _compositeEnabled = composite.enabled;
    _compositeMethod = composite.method;
    _compositeLayersFront = composite.layersFront;
    _compositeLayersBehind = composite.layersBehind;
    _overlayVisible = hasOverlaySelection() && _overlayOpacity > 0.0f;

    _viewerManager->setVolumeOverlay(this);
    refreshVolumeOptions();
    populateColormapOptions();
    applyOverlayVolume();
    setOpacity(_overlayOpacity);
    setWindowBounds(_overlayWindowLow, _overlayWindowHigh);
    if (_ui.maxDisplayedResolutionSpin) {
        const QSignalBlocker blocker(_ui.maxDisplayedResolutionSpin);
        _ui.maxDisplayedResolutionSpin->setValue(_overlayMaxDisplayedResolution);
    }
    _viewerManager->setOverlayMaxDisplayedResolution(_overlayMaxDisplayedResolution);
    syncCompositeUi();
    pushCompositeToManager();
    updateUiEnabled();
    _suspendPersistence = wasSuspended;
}

void VolumeOverlayController::setUi(const UiRefs& ui)
{
    disconnectUiSignals();
    _ui = ui;

    if (_ui.opacitySpin) {
        _ui.opacitySpin->setRange(0, 100);
        QSignalBlocker blocker(_ui.opacitySpin);
        _ui.opacitySpin->setValue(percentValueFromOpacity(_overlayOpacity));
    }

    if (_ui.thresholdSpin) {
        _ui.thresholdSpin->setRange(0, 255);
        _ui.thresholdSpin->setValue(spinValueFromWindow(_overlayWindowLow));
    }

    if (_ui.maxDisplayedResolutionSpin) {
        _ui.maxDisplayedResolutionSpin->setRange(0, 5);
        QSignalBlocker blocker(_ui.maxDisplayedResolutionSpin);
        _ui.maxDisplayedResolutionSpin->setValue(std::clamp(_overlayMaxDisplayedResolution, 0, 5));
    }

    if (_ui.compositeMethodSelect) {
        const QSignalBlocker blocker(_ui.compositeMethodSelect);
        _ui.compositeMethodSelect->clear();
        _ui.compositeMethodSelect->addItem(tr("Maximum"), QStringLiteral("max"));
        _ui.compositeMethodSelect->addItem(tr("Mean"), QStringLiteral("mean"));
        _ui.compositeMethodSelect->addItem(tr("Minimum"), QStringLiteral("min"));
    }
    if (_ui.compositeLayersFrontSpin) {
        _ui.compositeLayersFrontSpin->setRange(0, 64);
    }
    if (_ui.compositeLayersBehindSpin) {
        _ui.compositeLayersBehindSpin->setRange(0, 64);
    }
    syncCompositeUi();

    populateColormapOptions();
    refreshVolumeOptions();
    updateUiEnabled();
    connectUiSignals();
}

void VolumeOverlayController::setVolumePkg(const std::shared_ptr<VolumePkg>& pkg, const QString& path)
{
    saveState();

    _volumePkg = pkg;
    _volpkgPath = normalizedVolpkgPath(path);
    _overlayVolume.reset();

    _suspendPersistence = true;
    loadState();
    refreshVolumeOptions();
    populateColormapOptions();
    applyOverlayVolume();
    setColormap(_overlayColormapName);
    setOpacity(_overlayOpacity);
    setWindowBounds(_overlayWindowLow, _overlayWindowHigh);
    if (_ui.maxDisplayedResolutionSpin) {
        const QSignalBlocker blocker(_ui.maxDisplayedResolutionSpin);
        _ui.maxDisplayedResolutionSpin->setValue(std::clamp(_overlayMaxDisplayedResolution, 0, 5));
    }
    if (_viewerManager) {
        _viewerManager->setOverlayMaxDisplayedResolution(_overlayMaxDisplayedResolution);
    }
    syncCompositeUi();
    pushCompositeToManager();
    updateUiEnabled();
    _suspendPersistence = false;
}

void VolumeOverlayController::clearVolumePkg()
{
    saveState();

    _suspendPersistence = true;
    _volumePkg.reset();
    _volpkgPath.clear();
    _overlayVolume.reset();
    _overlayVolumeId.clear();
    _overlayVisible = false;

    if (_viewerManager) {
        _viewerManager->setOverlayVolume(nullptr, _overlayVolumeId);
    }

    if (_ui.volumeSelect) {
        const QSignalBlocker blocker(_ui.volumeSelect);
        _ui.volumeSelect->clear();
        _ui.volumeSelect->addItem(tr("None"));
        _ui.volumeSelect->setItemData(0, QVariant());
        _ui.volumeSelect->setCurrentIndex(0);
    }

    if (_ui.colormapSelect) {
        const QSignalBlocker blocker(_ui.colormapSelect);
        _ui.colormapSelect->clear();
    }

    _overlayOpacity = 0.5f;
    _overlayOpacityBeforeToggle = _overlayOpacity;
    _overlayWindowLow = 0.0f;
    _overlayWindowHigh = 255.0f;
    _overlayMaxDisplayedResolution = vc3d::settings::volume_overlay::MAX_DISPLAYED_RESOLUTION_DEFAULT;
    _compositeEnabled = vc3d::settings::volume_overlay::COMPOSITE_ENABLED_DEFAULT;
    _compositeMethod = vc3d::settings::volume_overlay::COMPOSITE_METHOD_DEFAULT;
    _compositeLayersFront = vc3d::settings::volume_overlay::COMPOSITE_LAYERS_FRONT_DEFAULT;
    _compositeLayersBehind = vc3d::settings::volume_overlay::COMPOSITE_LAYERS_BEHIND_DEFAULT;
    syncCompositeUi();
    if (_ui.opacitySpin) {
        const QSignalBlocker blocker(_ui.opacitySpin);
        _ui.opacitySpin->setValue(percentValueFromOpacity(_overlayOpacity));
    }
    if (_ui.thresholdSpin) {
        const QSignalBlocker blocker(_ui.thresholdSpin);
        _ui.thresholdSpin->setValue(spinValueFromWindow(_overlayWindowLow));
    }
    if (_ui.maxDisplayedResolutionSpin) {
        const QSignalBlocker blocker(_ui.maxDisplayedResolutionSpin);
        _ui.maxDisplayedResolutionSpin->setValue(_overlayMaxDisplayedResolution);
    }

    if (_viewerManager) {
        _viewerManager->setOverlayOpacity(_overlayOpacity);
        _viewerManager->setOverlayWindow(_overlayWindowLow, _overlayWindowHigh);
        _viewerManager->setOverlayColormap(std::string());
        _viewerManager->setOverlayMaxDisplayedResolution(_overlayMaxDisplayedResolution);
        _viewerManager->setOverlayComposite(currentCompositeSettings());
    }

    updateUiEnabled();
    _suspendPersistence = false;
}

void VolumeOverlayController::refreshVolumeOptions()
{
    if (!_ui.volumeSelect) {
        return;
    }

    const QSignalBlocker blocker(_ui.volumeSelect);
    _ui.volumeSelect->clear();
    _ui.volumeSelect->addItem(tr("None"));
    _ui.volumeSelect->setItemData(0, QVariant());

    int indexToSelect = 0;

    if (_volumePkg) {
        const std::string baseVolumeId = _viewerManager
            ? _viewerManager->currentVolumeId()
            : std::string{};
        for (const auto& id : _volumePkg->volumeIDs()) {
            if (!baseVolumeId.empty() &&
                !overlayCoordinatesCompatible(*_volumePkg, baseVolumeId, id))
                continue;
            std::shared_ptr<Volume> volume;
            try {
                volume = _volumePkg->volume(id);
            } catch (const std::out_of_range&) {
                continue;
            }

            const QString idStr = QString::fromStdString(id);
            const QString label = overlayVolumeLabel(volume, idStr);
            const int row = _ui.volumeSelect->count();
            _ui.volumeSelect->addItem(label, QVariant(idStr));
            if (!_overlayVolumeId.empty() && _overlayVolumeId == id) {
                indexToSelect = row;
            }
        }
    }

    _ui.volumeSelect->setCurrentIndex(indexToSelect);
    if (indexToSelect == 0 && !_overlayVolumeId.empty()) {
        _overlayVolumeId.clear();
    }
}

void VolumeOverlayController::refreshForCurrentVolume()
{
    refreshVolumeOptions();
    applyOverlayVolume();
    updateUiEnabled();
}

void VolumeOverlayController::toggleVisibility()
{
    if (_overlayVisible) {
        if (_overlayOpacity > 0.0f) {
            _overlayOpacityBeforeToggle = _overlayOpacity;
        }
        if (!_overlayVolumeId.empty()) {
            _overlayVolumeIdBeforeToggle = _overlayVolumeId;
        }

        if (_ui.volumeSelect) {
            if (_ui.volumeSelect->currentIndex() != 0) {
                _ui.volumeSelect->setCurrentIndex(0);
            } else if (!_overlayVolumeId.empty()) {
                // UI already points to "None", ensure internal state matches.
                _overlayVolumeId.clear();
                applyOverlayVolume();
                updateUiEnabled();
            }
        } else {
            _overlayVolumeId.clear();
            applyOverlayVolume();
            updateUiEnabled();
        }

        _overlayVisible = false;
        if (!_suspendPersistence) {
            saveState();
        }
        emit requestStatusMessage(tr("Volume overlay hidden"), 1200);
        return;
    }

    const std::string restoreId = !_overlayVolumeIdBeforeToggle.empty() ? _overlayVolumeIdBeforeToggle : _overlayVolumeId;
    if (restoreId.empty()) {
        emit requestStatusMessage(tr("No overlay volume selected"), 1200);
        return;
    }

    bool restored = false;
    if (_ui.volumeSelect) {
        const int count = _ui.volumeSelect->count();
        for (int row = 0; row < count; ++row) {
            const QVariant data = _ui.volumeSelect->itemData(row);
            if (!data.isValid()) {
                continue;
            }
            if (data.toString().toStdString() == restoreId) {
                if (_ui.volumeSelect->currentIndex() != row) {
                    _ui.volumeSelect->setCurrentIndex(row);
                } else if (_overlayVolumeId != restoreId) {
                    _overlayVolumeId = restoreId;
                    applyOverlayVolume();
                    updateUiEnabled();
                }
                restored = true;
                break;
            }
        }
    }

    if (!restored) {
        _overlayVolumeId = restoreId;
        applyOverlayVolume();
        updateUiEnabled();
        restored = hasOverlaySelection();
    }

    if (!restored) {
        emit requestStatusMessage(tr("Selected overlay volume unavailable"), 1200);
        return;
    }

    const float restoredOpacity = (_overlayOpacityBeforeToggle > 0.0f) ? _overlayOpacityBeforeToggle : 0.5f;
    setOpacity(restoredOpacity);

    const bool hasSelection = hasOverlaySelection();
    _overlayVisible = hasSelection && _overlayOpacity > 0.0f;
    if (_overlayVisible) {
        _overlayVolumeIdBeforeToggle.clear();
        _overlayOpacityBeforeToggle = _overlayOpacity;
    }

    if (!_suspendPersistence) {
        saveState();
    }

    if (_overlayVisible) {
        emit requestStatusMessage(tr("Volume overlay shown"), 1200);
    } else if (hasSelection) {
        emit requestStatusMessage(tr("Volume overlay shown (opacity 0%)"), 1200);
    } else {
        emit requestStatusMessage(tr("Selected overlay volume unavailable"), 1200);
    }
}

bool VolumeOverlayController::hasOverlaySelection() const
{
    return _overlayVolume && !_overlayVolumeId.empty();
}

void VolumeOverlayController::connectUiSignals()
{
    _connections.clear();

    if (_ui.volumeSelect) {
        _connections.push_back(QObject::connect(
            _ui.volumeSelect, qOverload<int>(&QComboBox::currentIndexChanged),
            this, [this](int index) { handleVolumeComboChanged(index); }));
    }

    if (_ui.colormapSelect) {
        _connections.push_back(QObject::connect(
            _ui.colormapSelect, qOverload<int>(&QComboBox::currentIndexChanged),
            this, [this](int index) { handleColormapChanged(index); }));
    }

    if (_ui.opacitySpin) {
        _connections.push_back(QObject::connect(
            _ui.opacitySpin, qOverload<int>(&QSpinBox::valueChanged),
            this, [this](int value) { handleOpacityChanged(value); }));
    }

    if (_ui.thresholdSpin) {
        _connections.push_back(QObject::connect(
            _ui.thresholdSpin, qOverload<int>(&QSpinBox::valueChanged),
            this, [this](int value) { handleThresholdChanged(value); }));
    }

    if (_ui.maxDisplayedResolutionSpin) {
        _connections.push_back(QObject::connect(
            _ui.maxDisplayedResolutionSpin, qOverload<int>(&QSpinBox::valueChanged),
            this, [this](int value) { handleMaxDisplayedResolutionChanged(value); }));
    }

    if (_ui.compositeEnabledCheck) {
        _connections.push_back(QObject::connect(
            _ui.compositeEnabledCheck, &QCheckBox::toggled,
            this, [this](bool checked) { handleCompositeEnabledChanged(checked); }));
    }

    if (_ui.compositeMethodSelect) {
        _connections.push_back(QObject::connect(
            _ui.compositeMethodSelect, qOverload<int>(&QComboBox::currentIndexChanged),
            this, [this](int index) { handleCompositeMethodChanged(index); }));
    }

    if (_ui.compositeLayersFrontSpin) {
        _connections.push_back(QObject::connect(
            _ui.compositeLayersFrontSpin, qOverload<int>(&QSpinBox::valueChanged),
            this, [this](int value) { handleCompositeLayersFrontChanged(value); }));
    }

    if (_ui.compositeLayersBehindSpin) {
        _connections.push_back(QObject::connect(
            _ui.compositeLayersBehindSpin, qOverload<int>(&QSpinBox::valueChanged),
            this, [this](int value) { handleCompositeLayersBehindChanged(value); }));
    }
}

void VolumeOverlayController::disconnectUiSignals()
{
    for (auto& connection : _connections) {
        QObject::disconnect(connection);
    }
    _connections.clear();
}

void VolumeOverlayController::populateColormapOptions()
{
    if (!_ui.colormapSelect) {
        return;
    }

    const auto& entries = volume_viewer_cmaps::entries(
        volume_viewer_cmaps::EntryScope::OverlayCompatible);
    const QSignalBlocker blocker(_ui.colormapSelect);
    _ui.colormapSelect->clear();

    int indexToSelect = 0;
    if (_overlayColormapName.empty() && !entries.empty()) {
        _overlayColormapName = entries.front().id;
    }

    for (int i = 0; i < static_cast<int>(entries.size()); ++i) {
        const auto& entry = entries.at(i);
        _ui.colormapSelect->addItem(entry.label, QVariant(QString::fromStdString(entry.id)));
        if (entry.id == _overlayColormapName) {
            indexToSelect = i;
        }
    }

    if (_ui.colormapSelect->count() > 0) {
        _ui.colormapSelect->setCurrentIndex(indexToSelect);
    }

    if (_viewerManager) {
        _viewerManager->setOverlayColormap(_overlayColormapName);
    }
}

void VolumeOverlayController::applyOverlayVolume()
{
    std::shared_ptr<Volume> overlayVolume;
    if (_volumePkg && !_overlayVolumeId.empty()) {
        const std::string baseVolumeId = _viewerManager
            ? _viewerManager->currentVolumeId()
            : std::string{};
        if (!baseVolumeId.empty() &&
            !overlayCoordinatesCompatible(*_volumePkg, baseVolumeId, _overlayVolumeId)) {
            emit requestStatusMessage(
                tr("Overlay rejected: volume coordinate spaces do not match."), 5000);
            _overlayVolumeId.clear();
            if (_ui.volumeSelect) {
                const QSignalBlocker blocker(_ui.volumeSelect);
                _ui.volumeSelect->setCurrentIndex(0);
            }
        }
    }
    if (_volumePkg && !_overlayVolumeId.empty()) {
        try {
            overlayVolume = _volumePkg->volume(_overlayVolumeId);
        } catch (const std::out_of_range&) {
            overlayVolume.reset();
            _overlayVolumeId.clear();
            if (_ui.volumeSelect) {
                const QSignalBlocker blocker(_ui.volumeSelect);
                _ui.volumeSelect->setCurrentIndex(0);
            }
        }
    }

    _overlayVolume = std::move(overlayVolume);
    if (_viewerManager) {
        _viewerManager->setOverlayVolume(_overlayVolume, _overlayVolumeId);
    }

    const bool visible = hasOverlaySelection() && _overlayOpacity > 0.0f;
    _overlayVisible = visible;
    if (_overlayVisible) {
        _overlayOpacityBeforeToggle = _overlayOpacity;
    }
}

void VolumeOverlayController::updateUiEnabled()
{
    const bool hasVolumeOptions = _ui.volumeSelect && _ui.volumeSelect->count() > 1;
    if (_ui.volumeSelect) {
        _ui.volumeSelect->setEnabled(hasVolumeOptions);
    }

    const bool hasOverlay = hasOverlaySelection();
    if (_ui.opacitySpin) {
        _ui.opacitySpin->setEnabled(hasOverlay);
    }
    if (_ui.thresholdSpin) {
        _ui.thresholdSpin->setEnabled(hasOverlay);
    }
    if (_ui.maxDisplayedResolutionSpin) {
        _ui.maxDisplayedResolutionSpin->setEnabled(hasOverlay);
    }
    if (_ui.colormapSelect) {
        const bool hasColormaps = _ui.colormapSelect->count() > 0;
        _ui.colormapSelect->setEnabled(hasOverlay && hasColormaps);
    }
    if (_ui.compositeEnabledCheck) {
        _ui.compositeEnabledCheck->setEnabled(hasOverlay);
    }
    const bool compositeControlsEnabled = hasOverlay && _compositeEnabled;
    if (_ui.compositeMethodSelect) {
        _ui.compositeMethodSelect->setEnabled(compositeControlsEnabled);
    }
    if (_ui.compositeLayersFrontSpin) {
        _ui.compositeLayersFrontSpin->setEnabled(compositeControlsEnabled);
    }
    if (_ui.compositeLayersBehindSpin) {
        _ui.compositeLayersBehindSpin->setEnabled(compositeControlsEnabled);
    }
}

void VolumeOverlayController::syncWindowFromManager(float low, float high)
{
    const bool wasSuspended = _suspendPersistence;
    _suspendPersistence = true;

    _overlayWindowLow = std::clamp(low, 0.0f, 255.0f);
    const float minHigh = std::min(_overlayWindowLow + 1.0f, 255.0f);
    _overlayWindowHigh = std::clamp(high, minHigh, 255.0f);

    if (_ui.thresholdSpin) {
        const QSignalBlocker blocker(_ui.thresholdSpin);
        _ui.thresholdSpin->setValue(spinValueFromWindow(_overlayWindowLow));
    }

    _suspendPersistence = wasSuspended;

    if (!wasSuspended) {
        saveState();
    }
}

void VolumeOverlayController::loadState()
{
    _overlayVolumeId.clear();
    _overlayOpacity = 0.5f;
    _overlayOpacityBeforeToggle = _overlayOpacity;
    _overlayWindowLow = 0.0f;
    _overlayWindowHigh = 255.0f;
    _overlayColormapName.clear();
    _overlayMaxDisplayedResolution = vc3d::settings::volume_overlay::MAX_DISPLAYED_RESOLUTION_DEFAULT;
    _compositeEnabled = vc3d::settings::volume_overlay::COMPOSITE_ENABLED_DEFAULT;
    _compositeMethod = vc3d::settings::volume_overlay::COMPOSITE_METHOD_DEFAULT;
    _compositeLayersFront = vc3d::settings::volume_overlay::COMPOSITE_LAYERS_FRONT_DEFAULT;
    _compositeLayersBehind = vc3d::settings::volume_overlay::COMPOSITE_LAYERS_BEHIND_DEFAULT;

    if (_volpkgPath.isEmpty()) {
        return;
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const QString groupKey = overlaySettingsGroupKey(_volpkgPath);
    if (groupKey.isEmpty()) {
        return;
    }

    using namespace vc3d::settings;
    settings.beginGroup(QString::fromLatin1(kOverlaySettingsGroup));
    settings.beginGroup(groupKey);

    const QString storedVolumeId = settings.value(volume_overlay::VOLUME_ID).toString();
    if (!storedVolumeId.isEmpty()) {
        _overlayVolumeId = storedVolumeId.toStdString();
    }

    _overlayOpacity = std::clamp(settings.value(volume_overlay::OPACITY, _overlayOpacity).toFloat(), 0.0f, 1.0f);
    _overlayOpacityBeforeToggle = _overlayOpacity;

    const QVariant storedWindowLow = settings.value(volume_overlay::WINDOW_LOW);
    if (storedWindowLow.isValid()) {
        _overlayWindowLow = std::clamp(storedWindowLow.toFloat(), 0.0f, 255.0f);
    } else {
        // Fall back to legacy threshold if present.
        const float legacyThreshold = std::max(0.0f, settings.value(volume_overlay::THRESHOLD, _overlayWindowLow).toFloat());
        _overlayWindowLow = std::clamp(legacyThreshold, 0.0f, 255.0f);
    }

    const QVariant storedWindowHigh = settings.value(volume_overlay::WINDOW_HIGH);
    if (storedWindowHigh.isValid()) {
        _overlayWindowHigh = std::clamp(storedWindowHigh.toFloat(), 0.0f, 255.0f);
    } else {
        // Default to full range when no explicit upper bound is stored.
        _overlayWindowHigh = 255.0f;
    }
    if (_overlayWindowHigh <= _overlayWindowLow) {
        _overlayWindowHigh = std::min(255.0f, _overlayWindowLow + 1.0f);
    }

    const QString storedColormap = settings.value(volume_overlay::COLORMAP).toString();
    if (!storedColormap.isEmpty()) {
        _overlayColormapName = storedColormap.toStdString();
    }

    _overlayMaxDisplayedResolution = std::clamp(
        settings.value(volume_overlay::MAX_DISPLAYED_RESOLUTION,
                       volume_overlay::MAX_DISPLAYED_RESOLUTION_DEFAULT).toInt(),
        0,
        5);

    _compositeEnabled = settings.value(volume_overlay::COMPOSITE_ENABLED,
                                       volume_overlay::COMPOSITE_ENABLED_DEFAULT).toBool();
    _compositeMethod = sanitizedCompositeMethod(
        settings.value(volume_overlay::COMPOSITE_METHOD,
                       QString::fromLatin1(volume_overlay::COMPOSITE_METHOD_DEFAULT))
            .toString().toStdString());
    _compositeLayersFront = std::clamp(
        settings.value(volume_overlay::COMPOSITE_LAYERS_FRONT,
                       volume_overlay::COMPOSITE_LAYERS_FRONT_DEFAULT).toInt(),
        0,
        64);
    _compositeLayersBehind = std::clamp(
        settings.value(volume_overlay::COMPOSITE_LAYERS_BEHIND,
                       volume_overlay::COMPOSITE_LAYERS_BEHIND_DEFAULT).toInt(),
        0,
        64);

    settings.endGroup();
    settings.endGroup();
}

void VolumeOverlayController::saveState() const
{
    if (_suspendPersistence || _volpkgPath.isEmpty()) {
        return;
    }

    const QString groupKey = overlaySettingsGroupKey(_volpkgPath);
    if (groupKey.isEmpty()) {
        return;
    }

    using namespace vc3d::settings;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(QString::fromLatin1(kOverlaySettingsGroup));
    settings.beginGroup(groupKey);
    settings.setValue(volume_overlay::PATH, _volpkgPath);
    settings.setValue(volume_overlay::VOLUME_ID, QString::fromStdString(_overlayVolumeId));
    settings.setValue(volume_overlay::OPACITY, _overlayOpacity);
    settings.setValue(volume_overlay::WINDOW_LOW, _overlayWindowLow);
    settings.setValue(volume_overlay::WINDOW_HIGH, _overlayWindowHigh);
    settings.setValue(volume_overlay::THRESHOLD, _overlayWindowLow); // legacy compatibility
    settings.setValue(volume_overlay::COLORMAP, QString::fromStdString(_overlayColormapName));
    settings.setValue(volume_overlay::MAX_DISPLAYED_RESOLUTION, _overlayMaxDisplayedResolution);
    settings.setValue(volume_overlay::COMPOSITE_ENABLED, _compositeEnabled);
    settings.setValue(volume_overlay::COMPOSITE_METHOD, QString::fromStdString(_compositeMethod));
    settings.setValue(volume_overlay::COMPOSITE_LAYERS_FRONT, _compositeLayersFront);
    settings.setValue(volume_overlay::COMPOSITE_LAYERS_BEHIND, _compositeLayersBehind);
    settings.endGroup();
    settings.endGroup();
}

void VolumeOverlayController::setColormap(const std::string& id)
{
    std::string newId = id;

    if (newId.empty()) {
        if (_ui.colormapSelect && _ui.colormapSelect->count() > 0) {
            const QVariant data = _ui.colormapSelect->itemData(_ui.colormapSelect->currentIndex());
            if (data.isValid()) {
                newId = data.toString().toStdString();
            }
        }

        if (newId.empty()) {
            const auto& entries = volume_viewer_cmaps::entries();
            if (!entries.empty()) {
                newId = entries.front().id;
            }
        }
    }

    _overlayColormapName = newId;

    if (_ui.colormapSelect) {
        const QSignalBlocker blocker(_ui.colormapSelect);
        const QString target = QString::fromStdString(_overlayColormapName);
        int index = _ui.colormapSelect->findData(target);
        if (index >= 0) {
            _ui.colormapSelect->setCurrentIndex(index);
        } else if (_ui.colormapSelect->count() > 0) {
            _ui.colormapSelect->setCurrentIndex(0);
            const QVariant data = _ui.colormapSelect->currentData();
            if (data.isValid()) {
                _overlayColormapName = data.toString().toStdString();
            } else {
                _overlayColormapName.clear();
            }
        }
    }

    if (_viewerManager) {
        _viewerManager->setOverlayColormap(_overlayColormapName);
    }
}

void VolumeOverlayController::setOpacity(float value)
{
    const float clamped = std::clamp(value, 0.0f, 1.0f);
    _overlayOpacity = clamped;

    if (_ui.opacitySpin) {
        const QSignalBlocker blocker(_ui.opacitySpin);
        _ui.opacitySpin->setValue(percentValueFromOpacity(_overlayOpacity));
    }

    if (_viewerManager) {
        _viewerManager->setOverlayOpacity(_overlayOpacity);
    }

    const bool visible = hasOverlaySelection() && _overlayOpacity > 0.0f;
    _overlayVisible = visible;
    if (_overlayVisible) {
        _overlayOpacityBeforeToggle = _overlayOpacity;
    }
}

void VolumeOverlayController::setThreshold(float value)
{
    const float clamped = std::clamp(value, 0.0f, 255.0f);
    setWindowBounds(clamped, _overlayWindowHigh);
}

void VolumeOverlayController::setWindowBounds(float low, float high)
{
    const float clampedLow = std::clamp(low, 0.0f, 255.0f);
    float clampedHigh = std::clamp(high, 0.0f, 255.0f);
    if (clampedHigh <= clampedLow) {
        clampedHigh = std::min(255.0f, clampedLow + 1.0f);
    }

    if (std::abs(clampedLow - _overlayWindowLow) < 1e-6f &&
        std::abs(clampedHigh - _overlayWindowHigh) < 1e-6f) {
        return;
    }

    _overlayWindowLow = clampedLow;
    _overlayWindowHigh = clampedHigh;

    if (_ui.thresholdSpin) {
        const QSignalBlocker blocker(_ui.thresholdSpin);
        _ui.thresholdSpin->setValue(spinValueFromWindow(_overlayWindowLow));
    }

    if (_viewerManager) {
        _viewerManager->setOverlayWindow(_overlayWindowLow, _overlayWindowHigh);
    }
}

OverlayCompositeSettings VolumeOverlayController::currentCompositeSettings() const
{
    OverlayCompositeSettings settings;
    settings.enabled = _compositeEnabled;
    settings.method = sanitizedCompositeMethod(_compositeMethod);
    settings.layersFront = std::clamp(_compositeLayersFront, 0, 64);
    settings.layersBehind = std::clamp(_compositeLayersBehind, 0, 64);
    return settings;
}

void VolumeOverlayController::syncCompositeUi()
{
    if (_ui.compositeEnabledCheck) {
        const QSignalBlocker blocker(_ui.compositeEnabledCheck);
        _ui.compositeEnabledCheck->setChecked(_compositeEnabled);
    }
    if (_ui.compositeMethodSelect) {
        const QSignalBlocker blocker(_ui.compositeMethodSelect);
        const int index = _ui.compositeMethodSelect->findData(
            QString::fromStdString(sanitizedCompositeMethod(_compositeMethod)));
        if (index >= 0) {
            _ui.compositeMethodSelect->setCurrentIndex(index);
        }
    }
    if (_ui.compositeLayersFrontSpin) {
        const QSignalBlocker blocker(_ui.compositeLayersFrontSpin);
        _ui.compositeLayersFrontSpin->setValue(std::clamp(_compositeLayersFront, 0, 64));
    }
    if (_ui.compositeLayersBehindSpin) {
        const QSignalBlocker blocker(_ui.compositeLayersBehindSpin);
        _ui.compositeLayersBehindSpin->setValue(std::clamp(_compositeLayersBehind, 0, 64));
    }
}

void VolumeOverlayController::pushCompositeToManager()
{
    if (_viewerManager) {
        _viewerManager->setOverlayComposite(currentCompositeSettings());
    }
}

void VolumeOverlayController::handleCompositeEnabledChanged(bool enabled)
{
    if (_compositeEnabled == enabled) {
        return;
    }

    _compositeEnabled = enabled;
    pushCompositeToManager();
    updateUiEnabled();

    if (!_suspendPersistence) {
        saveState();
    }
}

void VolumeOverlayController::handleCompositeMethodChanged(int index)
{
    if (!_ui.compositeMethodSelect || index < 0) {
        return;
    }

    const QVariant data = _ui.compositeMethodSelect->itemData(index);
    if (!data.isValid()) {
        return;
    }

    const std::string method = sanitizedCompositeMethod(data.toString().toStdString());
    if (_compositeMethod == method) {
        return;
    }

    _compositeMethod = method;
    pushCompositeToManager();

    if (!_suspendPersistence) {
        saveState();
    }
}

void VolumeOverlayController::handleCompositeLayersFrontChanged(int value)
{
    const int clamped = std::clamp(value, 0, 64);
    if (_compositeLayersFront == clamped) {
        return;
    }

    _compositeLayersFront = clamped;
    pushCompositeToManager();

    if (!_suspendPersistence) {
        saveState();
    }
}

void VolumeOverlayController::handleCompositeLayersBehindChanged(int value)
{
    const int clamped = std::clamp(value, 0, 64);
    if (_compositeLayersBehind == clamped) {
        return;
    }

    _compositeLayersBehind = clamped;
    pushCompositeToManager();

    if (!_suspendPersistence) {
        saveState();
    }
}

void VolumeOverlayController::handleMaxDisplayedResolutionChanged(int value)
{
    const int clamped = std::clamp(value, 0, 5);
    if (_overlayMaxDisplayedResolution == clamped) {
        return;
    }

    _overlayMaxDisplayedResolution = clamped;
    if (_ui.maxDisplayedResolutionSpin && _ui.maxDisplayedResolutionSpin->value() != clamped) {
        const QSignalBlocker blocker(_ui.maxDisplayedResolutionSpin);
        _ui.maxDisplayedResolutionSpin->setValue(clamped);
    }

    if (_viewerManager) {
        _viewerManager->setOverlayMaxDisplayedResolution(_overlayMaxDisplayedResolution);
    }

    if (!_suspendPersistence) {
        saveState();
    }
}

void VolumeOverlayController::handleVolumeComboChanged(int index)
{
    if (!_ui.volumeSelect) {
        return;
    }

    std::string newId;
    if (index >= 0) {
        const QVariant data = _ui.volumeSelect->itemData(index);
        if (data.isValid()) {
            newId = data.toString().toStdString();
        }
    }

    if (newId == _overlayVolumeId) {
        return;
    }

    _overlayVolumeId = std::move(newId);
    applyOverlayVolume();
    updateUiEnabled();

    if (!_suspendPersistence) {
        saveState();
    }
}

void VolumeOverlayController::handleColormapChanged(int index)
{
    if (!_ui.colormapSelect) {
        return;
    }

    std::string newId;
    if (index >= 0) {
        const QVariant data = _ui.colormapSelect->itemData(index);
        if (data.isValid()) {
            newId = data.toString().toStdString();
        }
    }

    setColormap(newId);

    if (!_suspendPersistence) {
        saveState();
    }
}

void VolumeOverlayController::handleOpacityChanged(int value)
{
    setOpacity(normalizedOpacityFromPercent(value));
    if (!_suspendPersistence) {
        saveState();
    }
}

void VolumeOverlayController::handleThresholdChanged(int value)
{
    setThreshold(windowValueFromSpin(value));
    if (!_suspendPersistence) {
        saveState();
    }
}
