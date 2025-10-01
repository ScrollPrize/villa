#include "VolumeOverlayController.hpp"

#include "../ViewerManager.hpp"
#include "../CVolumeViewer.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"

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

float normalizedThresholdFromSpin(int spinValue)
{
    return std::max(0.0f, static_cast<float>(spinValue));
}

int spinValueFromThreshold(float threshold)
{
    const float clamped = std::clamp(threshold, 0.0f, 65535.0f);
    return static_cast<int>(std::round(clamped));
}
} // namespace

VolumeOverlayController::VolumeOverlayController(ViewerManager* manager, QObject* parent)
    : QObject(parent)
    , _viewerManager(manager)
{
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
        _ui.thresholdSpin->setRange(0, 65535);
        _ui.thresholdSpin->setValue(spinValueFromThreshold(_overlayThreshold));
    }

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
    setThreshold(_overlayThreshold);
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
    _overlayThreshold = 1.0f;
    if (_ui.opacitySpin) {
        const QSignalBlocker blocker(_ui.opacitySpin);
        _ui.opacitySpin->setValue(percentValueFromOpacity(_overlayOpacity));
    }
    if (_ui.thresholdSpin) {
        const QSignalBlocker blocker(_ui.thresholdSpin);
        _ui.thresholdSpin->setValue(spinValueFromThreshold(_overlayThreshold));
    }

    if (_viewerManager) {
        _viewerManager->setOverlayOpacity(_overlayOpacity);
        _viewerManager->setOverlayThreshold(_overlayThreshold);
        _viewerManager->setOverlayColormap(std::string());
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
        for (const auto& id : _volumePkg->volumeIDs()) {
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

void VolumeOverlayController::toggleVisibility()
{
    if (!hasOverlaySelection()) {
        return;
    }

    if (_overlayVisible) {
        if (_overlayOpacity > 0.0f) {
            _overlayOpacityBeforeToggle = _overlayOpacity;
        }
        setOpacity(0.0f);
        _overlayVisible = false;
        if (!_suspendPersistence) {
            saveState();
        }
        emit requestStatusMessage(tr("Volume overlay hidden"), 1200);
    } else {
        const float restored = (_overlayOpacityBeforeToggle > 0.0f) ? _overlayOpacityBeforeToggle : 0.5f;
        setOpacity(restored);
        _overlayVisible = hasOverlaySelection() && _overlayOpacity > 0.0f;
        if (_overlayVisible) {
            _overlayOpacityBeforeToggle = _overlayOpacity;
        }
        if (!_suspendPersistence) {
            saveState();
        }
        emit requestStatusMessage(_overlayVisible ? tr("Volume overlay shown") : tr("Volume overlay hidden"), 1200);
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

    const auto& entries = CVolumeViewer::overlayColormapEntries();
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
    if (_ui.colormapSelect) {
        const bool hasColormaps = _ui.colormapSelect->count() > 0;
        _ui.colormapSelect->setEnabled(hasOverlay && hasColormaps);
    }
}

void VolumeOverlayController::loadState()
{
    _overlayVolumeId.clear();
    _overlayOpacity = 0.5f;
    _overlayOpacityBeforeToggle = _overlayOpacity;
    _overlayThreshold = 1.0f;
    _overlayColormapName.clear();

    if (_volpkgPath.isEmpty()) {
        return;
    }

    QSettings settings(QStringLiteral("VC.ini"), QSettings::IniFormat);
    const QString groupKey = overlaySettingsGroupKey(_volpkgPath);
    if (groupKey.isEmpty()) {
        return;
    }

    settings.beginGroup(QString::fromLatin1(kOverlaySettingsGroup));
    settings.beginGroup(groupKey);

    const QString storedVolumeId = settings.value(QStringLiteral("volume_id")).toString();
    if (!storedVolumeId.isEmpty()) {
        _overlayVolumeId = storedVolumeId.toStdString();
    }

    _overlayOpacity = std::clamp(settings.value(QStringLiteral("opacity"), _overlayOpacity).toFloat(), 0.0f, 1.0f);
    _overlayOpacityBeforeToggle = _overlayOpacity;
    _overlayThreshold = std::max(0.0f, settings.value(QStringLiteral("threshold"), _overlayThreshold).toFloat());

    const QString storedColormap = settings.value(QStringLiteral("colormap")).toString();
    if (!storedColormap.isEmpty()) {
        _overlayColormapName = storedColormap.toStdString();
    }

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

    QSettings settings(QStringLiteral("VC.ini"), QSettings::IniFormat);
    settings.beginGroup(QString::fromLatin1(kOverlaySettingsGroup));
    settings.beginGroup(groupKey);
    settings.setValue(QStringLiteral("path"), _volpkgPath);
    settings.setValue(QStringLiteral("volume_id"), QString::fromStdString(_overlayVolumeId));
    settings.setValue(QStringLiteral("opacity"), _overlayOpacity);
    settings.setValue(QStringLiteral("threshold"), _overlayThreshold);
    settings.setValue(QStringLiteral("colormap"), QString::fromStdString(_overlayColormapName));
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
            const auto& entries = CVolumeViewer::overlayColormapEntries();
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
    const float clamped = std::max(0.0f, std::min(value, 65535.0f));
    _overlayThreshold = clamped;

    if (_ui.thresholdSpin) {
        const QSignalBlocker blocker(_ui.thresholdSpin);
        _ui.thresholdSpin->setValue(spinValueFromThreshold(_overlayThreshold));
    }

    if (_viewerManager) {
        _viewerManager->setOverlayThreshold(_overlayThreshold);
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
    setThreshold(normalizedThresholdFromSpin(value));
    if (!_suspendPersistence) {
        saveState();
    }
}
