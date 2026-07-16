#pragma once

#include <QHash>
#include <QJsonArray>
#include <QJsonObject>
#include <QWidget>

#include "SpiralServiceProfile.hpp"
#include "elements/VolumeSelector.hpp"

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QPushButton;
class QSpinBox;
class QDoubleSpinBox;
class QPlainTextEdit;
class QToolButton;
class SpiralServiceManager;
class QFormLayout;

class SpiralPanel : public QWidget
{
    Q_OBJECT
public:
    explicit SpiralPanel(SpiralServiceManager* service, QWidget* parent = nullptr);
    void setVolumes(const QVector<VolumeSelector::VolumeOption>& volumes, const QString& selectedId);

signals:
    void volumeSelected(const QString& id);
    void visibilityChanged(const QString& category, bool visible);
    void windingRangeChanged(int minimum, int maximum);
    void surfaceIntersectionsChanged(bool shown);
    void pythonOutputRequested();

private:
    QLineEdit* addPathRow(QFormLayout* form, const QString& key, const QString& label,
                          bool directory);
    void addPclItem(const QString& path, const QString& role, bool required = false);
    QJsonObject sessionRequest() const;
    QJsonObject influenceConfig() const;
    QJsonObject durableAdvancedConfig() const;
    void applyResolution(const QJsonObject& resolution, bool force);
    void updateStatus(const QJsonObject& status);
    void markReloadRequired();
    void persist() const;
    void restore();

    // Service profiles
    void rebuildProfileCombo();
    void selectProfile(const QString& profileId);
    SpiralServiceProfile profileFromFields() const;
    void applyProfileFields(const SpiralServiceProfile& profile);
    void saveProfileList() const;
    void setRemoteMode(bool remote);
    void connectToSelectedProfile();
    QString formSettingsPrefix() const;

    SpiralServiceManager* _service = nullptr;
    QHash<QString, QLineEdit*> _paths;
    QHash<QString, QToolButton*> _pathBrowseButtons;
    QHash<QString, QCheckBox*> _visibilityChecks;
    QHash<QString, bool> _pathDirectories;
    QSpinBox* _minimumDisplayedWinding = nullptr;
    QSpinBox* _maximumDisplayedWinding = nullptr;
    QCheckBox* _showSurfaceIntersections = nullptr;
    QSpinBox* _zBegin = nullptr;
    QSpinBox* _zEnd = nullptr;
    QSpinBox* _iterations = nullptr;
    QSpinBox* _lasagnaScale = nullptr;
    QSpinBox* _legacyCheckpointStep = nullptr;
    QSpinBox* _renderVolumeScale = nullptr;
    QDoubleSpinBox* _voxelSize = nullptr;
    QLineEdit* _lasagnaGroup = nullptr;
    QLineEdit* _scrollName = nullptr;
    QLineEdit* _runTag = nullptr;
    QLineEdit* _pclPath = nullptr;
    QListWidget* _pclList = nullptr;
    QComboBox* _pclRole = nullptr;
    QPushButton* _removePcl = nullptr;
    QPushButton* _addPclButton = nullptr;
    QToolButton* _browsePclButton = nullptr;
    QComboBox* _outwardSense = nullptr;
    QComboBox* _storageBackend = nullptr;
    QCheckBox* _savePngVisualizations = nullptr;
    QCheckBox* _influenceEnabled = nullptr;
    QSpinBox* _influenceZ = nullptr;
    QDoubleSpinBox* _influenceWindings = nullptr;
    QSpinBox* _influenceThetaPct = nullptr;
    QSpinBox* _influenceDisableDtPct = nullptr;
    QDoubleSpinBox* _influenceAnchorWeight = nullptr;
    QPlainTextEdit* _advanced = nullptr;
    VolumeSelector* _volumeSelector = nullptr;
    QPushButton* _load = nullptr;
    QPushButton* _run = nullptr;
    QPushButton* _stop = nullptr;
    QPushButton* _save = nullptr;
    QPushButton* _downloadCheckpoint = nullptr;
    QPushButton* _refill = nullptr;
    QLabel* _state = nullptr;
    QLabel* _metrics = nullptr;
    QLabel* _warnings = nullptr;

    // Service section widgets
    QComboBox* _profileCombo = nullptr;
    QLineEdit* _endpointUrl = nullptr;
    QLineEdit* _sshDestination = nullptr;
    QSpinBox* _sshPort = nullptr;
    QLineEdit* _apiKey = nullptr;
    QLineEdit* _mapServiceRoot = nullptr;
    QLineEdit* _mapLocalRoot = nullptr;
    QLabel* _connectionStatus = nullptr;
    QPushButton* _connectButton = nullptr;
    QPushButton* _disconnectButton = nullptr;
    QWidget* _endpointRow = nullptr;
    QWidget* _sshRow = nullptr;
    QWidget* _apiKeyRow = nullptr;
    QWidget* _mappingRow = nullptr;

    // Ephemeral inputs
    QListWidget* _ephemeralList = nullptr;
    QPushButton* _commitInputs = nullptr;
    QPushButton* _removeInput = nullptr;
    QLabel* _commitHint = nullptr;
    QJsonArray _lastEphemeral;
    QJsonObject _loadedDurableAdvanced;

    QString _currentProfileId;
    QStringList _profileIds;
    QString _pendingDatasetRoot;
    bool _applyingResolution = false;
    bool _hasManualEdits = false;
    bool _hasSession = false;
    bool _reloadRequired = false;
    bool _remoteMode = false;
    bool _connected = false;
    int _ephemeralCount = 0;
    int _uncommittedCount = 0;
};
