#pragma once

#include <QHash>
#include <QElapsedTimer>
#include <QJsonArray>
#include <QJsonObject>
#include <QSet>
#include <QWidget>
#include <functional>

#include "SpiralServiceProfile.hpp"
#include "elements/VolumeSelector.hpp"

class QCheckBox;
class QComboBox;
class QDialog;
class QLabel;
class QLineEdit;
class QListWidget;
class QPushButton;
class QSpinBox;
class QDoubleSpinBox;
class QPlainTextEdit;
class QProgressBar;
class QSlider;
class QToolButton;
class QTimer;
class SpiralServiceManager;
class SpiralConfigProfileEditor;
class QFormLayout;

class SpiralPanel : public QWidget
{
    Q_OBJECT
public:
    explicit SpiralPanel(SpiralServiceManager* service, QWidget* parent = nullptr);
    void setVolumes(const QVector<VolumeSelector::VolumeOption>& volumes, const QString& selectedId);
    void setLossMapOptions(const QStringList& names);
    void setLossMapLegend(const QString& text);
    void setSessionExitGuard(
        std::function<void(std::function<void()>)> guard) { _sessionExitGuard = std::move(guard); }

signals:
    void volumeSelected(const QString& id);
    void visibilityChanged(const QString& category, bool visible);
    void runDiffChanged(bool visible);
    void lossMapChanged(const QString& name, qreal opacity);
    void windingRangeChanged(int minimum, int maximum);
    void surfaceIntersectionsChanged(bool shown);
    void surfaceIntersectionStrideChanged(int stride);
    void surfaceOverlapChanged(bool shown);
    void pythonOutputRequested();

private:
    QLineEdit* addPathRow(QFormLayout* form, const QString& key, const QString& label,
                          bool directory);
    void addPclItem(const QString& path, const QString& role, bool required = false);
    QJsonObject sessionRequest() const;
    QJsonObject influenceConfig() const;
    QJsonObject sessionAdvancedConfig() const;
    QJsonObject runAdvancedConfig() const;
    void applyTrackSamplingConfig(QJsonObject& config) const;
    void syncTrackSamplingControlsFromAdvanced();
    void writeTrackSamplingControlsToAdvanced();
    void updateTrackSamplingUi();
    void applySessionRunConfig(const QJsonObject& config, qint64 sessionGeneration);
    void synchronizeSession(const QJsonObject& request,
                            const QJsonObject& status);
    void applyResolution(const QJsonObject& resolution, bool force);
    void applyScrollSpec(const QJsonObject& spec);
    void updateStatus(const QJsonObject& status);
    QJsonObject normalizedReloadRequest(QJsonObject request) const;
    QString pendingRebuildStage() const;
    void refreshReloadRequired();
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
    void guardSessionExit(std::function<void()> action);

    SpiralServiceManager* _service = nullptr;
    QHash<QString, QLineEdit*> _paths;
    QHash<QString, QToolButton*> _pathBrowseButtons;
    QHash<QString, QCheckBox*> _visibilityChecks;
    QHash<QString, bool> _pathDirectories;
    QDialog* _displayDialog = nullptr;
    QSpinBox* _minimumDisplayedWinding = nullptr;
    QSpinBox* _maximumDisplayedWinding = nullptr;
    QCheckBox* _showSurfaceIntersections = nullptr;
    QComboBox* _lossMap = nullptr;
    QSlider* _lossMapOpacity = nullptr;
    QLabel* _lossMapLegend = nullptr;
    QSpinBox* _zBegin = nullptr;
    QSpinBox* _zEnd = nullptr;
    QSpinBox* _iterations = nullptr;
    QSpinBox* _legacyCheckpointStep = nullptr;
    QSpinBox* _renderVolumeScale = nullptr;
    // Read-only reports of what spiral-scroll.json specifies.
    QLabel* _scrollSummary = nullptr;
    QLabel* _lasagnaSummary = nullptr;
    QLineEdit* _runTag = nullptr;
    QLineEdit* _pclPath = nullptr;
    QListWidget* _pclList = nullptr;
    QComboBox* _pclRole = nullptr;
    QPushButton* _removePcl = nullptr;
    QPushButton* _addPclButton = nullptr;
    QToolButton* _browsePclButton = nullptr;
    QCheckBox* _savePngVisualizations = nullptr;
    QCheckBox* _trackLengthBinSampling = nullptr;
    QDoubleSpinBox* _trackShortWeight = nullptr;
    QDoubleSpinBox* _trackMediumWeight = nullptr;
    QDoubleSpinBox* _trackLongWeight = nullptr;
    QSpinBox* _maxTrackCrossings = nullptr;
    QCheckBox* _influenceEnabled = nullptr;
    QSpinBox* _influenceZ = nullptr;
    QDoubleSpinBox* _influenceWindings = nullptr;
    QSpinBox* _influenceThetaPct = nullptr;
    QSpinBox* _influenceDisableDtPct = nullptr;
    QDoubleSpinBox* _influenceAnchorWeight = nullptr;
    SpiralConfigProfileEditor* _advancedProfiles = nullptr;
    QPlainTextEdit* _advanced = nullptr;
    VolumeSelector* _volumeSelector = nullptr;
    QPushButton* _load = nullptr;
    QPushButton* _run = nullptr;
    QPushButton* _stop = nullptr;
    QPushButton* _save = nullptr;
    QPushButton* _downloadCheckpoint = nullptr;
    QPushButton* _loadCheckpoint = nullptr;
    QLineEdit* _datasetRoot = nullptr;
    QLineEdit* _outputRoot = nullptr;
    QLineEdit* _cacheRoot = nullptr;
    QWidget* _datasetRow = nullptr;
    QWidget* _outputRow = nullptr;
    QWidget* _cacheRow = nullptr;
    QLabel* _checkpointDownloadStatus = nullptr;
    QProgressBar* _checkpointDownloadProgress = nullptr;
    QTimer* _checkpointDownloadTimer = nullptr;
    QElapsedTimer _checkpointDownloadElapsed;
    QString _checkpointDownloadPhase;
    qint64 _checkpointBytesReceived = 0;
    qint64 _checkpointTotalBytes = 0;
    QLabel* _state = nullptr;
    QProgressBar* _previewProgress = nullptr;
    QLabel* _metrics = nullptr;
    QLabel* _warnings = nullptr;

    // Service section widgets
    QComboBox* _profileCombo = nullptr;
    QLineEdit* _endpointUrl = nullptr;
    QLineEdit* _sshDestination = nullptr;
    QSpinBox* _sshPort = nullptr;
    QLineEdit* _apiKey = nullptr;
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
    QJsonObject _loadedSessionRequest;
    QJsonObject _attachedAdvancedConfig;
    QJsonObject _defaultAdvancedConfig;
    QSet<QString> _runConfigKeys;
    QSet<QString> _runMutablePaths;
    // schema.model_stage_keys: the settings a rebuild can apply without
    // reloading the session's inputs.
    QSet<QString> _modelStageKeys;
    qint64 _advancedSessionGeneration = -1;

    QString _currentProfileId;
    QStringList _profileIds;
    bool _applyingResolution = false;
    bool _hasManualEdits = false;
    bool _hasSession = false;
    bool _reloadRequired = false;
    bool _sessionRunnable = false;
    bool _remoteMode = false;
    bool _connected = false;
    bool _previewTransferActive = false;
    bool _checkpointDownloadActive = false;
    QString _previewTransferText;
    // Last reported session lifecycle state; "Error" is the recovery case.
    QString _sessionState;
    int _ephemeralCount = 0;
    int _uncommittedCount = 0;
    std::function<void(std::function<void()>)> _sessionExitGuard;
    bool _runningGuardedExit = false;
};
