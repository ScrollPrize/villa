#pragma once

#include <QHash>
#include <QJsonObject>
#include <QWidget>

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
    void pythonOutputRequested();

private:
    QLineEdit* addPathRow(QFormLayout* form, const QString& key, const QString& label,
                          bool directory);
    void addPclItem(const QString& path, const QString& role, bool required = false);
    QJsonObject sessionRequest() const;
    void applyResolution(const QJsonObject& resolution, bool force);
    void updateStatus(const QJsonObject& status);
    void markReloadRequired();
    void persist() const;
    void restore();

    SpiralServiceManager* _service = nullptr;
    QHash<QString, QLineEdit*> _paths;
    QHash<QString, QCheckBox*> _visibilityChecks;
    QHash<QString, bool> _pathDirectories;
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
    QComboBox* _outwardSense = nullptr;
    QComboBox* _storageBackend = nullptr;
    QCheckBox* _savePngVisualizations = nullptr;
    QPlainTextEdit* _advanced = nullptr;
    VolumeSelector* _volumeSelector = nullptr;
    QPushButton* _load = nullptr;
    QPushButton* _run = nullptr;
    QPushButton* _stop = nullptr;
    QPushButton* _save = nullptr;
    QLabel* _state = nullptr;
    QLabel* _metrics = nullptr;
    QLabel* _warnings = nullptr;
    QString _pendingDatasetRoot;
    bool _applyingResolution = false;
    bool _hasManualEdits = false;
    bool _hasSession = false;
    bool _reloadRequired = false;
};
