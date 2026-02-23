#include "SegmentationLasagnaPanel.hpp"

#include "LasagnaServiceManager.hpp"
#include "VCSettings.hpp"
#include "elements/CollapsibleSettingsGroup.hpp"

#include <QComboBox>
#include <QFutureWatcher>
#include <QtConcurrent/QtConcurrent>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QStackedWidget>
#include <QToolButton>
#include <QVBoxLayout>

#include <nlohmann/json.hpp>

#include <iostream>

SegmentationLasagnaPanel::SegmentationLasagnaPanel(
    const QString& settingsGroup, QWidget* parent)
    : QWidget(parent)
    , _settingsGroup(settingsGroup)
{
    auto* panelLayout = new QVBoxLayout(this);
    panelLayout->setContentsMargins(0, 0, 0, 0);
    panelLayout->setSpacing(0);

    _group = new CollapsibleSettingsGroup(tr("Lasagna Model"), this);
    auto* content = _group->contentWidget();

    // -- Connection mode --
    _group->addRow(tr("Connection:"), [&](QHBoxLayout* row) {
        _connectionCombo = new QComboBox(content);
        _connectionCombo->addItem(tr("Internal (local)"));
        _connectionCombo->addItem(tr("External (remote)"));
        row->addWidget(_connectionCombo, 1);
    }, tr("Internal launches a local Python process. External connects to a running service."));

    // -- External widgets: discovery + host/port --
    _externalWidget = new QWidget(content);
    auto* extLayout = new QVBoxLayout(_externalWidget);
    extLayout->setContentsMargins(0, 0, 0, 0);
    extLayout->setSpacing(4);

    // Discovery row
    auto* discRow = new QHBoxLayout();
    auto* discLabel = new QLabel(tr("Service:"), _externalWidget);
    discLabel->setFixedWidth(60);
    _discoveryCombo = new QComboBox(_externalWidget);
    _discoveryCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    _refreshBtn = new QToolButton(_externalWidget);
    _refreshBtn->setText(QStringLiteral("\u21BB"));  // ↻
    _refreshBtn->setToolTip(tr("Refresh discovered services"));
    discRow->addWidget(discLabel);
    discRow->addWidget(_discoveryCombo, 1);
    discRow->addWidget(_refreshBtn);
    extLayout->addLayout(discRow);

    // Host/port row (hidden when a discovered service is selected)
    _hostPortWidget = new QWidget(_externalWidget);
    auto* hostRow = new QHBoxLayout(_hostPortWidget);
    hostRow->setContentsMargins(0, 0, 0, 0);
    auto* hostLabel = new QLabel(tr("Host:"), _hostPortWidget);
    hostLabel->setFixedWidth(60);
    _hostEdit = new QLineEdit(_hostPortWidget);
    _hostEdit->setText(QStringLiteral("127.0.0.1"));
    _hostEdit->setPlaceholderText(QStringLiteral("127.0.0.1"));
    auto* portLabel = new QLabel(tr("Port:"), _hostPortWidget);
    _portEdit = new QLineEdit(_hostPortWidget);
    _portEdit->setText(QStringLiteral("9999"));
    _portEdit->setPlaceholderText(QStringLiteral("9999"));
    _portEdit->setFixedWidth(70);
    hostRow->addWidget(hostLabel);
    hostRow->addWidget(_hostEdit, 1);
    hostRow->addWidget(portLabel);
    hostRow->addWidget(_portEdit);
    extLayout->addWidget(_hostPortWidget);

    _group->contentLayout()->addWidget(_externalWidget);
    _externalWidget->setVisible(false);  // Start hidden (internal mode)

    // -- Mode: Re-optimize / New Model --
    _group->addRow(tr("Mode:"), [&](QHBoxLayout* row) {
        _modeCombo = new QComboBox(content);
        _modeCombo->addItem(tr("Re-optimize"));
        _modeCombo->addItem(tr("New Model"));
        row->addWidget(_modeCombo, 1);
    }, tr("Re-optimize refines an existing model. New Model creates a fresh model centered at the cursor."));

    // -- New model dimensions (visible only in New Model mode) --
    _newModelWidget = new QWidget(content);
    auto* dimLayout = new QHBoxLayout(_newModelWidget);
    dimLayout->setContentsMargins(0, 0, 0, 0);
    dimLayout->setSpacing(4);

    dimLayout->addWidget(new QLabel(tr("W:"), _newModelWidget));
    _widthSpin = new QSpinBox(_newModelWidget);
    _widthSpin->setRange(1, 999999);
    _widthSpin->setValue(2048);
    _widthSpin->setSingleStep(64);
    dimLayout->addWidget(_widthSpin, 1);

    dimLayout->addWidget(new QLabel(tr("H:"), _newModelWidget));
    _heightSpin = new QSpinBox(_newModelWidget);
    _heightSpin->setRange(1, 999999);
    _heightSpin->setValue(2048);
    _heightSpin->setSingleStep(64);
    dimLayout->addWidget(_heightSpin, 1);

    dimLayout->addWidget(new QLabel(tr("D:"), _newModelWidget));
    _depthSpin = new QSpinBox(_newModelWidget);
    _depthSpin->setRange(1, 999999);
    _depthSpin->setValue(2048);
    _depthSpin->setSingleStep(64);
    _depthSpin->setToolTip(tr("Depth in full-resolution voxels"));
    dimLayout->addWidget(_depthSpin, 1);

    _group->contentLayout()->addWidget(_newModelWidget);
    _newModelWidget->setVisible(false);  // Hidden until "New Model" selected

    // -- Data input (zarr) — stacked: file browse (page 0) or dataset combo (page 1) --
    _dataInputStack = new QStackedWidget(content);

    // Page 0: file browse
    auto* browseWidget = new QWidget(_dataInputStack);
    auto* browseLayout = new QHBoxLayout(browseWidget);
    browseLayout->setContentsMargins(0, 0, 0, 0);
    _dataInputEdit = new QLineEdit(browseWidget);
    _dataInputEdit->setPlaceholderText(tr("Path to input data (.zarr)"));
    _dataInputBrowse = new QToolButton(browseWidget);
    _dataInputBrowse->setText(QStringLiteral("..."));
    browseLayout->addWidget(_dataInputEdit, 1);
    browseLayout->addWidget(_dataInputBrowse);
    _dataInputStack->addWidget(browseWidget);

    // Page 1: dataset combo
    _datasetCombo = new QComboBox(_dataInputStack);
    _dataInputStack->addWidget(_datasetCombo);

    _dataInputStack->setCurrentIndex(0);  // Default: file browse

    _group->addRow(tr("Data:"), [&](QHBoxLayout* row) {
        row->addWidget(_dataInputStack, 1);
    }, tr("Input data zarr (e.g. s5_cos.zarr) required by the lasagna."));

    // -- Config file selector (dropdown + browse button) --
    _group->addRow(tr("Config:"), [&](QHBoxLayout* row) {
        _configFileCombo = new QComboBox(content);
        _configFileCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        _configFileBrowse = new QToolButton(content);
        _configFileBrowse->setText(QStringLiteral("..."));
        _configFileBrowse->setToolTip(tr("Browse for a JSON config file"));
        row->addWidget(_configFileCombo, 1);
        row->addWidget(_configFileBrowse);
    }, tr("JSON config file for the optimizer. File is read when optimization starts."));

    // -- Run / Stop / Stop Service buttons --
    auto* btnRow = new QHBoxLayout();
    _runBtn = new QPushButton(tr("Run Optimization"), content);
    _stopBtn = new QPushButton(tr("Stop"), content);
    _stopBtn->setEnabled(false);
    _stopServiceBtn = new QPushButton(tr("Stop Service"), content);
    _stopServiceBtn->setEnabled(false);
    btnRow->addWidget(_runBtn);
    btnRow->addWidget(_stopBtn);
    btnRow->addWidget(_stopServiceBtn);
    btnRow->addStretch(1);
    _group->contentLayout()->addLayout(btnRow);

    // -- Progress bar --
    _progressBar = new QProgressBar(content);
    _progressBar->setRange(0, 100);
    _progressBar->setValue(0);
    _progressBar->setTextVisible(true);
    _progressBar->setVisible(false);
    _group->contentLayout()->addWidget(_progressBar);

    // -- Progress label --
    _progressLabel = new QLabel(content);
    _progressLabel->setWordWrap(true);
    _progressLabel->setVisible(false);
    _group->contentLayout()->addWidget(_progressLabel);

    panelLayout->addWidget(_group);

    // -----------------------------------------------------------------------
    // Signal wiring
    // -----------------------------------------------------------------------

    // Mode combo (Re-optimize / New Model)
    connect(_modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &SegmentationLasagnaPanel::onLasagnaModeChanged);

    // Persist dimension changes
    connect(_widthSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) {
        writeSetting(QStringLiteral("lasagna_new_model_width"), v);
    });
    connect(_heightSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) {
        writeSetting(QStringLiteral("lasagna_new_model_height"), v);
    });
    connect(_depthSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) {
        writeSetting(QStringLiteral("lasagna_new_model_depth"), v);
    });

    // Connection mode
    connect(_connectionCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &SegmentationLasagnaPanel::onConnectionModeChanged);

    // External: refresh button
    connect(_refreshBtn, &QToolButton::clicked, this,
            &SegmentationLasagnaPanel::refreshDiscoveredServices);

    // External: discovered service selection
    connect(_discoveryCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &SegmentationLasagnaPanel::onDiscoveredServiceSelected);

    // External: host/port manual editing
    connect(_hostEdit, &QLineEdit::textChanged, this, [this](const QString& text) {
        _externalHost = text.trimmed();
        writeSetting(QStringLiteral("lasagna_external_host"), _externalHost);
    });
    connect(_portEdit, &QLineEdit::textChanged, this, [this](const QString& text) {
        _externalPort = text.trimmed().toInt();
        writeSetting(QStringLiteral("lasagna_external_port"), _externalPort);
    });

    // Dataset combo selection
    connect(_datasetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int index) {
        if (index < 0) return;
        QString path = _datasetCombo->currentData().toString();
        if (!path.isEmpty()) {
            _lasagnaDataInputPath = path;
            writeSetting(QStringLiteral("lasagna_data_input_path"), _lasagnaDataInputPath);
        }
    });

    // Datasets received from service
    connect(&LasagnaServiceManager::instance(), &LasagnaServiceManager::datasetsReceived,
            this, [this](const QJsonArray& datasets) {
        _datasetCombo->clear();
        if (datasets.isEmpty()) {
            _dataInputStack->setCurrentIndex(0);  // Fall back to file browse
            return;
        }
        for (const auto& val : datasets) {
            QJsonObject ds = val.toObject();
            QString name = ds[QStringLiteral("name")].toString();
            QString path = ds[QStringLiteral("path")].toString();
            _datasetCombo->addItem(name, path);
        }
        _dataInputStack->setCurrentIndex(1);  // Show dataset combo
    });

    // Data input
    connect(_dataInputEdit, &QLineEdit::textChanged, this, [this](const QString& text) {
        _lasagnaDataInputPath = text.trimmed();
        writeSetting(QStringLiteral("lasagna_data_input_path"), _lasagnaDataInputPath);
    });
    connect(_dataInputBrowse, &QToolButton::clicked, this, [this]() {
        QString initial = _lasagnaDataInputPath.isEmpty()
            ? QDir::homePath() : QFileInfo(_lasagnaDataInputPath).absolutePath();
        QString path = QFileDialog::getExistingDirectory(
            this, tr("Select input data (.zarr directory)"), initial);
        if (!path.isEmpty()) {
            _lasagnaDataInputPath = path;
            _dataInputEdit->setText(path);
        }
    });

    // Config file combo selection
    connect(_configFileCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int index) {
        if (_restoringSettings || index < 0) return;
        QString path = _configFileCombo->currentData().toString();
        if (!path.isEmpty()) {
            _lasagnaConfigFilePath = path;
            writeSetting(QStringLiteral("lasagna_config_file_path"), _lasagnaConfigFilePath);
        }
    });

    // Config file browse button
    connect(_configFileBrowse, &QToolButton::clicked, this, [this]() {
        QString initial = _lasagnaConfigFilePath.isEmpty()
            ? QDir::homePath() : QFileInfo(_lasagnaConfigFilePath).absolutePath();
        QString path = QFileDialog::getOpenFileName(
            this, tr("Select optimizer config JSON file"), initial,
            tr("JSON files (*.json);;All files (*)"));
        if (!path.isEmpty()) {
            _lasagnaConfigFilePath = path;
            writeSetting(QStringLiteral("lasagna_config_file_path"), _lasagnaConfigFilePath);
            QFileInfo fi(path);
            populateConfigFileCombo(fi.absolutePath(), fi.fileName());
        }
    });

    // Run button
    connect(_runBtn, &QPushButton::clicked, this, [this]() {
        // Validate config file on run
        if (_lasagnaConfigFilePath.isEmpty()) {
            _progressLabel->setText(tr("No config file selected."));
            _progressLabel->setStyleSheet(QStringLiteral("color: #c0392b;"));
            _progressLabel->setVisible(true);
            return;
        }
        if (!QFileInfo::exists(_lasagnaConfigFilePath)) {
            _progressLabel->setText(tr("Config file not found: %1").arg(_lasagnaConfigFilePath));
            _progressLabel->setStyleSheet(QStringLiteral("color: #c0392b;"));
            _progressLabel->setVisible(true);
            return;
        }

        // If in external mode and not yet connected, connect first
        if (_connectionMode == 1) {
            auto& mgr = LasagnaServiceManager::instance();
            if (!mgr.isExternal() || !mgr.isRunning()) {
                mgr.connectToExternal(_externalHost, _externalPort);
                // Wait for serviceStarted signal to emit lasagnaOptimizeRequested
                auto* conn = new QMetaObject::Connection;
                auto* errConn = new QMetaObject::Connection;
                *conn = connect(&mgr, &LasagnaServiceManager::serviceStarted, this,
                    [this, conn, errConn]() {
                        QObject::disconnect(*conn);
                        QObject::disconnect(*errConn);
                        delete conn;
                        delete errConn;
                        emit lasagnaOptimizeRequested();
                    });
                *errConn = connect(&mgr, &LasagnaServiceManager::serviceError, this,
                    [this, conn, errConn](const QString&) {
                        QObject::disconnect(*conn);
                        QObject::disconnect(*errConn);
                        delete conn;
                        delete errConn;
                    });
                return;
            }
        }

        emit lasagnaOptimizeRequested();
    });

    // Stop button
    connect(_stopBtn, &QPushButton::clicked, this, [this]() {
        emit lasagnaStopRequested();
    });

    // Stop Service button — kills the Python process entirely
    connect(_stopServiceBtn, &QPushButton::clicked, this, []() {
        LasagnaServiceManager::instance().stopService();
    });

    // Connect to service manager signals
    auto& mgr = LasagnaServiceManager::instance();
    connect(&mgr, &LasagnaServiceManager::statusMessage, this, [this](const QString& msg) {
        if (_progressLabel) {
            _progressLabel->setText(msg);
            _progressLabel->setStyleSheet(QString());
            _progressLabel->setVisible(true);
        }
        emit lasagnaStatusMessage(msg);
    });
    connect(&mgr, &LasagnaServiceManager::serviceStarted, this, [this]() {
        if (_progressLabel) {
            _progressLabel->setText(tr("Service running"));
            _progressLabel->setStyleSheet(QStringLiteral("color: #27ae60;"));
        }
        if (_stopServiceBtn) _stopServiceBtn->setEnabled(true);
    });
    connect(&mgr, &LasagnaServiceManager::serviceStopped, this, [this]() {
        if (_progressBar) _progressBar->setVisible(false);
        if (_progressLabel) {
            _progressLabel->setText(tr("Service stopped"));
            _progressLabel->setStyleSheet(QString());
        }
        if (_stopBtn) _stopBtn->setEnabled(false);
        if (_stopServiceBtn) _stopServiceBtn->setEnabled(false);
        if (_runBtn) _runBtn->setEnabled(true);
    });
    connect(&mgr, &LasagnaServiceManager::serviceError, this, [this](const QString& err) {
        std::cerr << "[lasagna] service error: " << err.toStdString() << std::endl;
        if (_progressLabel) {
            _progressLabel->setText(tr("Error: %1").arg(err));
            _progressLabel->setStyleSheet(QStringLiteral("color: #c0392b;"));
            _progressLabel->setVisible(true);
        }
    });
    connect(&mgr, &LasagnaServiceManager::optimizationStarted, this, [this]() {
        if (_stopBtn) _stopBtn->setEnabled(true);
        if (_runBtn) _runBtn->setEnabled(false);
        if (_progressLabel) {
            _progressLabel->setText(tr("Optimization started..."));
            _progressLabel->setStyleSheet(QString());
            _progressLabel->setVisible(true);
        }
    });
    connect(&mgr, &LasagnaServiceManager::optimizationProgress, this,
            [this](const QString& /*stage*/, int /*step*/, int /*total*/, double loss,
                   double stageProgress, double overallProgress,
                   const QString& stageName) {
        if (_progressBar) {
            _progressBar->setRange(0, 1000);
            _progressBar->setValue(static_cast<int>(overallProgress * 1000.0));
            _progressBar->setFormat(
                tr("Overall: %1%").arg(overallProgress * 100.0, 0, 'f', 1));
            _progressBar->setVisible(true);
        }
        if (_progressLabel) {
            _progressLabel->setText(
                tr("Stage: %1 (%2%)  |  Overall: %3%  |  Loss: %4")
                    .arg(stageName.isEmpty() ? QStringLiteral("...") : stageName)
                    .arg(stageProgress * 100.0, 0, 'f', 1)
                    .arg(overallProgress * 100.0, 0, 'f', 1)
                    .arg(loss, 0, 'g', 5));
            _progressLabel->setStyleSheet(QString());
            _progressLabel->setVisible(true);
        }
    });
    connect(&mgr, &LasagnaServiceManager::optimizationFinished, this,
            [this](const QString& outputDir) {
        if (_stopBtn) _stopBtn->setEnabled(false);
        if (_runBtn) _runBtn->setEnabled(true);
        if (_progressBar) _progressBar->setVisible(false);
        if (_progressLabel) {
            _progressLabel->setText(tr("Optimization finished. Output: %1").arg(outputDir));
            _progressLabel->setStyleSheet(QStringLiteral("color: #27ae60;"));
            _progressLabel->setVisible(true);
        }
    });
    connect(&mgr, &LasagnaServiceManager::optimizationError, this,
            [this](const QString& err) {
        std::cerr << "[lasagna] optimization error: " << err.toStdString() << std::endl;
        if (_stopBtn) _stopBtn->setEnabled(false);
        if (_runBtn) _runBtn->setEnabled(true);
        if (_progressBar) _progressBar->setVisible(false);
        if (_progressLabel) {
            _progressLabel->setText(tr("Optimization error: %1").arg(err));
            _progressLabel->setStyleSheet(QStringLiteral("color: #c0392b;"));
            _progressLabel->setVisible(true);
        }
    });

    // Run service discovery once on startup (in background thread)
    auto* watcher = new QFutureWatcher<QJsonArray>(this);
    connect(watcher, &QFutureWatcher<QJsonArray>::finished, this, [this, watcher]() {
        QJsonArray services = watcher->result();
        watcher->deleteLater();
        if (!_discoveryCombo) return;
        _discoveryCombo->clear();
        _discoveryCombo->addItem(tr("(manual entry)"));
        for (const auto& val : services) {
            QJsonObject svc = val.toObject();
            QString host = svc[QStringLiteral("host")].toString();
            int port = svc[QStringLiteral("port")].toInt();
            QString label;
            if (svc.contains(QStringLiteral("name"))) {
                label = QStringLiteral("%1 (%2:%3)")
                    .arg(svc[QStringLiteral("name")].toString())
                    .arg(host).arg(port);
            } else {
                int pid = svc[QStringLiteral("pid")].toInt();
                label = QStringLiteral("%1:%2 (pid %3)").arg(host).arg(port).arg(pid);
            }
            _discoveryCombo->addItem(label, QJsonDocument(svc).toJson(QJsonDocument::Compact));
        }
    });
    watcher->setFuture(QtConcurrent::run([]() {
        return LasagnaServiceManager::discoverServices();
    }));
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

void SegmentationLasagnaPanel::writeSetting(const QString& key, const QVariant& value)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(_settingsGroup);
    settings.setValue(key, value);
    settings.endGroup();
}

void SegmentationLasagnaPanel::restoreSettings(QSettings& settings)
{
    _restoringSettings = true;

    _lasagnaDataInputPath = settings.value(QStringLiteral("lasagna_data_input_path"), QString()).toString();
    _lasagnaConfigFilePath = settings.value(QStringLiteral("lasagna_config_file_path"), QString()).toString();

    _lasagnaMode = settings.value(QStringLiteral("lasagna_mode"), 0).toInt();
    if (_modeCombo) {
        _modeCombo->setCurrentIndex(_lasagnaMode);
    }
    if (_newModelWidget) {
        _newModelWidget->setVisible(_lasagnaMode == 1);
    }
    if (_widthSpin) {
        const QSignalBlocker b(_widthSpin);
        _widthSpin->setValue(settings.value(QStringLiteral("lasagna_new_model_width"), 2048).toInt());
    }
    if (_heightSpin) {
        const QSignalBlocker b(_heightSpin);
        _heightSpin->setValue(settings.value(QStringLiteral("lasagna_new_model_height"), 2048).toInt());
    }
    if (_depthSpin) {
        const QSignalBlocker b(_depthSpin);
        _depthSpin->setValue(settings.value(QStringLiteral("lasagna_new_model_depth"), 2048).toInt());
    }

    _connectionMode = settings.value(QStringLiteral("lasagna_connection_mode"), 0).toInt();
    _externalHost = settings.value(QStringLiteral("lasagna_external_host"),
                                   QStringLiteral("127.0.0.1")).toString();
    _externalPort = settings.value(QStringLiteral("lasagna_external_port"), 9999).toInt();

    if (_connectionCombo) {
        _connectionCombo->setCurrentIndex(_connectionMode);
    }
    if (_hostEdit) {
        const QSignalBlocker b(_hostEdit);
        _hostEdit->setText(_externalHost);
    }
    if (_portEdit) {
        const QSignalBlocker b(_portEdit);
        _portEdit->setText(QString::number(_externalPort));
    }
    updateConnectionWidgets();

    // Populate config file combo from saved path
    if (!_lasagnaConfigFilePath.isEmpty()) {
        QFileInfo fi(_lasagnaConfigFilePath);
        if (fi.exists()) {
            populateConfigFileCombo(fi.absolutePath(), fi.fileName());
        } else {
            // File gone — just show the name as-is
            if (_configFileCombo) {
                _configFileCombo->clear();
                _configFileCombo->addItem(fi.fileName(), _lasagnaConfigFilePath);
            }
        }
    }

    const bool expanded = settings.value(
        QStringLiteral("group_lasagna_expanded"), false).toBool();
    if (_group) {
        _group->setExpanded(expanded);
    }

    _restoringSettings = false;
}

void SegmentationLasagnaPanel::syncUiState(bool /*editingEnabled*/, bool optimizing)
{
    if (_dataInputEdit) {
        const QSignalBlocker b(_dataInputEdit);
        _dataInputEdit->setText(_lasagnaDataInputPath);
    }

    if (_runBtn) _runBtn->setEnabled(!optimizing);
    if (_stopBtn) _stopBtn->setEnabled(optimizing);
}

// ---------------------------------------------------------------------------
// Setters
// ---------------------------------------------------------------------------

void SegmentationLasagnaPanel::setLasagnaDataInputPath(const QString& path)
{
    if (_lasagnaDataInputPath == path) return;
    _lasagnaDataInputPath = path;
    writeSetting(QStringLiteral("lasagna_data_input_path"), _lasagnaDataInputPath);
    if (_dataInputEdit) {
        const QSignalBlocker b(_dataInputEdit);
        _dataInputEdit->setText(path);
    }
}

// ---------------------------------------------------------------------------
// Config file selector
// ---------------------------------------------------------------------------

void SegmentationLasagnaPanel::populateConfigFileCombo(
    const QString& dir, const QString& selectName)
{
    if (!_configFileCombo) return;

    const QSignalBlocker b(_configFileCombo);
    _configFileCombo->clear();

    QDir d(dir);
    QStringList jsonFiles = d.entryList(
        QStringList{QStringLiteral("*.json")}, QDir::Files, QDir::Name);

    int selectIndex = -1;
    for (int i = 0; i < jsonFiles.size(); ++i) {
        QString fullPath = d.absoluteFilePath(jsonFiles[i]);
        _configFileCombo->addItem(jsonFiles[i], fullPath);
        if (jsonFiles[i] == selectName) {
            selectIndex = i;
        }
    }

    if (selectIndex >= 0) {
        _configFileCombo->setCurrentIndex(selectIndex);
        _lasagnaConfigFilePath = _configFileCombo->currentData().toString();
    } else if (_configFileCombo->count() > 0) {
        _configFileCombo->setCurrentIndex(0);
        _lasagnaConfigFilePath = _configFileCombo->currentData().toString();
    }
}

// ---------------------------------------------------------------------------
// Config JSON — reads file from disk on demand
// ---------------------------------------------------------------------------

QString SegmentationLasagnaPanel::lasagnaConfigText() const
{
    if (_lasagnaConfigFilePath.isEmpty()) return {};
    QFile f(_lasagnaConfigFilePath);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return {};
    return QString::fromUtf8(f.readAll());
}

std::optional<nlohmann::json> SegmentationLasagnaPanel::lasagnaConfigJson() const
{
    QString text = lasagnaConfigText().trimmed();
    if (text.isEmpty()) return std::nullopt;

    try {
        QByteArray utf8 = text.toUtf8();
        nlohmann::json parsed = nlohmann::json::parse(
            utf8.constData(), utf8.constData() + utf8.size());
        if (parsed.is_object()) return parsed;
    } catch (...) {}

    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Lasagna mode (Re-optimize / New Model)
// ---------------------------------------------------------------------------

int SegmentationLasagnaPanel::newModelWidth() const
{
    return _widthSpin ? _widthSpin->value() : 2048;
}

int SegmentationLasagnaPanel::newModelHeight() const
{
    return _heightSpin ? _heightSpin->value() : 2048;
}

int SegmentationLasagnaPanel::newModelDepth() const
{
    return _depthSpin ? _depthSpin->value() : 2048;
}

void SegmentationLasagnaPanel::onLasagnaModeChanged(int index)
{
    if (_restoringSettings) return;
    _lasagnaMode = index;
    writeSetting(QStringLiteral("lasagna_mode"), _lasagnaMode);
    if (_newModelWidget) {
        _newModelWidget->setVisible(index == 1);  // Show for "New Model"
    }
}

// ---------------------------------------------------------------------------
// Connection mode
// ---------------------------------------------------------------------------

void SegmentationLasagnaPanel::onConnectionModeChanged(int index)
{
    if (_restoringSettings) return;
    _connectionMode = index;
    writeSetting(QStringLiteral("lasagna_connection_mode"), _connectionMode);
    updateConnectionWidgets();

    // If switching to external, disconnect any internal service
    if (_connectionMode == 1) {
        auto& mgr = LasagnaServiceManager::instance();
        if (!mgr.isExternal() && mgr.isRunning()) {
            mgr.stopService();
        }
    }
}

void SegmentationLasagnaPanel::updateConnectionWidgets()
{
    bool external = (_connectionMode == 1);

    // Show/hide external widgets (discovery + host/port)
    if (_externalWidget) _externalWidget->setVisible(external);

    // Reset data input stack when switching to internal
    if (!external) {
        if (_dataInputStack) _dataInputStack->setCurrentIndex(0);
    }
}

void SegmentationLasagnaPanel::refreshDiscoveredServices()
{
    if (!_discoveryCombo) return;

    _discoveryCombo->clear();
    _discoveryCombo->addItem(tr("(manual entry)"));

    QJsonArray services = LasagnaServiceManager::discoverServices();
    for (const auto& val : services) {
        QJsonObject svc = val.toObject();
        QString host = svc[QStringLiteral("host")].toString();
        int port = svc[QStringLiteral("port")].toInt();

        // Build a descriptive label from whatever info is available
        QString label;
        if (svc.contains(QStringLiteral("name"))) {
            label = QStringLiteral("%1 (%2:%3)")
                .arg(svc[QStringLiteral("name")].toString())
                .arg(host).arg(port);
        } else {
            int pid = svc[QStringLiteral("pid")].toInt();
            label = QStringLiteral("%1:%2 (pid %3)").arg(host).arg(port).arg(pid);
        }
        _discoveryCombo->addItem(label, QJsonDocument(svc).toJson(QJsonDocument::Compact));
    }
}

void SegmentationLasagnaPanel::onDiscoveredServiceSelected(int index)
{
    // Show host/port only for manual entry (index 0)
    if (_hostPortWidget) {
        _hostPortWidget->setVisible(index <= 0);
    }

    if (index <= 0) return;  // "(manual entry)" or invalid
    if (!_discoveryCombo) return;

    QByteArray data = _discoveryCombo->currentData().toByteArray();
    QJsonObject svc = QJsonDocument::fromJson(data).object();

    QString host = svc[QStringLiteral("host")].toString();
    int port = svc[QStringLiteral("port")].toInt();

    if (_hostEdit) {
        _hostEdit->setText(host);
    }
    if (_portEdit) {
        _portEdit->setText(QString::number(port));
    }

    _externalHost = host;
    _externalPort = port;

    // Auto-connect; fetch datasets once connected
    auto& mgr = LasagnaServiceManager::instance();
    auto* conn = new QMetaObject::Connection;
    auto* errConn = new QMetaObject::Connection;
    *conn = connect(&mgr, &LasagnaServiceManager::serviceStarted, this,
        [this, conn, errConn]() {
            QObject::disconnect(*conn);
            QObject::disconnect(*errConn);
            delete conn;
            delete errConn;
            LasagnaServiceManager::instance().fetchDatasets();
        });
    *errConn = connect(&mgr, &LasagnaServiceManager::serviceError, this,
        [conn, errConn](const QString&) {
            QObject::disconnect(*conn);
            QObject::disconnect(*errConn);
            delete conn;
            delete errConn;
        });
    mgr.connectToExternal(host, port);
}
