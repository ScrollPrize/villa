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

    // Outer frame to visually group the whole lasagna panel
    auto* frame = new QFrame(this);
    frame->setFrameShape(QFrame::StyledPanel);
    auto* frameLayout = new QVBoxLayout(frame);
    frameLayout->setContentsMargins(4, 4, 4, 4);
    frameLayout->setSpacing(2);

    // =======================================================================
    // Connection section
    // =======================================================================
    _connectionGroup = new CollapsibleSettingsGroup(tr("Solver Connection"), frame);
    auto* connContent = _connectionGroup->contentWidget();

    // -- Connection mode --
    _connectionGroup->addRow(tr("Connection:"), [&](QHBoxLayout* row) {
        _connectionCombo = new QComboBox(connContent);
        _connectionCombo->addItem(tr("Internal (local)"));
        _connectionCombo->addItem(tr("External (remote)"));
        row->addWidget(_connectionCombo, 1);
    }, tr("Internal launches a local Python process. External connects to a running service."));

    // -- External widgets: discovery + host/port --
    _externalWidget = new QWidget(connContent);
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

    _connectionGroup->contentLayout()->addWidget(_externalWidget);
    _externalWidget->setVisible(false);  // Start hidden (internal mode)

    // -- Data input (zarr) — stacked: file browse (page 0) or dataset combo (page 1) --
    _dataInputStack = new QStackedWidget(connContent);

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

    _connectionGroup->addRow(tr("Data:"), [&](QHBoxLayout* row) {
        row->addWidget(_dataInputStack, 1);
    }, tr("Input data zarr (e.g. s5_cos.zarr) required by the lasagna."));

    panelLayout->addWidget(_connectionGroup);

    // =======================================================================
    // New Model button + settings section
    // =======================================================================
    _newModelBtn = new QPushButton(tr("New Model"), this);
    panelLayout->addWidget(_newModelBtn);

    _newModelGroup = new CollapsibleSettingsGroup(tr("New Model"), this);
    auto* nmContent = _newModelGroup->contentWidget();

    // Config file row
    _newModelGroup->addRow(tr("Config:"), [&](QHBoxLayout* row) {
        _newModelConfigCombo = new QComboBox(nmContent);
        _newModelConfigCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        _newModelConfigBrowse = new QToolButton(nmContent);
        _newModelConfigBrowse->setText(QStringLiteral("..."));
        _newModelConfigBrowse->setToolTip(tr("Browse for a JSON config file"));
        row->addWidget(_newModelConfigCombo, 1);
        row->addWidget(_newModelConfigBrowse);
    }, tr("JSON config file for new model optimization."));

    // Dimensions row
    {
        auto* dimWidget = new QWidget(nmContent);
        auto* dimLayout = new QHBoxLayout(dimWidget);
        dimLayout->setContentsMargins(0, 0, 0, 0);
        dimLayout->setSpacing(4);

        dimLayout->addWidget(new QLabel(tr("W:"), dimWidget));
        _widthSpin = new QSpinBox(dimWidget);
        _widthSpin->setRange(1, 999999);
        _widthSpin->setValue(2048);
        _widthSpin->setSingleStep(64);
        dimLayout->addWidget(_widthSpin, 1);

        dimLayout->addWidget(new QLabel(tr("H:"), dimWidget));
        _heightSpin = new QSpinBox(dimWidget);
        _heightSpin->setRange(1, 999999);
        _heightSpin->setValue(2048);
        _heightSpin->setSingleStep(64);
        dimLayout->addWidget(_heightSpin, 1);

        dimLayout->addWidget(new QLabel(tr("D:"), dimWidget));
        _depthSpin = new QSpinBox(dimWidget);
        _depthSpin->setRange(1, 999999);
        _depthSpin->setValue(2048);
        _depthSpin->setSingleStep(64);
        _depthSpin->setToolTip(tr("Depth in full-resolution voxels"));
        dimLayout->addWidget(_depthSpin, 1);

        _newModelGroup->contentLayout()->addWidget(dimWidget);
    }

    // Seed point row
    {
        auto* seedWidget = new QWidget(nmContent);
        auto* seedLayout = new QHBoxLayout(seedWidget);
        seedLayout->setContentsMargins(0, 0, 0, 0);
        seedLayout->setSpacing(4);

        seedLayout->addWidget(new QLabel(tr("Seed:"), seedWidget));
        _seedEdit = new QLineEdit(seedWidget);
        _seedEdit->setPlaceholderText(tr("auto (center)"));
        _seedEdit->setToolTip(tr("Dilation seed point in full-res voxel coords: X, Y, Z"));
        seedLayout->addWidget(_seedEdit, 1);

        _seedFromFocusBtn = new QPushButton(tr("Focus"), seedWidget);
        _seedFromFocusBtn->setToolTip(tr("Use current focus point as seed"));
        seedLayout->addWidget(_seedFromFocusBtn);

        _newModelGroup->contentLayout()->addWidget(seedWidget);
    }

    // Output name row
    {
        auto* nameWidget = new QWidget(nmContent);
        auto* nameLayout = new QHBoxLayout(nameWidget);
        nameLayout->setContentsMargins(0, 0, 0, 0);
        nameLayout->setSpacing(4);
        nameLayout->addWidget(new QLabel(tr("Name:"), nameWidget));
        _outputNameEdit = new QLineEdit(nameWidget);
        _outputNameEdit->setPlaceholderText(tr("new_model"));
        _outputNameEdit->setToolTip(tr("Output name prefix (auto-versioned, e.g. mysheet → mysheet_v001.tifxyz)"));
        nameLayout->addWidget(_outputNameEdit, 1);
        _newModelGroup->contentLayout()->addWidget(nameWidget);
    }

    panelLayout->addWidget(_newModelGroup);

    // =======================================================================
    // Re-optimize button + settings section
    // =======================================================================
    _reoptBtn = new QPushButton(tr("Re-optimize"), this);
    panelLayout->addWidget(_reoptBtn);

    _reoptGroup = new CollapsibleSettingsGroup(tr("Re-optimize"), this);
    auto* reoptContent = _reoptGroup->contentWidget();

    // Config file row
    _reoptGroup->addRow(tr("Config:"), [&](QHBoxLayout* row) {
        _reoptConfigCombo = new QComboBox(reoptContent);
        _reoptConfigCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        _reoptConfigBrowse = new QToolButton(reoptContent);
        _reoptConfigBrowse->setText(QStringLiteral("..."));
        _reoptConfigBrowse->setToolTip(tr("Browse for a JSON config file"));
        row->addWidget(_reoptConfigCombo, 1);
        row->addWidget(_reoptConfigBrowse);
    }, tr("JSON config file for re-optimization."));

    panelLayout->addWidget(_reoptGroup);

    // =======================================================================
    // Shared bottom area — stop buttons, progress
    // =======================================================================
    auto* btnRow = new QHBoxLayout();
    _stopBtn = new QPushButton(tr("Stop"), this);
    _stopBtn->setEnabled(false);
    _stopServiceBtn = new QPushButton(tr("Stop Service"), this);
    _stopServiceBtn->setEnabled(false);
    btnRow->addWidget(_stopBtn);
    btnRow->addWidget(_stopServiceBtn);
    btnRow->addStretch(1);
    panelLayout->addLayout(btnRow);

    _progressBar = new QProgressBar(this);
    _progressBar->setRange(0, 100);
    _progressBar->setValue(0);
    _progressBar->setTextVisible(true);
    _progressBar->setVisible(false);
    panelLayout->addWidget(_progressBar);

    _progressLabel = new QLabel(this);
    _progressLabel->setWordWrap(true);
    _progressLabel->setVisible(false);
    panelLayout->addWidget(_progressLabel);

    // -----------------------------------------------------------------------
    // Signal wiring
    // -----------------------------------------------------------------------

    // -- Connection --
    connect(_connectionCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &SegmentationLasagnaPanel::onConnectionModeChanged);

    connect(_refreshBtn, &QToolButton::clicked, this,
            &SegmentationLasagnaPanel::refreshDiscoveredServices);

    connect(_discoveryCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &SegmentationLasagnaPanel::onDiscoveredServiceSelected);

    connect(_hostEdit, &QLineEdit::textChanged, this, [this](const QString& text) {
        _externalHost = text.trimmed();
        writeSetting(QStringLiteral("lasagna_external_host"), _externalHost);
    });
    connect(_portEdit, &QLineEdit::textChanged, this, [this](const QString& text) {
        _externalPort = text.trimmed().toInt();
        writeSetting(QStringLiteral("lasagna_external_port"), _externalPort);
    });

    // -- Data input --
    connect(_datasetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int index) {
        if (index < 0) return;
        QString path = _datasetCombo->currentData().toString();
        if (!path.isEmpty()) {
            _lasagnaDataInputPath = path;
            writeSetting(QStringLiteral("lasagna_data_input_path"), _lasagnaDataInputPath);
        }
    });

    connect(&LasagnaServiceManager::instance(), &LasagnaServiceManager::datasetsReceived,
            this, [this](const QJsonArray& datasets) {
        _datasetCombo->clear();
        if (datasets.isEmpty()) {
            _dataInputStack->setCurrentIndex(0);
            return;
        }
        for (const auto& val : datasets) {
            QJsonObject ds = val.toObject();
            QString name = ds[QStringLiteral("name")].toString();
            QString path = ds[QStringLiteral("path")].toString();
            _datasetCombo->addItem(name, path);
        }
        _dataInputStack->setCurrentIndex(1);
    });

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

    // -- New model settings persistence --
    connect(_widthSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) {
        writeSetting(QStringLiteral("lasagna_new_model_width"), v);
    });
    connect(_heightSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) {
        writeSetting(QStringLiteral("lasagna_new_model_height"), v);
    });
    connect(_depthSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) {
        writeSetting(QStringLiteral("lasagna_new_model_depth"), v);
    });
    connect(_seedEdit, &QLineEdit::textChanged, this, [this](const QString& text) {
        writeSetting(QStringLiteral("lasagna_seed_point"), text.trimmed());
    });
    connect(_outputNameEdit, &QLineEdit::textChanged, this, [this](const QString& text) {
        writeSetting(QStringLiteral("lasagna_output_name"), text.trimmed());
    });
    connect(_seedFromFocusBtn, &QPushButton::clicked, this,
            &SegmentationLasagnaPanel::seedFromFocusRequested);

    // -- New model config combo --
    connect(_newModelConfigCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int index) {
        if (_restoringSettings || index < 0) return;
        QString path = _newModelConfigCombo->currentData().toString();
        if (!path.isEmpty()) {
            _newModelConfigFilePath = path;
            writeSetting(QStringLiteral("lasagna_new_model_config_file_path"), _newModelConfigFilePath);
        }
    });
    connect(_newModelConfigBrowse, &QToolButton::clicked, this, [this]() {
        QString initial = _newModelConfigFilePath.isEmpty()
            ? QDir::homePath() : QFileInfo(_newModelConfigFilePath).absolutePath();
        QString path = QFileDialog::getOpenFileName(
            this, tr("Select optimizer config JSON file"), initial,
            tr("JSON files (*.json);;All files (*)"));
        if (!path.isEmpty()) {
            _newModelConfigFilePath = path;
            writeSetting(QStringLiteral("lasagna_new_model_config_file_path"), _newModelConfigFilePath);
            QFileInfo fi(path);
            populateConfigCombo(_newModelConfigCombo, fi.absolutePath(), fi.fileName(), _newModelConfigFilePath);
        }
    });

    // -- Re-optimize config combo --
    connect(_reoptConfigCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int index) {
        if (_restoringSettings || index < 0) return;
        QString path = _reoptConfigCombo->currentData().toString();
        if (!path.isEmpty()) {
            _reoptConfigFilePath = path;
            writeSetting(QStringLiteral("lasagna_reopt_config_file_path"), _reoptConfigFilePath);
        }
    });
    connect(_reoptConfigBrowse, &QToolButton::clicked, this, [this]() {
        QString initial = _reoptConfigFilePath.isEmpty()
            ? QDir::homePath() : QFileInfo(_reoptConfigFilePath).absolutePath();
        QString path = QFileDialog::getOpenFileName(
            this, tr("Select optimizer config JSON file"), initial,
            tr("JSON files (*.json);;All files (*)"));
        if (!path.isEmpty()) {
            _reoptConfigFilePath = path;
            writeSetting(QStringLiteral("lasagna_reopt_config_file_path"), _reoptConfigFilePath);
            QFileInfo fi(path);
            populateConfigCombo(_reoptConfigCombo, fi.absolutePath(), fi.fileName(), _reoptConfigFilePath);
        }
    });

    // -- Action buttons --
    connect(_newModelBtn, &QPushButton::clicked, this, [this]() {
        _lasagnaMode = 1;
        triggerOptimization();
    });
    connect(_reoptBtn, &QPushButton::clicked, this, [this]() {
        _lasagnaMode = 0;
        triggerOptimization();
    });

    // -- Stop buttons --
    connect(_stopBtn, &QPushButton::clicked, this, [this]() {
        emit lasagnaStopRequested();
    });
    connect(_stopServiceBtn, &QPushButton::clicked, this, []() {
        LasagnaServiceManager::instance().stopService();
    });

    // -- Expand state persistence --
    connect(_connectionGroup, &CollapsibleSettingsGroup::toggled, this, [this](bool expanded) {
        writeSetting(QStringLiteral("group_lasagna_connection_expanded"), expanded);
    });
    connect(_newModelGroup, &CollapsibleSettingsGroup::toggled, this, [this](bool expanded) {
        writeSetting(QStringLiteral("group_lasagna_new_model_expanded"), expanded);
    });
    connect(_reoptGroup, &CollapsibleSettingsGroup::toggled, this, [this](bool expanded) {
        writeSetting(QStringLiteral("group_lasagna_reopt_expanded"), expanded);
    });

    // -----------------------------------------------------------------------
    // Service manager signals
    // -----------------------------------------------------------------------
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
        if (_newModelBtn) _newModelBtn->setEnabled(true);
        if (_reoptBtn) _reoptBtn->setEnabled(true);
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
        if (_newModelBtn) _newModelBtn->setEnabled(false);
        if (_reoptBtn) _reoptBtn->setEnabled(false);
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
        if (_newModelBtn) _newModelBtn->setEnabled(true);
        if (_reoptBtn) _reoptBtn->setEnabled(true);
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
        if (_newModelBtn) _newModelBtn->setEnabled(true);
        if (_reoptBtn) _reoptBtn->setEnabled(true);
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
// Trigger optimization (shared by both action buttons)
// ---------------------------------------------------------------------------

void SegmentationLasagnaPanel::triggerOptimization()
{
    const QString& configPath = (_lasagnaMode == 1)
        ? _newModelConfigFilePath : _reoptConfigFilePath;

    if (configPath.isEmpty()) {
        _progressLabel->setText(tr("No config file selected."));
        _progressLabel->setStyleSheet(QStringLiteral("color: #c0392b;"));
        _progressLabel->setVisible(true);
        return;
    }
    if (!QFileInfo::exists(configPath)) {
        _progressLabel->setText(tr("Config file not found: %1").arg(configPath));
        _progressLabel->setStyleSheet(QStringLiteral("color: #c0392b;"));
        _progressLabel->setVisible(true);
        return;
    }

    // If in external mode and not yet connected, connect first
    if (_connectionMode == 1) {
        auto& mgr = LasagnaServiceManager::instance();
        if (!mgr.isExternal() || !mgr.isRunning()) {
            mgr.connectToExternal(_externalHost, _externalPort);
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
                [conn, errConn](const QString&) {
                    QObject::disconnect(*conn);
                    QObject::disconnect(*errConn);
                    delete conn;
                    delete errConn;
                });
            return;
        }
    }

    emit lasagnaOptimizeRequested();
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

    // Config paths — with migration from old single key
    _newModelConfigFilePath = settings.value(QStringLiteral("lasagna_new_model_config_file_path"), QString()).toString();
    _reoptConfigFilePath = settings.value(QStringLiteral("lasagna_reopt_config_file_path"), QString()).toString();
    if (_newModelConfigFilePath.isEmpty() && _reoptConfigFilePath.isEmpty()) {
        QString oldPath = settings.value(QStringLiteral("lasagna_config_file_path"), QString()).toString();
        if (!oldPath.isEmpty()) {
            _newModelConfigFilePath = oldPath;
            _reoptConfigFilePath = oldPath;
            writeSetting(QStringLiteral("lasagna_new_model_config_file_path"), _newModelConfigFilePath);
            writeSetting(QStringLiteral("lasagna_reopt_config_file_path"), _reoptConfigFilePath);
        }
    }

    // Seed point
    if (_seedEdit) {
        const QSignalBlocker b(_seedEdit);
        _seedEdit->setText(settings.value(QStringLiteral("lasagna_seed_point"), QString()).toString());
    }
    // Output name
    if (_outputNameEdit) {
        const QSignalBlocker b(_outputNameEdit);
        _outputNameEdit->setText(settings.value(QStringLiteral("lasagna_output_name"), QString()).toString());
    }
    // Dimensions
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

    // Connection settings
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

    // Populate config combos from saved paths
    if (!_newModelConfigFilePath.isEmpty()) {
        QFileInfo fi(_newModelConfigFilePath);
        if (fi.exists()) {
            populateConfigCombo(_newModelConfigCombo, fi.absolutePath(), fi.fileName(), _newModelConfigFilePath);
        } else if (_newModelConfigCombo) {
            _newModelConfigCombo->clear();
            _newModelConfigCombo->addItem(fi.fileName(), _newModelConfigFilePath);
        }
    }
    if (!_reoptConfigFilePath.isEmpty()) {
        QFileInfo fi(_reoptConfigFilePath);
        if (fi.exists()) {
            populateConfigCombo(_reoptConfigCombo, fi.absolutePath(), fi.fileName(), _reoptConfigFilePath);
        } else if (_reoptConfigCombo) {
            _reoptConfigCombo->clear();
            _reoptConfigCombo->addItem(fi.fileName(), _reoptConfigFilePath);
        }
    }

    // Expand states
    if (_connectionGroup) {
        _connectionGroup->setExpanded(
            settings.value(QStringLiteral("group_lasagna_connection_expanded"), false).toBool());
    }
    if (_newModelGroup) {
        _newModelGroup->setExpanded(
            settings.value(QStringLiteral("group_lasagna_new_model_expanded"), false).toBool());
    }
    if (_reoptGroup) {
        _reoptGroup->setExpanded(
            settings.value(QStringLiteral("group_lasagna_reopt_expanded"), false).toBool());
    }

    _restoringSettings = false;
}

void SegmentationLasagnaPanel::syncUiState(bool /*editingEnabled*/, bool optimizing)
{
    if (_dataInputEdit) {
        const QSignalBlocker b(_dataInputEdit);
        _dataInputEdit->setText(_lasagnaDataInputPath);
    }

    if (_newModelBtn) _newModelBtn->setEnabled(!optimizing);
    if (_reoptBtn) _reoptBtn->setEnabled(!optimizing);
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
// Config file selector (reusable for both combos)
// ---------------------------------------------------------------------------

void SegmentationLasagnaPanel::populateConfigCombo(
    QComboBox* combo, const QString& dir,
    const QString& selectName, QString& outPath)
{
    if (!combo) return;

    const QSignalBlocker b(combo);
    combo->clear();

    QDir d(dir);
    QStringList jsonFiles = d.entryList(
        QStringList{QStringLiteral("*.json")}, QDir::Files, QDir::Name);

    int selectIndex = -1;
    for (int i = 0; i < jsonFiles.size(); ++i) {
        QString fullPath = d.absoluteFilePath(jsonFiles[i]);
        combo->addItem(jsonFiles[i], fullPath);
        if (jsonFiles[i] == selectName) {
            selectIndex = i;
        }
    }

    if (selectIndex >= 0) {
        combo->setCurrentIndex(selectIndex);
        outPath = combo->currentData().toString();
    } else if (combo->count() > 0) {
        combo->setCurrentIndex(0);
        outPath = combo->currentData().toString();
    }
}

// ---------------------------------------------------------------------------
// Config JSON — reads file from disk on demand
// ---------------------------------------------------------------------------

QString SegmentationLasagnaPanel::lasagnaConfigText() const
{
    const QString& path = (_lasagnaMode == 1)
        ? _newModelConfigFilePath : _reoptConfigFilePath;
    if (path.isEmpty()) return {};
    QFile f(path);
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
// Lasagna mode helpers
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

QString SegmentationLasagnaPanel::seedPointText() const
{
    return _seedEdit ? _seedEdit->text().trimmed() : QString();
}

QString SegmentationLasagnaPanel::newModelOutputName() const
{
    return _outputNameEdit ? _outputNameEdit->text().trimmed() : QString();
}

void SegmentationLasagnaPanel::setSeedFromFocus(int x, int y, int z)
{
    if (_seedEdit)
        _seedEdit->setText(QString("%1, %2, %3").arg(x).arg(y).arg(z));
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
