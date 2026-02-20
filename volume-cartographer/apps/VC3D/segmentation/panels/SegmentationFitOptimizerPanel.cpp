#include "SegmentationFitOptimizerPanel.hpp"

#include "FitServiceManager.hpp"
#include "VCSettings.hpp"
#include "elements/CollapsibleSettingsGroup.hpp"

#include <QComboBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QSettings>
#include <QSignalBlocker>
#include <QStackedWidget>
#include <QToolButton>
#include <QVBoxLayout>

#include <nlohmann/json.hpp>

#include <iostream>

// ---------------------------------------------------------------------------
// Predefined optimizer profiles
// ---------------------------------------------------------------------------
struct FitProfile {
    const char* name;
    const char* description;
    const char* json;
};

static const FitProfile kProfiles[] = {
    {"Quick Reopt (2k)", "Fast re-optimization of existing mesh",
     R"({
    "base": {
        "dir_v": 10.0,
        "dir_conn": 1.0,
        "step": 100.0,
        "smooth_y": 0.01,
        "gradmag": 1.0,
        "data": 0.4,
        "data_plain": 0.2,
        "z_straight": 10.0
    },
    "stages": [
        {
            "name": "reopt",
            "steps": 2000,
            "lr": [0.1, 0.01, 0.001],
            "params": ["mesh_ms", "conn_offset_ms", "amp", "bias"],
            "min_scaledown": 0
        }
    ]
})"},

    {"Final Refinement (10k)", "Single-stage full-resolution refinement",
     R"({
    "base": {
        "dir_v": 1.0,
        "dir_conn": 0.1,
        "step": 0.1,
        "smooth_y": 0.01,
        "gradmag": 0.01,
        "meshoff_sy": 0.001,
        "angle": 0.001,
        "data": 4.0,
        "data_plain": 2.0
    },
    "stages": [
        {
            "name": "sd0",
            "steps": 10000,
            "lr": [1.0, 0.1, 0.01],
            "params": ["mesh_ms", "conn_offset_ms", "amp", "bias"],
            "min_scaledown": 0
        }
    ]
})"},

    {"Scalespace (21k)", "Multi-scale coarse-to-fine optimization",
     R"({
    "base": {
        "dir_v": 1.0,
        "dir_conn": 0.1,
        "step": 0.1,
        "smooth_y": 0.01,
        "gradmag": 0.01,
        "angle": 0.001
    },
    "stages": [
        {
            "name": "sd4",
            "steps": 2000,
            "lr": 1.0,
            "params": ["mesh_ms", "conn_offset_ms"],
            "min_scaledown": 4
        },
        {
            "name": "sd3",
            "steps": 2000,
            "lr": 0.1,
            "params": ["mesh_ms", "conn_offset_ms"],
            "min_scaledown": 3,
            "w_fac": {"dir_conn": 10.0, "gradmag": 10.0}
        },
        {
            "name": "sd2",
            "steps": 5000,
            "lr": 0.1,
            "params": ["mesh_ms", "conn_offset_ms"],
            "min_scaledown": 2,
            "w_fac": {"dir_conn": 10.0, "gradmag": 10.0}
        },
        {
            "name": "sd1",
            "steps": 5000,
            "lr": 0.1,
            "params": ["mesh_ms", "conn_offset_ms"],
            "min_scaledown": 1,
            "w_fac": {"dir_conn": 10.0, "gradmag": 10.0}
        },
        {
            "name": "sd0",
            "steps": 5000,
            "lr": 0.01,
            "params": ["mesh_ms", "conn_offset_ms"],
            "min_scaledown": 0,
            "w_fac": {"dir_conn": 10000.0, "gradmag": 1.0}
        }
    ]
})"},

    {"Direct + Init (4k)", "Global transform init, then mesh optimization",
     R"({
    "base": {
        "dir_v": 1.0,
        "dir_conn": 1.0,
        "step": 1.0,
        "gradmag": 1.0,
        "mean_pos": 0.1
    },
    "stages": [
        {
            "name": "init",
            "steps": 1000,
            "lr": 0.01,
            "params": ["theta", "winding_scale"],
            "min_scaledown": 0
        },
        {
            "name": "mesh",
            "steps": 3000,
            "lr": 0.1,
            "params": ["mesh_ms", "conn_offset_ms"],
            "min_scaledown": 0
        }
    ]
})"},

    {"Direct + Grow", "Init, mesh opt, then grow up/down",
     R"({
    "base": {
        "dir_v": 1.0,
        "dir_conn": 1.0,
        "step": 1.0,
        "gradmag": 1.0,
        "mean_pos": 0.1
    },
    "stages": [
        {
            "name": "init",
            "steps": 500,
            "lr": 0.1,
            "params": ["theta", "winding_scale"],
            "min_scaledown": 0
        },
        {
            "name": "mesh",
            "steps": 3000,
            "lr": 0.1,
            "params": ["mesh_ms", "conn_offset_ms"],
            "min_scaledown": 0
        },
        {
            "name": "grow_only",
            "grow": {
                "directions": ["down", "up"],
                "generations": 30,
                "steps": 1
            },
            "global_opt": {
                "steps": 0,
                "lr": 0.1,
                "params": ["mesh_ms", "conn_offset_ms"],
                "min_scaledown": 0
            },
            "local_opt": {
                "opt_window": 2,
                "steps": 300,
                "lr": 0.1,
                "params": ["mesh_ms", "conn_offset_ms"],
                "min_scaledown": 0,
                "w_fac": {"mean_pos": 0.0}
            }
        }
    ]
})"},

    {"Grow Left", "Extend mesh leftward with local optimization",
     R"({
    "base": {
        "dir_v": 1.0,
        "dir_conn": 100.0,
        "step": 0.1,
        "smooth_y": 0.01,
        "contr": 0.1,
        "gradmag": 10.0,
        "angle": 0.001,
        "data": 4.0,
        "data_plain": 2.0
    },
    "stages": [
        {
            "name": "grow_left",
            "grow": {
                "directions": ["left"],
                "generations": 10,
                "steps": 1
            },
            "global_opt": {
                "steps": 0,
                "lr": 0.01,
                "params": ["mesh_ms", "conn_offset_ms", "amp", "bias"]
            },
            "local_opt": {
                "steps": 1000,
                "lr": 0.01,
                "params": ["mesh_ms", "conn_offset_ms", "amp", "bias"]
            }
        }
    ]
})"},

    {"Custom", "User-defined configuration", nullptr},
};

static constexpr int kProfileCount = sizeof(kProfiles) / sizeof(kProfiles[0]);
static constexpr int kCustomProfileIndex = kProfileCount - 1;

SegmentationFitOptimizerPanel::SegmentationFitOptimizerPanel(
    const QString& settingsGroup, QWidget* parent)
    : QWidget(parent)
    , _settingsGroup(settingsGroup)
{
    auto* panelLayout = new QVBoxLayout(this);
    panelLayout->setContentsMargins(0, 0, 0, 0);
    panelLayout->setSpacing(0);

    _group = new CollapsibleSettingsGroup(tr("Fit Optimizer"), this);
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

    // Host/port row
    auto* hostRow = new QHBoxLayout();
    auto* hostLabel = new QLabel(tr("Host:"), _externalWidget);
    hostLabel->setFixedWidth(60);
    _hostEdit = new QLineEdit(_externalWidget);
    _hostEdit->setText(QStringLiteral("127.0.0.1"));
    _hostEdit->setPlaceholderText(QStringLiteral("127.0.0.1"));
    auto* portLabel = new QLabel(tr("Port:"), _externalWidget);
    _portEdit = new QLineEdit(_externalWidget);
    _portEdit->setText(QStringLiteral("9999"));
    _portEdit->setPlaceholderText(QStringLiteral("9999"));
    _portEdit->setFixedWidth(70);
    hostRow->addWidget(hostLabel);
    hostRow->addWidget(_hostEdit, 1);
    hostRow->addWidget(portLabel);
    hostRow->addWidget(_portEdit);
    extLayout->addLayout(hostRow);

    _group->contentLayout()->addWidget(_externalWidget);
    _externalWidget->setVisible(false);  // Start hidden (internal mode)

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
    }, tr("Input data zarr (e.g. s5_cos.zarr) required by the fit optimizer."));

    // -- Profile dropdown --
    _group->addRow(tr("Profile:"), [&](QHBoxLayout* row) {
        _profileCombo = new QComboBox(content);
        for (int i = 0; i < kProfileCount; ++i) {
            _profileCombo->addItem(
                QString::fromUtf8(kProfiles[i].name),
                QString::fromUtf8(kProfiles[i].description));
        }
        row->addWidget(_profileCombo, 1);
    }, tr("Predefined optimizer configurations. Select 'Custom' to edit freely."));

    // -- JSON config editor --
    auto* configLabel = new QLabel(tr("Optimizer config (JSON):"), content);
    _group->contentLayout()->addWidget(configLabel);

    _configEdit = new QPlainTextEdit(content);
    _configEdit->setPlaceholderText(tr("Paste optimizer JSON config here"));
    _configEdit->setTabStopDistance(20);
    _configEdit->setMinimumHeight(200);
    _configEdit->setLineWrapMode(QPlainTextEdit::NoWrap);
    QFont monoFont("monospace");
    monoFont.setStyleHint(QFont::Monospace);
    monoFont.setPointSize(9);
    _configEdit->setFont(monoFont);
    _group->contentLayout()->addWidget(_configEdit);

    _configStatus = new QLabel(content);
    _configStatus->setWordWrap(true);
    _configStatus->setVisible(false);
    _group->contentLayout()->addWidget(_configStatus);

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

    // Connection mode
    connect(_connectionCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &SegmentationFitOptimizerPanel::onConnectionModeChanged);

    // External: refresh button
    connect(_refreshBtn, &QToolButton::clicked, this,
            &SegmentationFitOptimizerPanel::refreshDiscoveredServices);

    // External: discovered service selection
    connect(_discoveryCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &SegmentationFitOptimizerPanel::onDiscoveredServiceSelected);

    // External: host/port manual editing
    connect(_hostEdit, &QLineEdit::textChanged, this, [this](const QString& text) {
        _externalHost = text.trimmed();
        writeSetting(QStringLiteral("fit_external_host"), _externalHost);
    });
    connect(_portEdit, &QLineEdit::textChanged, this, [this](const QString& text) {
        _externalPort = text.trimmed().toInt();
        writeSetting(QStringLiteral("fit_external_port"), _externalPort);
    });

    // Dataset combo selection
    connect(_datasetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int index) {
        if (index < 0) return;
        QString path = _datasetCombo->currentData().toString();
        if (!path.isEmpty()) {
            _fitDataInputPath = path;
            writeSetting(QStringLiteral("fit_data_input_path"), _fitDataInputPath);
        }
    });

    // Datasets received from service
    connect(&FitServiceManager::instance(), &FitServiceManager::datasetsReceived,
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
        _fitDataInputPath = text.trimmed();
        writeSetting(QStringLiteral("fit_data_input_path"), _fitDataInputPath);
    });
    connect(_dataInputBrowse, &QToolButton::clicked, this, [this]() {
        QString initial = _fitDataInputPath.isEmpty()
            ? QDir::homePath() : QFileInfo(_fitDataInputPath).absolutePath();
        QString path = QFileDialog::getExistingDirectory(
            this, tr("Select input data (.zarr directory)"), initial);
        if (!path.isEmpty()) {
            _fitDataInputPath = path;
            _dataInputEdit->setText(path);
        }
    });

    // Profile combo
    connect(_profileCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int index) {
        if (_restoringSettings) return;
        writeSetting(QStringLiteral("fit_profile_index"), index);
        loadProfile(index);
    });

    // Config editor
    connect(_configEdit, &QPlainTextEdit::textChanged, this, [this]() {
        if (_restoringSettings) return;
        _fitConfigText = _configEdit->toPlainText();
        writeSetting(QStringLiteral("fit_config_text"), _fitConfigText);
        validateConfigText();

        // Switch to "Custom" if user manually edits while a preset is selected
        if (_profileCombo && _profileCombo->currentIndex() != kCustomProfileIndex) {
            int idx = _profileCombo->currentIndex();
            if (idx >= 0 && idx < kCustomProfileIndex && kProfiles[idx].json) {
                QString profileText = QString::fromUtf8(kProfiles[idx].json).trimmed();
                if (_fitConfigText.trimmed() != profileText) {
                    const QSignalBlocker b(_profileCombo);
                    _profileCombo->setCurrentIndex(kCustomProfileIndex);
                    writeSetting(QStringLiteral("fit_profile_index"), kCustomProfileIndex);
                }
            }
        }
    });

    // Run button
    connect(_runBtn, &QPushButton::clicked, this, [this]() {
        validateConfigText();
        if (!_configError.isEmpty()) {
            _progressLabel->setText(tr("Fix JSON errors before running."));
            _progressLabel->setStyleSheet(QStringLiteral("color: #c0392b;"));
            _progressLabel->setVisible(true);
            return;
        }

        // If in external mode and not yet connected, connect first
        if (_connectionMode == 1) {
            auto& mgr = FitServiceManager::instance();
            if (!mgr.isExternal() || !mgr.isRunning()) {
                mgr.connectToExternal(_externalHost, _externalPort);
                // Wait for serviceStarted signal to emit fitOptimizeRequested
                auto* conn = new QMetaObject::Connection;
                *conn = connect(&mgr, &FitServiceManager::serviceStarted, this,
                    [this, conn]() {
                        QObject::disconnect(*conn);
                        delete conn;
                        emit fitOptimizeRequested();
                    });
                auto* errConn = new QMetaObject::Connection;
                *errConn = connect(&mgr, &FitServiceManager::serviceError, this,
                    [this, conn, errConn](const QString&) {
                        QObject::disconnect(*conn);
                        QObject::disconnect(*errConn);
                        delete conn;
                        delete errConn;
                    });
                return;
            }
        }

        emit fitOptimizeRequested();
    });

    // Stop button
    connect(_stopBtn, &QPushButton::clicked, this, [this]() {
        emit fitStopRequested();
    });

    // Stop Service button — kills the Python process entirely
    connect(_stopServiceBtn, &QPushButton::clicked, this, []() {
        FitServiceManager::instance().stopService();
    });

    // Connect to service manager signals
    auto& mgr = FitServiceManager::instance();
    connect(&mgr, &FitServiceManager::statusMessage, this, [this](const QString& msg) {
        if (_progressLabel) {
            _progressLabel->setText(msg);
            _progressLabel->setStyleSheet(QString());
            _progressLabel->setVisible(true);
        }
        emit fitStatusMessage(msg);
    });
    connect(&mgr, &FitServiceManager::serviceStarted, this, [this]() {
        if (_progressLabel) {
            _progressLabel->setText(tr("Service running"));
            _progressLabel->setStyleSheet(QStringLiteral("color: #27ae60;"));
        }
        if (_stopServiceBtn) _stopServiceBtn->setEnabled(true);
    });
    connect(&mgr, &FitServiceManager::serviceStopped, this, [this]() {
        if (_progressBar) _progressBar->setVisible(false);
        if (_progressLabel) {
            _progressLabel->setText(tr("Service stopped"));
            _progressLabel->setStyleSheet(QString());
        }
        if (_stopBtn) _stopBtn->setEnabled(false);
        if (_stopServiceBtn) _stopServiceBtn->setEnabled(false);
        if (_runBtn) _runBtn->setEnabled(true);
    });
    connect(&mgr, &FitServiceManager::serviceError, this, [this](const QString& err) {
        std::cerr << "[fit-optimizer] service error: " << err.toStdString() << std::endl;
        if (_progressLabel) {
            _progressLabel->setText(tr("Error: %1").arg(err));
            _progressLabel->setStyleSheet(QStringLiteral("color: #c0392b;"));
            _progressLabel->setVisible(true);
        }
    });
    connect(&mgr, &FitServiceManager::optimizationStarted, this, [this]() {
        if (_stopBtn) _stopBtn->setEnabled(true);
        if (_runBtn) _runBtn->setEnabled(false);
        if (_progressLabel) {
            _progressLabel->setText(tr("Optimization started..."));
            _progressLabel->setStyleSheet(QString());
            _progressLabel->setVisible(true);
        }
    });
    connect(&mgr, &FitServiceManager::optimizationProgress, this,
            [this](const QString& stage, int step, int total, double loss) {
        if (_progressBar) {
            if (total > 0) {
                _progressBar->setRange(0, total);
                _progressBar->setValue(step);
                _progressBar->setFormat(
                    tr("%1/%2  Loss: %3")
                        .arg(step).arg(total).arg(loss, 0, 'g', 5));
                _progressBar->setVisible(true);
            } else {
                _progressBar->setVisible(false);
            }
        }
        if (_progressLabel) {
            _progressLabel->setText(
                tr("Step %1/%2  |  Loss: %3  |  Stage: %4")
                    .arg(step).arg(total).arg(loss, 0, 'g', 5).arg(stage));
            _progressLabel->setStyleSheet(QString());
            _progressLabel->setVisible(true);
        }
    });
    connect(&mgr, &FitServiceManager::optimizationFinished, this,
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
    connect(&mgr, &FitServiceManager::optimizationError, this,
            [this](const QString& err) {
        std::cerr << "[fit-optimizer] optimization error: " << err.toStdString() << std::endl;
        if (_stopBtn) _stopBtn->setEnabled(false);
        if (_runBtn) _runBtn->setEnabled(true);
        if (_progressBar) _progressBar->setVisible(false);
        if (_progressLabel) {
            _progressLabel->setText(tr("Optimization error: %1").arg(err));
            _progressLabel->setStyleSheet(QStringLiteral("color: #c0392b;"));
            _progressLabel->setVisible(true);
        }
    });
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

void SegmentationFitOptimizerPanel::writeSetting(const QString& key, const QVariant& value)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(_settingsGroup);
    settings.setValue(key, value);
    settings.endGroup();
}

void SegmentationFitOptimizerPanel::restoreSettings(QSettings& settings)
{
    _restoringSettings = true;

    _fitDataInputPath = settings.value(QStringLiteral("fit_data_input_path"), QString()).toString();
    _fitConfigText = settings.value(QStringLiteral("fit_config_text"), QString()).toString();

    _connectionMode = settings.value(QStringLiteral("fit_connection_mode"), 0).toInt();
    _externalHost = settings.value(QStringLiteral("fit_external_host"),
                                   QStringLiteral("127.0.0.1")).toString();
    _externalPort = settings.value(QStringLiteral("fit_external_port"), 9999).toInt();

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

    int profileIndex = settings.value(QStringLiteral("fit_profile_index"), 0).toInt();
    if (profileIndex < 0 || profileIndex >= kProfileCount) {
        profileIndex = 0;
    }

    if (_fitConfigText.trimmed().isEmpty()) {
        if (profileIndex < kCustomProfileIndex && kProfiles[profileIndex].json) {
            _fitConfigText = QString::fromUtf8(kProfiles[profileIndex].json);
        } else {
            _fitConfigText = QString::fromUtf8(kProfiles[0].json);
        }
    }

    if (_profileCombo) {
        _profileCombo->setCurrentIndex(profileIndex);
    }

    const bool expanded = settings.value(
        QStringLiteral("group_fit_optimizer_expanded"), false).toBool();
    if (_group) {
        _group->setExpanded(expanded);
    }

    _restoringSettings = false;
}

void SegmentationFitOptimizerPanel::syncUiState(bool /*editingEnabled*/, bool optimizing)
{
    if (_dataInputEdit) {
        const QSignalBlocker b(_dataInputEdit);
        _dataInputEdit->setText(_fitDataInputPath);
    }
    if (_configEdit && _configEdit->toPlainText() != _fitConfigText) {
        const QSignalBlocker b(_configEdit);
        _configEdit->setPlainText(_fitConfigText);
    }

    if (_runBtn) _runBtn->setEnabled(!optimizing);
    if (_stopBtn) _stopBtn->setEnabled(optimizing);

    validateConfigText();
}

// ---------------------------------------------------------------------------
// Setters
// ---------------------------------------------------------------------------

void SegmentationFitOptimizerPanel::setFitDataInputPath(const QString& path)
{
    if (_fitDataInputPath == path) return;
    _fitDataInputPath = path;
    writeSetting(QStringLiteral("fit_data_input_path"), _fitDataInputPath);
    if (_dataInputEdit) {
        const QSignalBlocker b(_dataInputEdit);
        _dataInputEdit->setText(path);
    }
}

// ---------------------------------------------------------------------------
// Profile loading
// ---------------------------------------------------------------------------

void SegmentationFitOptimizerPanel::loadProfile(int index)
{
    if (index < 0 || index >= kProfileCount) return;
    if (index == kCustomProfileIndex) return;  // Don't overwrite on "Custom"

    const char* json = kProfiles[index].json;
    if (!json) return;

    _fitConfigText = QString::fromUtf8(json);
    writeSetting(QStringLiteral("fit_config_text"), _fitConfigText);

    if (_configEdit) {
        const QSignalBlocker b(_configEdit);
        _configEdit->setPlainText(_fitConfigText);
    }
    validateConfigText();
}

// ---------------------------------------------------------------------------
// Config JSON
// ---------------------------------------------------------------------------

void SegmentationFitOptimizerPanel::validateConfigText()
{
    _configError.clear();

    QString trimmed = _fitConfigText.trimmed();
    if (trimmed.isEmpty()) {
        if (_configStatus) _configStatus->setVisible(false);
        return;
    }

    try {
        QByteArray utf8 = trimmed.toUtf8();
        nlohmann::json parsed = nlohmann::json::parse(
            utf8.constData(), utf8.constData() + utf8.size());
        if (!parsed.is_object()) {
            _configError = tr("Config must be a JSON object.");
        }
    } catch (const nlohmann::json::parse_error& ex) {
        _configError = tr("JSON parse error (byte %1): %2")
                           .arg(static_cast<qulonglong>(ex.byte))
                           .arg(QString::fromStdString(ex.what()));
    } catch (const std::exception& ex) {
        _configError = tr("JSON error: %1").arg(QString::fromStdString(ex.what()));
    }

    if (_configStatus) {
        if (_configError.isEmpty()) {
            _configStatus->setText(tr("JSON valid"));
            _configStatus->setStyleSheet(QStringLiteral("color: #27ae60;"));
        } else {
            _configStatus->setText(_configError);
            _configStatus->setStyleSheet(QStringLiteral("color: #c0392b;"));
        }
        _configStatus->setVisible(true);
    }
}

std::optional<nlohmann::json> SegmentationFitOptimizerPanel::fitConfigJson() const
{
    QString trimmed = _fitConfigText.trimmed();
    if (trimmed.isEmpty()) return std::nullopt;

    try {
        QByteArray utf8 = trimmed.toUtf8();
        nlohmann::json parsed = nlohmann::json::parse(
            utf8.constData(), utf8.constData() + utf8.size());
        if (parsed.is_object()) return parsed;
    } catch (...) {}

    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Connection mode
// ---------------------------------------------------------------------------

void SegmentationFitOptimizerPanel::onConnectionModeChanged(int index)
{
    if (_restoringSettings) return;
    _connectionMode = index;
    writeSetting(QStringLiteral("fit_connection_mode"), _connectionMode);
    updateConnectionWidgets();

    // If switching to external, disconnect any internal service
    if (_connectionMode == 1) {
        auto& mgr = FitServiceManager::instance();
        if (!mgr.isExternal() && mgr.isRunning()) {
            mgr.stopService();
        }
    }
}

void SegmentationFitOptimizerPanel::updateConnectionWidgets()
{
    bool external = (_connectionMode == 1);

    // Show/hide external widgets (discovery + host/port)
    if (_externalWidget) _externalWidget->setVisible(external);

    // Reset data input stack when switching to internal
    if (!external) {
        if (_dataInputStack) _dataInputStack->setCurrentIndex(0);
    }
}

void SegmentationFitOptimizerPanel::refreshDiscoveredServices()
{
    if (!_discoveryCombo) return;

    _discoveryCombo->clear();
    _discoveryCombo->addItem(tr("(manual entry)"));

    QJsonArray services = FitServiceManager::discoverServices();
    for (const auto& val : services) {
        QJsonObject svc = val.toObject();
        QString host = svc[QStringLiteral("host")].toString();
        int port = svc[QStringLiteral("port")].toInt();
        int pid = svc[QStringLiteral("pid")].toInt();
        QString label = QStringLiteral("%1:%2 (pid %3)").arg(host).arg(port).arg(pid);
        _discoveryCombo->addItem(label, QJsonDocument(svc).toJson(QJsonDocument::Compact));
    }
}

void SegmentationFitOptimizerPanel::onDiscoveredServiceSelected(int index)
{
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
    auto& mgr = FitServiceManager::instance();
    auto* conn = new QMetaObject::Connection;
    *conn = connect(&mgr, &FitServiceManager::serviceStarted, this,
        [this, conn]() {
            QObject::disconnect(*conn);
            delete conn;
            FitServiceManager::instance().fetchDatasets();
        });
    mgr.connectToExternal(host, port);
}
