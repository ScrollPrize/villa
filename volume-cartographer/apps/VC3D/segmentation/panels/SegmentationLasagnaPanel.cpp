#include "SegmentationLasagnaPanel.hpp"

#include "CState.hpp"
#include "LasagnaBatchWindow.hpp"
#include "LasagnaServiceManager.hpp"
#include "SurfaceCatalog.hpp"
#include "VCSettings.hpp"
#include "elements/CollapsibleSettingsGroup.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/ui/VCCollection.hpp"

#include <QAbstractItemView>
#include <QCheckBox>
#include <QComboBox>
#include <QCryptographicHash>
#include <QElapsedTimer>
#include <QFutureWatcher>
#include <QtConcurrent/QtConcurrent>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QDoubleSpinBox>
#include <QHash>
#include <QRegularExpression>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QSignalBlocker>
#include <QSplitter>
#include <QSpinBox>
#include <QStackedWidget>
#include <QStandardItem>
#include <QStandardItemModel>
#include <QStatusBar>
#include <QStringList>
#include <QTableView>
#include <QToolButton>
#include <QVBoxLayout>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace
{
QString lasagnaModeDebugName(int mode)
{
    switch (mode) {
    case 1:
        return QStringLiteral("new_model");
    case 3:
        return QStringLiteral("offset");
    default:
        return QStringLiteral("reopt");
    }
}

QStringList linkedSurfaceNamesFromRefs(const QJsonArray& refs)
{
    QStringList names;
    for (const QJsonValue& value : refs) {
        const QString name = value.toObject()[QStringLiteral("name")].toString().trimmed();
        if (!name.isEmpty()) {
            names.append(name);
        }
    }
    return names;
}

QStringList linkedSurfaceNamesFromJobSpec(const QJsonObject& jobSpec)
{
    return linkedSurfaceNamesFromRefs(jobSpec[QStringLiteral("linked_surfaces")].toArray());
}

QJsonArray volumeShapeZyxForState(CState* state)
{
    if (!state || !state->currentVolume()) {
        return {};
    }
    const auto shape = state->currentVolume()->shape();
    return QJsonArray{
        static_cast<int>(shape[0]),
        static_cast<int>(shape[1]),
        static_cast<int>(shape[2]),
    };
}

// A stored surface for a Lasagna job: the service reads it from the store by
// UUID (at `revision` when given); the name is for display.
QJsonObject surfaceRef(const std::string& uuid, const std::string& name,
                       const std::optional<std::string>& revision = std::nullopt)
{
    QJsonObject ref;
    ref[QStringLiteral("type")] = QStringLiteral("surface");
    ref[QStringLiteral("uuid")] = QString::fromStdString(uuid);
    ref[QStringLiteral("name")] = QString::fromStdString(name);
    if (revision) {
        ref[QStringLiteral("revision")] = QString::fromStdString(*revision);
    }
    return ref;
}

// The name a result of `name` is versioned from: without its _vNNN version
// and _off<N> offset suffixes.
QString lasagnaRootName(QString name)
{
    static const QRegularExpression version(QStringLiteral("_v\\d+$"));
    name.remove(version);
    const int offset = name.indexOf(QStringLiteral("_off"));
    return offset >= 0 ? name.left(offset) : name;
}

// The segmentation surface's catalog row, when it is a stored surface.
const SurfaceCatalog::Row* activeSurfaceRow(CState* state, const SurfaceCatalog* catalog)
{
    const auto active = state && catalog ? state->surface("segmentation") : nullptr;
    return active ? catalog->row(active->id) : nullptr;
}

QJsonObject recordMeta(const SurfaceCatalog::Row& row)
{
    return QJsonDocument::fromJson(QByteArray::fromStdString(row.record.meta)).object();
}

// The surfaces a job of `row` reads: those the job that made it read (its
// provenance sources), else itself.
QJsonArray linkedSurfacesOf(const SurfaceCatalog::Row& row, const SurfaceCatalog& catalog)
{
    QJsonArray links;
    const auto sources = recordMeta(row)[QStringLiteral("provenance")].toObject()[QStringLiteral("sources")].toArray();
    for (const QJsonValue& source : sources) {
        const auto uuid = source.toString().toStdString();
        const auto* listed = catalog.row(uuid);
        links.append(surfaceRef(uuid, listed ? listed->record.name : uuid));
    }
    return links.isEmpty() ? QJsonArray{surfaceRef(row.record.uuid, row.record.name)} : links;
}

// Names outputs must not reuse: the volume's surfaces and outputs already requested.
}  // namespace

SegmentationLasagnaPanel::SegmentationLasagnaPanel(
    const QString& settingsGroup, QWidget* parent)
    : QWidget(parent)
    , _settingsGroup(settingsGroup)
{
    auto* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(0);

    auto* splitter = new QSplitter(Qt::Horizontal, this);
    splitter->setChildrenCollapsible(false);
    rootLayout->addWidget(splitter);

    auto* controlsScrollArea = new QScrollArea(splitter);
    controlsScrollArea->setFrameShape(QFrame::NoFrame);
    controlsScrollArea->setWidgetResizable(true);
    controlsScrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    controlsScrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

    auto* controlsWidget = new QWidget(controlsScrollArea);
    auto* panelLayout = new QVBoxLayout(controlsWidget);
    panelLayout->setContentsMargins(0, 0, 0, 0);
    panelLayout->setSpacing(2);
    controlsScrollArea->setWidget(controlsWidget);
    splitter->addWidget(controlsScrollArea);

    // =======================================================================
    // Connection section
    // =======================================================================
    _connectionGroup = new CollapsibleSettingsGroup(tr("Solver Connection"), this);
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
    _dataInputEdit->setPlaceholderText(tr("Path to input data (.lasagna.json)"));
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
    }, tr("Input data (.lasagna.json) required by the lasagna."));

    panelLayout->addWidget(_connectionGroup);

    // =======================================================================
    // New Model settings + action button
    // =======================================================================
    {
        auto* sep = new QFrame(this);
        sep->setFrameShape(QFrame::HLine);
        sep->setFrameShadow(QFrame::Sunken);
        panelLayout->addWidget(sep);
    }

    _newModelGroup = new CollapsibleSettingsGroup(tr("New Model Settings"), this);
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
        _widthSpin = new QDoubleSpinBox(dimWidget);
        _widthSpin->setRange(0.001, 999999.0);
        _widthSpin->setDecimals(0);
        _widthSpin->setValue(2048.0);
        _widthSpin->setSingleStep(64.0);
        dimLayout->addWidget(_widthSpin, 1);
        _widthUnitCombo = new QComboBox(dimWidget);
        _widthUnitCombo->addItem(QStringLiteral("vx"), QStringLiteral("voxels"));
        _widthUnitCombo->addItem(QStringLiteral("wraps"), QStringLiteral("wraps"));
        dimLayout->addWidget(_widthUnitCombo);

        dimLayout->addWidget(new QLabel(tr("H:"), dimWidget));
        _heightSpin = new QSpinBox(dimWidget);
        _heightSpin->setRange(1, 999999);
        _heightSpin->setValue(2048);
        _heightSpin->setSingleStep(64);
        dimLayout->addWidget(_heightSpin, 1);

        dimLayout->addWidget(new QLabel(tr("D:"), dimWidget));
        _windingsSpin = new QSpinBox(dimWidget);
        _windingsSpin->setRange(1, 999);
        _windingsSpin->setValue(3);
        _windingsSpin->setSingleStep(1);
        _windingsSpin->setToolTip(tr("Number of windings / model depth layers"));
        dimLayout->addWidget(_windingsSpin, 1);

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
        _outputNameEdit->setToolTip(tr("Output name prefix (auto-versioned, e.g. mysheet → mysheet_v001)"));
        nameLayout->addWidget(_outputNameEdit, 1);
        _newModelGroup->contentLayout()->addWidget(nameWidget);
    }

    panelLayout->addWidget(_newModelGroup);

    _newModelBtn = new QPushButton(tr("New Model"), this);
    panelLayout->addWidget(_newModelBtn);

    // =======================================================================
    // Re-optimize settings + action button
    // =======================================================================
    {
        auto* sep = new QFrame(this);
        sep->setFrameShape(QFrame::HLine);
        sep->setFrameShadow(QFrame::Sunken);
        panelLayout->addWidget(sep);
    }

    _reoptGroup = new CollapsibleSettingsGroup(tr("Re-optimize Settings"), this);
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

    _reoptBtn = new QPushButton(tr("Re-optimize"), this);
    panelLayout->addWidget(_reoptBtn);

    // =======================================================================
    // Offset settings + action button
    // =======================================================================
    {
        auto* sep = new QFrame(this);
        sep->setFrameShape(QFrame::HLine);
        sep->setFrameShadow(QFrame::Sunken);
        panelLayout->addWidget(sep);
    }

    _offsetGroup = new CollapsibleSettingsGroup(tr("Offset Settings"), this);
    auto* offsetContent = _offsetGroup->contentWidget();

    _offsetGroup->addRow(tr("Config:"), [&](QHBoxLayout* row) {
        _offsetConfigCombo = new QComboBox(offsetContent);
        _offsetConfigCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        _offsetConfigBrowse = new QToolButton(offsetContent);
        _offsetConfigBrowse->setText(QStringLiteral("..."));
        _offsetConfigBrowse->setToolTip(tr("Browse for a JSON config file"));
        row->addWidget(_offsetConfigCombo, 1);
        row->addWidget(_offsetConfigBrowse);
    }, tr("JSON config file for offset optimization."));

    _offsetGroup->addRow(tr("Offset:"), [&](QHBoxLayout* row) {
        _offsetValueSpin = new QDoubleSpinBox(offsetContent);
        _offsetValueSpin->setRange(-5.0, 5.0);
        _offsetValueSpin->setSingleStep(1.0);
        _offsetValueSpin->setDecimals(2);
        _offsetValueSpin->setValue(1.0);
        _offsetValueSpin->setToolTip(tr("Target grad_mag integral offset (-1, 0, +1 typical)"));
        row->addWidget(_offsetValueSpin);
    }, tr("Offset in winding-integral space. 0=reoptimize in place, ±1=adjacent winding."));

    panelLayout->addWidget(_offsetGroup);

    _offsetBtn = new QPushButton(tr("Offset"), this);
    panelLayout->addWidget(_offsetBtn);

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

    auto* progressState = new QWidget(this);
    progressState->hide();
    _progressBar = new QProgressBar(progressState);
    _progressBar->setRange(0, 100);
    _progressBar->setValue(0);
    _progressBar->setTextVisible(true);
    _progressBar->setVisible(false);

    _progressLabel = new QLabel(progressState);
    _progressLabel->setWordWrap(true);
    _progressLabel->setVisible(false);
    panelLayout->addStretch(1);

    _batchWindow = new LasagnaBatchWindow(splitter);
    _batchWindow->setMinimumHeight(180);
    _batchWindow->setMinimumWidth(280);
    splitter->addWidget(_batchWindow);
    updateLinkedSurfaceTables();
    connect(_batchWindow, &LasagnaBatchWindow::finishedOutputActivated,
            this, &SegmentationLasagnaPanel::lasagnaOutputActivated);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 2);
    splitter->setSizes({360, 640});

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
    connect(_hostEdit, &QLineEdit::editingFinished, this, [this]() {
        if (_connectionMode == 1 && !_externalHost.isEmpty() && _externalPort > 0) {
            LasagnaServiceManager::instance().connectToExternal(_externalHost, _externalPort);
        }
    });
    connect(_portEdit, &QLineEdit::textChanged, this, [this](const QString& text) {
        _externalPort = text.trimmed().toInt();
        writeSetting(QStringLiteral("lasagna_external_port"), _externalPort);
    });
    connect(_portEdit, &QLineEdit::editingFinished, this, [this]() {
        if (_connectionMode == 1 && !_externalHost.isEmpty() && _externalPort > 0) {
            LasagnaServiceManager::instance().connectToExternal(_externalHost, _externalPort);
        }
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
        QString prevPath = _lasagnaDataInputPath;
        _datasetCombo->clear();
        if (datasets.isEmpty()) {
            _datasetCombo->setEnabled(false);
            if (_newModelBtn) _newModelBtn->setEnabled(false);
            if (_reoptBtn) _reoptBtn->setEnabled(false);
            if (_offsetBtn) _offsetBtn->setEnabled(false);
            syncCompactStatusFromFull();
            return;
        }
        int restoreIdx = 0;
        for (int i = 0; i < datasets.size(); ++i) {
            QJsonObject ds = datasets[i].toObject();
            QString name = ds[QStringLiteral("name")].toString();
            QString path = ds[QStringLiteral("path")].toString();
            _datasetCombo->addItem(name, path);
            if (path == prevPath)
                restoreIdx = i;
        }
        _datasetCombo->setCurrentIndex(restoreIdx);
        _datasetCombo->setEnabled(true);
        _dataInputStack->setCurrentIndex(1);
        if (_newModelBtn) _newModelBtn->setEnabled(true);
        if (_reoptBtn) _reoptBtn->setEnabled(true);
        if (_offsetBtn) _offsetBtn->setEnabled(true);
    });

    connect(_dataInputEdit, &QLineEdit::textChanged, this, [this](const QString& text) {
        _lasagnaDataInputPath = text.trimmed();
        writeSetting(QStringLiteral("lasagna_data_input_path"), _lasagnaDataInputPath);
    });
    connect(_dataInputBrowse, &QToolButton::clicked, this, [this]() {
        QString initial = _lasagnaDataInputPath.isEmpty()
            ? QDir::homePath() : QFileInfo(_lasagnaDataInputPath).absolutePath();
        QString path = QFileDialog::getOpenFileName(
            this, tr("Select lasagna volume (.lasagna.json)"), initial,
            tr("Lasagna volumes (*.lasagna.json);;All files (*)"));
        if (!path.isEmpty()) {
            _lasagnaDataInputPath = path;
            _dataInputEdit->setText(path);
        }
    });

    // -- New model settings persistence --
    connect(_widthSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) {
        const QString unit = newModelWidthUnit();
        if (unit == QStringLiteral("wraps")) {
            writeSetting(QStringLiteral("lasagna_new_model_width_wraps"), v);
        } else {
            writeSetting(QStringLiteral("lasagna_new_model_width"), v);
        }
    });
    connect(_widthUnitCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0) return;
        const QString unit = newModelWidthUnit();
        writeSetting(QStringLiteral("lasagna_new_model_width_unit"), unit);
        if (!_widthSpin) return;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.beginGroup(_settingsGroup);
        const double value = unit == QStringLiteral("wraps")
            ? settings.value(QStringLiteral("lasagna_new_model_width_wraps"), 1.0).toDouble()
            : settings.value(QStringLiteral("lasagna_new_model_width"), 2048.0).toDouble();
        settings.endGroup();
        const QSignalBlocker b(_widthSpin);
        if (unit == QStringLiteral("wraps")) {
            _widthSpin->setDecimals(3);
            _widthSpin->setSingleStep(0.25);
        } else {
            _widthSpin->setDecimals(0);
            _widthSpin->setSingleStep(64.0);
        }
        _widthSpin->setValue(value);
    });
    connect(_heightSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) {
        writeSetting(QStringLiteral("lasagna_new_model_height"), v);
    });
    connect(_windingsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) {
        writeSetting(QStringLiteral("lasagna_new_model_windings"), v);
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
            syncCompactConfigCombos();
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
            syncCompactConfigCombos();
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
            syncCompactConfigCombos();
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
            syncCompactConfigCombos();
        }
    });

    // -- Offset config combo --
    connect(_offsetConfigCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int index) {
        if (_restoringSettings || index < 0) return;
        QString path = _offsetConfigCombo->currentData().toString();
        if (!path.isEmpty()) {
            _offsetConfigFilePath = path;
            writeSetting(QStringLiteral("lasagna_offset_config_file_path"), _offsetConfigFilePath);
            syncCompactConfigCombos();
        }
    });
    connect(_offsetConfigBrowse, &QToolButton::clicked, this, [this]() {
        QString initial = _offsetConfigFilePath.isEmpty()
            ? QDir::homePath() : QFileInfo(_offsetConfigFilePath).absolutePath();
        QString path = QFileDialog::getOpenFileName(
            this, tr("Select optimizer config JSON file"), initial,
            tr("JSON files (*.json);;All files (*)"));
        if (!path.isEmpty()) {
            _offsetConfigFilePath = path;
            writeSetting(QStringLiteral("lasagna_offset_config_file_path"), _offsetConfigFilePath);
            QFileInfo fi(path);
            populateConfigCombo(_offsetConfigCombo, fi.absolutePath(), fi.fileName(), _offsetConfigFilePath);
            syncCompactConfigCombos();
        }
    });
    connect(_offsetValueSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) {
        writeSetting(QStringLiteral("lasagna_offset_value"), v);
    });

    // -- Action buttons --
    connect(_newModelBtn, &QPushButton::clicked, this, [this]() {
        launchLasagnaMode(LasagnaMode::NewModel);
    });
    connect(_reoptBtn, &QPushButton::clicked, this, [this]() {
        launchLasagnaMode(LasagnaMode::ReOptimize);
    });
    connect(_offsetBtn, &QPushButton::clicked, this, [this]() {
        launchLasagnaMode(LasagnaMode::Offset);
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
    connect(_offsetGroup, &CollapsibleSettingsGroup::toggled, this, [this](bool expanded) {
        writeSetting(QStringLiteral("group_lasagna_offset_expanded"), expanded);
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
        syncCompactStatusFromFull();
        emit lasagnaStatusMessage(msg);
    });
    connect(&mgr, &LasagnaServiceManager::serviceStarted, this, [this]() {
        if (_progressLabel) {
            _progressLabel->setText(tr("Service running"));
            _progressLabel->setStyleSheet(QStringLiteral("color: #27ae60;"));
        }
        if (_stopServiceBtn) _stopServiceBtn->setEnabled(true);
        // Always fetch datasets from the connected service
        LasagnaServiceManager::instance().fetchDatasets();
        syncCompactStatusFromFull();
    });
    connect(&mgr, &LasagnaServiceManager::serviceStopped, this, [this]() {
        if (_progressBar) _progressBar->setVisible(false);
        if (_progressLabel) {
            _progressLabel->setText(tr("Service stopped"));
            _progressLabel->setStyleSheet(QString());
        }
        if (_stopBtn) _stopBtn->setEnabled(false);
        if (_stopServiceBtn) _stopServiceBtn->setEnabled(false);
        if (_connectionMode == 1) {
            // External mode: clear datasets and disable controls
            if (_datasetCombo) { _datasetCombo->clear(); _datasetCombo->setEnabled(false); }
            if (_newModelBtn) _newModelBtn->setEnabled(false);
            if (_reoptBtn) _reoptBtn->setEnabled(false);
            if (_offsetBtn) _offsetBtn->setEnabled(false);
        } else {
            if (_newModelBtn) _newModelBtn->setEnabled(true);
            if (_reoptBtn) _reoptBtn->setEnabled(true);
            if (_offsetBtn) _offsetBtn->setEnabled(true);
                }
        syncCompactStatusFromFull();
    });
    connect(&mgr, &LasagnaServiceManager::serviceError, this, [this](const QString& err) {
        std::cerr << "[lasagna] service error: " << err.toStdString() << std::endl;
        if (_progressLabel) {
            _progressLabel->setText(tr("Error: %1").arg(err));
            _progressLabel->setStyleSheet(QStringLiteral("color: #c0392b;"));
            _progressLabel->setVisible(true);
        }
        if (_connectionMode == 1) {
            if (_datasetCombo) _datasetCombo->setEnabled(false);
            if (_newModelBtn) _newModelBtn->setEnabled(false);
            if (_reoptBtn) _reoptBtn->setEnabled(false);
            if (_offsetBtn) _offsetBtn->setEnabled(false);
        }
        syncCompactStatusFromFull();
    });
    connect(&mgr, &LasagnaServiceManager::optimizationStarted, this, [this]() {
        if (_stopBtn) _stopBtn->setEnabled(true);

        if (_progressLabel) {
            _progressLabel->setText(tr("Optimization started..."));
            _progressLabel->setStyleSheet(QString());
            _progressLabel->setVisible(true);
        }
        syncCompactStatusFromFull();
    });
    connect(&mgr, &LasagnaServiceManager::optimizationProgress, this,
            [this](const QString& /*stage*/, int step, int total, double loss,
                   double stageProgress, double overallProgress,
                   const QString& stageName) {
        if ((overallProgress <= 0.0) && step > 0 && total > 0) {
            overallProgress = static_cast<double>(step) / static_cast<double>(total);
        }
        overallProgress = std::clamp(overallProgress, 0.0, 1.0);
        stageProgress = std::clamp(stageProgress, 0.0, 1.0);
        if (_progressBar) {
            _progressBar->setRange(0, 1000);
            _progressBar->setValue(static_cast<int>(overallProgress * 1000.0));
            _progressBar->setFormat(
                tr("Overall: %1%").arg(overallProgress * 100.0, 0, 'f', 1));
            _progressBar->setVisible(true);
        }
        if (_progressLabel) {
            const QString stageText = stageName.isEmpty() ? QStringLiteral("...") : stageName;
            const QString label = total > 0
                ? tr("Stage: %1 (%2%)  |  Overall: %3%  |  Step: %4/%5  |  Loss: %6")
                    .arg(stageText)
                    .arg(stageProgress * 100.0, 0, 'f', 1)
                    .arg(overallProgress * 100.0, 0, 'f', 1)
                    .arg(step)
                    .arg(total)
                    .arg(loss, 0, 'g', 5)
                : tr("Stage: %1 (%2%)  |  Overall: %3%  |  Loss: %4")
                    .arg(stageText)
                    .arg(stageProgress * 100.0, 0, 'f', 1)
                    .arg(overallProgress * 100.0, 0, 'f', 1)
                    .arg(loss, 0, 'g', 5);
            _progressLabel->setText(label);
            _progressLabel->setStyleSheet(QString());
            _progressLabel->setVisible(true);
        }
        syncCompactStatusFromFull();
    });
    connect(&mgr, &LasagnaServiceManager::jobsUpdated, this, [this](const QJsonArray& jobs) {
        QStringList queued;
        bool running = false;
        for (const QJsonValue& value : jobs) {
            QJsonObject job = value.toObject();
            const QString state = job[QStringLiteral("state")].toString();
            if (state == QStringLiteral("running")) {
                running = true;
            } else if (state == QStringLiteral("upload")) {
                running = true;
            } else if (state == QStringLiteral("waiting")) {
                const int pos = job[QStringLiteral("queue_position")].toInt();
                if (pos > 0) {
                    const QString outputName = job[QStringLiteral("output_name")].toString().trimmed();
                    queued << (outputName.isEmpty()
                        ? QStringLiteral("#%1").arg(pos)
                        : QStringLiteral("#%1 %2").arg(pos).arg(outputName));
                }
            }
        }
        if (!running && !queued.isEmpty() && _progressLabel) {
            _progressLabel->setText(tr("Queue: %1").arg(queued.join(QStringLiteral(", "))));
            _progressLabel->setStyleSheet(QString());
            _progressLabel->setVisible(true);
        }
        syncCompactStatusFromFull();
    });
    connect(&mgr, &LasagnaServiceManager::optimizationFinished, this,
            [this](const QStringList& surfaceUuids) {
        if (_stopBtn) _stopBtn->setEnabled(false);
        if (_newModelBtn) _newModelBtn->setEnabled(true);
        if (_reoptBtn) _reoptBtn->setEnabled(true);
        if (_offsetBtn) _offsetBtn->setEnabled(true);
        if (_progressBar) _progressBar->setVisible(false);
        if (_progressLabel) {
            QStringList names;
            for (const QString& uuid : surfaceUuids) {
                const auto* row = _surfaceCatalog ? _surfaceCatalog->row(uuid.toStdString()) : nullptr;
                names.append(row ? QString::fromStdString(row->record.name) : uuid);
            }
            _progressLabel->setText(tr("Optimization finished. Output: %1").arg(names.join(QStringLiteral(", "))));
            _progressLabel->setStyleSheet(QStringLiteral("color: #27ae60;"));
            _progressLabel->setVisible(true);
        }
        syncCompactStatusFromFull();
    });
    connect(&mgr, &LasagnaServiceManager::optimizationError, this,
            [this](const QString& err) {
        std::cerr << "[lasagna] optimization error: " << err.toStdString() << std::endl;
        if (_stopBtn) _stopBtn->setEnabled(false);
        if (_newModelBtn) _newModelBtn->setEnabled(true);
        if (_reoptBtn) _reoptBtn->setEnabled(true);
        if (_offsetBtn) _offsetBtn->setEnabled(true);
        if (_progressBar) _progressBar->setVisible(false);
        if (_progressLabel) {
            _progressLabel->setText(tr("Optimization error: %1").arg(err));
            _progressLabel->setStyleSheet(QStringLiteral("color: #c0392b;"));
            _progressLabel->setVisible(true);
        }
        syncCompactStatusFromFull();
    });

#ifndef VC_TEST_DISABLE_LASAGNA_DISCOVERY
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
#endif
}

void SegmentationLasagnaPanel::setState(CState* state)
{
    if (_state == state) {
        updateLinkedSurfaceTables();
        return;
    }
    if (_stateSurfaceChangedConnection) {
        QObject::disconnect(_stateSurfaceChangedConnection);
        _stateSurfaceChangedConnection = {};
    }
    _state = state;
    if (_state) {
        _stateSurfaceChangedConnection = connect(
            _state,
            &CState::surfaceChanged,
            this,
            [this](const std::string& name, std::shared_ptr<Surface>, bool) {
                if (name == "segmentation") {
                    updateLinkedSurfaceTables();
                }
            });
    }
    updateLinkedSurfaceTables();
}

QWidget* SegmentationLasagnaPanel::createCompactView(QWidget* parent)
{
    if (_compactView) {
        _compactView->setParent(parent);
        return _compactView;
    }

    _compactView = new QWidget(parent);
    auto* layout = new QVBoxLayout(_compactView);
    layout->setContentsMargins(6, 6, 6, 6);
    layout->setSpacing(6);

    auto* openBtn = new QPushButton(tr("settings"), _compactView);
    layout->addWidget(openBtn);
    connect(openBtn, &QPushButton::clicked, this, &SegmentationLasagnaPanel::openLasagnaWorkspaceRequested);

    auto addConfigRow = [this, layout](const QString& labelText, QComboBox*& combo) {
        auto* row = new QHBoxLayout();
        row->addWidget(new QLabel(labelText, _compactView));
        combo = new QComboBox(_compactView);
        combo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        row->addWidget(combo, 1);
        layout->addLayout(row);
    };

    addConfigRow(tr("New:"), _compactNewModelConfigCombo);
    _compactNewModelBtn = new QPushButton(tr("New Model"), _compactView);
    layout->addWidget(_compactNewModelBtn);

    addConfigRow(tr("Re-opt:"), _compactReoptConfigCombo);
    _compactReoptBtn = new QPushButton(tr("Re-optimize"), _compactView);
    layout->addWidget(_compactReoptBtn);

    layout->addWidget(new QLabel(tr("Linked Surfaces"), _compactView));
    _compactLinkedSurfaceModel = new QStandardItemModel(_compactView);
    _compactLinkedSurfaceModel->setHorizontalHeaderLabels({tr("Name")});
    _compactLinkedSurfaceTable = new QTableView(_compactView);
    _compactLinkedSurfaceTable->setModel(_compactLinkedSurfaceModel);
    _compactLinkedSurfaceTable->setSelectionMode(QAbstractItemView::NoSelection);
    _compactLinkedSurfaceTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _compactLinkedSurfaceTable->horizontalHeader()->setStretchLastSection(true);
    _compactLinkedSurfaceTable->verticalHeader()->setVisible(false);
    _compactLinkedSurfaceTable->setMinimumHeight(80);
    _compactLinkedSurfaceTable->setMaximumHeight(140);
    layout->addWidget(_compactLinkedSurfaceTable);

    auto* stopRow = new QHBoxLayout();
    _compactStopBtn = new QPushButton(tr("Stop"), _compactView);
    _compactStopServiceBtn = new QPushButton(tr("Stop Service"), _compactView);
    stopRow->addWidget(_compactStopBtn);
    stopRow->addWidget(_compactStopServiceBtn);
    layout->addLayout(stopRow);

    _compactProgressBar = new QProgressBar(_compactView);
    layout->addWidget(_compactProgressBar);
    _compactProgressLabel = new QLabel(_compactView);
    _compactProgressLabel->setWordWrap(true);
    layout->addWidget(_compactProgressLabel);
    layout->addStretch(1);

    connect(_compactNewModelBtn, &QPushButton::clicked, this, [this]() {
        launchLasagnaMode(LasagnaMode::NewModel);
    });
    connect(_compactReoptBtn, &QPushButton::clicked, this, [this]() {
        launchLasagnaMode(LasagnaMode::ReOptimize);
    });
    connect(_compactStopBtn, &QPushButton::clicked, this, [this]() {
        emit lasagnaStopRequested();
    });
    connect(_compactStopServiceBtn, &QPushButton::clicked, this, []() {
        LasagnaServiceManager::instance().stopService();
    });
    connect(_compactNewModelConfigCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int index) {
        if (index < 0 || !_compactNewModelConfigCombo) return;
        const QString path = _compactNewModelConfigCombo->currentData().toString();
        if (path.isEmpty() || path == _newModelConfigFilePath) return;
        _newModelConfigFilePath = path;
        writeSetting(QStringLiteral("lasagna_new_model_config_file_path"), _newModelConfigFilePath);
        if (_newModelConfigCombo) {
            const QSignalBlocker blocker(_newModelConfigCombo);
            const int fullIndex = _newModelConfigCombo->findData(path);
            if (fullIndex >= 0) _newModelConfigCombo->setCurrentIndex(fullIndex);
        }
    });
    connect(_compactReoptConfigCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int index) {
        if (index < 0 || !_compactReoptConfigCombo) return;
        const QString path = _compactReoptConfigCombo->currentData().toString();
        if (path.isEmpty() || path == _reoptConfigFilePath) return;
        _reoptConfigFilePath = path;
        writeSetting(QStringLiteral("lasagna_reopt_config_file_path"), _reoptConfigFilePath);
        if (_reoptConfigCombo) {
            const QSignalBlocker blocker(_reoptConfigCombo);
            const int fullIndex = _reoptConfigCombo->findData(path);
            if (fullIndex >= 0) _reoptConfigCombo->setCurrentIndex(fullIndex);
        }
    });
    syncCompactConfigCombos();
    syncCompactStatusFromFull();
    updateLinkedSurfaceTables();
    return _compactView;
}

void SegmentationLasagnaPanel::syncCompactConfigCombos()
{
    auto syncCombo = [](QComboBox* compact, const QComboBox* full, const QString& selectedPath) {
        if (!compact || !full) {
            return;
        }
        const QSignalBlocker blocker(compact);
        compact->clear();
        for (int i = 0; i < full->count(); ++i) {
            compact->addItem(full->itemText(i), full->itemData(i));
        }
        const int index = compact->findData(selectedPath);
        if (index >= 0) {
            compact->setCurrentIndex(index);
        }
    };

    syncCombo(_compactNewModelConfigCombo, _newModelConfigCombo, _newModelConfigFilePath);
    syncCombo(_compactReoptConfigCombo, _reoptConfigCombo, _reoptConfigFilePath);
}

void SegmentationLasagnaPanel::syncCompactStatusFromFull()
{
    if (_compactNewModelBtn && _newModelBtn) {
        _compactNewModelBtn->setEnabled(_newModelBtn->isEnabled());
    }
    if (_compactReoptBtn && _reoptBtn) {
        _compactReoptBtn->setEnabled(_reoptBtn->isEnabled());
    }
    if (_compactStopBtn && _stopBtn) {
        _compactStopBtn->setEnabled(_stopBtn->isEnabled());
    }
    if (_compactStopServiceBtn && _stopServiceBtn) {
        _compactStopServiceBtn->setEnabled(_stopServiceBtn->isEnabled());
    }
    if (_compactProgressBar && _progressBar) {
        _compactProgressBar->setRange(_progressBar->minimum(), _progressBar->maximum());
        _compactProgressBar->setValue(_progressBar->value());
        _compactProgressBar->setFormat(_progressBar->format());
        _compactProgressBar->setVisible(!_progressBar->isHidden());
    }
    if (_compactProgressLabel && _progressLabel) {
        _compactProgressLabel->setText(_progressLabel->text());
        _compactProgressLabel->setStyleSheet(_progressLabel->styleSheet());
        _compactProgressLabel->setVisible(!_progressLabel->isHidden());
    }
}

void SegmentationLasagnaPanel::updateCompactLinkedSurfaceTable(const QStringList& names)
{
    if (!_compactLinkedSurfaceModel) {
        return;
    }
    _compactLinkedSurfaceModel->removeRows(0, _compactLinkedSurfaceModel->rowCount());
    for (const QString& name : names) {
        _compactLinkedSurfaceModel->appendRow(new QStandardItem(name));
    }
}

void SegmentationLasagnaPanel::updateLinkedSurfaceTables()
{
    const QStringList names = currentLinkedSurfaceNames();
    updateCompactLinkedSurfaceTable(names);
    if (_batchWindow) {
        _batchWindow->setLinkedSurfaceNames(names);
    }
}

QStringList SegmentationLasagnaPanel::currentLinkedSurfaceNames() const
{
    const auto* row = activeSurfaceRow(_state, _surfaceCatalog);
    return row ? linkedSurfaceNamesFromRefs(linkedSurfacesOf(*row, *_surfaceCatalog)) : QStringList{};
}

void SegmentationLasagnaPanel::setSurfaceCatalog(SurfaceCatalog* catalog)
{
    _surfaceCatalog = catalog;
    if (catalog) {
        connect(catalog, &SurfaceCatalog::rowsChanged, this, &SegmentationLasagnaPanel::updateLinkedSurfaceTables);
    }
    updateLinkedSurfaceTables();
}

// ---------------------------------------------------------------------------
// Trigger optimization (shared by both action buttons)
// ---------------------------------------------------------------------------

void SegmentationLasagnaPanel::triggerOptimization()
{
    const QString& configPath = (_lasagnaMode == 1) ? _newModelConfigFilePath
                              : (_lasagnaMode == 3) ? _offsetConfigFilePath
                              : _reoptConfigFilePath;

    std::cerr << "[lasagna] task requested:"
              << " mode=" << lasagnaModeDebugName(_lasagnaMode).toStdString()
              << " config=" << configPath.toStdString()
              << " connection=" << (_connectionMode == 1 ? "external" : "internal")
              << std::endl;

    if (!validateLasagnaConfigPath(configPath, nullptr)) {
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

void SegmentationLasagnaPanel::launchLasagnaMode(LasagnaMode mode)
{
    _lastLasagnaMode = mode;
    _lasagnaMode = static_cast<int>(mode);
    updateLinkedSurfaceTables();
    triggerOptimization();
}

void SegmentationLasagnaPanel::repeatLastLasagnaAction()
{
    launchLasagnaMode(_lastLasagnaMode);
}

bool SegmentationLasagnaPanel::repeatLastLasagnaActionHeadless(CState* state, QString* errorMessage)
{
    return startOptimizationHeadless(state, _lastLasagnaMode, QString(), std::nullopt,
                                      errorMessage);
}

void SegmentationLasagnaPanel::startOptimization(CState* state, QStatusBar* statusBar)
{
    (void)startOptimizationWithOverrides(
        state, statusBar, -1, QString(), false, 0, 0, 0);
}

void SegmentationLasagnaPanel::startOptimizationAtSeed(CState* state,
                                                       QStatusBar* statusBar,
                                                       LasagnaMode mode,
                                                       const QString& configPath,
                                                       int seedX,
                                                       int seedY,
                                                       int seedZ)
{
    _lastLasagnaMode = mode;

    if (!validateLasagnaConfigPath(configPath, statusBar)) {
        return;
    }

    if (_connectionMode == 1) {
        auto& mgr = LasagnaServiceManager::instance();
        if (!mgr.isExternal() || !mgr.isRunning()) {
            mgr.connectToExternal(_externalHost, _externalPort);
            auto* conn = new QMetaObject::Connection;
            auto* errConn = new QMetaObject::Connection;
            *conn = connect(&mgr, &LasagnaServiceManager::serviceStarted, this,
                [this, conn, errConn, state, statusBar, mode, configPath, seedX, seedY, seedZ]() {
                    QObject::disconnect(*conn);
                    QObject::disconnect(*errConn);
                    delete conn;
                    delete errConn;
                    (void)startOptimizationWithOverrides(
                        state, statusBar, static_cast<int>(mode), configPath,
                        true, seedX, seedY, seedZ);
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

    (void)startOptimizationWithOverrides(
        state, statusBar, static_cast<int>(mode), configPath,
        true, seedX, seedY, seedZ);
}

bool SegmentationLasagnaPanel::startOptimizationHeadless(CState* state,
                                                         LasagnaMode mode,
                                                         const QString& configPath,
                                                         std::optional<cv::Vec3i> seed,
                                                         QString* errorMessage)
{
    auto fail = [errorMessage](const QString& msg) {
        if (errorMessage) {
            *errorMessage = msg;
        }
        return false;
    };

    _lastLasagnaMode = mode;

    // Resolve the config path without validateLasagnaConfigPath, which can open
    // a message box.
    const QString resolvedConfig = configPath.isEmpty()
        ? selectedLasagnaConfigPathForMode(mode)
        : configPath;
    {
        const QFileInfo info(resolvedConfig);
        if (resolvedConfig.isEmpty() || !info.exists() || !info.isFile()) {
            return fail(QStringLiteral("Lasagna config file not found: %1")
                            .arg(resolvedConfig.isEmpty()
                                     ? QStringLiteral("(none selected)")
                                     : resolvedConfig));
        }
    }

    // Require a live service so failure is returned before dialog-capable
    // submission paths.
    auto& mgr = LasagnaServiceManager::instance();
    if (!mgr.isRunning()) {
        return fail(QStringLiteral("lasagna service is not running"));
    }

    int seedX = 0;
    int seedY = 0;
    int seedZ = 0;
    const bool hasSeed = seed.has_value();
    if (hasSeed) {
        seedX = (*seed)[0];
        seedY = (*seed)[1];
        seedZ = (*seed)[2];
    }

    const SubmissionResult result = startOptimizationWithOverrides(
        state, nullptr, static_cast<int>(mode), resolvedConfig,
        hasSeed, seedX, seedY, seedZ, Presentation::Silent);

    if (!result.submitted) {
        return fail(result.error.isEmpty()
            ? QStringLiteral("Lasagna optimization submission failed before reaching the service")
            : result.error);
    }
    return true;
}

QString SegmentationLasagnaPanel::selectedLasagnaConfigPathForMode(LasagnaMode mode) const
{
    return (mode == LasagnaMode::NewModel) ? _newModelConfigFilePath
         : (mode == LasagnaMode::Offset) ? _offsetConfigFilePath
         : _reoptConfigFilePath;
}

QStringList SegmentationLasagnaPanel::lasagnaConfigPathsForMode(LasagnaMode mode) const
{
    const QComboBox* combo = (mode == LasagnaMode::NewModel) ? _newModelConfigCombo
                          : (mode == LasagnaMode::Offset) ? _offsetConfigCombo
                          : _reoptConfigCombo;
    const QString currentPath = selectedLasagnaConfigPathForMode(mode);
    QStringList paths;
    auto addPath = [&paths](const QString& path) {
        if (!path.isEmpty() && !paths.contains(path)) {
            paths.append(path);
        }
    };

    if (combo) {
        for (int i = 0; i < combo->count(); ++i) {
            addPath(combo->itemData(i).toString());
        }
    }
    addPath(currentPath);
    return paths;
}

void SegmentationLasagnaPanel::showLasagnaConfigError(const QString& message,
                                                      QStatusBar* statusBar,
                                                      int timeoutMs,
                                                      Presentation presentation)
{
    std::cerr << "[lasagna] " << message.toStdString() << std::endl;
    if (statusBar) {
        statusBar->showMessage(message, timeoutMs);
    }
    if (_progressLabel) {
        _progressLabel->setText(message);
        _progressLabel->setStyleSheet(QStringLiteral("color: #c0392b;"));
        _progressLabel->setVisible(true);
    }
#ifndef VC_TEST_DISABLE_LASAGNA_DIALOGS
    if (presentation == Presentation::Interactive) {
        QMessageBox::warning(this, tr("Lasagna config error"), message);
    }
#endif
}

bool SegmentationLasagnaPanel::validateLasagnaConfigPath(const QString& configPath,
                                                         QStatusBar* statusBar,
                                                         Presentation presentation)
{
    if (configPath.isEmpty()) {
        showLasagnaConfigError(
            tr("No Lasagna config file selected."), statusBar, 5000, presentation);
        return false;
    }

    const QFileInfo configInfo(configPath);
    if (!configInfo.exists()) {
        showLasagnaConfigError(tr("Lasagna config file not found: %1").arg(configPath),
                               statusBar,
                               7000,
                               presentation);
        return false;
    }
    if (!configInfo.isFile()) {
        showLasagnaConfigError(tr("Lasagna config path is not a file: %1").arg(configPath),
                               statusBar,
                               7000,
                               presentation);
        return false;
    }

    return true;
}

SegmentationLasagnaPanel::SubmissionResult
SegmentationLasagnaPanel::startOptimizationWithOverrides(
    CState* state,
    QStatusBar* statusBar,
    int modeOverride,
    const QString& configPathOverride,
    bool hasSeedOverride,
    int seedX,
    int seedY,
    int seedZ,
    Presentation presentation)
{
    auto showStatus = [statusBar](const QString& msg, int timeout) {
        if (statusBar) {
            statusBar->showMessage(msg, timeout);
        }
    };

    auto& mgr = LasagnaServiceManager::instance();
    const LasagnaMode launchMode = modeOverride >= 0
        ? static_cast<LasagnaMode>(modeOverride)
        : lasagnaMode();
    const QString configPath = !configPathOverride.isEmpty()
        ? configPathOverride
        : (launchMode == LasagnaMode::NewModel) ? _newModelConfigFilePath
        : (launchMode == LasagnaMode::Offset) ? _offsetConfigFilePath
        : _reoptConfigFilePath;
    const bool isNewModel = (launchMode == LasagnaMode::NewModel);

    if (!validateLasagnaConfigPath(configPath, statusBar, presentation)) {
        return SubmissionResult::failure(
            tr("Lasagna configuration is invalid."));
    }

    QString dataInput = lasagnaDataInputPath();
    if (dataInput.isEmpty()) {
        auto msg = tr("No data input path set. Set the zarr path in the Lasagna Model panel.");
        std::cerr << "[lasagna] " << msg.toStdString() << std::endl;
        showStatus(msg, 5000);
        return SubmissionResult::failure(msg);
    }

    if (mgr.isExternal()) {
        if (!mgr.isRunning()) {
            auto msg = tr("External service not connected. Select a service or check host/port.");
            std::cerr << "[lasagna] " << msg.toStdString() << std::endl;
            showStatus(msg, 5000);
            return SubmissionResult::failure(msg);
        }
        // An external service reads its own datasets; a local path means nothing there.
        if (_datasetCombo->findData(dataInput) < 0) {
            auto msg = tr("Choose one of the external service's datasets as the data input.");
            std::cerr << "[lasagna] " << msg.toStdString() << std::endl;
            showStatus(msg, 5000);
            return SubmissionResult::failure(msg);
        }
    } else {
        if (!mgr.ensureServiceRunning(
                {}, QFileInfo(dataInput).absolutePath())) {
            auto msg = tr("Failed to start lasagna service: %1").arg(mgr.lastError());
            std::cerr << "[lasagna] " << msg.toStdString() << std::endl;
            showStatus(msg, 5000);
            return SubmissionResult::failure(msg);
        }
    }

    QElapsedTimer prepTimer;
    prepTimer.start();

    if (!_surfaceCatalog || !_surfaceCatalog->attachment() || !_surfaceCatalog->volume()) {
        auto msg = tr("Lasagna needs an attached object store and an open volume.");
        showStatus(msg, 5000);
        return SubmissionResult::failure(msg);
    }
    // The segmentation surface is the job's input and, for re-optimization,
    // the source of its model (left with it in the store by the fit that made it).
    const SurfaceCatalog::Row* activeRow = activeSurfaceRow(state, _surfaceCatalog);
    const bool isOffsetMode = (launchMode == LasagnaMode::Offset);
    const std::string activeUuid = activeRow ? activeRow->record.uuid : std::string{};
    const bool hasModel = activeRow && mgr.surfaceModel(activeUuid).has_value();

    QString rootName = QStringLiteral("new_model");
    if (isNewModel) {
        if (const QString nmName = newModelOutputName().trimmed(); !nmName.isEmpty()) {
            rootName = lasagnaRootName(nmName);
        }
    } else if (activeRow) {
        rootName = lasagnaRootName(QString::fromStdString(activeRow->record.name));
    }
    // The service names the result in its store: the next free version of the
    // root, or (offset mode) the first free offset of it.
    QString outputNameMode = QStringLiteral("version");

    QJsonObject config;
    QString configText;
    {
        QFile f(configPath);
        if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
            auto msg = tr("Cannot read Lasagna config %1: %2")
                .arg(configPath, f.errorString());
            showLasagnaConfigError(msg, statusBar, 7000, presentation);
            return SubmissionResult::failure(msg);
        }
        configText = QString::fromUtf8(f.readAll()).trimmed();
    }
    if (!configText.isEmpty()) {
        QJsonParseError parseError;
        QJsonDocument doc = QJsonDocument::fromJson(configText.toUtf8(), &parseError);
        if (parseError.error != QJsonParseError::NoError) {
            auto msg = tr("Invalid Lasagna config JSON at byte %1: %2")
                .arg(parseError.offset)
                .arg(parseError.errorString());
            showLasagnaConfigError(msg, statusBar, 7000, presentation);
            return SubmissionResult::failure(msg);
        }
        if (!doc.isObject()) {
            auto msg = tr("Invalid Lasagna config JSON: top-level value must be an object.");
            showLasagnaConfigError(msg, statusBar, 7000, presentation);
            return SubmissionResult::failure(msg);
        }
        config = doc.object();
    }

    double nmW = newModelWidth();
    QString nmWUnit = newModelWidthUnit();
    int nmH = newModelHeight();
    int nmN = newModelWindings();

    int cx = 0;
    int cy = 0;
    int cz = 0;
    bool seedOk = hasSeedOverride;
    if (hasSeedOverride) {
        cx = seedX;
        cy = seedY;
        cz = seedZ;
    }
    QString seedText = seedOk ? QString() : seedPointText();
    if (!seedOk && !seedText.isEmpty()) {
        QStringList parts = seedText.split(',');
        if (parts.size() == 3) {
            bool ok0 = false;
            bool ok1 = false;
            bool ok2 = false;
            cx = parts[0].trimmed().toInt(&ok0);
            cy = parts[1].trimmed().toInt(&ok1);
            cz = parts[2].trimmed().toInt(&ok2);
            seedOk = ok0 && ok1 && ok2;
        }
    }
    if (!seedOk) {
        POI* focus = state ? state->poi("focus") : nullptr;
        if (!focus) {
            auto msg = tr("No focus position or seed point set. Place the cursor or enter a seed.");
            std::cerr << "[lasagna] " << msg.toStdString() << std::endl;
            showStatus(msg, 5000);
            return SubmissionResult::failure(msg);
        }
        cx = static_cast<int>(focus->p[0]);
        cy = static_cast<int>(focus->p[1]);
        cz = static_cast<int>(focus->p[2]);
    }

    const double offsetVal = offsetValue();

    QJsonObject args = config[QStringLiteral("args")].toObject();
    // VC3D is transport only. Do not add config-semantic branching here.
    // Config interpretation belongs in fit_service.py / fit.py.
    args[QStringLiteral("seed")] = QJsonArray{cx, cy, cz};
    args[QStringLiteral("model-w")] = nmW;
    args[QStringLiteral("model-w-unit")] = nmWUnit;
    args[QStringLiteral("model-h")] = nmH;
    args[QStringLiteral("depth")] = nmN;
    config[QStringLiteral("args")] = args;

    std::cerr << "[lasagna] request settings:"
              << " seed=(" << cx << "," << cy << "," << cz << ")"
              << " w=" << nmW << " " << nmWUnit.toStdString() << " h=" << nmH
              << " windings/depth=" << nmN
              << " offset=" << offsetVal << std::endl;

    if (isOffsetMode && activeRow) {
        outputNameMode = QStringLiteral("offset");
        std::cerr << "[lasagna] offset mode: offset=" << offsetVal
                  << " outputName=" << rootName.toStdString() << "_off<N>" << std::endl;
    }

    // The point collections go by UUID: the service reads them from the store.
    if (state && state->pointCollection()) {
        const auto& cols = state->pointCollection()->getAllCollections();
        if (!cols.empty()) {
            state->savePointCollections();
            QJsonArray uuids;
            for (const auto& [cid, col] : cols) uuids.append(QString::fromStdString(col.uuid));
            config[QStringLiteral("corr_point_collections")] = uuids;
            std::cerr << "[lasagna] sending " << cols.size() << " point collection(s) by UUID" << std::endl;
        }
    }

    if (state && state->currentVolume()) {
        try {
            double vs = state->currentVolume()->voxelSize();
            if (std::isfinite(vs) && vs > 0.0) {
                config[QStringLiteral("voxel_size_um")] = vs;
            }
        } catch (...) {
        }
    }
    const QJsonArray volumeShapeZyx = volumeShapeZyxForState(state);
    const QString modelInit = args[QStringLiteral("model-init")].toString().trimmed().toLower();
    if (!isNewModel && modelInit == QStringLiteral("model") && !hasModel) {
        auto msg = tr("Re-optimize needs a surface made by a Lasagna fit (one with its model in the object store).");
        std::cerr << "[lasagna] " << msg.toStdString() << " surface=" << activeUuid << std::endl;
        showStatus(msg, 7000);
        return SubmissionResult::failure(msg);
    }

    QJsonObject request;
    request[QStringLiteral("data_input")] = dataInput;
    request[QStringLiteral("single_segment")] = true;
    request[QStringLiteral("config_name")] = QFileInfo(configPath).fileName();
    request[QStringLiteral("output_name")] = rootName;
    request[QStringLiteral("output_name_mode")] = outputNameMode;
    // Stored surfaces and point collections are read, and results bound, in the
    // open volume's voxels: its store volume at its working level's scale.
    request[QStringLiteral("volume_uuid")] = QString::fromStdString(_surfaceCatalog->voxels().volume);
    request[QStringLiteral("voxel_scale")] = _surfaceCatalog->voxels().scale;
    if (!volumeShapeZyx.isEmpty()) {
        request[QStringLiteral("volume_shape_zyx")] = volumeShapeZyx;
    }
    QJsonObject jobSpec;
    QJsonArray linkedSurfaces;
    if (activeRow) {
        // The job reads the revision on screen now; earlier links read their current one.
        std::optional<std::string> revision;
        _surfaceCatalog->session()->read(*_surfaceCatalog->attachment(), [&](const auto& db, const auto& txn, const auto&) {
            if (const auto head = db.currentGeometry(txn, activeUuid)) revision = head->uuid;
        });
        linkedSurfaces = linkedSurfacesOf(*activeRow, *_surfaceCatalog);
        for (qsizetype i = 0; i < linkedSurfaces.size(); ++i) {
            QJsonObject link = linkedSurfaces[i].toObject();
            if (link[QStringLiteral("uuid")].toString() == QString::fromStdString(activeUuid) && revision) {
                link[QStringLiteral("revision")] = QString::fromStdString(*revision);
                linkedSurfaces[i] = link;
            }
        }
    }
    if (!isNewModel && hasModel) {
        jobSpec[QStringLiteral("model")] = surfaceRef(activeUuid, activeRow->record.name);
    }
    if (!linkedSurfaces.isEmpty()) {
        QJsonArray externalSurfaces;
        for (const QJsonValue& value : linkedSurfaces) {
            QJsonObject surface = value.toObject();
            surface[QStringLiteral("offset")] = offsetVal;
            externalSurfaces.append(surface);
        }
        config[QStringLiteral("external_surfaces")] = externalSurfaces;
    }
    jobSpec[QStringLiteral("config")] = config;
    jobSpec[QStringLiteral("linked_surfaces")] = linkedSurfaces;
    if (!volumeShapeZyx.isEmpty()) {
        jobSpec[QStringLiteral("volume_shape_zyx")] = volumeShapeZyx;
    }
    request[QStringLiteral("config")] = config;
    request[QStringLiteral("job_spec")] = jobSpec;
    const QStringList linkedSurfaceNames = linkedSurfaceNamesFromJobSpec(jobSpec);
    updateCompactLinkedSurfaceTable(linkedSurfaceNames);
    if (_batchWindow) {
        _batchWindow->setLinkedSurfaceNames(linkedSurfaceNames);
    }

    std::cerr << "[lasagna] request prep:"
              << " mode=" << lasagnaModeDebugName(static_cast<int>(launchMode)).toStdString()
              << " elapsed=" << (static_cast<double>(prepTimer.elapsed()) / 1000.0) << "s"
              << " surface=" << activeUuid
              << " model=" << (jobSpec.contains(QStringLiteral("model")) ? "yes" : "no")
              << std::endl;

    mgr.startOptimization(request);
    showStatus(
        tr("Lasagna optimization started. Output: the next %1 of %2")
            .arg(outputNameMode == QStringLiteral("offset") ? tr("offset") : tr("version"), rootName),
        3000);
    return SubmissionResult::success();
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
    QString widthUnit = settings.value(
        QStringLiteral("lasagna_new_model_width_unit"),
        QStringLiteral("voxels")).toString().trimmed().toLower();
    if (widthUnit != QStringLiteral("wraps")) {
        widthUnit = QStringLiteral("voxels");
    }
    if (_widthUnitCombo) {
        const QSignalBlocker b(_widthUnitCombo);
        const int idx = _widthUnitCombo->findData(widthUnit);
        _widthUnitCombo->setCurrentIndex(idx >= 0 ? idx : 0);
    }
    if (_widthSpin) {
        const QSignalBlocker b(_widthSpin);
        if (widthUnit == QStringLiteral("wraps")) {
            _widthSpin->setDecimals(3);
            _widthSpin->setSingleStep(0.25);
            _widthSpin->setValue(
                settings.value(QStringLiteral("lasagna_new_model_width_wraps"), 1.0).toDouble());
        } else {
            _widthSpin->setDecimals(0);
            _widthSpin->setSingleStep(64.0);
            _widthSpin->setValue(
                settings.value(QStringLiteral("lasagna_new_model_width"), 2048).toDouble());
        }
    }
    if (_heightSpin) {
        const QSignalBlocker b(_heightSpin);
        _heightSpin->setValue(settings.value(QStringLiteral("lasagna_new_model_height"), 2048).toInt());
    }
    if (_windingsSpin) {
        const QSignalBlocker b(_windingsSpin);
        _windingsSpin->setValue(settings.value(QStringLiteral("lasagna_new_model_windings"), 3).toInt());
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

    // Auto-reconnect to saved external service (triggers dataset fetch via serviceStarted)
    if (_connectionMode == 1 && !_externalHost.isEmpty() && _externalPort > 0) {
        LasagnaServiceManager::instance().connectToExternal(_externalHost, _externalPort);
    }

    // Offset settings
    _offsetConfigFilePath = settings.value(QStringLiteral("lasagna_offset_config_file_path"), QString()).toString();
    if (_offsetValueSpin) {
        const QSignalBlocker b(_offsetValueSpin);
        _offsetValueSpin->setValue(
            settings.value(QStringLiteral("lasagna_offset_value"), 1.0).toDouble());
    }
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
    if (!_offsetConfigFilePath.isEmpty()) {
        QFileInfo fi(_offsetConfigFilePath);
        if (fi.exists()) {
            populateConfigCombo(_offsetConfigCombo, fi.absolutePath(), fi.fileName(), _offsetConfigFilePath);
        } else if (_offsetConfigCombo) {
            _offsetConfigCombo->clear();
            _offsetConfigCombo->addItem(fi.fileName(), _offsetConfigFilePath);
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
    if (_offsetGroup) {
        _offsetGroup->setExpanded(
            settings.value(QStringLiteral("group_lasagna_offset_expanded"), false).toBool());
    }

    _restoringSettings = false;
    syncCompactConfigCombos();
    syncCompactStatusFromFull();
}

void SegmentationLasagnaPanel::syncUiState(bool /*editingEnabled*/, bool optimizing)
{
    if (_dataInputEdit) {
        const QSignalBlocker b(_dataInputEdit);
        _dataInputEdit->setText(_lasagnaDataInputPath);
    }

    if (_newModelBtn) _newModelBtn->setEnabled(!optimizing);
    if (_reoptBtn) _reoptBtn->setEnabled(!optimizing);
    if (_offsetBtn) _offsetBtn->setEnabled(!optimizing);
    if (_stopBtn) _stopBtn->setEnabled(optimizing);
    syncCompactStatusFromFull();
    updateLinkedSurfaceTables();
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
    int addedCount = 0;
    for (const auto& fileName : jsonFiles) {
        QString fullPath = d.absoluteFilePath(fileName);
        combo->addItem(fileName, fullPath);
        if (fileName == selectName) {
            selectIndex = addedCount;
        }
        ++addedCount;
    }

    if (selectIndex >= 0) {
        combo->setCurrentIndex(selectIndex);
        outPath = combo->currentData().toString();
    } else if (combo->count() > 0) {
        combo->setCurrentIndex(0);
        outPath = combo->currentData().toString();
    }
    syncCompactConfigCombos();
}

// ---------------------------------------------------------------------------
// Config JSON — reads file from disk on demand
// ---------------------------------------------------------------------------

QString SegmentationLasagnaPanel::lasagnaConfigText() const
{
    const QString& path = (_lasagnaMode == 1) ? _newModelConfigFilePath
                        : (_lasagnaMode == 3) ? _offsetConfigFilePath
                        : _reoptConfigFilePath;
    if (path.isEmpty()) return {};
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return {};
    QByteArray raw = f.readAll();
    return QString::fromUtf8(raw);
}

nlohmann::json SegmentationLasagnaPanel::lasagnaConfigJson() const
{
    QString text = lasagnaConfigText().trimmed();
    if (text.isEmpty()) return nlohmann::json{};

    try {
        QByteArray utf8 = text.toUtf8();
        nlohmann::json parsed = nlohmann::json::parse(
            std::string_view(utf8.constData(), utf8.size()));
        if (parsed.is_object()) return parsed;
    } catch (const std::exception& e) {
        std::cerr << "[lasagna] ERROR: invalid config JSON: " << e.what() << std::endl;
    }

    return nlohmann::json{};
}

// ---------------------------------------------------------------------------
// Lasagna mode helpers
// ---------------------------------------------------------------------------

double SegmentationLasagnaPanel::newModelWidth() const
{
    return _widthSpin ? _widthSpin->value() : 2048.0;
}

QString SegmentationLasagnaPanel::newModelWidthUnit() const
{
    if (!_widthUnitCombo) {
        return QStringLiteral("voxels");
    }
    const QString unit = _widthUnitCombo->currentData().toString();
    return unit == QStringLiteral("wraps") ? unit : QStringLiteral("voxels");
}

int SegmentationLasagnaPanel::newModelHeight() const
{
    return _heightSpin ? _heightSpin->value() : 2048;
}

int SegmentationLasagnaPanel::newModelWindings() const
{
    return _windingsSpin ? _windingsSpin->value() : 3;
}

QString SegmentationLasagnaPanel::seedPointText() const
{
    return _seedEdit ? _seedEdit->text().trimmed() : QString();
}

QString SegmentationLasagnaPanel::newModelOutputName() const
{
    return _outputNameEdit ? _outputNameEdit->text().trimmed() : QString();
}

double SegmentationLasagnaPanel::offsetValue() const
{
    return _offsetValueSpin ? _offsetValueSpin->value() : 1.0;
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
    if (_externalWidget) _externalWidget->setVisible(external);

    if (_dataInputStack) {
        _dataInputStack->setCurrentIndex(external ? 1 : 0);
    }
    // When in external mode with no datasets yet, disable combo + action buttons
    if (external && _datasetCombo) {
        bool hasDatasets = (_datasetCombo->count() > 0);
        _datasetCombo->setEnabled(hasDatasets);
        if (_newModelBtn) _newModelBtn->setEnabled(hasDatasets);
        if (_reoptBtn) _reoptBtn->setEnabled(hasDatasets);
        if (_offsetBtn) _offsetBtn->setEnabled(hasDatasets);
    } else {
        // Internal mode: re-enable controls
        if (_datasetCombo) _datasetCombo->setEnabled(true);
        if (_newModelBtn) _newModelBtn->setEnabled(true);
        if (_reoptBtn) _reoptBtn->setEnabled(true);
        if (_offsetBtn) _offsetBtn->setEnabled(true);
    }

    if (external && !_restoringSettings && !_externalHost.isEmpty() && _externalPort > 0) {
        LasagnaServiceManager::instance().connectToExternal(_externalHost, _externalPort);
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

    // Auto-connect (datasets are fetched via the permanent serviceStarted handler)
    LasagnaServiceManager::instance().connectToExternal(host, port);
}
