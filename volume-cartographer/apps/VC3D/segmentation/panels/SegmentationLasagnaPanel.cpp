#include "SegmentationLasagnaPanel.hpp"

#include "CState.hpp"
#include "LasagnaBatchWindow.hpp"
#include "LasagnaServiceManager.hpp"
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

#include "utils/Json.hpp"

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

double bytesToMiB(qint64 bytes)
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

QString md5Ref(const QByteArray& bytes)
{
    return QStringLiteral("md5:%1").arg(QString::fromLatin1(
        QCryptographicHash::hash(bytes, QCryptographicHash::Md5).toHex()));
}

QJsonObject makeObjectRef(const QString& type, const QString& name, const QString& hash)
{
    QJsonObject ref;
    ref[QStringLiteral("type")] = type;
    ref[QStringLiteral("name")] = name;
    ref[QStringLiteral("hash")] = hash;
    return ref;
}

QJsonObject modelArtifactForPath(const std::filesystem::path& modelPath,
                                 const QString& refName,
                                 qint64* rawBytes)
{
    QFile f(QString::fromStdString(modelPath.string()));
    if (!f.open(QIODevice::ReadOnly)) {
        return {};
    }
    const QByteArray bytes = f.readAll();
    if (rawBytes) {
        *rawBytes += bytes.size();
    }
    QJsonObject upload;
    upload[QStringLiteral("object")] = makeObjectRef(
        QStringLiteral("lasagna_model"), refName, md5Ref(bytes));
    upload[QStringLiteral("data")] = QString::fromLatin1(bytes.toBase64());
    return upload;
}

QJsonObject segmentArtifactForPath(const std::filesystem::path& segPath,
                                   qint64* rawBytes,
                                   int* fileCount)
{
    std::vector<std::filesystem::path> files;
    std::error_code ec;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(segPath, ec)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    QJsonObject encodedFiles;
    QByteArray manifest;
    for (const auto& path : files) {
        QFile f(QString::fromStdString(path.string()));
        if (!f.open(QIODevice::ReadOnly)) {
            return {};
        }
        const QByteArray bytes = f.readAll();
        const auto relPath = std::filesystem::relative(path, segPath, ec);
        if (ec) {
            return {};
        }
        const QString rel = QString::fromStdString(relPath.generic_string());
        const QString fileHash = md5Ref(bytes);
        manifest.append(rel.toUtf8());
        manifest.append('\t');
        manifest.append(fileHash.toUtf8());
        manifest.append('\n');
        encodedFiles[rel] = QString::fromLatin1(bytes.toBase64());
        if (rawBytes) {
            *rawBytes += bytes.size();
        }
        if (fileCount) {
            ++(*fileCount);
        }
    }

    const QString segName = QString::fromStdString(segPath.filename().string());
    QJsonObject upload;
    upload[QStringLiteral("object")] = makeObjectRef(
        QStringLiteral("tifxyz_segment"), segName, md5Ref(manifest));
    upload[QStringLiteral("files")] = encodedFiles;
    return upload;
}

QJsonArray linkedSurfacesFromMeta(const std::filesystem::path& segPath)
{
    QFile metaFile(QString::fromStdString((segPath / "meta.json").string()));
    if (!metaFile.open(QIODevice::ReadOnly)) {
        return {};
    }
    const QJsonObject meta = QJsonDocument::fromJson(metaFile.readAll()).object();
    const QJsonObject job = meta[QStringLiteral("lasagna_job")].toObject();
    QJsonArray links = job[QStringLiteral("linked_surfaces")].toArray();
    if (!links.isEmpty()) {
        return links;
    }
    links = meta[QStringLiteral("job_spec")].toObject()[QStringLiteral("linked_surfaces")].toArray();
    if (!links.isEmpty()) {
        return links;
    }
    return meta[QStringLiteral("linked_surfaces")].toArray();
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

std::filesystem::path outputSegmentsPathForState(CState* state)
{
    if (!state || !state->vpkg()) {
        return {};
    }
    std::filesystem::path path = state->vpkg()->outputSegmentsPath();
    if (path.empty()) {
        path = state->vpkg()->findSegmentPathByName(
            state->vpkg()->getSegmentationDirectory());
    }
    if (path.empty()) {
        const auto vpkgRoot = std::filesystem::path(state->vpkg()->getVolpkgDirectory());
        path = vpkgRoot / "paths";
    }
    return std::filesystem::absolute(path).lexically_normal();
}

std::filesystem::path selectedSegmentPathForState(CState* state)
{
    const std::filesystem::path outputSegmentsPath = outputSegmentsPathForState(state);
    if (!state) {
        return {};
    }
    auto activeSurface = std::dynamic_pointer_cast<QuadSurface>(state->surface("segmentation"));
    if (!activeSurface || activeSurface->path.empty()) {
        return {};
    }
    std::filesystem::path segPath = activeSurface->path;
    if (segPath.is_relative() && !outputSegmentsPath.empty()) {
        segPath = outputSegmentsPath / segPath.filename();
    }
    return segPath;
}
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

        dimLayout->addWidget(new QLabel(tr("N:"), dimWidget));
        _windingsSpin = new QSpinBox(dimWidget);
        _windingsSpin->setRange(1, 999);
        _windingsSpin->setValue(3);
        _windingsSpin->setSingleStep(1);
        _windingsSpin->setToolTip(tr("Number of windings"));
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
        _outputNameEdit->setToolTip(tr("Output name prefix (auto-versioned, e.g. mysheet → mysheet_v001.tifxyz)"));
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

    _offsetGroup->addRow(tr("Window:"), [&](QHBoxLayout* row) {
        _windowSizeSpin = new QSpinBox(offsetContent);
        _windowSizeSpin->setRange(0, 100000);
        _windowSizeSpin->setSingleStep(1000);
        _windowSizeSpin->setValue(5000);
        _windowSizeSpin->setToolTip(tr("Window size in fullres voxels (0 = no windowing)"));
        row->addWidget(_windowSizeSpin);
    }, tr("Split large surfaces into windows for memory efficiency. 0 = process whole surface."));

    _offsetGroup->addRow(tr("Overlap:"), [&](QHBoxLayout* row) {
        _windowOverlapSpin = new QSpinBox(offsetContent);
        _windowOverlapSpin->setRange(0, 50000);
        _windowOverlapSpin->setSingleStep(100);
        _windowOverlapSpin->setValue(500);
        _windowOverlapSpin->setToolTip(tr("Overlap between windows in fullres voxels"));
        row->addWidget(_windowOverlapSpin);
    }, tr("Overlap ensures smooth transitions at window boundaries."));

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
    connect(_widthSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) {
        writeSetting(QStringLiteral("lasagna_new_model_width"), v);
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
    connect(_windowSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) {
        writeSetting(QStringLiteral("lasagna_window_size"), v);
    });
    connect(_windowOverlapSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) {
        writeSetting(QStringLiteral("lasagna_window_overlap"), v);
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
            [this](const QString& outputDir) {
        if (_stopBtn) _stopBtn->setEnabled(false);
        if (_newModelBtn) _newModelBtn->setEnabled(true);
        if (_reoptBtn) _reoptBtn->setEnabled(true);
        if (_offsetBtn) _offsetBtn->setEnabled(true);
        if (_progressBar) _progressBar->setVisible(false);
        if (_progressLabel) {
            _progressLabel->setText(tr("Optimization finished. Output: %1").arg(outputDir));
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
    const std::filesystem::path segPath = selectedSegmentPathForState(_state);
    if (segPath.empty()) {
        return {};
    }

    const QStringList storedNames = linkedSurfaceNamesFromRefs(linkedSurfacesFromMeta(segPath));
    const bool selectedLasagnaModel = std::filesystem::exists(segPath / "model.pt");
    if (selectedLasagnaModel || !storedNames.isEmpty()) {
        return storedNames;
    }

    return {QString::fromStdString(segPath.filename().string())};
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

void SegmentationLasagnaPanel::startOptimization(CState* state, QStatusBar* statusBar)
{
    startOptimizationWithOverrides(state, statusBar, -1, QString(), false, 0, 0, 0);
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
                    startOptimizationWithOverrides(
                        state, statusBar, static_cast<int>(mode), configPath, true, seedX, seedY, seedZ);
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

    startOptimizationWithOverrides(
        state, statusBar, static_cast<int>(mode), configPath, true, seedX, seedY, seedZ);
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
                                                      int timeoutMs)
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
    QMessageBox::warning(this, tr("Lasagna config error"), message);
}

bool SegmentationLasagnaPanel::validateLasagnaConfigPath(const QString& configPath,
                                                         QStatusBar* statusBar)
{
    if (configPath.isEmpty()) {
        showLasagnaConfigError(tr("No Lasagna config file selected."), statusBar, 5000);
        return false;
    }

    const QFileInfo configInfo(configPath);
    if (!configInfo.exists()) {
        showLasagnaConfigError(tr("Lasagna config file not found: %1").arg(configPath),
                               statusBar,
                               7000);
        return false;
    }
    if (!configInfo.isFile()) {
        showLasagnaConfigError(tr("Lasagna config path is not a file: %1").arg(configPath),
                               statusBar,
                               7000);
        return false;
    }

    return true;
}

void SegmentationLasagnaPanel::startOptimizationWithOverrides(CState* state,
                                                              QStatusBar* statusBar,
                                                              int modeOverride,
                                                              const QString& configPathOverride,
                                                              bool hasSeedOverride,
                                                              int seedX,
                                                              int seedY,
                                                              int seedZ)
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

    if (!validateLasagnaConfigPath(configPath, statusBar)) {
        return;
    }

    if (mgr.isExternal()) {
        if (!mgr.isRunning()) {
            auto msg = tr("External service not connected. Select a service or check host/port.");
            std::cerr << "[lasagna] " << msg.toStdString() << std::endl;
            showStatus(msg, 5000);
            return;
        }
    } else {
        if (!mgr.ensureServiceRunning()) {
            auto msg = tr("Failed to start lasagna service: %1").arg(mgr.lastError());
            std::cerr << "[lasagna] " << msg.toStdString() << std::endl;
            showStatus(msg, 5000);
            return;
        }
    }

    QElapsedTimer prepTimer;
    prepTimer.start();

    std::filesystem::path outputSegmentsPath;
    if (state && state->vpkg()) {
        outputSegmentsPath = state->vpkg()->outputSegmentsPath();
        if (outputSegmentsPath.empty()) {
            outputSegmentsPath = state->vpkg()->findSegmentPathByName(
                state->vpkg()->getSegmentationDirectory());
        }
        if (outputSegmentsPath.empty()) {
            auto vpkgRoot = std::filesystem::path(state->vpkg()->getVolpkgDirectory());
            outputSegmentsPath = vpkgRoot / "paths";
        }
        outputSegmentsPath = std::filesystem::absolute(outputSegmentsPath).lexically_normal();
    }

    std::filesystem::path segPath;
    if (state) {
        auto activeSurface = std::dynamic_pointer_cast<QuadSurface>(state->surface("segmentation"));
        if (activeSurface && !activeSurface->path.empty()) {
            segPath = activeSurface->path;
            if (segPath.is_relative() && !outputSegmentsPath.empty()) {
                segPath = outputSegmentsPath / segPath.filename();
            }
        }
    }

    const bool isOffsetMode = (launchMode == LasagnaMode::Offset);

    QString modelPath;
    if (!segPath.empty()) {
        auto modelFile = segPath / "model.pt";
        if (std::filesystem::exists(modelFile)) {
            try {
                modelPath = QString::fromStdString(std::filesystem::canonical(modelFile).string());
            } catch (const std::filesystem::filesystem_error&) {
            }
        }
    }

    QString dataInput = lasagnaDataInputPath();
    if (dataInput.isEmpty()) {
        auto msg = tr("No data input path set. Set the zarr path in the Lasagna Model panel.");
        std::cerr << "[lasagna] " << msg.toStdString() << std::endl;
        showStatus(msg, 5000);
        return;
    }

    QString outputDir;
    if (!outputSegmentsPath.empty()) {
        outputDir = QString::fromStdString(outputSegmentsPath.string());
    } else if (!segPath.empty()) {
        outputDir = QString::fromStdString(
            std::filesystem::absolute(segPath.parent_path()).lexically_normal().string());
    }

    const std::string tifxyzSuffix = ".tifxyz";
    std::string rootName = "new_model";
    QString outputName;
    {
        if (isNewModel) {
            QString nmName = newModelOutputName();
            if (!nmName.isEmpty()) {
                rootName = nmName.toStdString();
            }
        } else if (!segPath.empty()) {
            auto segName = segPath.filename().string();
            std::string baseName = segName;
            if (baseName.size() > tifxyzSuffix.size() &&
                baseName.compare(baseName.size() - tifxyzSuffix.size(),
                                 tifxyzSuffix.size(), tifxyzSuffix) == 0) {
                baseName = baseName.substr(0, baseName.size() - tifxyzSuffix.size());
            }
            rootName = baseName;
            if (rootName.size() > 5) {
                auto pos = rootName.rfind("_v");
                if (pos != std::string::npos && pos + 2 < rootName.size()) {
                    bool allDigits = true;
                    for (size_t i = pos + 2; i < rootName.size(); ++i) {
                        if (!std::isdigit(static_cast<unsigned char>(rootName[i]))) {
                            allDigits = false;
                            break;
                        }
                    }
                    if (allDigits) {
                        rootName = rootName.substr(0, pos);
                    }
                }
            }
            auto pos = rootName.find("_off");
            if (pos != std::string::npos) {
                rootName = rootName.substr(0, pos);
            }
        }

        int maxVersion = 0;
        if (!outputDir.isEmpty()) {
            std::error_code ec;
            for (auto& entry : std::filesystem::directory_iterator(outputDir.toStdString(), ec)) {
                auto name = entry.path().filename().string();
                std::string prefix = rootName + "_v";
                if (name.size() > prefix.size() + tifxyzSuffix.size() &&
                    name.compare(0, prefix.size(), prefix) == 0 &&
                    name.compare(name.size() - tifxyzSuffix.size(),
                                 tifxyzSuffix.size(), tifxyzSuffix) == 0) {
                    auto numStr = name.substr(prefix.size(),
                        name.size() - prefix.size() - tifxyzSuffix.size());
                    bool allDigits = true;
                    for (auto c : numStr) {
                        if (!std::isdigit(static_cast<unsigned char>(c))) {
                            allDigits = false;
                        }
                    }
                    if (allDigits && !numStr.empty()) {
                        int v = std::stoi(numStr);
                        if (v > maxVersion) {
                            maxVersion = v;
                        }
                    }
                }
            }
        }
        const QString versionPrefix = QString::fromStdString(rootName + "_v");
        for (const QString& reserved : std::as_const(_submittedOutputNames)) {
            if (!reserved.startsWith(versionPrefix) ||
                !reserved.endsWith(QString::fromStdString(tifxyzSuffix))) {
                continue;
            }
            const QString numStr = reserved.mid(
                versionPrefix.size(),
                reserved.size() - versionPrefix.size() - static_cast<int>(tifxyzSuffix.size()));
            bool ok = false;
            const int version = numStr.toInt(&ok);
            if (ok && version > maxVersion) {
                maxVersion = version;
            }
        }
        char numBuf[16];
        std::snprintf(numBuf, sizeof(numBuf), "_v%03d", maxVersion + 1);
        outputName = QString::fromStdString(rootName + numBuf + ".tifxyz");
    }

    QJsonObject config;
    QString configText;
    {
        QFile f(configPath);
        if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
            auto msg = tr("Cannot read Lasagna config %1: %2")
                .arg(configPath, f.errorString());
            showLasagnaConfigError(msg, statusBar, 7000);
            return;
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
            showLasagnaConfigError(msg, statusBar, 7000);
            return;
        }
        if (!doc.isObject()) {
            auto msg = tr("Invalid Lasagna config JSON: top-level value must be an object.");
            showLasagnaConfigError(msg, statusBar, 7000);
            return;
        }
        config = doc.object();
    }

    int nmW = newModelWidth();
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
            return;
        }
        cx = static_cast<int>(focus->p[0]);
        cy = static_cast<int>(focus->p[1]);
        cz = static_cast<int>(focus->p[2]);
    }

    const double offsetVal = offsetValue();
    const int size = windowSize();
    const int overlap = windowOverlap();

    QJsonObject args = config[QStringLiteral("args")].toObject();
    // VC3D is transport only. Do not add config-semantic branching here.
    // Config interpretation belongs in fit_service.py / fit.py.
    args[QStringLiteral("seed")] = QJsonArray{cx, cy, cz};
    args[QStringLiteral("model-w")] = nmW;
    args[QStringLiteral("model-h")] = nmH;
    args[QStringLiteral("windings")] = nmN;
    config[QStringLiteral("args")] = args;

    std::cerr << "[lasagna] request settings:"
              << " seed=(" << cx << "," << cy << "," << cz << ")"
              << " w=" << nmW << " h=" << nmH
              << " windings=" << nmN
              << " offset=" << offsetVal
              << " window_size=" << size
              << " window_overlap=" << overlap << std::endl;

    if (isOffsetMode && !segPath.empty()) {

        if (size > 0 && !outputDir.isEmpty()) {
            int offIdx = 1;
            std::error_code ec2;
            for (bool collision = true; collision; ++offIdx) {
                collision = false;
                std::string offPrefix = rootName + "_off" + std::to_string(offIdx) + "_w";
                const QString reservedPrefix = QString::fromStdString(
                    rootName + "_off" + std::to_string(offIdx));
                if (_submittedOutputNames.contains(reservedPrefix)) {
                    collision = true;
                    continue;
                }
                for (auto& entry : std::filesystem::directory_iterator(outputDir.toStdString(), ec2)) {
                    auto name = entry.path().filename().string();
                    if (name.size() > offPrefix.size() + tifxyzSuffix.size() &&
                        name.compare(0, offPrefix.size(), offPrefix) == 0 &&
                        name.compare(name.size() - tifxyzSuffix.size(),
                                     tifxyzSuffix.size(), tifxyzSuffix) == 0) {
                        collision = true;
                        break;
                    }
                }
            }
            outputName = QString::fromStdString(rootName + "_off" + std::to_string(offIdx - 1));
        }

        std::cerr << "[lasagna] offset mode: offset=" << offsetVal
                  << " window_size=" << size
                  << " window_overlap=" << overlap
                  << " outputName=" << outputName.toStdString() << std::endl;
    }

    if (state && state->pointCollection()) {
        const auto& cols = state->pointCollection()->getAllCollections();
        if (!cols.empty()) {
            utils::Json corrJson;
            utils::Json colsJson = utils::Json::object();
            for (const auto& [cid, col] : cols) {
                utils::Json colJson;
                to_json(colJson, col);
                colsJson[std::to_string(cid)] = colJson;
            }
            corrJson["collections"] = colsJson;
            QJsonDocument corrDoc = QJsonDocument::fromJson(
                QByteArray::fromStdString(corrJson.dump()));
            if (corrDoc.isObject()) {
                config[QStringLiteral("corr_points")] = corrDoc.object();
                std::cerr << "[lasagna] injected " << cols.size()
                          << " point collection(s) as corr_points" << std::endl;
            }
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

    QJsonObject request;
    request[QStringLiteral("data_input")] = dataInput;
    request[QStringLiteral("single_segment")] = true;
    request[QStringLiteral("copy_model")] = true;
    request[QStringLiteral("config_name")] = QFileInfo(configPath).fileName();
    if (!outputName.isEmpty()) {
        request[QStringLiteral("output_name")] = outputName;
    }
    if (size > 0) {
        request[QStringLiteral("window_size")] = size;
        request[QStringLiteral("window_overlap")] = overlap;
    }

    QJsonObject jobSpec;
    QJsonArray objectUploads;
    QJsonArray linkedSurfaces;
    const bool sendModelData = !isNewModel && !modelPath.isEmpty();
    const bool sendTifxyz = !segPath.empty();
    qint64 rawTifxyzBytes = 0;
    int tifxyzFileCount = 0;
    qint64 rawModelBytes = 0;
    std::cerr << "[lasagna] request payload: send_model="
              << (sendModelData ? "yes" : "no")
              << " send_tifxyz=" << (sendTifxyz ? "yes" : "no")
              << std::endl;

    auto appendUploadIfNew = [&](const QJsonObject& upload) {
        const QJsonObject ref = upload[QStringLiteral("object")].toObject();
        if (ref.isEmpty()) {
            return;
        }
        const QString key = ref[QStringLiteral("type")].toString() + QStringLiteral("\n")
            + ref[QStringLiteral("name")].toString() + QStringLiteral("\n")
            + ref[QStringLiteral("hash")].toString();
        for (const QJsonValue& existingValue : objectUploads) {
            const QJsonObject existingRef = existingValue.toObject()[QStringLiteral("object")].toObject();
            const QString existingKey = existingRef[QStringLiteral("type")].toString() + QStringLiteral("\n")
                + existingRef[QStringLiteral("name")].toString() + QStringLiteral("\n")
                + existingRef[QStringLiteral("hash")].toString();
            if (existingKey == key) {
                return;
            }
        }
        objectUploads.append(upload);
    };

    if (sendTifxyz) {
        QJsonObject currentSegmentUpload = segmentArtifactForPath(segPath, &rawTifxyzBytes, &tifxyzFileCount);
        if (currentSegmentUpload.isEmpty()) {
            auto msg = tr("Cannot pack selected tifxyz segment for Lasagna artifact sync.");
            std::cerr << "[lasagna] " << msg.toStdString() << std::endl;
            showStatus(msg, 5000);
            return;
        }

        linkedSurfaces = linkedSurfacesFromMeta(segPath);
        const bool selectedLasagnaModel = std::filesystem::exists(segPath / "model.pt");
        if (linkedSurfaces.isEmpty() && !selectedLasagnaModel) {
            linkedSurfaces.append(currentSegmentUpload[QStringLiteral("object")].toObject());
            appendUploadIfNew(currentSegmentUpload);
        } else if (!outputSegmentsPath.empty()) {
            for (const QJsonValue& value : linkedSurfaces) {
                const QJsonObject ref = value.toObject();
                const QString name = ref[QStringLiteral("name")].toString();
                if (name.isEmpty()) {
                    continue;
                }
                std::filesystem::path localPath = outputSegmentsPath / name.toStdString();
                if (std::filesystem::exists(localPath) && std::filesystem::is_directory(localPath)) {
                    qint64 extraBytes = 0;
                    int extraFiles = 0;
                    QJsonObject upload = segmentArtifactForPath(localPath, &extraBytes, &extraFiles);
                    if (!upload.isEmpty() &&
                        upload[QStringLiteral("object")].toObject()[QStringLiteral("hash")].toString()
                            == ref[QStringLiteral("hash")].toString()) {
                        rawTifxyzBytes += extraBytes;
                        tifxyzFileCount += extraFiles;
                        appendUploadIfNew(upload);
                    }
                }
            }
        }
    }

    if (sendModelData) {
        const QString modelRefName = QString::fromStdString(segPath.filename().string())
            + QStringLiteral("/model.pt");
        QJsonObject modelUpload = modelArtifactForPath(
            std::filesystem::path(modelPath.toStdString()), modelRefName, &rawModelBytes);
        if (modelUpload.isEmpty()) {
            auto msg = tr("Cannot read model file: %1").arg(modelPath);
            std::cerr << "[lasagna] " << msg.toStdString() << std::endl;
            showStatus(msg, 5000);
            return;
        }
        jobSpec[QStringLiteral("model")] = modelUpload[QStringLiteral("object")].toObject();
        appendUploadIfNew(modelUpload);
    }
    if (!linkedSurfaces.isEmpty()) {
        QJsonArray externalSurfaces;
        for (const QJsonValue& value : linkedSurfaces) {
            QJsonObject surface = value.toObject();
            if (surface.isEmpty()) {
                continue;
            }
            surface[QStringLiteral("offset")] = offsetVal;
            externalSurfaces.append(surface);
        }
        config[QStringLiteral("external_surfaces")] = externalSurfaces;
    }
    jobSpec[QStringLiteral("config")] = config;
    jobSpec[QStringLiteral("linked_surfaces")] = linkedSurfaces;
    request[QStringLiteral("config")] = config;
    request[QStringLiteral("job_spec")] = jobSpec;
    request[QStringLiteral("_objects")] = objectUploads;
    const QStringList linkedSurfaceNames = linkedSurfaceNamesFromJobSpec(jobSpec);
    updateCompactLinkedSurfaceTable(linkedSurfaceNames);
    if (_batchWindow) {
        _batchWindow->setLinkedSurfaceNames(linkedSurfaceNames);
    }

    std::cerr << "[lasagna] request prep:"
              << " mode=" << lasagnaModeDebugName(static_cast<int>(launchMode)).toStdString()
              << " elapsed=" << (static_cast<double>(prepTimer.elapsed()) / 1000.0) << "s"
              << " tifxyz_files=" << tifxyzFileCount
              << " tifxyz_raw=" << bytesToMiB(rawTifxyzBytes) << " MiB"
              << " model_raw=" << bytesToMiB(rawModelBytes) << " MiB"
              << std::endl;

    mgr.startOptimization(request, outputDir);
    if (!outputName.isEmpty()) {
        _submittedOutputNames.insert(outputName);
    }
    showStatus(
        tr("Lasagna optimization started. Output: %1")
            .arg(outputName),
        3000);
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
    if (_windowSizeSpin) {
        const QSignalBlocker b(_windowSizeSpin);
        _windowSizeSpin->setValue(
            settings.value(QStringLiteral("lasagna_window_size"), 5000).toInt());
    }
    if (_windowOverlapSpin) {
        const QSignalBlocker b(_windowOverlapSpin);
        _windowOverlapSpin->setValue(
            settings.value(QStringLiteral("lasagna_window_overlap"), 500).toInt());
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
    const QString& selectName, QString& outPath,
    bool /*growOnly*/)
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

utils::Json SegmentationLasagnaPanel::lasagnaConfigJson() const
{
    QString text = lasagnaConfigText().trimmed();
    if (text.isEmpty()) return utils::Json{};

    try {
        QByteArray utf8 = text.toUtf8();
        utils::Json parsed = utils::Json::parse(
            std::string_view(utf8.constData(), utf8.size()));
        if (parsed.is_object()) return parsed;
    } catch (const std::exception& e) {
        std::cerr << "[lasagna] ERROR: invalid config JSON: " << e.what() << std::endl;
    }

    return utils::Json{};
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

int SegmentationLasagnaPanel::windowSize() const
{
    return _windowSizeSpin ? _windowSizeSpin->value() : 5000;
}

int SegmentationLasagnaPanel::windowOverlap() const
{
    return _windowOverlapSpin ? _windowOverlapSpin->value() : 500;
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
