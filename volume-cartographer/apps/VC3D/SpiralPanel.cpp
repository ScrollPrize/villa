#include "SpiralPanel.hpp"

#include "SpiralReloadComparison.hpp"
#include "SpiralServiceManager.hpp"
#include "VCSettings.hpp"
#include "elements/CollapsibleSettingsGroup.hpp"
#include "elements/SpiralConfigProfileEditor.hpp"
#include "elements/VolumeSelector.hpp"

#include <QAbstractItemView>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QDialog>
#include <QDir>
#include <QFileDialog>
#include <QFormLayout>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QListWidgetItem>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QSlider>
#include <QToolButton>
#include <QUuid>
#include <QVBoxLayout>

namespace {
const QString kLocalhostProfileId = QStringLiteral("localhost");
}

SpiralPanel::SpiralPanel(SpiralServiceManager* service, QWidget* parent)
    : QWidget(parent), _service(service)
{
    auto* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(4, 4, 4, 4);
    auto* scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    auto* contents = new QWidget(scroll);
    auto* layout = new QVBoxLayout(contents);

    auto makeSection = [this, contents, layout](const QString& title,
                                                const QString& objectName,
                                                const QString& settingsKey) {
        auto* group = new CollapsibleSettingsGroup(title, contents);
        group->setObjectName(objectName);
        layout->addWidget(group);
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        group->setExpanded(settings.value(settingsKey, true).toBool());
        connect(group, &CollapsibleSettingsGroup::toggled, this,
                [settingsKey](bool expanded) {
                    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
                    settings.setValue(settingsKey, expanded);
                });
        return group;
    };

    // ------------------------------------------------------------------
    // Spiral Service (connection) section
    // ------------------------------------------------------------------
    auto* serviceGroup = makeSection(tr("Spiral Service"),
                                     QStringLiteral("spiralServiceGroup"),
                                     QStringLiteral("spiral/groups/service_expanded"));
    auto* serviceContents = new QWidget(serviceGroup->contentWidget());
    auto* serviceForm = new QFormLayout(serviceContents);
    serviceGroup->contentLayout()->addWidget(serviceContents);

    auto* profileRow = new QWidget(serviceContents);
    auto* profileLayout = new QHBoxLayout(profileRow);
    profileLayout->setContentsMargins(0, 0, 0, 0);
    _profileCombo = new QComboBox(profileRow);
    _profileCombo->setObjectName(QStringLiteral("spiralProfileCombo"));
    auto* addLan = new QToolButton(profileRow);
    addLan->setText(QStringLiteral("+LAN"));
    addLan->setToolTip(tr("Add a Remote (LAN) profile: direct HTTP to a service on a trusted network"));
    auto* addSsh = new QToolButton(profileRow);
    addSsh->setText(QStringLiteral("+SSH"));
    addSsh->setToolTip(tr("Add a Remote (SSH) profile: VC3D manages an SSH tunnel to a persistent service"));
    auto* removeProfile = new QToolButton(profileRow);
    removeProfile->setText(QStringLiteral("−"));
    removeProfile->setToolTip(tr("Remove the selected profile"));
    profileLayout->addWidget(_profileCombo, 1);
    profileLayout->addWidget(addLan);
    profileLayout->addWidget(addSsh);
    profileLayout->addWidget(removeProfile);
    serviceForm->addRow(tr("Service"), profileRow);

    _endpointRow = new QWidget(serviceContents);
    auto* endpointLayout = new QHBoxLayout(_endpointRow);
    endpointLayout->setContentsMargins(0, 0, 0, 0);
    _endpointUrl = new QLineEdit(_endpointRow);
    _endpointUrl->setPlaceholderText(QStringLiteral("http://gpu-host:8765"));
    endpointLayout->addWidget(_endpointUrl);
    serviceForm->addRow(tr("Endpoint"), _endpointRow);

    _sshRow = new QWidget(serviceContents);
    auto* sshLayout = new QHBoxLayout(_sshRow);
    sshLayout->setContentsMargins(0, 0, 0, 0);
    _sshDestination = new QLineEdit(_sshRow);
    _sshDestination->setPlaceholderText(tr("[user@]host (uses ~/.ssh/config)"));
    _sshPort = new QSpinBox(_sshRow);
    _sshPort->setRange(1, 65535);
    _sshPort->setValue(8765);
    _sshPort->setToolTip(tr("Loopback port of the persistent service on the host"));
    sshLayout->addWidget(_sshDestination, 1);
    sshLayout->addWidget(new QLabel(tr("port"), _sshRow));
    sshLayout->addWidget(_sshPort);
    serviceForm->addRow(tr("SSH host"), _sshRow);

    _apiKeyRow = new QWidget(serviceContents);
    auto* apiKeyLayout = new QHBoxLayout(_apiKeyRow);
    apiKeyLayout->setContentsMargins(0, 0, 0, 0);
    _apiKey = new QLineEdit(_apiKeyRow);
    _apiKey->setEchoMode(QLineEdit::Password);
    _apiKey->setPlaceholderText(tr("Printed by the service at startup (or set SPIRAL_API_KEY)"));
    apiKeyLayout->addWidget(_apiKey);
    serviceForm->addRow(tr("API key"), _apiKeyRow);

    _mappingRow = new QWidget(serviceContents);
    auto* mappingLayout = new QHBoxLayout(_mappingRow);
    mappingLayout->setContentsMargins(0, 0, 0, 0);
    _mapServiceRoot = new QLineEdit(_mappingRow);
    _mapServiceRoot->setPlaceholderText(tr("service path prefix"));
    _mapLocalRoot = new QLineEdit(_mappingRow);
    _mapLocalRoot->setPlaceholderText(tr("local path prefix"));
    mappingLayout->addWidget(_mapServiceRoot, 1);
    mappingLayout->addWidget(new QLabel(QStringLiteral("→"), _mappingRow));
    mappingLayout->addWidget(_mapLocalRoot, 1);
    _mappingRow->setToolTip(tr("Optional: when both machines mount the same dataset under "
                               "different roots, map service paths to local paths so input "
                               "overlays can be displayed."));
    serviceForm->addRow(tr("Path map"), _mappingRow);

    auto* connectRow = new QWidget(serviceContents);
    auto* connectLayout = new QHBoxLayout(connectRow);
    connectLayout->setContentsMargins(0, 0, 0, 0);
    _connectButton = new QPushButton(tr("Connect"), connectRow);
    _disconnectButton = new QPushButton(tr("Disconnect"), connectRow);
    _disconnectButton->setEnabled(false);
    connectLayout->addWidget(_connectButton);
    connectLayout->addWidget(_disconnectButton);
    connectLayout->addStretch(1);
    serviceForm->addRow(QString(), connectRow);
    _connectionStatus = new QLabel(tr("Disconnected"), serviceContents);
    _connectionStatus->setWordWrap(true);
    serviceForm->addRow(tr("Status"), _connectionStatus);

    // ------------------------------------------------------------------
    // Dataset and fit geometry
    // ------------------------------------------------------------------
    auto* pathsGroup = makeSection(tr("Dataset and fit geometry"),
                                   QStringLiteral("spiralDatasetGeometryGroup"),
                                   QStringLiteral("spiral/groups/dataset_geometry_expanded"));
    auto* pathsContents = new QWidget(pathsGroup->contentWidget());
    auto* pathsForm = new QFormLayout(pathsContents);
    pathsGroup->contentLayout()->addWidget(pathsContents);

    auto* optionalInputs = new QWidget(pathsContents);
    auto* optionalGrid = new QGridLayout(optionalInputs);
    optionalGrid->setContentsMargins(0, 0, 0, 0);
    optionalGrid->setHorizontalSpacing(12);
    optionalGrid->setVerticalSpacing(2);
    const auto optionalSpecs = std::initializer_list<std::pair<const char*, const char*>>{
        {"verified_patches", "Verified"}, {"unverified_patches", "Unverified"},
        {"normals", "Normals"}, {"surf_sdt", "SDT"},
        {"tracks_dbm", "Tracks"}, {"gradient_magnitude", "Grad mag"},
        {"fibers", "Fibers"},
    };
    int optionalIndex = 0;
    for (const auto& spec : optionalSpecs) {
        const QString key = QString::fromLatin1(spec.first);
        auto* check = new QCheckBox(tr(spec.second), optionalInputs);
        check->setObjectName(QStringLiteral("spiralUse_") + key);
        check->setChecked(true);
        check->setToolTip(tr("Include this dataset input and its associated losses and sampling"));
        _optionalInputs.insert(key, check);
        optionalGrid->addWidget(check, optionalIndex / 4, optionalIndex % 4);
        ++optionalIndex;
        connect(check, &QCheckBox::toggled, this, [this](bool) {
            updateOptionalInputUi();
            refreshReloadRequired();
        });
    }
    pathsForm->addRow(tr("Use inputs"), optionalInputs);
    addPathRow(pathsForm, "dataset_root", tr("Dataset root"), true);
    _refill = new QPushButton(tr("Refill from Dataset Root"), pathsContents);
    pathsForm->addRow(_refill);
    addPathRow(pathsForm, "umbilicus", tr("Umbilicus"), false);

    auto* pclContainer = new QWidget(pathsContents);
    auto* pclLayout = new QVBoxLayout(pclContainer);
    pclLayout->setContentsMargins(0, 0, 0, 0);
    _pclList = new QListWidget(pclContainer);
    _pclList->setObjectName(QStringLiteral("spiralPclList"));
    _pclList->setSelectionMode(QAbstractItemView::SingleSelection);
    _pclList->setMinimumHeight(90);
    pclLayout->addWidget(_pclList);
    auto* pclInputRow = new QHBoxLayout;
    _pclRole = new QComboBox(pclContainer);
    _pclRole->setObjectName(QStringLiteral("spiralPclRole"));
    _pclRole->addItem(tr("Absolute"), QStringLiteral("absolute"));
    _pclRole->addItem(tr("Patch overlap"), QStringLiteral("patch_overlap"));
    _pclRole->addItem(tr("Relative"), QStringLiteral("relative"));
    _pclRole->addItem(tr("Same winding"), QStringLiteral("same_winding"));
    _pclRole->addItem(tr("Drawn control points"), QStringLiteral("drawn_control_points"));
    _pclPath = new QLineEdit(pclContainer);
    _pclPath->setObjectName(QStringLiteral("spiralPclPath"));
    _pclPath->setPlaceholderText(tr("PCL path"));
    _browsePclButton = new QToolButton(pclContainer);
    _browsePclButton->setText(QStringLiteral("…"));
    _browsePclButton->setToolTip(tr("Select PCL file"));
    _addPclButton = new QPushButton(QStringLiteral("+"), pclContainer);
    _addPclButton->setObjectName(QStringLiteral("spiralAddPcl"));
    _addPclButton->setToolTip(tr("Add PCL"));
    _removePcl = new QPushButton(QStringLiteral("-"), pclContainer);
    _removePcl->setObjectName(QStringLiteral("spiralRemovePcl"));
    _removePcl->setToolTip(tr("Remove selected PCL"));
    _removePcl->setEnabled(false);
    pclInputRow->addWidget(_pclRole);
    pclInputRow->addWidget(_pclPath, 1);
    pclInputRow->addWidget(_browsePclButton);
    pclInputRow->addWidget(_addPclButton);
    pclInputRow->addWidget(_removePcl);
    pclLayout->addLayout(pclInputRow);
    pathsForm->addRow(tr("PCLs"), pclContainer);

    addPathRow(pathsForm, "fibers", tr("Fibers"), true);
    addPathRow(pathsForm, "tracks_dbm", tr("Tracks DBM"), false);

    _trackLengthBinSampling = new QCheckBox(tr("Sample tracks by length bins"), pathsContents);
    _trackLengthBinSampling->setObjectName(QStringLiteral("spiralTrackLengthBinSampling"));
    _trackLengthBinSampling->setChecked(false);
    _trackLengthBinSampling->setToolTip(
        tr("Draw short, medium, and long tracks using configurable weights. "
           "The bin boundaries are computed from eligible-track arclength tertiles."));
    pathsForm->addRow(_trackLengthBinSampling);

    auto* trackWeights = new QWidget(pathsContents);
    auto* trackWeightsLayout = new QHBoxLayout(trackWeights);
    trackWeightsLayout->setContentsMargins(0, 0, 0, 0);
    _trackShortWeight = new QDoubleSpinBox(trackWeights);
    _trackMediumWeight = new QDoubleSpinBox(trackWeights);
    _trackLongWeight = new QDoubleSpinBox(trackWeights);
    const auto configureTrackWeight = [](QDoubleSpinBox* spin, double value,
                                         const QString& objectName) {
        spin->setObjectName(objectName);
        spin->setRange(0.0, 1000.0);
        spin->setDecimals(3);
        spin->setSingleStep(0.05);
        spin->setValue(value);
    };
    configureTrackWeight(_trackShortWeight, 0.15,
                         QStringLiteral("spiralTrackShortWeight"));
    configureTrackWeight(_trackMediumWeight, 0.25,
                         QStringLiteral("spiralTrackMediumWeight"));
    configureTrackWeight(_trackLongWeight, 0.60,
                         QStringLiteral("spiralTrackLongWeight"));
    trackWeightsLayout->addWidget(new QLabel(tr("Short"), trackWeights));
    trackWeightsLayout->addWidget(_trackShortWeight);
    trackWeightsLayout->addWidget(new QLabel(tr("Medium"), trackWeights));
    trackWeightsLayout->addWidget(_trackMediumWeight);
    trackWeightsLayout->addWidget(new QLabel(tr("Long"), trackWeights));
    trackWeightsLayout->addWidget(_trackLongWeight);
    pathsForm->addRow(tr("Length-bin weights"), trackWeights);

    _maxTrackCrossings = new QSpinBox(pathsContents);
    _maxTrackCrossings->setObjectName(QStringLiteral("spiralMaxTrackCrossings"));
    _maxTrackCrossings->setRange(0, 8);
    _maxTrackCrossings->setValue(0);
    _maxTrackCrossings->setToolTip(
        tr("Maximum differently oriented crossing partners appended for each sampled "
           "primary track. Applies on the next Run; zero disables crossing-pair "
           "sampling. The upper bound is prepared when the session loads."));
    pathsForm->addRow(tr("Max crossings / sampled track"), _maxTrackCrossings);
    updateTrackSamplingUi();

    addPathRow(pathsForm, "verified_patches", tr("Verified patches"), true);
    addPathRow(pathsForm, "unverified_patches", tr("Unverified patches"), true);
    addPathRow(pathsForm, "outer_shell", tr("Outer shell"), true);

    auto* lasagnaSection = makeSection(tr("Lasagna inputs"),
                                       QStringLiteral("spiralLasagnaInputsGroup"),
                                       QStringLiteral("spiral/groups/lasagna_inputs_expanded"));
    auto* lasagnaContents = new QWidget(lasagnaSection->contentWidget());
    auto* lasagnaForm = new QFormLayout(lasagnaContents);
    lasagnaSection->contentLayout()->addWidget(lasagnaContents);
    addPathRow(lasagnaForm, "normal_x", tr("Normal X"), true);
    addPathRow(lasagnaForm, "normal_y", tr("Normal Y"), true);
    addPathRow(lasagnaForm, "surf_sdt", tr("Surface SDT"), true);
    addPathRow(lasagnaForm, "gradient_magnitude", tr("Gradient magnitude"), true);
    _lasagnaGroup = new QLineEdit(QStringLiteral("4"), lasagnaContents);
    _lasagnaScale = new QSpinBox(lasagnaContents);
    _lasagnaScale->setRange(1, 1024);
    _lasagnaScale->setValue(4);
    _storageBackend = new QComboBox(lasagnaContents);
    _storageBackend->addItem(tr("Auto"), QStringLiteral("auto"));
    _storageBackend->addItem(tr("Memory mapped"), QStringLiteral("mmap"));
    _storageBackend->addItem(tr("Dense CUDA (legacy)"), QStringLiteral("dense_cuda"));
    addPathRow(lasagnaForm, "cache_directory", tr("Cache directory"), true);
    lasagnaForm->addRow(tr("Zarr group"), _lasagnaGroup);
    lasagnaForm->addRow(tr("Coordinate scale"), _lasagnaScale);
    lasagnaForm->addRow(tr("Storage backend"), _storageBackend);

    auto* outputGroup = makeSection(tr("Fit and output"),
                                    QStringLiteral("spiralFitOutputGroup"),
                                    QStringLiteral("spiral/groups/fit_output_expanded"));
    auto* outputContents = new QWidget(outputGroup->contentWidget());
    auto* outputForm = new QFormLayout(outputContents);
    outputGroup->contentLayout()->addWidget(outputContents);
    addPathRow(outputForm, "output_directory", tr("Output directory"), true);
    addPathRow(outputForm, "checkpoint", tr("Checkpoint"), false);
    addPathRow(outputForm, "scroll_zarr", tr("Scroll/render Zarr"), true);
    _zBegin = new QSpinBox(outputContents); _zBegin->setRange(0, 1000000); _zBegin->setValue(4000);
    _zEnd = new QSpinBox(outputContents); _zEnd->setRange(1, 1000000); _zEnd->setValue(17000);
    _scrollName = new QLineEdit(QStringLiteral("s1"), outputContents);
    _outwardSense = new QComboBox(outputContents); _outwardSense->addItems({QStringLiteral("CW"), QStringLiteral("ACW")});
    _voxelSize = new QDoubleSpinBox(outputContents); _voxelSize->setRange(0.001, 10000); _voxelSize->setDecimals(4); _voxelSize->setValue(9.6);
    _legacyCheckpointStep = new QSpinBox(outputContents); _legacyCheckpointStep->setRange(0, 1000000000);
    _renderVolumeScale = new QSpinBox(outputContents); _renderVolumeScale->setRange(1, 4096); _renderVolumeScale->setValue(16);
    _savePngVisualizations = new QCheckBox(tr("Save diagnostic PNG visualizations"), outputContents);
    _savePngVisualizations->setChecked(false);
    _influenceEnabled = new QCheckBox(tr("Localize fit around added inputs"), outputContents);
    _influenceEnabled->setChecked(false);
    _influenceEnabled->setToolTip(tr("Restrict optimization to a region around each input added to "
                                     "the running fit; the rest of the fit is held in place. Takes "
                                     "effect when inputs are added to a running session."));
    _influenceZ = new QSpinBox(outputContents);
    _influenceZ->setRange(1, 1000000); _influenceZ->setValue(3000);
    _influenceZ->setToolTip(tr("Max influence half-extent above/below the added input, in voxels"));
    _influenceWindings = new QDoubleSpinBox(outputContents);
    _influenceWindings->setRange(0.1, 100.0); _influenceWindings->setDecimals(1); _influenceWindings->setValue(5.0);
    _influenceWindings->setToolTip(tr("Max influence half-extent across wraps, in windings"));
    _influenceThetaPct = new QSpinBox(outputContents);
    _influenceThetaPct->setRange(1, 100); _influenceThetaPct->setSuffix(tr("% of wrap")); _influenceThetaPct->setValue(50);
    _influenceThetaPct->setToolTip(tr("Max influence half-extent along the wrap, as a fraction of a full turn"));
    _influenceDisableDtPct = new QSpinBox(outputContents);
    _influenceDisableDtPct->setRange(0, 100);
    _influenceDisableDtPct->setSuffix(tr("%"));
    _influenceDisableDtPct->setValue(75);
    _influenceDisableDtPct->setToolTip(
        tr("Fraction of each requested Run window that keeps directional DT losses disabled "
           "after pending inputs are incorporated"));
    _influenceAnchorWeight = new QDoubleSpinBox(outputContents);
    _influenceAnchorWeight->setRange(0.0, 10000.0); _influenceAnchorWeight->setDecimals(1); _influenceAnchorWeight->setValue(20.0);
    _influenceAnchorWeight->setToolTip(tr("Weight of the loss holding the fit in place outside the influence region"));
    _runTag = new QLineEdit(outputContents);
    _advancedProfiles = new SpiralConfigProfileEditor(outputContents);
    _advanced = _advancedProfiles->textEdit();
    outputForm->addRow(tr("z begin"), _zBegin);
    outputForm->addRow(tr("z end"), _zEnd);
    outputForm->addRow(tr("Scroll name"), _scrollName);
    outputForm->addRow(tr("Outward sense"), _outwardSense);
    outputForm->addRow(tr("Voxel size (µm)"), _voxelSize);
    outputForm->addRow(tr("Legacy checkpoint step"), _legacyCheckpointStep);
    outputForm->addRow(tr("Run tag"), _runTag);
    outputForm->addRow(tr("Render-volume scale"), _renderVolumeScale);
    outputForm->addRow(_savePngVisualizations);
    outputForm->addRow(_influenceEnabled);
    outputForm->addRow(tr("Influence z extent"), _influenceZ);
    outputForm->addRow(tr("Influence windings"), _influenceWindings);
    outputForm->addRow(tr("Influence theta"), _influenceThetaPct);
    outputForm->addRow(tr("% of iters to disable DT"), _influenceDisableDtPct);
    outputForm->addRow(tr("Influence anchor weight"), _influenceAnchorWeight);
    outputForm->addRow(tr("Advanced config JSON"), _advancedProfiles);

    auto* displayGroup = makeSection(tr("Display"),
                                     QStringLiteral("spiralDisplayGroup"),
                                     QStringLiteral("spiral/groups/display_expanded"));
    auto* displayContents = new QWidget(displayGroup->contentWidget());
    auto* displayLayout = new QVBoxLayout(displayContents);
    displayGroup->contentLayout()->addWidget(displayContents);
    _volumeSelector = new VolumeSelector(displayContents);
    displayLayout->addWidget(_volumeSelector);
    auto* windingRange = new QWidget(displayContents);
    auto* windingRangeLayout = new QHBoxLayout(windingRange);
    windingRangeLayout->setContentsMargins(0, 0, 0, 0);
    _minimumDisplayedWinding = new QSpinBox(windingRange);
    _minimumDisplayedWinding->setObjectName(QStringLiteral("spiralMinimumDisplayedWinding"));
    _minimumDisplayedWinding->setRange(0, 1000000);
    _minimumDisplayedWinding->setValue(10);
    _minimumDisplayedWinding->setToolTip(tr("First winding to display (inclusive)"));
    _maximumDisplayedWinding = new QSpinBox(windingRange);
    _maximumDisplayedWinding->setObjectName(QStringLiteral("spiralMaximumDisplayedWinding"));
    _maximumDisplayedWinding->setRange(-1, 1000000);
    _maximumDisplayedWinding->setValue(130);
    _maximumDisplayedWinding->setToolTip(
        tr("Last winding to display (inclusive); -1 displays through the final winding"));
    windingRangeLayout->addWidget(new QLabel(tr("Min winding"), windingRange));
    windingRangeLayout->addWidget(_minimumDisplayedWinding);
    windingRangeLayout->addWidget(new QLabel(tr("Max winding"), windingRange));
    windingRangeLayout->addWidget(_maximumDisplayedWinding);
    displayLayout->addWidget(windingRange);
    auto emitWindingRange = [this](int) {
        emit windingRangeChanged(_minimumDisplayedWinding->value(),
                                 _maximumDisplayedWinding->value());
    };
    connect(_minimumDisplayedWinding, QOverload<int>::of(&QSpinBox::valueChanged),
            this, emitWindingRange);
    connect(_maximumDisplayedWinding, QOverload<int>::of(&QSpinBox::valueChanged),
            this, emitWindingRange);

    auto* displayButton = new QPushButton(tr("Display ->"), displayContents);
    displayButton->setObjectName(QStringLiteral("spiralDisplayButton"));
    displayButton->setToolTip(tr("Choose which Spiral overlays are shown"));
    displayLayout->addWidget(displayButton);

    _displayDialog = new QDialog(this);
    _displayDialog->setObjectName(QStringLiteral("spiralDisplayDialog"));
    _displayDialog->setWindowTitle(tr("Display"));
    _displayDialog->setModal(false);
    auto* displayDialogLayout = new QVBoxLayout(_displayDialog);

    connect(displayButton, &QPushButton::clicked, this, [this]() {
        _displayDialog->show();
        _displayDialog->raise();
        _displayDialog->activateWindow();
    });

    _showSurfaceIntersections = new QCheckBox(tr("Show surface intersections"), _displayDialog);
    _showSurfaceIntersections->setObjectName(QStringLiteral("spiralShowSurfaceIntersections"));
    _showSurfaceIntersections->setChecked(true);
    _showSurfaceIntersections->setToolTip(
        tr("Show rendered surface intersections on the plane views"));
    connect(_showSurfaceIntersections, &QCheckBox::toggled,
            this, &SpiralPanel::surfaceIntersectionsChanged);
    displayDialogLayout->addWidget(_showSurfaceIntersections);

    auto* intersectionStrideRow = new QWidget(_displayDialog);
    auto* intersectionStrideLayout = new QHBoxLayout(intersectionStrideRow);
    intersectionStrideLayout->setContentsMargins(0, 0, 0, 0);
    auto* intersectionStride = new QSpinBox(intersectionStrideRow);
    intersectionStride->setObjectName(QStringLiteral("spiralSurfaceIntersectionStride"));
    intersectionStride->setRange(1, 16);
    intersectionStride->setValue(4);
    intersectionStride->setToolTip(
        tr("Sample every Nth input-surface grid cell when building Spiral intersections; "
           "larger values are faster but less detailed"));
    intersectionStrideLayout->addWidget(
        new QLabel(tr("Surface intersection stride"), intersectionStrideRow));
    intersectionStrideLayout->addWidget(intersectionStride);
    intersectionStrideLayout->addStretch(1);
    connect(intersectionStride, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &SpiralPanel::surfaceIntersectionStrideChanged);
    displayDialogLayout->addWidget(intersectionStrideRow);

    auto* surfaceOverlap = new QCheckBox(tr("Show patch overlap on flattened output"),
                                         _displayDialog);
    surfaceOverlap->setObjectName(QStringLiteral("spiralShowSurfaceOverlap"));
    surfaceOverlap->setChecked(true);
    surfaceOverlap->setToolTip(
        tr("Color areas of the flattened output that overlap the selected patch categories"));
    connect(surfaceOverlap, &QCheckBox::toggled,
            this, &SpiralPanel::surfaceOverlapChanged);
    displayDialogLayout->addWidget(surfaceOverlap);

    auto* runDiff = new QCheckBox(tr("Run diff"), _displayDialog);
    runDiff->setObjectName(QStringLiteral("spiralRunDiff"));
    runDiff->setToolTip(
        tr("Overlay the XYZ displacement magnitude between the previous and current "
           "completed runs. Magnitude increases from blue through green, yellow, and "
           "orange to red; the first run has no diff."));
    connect(runDiff, &QCheckBox::toggled, this, &SpiralPanel::runDiffChanged);
    displayDialogLayout->addWidget(runDiff);

    auto* lossMapRow = new QWidget(_displayDialog);
    auto* lossMapLayout = new QHBoxLayout(lossMapRow);
    lossMapLayout->setContentsMargins(0, 0, 0, 0);
    _lossMap = new QComboBox(lossMapRow);
    _lossMap->setObjectName(QStringLiteral("spiralLossMap"));
    _lossMap->addItem(tr("No loss overlay"), QString());
    _lossMap->setToolTip(
        tr("Overlay a weighted per-sample loss residual on the flattened output"));
    _lossMapOpacity = new QSlider(Qt::Horizontal, lossMapRow);
    _lossMapOpacity->setObjectName(QStringLiteral("spiralLossMapOpacity"));
    _lossMapOpacity->setRange(0, 100);
    _lossMapOpacity->setValue(80);
    _lossMapOpacity->setToolTip(tr("Loss overlay opacity"));
    lossMapLayout->addWidget(_lossMap, 1);
    lossMapLayout->addWidget(new QLabel(tr("Opacity"), lossMapRow));
    lossMapLayout->addWidget(_lossMapOpacity);
    displayDialogLayout->addWidget(lossMapRow);
    _lossMapLegend = new QLabel(_displayDialog);
    _lossMapLegend->setWordWrap(true);
    _lossMapLegend->setTextInteractionFlags(Qt::TextSelectableByMouse);
    displayDialogLayout->addWidget(_lossMapLegend);
    auto emitLossMap = [this]() {
        emit lossMapChanged(_lossMap->currentData().toString(),
                            _lossMapOpacity->value() / 100.0);
    };
    connect(_lossMap, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [emitLossMap](int) { emitLossMap(); });
    connect(_lossMapOpacity, &QSlider::valueChanged,
            this, [emitLossMap](int) { emitLossMap(); });

    for (const auto& item : std::initializer_list<std::pair<const char*, const char*>>{
             {"output", "Output"}, {"fibers", "Fibers"}, {"tracks", "Tracks"},
             {"pcls", "Winding/PCL inputs"}, {"verified", "Verified patches"},
             {"unverified", "Unverified patches"}, {"pending_only", "Pending patches only"},
             {"shell", "Shell"}, {"lasagna", "Lasagna inputs"}}) {
        auto* check = new QCheckBox(tr(item.second), _displayDialog);
        const QString key = QString::fromLatin1(item.first);
        _visibilityChecks[key] = check;
        if (key == QStringLiteral("pending_only")) {
            check->setObjectName(QStringLiteral("spiralPendingPatchesOnly"));
            check->setToolTip(tr("Replace the verified/unverified patch selections with only "
                                 "interactive-fit patches that are not yet committed to the dataset"));
        }
        check->setChecked(key == QStringLiteral("output"));
        connect(check, &QCheckBox::toggled, this, [this, key = QString::fromLatin1(item.first)](bool shown) {
            emit visibilityChanged(key, shown);
        });
        displayDialogLayout->addWidget(check);
    }
    _displayDialog->adjustSize();

    auto* runGroup = makeSection(tr("Run and status"),
                                 QStringLiteral("spiralRunStatusGroup"),
                                 QStringLiteral("spiral/groups/run_status_expanded"));
    auto* runContents = new QWidget(runGroup->contentWidget());
    auto* runLayout = new QVBoxLayout(runContents);
    runGroup->contentLayout()->addWidget(runContents);
    auto* controls = new QHBoxLayout;
    _load = new QPushButton(tr("Load/Reload Inputs"), runContents);
    _load->setEnabled(false);
    _iterations = new QSpinBox(runContents); _iterations->setRange(1, 1000000); _iterations->setValue(100);
    _run = new QPushButton(tr("Run"), runContents);
    _run->setEnabled(false);
    _stop = new QPushButton(tr("Stop after iteration"), runContents); _stop->setEnabled(false);
    controls->addWidget(_load); controls->addWidget(_iterations); controls->addWidget(_run);
    controls->addWidget(_stop);
    runLayout->addLayout(controls);
    auto* checkpointControls = new QHBoxLayout;
    _save = new QPushButton(tr("Save Checkpoint on Service"), runContents); _save->setEnabled(false);
    _downloadCheckpoint = new QPushButton(tr("Download Checkpoint…"), runContents);
    _downloadCheckpoint->setEnabled(false);
    checkpointControls->addWidget(_save);
    checkpointControls->addWidget(_downloadCheckpoint);
    checkpointControls->addStretch(1);
    runLayout->addLayout(checkpointControls);
    _flattenWithLasagna = new QPushButton(tr("Flatten with Lasagna"), runContents);
    _flattenWithLasagna->setObjectName(QStringLiteral("spiralFlattenWithLasagna"));
    _flattenWithLasagna->setEnabled(false);
    _flattenWithLasagna->setToolTip(
        tr("Flatten the complete latest Spiral output with Lasagna"));
    runLayout->addWidget(_flattenWithLasagna);

    auto* ephemeralLabel = new QLabel(tr("Inputs added to the running fit:"), runContents);
    _ephemeralList = new QListWidget(runContents);
    _ephemeralList->setObjectName(QStringLiteral("spiralEphemeralList"));
    _ephemeralList->setMaximumHeight(80);
    _ephemeralList->setSelectionMode(QAbstractItemView::SingleSelection);
    _commitInputs = new QPushButton(tr("Commit current inputs"), runContents);
    _commitInputs->setEnabled(false);
    _commitInputs->setToolTip(tr("Copy the session's added inputs into their dataset locations"));
    _removeInput = new QPushButton(tr("Remove"), runContents);
    _removeInput->setEnabled(false);
    _removeInput->setToolTip(tr("Remove the selected input before it joins the fit; "
                                "inputs that already joined need a session reload"));
    _commitHint = new QLabel(runContents);
    _commitHint->setWordWrap(true);
    auto* commitRow = new QHBoxLayout;
    commitRow->addWidget(_commitInputs);
    commitRow->addWidget(_removeInput);
    commitRow->addStretch(1);
    runLayout->addWidget(ephemeralLabel);
    runLayout->addWidget(_ephemeralList);
    runLayout->addLayout(commitRow);
    runLayout->addWidget(_commitHint);

    _state = new QLabel(tr("Service disconnected"), runContents);
    _metrics = new QLabel(runContents);
    _warnings = new QLabel(runContents); _warnings->setWordWrap(true);
    _warnings->setTextInteractionFlags(Qt::TextSelectableByMouse);
    runLayout->addWidget(_state); runLayout->addWidget(_metrics); runLayout->addWidget(_warnings);
    layout->addStretch(1);
    scroll->setWidget(contents);
    rootLayout->addWidget(scroll);

    // ------------------------------------------------------------------
    // Wiring
    // ------------------------------------------------------------------
    connect(_paths["dataset_root"], &QLineEdit::editingFinished, this, [this]() {
        if (_remoteMode) return;
        _pendingDatasetRoot = _paths["dataset_root"]->text();
        if (_service->isReady()) _service->resolveDataset(_pendingDatasetRoot);
    });
    connect(_refill, &QPushButton::clicked, this, [this]() {
        if (_remoteMode) return;
        _pendingDatasetRoot = _paths["dataset_root"]->text();
        if (_service->isReady()) _service->resolveDataset(_pendingDatasetRoot);
        else _connectionStatus->setText(tr("Connect to the service to resolve datasets"));
    });
    auto appendPcl = [this]() {
        const QString path = _pclPath->text().trimmed();
        if (path.isEmpty()) return;
        addPclItem(path, _pclRole->currentData().toString());
        _pclPath->clear();
        _hasManualEdits = true;
        refreshReloadRequired();
    };
    connect(_addPclButton, &QPushButton::clicked, this, appendPcl);
    connect(_pclPath, &QLineEdit::returnPressed, this, appendPcl);
    connect(_browsePclButton, &QToolButton::clicked, this, [this]() {
        const QString chosen = QFileDialog::getOpenFileName(
            this, tr("Select PCL file"), _pclPath->text(), tr("JSON files (*.json);;All files (*)"));
        if (!chosen.isEmpty()) _pclPath->setText(chosen);
    });
    connect(_pclList, &QListWidget::itemSelectionChanged, this, [this]() {
        _removePcl->setEnabled(!_remoteMode && _pclList->currentItem() != nullptr);
    });
    connect(_removePcl, &QPushButton::clicked, this, [this]() {
        if (auto* item = _pclList->takeItem(_pclList->currentRow())) {
            delete item;
            _hasManualEdits = true;
            refreshReloadRequired();
        }
    });

    connect(_service, &SpiralServiceManager::connectionStateChanged, this,
            [this](SpiralServiceManager::ConnectionState state, const QString& message) {
                using CS = SpiralServiceManager::ConnectionState;
                QString text;
                switch (state) {
                case CS::Disconnected: text = tr("Disconnected"); break;
                case CS::Starting: text = tr("Starting…"); break;
                case CS::Connecting: text = tr("Connecting…"); break;
                case CS::Ready: text = tr("Connected — API v6%1")
                        .arg(message.isEmpty() ? QString() : QStringLiteral(" — ") + message); break;
                case CS::Reconnecting: text = tr("Reconnecting… %1").arg(message); break;
                case CS::Failed: text = tr("Failed: %1").arg(message); break;
                }
                _connectionStatus->setText(text);
                _connected = state == CS::Ready;
                _connectButton->setEnabled(state == CS::Disconnected || state == CS::Failed);
                _disconnectButton->setEnabled(state != CS::Disconnected);
                // Connection must succeed before dataset resolution or fit
                // controls are enabled.
                _load->setEnabled(_connected);
                if (!_connected) { _run->setEnabled(false); _stop->setEnabled(false);
                                   _save->setEnabled(false); _downloadCheckpoint->setEnabled(false);
                                   _removeInput->setEnabled(false); }
                _state->setText(tr("Service: %1").arg(text));
                if (_connected && !_remoteMode && !_pendingDatasetRoot.isEmpty())
                    _service->resolveDataset(_pendingDatasetRoot);
            });
    connect(_service, &SpiralServiceManager::datasetResolved, this, [this](const QJsonObject& value) {
        applyResolution(value, _remoteMode || !_hasManualEdits);
        _pendingDatasetRoot.clear();
    });
    connect(_service, &SpiralServiceManager::sessionStatusChanged, this, &SpiralPanel::updateStatus);
    connect(_service, &SpiralServiceManager::sessionAccepted, this,
            [this](const QJsonObject&, qint64) {
                _hasSession = true;
                _loadedSessionRequest = _pendingSessionRequest;
                _pendingSessionRequest = {};
                _reloadRequired = false;
                _advancedSessionGeneration = -1;
                _defaultAdvancedConfig = {};
                _applyingResolution = true;
                _advancedProfiles->clearSessionDefault();
                _applyingResolution = false;
                for (auto it = _visibilityChecks.begin(); it != _visibilityChecks.end(); ++it)
                    it.value()->setChecked(it.key() == QStringLiteral("output"));
            });
    connect(_service, &SpiralServiceManager::errorOccurred, this, [this](const QString& error) {
        _warnings->setText(error);
    });
    connect(_service, &SpiralServiceManager::checkpointUploadProgress, this,
            [this](qint64 sent, qint64 total) {
                if (total > 0)
                    _state->setText(tr("Uploading resume checkpoint… %1 / %2 MB")
                                        .arg(sent / 1000000).arg(total / 1000000));
            });
    connect(_service, &SpiralServiceManager::checkpointDownloadFinished, this,
            [this](const QString& path, const QString& error) {
                if (error.isEmpty())
                    _warnings->setText(tr("Checkpoint downloaded to %1").arg(path));
                else
                    _warnings->setText(tr("Checkpoint download failed: %1").arg(error));
            });
    connect(_service, &SpiralServiceManager::inputUploadFinished, this,
            [this](const QString& inputId, const QString& error) {
                if (error.isEmpty())
                    _warnings->setText(tr("Added %1 to the current fit; it is used on the next run").arg(inputId));
                else
                    _warnings->setText(tr("Adding %1 failed: %2").arg(inputId, error));
            });

    connect(_load, &QPushButton::clicked, this, [this]() {
        QJsonParseError error;
        const QJsonDocument advanced = QJsonDocument::fromJson(_advanced->toPlainText().toUtf8(), &error);
        if (error.error != QJsonParseError::NoError || !advanced.isObject()) {
            QMessageBox::warning(this, tr("Invalid advanced configuration"),
                                 tr("Advanced config must be a JSON object: %1").arg(error.errorString()));
            return;
        }
        guardSessionExit([this]() {
            if (_uncommittedCount > 0
                && QMessageBox::question(
                       this, tr("Uncommitted inputs"),
                       tr("The current session has %1 added input(s) that were not committed "
                          "to the dataset. Reloading discards them. Continue?").arg(_uncommittedCount))
                       != QMessageBox::Yes)
                return;
            persist();
            emit pythonOutputRequested();
            _pendingSessionRequest = sessionRequest();
            _service->loadSession(_pendingSessionRequest);
        });
    });
    connect(_run, &QPushButton::clicked, this, [this]() {
        QJsonParseError error;
        const QJsonDocument advanced =
            QJsonDocument::fromJson(_advanced->toPlainText().toUtf8(), &error);
        if (error.error != QJsonParseError::NoError || !advanced.isObject()) {
            QMessageBox::warning(this, tr("Invalid advanced configuration"),
                                 tr("Advanced config must be a JSON object: %1")
                                     .arg(error.errorString()));
            return;
        }
        persist();
        emit pythonOutputRequested();
        _service->runIterations(_iterations->value(), influenceConfig(),
                                runAdvancedConfig());
    });
    connect(_stop, &QPushButton::clicked, _service, &SpiralServiceManager::stopAfterIteration);
    connect(_flattenWithLasagna, &QPushButton::clicked,
            this, &SpiralPanel::flattenWithLasagnaRequested);
    connect(_save, &QPushButton::clicked, this, [this]() {
        const QString initial = QDir(_paths["output_directory"]->text())
            .filePath(QStringLiteral("checkpoint_manual.ckpt"));
        const QString path = _remoteMode
            ? QInputDialog::getText(this, tr("Save Spiral checkpoint on the service host"),
                                    tr("Service-host path (must be under the session output directory):"),
                                    QLineEdit::Normal, initial)
            : QFileDialog::getSaveFileName(this, tr("Save Spiral checkpoint"),
                                           initial, tr("Checkpoint (*.ckpt)"));
        if (!path.isEmpty()) _service->saveCheckpoint(path);
    });
    connect(_downloadCheckpoint, &QPushButton::clicked, this, [this]() {
        const QString path = QFileDialog::getSaveFileName(
            this, tr("Download Spiral checkpoint"),
            QDir::home().filePath(QStringLiteral("spiral_checkpoint.ckpt")),
            tr("Checkpoint (*.ckpt)"));
        if (path.isEmpty()) return;
        _warnings->setText(tr("Downloading checkpoint… this can take a few minutes."));
        _service->downloadCheckpoint(path);
    });
    connect(_commitInputs, &QPushButton::clicked, this, [this]() {
        if (QMessageBox::question(this, tr("Commit inputs"),
                                  tr("Move the added inputs into the dataset? Patches go to "
                                     "verified_patches/, fibers to fibers/, and PCL documents "
                                     "merge into their conventional role file (control-point "
                                     "lines go to drawn_control_points.json; "
                                     "same-winding point collections go to same_windings.json)."))
            != QMessageBox::Yes) return;
        _service->commitInputs();
    });
    connect(_ephemeralList, &QListWidget::itemSelectionChanged, this, [this]() {
        const QListWidgetItem* item = _ephemeralList->currentItem();
        _removeInput->setEnabled(_connected && item
            && item->data(Qt::UserRole + 2).toString() == QStringLiteral("pending"));
    });
    connect(_removeInput, &QPushButton::clicked, this, [this]() {
        const QListWidgetItem* item = _ephemeralList->currentItem();
        if (!item) return;
        _service->removeEphemeralInput(item->data(Qt::UserRole).toString(),
                                       item->data(Qt::UserRole + 1).toString());
    });
    connect(_volumeSelector->comboBox(), qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this](int) { emit volumeSelected(_volumeSelector->selectedVolumeId()); });
    for (QSpinBox* spin : {_zBegin, _zEnd, _lasagnaScale, _legacyCheckpointStep,
                           _renderVolumeScale})
        connect(spin, qOverload<int>(&QSpinBox::valueChanged), this, [this](int) { refreshReloadRequired(); });
    for (QDoubleSpinBox* spin : {_voxelSize})
        connect(spin, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
                [this](double) { refreshReloadRequired(); });
    for (QLineEdit* edit : {_lasagnaGroup, _scrollName, _runTag})
        connect(edit, &QLineEdit::textEdited, this, [this](const QString&) { refreshReloadRequired(); });
    connect(_outwardSense, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this](int) { refreshReloadRequired(); });
    connect(_storageBackend, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this](int) { refreshReloadRequired(); });
    connect(_savePngVisualizations, &QCheckBox::toggled, this,
            [this](bool) { refreshReloadRequired(); });
    connect(_trackLengthBinSampling, &QCheckBox::toggled, this, [this](bool) {
        updateTrackSamplingUi();
        writeTrackSamplingControlsToAdvanced();
    });
    for (QDoubleSpinBox* spin : {_trackShortWeight, _trackMediumWeight,
                                 _trackLongWeight}) {
        connect(spin, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
                [this](double) { writeTrackSamplingControlsToAdvanced(); });
    }
    connect(_maxTrackCrossings, qOverload<int>(&QSpinBox::valueChanged), this,
            [this](int) { writeTrackSamplingControlsToAdvanced(); });
    connect(_advancedProfiles, &SpiralConfigProfileEditor::textChanged, this, [this]() {
        syncTrackSamplingControlsFromAdvanced();
        refreshReloadRequired();
    });

    // Profile management
    connect(_profileCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0) return;
        const QString profileId = _profileCombo->itemData(index).toString();
        if (!profileId.isEmpty() && profileId != _currentProfileId) selectProfile(profileId);
    });
    connect(addLan, &QToolButton::clicked, this, [this]() {
        bool ok = false;
        const QString name = QInputDialog::getText(this, tr("New Remote (LAN) profile"),
                                                   tr("Profile name:"), QLineEdit::Normal,
                                                   tr("GPU workstation"), &ok);
        if (!ok || name.trimmed().isEmpty()) return;
        SpiralServiceProfile profile;
        profile.id = QStringLiteral("lan-") + QUuid::createUuid().toString(QUuid::WithoutBraces).left(8);
        profile.name = name.trimmed();
        profile.transport = SpiralServiceProfile::Transport::Direct;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        profile.save(settings);
        _profileIds.push_back(profile.id);
        saveProfileList();
        rebuildProfileCombo();
        selectProfile(profile.id);
    });
    connect(addSsh, &QToolButton::clicked, this, [this]() {
        bool ok = false;
        const QString name = QInputDialog::getText(this, tr("New Remote (SSH) profile"),
                                                   tr("Profile name:"), QLineEdit::Normal,
                                                   tr("GPU host over SSH"), &ok);
        if (!ok || name.trimmed().isEmpty()) return;
        SpiralServiceProfile profile;
        profile.id = QStringLiteral("ssh-") + QUuid::createUuid().toString(QUuid::WithoutBraces).left(8);
        profile.name = name.trimmed();
        profile.transport = SpiralServiceProfile::Transport::SshTunnel;
        profile.remoteServicePort = 8765;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        profile.save(settings);
        _profileIds.push_back(profile.id);
        saveProfileList();
        rebuildProfileCombo();
        selectProfile(profile.id);
    });
    connect(removeProfile, &QToolButton::clicked, this, [this]() {
        if (_currentProfileId == kLocalhostProfileId) return;
        if (QMessageBox::question(this, tr("Remove profile"),
                                  tr("Remove the profile '%1'?").arg(_profileCombo->currentText()))
            != QMessageBox::Yes) return;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.remove(QStringLiteral("spiral/profiles/") + _currentProfileId);
        _profileIds.removeAll(_currentProfileId);
        saveProfileList();
        _currentProfileId.clear();
        rebuildProfileCombo();
        selectProfile(kLocalhostProfileId);
    });
    connect(_connectButton, &QPushButton::clicked, this, [this]() {
        guardSessionExit([this]() { connectToSelectedProfile(); });
    });
    connect(_disconnectButton, &QPushButton::clicked, this, [this]() {
        guardSessionExit([this]() { _service->disconnectFromService(); });
    });

    // Load saved profiles and select the previous one.
    {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        _profileIds = settings.value(QStringLiteral("spiral/profile_ids")).toStringList();
        rebuildProfileCombo();
        const QString selected = settings.value(QStringLiteral("spiral/selected_profile"),
                                                kLocalhostProfileId).toString();
        selectProfile(_profileIds.contains(selected) || selected == kLocalhostProfileId
                          ? selected : kLocalhostProfileId);
    }
    // Preserve the easy local experience: auto-connect the built-in profile.
    if (_currentProfileId == kLocalhostProfileId) connectToSelectedProfile();
}

void SpiralPanel::rebuildProfileCombo()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const QSignalBlocker blocker(_profileCombo);
    _profileCombo->clear();
    _profileCombo->addItem(tr("Local (this computer)"), kLocalhostProfileId);
    for (const QString& profileId : _profileIds) {
        const SpiralServiceProfile profile = SpiralServiceProfile::load(settings, profileId);
        const QString badge = profile.transport == SpiralServiceProfile::Transport::SshTunnel
            ? tr("Remote (SSH)") : tr("Remote (LAN)");
        _profileCombo->addItem(QStringLiteral("%1 — %2").arg(profile.name, badge), profileId);
    }
}

void SpiralPanel::saveProfileList() const
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(QStringLiteral("spiral/profile_ids"), _profileIds);
}

QString SpiralPanel::formSettingsPrefix() const
{
    return QStringLiteral("spiral/profiles/%1/form/").arg(
        _currentProfileId.isEmpty() ? kLocalhostProfileId : _currentProfileId);
}

void SpiralPanel::selectProfile(const QString& profileId)
{
    if (profileId == _currentProfileId) return;
    if (!_runningGuardedExit && _sessionExitGuard) {
        const int oldIndex = _profileCombo->findData(_currentProfileId);
        if (oldIndex >= 0) {
            const QSignalBlocker blocker(_profileCombo);
            _profileCombo->setCurrentIndex(oldIndex);
        }
        guardSessionExit([this, profileId]() { selectProfile(profileId); });
        return;
    }
    if (!_currentProfileId.isEmpty()) persist();
    _service->disconnectFromService();
    _currentProfileId = profileId;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(QStringLiteral("spiral/selected_profile"), profileId);
    const int index = _profileCombo->findData(profileId);
    if (index >= 0 && index != _profileCombo->currentIndex()) {
        const QSignalBlocker blocker(_profileCombo);
        _profileCombo->setCurrentIndex(index);
    }
    SpiralServiceProfile profile = profileId == kLocalhostProfileId
        ? SpiralServiceProfile::localhostProfile()
        : SpiralServiceProfile::load(settings, profileId);
    applyProfileFields(profile);
    setRemoteMode(profile.isRemote());
    // Switching profiles must never carry one host's persisted values into
    // another profile's form.
    restore();
}

void SpiralPanel::applyProfileFields(const SpiralServiceProfile& profile)
{
    const bool localhost = profile.isLocalhost();
    const bool ssh = profile.transport == SpiralServiceProfile::Transport::SshTunnel;
    _endpointRow->setVisible(!localhost && !ssh);
    _sshRow->setVisible(ssh);
    // An SSH profile's credential is read from the host over SSH.
    _apiKeyRow->setVisible(!localhost && !ssh);
    _mappingRow->setVisible(!localhost);
    _endpointUrl->setText(profile.baseUrl.toString());
    _sshDestination->setText(profile.sshDestination);
    if (profile.remoteServicePort > 0) _sshPort->setValue(profile.remoteServicePort);
    _mapServiceRoot->setText(profile.serviceRootPrefix);
    _mapLocalRoot->setText(profile.localRootPrefix);
    _apiKey->clear();
}

SpiralServiceProfile SpiralPanel::profileFromFields() const
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    if (_currentProfileId == kLocalhostProfileId)
        return SpiralServiceProfile::localhostProfile();
    SpiralServiceProfile profile = SpiralServiceProfile::load(settings, _currentProfileId);
    profile.baseUrl = QUrl(_endpointUrl->text().trimmed());
    profile.sshDestination = _sshDestination->text().trimmed();
    profile.remoteServicePort = _sshPort->value();
    profile.serviceRootPrefix = _mapServiceRoot->text().trimmed();
    profile.localRootPrefix = _mapLocalRoot->text().trimmed();
    profile.apiKey = _apiKey->text();
    return profile;
}

void SpiralPanel::connectToSelectedProfile()
{
    SpiralServiceProfile profile = profileFromFields();
    if (profile.isRemote()) {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        profile.save(settings); // never persists the API key
    }
    _service->connectToService(profile);
}

void SpiralPanel::guardSessionExit(std::function<void()> action)
{
    if (!_sessionExitGuard || _runningGuardedExit) {
        action();
        return;
    }
    _sessionExitGuard([this, action = std::move(action)]() mutable {
        _runningGuardedExit = true;
        action();
        _runningGuardedExit = false;
    });
}

void SpiralPanel::setRemoteMode(bool remote)
{
    _remoteMode = remote;
    // Remote services own their base inputs: the path rows populate read-only
    // from the service's advertised dataset resolution, and there is nothing
    // local to browse. The checkpoint and tracks selections stay editable
    // because the client chooses among service-advertised values.
    const QStringList clientSelectable{QStringLiteral("checkpoint"), QStringLiteral("tracks_dbm")};
    for (auto it = _paths.begin(); it != _paths.end(); ++it) {
        const bool selectable = clientSelectable.contains(it.key());
        // The checkpoint browse button stays in remote mode: a client-local
        // .ckpt selected here is uploaded to the service before the load.
        const bool browsable = !remote || it.key() == QStringLiteral("checkpoint");
        it.value()->setReadOnly(remote && !selectable);
        if (_pathBrowseButtons.contains(it.key()))
            _pathBrowseButtons[it.key()]->setVisible(browsable);
    }
    _paths[QStringLiteral("checkpoint")]->setToolTip(
        remote ? tr("A service-advertised checkpoint, a service path under the output "
                    "directory, or a local .ckpt file to upload for resume")
               : QString());
    _refill->setVisible(!remote);
    _pclRole->setEnabled(!remote);
    _pclPath->setEnabled(!remote);
    _addPclButton->setEnabled(!remote);
    _browsePclButton->setVisible(!remote);
    _removePcl->setEnabled(!remote && _pclList->currentItem() != nullptr);
    _save->setVisible(true);
    if (remote) {
        for (QLineEdit* edit : {_paths["dataset_root"], _paths["umbilicus"]})
            edit->setToolTip(tr("Service-host path, owned by the service"));
    }
    updateOptionalInputUi();
}

QLineEdit* SpiralPanel::addPathRow(QFormLayout* form, const QString& key, const QString& label, bool directory)
{
    auto* container = new QWidget(form->parentWidget());
    auto* row = new QHBoxLayout(container); row->setContentsMargins(0, 0, 0, 0);
    auto* edit = new QLineEdit(container);
    auto* browse = new QToolButton(container); browse->setText(QStringLiteral("…"));
    row->addWidget(edit, 1); row->addWidget(browse);
    form->addRow(label, container);
    _paths[key] = edit; _pathDirectories[key] = directory;
    _pathBrowseButtons[key] = browse;
    connect(edit, &QLineEdit::textEdited, this, [this, key](const QString&) {
        if (!_applyingResolution) {
            if (key != QStringLiteral("dataset_root")) _hasManualEdits = true;
            refreshReloadRequired();
        }
    });
    connect(browse, &QToolButton::clicked, this, [this, edit, directory, key]() {
        const QString chosen = directory
            ? QFileDialog::getExistingDirectory(this, tr("Select directory"), edit->text())
            : QFileDialog::getOpenFileName(this, tr("Select file"), edit->text());
        if (!chosen.isEmpty()) {
            edit->setText(chosen);
            if (key != QStringLiteral("dataset_root")) _hasManualEdits = true;
            refreshReloadRequired();
        }
    });
    return edit;
}

void SpiralPanel::addPclItem(const QString& path, const QString& role, bool required)
{
    if (path.trimmed().isEmpty()) return;
    const int roleIndex = _pclRole->findData(role);
    const QString roleLabel = roleIndex >= 0 ? _pclRole->itemText(roleIndex) : role;
    auto* item = new QListWidgetItem(tr("%1 — %2").arg(roleLabel, path), _pclList);
    item->setData(Qt::UserRole, path);
    item->setData(Qt::UserRole + 1, role);
    item->setData(Qt::UserRole + 2, required);
    item->setToolTip(path);
}

void SpiralPanel::setVolumes(const QVector<VolumeSelector::VolumeOption>& volumes, const QString& selectedId)
{
    _volumeSelector->setVolumes(volumes, selectedId);
}

void SpiralPanel::setLossMapOptions(const QStringList& names)
{
    const QString selected = _lossMap ? _lossMap->currentData().toString() : QString();
    if (!_lossMap) return;
    const QSignalBlocker blocker(_lossMap);
    _lossMap->clear();
    _lossMap->addItem(tr("No loss overlay"), QString());
    for (const QString& name : names)
        _lossMap->addItem(name, name);
    const int selectedIndex = _lossMap->findData(selected);
    _lossMap->setCurrentIndex(selectedIndex >= 0 ? selectedIndex : 0);
    if (selectedIndex < 0) {
        _lossMapLegend->clear();
        emit lossMapChanged({}, _lossMapOpacity->value() / 100.0);
    }
}

void SpiralPanel::setLossMapLegend(const QString& text)
{
    if (_lossMapLegend) _lossMapLegend->setText(text);
}

void SpiralPanel::setLasagnaFlattenAvailable(bool available, const QString& reason)
{
    if (!_flattenWithLasagna) return;
    _lasagnaFlattenAvailable = available;
    _flattenWithLasagna->setEnabled(available);
    _flattenWithLasagna->setToolTip(
        reason.isEmpty()
            ? tr("Flatten the complete latest Spiral output with Lasagna")
            : reason);
}

void SpiralPanel::setLasagnaFlattenRunning(bool running)
{
    if (!_flattenWithLasagna) return;
    _flattenWithLasagna->setEnabled(!running && _lasagnaFlattenAvailable);
    _flattenWithLasagna->setText(
        running ? tr("Flattening with Lasagna…") : tr("Flatten with Lasagna"));
}

void SpiralPanel::applyResolution(const QJsonObject& resolution, bool force)
{
    if (!force && _hasManualEdits) {
        if (QMessageBox::question(this, tr("Refill Spiral paths"),
                tr("Replace manually edited path rows with the Dataset Root proposals?")) != QMessageBox::Yes) return;
    }
    const QJsonObject before = sessionRequest();
    _applyingResolution = true;
    if (_remoteMode) {
        const QString root = resolution.value(QStringLiteral("root")).toString();
        if (!root.isEmpty()) _paths[QStringLiteral("dataset_root")]->setText(root);
    }
    const QJsonObject resolved = resolution.value("resolved").toObject();
    for (auto it = resolved.begin(); it != resolved.end(); ++it)
        if (_paths.contains(it.key())) _paths[it.key()]->setText(it.value().toString());
    _pclList->clear();
    for (const QJsonValue& value : resolution.value("pcl_inputs").toArray()) {
        const QJsonObject item = value.toObject();
        addPclItem(item.value("path").toString(), item.value("role").toString(),
                   item.value("required").toBool());
    }
    _applyingResolution = false; _hasManualEdits = false;
    // A refreshed resolution that resolves to the very same inputs (e.g. the
    // re-advertisement after committing ephemeral inputs) must not disable Run
    // on the live session; only an actual form change requires a reload.
    if (sessionRequest() != before) refreshReloadRequired();
    const QStringList missing = resolution.value("missing_required").toVariant().toStringList();
    QStringList notes;
    if (!missing.isEmpty()) notes << tr("Missing required: %1").arg(missing.join(", "));
    const QStringList checkpoints = resolution.value("detected_checkpoints").toVariant().toStringList();
    if (_remoteMode && !checkpoints.isEmpty())
        notes << tr("Detected resume checkpoints on the service: %1").arg(checkpoints.join(", "));
    _warnings->setText(notes.join(QStringLiteral("\n")));
}

QJsonObject SpiralPanel::sessionRequest() const
{
    QJsonObject paths;
    for (auto it = _paths.begin(); it != _paths.end(); ++it) {
        QString optionalKey = it.key();
        if (optionalKey == QStringLiteral("normal_x")
            || optionalKey == QStringLiteral("normal_y"))
            optionalKey = QStringLiteral("normals");
        paths[it.key()] = optionalInputEnabled(optionalKey) ? it.value()->text() : QString();
    }
    QJsonArray pcls;
    for (int row = 0; row < _pclList->count(); ++row) {
        const QListWidgetItem* item = _pclList->item(row);
        pcls.append(QJsonObject{{"path", item->data(Qt::UserRole).toString()},
                                {"role", item->data(Qt::UserRole + 1).toString()},
                                {"required", item->data(Qt::UserRole + 2).toBool()}});
    }
    paths["pcls"] = pcls;
    QJsonObject config = sessionAdvancedConfig();
    config[QStringLiteral("save_png_visualizations")] = _savePngVisualizations->isChecked();
    QJsonObject run{{"z_begin", _zBegin->value()}, {"z_end", _zEnd->value()},
                    {"scroll_name", _scrollName->text()}, {"outward_sense", _outwardSense->currentText()},
                    {"voxel_size_um", _voxelSize->value()}, {"lasagna_group", _lasagnaGroup->text()},
                    {"lasagna_scale", _lasagnaScale->value()},
                    {"storage_backend", _storageBackend->currentData().toString()},
                    {"legacy_checkpoint_step", _legacyCheckpointStep->value()},
                    {"run_tag", _runTag->text()},
                    {"render_volume_scale", _renderVolumeScale->value()},
                    {"config", config}};
    return {{"paths", paths}, {"run", run}, {"preview", QJsonObject{{"first_winding", 10}}}};
}

QJsonObject SpiralPanel::influenceConfig() const
{
    const QJsonDocument advanced =
        QJsonDocument::fromJson(_advanced->toPlainText().toUtf8());
    QJsonObject config;
    if (advanced.isObject()) {
        const QJsonObject all = advanced.object();
        for (auto it = all.begin(); it != all.end(); ++it) {
            if (it.key().startsWith(QStringLiteral("interactive_influence_"))
                || it.key() == QStringLiteral("loss_weight_anchor"))
                config[it.key()] = it.value();
        }
    }
    config[QStringLiteral("interactive_influence_enabled")] =
        _influenceEnabled->isChecked();
    config[QStringLiteral("interactive_influence_disable_dt_frac")] =
        _influenceDisableDtPct->value() / 100.0;
    if (_influenceEnabled->isChecked()) {
        config[QStringLiteral("interactive_influence_z")] =
            static_cast<double>(_influenceZ->value());
        config[QStringLiteral("interactive_influence_windings")] =
            _influenceWindings->value();
        config[QStringLiteral("interactive_influence_theta_frac")] =
            _influenceThetaPct->value() / 100.0;
        config[QStringLiteral("loss_weight_anchor")] =
            _influenceAnchorWeight->value();
    }
    return config;
}

QJsonObject SpiralPanel::sessionAdvancedConfig() const
{
    const QJsonDocument advanced =
        QJsonDocument::fromJson(_advanced->toPlainText().toUtf8());
    QJsonObject config = !_advancedProfiles->isDefaultProfile() && advanced.isObject()
        ? advanced.object() : QJsonObject{};
    for (auto it = config.begin(); it != config.end();) {
        if (it.key().startsWith(QStringLiteral("interactive_influence_"))
            || it.key() == QStringLiteral("loss_weight_anchor"))
            it = config.erase(it);
        else
            ++it;
    }
    applyOptionalInputConfig(config, true);
    return config;
}

QJsonObject SpiralPanel::runAdvancedConfig() const
{
    const QJsonDocument advanced =
        QJsonDocument::fromJson(_advanced->toPlainText().toUtf8());
    QJsonObject config;
    if (!advanced.isObject()) return config;
    const QJsonObject all = advanced.object();
    for (const QString& key : _runConfigKeys) {
        if (all.contains(key)) config.insert(key, all.value(key));
    }
    if (_runConfigKeys.contains(QStringLiteral("save_png_visualizations")))
        config[QStringLiteral("save_png_visualizations")] =
            _savePngVisualizations->isChecked();
    applyOptionalInputConfig(config, false);
    for (auto it = config.begin(); it != config.end();) {
        if (!_runConfigKeys.contains(it.key()))
            it = config.erase(it);
        else
            ++it;
    }
    return config;
}

bool SpiralPanel::optionalInputEnabled(const QString& key) const
{
    const auto it = _optionalInputs.constFind(key);
    return it == _optionalInputs.cend() || it.value()->isChecked();
}

void SpiralPanel::applyTrackSamplingConfig(QJsonObject& config) const
{
    if (_trackLengthBinSampling->isChecked()) {
        config[QStringLiteral("track_length_bin_weights")] = QJsonArray{
            _trackShortWeight->value(),
            _trackMediumWeight->value(),
            _trackLongWeight->value(),
        };
    } else {
        config[QStringLiteral("track_length_bin_weights")] = QJsonValue::Null;
    }
    config[QStringLiteral("max_track_crossing_per_step")] =
        _maxTrackCrossings->value();
}

void SpiralPanel::syncTrackSamplingControlsFromAdvanced()
{
    const QJsonDocument document =
        QJsonDocument::fromJson(_advanced->toPlainText().toUtf8());
    if (!document.isObject()) return;
    const QJsonObject config = document.object();

    bool binSampling = false;
    const QJsonValue weightsValue =
        config.value(QStringLiteral("track_length_bin_weights"));
    if (weightsValue.isArray()) {
        const QJsonArray weights = weightsValue.toArray();
        if (weights.size() == 3
            && weights[0].isDouble() && weights[1].isDouble()
            && weights[2].isDouble()) {
            binSampling = true;
            const QSignalBlocker shortBlocker(_trackShortWeight);
            const QSignalBlocker mediumBlocker(_trackMediumWeight);
            const QSignalBlocker longBlocker(_trackLongWeight);
            _trackShortWeight->setValue(weights[0].toDouble());
            _trackMediumWeight->setValue(weights[1].toDouble());
            _trackLongWeight->setValue(weights[2].toDouble());
        }
    }
    {
        const QSignalBlocker blocker(_trackLengthBinSampling);
        _trackLengthBinSampling->setChecked(binSampling);
    }

    const QJsonValue crossings =
        config.value(QStringLiteral("max_track_crossing_per_step"));
    {
        const QSignalBlocker blocker(_maxTrackCrossings);
        const int preparedMaximum = config
            .value(QStringLiteral("track_crossing_precompute_max"))
            .toInt(8);
        _maxTrackCrossings->setMaximum(qMax(0, preparedMaximum));
        _maxTrackCrossings->setValue(crossings.isDouble() ? crossings.toInt() : 0);
    }
    updateTrackSamplingUi();
}

void SpiralPanel::writeTrackSamplingControlsToAdvanced()
{
    QJsonDocument document =
        QJsonDocument::fromJson(_advanced->toPlainText().toUtf8());
    if (!document.isObject()) {
        refreshReloadRequired();
        return;
    }
    QJsonObject config = document.object();
    applyTrackSamplingConfig(config);
    _advancedProfiles->setCurrentText(QString::fromUtf8(
        QJsonDocument(config).toJson(QJsonDocument::Indented)).trimmed());
    refreshReloadRequired();
}

void SpiralPanel::updateTrackSamplingUi()
{
    const bool tracksEnabled = optionalInputEnabled(QStringLiteral("tracks_dbm"));
    _trackLengthBinSampling->setEnabled(tracksEnabled);
    _trackShortWeight->setEnabled(tracksEnabled && _trackLengthBinSampling->isChecked());
    _trackMediumWeight->setEnabled(tracksEnabled && _trackLengthBinSampling->isChecked());
    _trackLongWeight->setEnabled(tracksEnabled && _trackLengthBinSampling->isChecked());
    _maxTrackCrossings->setEnabled(tracksEnabled);
}

void SpiralPanel::applyOptionalInputConfig(QJsonObject& config,
                                            bool includeSelectionFlags) const
{
    const bool verified = optionalInputEnabled(QStringLiteral("verified_patches"));
    const bool unverified = optionalInputEnabled(QStringLiteral("unverified_patches"));
    const bool normals = optionalInputEnabled(QStringLiteral("normals"));
    const bool sdt = optionalInputEnabled(QStringLiteral("surf_sdt"));
    const bool tracks = optionalInputEnabled(QStringLiteral("tracks_dbm"));
    const bool gradMag = optionalInputEnabled(QStringLiteral("gradient_magnitude"));
    const bool fibers = optionalInputEnabled(QStringLiteral("fibers"));

    if (includeSelectionFlags) {
        config[QStringLiteral("use_verified_patches")] = verified;
        config[QStringLiteral("use_unverified_patches")] = unverified;
        config[QStringLiteral("use_normals")] = normals;
        config[QStringLiteral("use_surf_sdt")] = sdt;
        config[QStringLiteral("use_tracks")] = tracks;
        config[QStringLiteral("use_gradient_magnitude")] = gradMag;
        config[QStringLiteral("use_fibers")] = fibers;
    }

    auto zero = [&config](std::initializer_list<const char*> keys) {
        for (const char* key : keys) config[QString::fromLatin1(key)] = 0;
    };
    if (!verified) {
        zero({"loss_weight_patch_radius", "loss_weight_patch_dt",
              "loss_weight_umbilicus", "loss_weight_shell_patch_radius",
              "num_patches_per_step", "num_patches_per_step_for_dt",
              "num_points_per_patch"});
    }
    if (!unverified) {
        zero({"loss_weight_unverified_patch_radius", "loss_weight_unverified_patch_dt",
              "unverified_num_patches_per_step",
              "unverified_num_patches_per_step_for_dt",
              "unverified_num_points_per_patch"});
    }
    if (!normals)
        zero({"loss_weight_dense_normals", "dense_normals_num_points"});
    if (!tracks) {
        zero({"loss_weight_track_radius", "loss_weight_track_dt",
              "track_num_per_step", "track_num_points_per_step"});
    }
    if (!fibers) {
        zero({"loss_weight_unattached_pcl_radius", "loss_weight_unattached_pcl_dt",
              "unattached_pcl_num_per_step", "unattached_pcl_num_points_per_step"});
    }

    const QString spacingMode =
        config.value(QStringLiteral("dense_spacing_mode")).toString(QStringLiteral("grad_mag"));
    if (!sdt || !normals) {
        zero({"loss_weight_dense_spacing_count", "loss_weight_dense_spacing_density",
              "loss_weight_dense_attachment",
              "dense_spacing_count_extra_pairs", "dense_spacing_density_extra_pairs",
              "dense_attachment_num_points"});
        if (spacingMode == QStringLiteral("phase"))
            zero({"loss_weight_dense_spacing", "dense_spacing_num_pairs"});
    }
    if (!gradMag && spacingMode == QStringLiteral("grad_mag"))
        zero({"loss_weight_dense_spacing", "dense_spacing_num_pairs"});
}

void SpiralPanel::updateOptionalInputUi()
{
    for (auto it = _paths.begin(); it != _paths.end(); ++it) {
        QString optionalKey = it.key();
        if (optionalKey == QStringLiteral("normal_x")
            || optionalKey == QStringLiteral("normal_y"))
            optionalKey = QStringLiteral("normals");
        const bool enabled = optionalInputEnabled(optionalKey);
        it.value()->setEnabled(enabled);
        if (_pathBrowseButtons.contains(it.key()))
            _pathBrowseButtons[it.key()]->setEnabled(enabled);
    }
    updateTrackSamplingUi();
}

void SpiralPanel::applySessionRunConfig(const QJsonObject& config, qint64 sessionGeneration)
{
    _runConfigKeys.clear();
    for (auto it = config.begin(); it != config.end(); ++it)
        _runConfigKeys.insert(it.key());
    _advancedSessionGeneration = sessionGeneration;
}

void SpiralPanel::updateStatus(const QJsonObject& status)
{
    const qint64 sessionGeneration =
        status.value(QStringLiteral("session_generation")).toInteger(-1);
    const QJsonObject runConfig = status.value(QStringLiteral("run_config")).toObject();
    const QJsonObject runConfigLimits =
        status.value(QStringLiteral("run_config_limits")).toObject();
    const QJsonObject defaultConfig =
        status.value(QStringLiteral("default_advanced_config")).toObject();
    if (sessionGeneration >= 0 && sessionGeneration != _advancedSessionGeneration
        && !runConfig.isEmpty()) {
        if (!defaultConfig.isEmpty()) {
            _defaultAdvancedConfig = defaultConfig;
            _applyingResolution = true;
            _advancedProfiles->setSessionDefault(defaultConfig);
            _applyingResolution = false;
        }
        applySessionRunConfig(runConfig, sessionGeneration);
        refreshReloadRequired();
    }
    const QJsonValue crossingLimit =
        runConfigLimits.value(QStringLiteral("max_track_crossing_per_step"));
    if (crossingLimit.isDouble()) {
        const QSignalBlocker blocker(_maxTrackCrossings);
        _maxTrackCrossings->setMaximum(qMax(0, crossingLimit.toInt()));
    }

    const QString state = status.value("state").toString();
    QString stateText = tr("Session: %1 — %2 — iteration %3/%4")
        .arg(state, status.value("phase").toString())
        .arg(status.value("current_iteration").toInteger())
        .arg(status.value("target_iteration").toInteger());
    // Status polls arrive every second; without this the reload-required
    // notice set by refreshReloadRequired() vanishes immediately, leaving an
    // unexplained disabled Run button.
    if (_reloadRequired)
        stateText += QStringLiteral("\n")
            + tr("Reload required — fit inputs or training configuration changed");
    _state->setText(stateText);
    const QJsonObject metrics = status.value("latest_metrics").toObject();
    _metrics->setText(metrics.contains("total_loss") ? tr("Loss: %1").arg(metrics.value("total_loss").toDouble()) : QString());
    QStringList diagnostics;
    const QString error = status.value(QStringLiteral("error")).toString();
    if (!error.isEmpty()) diagnostics.push_back(error);
    for (const QJsonValue& warning : status.value(QStringLiteral("warnings")).toArray()) {
        const QString text = warning.toString().trimmed();
        if (!text.isEmpty()) diagnostics.push_back(text);
    }
    _warnings->setText(diagnostics.join(QStringLiteral("\n\n")));
    const bool runnable = state == "Ready" || state == "Paused";
    _sessionRunnable = runnable;
    _run->setEnabled(_connected && runnable && !_reloadRequired);
    _stop->setEnabled(state == "Running");
    _save->setEnabled(_connected && runnable);
    _downloadCheckpoint->setEnabled(_connected && runnable);

    // Ephemeral inputs added to the running fit.
    const QJsonArray ephemeral = status.value(QStringLiteral("ephemeral_inputs")).toArray();
    _ephemeralCount = ephemeral.size();
    int pendingCount = 0;
    _uncommittedCount = 0;
    for (const QJsonValue& value : ephemeral) {
        const QJsonObject input = value.toObject();
        if (input.value(QStringLiteral("state")).toString() == QStringLiteral("pending"))
            ++pendingCount;
        if (!input.value(QStringLiteral("committed")).toBool())
            ++_uncommittedCount;
    }
    // Rebuild only on change: the 1 Hz status poll must not wipe the row the
    // user selected while aiming for Remove.
    if (ephemeral != _lastEphemeral) {
        _lastEphemeral = ephemeral;
        QString selectedKind, selectedId;
        if (const QListWidgetItem* current = _ephemeralList->currentItem()) {
            selectedKind = current->data(Qt::UserRole).toString();
            selectedId = current->data(Qt::UserRole + 1).toString();
        }
        _ephemeralList->clear();
        for (const QJsonValue& value : ephemeral) {
            const QJsonObject input = value.toObject();
            const QString kind = input.value(QStringLiteral("kind")).toString();
            const QString id = input.value(QStringLiteral("id")).toString();
            QString label = tr("%1 %2 — %3")
                .arg(kind, id, input.value(QStringLiteral("state")).toString());
            if (input.value(QStringLiteral("committed")).toBool())
                label += tr(", committed");
            const QString role = input.value(QStringLiteral("role")).toString();
            if (!role.isEmpty()) label += tr(" (%1)").arg(role);
            auto* item = new QListWidgetItem(label, _ephemeralList);
            item->setData(Qt::UserRole, kind);
            item->setData(Qt::UserRole + 1, id);
            item->setData(Qt::UserRole + 2, input.value(QStringLiteral("state")).toString());
            if (kind == selectedKind && id == selectedId) _ephemeralList->setCurrentItem(item);
        }
    }
    const bool commitAvailable = status.value(QStringLiteral("commit_available")).toBool();
    _commitInputs->setEnabled(commitAvailable);
    if (_ephemeralCount > 0 && !commitAvailable && _uncommittedCount > 0)
        _commitHint->setText(tr("Commit unavailable: %1")
                                 .arg(status.value(QStringLiteral("commit_unavailable_reason")).toString()));
    else if (pendingCount > 0)
        _commitHint->setText(state == QStringLiteral("Running")
            ? tr("Pending inputs join the fit when this run pauses and the next run starts")
            : tr("Pending inputs join the fit on the next run"));
    else
        _commitHint->clear();
}

QJsonObject SpiralPanel::normalizedReloadRequest(QJsonObject request) const
{
    // Run-mutable settings are excluded because they can be applied at the
    // next Run boundary without rebuilding the resident session.
    return vc3d::normalizedSpiralReloadRequest(
        std::move(request), _defaultAdvancedConfig, _runConfigKeys);
}

void SpiralPanel::refreshReloadRequired()
{
    if (!_hasSession || _applyingResolution || _loadedSessionRequest.isEmpty()) return;

    QJsonObject current = sessionRequest();

    // A loaded input may be disabled at a Run boundary by forcing its mutable
    // loss weights and sample counts to zero. Re-enabling an input that was not
    // loaded, or changing its path, still requires rebuilding the session.
    QJsonObject currentRun = current.value(QStringLiteral("run")).toObject();
    QJsonObject currentConfig = currentRun.value(QStringLiteral("config")).toObject();
    const QJsonObject loadedRun =
        _loadedSessionRequest.value(QStringLiteral("run")).toObject();
    const QJsonObject loadedConfig =
        loadedRun.value(QStringLiteral("config")).toObject();
    QJsonObject currentPaths = current.value(QStringLiteral("paths")).toObject();
    const QJsonObject loadedPaths =
        _loadedSessionRequest.value(QStringLiteral("paths")).toObject();
    const QHash<QString, QStringList> optionalInputs{
        {QStringLiteral("use_verified_patches"), {QStringLiteral("verified_patches")}},
        {QStringLiteral("use_unverified_patches"), {QStringLiteral("unverified_patches")}},
        {QStringLiteral("use_normals"), {QStringLiteral("normal_x"), QStringLiteral("normal_y")}},
        {QStringLiteral("use_surf_sdt"), {QStringLiteral("surf_sdt")}},
        {QStringLiteral("use_tracks"), {QStringLiteral("tracks_dbm")}},
        {QStringLiteral("use_gradient_magnitude"), {QStringLiteral("gradient_magnitude")}},
        {QStringLiteral("use_fibers"), {QStringLiteral("fibers")}},
    };
    for (auto it = optionalInputs.begin(); it != optionalInputs.end(); ++it) {
        const QString& flag = it.key();
        const bool loaded = loadedConfig.value(flag).toBool(true);
        const bool enabled = currentConfig.value(flag).toBool(true);
        if (!loaded || enabled) continue;
        currentConfig[flag] = true;
        for (const QString& pathKey : it.value()) {
            currentPaths[pathKey] = loadedPaths.value(pathKey);
        }
    }
    currentRun[QStringLiteral("config")] = currentConfig;
    current[QStringLiteral("run")] = currentRun;
    current[QStringLiteral("paths")] = currentPaths;

    const bool wasReloadRequired = _reloadRequired;
    _reloadRequired = normalizedReloadRequest(current)
        != normalizedReloadRequest(_loadedSessionRequest);
    _run->setEnabled(_connected && _sessionRunnable && !_reloadRequired);
    if (_reloadRequired)
        _state->setText(tr("Reload required — fit inputs or session configuration changed"));
    else if (wasReloadRequired)
        _state->setText(tr("Session configuration restored — no reload required"));
}

void SpiralPanel::persist() const
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const QString prefix = formSettingsPrefix();
    for (auto it = _paths.begin(); it != _paths.end(); ++it)
        settings.setValue(prefix + QStringLiteral("paths/") + it.key(), it.value()->text());
    for (auto it = _optionalInputs.begin(); it != _optionalInputs.end(); ++it)
        settings.setValue(prefix + QStringLiteral("use_inputs/") + it.key(),
                          it.value()->isChecked());
    QJsonArray pcls;
    for (int row = 0; row < _pclList->count(); ++row) {
        const QListWidgetItem* item = _pclList->item(row);
        pcls.append(QJsonObject{{"path", item->data(Qt::UserRole).toString()},
                                {"role", item->data(Qt::UserRole + 1).toString()},
                                {"required", item->data(Qt::UserRole + 2).toBool()}});
    }
    settings.setValue(prefix + QStringLiteral("pcls"),
                      QJsonDocument(pcls).toJson(QJsonDocument::Compact));
    settings.setValue(prefix + "z_begin", _zBegin->value());
    settings.setValue(prefix + "z_end", _zEnd->value());
    settings.setValue(prefix + "scroll_name", _scrollName->text());
    settings.setValue(prefix + "outward_sense", _outwardSense->currentText());
    settings.setValue(prefix + "voxel_size_um", _voxelSize->value());
    settings.setValue(prefix + "lasagna_group", _lasagnaGroup->text());
    settings.setValue(prefix + "lasagna_scale", _lasagnaScale->value());
    settings.setValue(prefix + "storage_backend", _storageBackend->currentData().toString());
    settings.setValue(prefix + "legacy_checkpoint_step", _legacyCheckpointStep->value());
    settings.setValue(prefix + "run_tag", _runTag->text());
    settings.setValue(prefix + "render_volume_scale", _renderVolumeScale->value());
    settings.setValue(prefix + "save_png_visualizations", _savePngVisualizations->isChecked());
    settings.setValue(prefix + "influence_enabled", _influenceEnabled->isChecked());
    settings.setValue(prefix + "influence_z", _influenceZ->value());
    settings.setValue(prefix + "influence_windings", _influenceWindings->value());
    settings.setValue(prefix + "influence_theta_pct", _influenceThetaPct->value());
    settings.setValue(prefix + "influence_disable_dt_pct", _influenceDisableDtPct->value());
    settings.setValue(prefix + "influence_anchor_weight", _influenceAnchorWeight->value());
    settings.setValue(prefix + "iterations", _iterations->value());
    settings.remove(prefix + "advanced_config");
}

void SpiralPanel::restore()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    QString prefix = formSettingsPrefix();
    // One-time migration: the original panel persisted everything under
    // global keys; import them into the built-in local profile.
    const bool legacy = _currentProfileId == kLocalhostProfileId
        && !settings.contains(prefix + "z_begin")
        && settings.contains(QStringLiteral("spiral/z_begin"));
    const QString pathsPrefix = legacy ? QStringLiteral("spiral/paths/")
                                       : prefix + QStringLiteral("paths/");
    const QString valuePrefix = legacy ? QStringLiteral("spiral/") : prefix;

    _applyingResolution = true;
    for (auto it = _paths.begin(); it != _paths.end(); ++it)
        it.value()->setText(settings.value(pathsPrefix + it.key()).toString());
    for (auto it = _optionalInputs.begin(); it != _optionalInputs.end(); ++it)
        it.value()->setChecked(settings.value(
            valuePrefix + QStringLiteral("use_inputs/") + it.key(), true).toBool());
    _pclList->clear();
    const QByteArray savedPcls = settings.value(valuePrefix + QStringLiteral("pcls")).toByteArray();
    const QJsonDocument pclDocument = QJsonDocument::fromJson(savedPcls);
    if (pclDocument.isArray()) {
        for (const QJsonValue& value : pclDocument.array()) {
            const QJsonObject item = value.toObject();
            addPclItem(item.value(QStringLiteral("path")).toString(),
                       item.value(QStringLiteral("role")).toString(),
                       item.value(QStringLiteral("required")).toBool());
        }
    } else if (legacy) {
        // Import settings written by the original four-row PCL UI once.
        for (const auto& pair : std::initializer_list<std::pair<const char*, const char*>>{
                 {"pcl_absolute", "absolute"}, {"pcl_patch_overlap", "patch_overlap"},
                 {"pcl_relative", "relative"}, {"pcl_same_winding", "same_winding"},
                 {"pcl_drawn_control_points", "drawn_control_points"}}) {
            const QString path = settings.value(
                QStringLiteral("spiral/paths/") + QString::fromLatin1(pair.first)).toString();
            addPclItem(path, QString::fromLatin1(pair.second));
        }
    }
    _zBegin->setValue(settings.value(valuePrefix + "z_begin", 4000).toInt());
    _zEnd->setValue(settings.value(valuePrefix + "z_end", 17000).toInt());
    _scrollName->setText(settings.value(valuePrefix + "scroll_name", "s1").toString());
    _outwardSense->setCurrentText(settings.value(valuePrefix + "outward_sense", "CW").toString());
    _voxelSize->setValue(settings.value(valuePrefix + "voxel_size_um", 9.6).toDouble());
    _lasagnaGroup->setText(settings.value(valuePrefix + "lasagna_group", "4").toString());
    _lasagnaScale->setValue(settings.value(valuePrefix + "lasagna_scale", 4).toInt());
    const int backend = _storageBackend->findData(settings.value(valuePrefix + "storage_backend", "auto").toString());
    if (backend >= 0) _storageBackend->setCurrentIndex(backend);
    _legacyCheckpointStep->setValue(settings.value(valuePrefix + "legacy_checkpoint_step", 0).toInt());
    _runTag->setText(settings.value(valuePrefix + "run_tag").toString());
    _renderVolumeScale->setValue(settings.value(valuePrefix + "render_volume_scale", 16).toInt());
    _savePngVisualizations->setChecked(
        settings.value(valuePrefix + "save_png_visualizations", false).toBool());
    _influenceEnabled->setChecked(settings.value(valuePrefix + "influence_enabled", false).toBool());
    _influenceZ->setValue(settings.value(valuePrefix + "influence_z", 3000).toInt());
    _influenceWindings->setValue(settings.value(valuePrefix + "influence_windings", 5.0).toDouble());
    _influenceThetaPct->setValue(settings.value(valuePrefix + "influence_theta_pct", 50).toInt());
    _influenceDisableDtPct->setValue(
        settings.value(valuePrefix + "influence_disable_dt_pct", 75).toInt());
    _influenceAnchorWeight->setValue(
        settings.value(valuePrefix + "influence_anchor_weight", 20.0).toDouble());
    _iterations->setValue(settings.value(valuePrefix + "iterations", 100).toInt());
    _advancedProfiles->clearSessionDefault();
    _applyingResolution = false;
    _hasManualEdits = false;
    _hasSession = false;
    _reloadRequired = false;
    _advancedSessionGeneration = -1;
    _runConfigKeys.clear();
    _loadedSessionRequest = {};
    _pendingSessionRequest = {};
    _defaultAdvancedConfig = {};
    updateOptionalInputUi();
}
