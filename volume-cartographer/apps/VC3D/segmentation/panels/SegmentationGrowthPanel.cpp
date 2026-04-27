#include "SegmentationGrowthPanel.hpp"

#include "VCSettings.hpp"
#include "elements/VolumeSelector.hpp"
#include "segmentation/SegmentationCommon.hpp"
#include "segmentation/growth/SegmentationGrowth.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QDir>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QRegularExpression>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QVariant>

#include <algorithm>
#include <cmath>
#include <filesystem>

namespace {
constexpr int kGrowDirUpBit = 1 << 0;
constexpr int kGrowDirDownBit = 1 << 1;
constexpr int kGrowDirLeftBit = 1 << 2;
constexpr int kGrowDirRightBit = 1 << 3;
constexpr int kGrowDirAllMask = kGrowDirUpBit | kGrowDirDownBit | kGrowDirLeftBit | kGrowDirRightBit;
constexpr int kDefaultGrowthMaxSteps = 1024;
constexpr int kPatchTracerMaxSteps = 100000;

bool containsSurfKeyword(const QString& text)
{
    if (text.isEmpty()) {
        return false;
    }
    const QString lowered = text.toLower();
    return lowered.contains(QStringLiteral("surface")) || lowered.contains(QStringLiteral("surf"));
}

std::optional<int> trailingNumber(const QString& text)
{
    static const QRegularExpression numberSuffix(QStringLiteral("(\\d+)$"));
    const auto match = numberSuffix.match(text.trimmed());
    if (match.hasMatch()) {
        return match.captured(1).toInt();
    }
    return std::nullopt;
}

int growthStepMaximumForMethod(SegmentationGrowthMethod method)
{
    return method == SegmentationGrowthMethod::PatchTracer
        ? kPatchTracerMaxSteps
        : kDefaultGrowthMaxSteps;
}
} // namespace

SegmentationGrowthPanel::SegmentationGrowthPanel(const QString& settingsGroup, QWidget* parent)
    : QWidget(parent)
    , _settingsGroup(settingsGroup)
{
    _growthDirectionMask = kGrowDirAllMask;

    auto* panelLayout = new QVBoxLayout(this);
    panelLayout->setContentsMargins(0, 0, 0, 0);
    panelLayout->setSpacing(12);

    _groupGrowth = new QGroupBox(tr("Surface Growth"), this);
    auto* growthLayout = new QVBoxLayout(_groupGrowth);

    // Method selection row
    auto* methodRow = new QHBoxLayout();
    auto* methodLabel = new QLabel(tr("Method:"), _groupGrowth);
    _comboGrowthMethod = new QComboBox(_groupGrowth);
    _comboGrowthMethod->addItem(tr("Tracer"), static_cast<int>(SegmentationGrowthMethod::Tracer));
    _comboGrowthMethod->addItem(tr("Patch Tracer"), static_cast<int>(SegmentationGrowthMethod::PatchTracer));
    _comboGrowthMethod->addItem(tr("Extrapolation"), static_cast<int>(SegmentationGrowthMethod::Extrapolation));
    _comboGrowthMethod->setToolTip(tr("Select the growth algorithm:\n"
                                      "- Tracer: Neural-guided growth using volume data\n"
                                      "- Patch Tracer: Patch-index based growth from loaded surfaces\n"
                                      "- Extrapolation: Simple polynomial extrapolation from boundary points"));
    methodRow->addWidget(methodLabel);
    methodRow->addWidget(_comboGrowthMethod);
    methodRow->addStretch(1);
    growthLayout->addLayout(methodRow);

    // Extrapolation options panel (shown only when Extrapolation method is selected)
    _extrapolationOptionsPanel = new QWidget(_groupGrowth);
    auto* extrapLayout = new QHBoxLayout(_extrapolationOptionsPanel);
    extrapLayout->setContentsMargins(0, 0, 0, 0);
    _lblExtrapolationPoints = new QLabel(tr("Fit points:"), _extrapolationOptionsPanel);
    _spinExtrapolationPoints = new QSpinBox(_extrapolationOptionsPanel);
    _spinExtrapolationPoints->setRange(3, 20);
    _spinExtrapolationPoints->setValue(7);
    _spinExtrapolationPoints->setToolTip(tr("Number of boundary points to use for polynomial fitting."));
    auto* typeLabel = new QLabel(tr("Type:"), _extrapolationOptionsPanel);
    _comboExtrapolationType = new QComboBox(_extrapolationOptionsPanel);
    _comboExtrapolationType->addItem(tr("Linear"), static_cast<int>(ExtrapolationType::Linear));
    _comboExtrapolationType->addItem(tr("Quadratic"), static_cast<int>(ExtrapolationType::Quadratic));
    _comboExtrapolationType->addItem(tr("Linear+Fit"), static_cast<int>(ExtrapolationType::LinearFit));
    _comboExtrapolationType->addItem(tr("Skeleton Path"), static_cast<int>(ExtrapolationType::SkeletonPath));
    _comboExtrapolationType->setToolTip(tr("Extrapolation method:\n"
                                           "- Linear: Fit a straight line (faster, simpler)\n"
                                           "- Quadratic: Fit a parabola (better for curved surfaces)\n"
                                           "- Linear+Fit: Linear extrapolation + Newton refinement to fit selected volume\n"
                                           "- Skeleton Path: Use 2D skeleton analysis + 3D Dijkstra path following"));
    extrapLayout->addWidget(_lblExtrapolationPoints);
    extrapLayout->addWidget(_spinExtrapolationPoints);
    extrapLayout->addSpacing(12);
    extrapLayout->addWidget(typeLabel);
    extrapLayout->addWidget(_comboExtrapolationType);
    extrapLayout->addStretch(1);
    growthLayout->addWidget(_extrapolationOptionsPanel);
    _extrapolationOptionsPanel->setVisible(false);

    // SDT/Newton refinement params (shown only when Linear+Fit is selected)
    _sdtParamsContainer = new QWidget(_groupGrowth);
    auto* sdtLayout = new QHBoxLayout(_sdtParamsContainer);
    sdtLayout->setContentsMargins(0, 0, 0, 0);

    auto* maxStepsLabel = new QLabel(tr("Newton steps:"), _sdtParamsContainer);
    _spinSDTMaxSteps = new QSpinBox(_sdtParamsContainer);
    _spinSDTMaxSteps->setRange(1, 10);
    _spinSDTMaxSteps->setValue(5);
    _spinSDTMaxSteps->setToolTip(tr("Maximum Newton iterations for surface refinement (1-10)."));

    auto* stepSizeLabel = new QLabel(tr("Step size:"), _sdtParamsContainer);
    _spinSDTStepSize = new QDoubleSpinBox(_sdtParamsContainer);
    _spinSDTStepSize->setRange(0.1, 2.0);
    _spinSDTStepSize->setSingleStep(0.1);
    _spinSDTStepSize->setValue(0.8);
    _spinSDTStepSize->setToolTip(tr("Newton step size multiplier (0.1-2.0). Smaller values are more stable."));

    auto* convergenceLabel = new QLabel(tr("Convergence:"), _sdtParamsContainer);
    _spinSDTConvergence = new QDoubleSpinBox(_sdtParamsContainer);
    _spinSDTConvergence->setRange(0.1, 2.0);
    _spinSDTConvergence->setSingleStep(0.1);
    _spinSDTConvergence->setValue(0.5);
    _spinSDTConvergence->setToolTip(tr("Stop refinement when distance < this threshold in voxels (0.1-2.0)."));

    auto* chunkSizeLabel = new QLabel(tr("Chunk:"), _sdtParamsContainer);
    _spinSDTChunkSize = new QSpinBox(_sdtParamsContainer);
    _spinSDTChunkSize->setRange(32, 256);
    _spinSDTChunkSize->setSingleStep(32);
    _spinSDTChunkSize->setValue(128);
    _spinSDTChunkSize->setToolTip(tr("Size of SDT chunks in voxels (32-256). Larger = faster but more memory."));

    sdtLayout->addWidget(maxStepsLabel);
    sdtLayout->addWidget(_spinSDTMaxSteps);
    sdtLayout->addSpacing(8);
    sdtLayout->addWidget(stepSizeLabel);
    sdtLayout->addWidget(_spinSDTStepSize);
    sdtLayout->addSpacing(8);
    sdtLayout->addWidget(convergenceLabel);
    sdtLayout->addWidget(_spinSDTConvergence);
    sdtLayout->addSpacing(8);
    sdtLayout->addWidget(chunkSizeLabel);
    sdtLayout->addWidget(_spinSDTChunkSize);
    sdtLayout->addStretch(1);
    growthLayout->addWidget(_sdtParamsContainer);
    _sdtParamsContainer->setVisible(false);

    // Skeleton path params (shown only when Skeleton Path is selected)
    _skeletonParamsContainer = new QWidget(_groupGrowth);
    auto* skeletonLayout = new QHBoxLayout(_skeletonParamsContainer);
    skeletonLayout->setContentsMargins(0, 0, 0, 0);

    auto* connectivityLabel = new QLabel(tr("Connectivity:"), _skeletonParamsContainer);
    _comboSkeletonConnectivity = new QComboBox(_skeletonParamsContainer);
    _comboSkeletonConnectivity->addItem(tr("6"), 6);
    _comboSkeletonConnectivity->addItem(tr("18"), 18);
    _comboSkeletonConnectivity->addItem(tr("26"), 26);
    _comboSkeletonConnectivity->setCurrentIndex(2);  // Default to 26
    _comboSkeletonConnectivity->setToolTip(tr("3D neighborhood connectivity for Dijkstra pathfinding:\n"
                                              "- 6: Face neighbors only\n"
                                              "- 18: Face + edge neighbors\n"
                                              "- 26: Face + edge + corner neighbors"));

    auto* sliceOrientLabel = new QLabel(tr("Up/Down slice:"), _skeletonParamsContainer);
    _comboSkeletonSliceOrientation = new QComboBox(_skeletonParamsContainer);
    _comboSkeletonSliceOrientation->addItem(tr("YZ (X-slice)"), 0);
    _comboSkeletonSliceOrientation->addItem(tr("XZ (Y-slice)"), 1);
    _comboSkeletonSliceOrientation->setToolTip(tr("For Up/Down growth, which plane to use for 2D skeleton analysis:\n"
                                                   "- YZ (X-slice): Extract slice perpendicular to X axis\n"
                                                   "- XZ (Y-slice): Extract slice perpendicular to Y axis\n"
                                                   "(Left/Right growth always uses XY Z-slices)"));

    auto* skeletonChunkLabel = new QLabel(tr("Chunk:"), _skeletonParamsContainer);
    _spinSkeletonChunkSize = new QSpinBox(_skeletonParamsContainer);
    _spinSkeletonChunkSize->setRange(32, 256);
    _spinSkeletonChunkSize->setSingleStep(32);
    _spinSkeletonChunkSize->setValue(128);
    _spinSkeletonChunkSize->setToolTip(tr("Size of chunks for binary volume loading (32-256). Larger = faster but more memory."));

    auto* searchRadiusLabel = new QLabel(tr("Search:"), _skeletonParamsContainer);
    _spinSkeletonSearchRadius = new QSpinBox(_skeletonParamsContainer);
    _spinSkeletonSearchRadius->setRange(1, 100);
    _spinSkeletonSearchRadius->setSingleStep(1);
    _spinSkeletonSearchRadius->setValue(5);
    _spinSkeletonSearchRadius->setToolTip(tr("When starting point is on background, search this many pixels for nearest component (1-100)."));

    skeletonLayout->addWidget(connectivityLabel);
    skeletonLayout->addWidget(_comboSkeletonConnectivity);
    skeletonLayout->addSpacing(12);
    skeletonLayout->addWidget(sliceOrientLabel);
    skeletonLayout->addWidget(_comboSkeletonSliceOrientation);
    skeletonLayout->addSpacing(12);
    skeletonLayout->addWidget(skeletonChunkLabel);
    skeletonLayout->addWidget(_spinSkeletonChunkSize);
    skeletonLayout->addSpacing(12);
    skeletonLayout->addWidget(searchRadiusLabel);
    skeletonLayout->addWidget(_spinSkeletonSearchRadius);
    skeletonLayout->addStretch(1);
    growthLayout->addWidget(_skeletonParamsContainer);
    _skeletonParamsContainer->setVisible(false);

    auto* dirRow = new QHBoxLayout();
    auto* stepsLabel = new QLabel(tr("Steps:"), _groupGrowth);
    _spinGrowthSteps = new QSpinBox(_groupGrowth);
    _spinGrowthSteps->setRange(0, kDefaultGrowthMaxSteps);
    _spinGrowthSteps->setSingleStep(1);
    _spinGrowthSteps->setToolTip(tr("Number of iterations to run when growing the segmentation."));
    dirRow->addWidget(stepsLabel);
    dirRow->addWidget(_spinGrowthSteps);
    dirRow->addSpacing(16);

    auto* dirLabel = new QLabel(tr("Allowed directions:"), _groupGrowth);
    dirRow->addWidget(dirLabel);
    auto addDirectionCheckbox = [&](const QString& text) {
        auto* box = new QCheckBox(text, _groupGrowth);
        dirRow->addWidget(box);
        return box;
    };
    _chkGrowthDirUp = addDirectionCheckbox(tr("Up"));
    _chkGrowthDirUp->setToolTip(tr("Allow growth steps to move upward along the volume."));
    _chkGrowthDirDown = addDirectionCheckbox(tr("Down"));
    _chkGrowthDirDown->setToolTip(tr("Allow growth steps to move downward along the volume."));
    _chkGrowthDirLeft = addDirectionCheckbox(tr("Left"));
    _chkGrowthDirLeft->setToolTip(tr("Allow growth steps to move left across the volume."));
    _chkGrowthDirRight = addDirectionCheckbox(tr("Right"));
    _chkGrowthDirRight->setToolTip(tr("Allow growth steps to move right across the volume."));
    dirRow->addStretch(1);
    growthLayout->addLayout(dirRow);

    auto* keybindsRow = new QHBoxLayout();
    _chkGrowthKeybindsEnabled = new QCheckBox(tr("Enable growth keybinds (1-6)"), _groupGrowth);
    _chkGrowthKeybindsEnabled->setToolTip(tr("When enabled, keys 1-6 trigger growth in different directions."));
    keybindsRow->addWidget(_chkGrowthKeybindsEnabled);
    keybindsRow->addStretch(1);
    growthLayout->addLayout(keybindsRow);

    auto* zRow = new QHBoxLayout();
    _chkCorrectionsUseZRange = new QCheckBox(tr("Limit Z range"), _groupGrowth);
    _chkCorrectionsUseZRange->setToolTip(tr("Restrict growth requests to the specified slice range."));
    zRow->addWidget(_chkCorrectionsUseZRange);
    zRow->addSpacing(12);
    auto* zMinLabel = new QLabel(tr("Z min"), _groupGrowth);
    _spinCorrectionsZMin = new QSpinBox(_groupGrowth);
    _spinCorrectionsZMin->setRange(-100000, 100000);
    _spinCorrectionsZMin->setToolTip(tr("Lowest slice index used when Z range limits are enabled."));
    auto* zMaxLabel = new QLabel(tr("Z max"), _groupGrowth);
    _spinCorrectionsZMax = new QSpinBox(_groupGrowth);
    _spinCorrectionsZMax->setRange(-100000, 100000);
    _spinCorrectionsZMax->setToolTip(tr("Highest slice index used when Z range limits are enabled."));
    zRow->addWidget(zMinLabel);
    zRow->addWidget(_spinCorrectionsZMin);
    zRow->addSpacing(8);
    zRow->addWidget(zMaxLabel);
    zRow->addWidget(_spinCorrectionsZMax);
    zRow->addStretch(1);
    growthLayout->addLayout(zRow);

    auto* growButtonsRow = new QHBoxLayout();
    _btnGrow = new QPushButton(tr("Grow"), _groupGrowth);
    _btnGrow->setToolTip(tr("Run surface growth using the configured steps and directions."));
    growButtonsRow->addWidget(_btnGrow);

    _btnInpaint = new QPushButton(tr("Inpaint"), _groupGrowth);
    _btnInpaint->setToolTip(tr("Resume the current surface and run tracer inpainting without additional growth."));
    growButtonsRow->addWidget(_btnInpaint);
    growButtonsRow->addStretch(1);
    growthLayout->addLayout(growButtonsRow);

    auto* volumeRow = new QHBoxLayout();
    auto* volumeLabel = new QLabel(tr("Volume:"), _groupGrowth);
    auto* volumeSelector = new VolumeSelector(_groupGrowth);
    volumeSelector->setLabelVisible(false);
    _comboVolumes = volumeSelector->comboBox();
    _comboVolumes->setEnabled(false);
    _comboVolumes->setToolTip(tr("Select which volume provides source data for segmentation growth."));
    volumeRow->addWidget(volumeLabel);
    volumeRow->addWidget(volumeSelector, 1);
    growthLayout->addLayout(volumeRow);

    _groupGrowth->setLayout(growthLayout);
    panelLayout->addWidget(_groupGrowth);

    {
        auto* normalGridRow = new QHBoxLayout();
        _lblNormalGrid = new QLabel(this);
        _lblNormalGrid->setTextFormat(Qt::RichText);
        _lblNormalGrid->setToolTip(tr("Shows whether precomputed normal grids are available for push/pull tools."));
        _lblNormalGrid->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        normalGridRow->addWidget(_lblNormalGrid, 0);

        _editNormalGridPath = new QLineEdit(this);
        _editNormalGridPath->setReadOnly(true);
        _editNormalGridPath->setClearButtonEnabled(false);
        _editNormalGridPath->setVisible(false);
        normalGridRow->addWidget(_editNormalGridPath, 1);
        panelLayout->addLayout(normalGridRow);
    }

    // Normal3D zarr selection (optional)
    {
        auto* normal3dRow = new QHBoxLayout();
        _lblNormal3d = new QLabel(this);
        _lblNormal3d->setTextFormat(Qt::RichText);
        _lblNormal3d->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        normal3dRow->addWidget(_lblNormal3d, 0);

        _editNormal3dPath = new QLineEdit(this);
        _editNormal3dPath->setReadOnly(true);
        _editNormal3dPath->setClearButtonEnabled(false);
        _editNormal3dPath->setVisible(false);
        normal3dRow->addWidget(_editNormal3dPath, 1);

        _comboNormal3d = new QComboBox(this);
        _comboNormal3d->setToolTip(tr("Select Normal3D zarr volume to use for normal3dline constraints."));
        _comboNormal3d->setVisible(false);
        normal3dRow->addWidget(_comboNormal3d, 0);

        panelLayout->addLayout(normal3dRow);
    }

    {
        _patchTracerSourceContainer = new QWidget(this);
        auto* patchTracerRow = new QHBoxLayout();
        patchTracerRow->setContentsMargins(0, 0, 0, 0);
        _patchTracerSourceContainer->setLayout(patchTracerRow);
        _lblPatchTracerSource = new QLabel(tr("Patch Tracer source:"), this);
        patchTracerRow->addWidget(_lblPatchTracerSource, 0);

        _editPatchTracerSourcePath = new QLineEdit(this);
        _editPatchTracerSourcePath->setReadOnly(true);
        patchTracerRow->addWidget(_editPatchTracerSourcePath, 1);

        _btnPatchTracerSourceBrowse = new QPushButton(tr("Browse"), this);
        _btnPatchTracerSourceBrowse->setToolTip(tr("Select a folder of patch surfaces for Patch Tracer growth."));
        patchTracerRow->addWidget(_btnPatchTracerSourceBrowse, 0);
        panelLayout->addWidget(_patchTracerSourceContainer);
    }

    {
        _patchTracerUmbilicusContainer = new QWidget(this);
        auto* umbilicusRow = new QHBoxLayout();
        umbilicusRow->setContentsMargins(0, 0, 0, 0);
        _patchTracerUmbilicusContainer->setLayout(umbilicusRow);
        _lblPatchTracerUmbilicus = new QLabel(tr("Umbilicus:"), this);
        umbilicusRow->addWidget(_lblPatchTracerUmbilicus, 0);

        _editPatchTracerUmbilicusPath = new QLineEdit(this);
        _editPatchTracerUmbilicusPath->setReadOnly(true);
        umbilicusRow->addWidget(_editPatchTracerUmbilicusPath, 1);

        _btnPatchTracerUmbilicusBrowse = new QPushButton(tr("Browse"), this);
        _btnPatchTracerUmbilicusBrowse->setToolTip(tr("Select an umbilicus JSON file for Patch Tracer single-wrap growth."));
        umbilicusRow->addWidget(_btnPatchTracerUmbilicusBrowse, 0);
        panelLayout->addWidget(_patchTracerUmbilicusContainer);
    }

    _groupPatchTracerParams = new QGroupBox(tr("Patch Tracer"), this);
    auto* patchParamsLayout = new QVBoxLayout(_groupPatchTracerParams);
    auto* patchForm = new QFormLayout();
    patchForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);

    auto addIntParam = [&](const QString& label, int min, int max, int step, int value) {
        auto* spin = new QSpinBox(_groupPatchTracerParams);
        spin->setRange(min, max);
        spin->setSingleStep(step);
        spin->setValue(value);
        patchForm->addRow(label, spin);
        connect(spin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this]() { persistPatchTracerParams(); });
        return spin;
    };
    auto addDoubleParam = [&](const QString& label, double min, double max, double step, double value) {
        auto* spin = new QDoubleSpinBox(_groupPatchTracerParams);
        spin->setRange(min, max);
        spin->setSingleStep(step);
        spin->setDecimals(4);
        spin->setValue(value);
        patchForm->addRow(label, spin);
        connect(spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this]() { persistPatchTracerParams(); });
        return spin;
    };

    _spinPatchGlobalStepsPerWindow = addIntParam(tr("Global opt windows"), 0, 100, 1, _patchGlobalStepsPerWindow);
    _spinPatchSrcStep = addIntParam(tr("Source step"), 1, 1000, 1, _patchSrcStep);
    _spinPatchStep = addIntParam(tr("Grid step"), 1, 1000, 1, _patchStep);
    _spinPatchMaxWidth = addIntParam(tr("Max width"), 1, 1000000, 1000, _patchMaxWidth);
    _spinPatchLocalCostInlierThreshold = addDoubleParam(tr("Local cost inlier"), 0.0, 100.0, 0.01, _patchLocalCostInlierThreshold);
    _spinPatchSameSurfaceThreshold = addDoubleParam(tr("Same surface threshold"), 0.0, 1000.0, 0.1, _patchSameSurfaceThreshold);
    _spinPatchStraightWeight = addDoubleParam(tr("Straight weight 2D"), 0.0, 100.0, 0.1, _patchStraightWeight);
    _spinPatchStraightWeight3d = addDoubleParam(tr("Straight weight 3D"), 0.0, 100.0, 0.1, _patchStraightWeight3d);
    _spinPatchSlidingWindowScale = addDoubleParam(tr("Sliding window scale"), 0.01, 100.0, 0.1, _patchSlidingWindowScale);
    _spinPatchZLocationLossWeight = addDoubleParam(tr("Z loc loss weight"), 0.0, 100.0, 0.1, _patchZLocationLossWeight);
    _spinPatchDistLoss2dWeight = addDoubleParam(tr("Dist loss 2D"), 0.0, 100.0, 0.1, _patchDistLoss2dWeight);
    _spinPatchDistLoss3dWeight = addDoubleParam(tr("Dist loss 3D"), 0.0, 100.0, 0.1, _patchDistLoss3dWeight);
    _spinPatchStraightMinCount = addDoubleParam(tr("Straight min count"), 0.0, 16.0, 0.5, _patchStraightMinCount);
    _spinPatchInlierBaseThreshold = addIntParam(tr("Inlier base threshold"), 0, 10000, 1, _patchInlierBaseThreshold);

    _chkPatchFlipX = new QCheckBox(tr("Flip X"), _groupPatchTracerParams);
    _chkPatchDebugImages = new QCheckBox(tr("Debug images"), _groupPatchTracerParams);
    _chkPatchSingleWrap = new QCheckBox(tr("Single wrap"), _groupPatchTracerParams);
    connect(_chkPatchFlipX, &QCheckBox::toggled, this, [this]() { persistPatchTracerParams(); });
    connect(_chkPatchDebugImages, &QCheckBox::toggled, this, [this]() { persistPatchTracerParams(); });
    connect(_chkPatchSingleWrap, &QCheckBox::toggled, this, [this]() { persistPatchTracerParams(); });
    patchForm->addRow(QString(), _chkPatchFlipX);
    patchForm->addRow(QString(), _chkPatchDebugImages);
    patchForm->addRow(QString(), _chkPatchSingleWrap);

    patchParamsLayout->addLayout(patchForm);
    auto* patchButtons = new QHBoxLayout();
    _btnPatchTracerResetDefaults = new QPushButton(tr("Reset to Defaults"), _groupPatchTracerParams);
    connect(_btnPatchTracerResetDefaults, &QPushButton::clicked, this, [this]() {
        resetPatchTracerParams(true);
    });
    patchButtons->addWidget(_btnPatchTracerResetDefaults);
    patchButtons->addStretch(1);
    patchParamsLayout->addLayout(patchButtons);
    panelLayout->addWidget(_groupPatchTracerParams);

    // --- Signal wiring ---

    auto connectDirectionCheckbox = [this](QCheckBox* box) {
        if (!box) return;
        connect(box, &QCheckBox::toggled, this, [this, box](bool) {
            updateGrowthDirectionMaskFromUi(box);
        });
    };
    connectDirectionCheckbox(_chkGrowthDirUp);
    connectDirectionCheckbox(_chkGrowthDirDown);
    connectDirectionCheckbox(_chkGrowthDirLeft);
    connectDirectionCheckbox(_chkGrowthDirRight);

    connect(_chkGrowthKeybindsEnabled, &QCheckBox::toggled, this, [this](bool checked) {
        _growthKeybindsEnabled = checked;
        writeSetting(QStringLiteral("growth_keybinds_enabled"), _growthKeybindsEnabled);
    });

    connect(_spinGrowthSteps, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int value) { applyGrowthSteps(value, true, true); });

    connect(_comboGrowthMethod, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int index) {
                const auto method = static_cast<SegmentationGrowthMethod>(
                    _comboGrowthMethod->itemData(index).toInt());
                setGrowthMethod(method);
            });

    connect(_spinExtrapolationPoints, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int value) {
                _extrapolationPointCount = std::clamp(value, 3, 20);
                writeSetting(QStringLiteral("extrapolation_point_count"), _extrapolationPointCount);
            });

    connect(_comboExtrapolationType, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int index) {
                _extrapolationType = extrapolationTypeFromInt(
                    _comboExtrapolationType->itemData(index).toInt());
                writeSetting(QStringLiteral("extrapolation_type"), static_cast<int>(_extrapolationType));
                if (_sdtParamsContainer) {
                    _sdtParamsContainer->setVisible(
                        _growthMethod == SegmentationGrowthMethod::Extrapolation &&
                        _extrapolationType == ExtrapolationType::LinearFit);
                }
                if (_skeletonParamsContainer) {
                    _skeletonParamsContainer->setVisible(
                        _growthMethod == SegmentationGrowthMethod::Extrapolation &&
                        _extrapolationType == ExtrapolationType::SkeletonPath);
                }
                bool showFitPoints = _extrapolationType != ExtrapolationType::SkeletonPath;
                if (_lblExtrapolationPoints) {
                    _lblExtrapolationPoints->setVisible(showFitPoints);
                }
                if (_spinExtrapolationPoints) {
                    _spinExtrapolationPoints->setVisible(showFitPoints);
                }
            });

    // SDT parameter connections
    connect(_spinSDTMaxSteps, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        _sdtMaxSteps = std::clamp(value, 1, 10);
        writeSetting(QStringLiteral("sdt_max_steps"), _sdtMaxSteps);
    });
    connect(_spinSDTStepSize, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        _sdtStepSize = std::clamp(static_cast<float>(value), 0.1f, 2.0f);
        writeSetting(QStringLiteral("sdt_step_size"), static_cast<double>(_sdtStepSize));
    });
    connect(_spinSDTConvergence, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        _sdtConvergence = std::clamp(static_cast<float>(value), 0.1f, 2.0f);
        writeSetting(QStringLiteral("sdt_convergence"), static_cast<double>(_sdtConvergence));
    });
    connect(_spinSDTChunkSize, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        _sdtChunkSize = std::clamp(value, 32, 256);
        writeSetting(QStringLiteral("sdt_chunk_size"), _sdtChunkSize);
    });

    // Skeleton parameter connections
    connect(_comboSkeletonConnectivity, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        _skeletonConnectivity = _comboSkeletonConnectivity->itemData(index).toInt();
        writeSetting(QStringLiteral("skeleton_connectivity"), _skeletonConnectivity);
    });
    connect(_comboSkeletonSliceOrientation, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        _skeletonSliceOrientation = _comboSkeletonSliceOrientation->itemData(index).toInt();
        writeSetting(QStringLiteral("skeleton_slice_orientation"), _skeletonSliceOrientation);
    });
    connect(_spinSkeletonChunkSize, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        _skeletonChunkSize = std::clamp(value, 32, 256);
        writeSetting(QStringLiteral("skeleton_chunk_size"), _skeletonChunkSize);
    });
    connect(_spinSkeletonSearchRadius, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        _skeletonSearchRadius = std::clamp(value, 1, 100);
        writeSetting(QStringLiteral("skeleton_search_radius"), _skeletonSearchRadius);
    });

    // Grow/inpaint buttons
    connect(_btnGrow, &QPushButton::clicked, this, [this]() {
        const auto allowed = allowedGrowthDirections();
        auto direction = SegmentationGrowthDirection::All;
        if (allowed.size() == 1) {
            direction = allowed.front();
        }
        triggerGrowthRequest(direction, _growthSteps, false);
    });
    connect(_btnInpaint, &QPushButton::clicked, this, [this]() {
        triggerGrowthRequest(SegmentationGrowthDirection::All, 0, true);
    });

    // Volume combo
    connect(_comboVolumes, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0) return;
        const QString volumeId = _comboVolumes->itemData(index).toString();
        if (volumeId.isEmpty() || volumeId == _activeVolumeId) return;
        _activeVolumeId = volumeId;
        emit volumeSelectionChanged(volumeId);
    });

    // Normal3D combo
    connect(_comboNormal3d, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int idx) {
        if (_restoringSettings) return;
        if (idx < 0) return;
        const QString path = _comboNormal3d->itemData(idx).toString();
        if (path.isEmpty() || path == _normal3dSelectedPath) return;
        _normal3dSelectedPath = path;
        writeSetting(QStringLiteral("normal3d_selected_path"), _normal3dSelectedPath);
        updateNormal3dUi();
    });

    connect(_btnPatchTracerSourceBrowse, &QPushButton::clicked, this, [this]() {
        QString initial = _patchTracerSourcePath;
        if (initial.isEmpty() && !_volumePackagePath.isEmpty()) {
            initial = QDir(_volumePackagePath).filePath(QStringLiteral("paths"));
        }
        const QString dir = QFileDialog::getExistingDirectory(this,
                                                              tr("Select Patch Tracer source folder"),
                                                              initial,
                                                              QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
        if (dir.isEmpty()) {
            return;
        }
        _patchTracerSourcePath = QDir::cleanPath(dir);
        writeSetting(QStringLiteral("patch_tracer_source_path"), _patchTracerSourcePath);
        updatePatchTracerSourceUi();
    });

    connect(_btnPatchTracerUmbilicusBrowse, &QPushButton::clicked, this, [this]() {
        QString initial = _patchTracerUmbilicusPath;
        if (initial.isEmpty() && !_volumePackagePath.isEmpty()) {
            initial = _volumePackagePath;
        }
        const QString file = QFileDialog::getOpenFileName(this,
                                                          tr("Select umbilicus JSON"),
                                                          initial,
                                                          tr("JSON Files (*.json);;All Files (*)"));
        if (file.isEmpty()) {
            return;
        }
        _patchTracerUmbilicusPath = QDir::cleanPath(file);
        _patchTracerUmbilicusPathUserSet = true;
        writeSetting(QStringLiteral("patch_tracer_umbilicus_path"), _patchTracerUmbilicusPath);
        writeSetting(QStringLiteral("patch_tracer_umbilicus_path_user_set"), _patchTracerUmbilicusPathUserSet);
        updatePatchTracerUmbilicusUi();
    });

    // Z-range connections
    connect(_chkCorrectionsUseZRange, &QCheckBox::toggled, this, [this](bool enabled) {
        _correctionsZRangeEnabled = enabled;
        writeSetting(QStringLiteral("corrections_z_range_enabled"), _correctionsZRangeEnabled);
        updateGrowthUiState();
        emit correctionsZRangeChanged(enabled, _correctionsZMin, _correctionsZMax);
    });
    connect(_spinCorrectionsZMin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (_correctionsZMin == value) return;
        _correctionsZMin = value;
        writeSetting(QStringLiteral("corrections_z_min"), _correctionsZMin);
        if (_correctionsZRangeEnabled) {
            emit correctionsZRangeChanged(true, _correctionsZMin, _correctionsZMax);
        }
    });
    connect(_spinCorrectionsZMax, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (_correctionsZMax == value) return;
        _correctionsZMax = value;
        writeSetting(QStringLiteral("corrections_z_max"), _correctionsZMax);
        if (_correctionsZRangeEnabled) {
            emit correctionsZRangeChanged(true, _correctionsZMin, _correctionsZMax);
        }
    });
}

void SegmentationGrowthPanel::writeSetting(const QString& key, const QVariant& value)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(_settingsGroup);
    settings.setValue(key, value);
    settings.endGroup();
}

QString SegmentationGrowthPanel::determineDefaultVolumeId(const QVector<QPair<QString, QString>>& volumes,
                                                           const QString& requestedId) const
{
    const auto hasId = [&volumes](const QString& id) {
        return std::any_of(volumes.cbegin(), volumes.cend(), [&](const auto& entry) {
            return entry.first == id;
        });
    };

    QString numericCandidate;
    int numericValue = -1;
    QString keywordCandidate;

    for (const auto& entry : volumes) {
        const QString& id = entry.first;
        const QString& label = entry.second;

        if (!containsSurfKeyword(id) && !containsSurfKeyword(label)) {
            continue;
        }

        const auto numberFromId = trailingNumber(id);
        const auto numberFromLabel = trailingNumber(label);
        const std::optional<int> number = numberFromId ? numberFromId : numberFromLabel;

        if (number) {
            if (*number > numericValue) {
                numericValue = *number;
                numericCandidate = id;
            }
        } else if (keywordCandidate.isEmpty()) {
            keywordCandidate = id;
        }
    }

    if (!numericCandidate.isEmpty()) {
        return numericCandidate;
    }
    if (!keywordCandidate.isEmpty()) {
        return keywordCandidate;
    }
    if (!requestedId.isEmpty() && hasId(requestedId)) {
        return requestedId;
    }
    if (!volumes.isEmpty()) {
        return volumes.front().first;
    }
    return {};
}

void SegmentationGrowthPanel::applyGrowthSteps(int steps, bool persist, bool fromUi)
{
    const int minimum = (_growthMethod == SegmentationGrowthMethod::Corrections) ? 0 : 1;
    const int maximum = growthStepMaximumForMethod(_growthMethod);
    const int clamped = std::clamp(steps, minimum, maximum);

    if (_spinGrowthSteps && _spinGrowthSteps->maximum() != maximum) {
        QSignalBlocker blocker(_spinGrowthSteps);
        _spinGrowthSteps->setMaximum(maximum);
    }

    if ((!fromUi || clamped != steps) && _spinGrowthSteps) {
        QSignalBlocker blocker(_spinGrowthSteps);
        _spinGrowthSteps->setValue(clamped);
    }

    if (clamped > 0) {
        if (_growthMethod == SegmentationGrowthMethod::PatchTracer) {
            _patchTracerGrowthSteps = std::max(1, clamped);
        } else {
            _tracerGrowthSteps = std::max(1, std::min(clamped, kDefaultGrowthMaxSteps));
        }
    }

    _growthSteps = clamped;

    if (persist) {
        if (_growthMethod == SegmentationGrowthMethod::PatchTracer) {
            writeSetting(QStringLiteral("growth_steps_patch_tracer"), _patchTracerGrowthSteps);
        } else {
            writeSetting(QStringLiteral("growth_steps"), _growthSteps);
        }
        writeSetting(QStringLiteral("growth_steps_tracer"), _tracerGrowthSteps);
    }
}

void SegmentationGrowthPanel::setGrowthSteps(int steps, bool persist)
{
    applyGrowthSteps(steps, persist, false);
}

void SegmentationGrowthPanel::setGrowthMethod(SegmentationGrowthMethod method)
{
    if (_growthMethod == method) {
        return;
    }
    const int currentSteps = _growthSteps;
    const SegmentationGrowthMethod previousMethod = _growthMethod;
    if (previousMethod == SegmentationGrowthMethod::PatchTracer) {
        _patchTracerGrowthSteps = (currentSteps > 0) ? currentSteps : std::max(1, _patchTracerGrowthSteps);
    } else if (previousMethod != SegmentationGrowthMethod::Corrections) {
        _tracerGrowthSteps = (currentSteps > 0)
            ? std::min(currentSteps, kDefaultGrowthMaxSteps)
            : std::max(1, _tracerGrowthSteps);
    }
    _growthMethod = method;
    int targetSteps = std::max(1, _tracerGrowthSteps);
    if (method == SegmentationGrowthMethod::Corrections) {
        targetSteps = 0;
    } else if (method == SegmentationGrowthMethod::PatchTracer) {
        targetSteps = std::max(1, _patchTracerGrowthSteps);
    } else {
        targetSteps = std::max(1, _tracerGrowthSteps);
    }
    applyGrowthSteps(targetSteps, true, false);
    writeSetting(QStringLiteral("growth_method"), static_cast<int>(_growthMethod));
    syncUiState(_editingEnabled, _growthInProgress);
    emit growthMethodChanged(_growthMethod);
}

void SegmentationGrowthPanel::setGrowthInProgress(bool running)
{
    if (_growthInProgress == running) {
        return;
    }
    _growthInProgress = running;
    updateGrowthUiState();
}

void SegmentationGrowthPanel::setNormalGridAvailable(bool available)
{
    _normalGridAvailable = available;
    syncUiState(_editingEnabled, _growthInProgress);
}

void SegmentationGrowthPanel::setNormalGridPathHint(const QString& hint)
{
    _normalGridHint = hint;
    QString display = hint.trimmed();
    const int colonIndex = display.indexOf(QLatin1Char(':'));
    if (colonIndex >= 0 && colonIndex + 1 < display.size()) {
        display = display.mid(colonIndex + 1).trimmed();
    }
    _normalGridDisplayPath = display;
    syncUiState(_editingEnabled, _growthInProgress);
}

void SegmentationGrowthPanel::setNormalGridPath(const QString& path)
{
    _normalGridPath = path.trimmed();
    syncUiState(_editingEnabled, _growthInProgress);
}

void SegmentationGrowthPanel::setNormal3dZarrCandidates(const QStringList& candidates, const QString& hint)
{
    _normal3dCandidates = candidates;
    _normal3dHint = hint;
    syncUiState(_editingEnabled, _growthInProgress);
}

void SegmentationGrowthPanel::setVolumePackagePath(const QString& path)
{
    _volumePackagePath = path;
    if (_patchTracerSourcePath.isEmpty() && !_volumePackagePath.isEmpty()) {
        _patchTracerSourcePath = QDir(_volumePackagePath).filePath(QStringLiteral("paths"));
    }
    if (!_patchTracerUmbilicusPathUserSet && !_volumePackagePath.isEmpty()) {
        const QString defaultUmbilicus = QDir(_volumePackagePath).filePath(QStringLiteral("umbilicus.json"));
        std::error_code ec;
        if (std::filesystem::exists(defaultUmbilicus.toStdString(), ec) && !ec) {
            _patchTracerUmbilicusPath = defaultUmbilicus;
        } else {
            _patchTracerUmbilicusPath.clear();
        }
    }
    syncUiState(_editingEnabled, _growthInProgress);
}

void SegmentationGrowthPanel::setAvailableVolumes(const QVector<QPair<QString, QString>>& volumes,
                                                   const QString& activeId)
{
    _volumeEntries = volumes;
    _activeVolumeId = determineDefaultVolumeId(_volumeEntries, activeId);
    if (_comboVolumes) {
        const QSignalBlocker blocker(_comboVolumes);
        _comboVolumes->clear();
        for (const auto& entry : _volumeEntries) {
            const QString& id = entry.first;
            const QString& label = entry.second.isEmpty() ? id : entry.second;
            _comboVolumes->addItem(label, id);
        }
        int idx = _comboVolumes->findData(_activeVolumeId);
        if (idx < 0 && !_volumeEntries.isEmpty()) {
            _activeVolumeId = _comboVolumes->itemData(0).toString();
            idx = 0;
        }
        if (idx >= 0) {
            _comboVolumes->setCurrentIndex(idx);
        }
        _comboVolumes->setEnabled(!_volumeEntries.isEmpty());
    }
}

void SegmentationGrowthPanel::setActiveVolume(const QString& volumeId)
{
    if (_activeVolumeId == volumeId) {
        return;
    }
    _activeVolumeId = volumeId;
    if (_comboVolumes) {
        const QSignalBlocker blocker(_comboVolumes);
        int idx = _comboVolumes->findData(_activeVolumeId);
        if (idx >= 0) {
            _comboVolumes->setCurrentIndex(idx);
        }
    }
}

std::vector<SegmentationGrowthDirection> SegmentationGrowthPanel::allowedGrowthDirections() const
{
    std::vector<SegmentationGrowthDirection> dirs;
    if (_growthDirectionMask & kGrowDirUpBit) {
        dirs.push_back(SegmentationGrowthDirection::Up);
    }
    if (_growthDirectionMask & kGrowDirDownBit) {
        dirs.push_back(SegmentationGrowthDirection::Down);
    }
    if (_growthDirectionMask & kGrowDirLeftBit) {
        dirs.push_back(SegmentationGrowthDirection::Left);
    }
    if (_growthDirectionMask & kGrowDirRightBit) {
        dirs.push_back(SegmentationGrowthDirection::Right);
    }
    if (dirs.empty()) {
        dirs = {
            SegmentationGrowthDirection::Up,
            SegmentationGrowthDirection::Down,
            SegmentationGrowthDirection::Left,
            SegmentationGrowthDirection::Right
        };
    }
    return dirs;
}

std::optional<std::pair<int, int>> SegmentationGrowthPanel::correctionsZRange() const
{
    if (!_correctionsZRangeEnabled) {
        return std::nullopt;
    }
    return std::make_pair(_correctionsZMin, _correctionsZMax);
}

utils::Json SegmentationGrowthPanel::patchTracerParamsJson() const
{
    utils::Json params = utils::Json::object();
    params["global_steps_per_window"] = _patchGlobalStepsPerWindow;
    params["flip_x"] = _patchFlipX;
    params["src_step"] = _patchSrcStep;
    params["step"] = _patchStep;
    params["max_width"] = _patchMaxWidth;
    params["debug_images"] = _patchDebugImages;
    params["single wrap"] = _patchSingleWrap;
    if (!_patchTracerUmbilicusPath.trimmed().isEmpty()) {
        params["umbilicus_path"] = _patchTracerUmbilicusPath.toStdString();
    }
    params["local_cost_inl_th"] = _patchLocalCostInlierThreshold;
    params["same_surface_th"] = _patchSameSurfaceThreshold;
    params["straight_weight"] = _patchStraightWeight;
    params["straight_weight_3D"] = _patchStraightWeight3d;
    params["sliding_w_scale"] = _patchSlidingWindowScale;
    params["z_loc_loss_w"] = _patchZLocationLossWeight;
    params["dist_loss_2d_w"] = _patchDistLoss2dWeight;
    params["dist_loss_3d_w"] = _patchDistLoss3dWeight;
    params["straight_min_count"] = _patchStraightMinCount;
    params["inlier_base_threshold"] = _patchInlierBaseThreshold;
    return params;
}

void SegmentationGrowthPanel::setGrowthDirectionMask(int mask)
{
    mask = normalizeGrowthDirectionMask(mask);
    if (_growthDirectionMask == mask) {
        return;
    }
    _growthDirectionMask = mask;
    writeSetting(QStringLiteral("growth_direction_mask"), _growthDirectionMask);
    applyGrowthDirectionMaskToUi();
}

void SegmentationGrowthPanel::updateGrowthDirectionMaskFromUi(QCheckBox* changedCheckbox)
{
    int mask = 0;
    if (_chkGrowthDirUp && _chkGrowthDirUp->isChecked()) {
        mask |= kGrowDirUpBit;
    }
    if (_chkGrowthDirDown && _chkGrowthDirDown->isChecked()) {
        mask |= kGrowDirDownBit;
    }
    if (_chkGrowthDirLeft && _chkGrowthDirLeft->isChecked()) {
        mask |= kGrowDirLeftBit;
    }
    if (_chkGrowthDirRight && _chkGrowthDirRight->isChecked()) {
        mask |= kGrowDirRightBit;
    }

    if (mask == 0) {
        if (changedCheckbox) {
            const QSignalBlocker blocker(changedCheckbox);
            changedCheckbox->setChecked(true);
        }
        mask = kGrowDirAllMask;
    }

    setGrowthDirectionMask(mask);
}

void SegmentationGrowthPanel::applyGrowthDirectionMaskToUi()
{
    if (_chkGrowthDirUp) {
        const QSignalBlocker blocker(_chkGrowthDirUp);
        _chkGrowthDirUp->setChecked((_growthDirectionMask & kGrowDirUpBit) != 0);
    }
    if (_chkGrowthDirDown) {
        const QSignalBlocker blocker(_chkGrowthDirDown);
        _chkGrowthDirDown->setChecked((_growthDirectionMask & kGrowDirDownBit) != 0);
    }
    if (_chkGrowthDirLeft) {
        const QSignalBlocker blocker(_chkGrowthDirLeft);
        _chkGrowthDirLeft->setChecked((_growthDirectionMask & kGrowDirLeftBit) != 0);
    }
    if (_chkGrowthDirRight) {
        const QSignalBlocker blocker(_chkGrowthDirRight);
        _chkGrowthDirRight->setChecked((_growthDirectionMask & kGrowDirRightBit) != 0);
    }
}

int SegmentationGrowthPanel::normalizeGrowthDirectionMask(int mask)
{
    mask &= kGrowDirAllMask;
    if (mask == 0) {
        // If no directions are selected, enable all directions by default.
        // This ensures that growth is not unintentionally disabled.
        mask = kGrowDirAllMask;
    }
    return mask;
}

void SegmentationGrowthPanel::updateGrowthUiState()
{
    const bool enableGrowth = _editingEnabled && !_growthInProgress;
    if (_spinGrowthSteps) {
        _spinGrowthSteps->setEnabled(enableGrowth);
    }
    if (_btnGrow) {
        _btnGrow->setEnabled(enableGrowth);
    }
    if (_btnInpaint) {
        _btnInpaint->setEnabled(enableGrowth);
    }
    const bool enableDirCheckbox = enableGrowth;
    if (_chkGrowthDirUp) {
        _chkGrowthDirUp->setEnabled(enableDirCheckbox);
    }
    if (_chkGrowthDirDown) {
        _chkGrowthDirDown->setEnabled(enableDirCheckbox);
    }
    if (_chkGrowthDirLeft) {
        _chkGrowthDirLeft->setEnabled(enableDirCheckbox);
    }
    if (_chkGrowthDirRight) {
        _chkGrowthDirRight->setEnabled(enableDirCheckbox);
    }

    const bool allowZRange = _editingEnabled && !_growthInProgress;
    if (_chkCorrectionsUseZRange) {
        _chkCorrectionsUseZRange->setEnabled(allowZRange);
    }
    if (_spinCorrectionsZMin) {
        _spinCorrectionsZMin->setEnabled(allowZRange && _correctionsZRangeEnabled);
    }
    if (_spinCorrectionsZMax) {
        _spinCorrectionsZMax->setEnabled(allowZRange && _correctionsZRangeEnabled);
    }
    if (_btnPatchTracerSourceBrowse) {
        _btnPatchTracerSourceBrowse->setEnabled(enableGrowth);
    }
    if (_btnPatchTracerUmbilicusBrowse) {
        _btnPatchTracerUmbilicusBrowse->setEnabled(enableGrowth);
    }
    if (_groupPatchTracerParams) {
        _groupPatchTracerParams->setEnabled(enableGrowth);
    }
}

void SegmentationGrowthPanel::updateNormal3dUi()
{
    if (!_lblNormal3d) {
        return;
    }

    const int count = _normal3dCandidates.size();
    const bool hasAny = count > 0;

    // Keep selection valid.
    if (hasAny) {
        if (_normal3dSelectedPath.isEmpty() || !_normal3dCandidates.contains(_normal3dSelectedPath)) {
            _normal3dSelectedPath = _normal3dCandidates.front();
        }
    } else {
        _normal3dSelectedPath.clear();
    }

    const bool showCombo = count > 1;
    if (_comboNormal3d) {
        _comboNormal3d->setVisible(showCombo);
        _comboNormal3d->setEnabled(_editingEnabled && hasAny);
        if (showCombo) {
            const QSignalBlocker blocker(_comboNormal3d);
            _comboNormal3d->clear();
            for (const QString& p : _normal3dCandidates) {
                _comboNormal3d->addItem(p, p);
            }
            const int idx = _comboNormal3d->findData(_normal3dSelectedPath);
            if (idx >= 0) {
                _comboNormal3d->setCurrentIndex(idx);
            }
        }
    }

    const QString icon = hasAny
        ? QStringLiteral("<span style=\"color:#2e7d32; font-size:16px;\">&#10003;</span>")
        : QStringLiteral("<span style=\"color:#c62828; font-size:16px;\">&#10007;</span>");

    QString message;
    if (!hasAny) {
        message = tr("Normal3D volume not found.");
    } else if (count == 1) {
        message = tr("Normal3D volume found.");
    } else {
        message = tr("Normal3D volumes found (%1). Select one:").arg(count);
    }

    QString tooltip = message;
    if (!_normal3dHint.isEmpty()) {
        tooltip.append(QStringLiteral("\n"));
        tooltip.append(_normal3dHint);
    }
    if (!_volumePackagePath.isEmpty()) {
        tooltip.append(QStringLiteral("\n"));
        tooltip.append(tr("Volume package: %1").arg(_volumePackagePath));
    }

    _lblNormal3d->setText(icon + QStringLiteral("&nbsp;") + message);
    _lblNormal3d->setToolTip(tooltip);
    _lblNormal3d->setAccessibleDescription(message);

    if (_editNormal3dPath) {
        const bool show = hasAny && !showCombo;
        _editNormal3dPath->setVisible(show);
        _editNormal3dPath->setText(_normal3dSelectedPath);
        _editNormal3dPath->setToolTip(_normal3dSelectedPath);
    }
}

void SegmentationGrowthPanel::updatePatchTracerSourceUi()
{
    if (_patchTracerSourcePath.isEmpty() && !_volumePackagePath.isEmpty()) {
        _patchTracerSourcePath = QDir(_volumePackagePath).filePath(QStringLiteral("paths"));
    }
    if (_editPatchTracerSourcePath) {
        _editPatchTracerSourcePath->setText(_patchTracerSourcePath);
        _editPatchTracerSourcePath->setToolTip(_patchTracerSourcePath);
    }
}

void SegmentationGrowthPanel::updatePatchTracerUmbilicusUi()
{
    if (!_patchTracerUmbilicusPathUserSet && _patchTracerUmbilicusPath.isEmpty() && !_volumePackagePath.isEmpty()) {
        const QString defaultUmbilicus = QDir(_volumePackagePath).filePath(QStringLiteral("umbilicus.json"));
        std::error_code ec;
        if (std::filesystem::exists(defaultUmbilicus.toStdString(), ec) && !ec) {
            _patchTracerUmbilicusPath = defaultUmbilicus;
        }
    }
    if (_editPatchTracerUmbilicusPath) {
        _editPatchTracerUmbilicusPath->setText(_patchTracerUmbilicusPath);
        _editPatchTracerUmbilicusPath->setToolTip(_patchTracerUmbilicusPath);
    }
}

void SegmentationGrowthPanel::resetPatchTracerParams(bool persist)
{
    _patchGlobalStepsPerWindow = 0;
    _patchFlipX = false;
    _patchSrcStep = 20;
    _patchStep = 10;
    _patchMaxWidth = 80000;
    _patchDebugImages = false;
    _patchSingleWrap = false;
    _patchLocalCostInlierThreshold = 0.2;
    _patchSameSurfaceThreshold = 2.0;
    _patchStraightWeight = 0.7;
    _patchStraightWeight3d = 4.0;
    _patchSlidingWindowScale = 1.0;
    _patchZLocationLossWeight = 0.1;
    _patchDistLoss2dWeight = 1.0;
    _patchDistLoss3dWeight = 2.0;
    _patchStraightMinCount = 1.0;
    _patchInlierBaseThreshold = 20;
    syncPatchTracerParamsUi();
    if (persist) {
        persistPatchTracerParams();
    }
}

void SegmentationGrowthPanel::syncPatchTracerParamsUi()
{
    auto setSpin = [](auto* spin, auto value) {
        if (!spin) {
            return;
        }
        const QSignalBlocker blocker(spin);
        spin->setValue(value);
    };
    setSpin(_spinPatchGlobalStepsPerWindow, _patchGlobalStepsPerWindow);
    setSpin(_spinPatchSrcStep, _patchSrcStep);
    setSpin(_spinPatchStep, _patchStep);
    setSpin(_spinPatchMaxWidth, _patchMaxWidth);
    setSpin(_spinPatchLocalCostInlierThreshold, _patchLocalCostInlierThreshold);
    setSpin(_spinPatchSameSurfaceThreshold, _patchSameSurfaceThreshold);
    setSpin(_spinPatchStraightWeight, _patchStraightWeight);
    setSpin(_spinPatchStraightWeight3d, _patchStraightWeight3d);
    setSpin(_spinPatchSlidingWindowScale, _patchSlidingWindowScale);
    setSpin(_spinPatchZLocationLossWeight, _patchZLocationLossWeight);
    setSpin(_spinPatchDistLoss2dWeight, _patchDistLoss2dWeight);
    setSpin(_spinPatchDistLoss3dWeight, _patchDistLoss3dWeight);
    setSpin(_spinPatchStraightMinCount, _patchStraightMinCount);
    setSpin(_spinPatchInlierBaseThreshold, _patchInlierBaseThreshold);
    if (_chkPatchFlipX) {
        const QSignalBlocker blocker(_chkPatchFlipX);
        _chkPatchFlipX->setChecked(_patchFlipX);
    }
    if (_chkPatchDebugImages) {
        const QSignalBlocker blocker(_chkPatchDebugImages);
        _chkPatchDebugImages->setChecked(_patchDebugImages);
    }
    if (_chkPatchSingleWrap) {
        const QSignalBlocker blocker(_chkPatchSingleWrap);
        _chkPatchSingleWrap->setChecked(_patchSingleWrap);
    }
}

void SegmentationGrowthPanel::persistPatchTracerParams()
{
    if (_restoringSettings) {
        return;
    }
    if (_spinPatchGlobalStepsPerWindow) _patchGlobalStepsPerWindow = _spinPatchGlobalStepsPerWindow->value();
    if (_spinPatchSrcStep) _patchSrcStep = _spinPatchSrcStep->value();
    if (_spinPatchStep) _patchStep = _spinPatchStep->value();
    if (_spinPatchMaxWidth) _patchMaxWidth = _spinPatchMaxWidth->value();
    if (_spinPatchLocalCostInlierThreshold) _patchLocalCostInlierThreshold = _spinPatchLocalCostInlierThreshold->value();
    if (_spinPatchSameSurfaceThreshold) _patchSameSurfaceThreshold = _spinPatchSameSurfaceThreshold->value();
    if (_spinPatchStraightWeight) _patchStraightWeight = _spinPatchStraightWeight->value();
    if (_spinPatchStraightWeight3d) _patchStraightWeight3d = _spinPatchStraightWeight3d->value();
    if (_spinPatchSlidingWindowScale) _patchSlidingWindowScale = _spinPatchSlidingWindowScale->value();
    if (_spinPatchZLocationLossWeight) _patchZLocationLossWeight = _spinPatchZLocationLossWeight->value();
    if (_spinPatchDistLoss2dWeight) _patchDistLoss2dWeight = _spinPatchDistLoss2dWeight->value();
    if (_spinPatchDistLoss3dWeight) _patchDistLoss3dWeight = _spinPatchDistLoss3dWeight->value();
    if (_spinPatchStraightMinCount) _patchStraightMinCount = _spinPatchStraightMinCount->value();
    if (_spinPatchInlierBaseThreshold) _patchInlierBaseThreshold = _spinPatchInlierBaseThreshold->value();
    if (_chkPatchFlipX) _patchFlipX = _chkPatchFlipX->isChecked();
    if (_chkPatchDebugImages) _patchDebugImages = _chkPatchDebugImages->isChecked();
    if (_chkPatchSingleWrap) _patchSingleWrap = _chkPatchSingleWrap->isChecked();

    writeSetting(QStringLiteral("patch_global_steps_per_window"), _patchGlobalStepsPerWindow);
    writeSetting(QStringLiteral("patch_flip_x"), _patchFlipX);
    writeSetting(QStringLiteral("patch_src_step"), _patchSrcStep);
    writeSetting(QStringLiteral("patch_step"), _patchStep);
    writeSetting(QStringLiteral("patch_max_width"), _patchMaxWidth);
    writeSetting(QStringLiteral("patch_debug_images"), _patchDebugImages);
    writeSetting(QStringLiteral("patch_single_wrap"), _patchSingleWrap);
    writeSetting(QStringLiteral("patch_local_cost_inl_th"), _patchLocalCostInlierThreshold);
    writeSetting(QStringLiteral("patch_same_surface_th"), _patchSameSurfaceThreshold);
    writeSetting(QStringLiteral("patch_straight_weight"), _patchStraightWeight);
    writeSetting(QStringLiteral("patch_straight_weight_3D"), _patchStraightWeight3d);
    writeSetting(QStringLiteral("patch_sliding_w_scale"), _patchSlidingWindowScale);
    writeSetting(QStringLiteral("patch_z_loc_loss_w"), _patchZLocationLossWeight);
    writeSetting(QStringLiteral("patch_dist_loss_2d_w"), _patchDistLoss2dWeight);
    writeSetting(QStringLiteral("patch_dist_loss_3d_w"), _patchDistLoss3dWeight);
    writeSetting(QStringLiteral("patch_straight_min_count"), _patchStraightMinCount);
    writeSetting(QStringLiteral("patch_inlier_base_threshold"), _patchInlierBaseThreshold);
}

void SegmentationGrowthPanel::triggerGrowthRequest(SegmentationGrowthDirection direction,
                                                    int steps,
                                                    bool inpaintOnly)
{
    if (!_editingEnabled || _growthInProgress) {
        return;
    }

    const SegmentationGrowthMethod method = inpaintOnly
        ? SegmentationGrowthMethod::Tracer
        : _growthMethod;

    const bool allowZeroSteps = inpaintOnly || method == SegmentationGrowthMethod::Corrections;
    const int minSteps = allowZeroSteps ? 0 : 1;
    const int clampedSteps = std::clamp(steps, minSteps, growthStepMaximumForMethod(method));
    const int finalSteps = clampedSteps;

    qCInfo(lcSegWidget) << "Grow request" << segmentationGrowthMethodToString(method)
                        << segmentationGrowthDirectionToString(direction)
                        << "steps" << finalSteps
                        << "inpaintOnly" << inpaintOnly;
    emit growSurfaceRequested(method, direction, finalSteps, inpaintOnly);
}

void SegmentationGrowthPanel::restoreSettings(QSettings& settings)
{
    using namespace vc3d::settings;
    _restoringSettings = true;

    _growthMethod = segmentationGrowthMethodFromInt(
        settings.value(segmentation::GROWTH_METHOD, static_cast<int>(_growthMethod)).toInt());
    _extrapolationPointCount = settings.value(QStringLiteral("extrapolation_point_count"), _extrapolationPointCount).toInt();
    _extrapolationPointCount = std::clamp(_extrapolationPointCount, 3, 20);
    _extrapolationType = extrapolationTypeFromInt(
        settings.value(QStringLiteral("extrapolation_type"), static_cast<int>(_extrapolationType)).toInt());

    _sdtMaxSteps = std::clamp(settings.value(QStringLiteral("sdt_max_steps"), _sdtMaxSteps).toInt(), 1, 10);
    _sdtStepSize = std::clamp(settings.value(QStringLiteral("sdt_step_size"), static_cast<double>(_sdtStepSize)).toFloat(), 0.1f, 2.0f);
    _sdtConvergence = std::clamp(settings.value(QStringLiteral("sdt_convergence"), static_cast<double>(_sdtConvergence)).toFloat(), 0.1f, 2.0f);
    _sdtChunkSize = std::clamp(settings.value(QStringLiteral("sdt_chunk_size"), _sdtChunkSize).toInt(), 32, 256);

    int storedConnectivity = settings.value(QStringLiteral("skeleton_connectivity"), _skeletonConnectivity).toInt();
    if (storedConnectivity == 6 || storedConnectivity == 18 || storedConnectivity == 26) {
        _skeletonConnectivity = storedConnectivity;
    }
    _skeletonSliceOrientation = std::clamp(settings.value(QStringLiteral("skeleton_slice_orientation"), _skeletonSliceOrientation).toInt(), 0, 1);
    _skeletonChunkSize = std::clamp(settings.value(QStringLiteral("skeleton_chunk_size"), _skeletonChunkSize).toInt(), 32, 256);
    _skeletonSearchRadius = std::clamp(settings.value(QStringLiteral("skeleton_search_radius"), _skeletonSearchRadius).toInt(), 1, 100);

    int storedGrowthSteps = settings.value(segmentation::GROWTH_STEPS, _growthSteps).toInt();
    storedGrowthSteps = std::clamp(storedGrowthSteps, 0, 1024);
    _tracerGrowthSteps = settings
                             .value(QStringLiteral("growth_steps_tracer"),
                                    std::max(1, storedGrowthSteps))
                             .toInt();
    _tracerGrowthSteps = std::clamp(_tracerGrowthSteps, 1, kDefaultGrowthMaxSteps);
    _patchTracerGrowthSteps = settings
                                  .value(QStringLiteral("growth_steps_patch_tracer"),
                                         _patchTracerGrowthSteps)
                                  .toInt();
    _patchTracerGrowthSteps = std::clamp(_patchTracerGrowthSteps, 1, kPatchTracerMaxSteps);
    applyGrowthSteps(_growthMethod == SegmentationGrowthMethod::PatchTracer
                         ? _patchTracerGrowthSteps
                         : storedGrowthSteps,
                     false,
                     false);
    _growthDirectionMask = normalizeGrowthDirectionMask(
        settings.value(segmentation::GROWTH_DIRECTION_MASK, kGrowDirAllMask).toInt());
    _growthKeybindsEnabled = settings.value(segmentation::GROWTH_KEYBINDS_ENABLED,
                                            segmentation::GROWTH_KEYBINDS_ENABLED_DEFAULT).toBool();

    _correctionsZRangeEnabled = settings.value(segmentation::CORRECTIONS_Z_RANGE_ENABLED, segmentation::CORRECTIONS_Z_RANGE_ENABLED_DEFAULT).toBool();
    _correctionsZMin = settings.value(segmentation::CORRECTIONS_Z_MIN, segmentation::CORRECTIONS_Z_MIN_DEFAULT).toInt();
    _correctionsZMax = settings.value(segmentation::CORRECTIONS_Z_MAX, _correctionsZMin).toInt();
    if (_correctionsZMax < _correctionsZMin) {
        _correctionsZMax = _correctionsZMin;
    }

    _normal3dSelectedPath = settings.value(QStringLiteral("normal3d_selected_path"), QString()).toString();
    _patchTracerSourcePath = settings.value(QStringLiteral("patch_tracer_source_path"), QString()).toString();
    _patchTracerUmbilicusPath = settings.value(QStringLiteral("patch_tracer_umbilicus_path"), QString()).toString();
    _patchTracerUmbilicusPathUserSet = settings.value(QStringLiteral("patch_tracer_umbilicus_path_user_set"), false).toBool();
    _patchGlobalStepsPerWindow = settings.value(QStringLiteral("patch_global_steps_per_window"), _patchGlobalStepsPerWindow).toInt();
    _patchFlipX = settings.value(QStringLiteral("patch_flip_x"), _patchFlipX).toBool();
    _patchSrcStep = std::clamp(settings.value(QStringLiteral("patch_src_step"), _patchSrcStep).toInt(), 1, 1000);
    _patchStep = std::clamp(settings.value(QStringLiteral("patch_step"), _patchStep).toInt(), 1, 1000);
    _patchMaxWidth = std::clamp(settings.value(QStringLiteral("patch_max_width"), _patchMaxWidth).toInt(), 1, 1000000);
    _patchDebugImages = settings.value(QStringLiteral("patch_debug_images"), _patchDebugImages).toBool();
    _patchSingleWrap = settings.value(QStringLiteral("patch_single_wrap"), _patchSingleWrap).toBool();
    _patchLocalCostInlierThreshold = std::clamp(settings.value(QStringLiteral("patch_local_cost_inl_th"), _patchLocalCostInlierThreshold).toDouble(), 0.0, 100.0);
    _patchSameSurfaceThreshold = std::clamp(settings.value(QStringLiteral("patch_same_surface_th"), _patchSameSurfaceThreshold).toDouble(), 0.0, 1000.0);
    _patchStraightWeight = std::clamp(settings.value(QStringLiteral("patch_straight_weight"), _patchStraightWeight).toDouble(), 0.0, 100.0);
    _patchStraightWeight3d = std::clamp(settings.value(QStringLiteral("patch_straight_weight_3D"), _patchStraightWeight3d).toDouble(), 0.0, 100.0);
    _patchSlidingWindowScale = std::clamp(settings.value(QStringLiteral("patch_sliding_w_scale"), _patchSlidingWindowScale).toDouble(), 0.01, 100.0);
    _patchZLocationLossWeight = std::clamp(settings.value(QStringLiteral("patch_z_loc_loss_w"), _patchZLocationLossWeight).toDouble(), 0.0, 100.0);
    _patchDistLoss2dWeight = std::clamp(settings.value(QStringLiteral("patch_dist_loss_2d_w"), _patchDistLoss2dWeight).toDouble(), 0.0, 100.0);
    _patchDistLoss3dWeight = std::clamp(settings.value(QStringLiteral("patch_dist_loss_3d_w"), _patchDistLoss3dWeight).toDouble(), 0.0, 100.0);
    _patchStraightMinCount = std::clamp(settings.value(QStringLiteral("patch_straight_min_count"), _patchStraightMinCount).toDouble(), 0.0, 16.0);
    _patchInlierBaseThreshold = std::clamp(settings.value(QStringLiteral("patch_inlier_base_threshold"), _patchInlierBaseThreshold).toInt(), 0, 10000);

    _restoringSettings = false;
}

void SegmentationGrowthPanel::syncUiState(bool editingEnabled, bool growthInProgress)
{
    _editingEnabled = editingEnabled;
    _growthInProgress = growthInProgress;

    if (_spinGrowthSteps) {
        const QSignalBlocker blocker(_spinGrowthSteps);
        _spinGrowthSteps->setMaximum(growthStepMaximumForMethod(_growthMethod));
        _spinGrowthSteps->setValue(_growthSteps);
    }

    if (_comboGrowthMethod) {
        const QSignalBlocker blocker(_comboGrowthMethod);
        int idx = _comboGrowthMethod->findData(static_cast<int>(_growthMethod));
        if (idx >= 0) {
            _comboGrowthMethod->setCurrentIndex(idx);
        }
    }

    if (_extrapolationOptionsPanel) {
        _extrapolationOptionsPanel->setVisible(_growthMethod == SegmentationGrowthMethod::Extrapolation);
    }
    if (_groupPatchTracerParams) {
        _groupPatchTracerParams->setVisible(_growthMethod == SegmentationGrowthMethod::PatchTracer);
    }
    if (_patchTracerSourceContainer) {
        _patchTracerSourceContainer->setVisible(_growthMethod == SegmentationGrowthMethod::PatchTracer);
    }
    if (_patchTracerUmbilicusContainer) {
        _patchTracerUmbilicusContainer->setVisible(_growthMethod == SegmentationGrowthMethod::PatchTracer);
    }

    if (_spinExtrapolationPoints) {
        const QSignalBlocker blocker(_spinExtrapolationPoints);
        _spinExtrapolationPoints->setValue(_extrapolationPointCount);
    }

    if (_comboExtrapolationType) {
        const QSignalBlocker blocker(_comboExtrapolationType);
        int idx = _comboExtrapolationType->findData(static_cast<int>(_extrapolationType));
        if (idx >= 0) {
            _comboExtrapolationType->setCurrentIndex(idx);
        }
    }

    if (_sdtParamsContainer) {
        _sdtParamsContainer->setVisible(
            _growthMethod == SegmentationGrowthMethod::Extrapolation &&
            _extrapolationType == ExtrapolationType::LinearFit);
    }
    if (_spinSDTMaxSteps) {
        const QSignalBlocker blocker(_spinSDTMaxSteps);
        _spinSDTMaxSteps->setValue(_sdtMaxSteps);
    }
    if (_spinSDTStepSize) {
        const QSignalBlocker blocker(_spinSDTStepSize);
        _spinSDTStepSize->setValue(static_cast<double>(_sdtStepSize));
    }
    if (_spinSDTConvergence) {
        const QSignalBlocker blocker(_spinSDTConvergence);
        _spinSDTConvergence->setValue(static_cast<double>(_sdtConvergence));
    }
    if (_spinSDTChunkSize) {
        const QSignalBlocker blocker(_spinSDTChunkSize);
        _spinSDTChunkSize->setValue(_sdtChunkSize);
    }

    if (_skeletonParamsContainer) {
        _skeletonParamsContainer->setVisible(
            _growthMethod == SegmentationGrowthMethod::Extrapolation &&
            _extrapolationType == ExtrapolationType::SkeletonPath);
    }
    if (_comboSkeletonConnectivity) {
        const QSignalBlocker blocker(_comboSkeletonConnectivity);
        int idx = _comboSkeletonConnectivity->findData(_skeletonConnectivity);
        if (idx >= 0) {
            _comboSkeletonConnectivity->setCurrentIndex(idx);
        }
    }
    if (_comboSkeletonSliceOrientation) {
        const QSignalBlocker blocker(_comboSkeletonSliceOrientation);
        int idx = _comboSkeletonSliceOrientation->findData(_skeletonSliceOrientation);
        if (idx >= 0) {
            _comboSkeletonSliceOrientation->setCurrentIndex(idx);
        }
    }
    if (_spinSkeletonChunkSize) {
        const QSignalBlocker blocker(_spinSkeletonChunkSize);
        _spinSkeletonChunkSize->setValue(_skeletonChunkSize);
    }
    if (_spinSkeletonSearchRadius) {
        const QSignalBlocker blocker(_spinSkeletonSearchRadius);
        _spinSkeletonSearchRadius->setValue(_skeletonSearchRadius);
    }
    bool showFitPoints = _extrapolationType != ExtrapolationType::SkeletonPath;
    if (_lblExtrapolationPoints) {
        _lblExtrapolationPoints->setVisible(showFitPoints);
    }
    if (_spinExtrapolationPoints) {
        _spinExtrapolationPoints->setVisible(showFitPoints);
    }

    applyGrowthDirectionMaskToUi();
    if (_chkGrowthKeybindsEnabled) {
        const QSignalBlocker blocker(_chkGrowthKeybindsEnabled);
        _chkGrowthKeybindsEnabled->setChecked(_growthKeybindsEnabled);
    }

    if (_chkCorrectionsUseZRange) {
        const QSignalBlocker blocker(_chkCorrectionsUseZRange);
        _chkCorrectionsUseZRange->setChecked(_correctionsZRangeEnabled);
    }
    if (_spinCorrectionsZMin) {
        const QSignalBlocker blocker(_spinCorrectionsZMin);
        _spinCorrectionsZMin->setValue(_correctionsZMin);
    }
    if (_spinCorrectionsZMax) {
        const QSignalBlocker blocker(_spinCorrectionsZMax);
        _spinCorrectionsZMax->setValue(_correctionsZMax);
    }

    // Normal grid UI
    if (_lblNormalGrid) {
        const QString icon = _normalGridAvailable
            ? QStringLiteral("<span style=\"color:#2e7d32; font-size:16px;\">&#10003;</span>")
            : QStringLiteral("<span style=\"color:#c62828; font-size:16px;\">&#10007;</span>");
        const bool hasExplicitLocation = !_normalGridDisplayPath.isEmpty() && _normalGridDisplayPath != _normalGridHint;
        QString message;
        message = _normalGridAvailable ? tr("Normal grids found.") : tr("Normal grids not found.");
        if (!_normalGridHint.isEmpty()) {
            message.append(QStringLiteral(" ("));
            message.append(_normalGridHint);
            message.append(QLatin1Char(')'));
        }

        QString tooltip = message;
        if (hasExplicitLocation && !_normalGridHint.isEmpty()) {
            tooltip.append(QStringLiteral("\n"));
            tooltip.append(_normalGridHint);
        }
        if (!_volumePackagePath.isEmpty()) {
            tooltip.append(QStringLiteral("\n"));
            tooltip.append(tr("Volume package: %1").arg(_volumePackagePath));
        }

        _lblNormalGrid->setText(icon + QStringLiteral("&nbsp;") + message);
        _lblNormalGrid->setToolTip(tooltip);
        _lblNormalGrid->setAccessibleDescription(message);
    }

    if (_editNormalGridPath) {
        const bool show = _normalGridAvailable && !_normalGridPath.isEmpty();
        _editNormalGridPath->setVisible(show);
        _editNormalGridPath->setText(_normalGridPath);
        _editNormalGridPath->setToolTip(_normalGridPath);
    }

    updateNormal3dUi();
    updatePatchTracerSourceUi();
    updatePatchTracerUmbilicusUi();
    syncPatchTracerParamsUi();

    updateGrowthUiState();
}
