#include "SegmentationWidget.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSettings>
#include <QSpinBox>
#include <QVariant>
#include <QVBoxLayout>
#include <QSignalBlocker>
#include <QLoggingCategory>
#include <QStyle>

#include <algorithm>

#include <cmath>

Q_LOGGING_CATEGORY(lcSegWidget, "vc.segmentation.widget");

SegmentationWidget::SegmentationWidget(QWidget* parent)
    : QWidget(parent)
    , _chkEditing(nullptr)
    , _editingStatus(nullptr)
    , _groupGrowth(nullptr)
    , _groupMasking(nullptr)
    , _groupSampling(nullptr)
    , _spinDownsample(nullptr)
    , _spinRadius(nullptr)
    , _spinSigma(nullptr)
    , _comboInfluenceMode(nullptr)
    , _groupInfluence(nullptr)
    , _groupSliceVisibility(nullptr)
    , _spinSliceFadeDistance(nullptr)
    , _comboSliceDisplayMode(nullptr)
    , _comboRowColMode(nullptr)
    , _spinHighlightDistance(nullptr)
    , _groupHole(nullptr)
    , _spinHoleRadius(nullptr)
    , _spinHoleIterations(nullptr)
    , _chkFillInvalidRegions(nullptr)
    , _groupHandleDisplay(nullptr)
    , _chkHandlesAlwaysVisible(nullptr)
    , _spinHandleDisplayDistance(nullptr)
    , _btnApply(nullptr)
    , _btnReset(nullptr)
    , _btnStopTools(nullptr)
    , _comboGrowthMethod(nullptr)
    , _comboGrowthDirection(nullptr)
    , _spinGrowthSteps(nullptr)
    , _btnGrow(nullptr)
    , _chkGrowthDirUp(nullptr)
    , _chkGrowthDirDown(nullptr)
    , _chkGrowthDirLeft(nullptr)
    , _chkGrowthDirRight(nullptr)
    , _btnMaskEdit(nullptr)
    , _btnMaskApply(nullptr)
    , _spinMaskSampling(nullptr)
    , _comboVolume(nullptr)
    , _groupCorrections(nullptr)
    , _comboCorrections(nullptr)
    , _btnCorrectionsNew(nullptr)
    , _chkCorrectionsAnnotate(nullptr)
    , _chkCorrectionsUseZRange(nullptr)
    , _spinCorrectionsZMin(nullptr)
    , _spinCorrectionsZMax(nullptr)
    , _normalGridStatusWidget(nullptr)
    , _normalGridStatusIcon(nullptr)
    , _normalGridStatusText(nullptr)
{
    setupUI();
    updateEditingUi();
}

void SegmentationWidget::setupUI()
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(6, 6, 6, 6);
    layout->setSpacing(10);

    // Editing toggle and status
    _chkEditing = new QCheckBox(tr("Enable Editing"), this);
    _chkEditing->setToolTip(tr("Toggle interactive segmentation editing mode"));
    layout->addWidget(_chkEditing);

    _editingStatus = new QLabel(tr("Editing disabled"), this);
    layout->addWidget(_editingStatus);

    // Surface growth controls (moved to the top for quicker access)
    _groupGrowth = new QGroupBox(tr("Surface Growth"), this);
    auto* growthLayout = new QVBoxLayout(_groupGrowth);

    auto* methodLayout = new QHBoxLayout();
    auto* methodLabel = new QLabel(tr("Method:"), _groupGrowth);
    _comboGrowthMethod = new QComboBox(_groupGrowth);
    _comboGrowthMethod->addItem(tr("Tracer"), static_cast<int>(SegmentationGrowthMethod::Tracer));
    _comboGrowthMethod->addItem(tr("Corrections"), static_cast<int>(SegmentationGrowthMethod::Corrections));
    methodLayout->addWidget(methodLabel);
    methodLayout->addWidget(_comboGrowthMethod);
    methodLayout->addStretch();
    growthLayout->addLayout(methodLayout);

    auto* directionLayout = new QHBoxLayout();
    auto* directionLabel = new QLabel(tr("Direction:"), _groupGrowth);
    _comboGrowthDirection = new QComboBox(_groupGrowth);
    _comboGrowthDirection->addItem(tr("All"), static_cast<int>(SegmentationGrowthDirection::All));
    _comboGrowthDirection->addItem(tr("Up"), static_cast<int>(SegmentationGrowthDirection::Up));
    _comboGrowthDirection->addItem(tr("Down"), static_cast<int>(SegmentationGrowthDirection::Down));
    _comboGrowthDirection->addItem(tr("Left"), static_cast<int>(SegmentationGrowthDirection::Left));
    _comboGrowthDirection->addItem(tr("Right"), static_cast<int>(SegmentationGrowthDirection::Right));
    directionLayout->addWidget(directionLabel);
    directionLayout->addWidget(_comboGrowthDirection);
    directionLayout->addStretch();
    growthLayout->addLayout(directionLayout);

    auto* allowedLayout = new QHBoxLayout();
    auto* allowedLabel = new QLabel(tr("Allowed directions:"), _groupGrowth);
    allowedLayout->addWidget(allowedLabel);

    _chkGrowthDirUp = new QCheckBox(tr("Up"), _groupGrowth);
    _chkGrowthDirDown = new QCheckBox(tr("Down"), _groupGrowth);
    _chkGrowthDirLeft = new QCheckBox(tr("Left"), _groupGrowth);
    _chkGrowthDirRight = new QCheckBox(tr("Right"), _groupGrowth);

    const QString dirTooltip = tr("Restrict tracer expansion to specific grid directions."
                                   " At least one direction must remain enabled.");
    _chkGrowthDirUp->setToolTip(dirTooltip);
    _chkGrowthDirDown->setToolTip(dirTooltip);
    _chkGrowthDirLeft->setToolTip(dirTooltip);
    _chkGrowthDirRight->setToolTip(dirTooltip);

    allowedLayout->addWidget(_chkGrowthDirUp);
    allowedLayout->addWidget(_chkGrowthDirDown);
    allowedLayout->addWidget(_chkGrowthDirLeft);
    allowedLayout->addWidget(_chkGrowthDirRight);
    allowedLayout->addStretch();
    growthLayout->addLayout(allowedLayout);

    auto* zRangeLayout = new QHBoxLayout();
    _chkCorrectionsUseZRange = new QCheckBox(tr("Limit Z range"), _groupGrowth);
    _chkCorrectionsUseZRange->setToolTip(tr("Restrict tracer growth between the specified Z planes when correction-guided growth runs."));
    zRangeLayout->addWidget(_chkCorrectionsUseZRange);

    auto* zMinLabel = new QLabel(tr("Z min:"), _groupGrowth);
    _spinCorrectionsZMin = new QSpinBox(_groupGrowth);
    _spinCorrectionsZMin->setRange(0, 1000000);
    _spinCorrectionsZMin->setSingleStep(1);
    _spinCorrectionsZMin->setValue(_correctionsZMin);
    _spinCorrectionsZMin->setEnabled(false);

    auto* zMaxLabel = new QLabel(tr("Z max:"), _groupGrowth);
    _spinCorrectionsZMax = new QSpinBox(_groupGrowth);
    _spinCorrectionsZMax->setRange(0, 1000000);
    _spinCorrectionsZMax->setSingleStep(1);
    _spinCorrectionsZMax->setValue(_correctionsZMax);
    _spinCorrectionsZMax->setEnabled(false);

    zRangeLayout->addSpacing(12);
    zRangeLayout->addWidget(zMinLabel);
    zRangeLayout->addWidget(_spinCorrectionsZMin);
    zRangeLayout->addSpacing(8);
    zRangeLayout->addWidget(zMaxLabel);
    zRangeLayout->addWidget(_spinCorrectionsZMax);
    zRangeLayout->addStretch();
    growthLayout->addLayout(zRangeLayout);

    applyGrowthDirectionMaskToUi();

    auto* stepsLayout = new QHBoxLayout();
    auto* stepsLabel = new QLabel(tr("Steps:"), _groupGrowth);
    _spinGrowthSteps = new QSpinBox(_groupGrowth);
    _spinGrowthSteps->setRange(1, 1024);
    _spinGrowthSteps->setSingleStep(1);
    _spinGrowthSteps->setValue(_growthSteps);
    _spinGrowthSteps->setToolTip(tr("Number of grid units to add in the chosen direction"));
    stepsLayout->addWidget(stepsLabel);
    stepsLayout->addWidget(_spinGrowthSteps);
    stepsLayout->addStretch();
    growthLayout->addLayout(stepsLayout);

    _btnGrow = new QPushButton(tr("Grow"), _groupGrowth);
    _btnGrow->setToolTip(tr("Extend the segmentation surface using the selected method"));
    growthLayout->addWidget(_btnGrow);

    auto* volumeLayout = new QHBoxLayout();
    auto* volumeLabel = new QLabel(tr("Volume:"), _groupGrowth);
    _comboVolume = new QComboBox(_groupGrowth);
    _comboVolume->setEnabled(false);
    volumeLayout->addWidget(volumeLabel);
    volumeLayout->addWidget(_comboVolume);
    volumeLayout->addStretch();
    growthLayout->addLayout(volumeLayout);

    _normalGridStatusWidget = new QWidget(_groupGrowth);
    auto* normalGridLayout = new QHBoxLayout(_normalGridStatusWidget);
    normalGridLayout->setContentsMargins(0, 0, 0, 0);
    normalGridLayout->setSpacing(6);
    _normalGridStatusIcon = new QLabel(_normalGridStatusWidget);
    _normalGridStatusText = new QLabel(tr("Normal grid data available"), _normalGridStatusWidget);
    normalGridLayout->addWidget(_normalGridStatusIcon);
    normalGridLayout->addWidget(_normalGridStatusText);
    normalGridLayout->addStretch();
    _normalGridStatusWidget->setVisible(false);
    growthLayout->addWidget(_normalGridStatusWidget);

    layout->addWidget(_groupGrowth);

    _groupMasking = new QGroupBox(tr("Masking"), this);
    auto* maskingLayout = new QVBoxLayout(_groupMasking);

    _btnMaskEdit = new QPushButton(tr("Edit Mask"), _groupMasking);
    _btnMaskEdit->setCheckable(true);
    maskingLayout->addWidget(_btnMaskEdit);

    _btnMaskApply = new QPushButton(tr("Apply Mask"), _groupMasking);
    _btnMaskApply->setEnabled(false);
    maskingLayout->addWidget(_btnMaskApply);

    auto* samplingRow = new QHBoxLayout();
    auto* samplingLabel = new QLabel(tr("Sampling:"), _groupMasking);
    _spinMaskSampling = new QSpinBox(_groupMasking);
    _spinMaskSampling->setRange(1, 64);
    _spinMaskSampling->setSingleStep(1);
    _spinMaskSampling->setValue(_maskSampling);
    _spinMaskSampling->setToolTip(tr("Controls how densely mask preview points are sampled"));
    samplingRow->addWidget(samplingLabel);
    samplingRow->addWidget(_spinMaskSampling);
    samplingRow->addStretch();
    maskingLayout->addLayout(samplingRow);

    auto* maskRadiusRow = new QHBoxLayout();
    auto* maskRadiusLabel = new QLabel(tr("Brush radius:"), _groupMasking);
    _spinMaskRadius = new QSpinBox(_groupMasking);
    _spinMaskRadius->setRange(1, 64);
    _spinMaskRadius->setSingleStep(1);
    _spinMaskRadius->setValue(_maskBrushRadius);
    _spinMaskRadius->setToolTip(tr("Size of the eraser brush in grid cells while mask editing"));
    maskRadiusRow->addWidget(maskRadiusLabel);
    maskRadiusRow->addWidget(_spinMaskRadius);
    maskRadiusRow->addStretch();
    maskingLayout->addLayout(maskRadiusRow);

    layout->addWidget(_groupMasking);

    // Parameter controls
    _groupSampling = new QGroupBox(tr("Sampling"), this);
    auto* samplingLayout = new QVBoxLayout(_groupSampling);

    auto* downsampleLayout = new QHBoxLayout();
    auto* downsampleLabel = new QLabel(tr("Downsample factor:"), _groupSampling);
    _spinDownsample = new QSpinBox(_groupSampling);
    _spinDownsample->setRange(2, 64);
    _spinDownsample->setSingleStep(2);
    _spinDownsample->setValue(_downsample);
    _spinDownsample->setToolTip(tr("Controls how densely surface control points are sampled"));
    downsampleLayout->addWidget(downsampleLabel);
    downsampleLayout->addWidget(_spinDownsample);
    downsampleLayout->addStretch();
    samplingLayout->addLayout(downsampleLayout);

    layout->addWidget(_groupSampling);

    _groupInfluence = new QGroupBox(tr("Influence"), this);
    auto* influenceLayout = new QVBoxLayout(_groupInfluence);

    auto* modeLayout = new QHBoxLayout();
    auto* modeLabel = new QLabel(tr("Falloff mode:"), _groupInfluence);
    _comboInfluenceMode = new QComboBox(_groupInfluence);
    _comboInfluenceMode->addItem(tr("Grid (square)"), static_cast<int>(SegmentationInfluenceMode::GridChebyshev));
    _comboInfluenceMode->addItem(tr("Geodesic (circular)"), static_cast<int>(SegmentationInfluenceMode::GeodesicCircular));
    _comboInfluenceMode->addItem(tr("Row / Column"), static_cast<int>(SegmentationInfluenceMode::RowColumn));
    _comboInfluenceMode->setToolTip(tr("Choose how handle influence decays across the surface"));
    int modeIndex = _comboInfluenceMode->findData(static_cast<int>(_influenceMode));
    if (modeIndex >= 0) {
        _comboInfluenceMode->setCurrentIndex(modeIndex);
    }
    modeLayout->addWidget(modeLabel);
    modeLayout->addWidget(_comboInfluenceMode);
    modeLayout->addStretch();
    influenceLayout->addLayout(modeLayout);

    auto* rowColLayout = new QHBoxLayout();
    auto* rowColLabel = new QLabel(tr("Row/Col preference:"), _groupInfluence);
    _comboRowColMode = new QComboBox(_groupInfluence);
    _comboRowColMode->addItem(tr("Row only"), static_cast<int>(SegmentationRowColMode::RowOnly));
    _comboRowColMode->addItem(tr("Column only"), static_cast<int>(SegmentationRowColMode::ColumnOnly));
    _comboRowColMode->addItem(tr("Dynamic"), static_cast<int>(SegmentationRowColMode::Dynamic));
    _comboRowColMode->setToolTip(tr("When using Row / Column mode, choose if influence spreads along rows, columns, or matches the viewer orientation"));
    int rowColIndex = _comboRowColMode->findData(static_cast<int>(_rowColMode));
    if (rowColIndex >= 0) {
        _comboRowColMode->setCurrentIndex(rowColIndex);
    }
    rowColLayout->addWidget(rowColLabel);
    rowColLayout->addWidget(_comboRowColMode);
    rowColLayout->addStretch();
    influenceLayout->addLayout(rowColLayout);
    _comboRowColMode->setEnabled(false);

    auto* radiusLayout = new QHBoxLayout();
    auto* radiusLabel = new QLabel(tr("Radius:"), _groupInfluence);
    _spinRadius = new QSpinBox(_groupInfluence);
    _spinRadius->setRange(1, 32);
    _spinRadius->setSingleStep(1);
    _spinRadius->setValue(static_cast<int>(std::lround(_radius)));
    _spinRadius->setSuffix(tr(" steps"));
    _spinRadius->setToolTip(tr("Number of grid steps (Chebyshev) influenced around the active handle"));
    radiusLayout->addWidget(radiusLabel);
    radiusLayout->addWidget(_spinRadius);
    radiusLayout->addStretch();
    influenceLayout->addLayout(radiusLayout);

    auto* sigmaLayout = new QHBoxLayout();
    auto* sigmaLabel = new QLabel(tr("Strength (sigma):"), _groupInfluence);
    _spinSigma = new QDoubleSpinBox(_groupInfluence);
    _spinSigma->setDecimals(2);
    _spinSigma->setRange(0.10, 2.00);
    _spinSigma->setSingleStep(0.05);
    _spinSigma->setValue(static_cast<double>(_sigma));
    _spinSigma->setSuffix(tr(" x"));
    _spinSigma->setToolTip(tr("Multiplier for how strongly neighbouring grid points follow the dragged handle"));
    sigmaLayout->addWidget(sigmaLabel);
    sigmaLayout->addWidget(_spinSigma);
    sigmaLayout->addStretch();
    influenceLayout->addLayout(sigmaLayout);

    layout->addWidget(_groupInfluence);

    _groupSliceVisibility = new QGroupBox(tr("Slice Visibility"), this);
    auto* sliceLayout = new QVBoxLayout(_groupSliceVisibility);

    auto* sliceFadeLayout = new QHBoxLayout();
    auto* sliceFadeLabel = new QLabel(tr("Fade distance:"), _groupSliceVisibility);
    _spinSliceFadeDistance = new QDoubleSpinBox(_groupSliceVisibility);
    _spinSliceFadeDistance->setDecimals(1);
    _spinSliceFadeDistance->setRange(0.1, 500.0);
    _spinSliceFadeDistance->setSingleStep(0.5);
    _spinSliceFadeDistance->setValue(static_cast<double>(_sliceFadeDistance));
    _spinSliceFadeDistance->setToolTip(tr("World-space distance from the slice plane where handles begin to fade or hide"));
    sliceFadeLayout->addWidget(sliceFadeLabel);
    sliceFadeLayout->addWidget(_spinSliceFadeDistance);
    sliceFadeLayout->addStretch();
    sliceLayout->addLayout(sliceFadeLayout);

    auto* sliceModeLayout = new QHBoxLayout();
    auto* sliceModeLabel = new QLabel(tr("Beyond distance:"), _groupSliceVisibility);
    _comboSliceDisplayMode = new QComboBox(_groupSliceVisibility);
    _comboSliceDisplayMode->addItem(tr("Fade"), static_cast<int>(SegmentationSliceDisplayMode::Fade));
    _comboSliceDisplayMode->addItem(tr("Hide"), static_cast<int>(SegmentationSliceDisplayMode::Hide));
    int sliceModeIndex = _comboSliceDisplayMode->findData(static_cast<int>(_sliceDisplayMode));
    if (sliceModeIndex >= 0) {
        _comboSliceDisplayMode->setCurrentIndex(sliceModeIndex);
    }
    _comboSliceDisplayMode->setToolTip(tr("Choose whether slice viewers fade handles out or hide them past the distance"));
    sliceModeLayout->addWidget(sliceModeLabel);
    sliceModeLayout->addWidget(_comboSliceDisplayMode);
    sliceModeLayout->addStretch();
    sliceLayout->addLayout(sliceModeLayout);

    layout->addWidget(_groupSliceVisibility);

    _groupHole = new QGroupBox(tr("Hole Filling"), this);
    auto* holeLayout = new QVBoxLayout(_groupHole);

    _chkFillInvalidRegions = new QCheckBox(tr("Fill invalid regions"), _groupHole);
    _chkFillInvalidRegions->setChecked(_fillInvalidRegions);
    _chkFillInvalidRegions->setToolTip(tr("When enabled, gap clicks try to solve and smooth missing grid cells before adding a handle."));
    holeLayout->addWidget(_chkFillInvalidRegions);

    auto* holeRadiusLayout = new QHBoxLayout();
    auto* holeRadiusLabel = new QLabel(tr("Search radius:"), _groupHole);
    _spinHoleRadius = new QSpinBox(_groupHole);
    _spinHoleRadius->setRange(1, 64);
    _spinHoleRadius->setSingleStep(1);
    _spinHoleRadius->setValue(_holeSearchRadius);
    _spinHoleRadius->setSuffix(tr(" cells"));
    _spinHoleRadius->setToolTip(tr("Maximum grid distance flood-filled when creating new points inside holes"));
    holeRadiusLayout->addWidget(holeRadiusLabel);
    holeRadiusLayout->addWidget(_spinHoleRadius);
    holeRadiusLayout->addStretch();
    holeLayout->addLayout(holeRadiusLayout);

    auto* holeIterationsLayout = new QHBoxLayout();
    auto* holeIterationsLabel = new QLabel(tr("Relax iterations:"), _groupHole);
    _spinHoleIterations = new QSpinBox(_groupHole);
    _spinHoleIterations->setRange(1, 200);
    _spinHoleIterations->setSingleStep(1);
    _spinHoleIterations->setValue(_holeSmoothIterations);
    _spinHoleIterations->setToolTip(tr("Number of smoothing passes applied to the filled patch"));
    holeIterationsLayout->addWidget(holeIterationsLabel);
    holeIterationsLayout->addWidget(_spinHoleIterations);
    holeIterationsLayout->addStretch();
    holeLayout->addLayout(holeIterationsLayout);

    layout->addWidget(_groupHole);

    _groupHandleDisplay = new QGroupBox(tr("Handle Display"), this);
    auto* handleDisplayLayout = new QVBoxLayout(_groupHandleDisplay);

    _chkHandlesAlwaysVisible = new QCheckBox(tr("Show all handles"), _groupHandleDisplay);
    _chkHandlesAlwaysVisible->setChecked(_handlesAlwaysVisible);
    _chkHandlesAlwaysVisible->setToolTip(tr("When unchecked, only handles within the specified world distance from the cursor are shown"));
    handleDisplayLayout->addWidget(_chkHandlesAlwaysVisible);

    auto* handleDistanceLayout = new QHBoxLayout();
    auto* handleDistanceLabel = new QLabel(tr("Display distance:"), _groupHandleDisplay);
    _spinHandleDisplayDistance = new QDoubleSpinBox(_groupHandleDisplay);
    _spinHandleDisplayDistance->setRange(1.0, 500.0);
    _spinHandleDisplayDistance->setSingleStep(1.0);
    _spinHandleDisplayDistance->setDecimals(1);
    _spinHandleDisplayDistance->setValue(static_cast<double>(_handleDisplayDistance));
    _spinHandleDisplayDistance->setToolTip(tr("Maximum world-space distance from the cursor used to show nearby handles"));
    handleDistanceLayout->addWidget(handleDistanceLabel);
    handleDistanceLayout->addWidget(_spinHandleDisplayDistance);
    handleDistanceLayout->addStretch();
    handleDisplayLayout->addLayout(handleDistanceLayout);

    auto* highlightLayout = new QHBoxLayout();
    auto* highlightLabel = new QLabel(tr("Highlight distance:"), _groupHandleDisplay);
    _spinHighlightDistance = new QDoubleSpinBox(_groupHandleDisplay);
    _spinHighlightDistance->setRange(0.5, 500.0);
    _spinHighlightDistance->setSingleStep(0.5);
    _spinHighlightDistance->setDecimals(1);
    _spinHighlightDistance->setValue(static_cast<double>(_highlightDistance));
    _spinHighlightDistance->setToolTip(tr("Screen-space radius (pixels) used to select the nearest handle for hover highlighting"));
    highlightLayout->addWidget(highlightLabel);
    highlightLayout->addWidget(_spinHighlightDistance);
    highlightLayout->addStretch();
    handleDisplayLayout->addLayout(highlightLayout);

    layout->addWidget(_groupHandleDisplay);

    _groupCorrections = new QGroupBox(tr("Corrections"), this);
    auto* correctionsLayout = new QVBoxLayout(_groupCorrections);

    auto* correctionsComboLayout = new QHBoxLayout();
    auto* correctionsLabel = new QLabel(tr("Active set:"), _groupCorrections);
    _comboCorrections = new QComboBox(_groupCorrections);
    _comboCorrections->addItem(tr("None"), static_cast<qulonglong>(0));
    correctionsComboLayout->addWidget(correctionsLabel);
    correctionsComboLayout->addWidget(_comboCorrections);
    correctionsComboLayout->addStretch();
    correctionsLayout->addLayout(correctionsComboLayout);

    _btnCorrectionsNew = new QPushButton(tr("New set (T)"), _groupCorrections);
    _btnCorrectionsNew->setToolTip(tr("Create a new correction collection (shortcut: T)"));
    correctionsLayout->addWidget(_btnCorrectionsNew);

    _chkCorrectionsAnnotate = new QCheckBox(tr("Annotate corrections"), _groupCorrections);
    _chkCorrectionsAnnotate->setToolTip(tr("When enabled, left-clicks add points to the active correction set; Ctrl+click removes the nearest point."));
    correctionsLayout->addWidget(_chkCorrectionsAnnotate);

    layout->addWidget(_groupCorrections);

    auto* actionsLayout = new QHBoxLayout();
    _btnApply = new QPushButton(tr("Apply"), this);
    _btnApply->setDefault(true);
    _btnReset = new QPushButton(tr("Reset"), this);
    actionsLayout->addWidget(_btnApply);
    actionsLayout->addWidget(_btnReset);
    layout->addLayout(actionsLayout);

    _btnStopTools = new QPushButton(tr("Stop"), this);
    layout->addWidget(_btnStopTools);

    layout->addStretch();

    restoreSettings();
    updateGrowthModeUi();

    // Signal wiring
    connect(_chkEditing, &QCheckBox::toggled, this, [this](bool enabled) {
        setEditingEnabled(enabled);
        emit editingModeChanged(enabled);
    });

    connect(_spinDownsample, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (value == _downsample) {
            return;
        }
        _downsample = value;
        writeSetting(QStringLiteral("downsample"), _downsample);
        emit downsampleChanged(value);
    });

    connect(_spinRadius, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        float radius = static_cast<float>(value);
        if (std::fabs(radius - _radius) < 1e-4f) {
            return;
        }
        _radius = radius;
        writeSetting(QStringLiteral("radius_steps"), static_cast<int>(std::lround(_radius)));
        emit radiusChanged(radius);
    });

    connect(_spinSigma, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        float sigma = std::max(0.10f, static_cast<float>(value));
        if (std::fabs(sigma - _sigma) < 1e-4f) {
            return;
        }
        _sigma = sigma;
        writeSetting(QStringLiteral("strength"), _sigma);
        emit sigmaChanged(sigma);
    });

    connect(_comboInfluenceMode, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0) {
            return;
        }
        const QVariant modeData = _comboInfluenceMode->itemData(index);
        if (!modeData.isValid()) {
            return;
        }
        const auto mode = static_cast<SegmentationInfluenceMode>(modeData.toInt());
        if (mode == _influenceMode) {
            return;
        }
        _influenceMode = mode;
        writeSetting(QStringLiteral("influence_mode"), static_cast<int>(_influenceMode));
        if (_comboRowColMode) {
            _comboRowColMode->setEnabled(_editingEnabled && _influenceMode == SegmentationInfluenceMode::RowColumn);
        }
        emit influenceModeChanged(_influenceMode);
    });

    connect(_comboRowColMode, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0) {
            return;
        }
        const QVariant modeData = _comboRowColMode->itemData(index);
        if (!modeData.isValid()) {
            return;
        }
        const auto mode = static_cast<SegmentationRowColMode>(modeData.toInt());
        if (mode == _rowColMode) {
            return;
        }
        _rowColMode = mode;
        writeSetting(QStringLiteral("row_col_mode"), static_cast<int>(_rowColMode));
        emit rowColModeChanged(_rowColMode);
    });

    connect(_spinSliceFadeDistance, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        float distance = static_cast<float>(std::max(0.1, value));
        if (std::fabs(distance - _sliceFadeDistance) < 1e-4f) {
            return;
        }
        _sliceFadeDistance = distance;
        writeSetting(QStringLiteral("slice_fade_distance"), _sliceFadeDistance);
        emit sliceFadeDistanceChanged(_sliceFadeDistance);
    });

    connect(_comboSliceDisplayMode, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0) {
            return;
        }
        const QVariant modeData = _comboSliceDisplayMode->itemData(index);
        if (!modeData.isValid()) {
            return;
        }
        const auto mode = static_cast<SegmentationSliceDisplayMode>(modeData.toInt());
        if (mode == _sliceDisplayMode) {
            return;
        }
        _sliceDisplayMode = mode;
        writeSetting(QStringLiteral("slice_display_mode"), static_cast<int>(_sliceDisplayMode));
        emit sliceDisplayModeChanged(_sliceDisplayMode);
    });

    connect(_spinHighlightDistance, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        float distance = static_cast<float>(std::max(0.5, value));
        if (std::fabs(distance - _highlightDistance) < 1e-4f) {
            return;
        }
        _highlightDistance = distance;
        writeSetting(QStringLiteral("highlight_distance"), _highlightDistance);
        emit highlightDistanceChanged(_highlightDistance);
    });

    connect(_comboCorrections, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0) {
            _activeCorrectionId.reset();
            emit correctionsCollectionSelected(0);
            return;
        }
        const QVariant data = _comboCorrections->itemData(index);
        const uint64_t collectionId = static_cast<uint64_t>(data.toULongLong());
        if (collectionId == 0) {
            _activeCorrectionId.reset();
        } else {
            _activeCorrectionId = collectionId;
        }
        emit correctionsCollectionSelected(collectionId);
    });

    connect(_btnCorrectionsNew, &QPushButton::clicked, this, [this]() {
        emit correctionsCreateRequested();
    });

    connect(_chkCorrectionsAnnotate, &QCheckBox::toggled, this, [this](bool enabled) {
        emit correctionsAnnotateToggled(enabled);
    });

    connect(_chkCorrectionsUseZRange, &QCheckBox::toggled, this, [this](bool enabled) {
        _correctionsZRangeEnabled = enabled;
        if (_spinCorrectionsZMin) {
            _spinCorrectionsZMin->setEnabled(enabled);
        }
        if (_spinCorrectionsZMax) {
            _spinCorrectionsZMax->setEnabled(enabled);
        }
        writeSetting(QStringLiteral("corrections_z_range_enabled"), enabled);
        emit correctionsZRangeChanged(enabled, _correctionsZMin, _correctionsZMax);
    });

    connect(_spinCorrectionsZMin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (value == _correctionsZMin) {
            return;
        }
        _correctionsZMin = value;
        writeSetting(QStringLiteral("corrections_z_min"), _correctionsZMin);
        if (_correctionsZRangeEnabled) {
            emit correctionsZRangeChanged(true, _correctionsZMin, _correctionsZMax);
        }
    });

    connect(_spinCorrectionsZMax, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (value == _correctionsZMax) {
            return;
        }
        _correctionsZMax = value;
        writeSetting(QStringLiteral("corrections_z_max"), _correctionsZMax);
        if (_correctionsZRangeEnabled) {
            emit correctionsZRangeChanged(true, _correctionsZMin, _correctionsZMax);
        }
    });

    connect(_comboGrowthMethod, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0) {
            return;
        }
        const QVariant methodData = _comboGrowthMethod->itemData(index);
        if (!methodData.isValid()) {
            return;
        }
        auto method = segmentationGrowthMethodFromInt(methodData.toInt());
        if (method == _growthMethod) {
            return;
        }
        _growthMethod = method;
        writeSetting(QStringLiteral("growth_method"), static_cast<int>(_growthMethod));
        updateGrowthModeUi();
        emit growthMethodChanged(_growthMethod);
    });

    connect(_comboGrowthDirection, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0) {
            return;
        }
        const QVariant dirData = _comboGrowthDirection->itemData(index);
        if (!dirData.isValid()) {
            return;
        }
        auto direction = segmentationGrowthDirectionFromInt(dirData.toInt());
        if (direction == _growthDirection) {
            return;
        }
        _growthDirection = direction;
        writeSetting(QStringLiteral("growth_direction"), static_cast<int>(_growthDirection));
    });

    auto connectDirectionCheckbox = [this](QCheckBox* box) {
        if (!box) {
            return;
        }
        connect(box, &QCheckBox::toggled, this, [this, box](bool) {
            updateGrowthDirectionMaskFromUi(box);
        });
    };

    connectDirectionCheckbox(_chkGrowthDirUp);
    connectDirectionCheckbox(_chkGrowthDirDown);
    connectDirectionCheckbox(_chkGrowthDirLeft);
    connectDirectionCheckbox(_chkGrowthDirRight);

    connect(_spinGrowthSteps, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (value == _growthSteps) {
            return;
        }
        _growthSteps = std::max(0, value);
        writeSetting(QStringLiteral("growth_steps"), _growthSteps);
    });

    connect(_btnGrow, &QPushButton::clicked, this, [this]() {
        qCInfo(lcSegWidget) << "Grow pressed" << segmentationGrowthMethodToString(_growthMethod)
                            << segmentationGrowthDirectionToString(_growthDirection)
                            << "steps" << _growthSteps;
        emit growSurfaceRequested(_growthMethod, _growthDirection, _growthSteps);
    });

    connect(_comboVolume, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0) {
            return;
        }
        const QVariant idData = _comboVolume->itemData(index);
        const QString volumeId = idData.toString();
        if (volumeId.isEmpty() || volumeId == _activeVolumeId) {
            return;
        }
        _activeVolumeId = volumeId;
        emit volumeSelectionChanged(volumeId);
    });

    connect(_spinHoleRadius, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        value = std::clamp(value, 1, 64);
        if (value == _holeSearchRadius) {
            return;
        }
        _holeSearchRadius = value;
        writeSetting(QStringLiteral("hole_search_radius"), _holeSearchRadius);
        emit holeSearchRadiusChanged(value);
    });

    connect(_spinHoleIterations, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        value = std::clamp(value, 1, 200);
        if (value == _holeSmoothIterations) {
            return;
        }
        _holeSmoothIterations = value;
        writeSetting(QStringLiteral("hole_smooth_iterations"), _holeSmoothIterations);
        emit holeSmoothIterationsChanged(value);
    });

    connect(_chkFillInvalidRegions, &QCheckBox::toggled, this, [this](bool checked) {
        if (checked == _fillInvalidRegions) {
            return;
        }
        _fillInvalidRegions = checked;
        writeSetting(QStringLiteral("fill_invalid_regions"), _fillInvalidRegions);
        emit fillInvalidRegionsChanged(checked);
    });

    connect(_chkHandlesAlwaysVisible, &QCheckBox::toggled, this, [this](bool checked) {
        if (checked == _handlesAlwaysVisible) {
            return;
        }
        _handlesAlwaysVisible = checked;
        writeSetting(QStringLiteral("handles_always_visible"), _handlesAlwaysVisible);
        if (_spinHandleDisplayDistance) {
            _spinHandleDisplayDistance->setEnabled(!checked && _editingEnabled);
        }
        emit handlesAlwaysVisibleChanged(checked);
    });

    connect(_spinHandleDisplayDistance, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        float distance = static_cast<float>(std::max(1.0, value));
        if (std::fabs(distance - _handleDisplayDistance) < 1e-4f) {
            return;
        }
        _handleDisplayDistance = distance;
        writeSetting(QStringLiteral("handle_display_distance"), _handleDisplayDistance);
        emit handleDisplayDistanceChanged(_handleDisplayDistance);
    });

    connect(_btnApply, &QPushButton::clicked, this, [this]() {
        emit applyRequested();
    });

    connect(_btnReset, &QPushButton::clicked, this, [this]() {
        emit resetRequested();
    });

    connect(_btnStopTools, &QPushButton::clicked, this, [this]() {
        emit stopToolsRequested();
    });

    connect(_btnMaskEdit, &QPushButton::toggled, this, [this](bool checked) {
        setMaskEditingActive(checked);
        emit maskEditingToggled(checked);
    });

    connect(_btnMaskApply, &QPushButton::clicked, this, [this]() {
        emit maskApplyRequested();
    });

    connect(_spinMaskSampling, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        setMaskSampling(value);
    });

    connect(_spinMaskRadius, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        setMaskBrushRadius(value);
    });
    setPendingChanges(false);
}

void SegmentationWidget::setEditingEnabled(bool enabled)
{
    if (_editingEnabled == enabled) {
        return;
    }

    _editingEnabled = enabled;
    const QSignalBlocker blocker(_chkEditing);
    _chkEditing->setChecked(enabled);
    if (!enabled) {
        _hasPendingChanges = false;
    }
    updateEditingUi();
}

void SegmentationWidget::setDownsample(int value)
{
    value = std::clamp(value, 2, 64);
    if (value == _downsample) {
        return;
    }
    _downsample = value;
    const QSignalBlocker blocker(_spinDownsample);
    _spinDownsample->setValue(value);
    writeSetting(QStringLiteral("downsample"), _downsample);
}

void SegmentationWidget::setRadius(float value)
{
    const int snapped = std::max(1, static_cast<int>(std::lround(value)));
    const float radius = static_cast<float>(snapped);
    if (std::fabs(radius - _radius) < 1e-4f) {
        return;
    }
    _radius = radius;
    const QSignalBlocker blocker(_spinRadius);
    _spinRadius->setValue(snapped);
    writeSetting(QStringLiteral("radius_steps"), snapped);
}

void SegmentationWidget::setSigma(float value)
{
    const float sigma = std::max(0.10f, value);
    if (std::fabs(sigma - _sigma) < 1e-4f) {
        return;
    }
    _sigma = sigma;
    const QSignalBlocker blocker(_spinSigma);
    _spinSigma->setValue(static_cast<double>(sigma));
    writeSetting(QStringLiteral("strength"), _sigma);
}

void SegmentationWidget::setInfluenceMode(SegmentationInfluenceMode mode)
{
    if (_influenceMode == mode) {
        return;
    }
    _influenceMode = mode;
    if (_comboInfluenceMode) {
        const QSignalBlocker blocker(_comboInfluenceMode);
        int modeIndex = _comboInfluenceMode->findData(static_cast<int>(_influenceMode));
        if (modeIndex >= 0) {
            _comboInfluenceMode->setCurrentIndex(modeIndex);
        }
    }
    if (_comboRowColMode) {
        _comboRowColMode->setEnabled(_editingEnabled && _influenceMode == SegmentationInfluenceMode::RowColumn);
    }
    writeSetting(QStringLiteral("influence_mode"), static_cast<int>(_influenceMode));
}

void SegmentationWidget::setSliceFadeDistance(float value)
{
    const float clamped = std::clamp(value, 0.1f, 500.0f);
    if (std::fabs(clamped - _sliceFadeDistance) < 1e-4f) {
        return;
    }
    _sliceFadeDistance = clamped;
    if (_spinSliceFadeDistance) {
        const QSignalBlocker blocker(_spinSliceFadeDistance);
        _spinSliceFadeDistance->setValue(static_cast<double>(_sliceFadeDistance));
    }
    writeSetting(QStringLiteral("slice_fade_distance"), _sliceFadeDistance);
}

void SegmentationWidget::setSliceDisplayMode(SegmentationSliceDisplayMode mode)
{
    if (_sliceDisplayMode == mode) {
        return;
    }
    _sliceDisplayMode = mode;
    if (_comboSliceDisplayMode) {
        const QSignalBlocker blocker(_comboSliceDisplayMode);
        int index = _comboSliceDisplayMode->findData(static_cast<int>(_sliceDisplayMode));
        if (index >= 0) {
            _comboSliceDisplayMode->setCurrentIndex(index);
        }
    }
    writeSetting(QStringLiteral("slice_display_mode"), static_cast<int>(_sliceDisplayMode));
}

void SegmentationWidget::setRowColMode(SegmentationRowColMode mode)
{
    if (_rowColMode == mode) {
        return;
    }
    _rowColMode = mode;
    if (_comboRowColMode) {
        const QSignalBlocker blocker(_comboRowColMode);
        int idx = _comboRowColMode->findData(static_cast<int>(_rowColMode));
        if (idx >= 0) {
            _comboRowColMode->setCurrentIndex(idx);
        }
        _comboRowColMode->setEnabled(_editingEnabled && _influenceMode == SegmentationInfluenceMode::RowColumn);
    }
    writeSetting(QStringLiteral("row_col_mode"), static_cast<int>(_rowColMode));
}

void SegmentationWidget::setHoleSearchRadius(int value)
{
    value = std::clamp(value, 1, 64);
    if (value == _holeSearchRadius) {
        return;
    }
    _holeSearchRadius = value;
    if (_spinHoleRadius) {
        const QSignalBlocker blocker(_spinHoleRadius);
        _spinHoleRadius->setValue(value);
    }
    writeSetting(QStringLiteral("hole_search_radius"), _holeSearchRadius);
}

void SegmentationWidget::setHoleSmoothIterations(int value)
{
    value = std::clamp(value, 1, 200);
    if (value == _holeSmoothIterations) {
        return;
    }
    _holeSmoothIterations = value;
    if (_spinHoleIterations) {
        const QSignalBlocker blocker(_spinHoleIterations);
        _spinHoleIterations->setValue(value);
    }
    writeSetting(QStringLiteral("hole_smooth_iterations"), _holeSmoothIterations);
}

void SegmentationWidget::setHandlesAlwaysVisible(bool value)
{
    if (value == _handlesAlwaysVisible) {
        return;
    }
    _handlesAlwaysVisible = value;
    if (_chkHandlesAlwaysVisible) {
        const QSignalBlocker blocker(_chkHandlesAlwaysVisible);
        _chkHandlesAlwaysVisible->setChecked(value);
    }
    if (_spinHandleDisplayDistance) {
        _spinHandleDisplayDistance->setEnabled(_editingEnabled && !_handlesAlwaysVisible);
    }
    if (_spinHighlightDistance) {
        _spinHighlightDistance->setEnabled(_editingEnabled);
    }
    writeSetting(QStringLiteral("handles_always_visible"), _handlesAlwaysVisible);
}

void SegmentationWidget::setHandleDisplayDistance(float value)
{
    const float clamped = std::clamp(value, 1.0f, 500.0f);
    if (std::fabs(clamped - _handleDisplayDistance) < 1e-4f) {
        return;
    }
    _handleDisplayDistance = clamped;
    if (_spinHandleDisplayDistance) {
        const QSignalBlocker blocker(_spinHandleDisplayDistance);
        _spinHandleDisplayDistance->setValue(static_cast<double>(clamped));
        _spinHandleDisplayDistance->setEnabled(_editingEnabled && !_handlesAlwaysVisible);
    }
    writeSetting(QStringLiteral("handle_display_distance"), _handleDisplayDistance);
}

void SegmentationWidget::setFillInvalidRegions(bool value)
{
    if (_fillInvalidRegions == value) {
        return;
    }
    _fillInvalidRegions = value;
    if (_chkFillInvalidRegions) {
        const QSignalBlocker blocker(_chkFillInvalidRegions);
        _chkFillInvalidRegions->setChecked(value);
    }
    writeSetting(QStringLiteral("fill_invalid_regions"), _fillInvalidRegions);
}

std::vector<SegmentationGrowthDirection> SegmentationWidget::allowedGrowthDirections() const
{
    std::vector<SegmentationGrowthDirection> selected;
    if (_growthDirectionMask & kGrowDirUpBit) {
        selected.push_back(SegmentationGrowthDirection::Up);
    }
    if (_growthDirectionMask & kGrowDirDownBit) {
        selected.push_back(SegmentationGrowthDirection::Down);
    }
    if (_growthDirectionMask & kGrowDirLeftBit) {
        selected.push_back(SegmentationGrowthDirection::Left);
    }
    if (_growthDirectionMask & kGrowDirRightBit) {
        selected.push_back(SegmentationGrowthDirection::Right);
    }

    if (selected.empty()) {
        selected = {
            SegmentationGrowthDirection::Up,
            SegmentationGrowthDirection::Down,
            SegmentationGrowthDirection::Left,
            SegmentationGrowthDirection::Right
        };
    }
    return selected;
}

void SegmentationWidget::setGrowthDirectionMask(int mask)
{
    mask = normalizeGrowthDirectionMask(mask);
    const bool changed = (mask != _growthDirectionMask);
    _growthDirectionMask = mask;
    applyGrowthDirectionMaskToUi();
    if (changed) {
        writeSetting(QStringLiteral("growth_direction_mask"), _growthDirectionMask);
    }
}

void SegmentationWidget::updateGrowthDirectionMaskFromUi(QCheckBox* changedCheckbox)
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
        applyGrowthDirectionMaskToUi();
        return;
    }

    setGrowthDirectionMask(mask);
}

void SegmentationWidget::applyGrowthDirectionMaskToUi()
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

int SegmentationWidget::normalizeGrowthDirectionMask(int mask)
{
    mask &= kGrowDirAllMask;
    if (mask == 0) {
        mask = kGrowDirAllMask;
    }
    return mask;
}

void SegmentationWidget::setHighlightDistance(float value)
{
    const float clamped = std::clamp(value, 0.5f, 500.0f);
    if (std::fabs(clamped - _highlightDistance) < 1e-4f) {
        return;
    }
    _highlightDistance = clamped;
    if (_spinHighlightDistance) {
        const QSignalBlocker blocker(_spinHighlightDistance);
        _spinHighlightDistance->setValue(static_cast<double>(_highlightDistance));
    }
    writeSetting(QStringLiteral("highlight_distance"), _highlightDistance);
}

void SegmentationWidget::setPendingChanges(bool pending)
{
    if (_hasPendingChanges == pending) {
        updateEditingUi();
        return;
    }
    _hasPendingChanges = pending;
    updateEditingUi();
}

void SegmentationWidget::updateEditingUi()
{
    if (!_editingStatus) {
        return;
    }

    if (_editingEnabled) {
        _editingStatus->setText(tr("Editing enabled"));
    } else {
        _editingStatus->setText(tr("Editing disabled"));
    }

    _spinDownsample->setEnabled(true);
    _spinRadius->setEnabled(_editingEnabled);
    _spinSigma->setEnabled(_editingEnabled);
    if (_comboInfluenceMode) {
        _comboInfluenceMode->setEnabled(_editingEnabled);
    }
    if (_comboRowColMode) {
        _comboRowColMode->setEnabled(_editingEnabled && _influenceMode == SegmentationInfluenceMode::RowColumn);
    }
    if (_groupSliceVisibility) {
        _groupSliceVisibility->setEnabled(_editingEnabled);
    }
    if (_spinHoleRadius) {
        _spinHoleRadius->setEnabled(_editingEnabled);
    }
    if (_spinHoleIterations) {
        _spinHoleIterations->setEnabled(_editingEnabled);
    }
    if (_chkFillInvalidRegions) {
        _chkFillInvalidRegions->setEnabled(_editingEnabled);
    }
    if (_chkHandlesAlwaysVisible) {
        _chkHandlesAlwaysVisible->setEnabled(true);
    }
    if (_spinHandleDisplayDistance) {
        _spinHandleDisplayDistance->setEnabled(_editingEnabled && !_handlesAlwaysVisible);
    }
    if (_spinHighlightDistance) {
        _spinHighlightDistance->setEnabled(_editingEnabled);
    }
    if (_comboGrowthMethod) {
        _comboGrowthMethod->setEnabled(_editingEnabled);
    }
    if (_comboGrowthDirection) {
        _comboGrowthDirection->setEnabled(_editingEnabled);
    }
    const bool growthDirCheckboxEnabled = _editingEnabled;
    if (_chkGrowthDirUp) {
        _chkGrowthDirUp->setEnabled(growthDirCheckboxEnabled);
    }
    if (_chkGrowthDirDown) {
        _chkGrowthDirDown->setEnabled(growthDirCheckboxEnabled);
    }
    if (_chkGrowthDirLeft) {
        _chkGrowthDirLeft->setEnabled(growthDirCheckboxEnabled);
    }
    if (_chkGrowthDirRight) {
        _chkGrowthDirRight->setEnabled(growthDirCheckboxEnabled);
    }
    if (_spinGrowthSteps) {
        _spinGrowthSteps->setEnabled(_editingEnabled);
    }
    if (_btnGrow) {
        _btnGrow->setEnabled(_editingEnabled);
    }
    if (_groupMasking) {
        _groupMasking->setEnabled(_editingEnabled);
    }
    if (_btnMaskEdit) {
        _btnMaskEdit->setEnabled(_editingEnabled);
    }
    if (_btnMaskApply) {
        _btnMaskApply->setEnabled(_editingEnabled && _maskEditingActive && _maskApplyEnabled);
    }
    if (_spinMaskSampling) {
        _spinMaskSampling->setEnabled(_editingEnabled);
    }
    if (_spinMaskRadius) {
        _spinMaskRadius->setEnabled(_editingEnabled);
    }
    if (_btnApply) {
        _btnApply->setEnabled(_editingEnabled && _hasPendingChanges);
    }
    if (_btnReset) {
        _btnReset->setEnabled(_editingEnabled && _hasPendingChanges);
    }

    updateGrowthModeUi();
    refreshCorrectionsUiState();
}

void SegmentationWidget::setMaskEditingActive(bool active)
{
    _maskEditingActive = active;
    if (!active) {
        _maskApplyEnabled = false;
    }
    if (_btnMaskEdit) {
        const QSignalBlocker blocker(_btnMaskEdit);
        _btnMaskEdit->setChecked(active);
        _btnMaskEdit->setText(active ? tr("Exit Mask Editing") : tr("Edit Mask"));
    }
    updateEditingUi();
}

void SegmentationWidget::setMaskApplyEnabled(bool enabled)
{
    _maskApplyEnabled = enabled;
    updateEditingUi();
}

void SegmentationWidget::setMaskSampling(int value)
{
    const int clamped = std::clamp(value, 1, 64);
    if (_maskSampling == clamped) {
        return;
    }
    _maskSampling = clamped;
    if (_spinMaskSampling) {
        const QSignalBlocker blocker(_spinMaskSampling);
        _spinMaskSampling->setValue(clamped);
    }
    writeSetting(QStringLiteral("mask_sampling"), _maskSampling);
    emit maskSamplingChanged(_maskSampling);
}

void SegmentationWidget::setMaskBrushRadius(int value)
{
    const int clamped = std::clamp(value, 1, 64);
    if (_maskBrushRadius == clamped) {
        return;
    }
    _maskBrushRadius = clamped;
    if (_spinMaskRadius) {
        const QSignalBlocker blocker(_spinMaskRadius);
        _spinMaskRadius->setValue(clamped);
    }
    writeSetting(QStringLiteral("mask_brush_radius"), _maskBrushRadius);
    emit maskBrushRadiusChanged(_maskBrushRadius);
}

void SegmentationWidget::restoreSettings()
{
    QSettings settings("VC.ini", QSettings::IniFormat);

    const int storedMaskSampling = settings.value(QStringLiteral("segmentation_edit/mask_sampling"), _maskSampling).toInt();
    setMaskSampling(std::clamp(storedMaskSampling, 1, 64));

    const int storedMaskRadius = settings.value(QStringLiteral("segmentation_edit/mask_brush_radius"), _maskBrushRadius).toInt();
    setMaskBrushRadius(std::clamp(storedMaskRadius, 1, 64));

    const int storedDownsample = settings.value(QStringLiteral("segmentation_edit/downsample"), _downsample).toInt();
    setDownsample(std::clamp(storedDownsample, 2, 64));

    const int storedRadius = settings.value(QStringLiteral("segmentation_edit/radius_steps"), static_cast<int>(std::lround(_radius))).toInt();
    setRadius(static_cast<float>(std::clamp(storedRadius, 1, 32)));

    const double storedStrength = settings.value(QStringLiteral("segmentation_edit/strength"), static_cast<double>(_sigma)).toDouble();
    const float clampedStrength = static_cast<float>(std::clamp(storedStrength, 0.10, 2.0));
    setSigma(clampedStrength);

    const int storedInfluence = settings.value(QStringLiteral("segmentation_edit/influence_mode"), static_cast<int>(_influenceMode)).toInt();
    const int clampedInfluence = std::clamp(storedInfluence,
                                            static_cast<int>(SegmentationInfluenceMode::GridChebyshev),
                                            static_cast<int>(SegmentationInfluenceMode::RowColumn));
    setInfluenceMode(static_cast<SegmentationInfluenceMode>(clampedInfluence));

    const double storedSliceFade = settings.value(QStringLiteral("segmentation_edit/slice_fade_distance"), static_cast<double>(_sliceFadeDistance)).toDouble();
    setSliceFadeDistance(static_cast<float>(std::clamp(storedSliceFade, 0.1, 500.0)));

    const int storedSliceMode = settings.value(QStringLiteral("segmentation_edit/slice_display_mode"), static_cast<int>(_sliceDisplayMode)).toInt();
    const int clampedSliceMode = std::clamp(storedSliceMode,
                                            static_cast<int>(SegmentationSliceDisplayMode::Fade),
                                            static_cast<int>(SegmentationSliceDisplayMode::Hide));
    setSliceDisplayMode(static_cast<SegmentationSliceDisplayMode>(clampedSliceMode));

    const int storedRowCol = settings.value(QStringLiteral("segmentation_edit/row_col_mode"), static_cast<int>(_rowColMode)).toInt();
    const int clampedRowCol = std::clamp(storedRowCol,
                                         static_cast<int>(SegmentationRowColMode::RowOnly),
                                         static_cast<int>(SegmentationRowColMode::Dynamic));
    setRowColMode(static_cast<SegmentationRowColMode>(clampedRowCol));

    const int storedHoleRadius = settings.value(QStringLiteral("segmentation_edit/hole_search_radius"), _holeSearchRadius).toInt();
    setHoleSearchRadius(std::clamp(storedHoleRadius, 1, 64));

    const int storedHoleIterations = settings.value(QStringLiteral("segmentation_edit/hole_smooth_iterations"), _holeSmoothIterations).toInt();
    setHoleSmoothIterations(std::clamp(storedHoleIterations, 1, 200));

    const bool storedFillInvalid = settings.value(QStringLiteral("segmentation_edit/fill_invalid_regions"), _fillInvalidRegions).toBool();
    setFillInvalidRegions(storedFillInvalid);

    const bool storedHandlesAlways = settings.value(QStringLiteral("segmentation_edit/handles_always_visible"), _handlesAlwaysVisible).toBool();
    setHandlesAlwaysVisible(storedHandlesAlways);

    const double storedHandleDistance = settings.value(QStringLiteral("segmentation_edit/handle_display_distance"), static_cast<double>(_handleDisplayDistance)).toDouble();
    setHandleDisplayDistance(static_cast<float>(std::clamp(storedHandleDistance, 1.0, 500.0)));

    const double storedHighlightDistance = settings.value(QStringLiteral("segmentation_edit/highlight_distance"), static_cast<double>(_highlightDistance)).toDouble();
    setHighlightDistance(static_cast<float>(std::clamp(storedHighlightDistance, 0.5, 500.0)));

    const int storedGrowthMethod = settings.value(QStringLiteral("segmentation_edit/growth_method"), static_cast<int>(_growthMethod)).toInt();
    if (storedGrowthMethod == static_cast<int>(SegmentationGrowthMethod::Corrections)) {
        _growthMethod = SegmentationGrowthMethod::Corrections;
    } else {
        _growthMethod = SegmentationGrowthMethod::Tracer;
    }
    if (_comboGrowthMethod) {
        const QSignalBlocker blocker(_comboGrowthMethod);
        int idx = _comboGrowthMethod->findData(static_cast<int>(_growthMethod));
        if (idx >= 0) {
            _comboGrowthMethod->setCurrentIndex(idx);
        }
    }

    const int storedGrowthDirection = settings.value(QStringLiteral("segmentation_edit/growth_direction"), static_cast<int>(_growthDirection)).toInt();
    _growthDirection = segmentationGrowthDirectionFromInt(storedGrowthDirection);
    if (_comboGrowthDirection) {
        const QSignalBlocker blocker(_comboGrowthDirection);
        int idx = _comboGrowthDirection->findData(static_cast<int>(_growthDirection));
        if (idx >= 0) {
            _comboGrowthDirection->setCurrentIndex(idx);
        }
    }

    const int storedGrowthDirectionMask = settings.value(QStringLiteral("segmentation_edit/growth_direction_mask"), _growthDirectionMask).toInt();
    setGrowthDirectionMask(storedGrowthDirectionMask);

    const int storedGrowthSteps = settings.value(QStringLiteral("segmentation_edit/growth_steps"), _growthSteps).toInt();
    _growthSteps = std::clamp(storedGrowthSteps, 0, 1024);
    if (_spinGrowthSteps) {
        const QSignalBlocker blocker(_spinGrowthSteps);
        _spinGrowthSteps->setValue(_growthSteps);
    }

    _correctionsZRangeEnabled = settings.value(QStringLiteral("segmentation_edit/corrections_z_range_enabled"), false).toBool();
    _correctionsZMin = settings.value(QStringLiteral("segmentation_edit/corrections_z_min"), 0).toInt();
    _correctionsZMax = settings.value(QStringLiteral("segmentation_edit/corrections_z_max"), _correctionsZMin).toInt();
    _correctionsZMin = std::max(0, _correctionsZMin);
    _correctionsZMax = std::max(_correctionsZMin, _correctionsZMax);

    if (_chkCorrectionsUseZRange) {
        const QSignalBlocker blocker(_chkCorrectionsUseZRange);
        _chkCorrectionsUseZRange->setChecked(_correctionsZRangeEnabled);
    }
    if (_spinCorrectionsZMin) {
        const QSignalBlocker blocker(_spinCorrectionsZMin);
        _spinCorrectionsZMin->setValue(_correctionsZMin);
        _spinCorrectionsZMin->setEnabled(_correctionsZRangeEnabled);
    }
    if (_spinCorrectionsZMax) {
        const QSignalBlocker blocker(_spinCorrectionsZMax);
        _spinCorrectionsZMax->setValue(_correctionsZMax);
        _spinCorrectionsZMax->setEnabled(_correctionsZRangeEnabled);
    }

    updateGrowthModeUi();
}

void SegmentationWidget::writeSetting(const QString& key, const QVariant& value)
{
    QSettings settings("VC.ini", QSettings::IniFormat);
    settings.setValue(QStringLiteral("segmentation_edit/%1").arg(key), value);
}

void SegmentationWidget::setCorrectionsEnabled(bool enabled)
{
    if (_correctionsEnabled == enabled) {
        return;
    }
    _correctionsEnabled = enabled;
    refreshCorrectionsUiState();
}

void SegmentationWidget::setCorrectionsAnnotateChecked(bool enabled)
{
    if (_chkCorrectionsAnnotate) {
        const QSignalBlocker blocker(_chkCorrectionsAnnotate);
        _chkCorrectionsAnnotate->setChecked(enabled);
    }
}

void SegmentationWidget::setCorrectionCollections(const QVector<QPair<uint64_t, QString>>& collections,
                                                  std::optional<uint64_t> activeId)
{
    _activeCorrectionId = activeId;
    if (!_comboCorrections) {
        return;
    }

    const QSignalBlocker blocker(_comboCorrections);
    _comboCorrections->clear();
    _comboCorrections->addItem(tr("None"), static_cast<qulonglong>(0));

    int activeIndex = 0;
    int idx = 1;
    for (const auto& entry : collections) {
        _comboCorrections->addItem(entry.second, static_cast<qulonglong>(entry.first));
        if (activeId && entry.first == *activeId) {
            activeIndex = idx;
        }
        ++idx;
    }

    if (activeIndex >= 0 && activeIndex < _comboCorrections->count()) {
        _comboCorrections->setCurrentIndex(activeIndex);
    } else {
        _comboCorrections->setCurrentIndex(0);
        if (activeId) {
            _activeCorrectionId.reset();
        }
    }

    refreshCorrectionsUiState();
}

void SegmentationWidget::refreshCorrectionsUiState()
{
    const bool allow = _editingEnabled && _correctionsEnabled;
    const bool hasCollections = _comboCorrections && _comboCorrections->count() > 1;

    if (_groupCorrections) {
        _groupCorrections->setEnabled(_editingEnabled);
    }

    if (_comboCorrections) {
        _comboCorrections->setEnabled(allow && hasCollections);
    }

    if (_btnCorrectionsNew) {
        _btnCorrectionsNew->setEnabled(allow);
    }

    if (_chkCorrectionsAnnotate) {
        const bool canAnnotate = allow && hasCollections;
        if (!canAnnotate) {
            const QSignalBlocker blocker(_chkCorrectionsAnnotate);
            _chkCorrectionsAnnotate->setChecked(false);
        }
        _chkCorrectionsAnnotate->setEnabled(canAnnotate);
    }

    if (_chkCorrectionsUseZRange) {
        _chkCorrectionsUseZRange->setEnabled(allow);
    }

    if (_spinCorrectionsZMin) {
        _spinCorrectionsZMin->setEnabled(allow && _correctionsZRangeEnabled);
    }

    if (_spinCorrectionsZMax) {
        _spinCorrectionsZMax->setEnabled(allow && _correctionsZRangeEnabled);
    }

}

void SegmentationWidget::updateGrowthModeUi()
{
    const bool showDetailedControls = _growthMethod != SegmentationGrowthMethod::Corrections;

    if (_groupSampling) {
        _groupSampling->setVisible(showDetailedControls);
    }
    if (_groupInfluence) {
        _groupInfluence->setVisible(showDetailedControls);
    }
    if (_groupHole) {
        _groupHole->setVisible(showDetailedControls);
    }
    if (_groupHandleDisplay) {
        _groupHandleDisplay->setVisible(showDetailedControls);
    }
    if (_groupSliceVisibility) {
        _groupSliceVisibility->setVisible(showDetailedControls);
    }
}

void SegmentationWidget::setHandlesLocked(bool locked)
{
    if (_handlesLocked == locked) {
        return;
    }
    _handlesLocked = locked;
}

void SegmentationWidget::setNormalGridAvailable(bool available)
{
    _normalGridAvailable = available;

    if (!_normalGridStatusWidget) {
        return;
    }

    if (available) {
        if (_normalGridStatusIcon) {
            const auto icon = style()->standardIcon(QStyle::SP_DialogApplyButton);
            _normalGridStatusIcon->setPixmap(icon.pixmap(16, 16));
        }
        if (_normalGridStatusText) {
            _normalGridStatusText->setText(tr("Normal grid data detected [OK]"));
            _normalGridStatusText->setStyleSheet(QStringLiteral("color: #2b8a3e;"));
        }
    } else {
        if (_normalGridStatusIcon) {
            const auto icon = style()->standardIcon(QStyle::SP_MessageBoxCritical);
            _normalGridStatusIcon->setPixmap(icon.pixmap(16, 16));
        }
        if (_normalGridStatusText) {
            _normalGridStatusText->setText(tr("Normal grid not found at {volpkg}/normal_grids"));
            _normalGridStatusText->setStyleSheet(QStringLiteral("color: #c92a2a;"));
        }
    }

    _normalGridStatusWidget->setVisible(true);
}

void SegmentationWidget::setAvailableVolumes(const QVector<QPair<QString, QString>>& volumes,
                                             const QString& activeId)
{
    _volumeEntries = volumes;
    _activeVolumeId = activeId;
    if (!_comboVolume) {
        return;
    }

    const QSignalBlocker blocker(_comboVolume);
    _comboVolume->clear();

    for (const auto& entry : volumes) {
        _comboVolume->addItem(entry.second, entry.first);
    }

    int index = _comboVolume->findData(_activeVolumeId);
    if (index < 0 && _comboVolume->count() > 0) {
        index = 0;
        _activeVolumeId = _comboVolume->itemData(index).toString();
    }

    if (index >= 0) {
        _comboVolume->setCurrentIndex(index);
    }

    _comboVolume->setEnabled(_comboVolume->count() > 0);
}

void SegmentationWidget::setActiveVolume(const QString& volumeId)
{
    if (_activeVolumeId == volumeId) {
        return;
    }
    _activeVolumeId = volumeId;
    if (!_comboVolume) {
        return;
    }
    const QSignalBlocker blocker(_comboVolume);
    int index = _comboVolume->findData(volumeId);
    if (index >= 0) {
        _comboVolume->setCurrentIndex(index);
    }
}

void SegmentationWidget::setNormalGridPathHint(const QString& hint)
{
    if (_normalGridStatusWidget) {
        _normalGridStatusWidget->setToolTip(hint);
    }
}
