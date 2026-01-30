#include "SegmentationGrowthPanel.hpp"

#include "elements/VolumeSelector.hpp"
#include "segmentation/SegmentationGrowth.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>

SegmentationGrowthPanel::SegmentationGrowthPanel(bool growthKeybindsEnabled, QWidget* parent)
    : QWidget(parent)
{
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
    _comboGrowthMethod->addItem(tr("Extrapolation"), static_cast<int>(SegmentationGrowthMethod::Extrapolation));
    _comboGrowthMethod->setToolTip(tr("Select the growth algorithm:\n"
                                      "- Tracer: Neural-guided growth using volume data\n"
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
    _spinGrowthSteps->setRange(0, 1024);
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
    _chkGrowthKeybindsEnabled->setChecked(growthKeybindsEnabled);
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
}
