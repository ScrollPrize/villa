#include "SegmentationManualAddPanel.hpp"

#include "VCSettings.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QVariant>
#include <QVBoxLayout>

#include <algorithm>

SegmentationManualAddPanel::SegmentationManualAddPanel(const QString& settingsGroup, QWidget* parent)
    : QWidget(parent)
    , _settingsGroup(settingsGroup)
{
    auto* outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);

    auto* group = new QGroupBox(tr("Manual Fill"), this);
    _group = group;
    auto* layout = new QVBoxLayout(group);
    auto* form = new QFormLayout();
    _form = form;

    _spinMaxPreviewSpan = new QSpinBox(group);
    _spinMaxPreviewSpan->setRange(4, 4096);
    _spinMaxPreviewSpan->setValue(256);
    form->addRow(tr("Max preview span"), _spinMaxPreviewSpan);

    _spinBoundaryBand = new QSpinBox(group);
    _spinBoundaryBand->setRange(1, 20);
    _spinBoundaryBand->setValue(2);
    form->addRow(tr("Boundary band"), _spinBoundaryBand);

    _spinRegularization = new QDoubleSpinBox(group);
    _spinRegularization->setRange(0.0, 1.0);
    _spinRegularization->setDecimals(6);
    _spinRegularization->setSingleStep(0.0001);
    _spinRegularization->setValue(0.0001);
    form->addRow(tr("RBF regularization"), _spinRegularization);

    _spinSampleCap = new QSpinBox(group);
    _spinSampleCap->setRange(16, 10000);
    _spinSampleCap->setValue(512);
    form->addRow(tr("RBF sample cap"), _spinSampleCap);

    _spinPreviewThrottle = new QSpinBox(group);
    _spinPreviewThrottle->setRange(0, 500);
    _spinPreviewThrottle->setSuffix(tr(" ms"));
    _spinPreviewThrottle->setValue(50);
    form->addRow(tr("Preview throttle"), _spinPreviewThrottle);

    _spinTintOpacity = new QSpinBox(group);
    _spinTintOpacity->setRange(0, 100);
    _spinTintOpacity->setSuffix(tr("%"));
    _spinTintOpacity->setValue(45);
    form->addRow(tr("Tint opacity"), _spinTintOpacity);

    _spinPlaneConstraintRadius = new QDoubleSpinBox(group);
    _spinPlaneConstraintRadius->setRange(0.5, 100.0);
    _spinPlaneConstraintRadius->setSingleStep(0.5);
    _spinPlaneConstraintRadius->setValue(30.0);
    form->addRow(tr("Plane constraint radius"), _spinPlaneConstraintRadius);

    _spinPlaneConstraintReplacementRadius = new QDoubleSpinBox(group);
    _spinPlaneConstraintReplacementRadius->setRange(0.0, 100.0);
    _spinPlaneConstraintReplacementRadius->setSingleStep(0.5);
    _spinPlaneConstraintReplacementRadius->setValue(16.0);
    form->addRow(tr("Constraint replace radius"), _spinPlaneConstraintReplacementRadius);

    _comboLinePreviewMode = new QComboBox(group);
    _comboLinePreviewMode->addItem(tr("Cross"), static_cast<int>(ManualAddTool::LinePreviewMode::Cross));
    _comboLinePreviewMode->addItem(tr("Cross-fill"), static_cast<int>(ManualAddTool::LinePreviewMode::CrossFill));
    _comboLinePreviewMode->addItem(tr("Vertical only"), static_cast<int>(ManualAddTool::LinePreviewMode::VerticalOnly));
    _comboLinePreviewMode->addItem(tr("Horizontal only"), static_cast<int>(ManualAddTool::LinePreviewMode::HorizontalOnly));
    form->addRow(tr("Yellow line"), _comboLinePreviewMode);

    _comboInterpolationMode = new QComboBox(group);
    _comboInterpolationMode->addItem(tr("Thin-plate spline"), static_cast<int>(ManualAddTool::InterpolationMode::ThinPlateSpline));
    _comboInterpolationMode->addItem(tr("Tracer inside fill"), static_cast<int>(ManualAddTool::InterpolationMode::TracerRestrictedToFill));
    form->addRow(tr("Fill method"), _comboInterpolationMode);

    layout->addLayout(form);

    _chkIncludeTouchedValidBorder = new QCheckBox(tr("Include touched valid border in fit"), group);
    _chkIncludeTouchedValidBorder->setChecked(true);
    layout->addWidget(_chkIncludeTouchedValidBorder);

    _chkAllowBoundarySmoothing = new QCheckBox(tr("Allow boundary smoothing"), group);
    _chkAllowBoundarySmoothing->setChecked(false);
    layout->addWidget(_chkAllowBoundarySmoothing);

    _growControls = new QWidget(group);
    auto* growLayout = new QVBoxLayout(_growControls);
    growLayout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(_growControls);

    _chkFollowSheet = new QCheckBox(tr("Extensions follow the sheet (CT)"), group);
    _chkFollowSheet->setChecked(true);
    _chkFollowSheet->setToolTip(tr("Walk each new extension along the papyrus using the CT structure tensor "
                                   "(vc_reader sheet normals) instead of continuing it straight. "
                                   "Your Shift+clicks still correct it."));
    growLayout->addWidget(_chkFollowSheet);
    auto* walkForm = new QFormLayout();
    _spinGrowWorkers = new QSpinBox(group);
    _spinGrowWorkers->setObjectName(QStringLiteral("manualGrowWorkers"));
    _spinGrowWorkers->setRange(1, 256);
    _spinGrowWorkers->setValue(8);
    _spinGrowWorkers->setToolTip(tr("Maximum parallel growth workers, limited by available hardware threads "
                                    "and growth lines. Changes apply to the next extension."));
    walkForm->addRow(tr("Workers"), _spinGrowWorkers);
    _spinNormalStep = new QDoubleSpinBox(group);
    _spinNormalStep->setRange(0, 10000);
    _spinNormalStep->setDecimals(2);
    _spinNormalStep->setSingleStep(0.25);
    _spinNormalStep->setSpecialValueText(tr("One step per vertex"));
    _spinNormalStep->setSuffix(tr(" voxels"));
    _spinNormalStep->setToolTip(tr("Maximum internal step length; mesh vertex spacing stays unchanged. "
                                   "Smaller steps take longer. Changes apply to the next extension."));
    walkForm->addRow(tr("Normal substep"), _spinNormalStep);
    _comboWalkMode = new QComboBox(group);
    _comboWalkMode->addItem(tr("Projection"));
    _comboWalkMode->addItem(tr("Streamline"));
    _comboWalkMode->setToolTip(tr("Streamline transports the tangent along the normal field. "
                                  "Changes apply to the next extension."));
    walkForm->addRow(tr("Sheet walk"), _comboWalkMode);
    _comboIntegrator = new QComboBox(group);
    _comboIntegrator->addItem(tr("Midpoint (RK2)"));
    _comboIntegrator->addItem(tr("Heun (RK2)"));
    _comboIntegrator->addItem(tr("RK4"));
    _comboIntegrator->setEnabled(false);
    walkForm->addRow(tr("Streamline method"), _comboIntegrator);
    connect(_comboWalkMode, &QComboBox::currentIndexChanged, this, [this](int mode) {
        _comboIntegrator->setEnabled(mode == 1);
    });
    growLayout->addLayout(walkForm);

    auto* extendRow = new QHBoxLayout();
    extendRow->addWidget(new QLabel(tr("Extend edge"), group));
    _spinExtendCells = new QSpinBox(group);
    _spinExtendCells->setRange(1, 4096);
    _spinExtendCells->setValue(16);
    _spinExtendCells->setSuffix(tr(" vertices"));
    _spinExtendCells->setToolTip(tr("How far an edge extension reaches past the last valid vertex of each line."));
    extendRow->addWidget(_spinExtendCells);
    auto makeExtendButton = [&](const QString& label, const QString& tip) {
        auto* button = new QPushButton(label, group);
        button->setToolTip(tip);
        extendRow->addWidget(button);
        return button;
    };
    const QString extendTip = tr("Continue the surface past its %1 edge along each line's own direction. "
                                 "Shift+click the sheet in a slice view to bend the preview onto it, "
                                 "Shift+right-click to remove a click, then Apply or Cancel. "
                                 "Pressing it again applies the pending strip and extends on from the new edge. "
                                 "Keys 1 to 4 do the same in Manual Grow, and Shift+drag along an edge "
                                 "in the segment view extends only that stretch.");
    _btnExtendUp = makeExtendButton(tr("Up"), extendTip.arg(tr("top")));
    _btnExtendDown = makeExtendButton(tr("Down"), extendTip.arg(tr("bottom")));
    _btnExtendLeft = makeExtendButton(tr("Left"), extendTip.arg(tr("left")));
    _btnExtendRight = makeExtendButton(tr("Right"), extendTip.arg(tr("right")));
    extendRow->addStretch(1);
    growLayout->addLayout(extendRow);

    auto* buttons = new QHBoxLayout();
    _btnClearPending = new QPushButton(tr("Clear Pending"), group);
    _btnRecompute = new QPushButton(tr("Recompute"), group);
    _btnApplyExit = new QPushButton(tr("Apply && Exit"), group);
    _btnCancel = new QPushButton(tr("Cancel"), group);
    buttons->addWidget(_btnClearPending);
    buttons->addWidget(_btnRecompute);
    buttons->addWidget(_btnApplyExit);
    buttons->addWidget(_btnCancel);
    buttons->addStretch(1);
    layout->addLayout(buttons);

    outer->addWidget(group);

    auto persist = [this]() {
        persistFromUi();
        emit configChanged();
    };
    for (auto* spin : {_spinMaxPreviewSpan, _spinBoundaryBand, _spinSampleCap, _spinPreviewThrottle, _spinTintOpacity}) {
        connect(spin, QOverload<int>::of(&QSpinBox::valueChanged), this, persist);
    }
    connect(_spinRegularization, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, persist);
    connect(_spinPlaneConstraintRadius, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, persist);
    connect(_spinPlaneConstraintReplacementRadius, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, persist);
    connect(_comboLinePreviewMode, QOverload<int>::of(&QComboBox::currentIndexChanged), this, persist);
    connect(_comboInterpolationMode, QOverload<int>::of(&QComboBox::currentIndexChanged), this, persist);
    connect(_chkIncludeTouchedValidBorder, &QCheckBox::toggled, this, persist);
    connect(_chkAllowBoundarySmoothing, &QCheckBox::toggled, this, persist);
    connect(_chkFollowSheet, &QCheckBox::toggled, this, persist);
    connect(_spinNormalStep, &QDoubleSpinBox::valueChanged, this, persist);
    connect(_comboWalkMode, &QComboBox::currentIndexChanged, this, persist);
    connect(_comboIntegrator, &QComboBox::currentIndexChanged, this, persist);
    connect(_btnClearPending, &QPushButton::clicked, this, &SegmentationManualAddPanel::clearPendingRequested);
    connect(_btnRecompute, &QPushButton::clicked, this, &SegmentationManualAddPanel::recomputeRequested);
    connect(_btnApplyExit, &QPushButton::clicked, this, &SegmentationManualAddPanel::applyExitRequested);
    connect(_btnCancel, &QPushButton::clicked, this, &SegmentationManualAddPanel::cancelRequested);
    connect(_spinExtendCells, QOverload<int>::of(&QSpinBox::valueChanged), this, [this]() { persistFromUi(); });
    // Worker count affects the next job; it must not recompute pending geometry.
    connect(_spinGrowWorkers, QOverload<int>::of(&QSpinBox::valueChanged), this, [this]() { persistFromUi(); });
    const std::pair<QPushButton*, ManualAddTool::ExtendSide> extendButtons[] = {
        {_btnExtendUp, ManualAddTool::ExtendSide::Up},
        {_btnExtendDown, ManualAddTool::ExtendSide::Down},
        {_btnExtendLeft, ManualAddTool::ExtendSide::Left},
        {_btnExtendRight, ManualAddTool::ExtendSide::Right},
    };
    for (const auto& [button, side] : extendButtons) {
        connect(button, &QPushButton::clicked, this, [this, side = side]() {
            emit extendRequested(static_cast<int>(side), _spinExtendCells->value());
        });
    }
}

void SegmentationManualAddPanel::setGrowMode(bool grow)
{
    _group->setTitle(grow ? tr("Manual Grow") : tr("Manual Fill"));
    _growControls->setVisible(grow);
    for (QWidget* field : {static_cast<QWidget*>(_spinMaxPreviewSpan),
                           static_cast<QWidget*>(_spinPreviewThrottle),
                           static_cast<QWidget*>(_comboLinePreviewMode),
                           static_cast<QWidget*>(_comboInterpolationMode)}) {
        _form->setRowVisible(field, !grow);
    }
}

int SegmentationManualAddPanel::growWorkers() const
{
    return _spinGrowWorkers->value();
}

int SegmentationManualAddPanel::extendCells() const
{
    return _spinExtendCells->value();
}

ManualAddTool::Config SegmentationManualAddPanel::config() const
{
    ManualAddTool::Config cfg;
    cfg.maxPreviewSpan = _spinMaxPreviewSpan->value();
    cfg.boundaryBand = _spinBoundaryBand->value();
    cfg.regularization = _spinRegularization->value();
    cfg.sampleCap = _spinSampleCap->value();
    cfg.previewThrottleMs = _spinPreviewThrottle->value();
    cfg.tintOpacity = static_cast<float>(_spinTintOpacity->value()) / 100.0f;
    cfg.planeConstraintRadius = _spinPlaneConstraintRadius->value();
    cfg.planeConstraintReplacementRadius = _spinPlaneConstraintReplacementRadius->value();
    cfg.linePreviewMode = static_cast<ManualAddTool::LinePreviewMode>(_comboLinePreviewMode->currentData().toInt());
    cfg.interpolationMode = static_cast<ManualAddTool::InterpolationMode>(_comboInterpolationMode->currentData().toInt());
    cfg.includeTouchedValidBorder = _chkIncludeTouchedValidBorder->isChecked();
    cfg.allowBoundarySmoothing = _chkAllowBoundarySmoothing->isChecked();
    cfg.followSheet = _chkFollowSheet->isChecked();
    cfg.sheetWalk.maxStep = _spinNormalStep->value();
    cfg.sheetWalk.mode = static_cast<ManualAddTool::SheetWalkMode>(_comboWalkMode->currentIndex());
    cfg.sheetWalk.integrator = static_cast<ManualAddTool::SheetIntegrator>(_comboIntegrator->currentIndex());
    return cfg;
}

ManualAddTool::LinePreviewMode SegmentationManualAddPanel::cycleLinePreviewMode()
{
    const int nextIndex = (_comboLinePreviewMode->currentIndex() + 1) % _comboLinePreviewMode->count();
    _comboLinePreviewMode->setCurrentIndex(nextIndex);
    return static_cast<ManualAddTool::LinePreviewMode>(_comboLinePreviewMode->currentData().toInt());
}

ManualAddTool::LinePreviewMode SegmentationManualAddPanel::setLinePreviewMode(ManualAddTool::LinePreviewMode mode)
{
    const int index = _comboLinePreviewMode->findData(static_cast<int>(mode));
    if (index >= 0) {
        // setCurrentIndex fires currentIndexChanged -> persist -> configChanged,
        // matching a human combo edit. Assigning the same index is a no-op, so
        // force the persist path when the mode is already selected.
        if (index == _comboLinePreviewMode->currentIndex()) {
            persistFromUi();
            emit configChanged();
        } else {
            _comboLinePreviewMode->setCurrentIndex(index);
        }
    }
    return static_cast<ManualAddTool::LinePreviewMode>(_comboLinePreviewMode->currentData().toInt());
}

ManualAddTool::InterpolationMode SegmentationManualAddPanel::setInterpolationMode(ManualAddTool::InterpolationMode mode)
{
    const int index = _comboInterpolationMode->findData(static_cast<int>(mode));
    if (index >= 0) {
        if (index == _comboInterpolationMode->currentIndex()) {
            persistFromUi();
            emit configChanged();
        } else {
            _comboInterpolationMode->setCurrentIndex(index);
        }
    }
    return static_cast<ManualAddTool::InterpolationMode>(_comboInterpolationMode->currentData().toInt());
}

void SegmentationManualAddPanel::restoreSettings(QSettings& settings)
{
    _restoringSettings = true;
    auto setSpin = [](auto* spin, auto value) {
        const QSignalBlocker blocker(spin);
        spin->setValue(value);
    };
    setSpin(_spinMaxPreviewSpan, settings.value(QStringLiteral("manual_add_max_preview_span"), 256).toInt());
    setSpin(_spinBoundaryBand, settings.value(QStringLiteral("manual_add_boundary_band"), 2).toInt());
    setSpin(_spinRegularization, settings.value(QStringLiteral("manual_add_regularization"), 0.0001).toDouble());
    setSpin(_spinSampleCap, settings.value(QStringLiteral("manual_add_sample_cap"), 512).toInt());
    setSpin(_spinPreviewThrottle, settings.value(QStringLiteral("manual_add_preview_throttle_ms"), 50).toInt());
    setSpin(_spinGrowWorkers, settings.value(QStringLiteral("manual_grow_workers"), 8).toInt());
    setSpin(_spinExtendCells, settings.value(QStringLiteral("manual_add_extend_cells"), 16).toInt());
    setSpin(_spinTintOpacity, settings.value(QStringLiteral("manual_add_tint_opacity_percent"), 45).toInt());
    setSpin(_spinPlaneConstraintRadius, settings.value(QStringLiteral("manual_add_plane_constraint_radius"), 30.0).toDouble());
    setSpin(_spinPlaneConstraintReplacementRadius, settings.value(QStringLiteral("manual_add_plane_constraint_replacement_radius"), 16.0).toDouble());
    {
        const QSignalBlocker blocker(_comboLinePreviewMode);
        const int mode = settings.value(QStringLiteral("manual_add_line_preview_mode"),
                                        static_cast<int>(ManualAddTool::LinePreviewMode::Cross)).toInt();
        const int index = std::max(0, _comboLinePreviewMode->findData(mode));
        _comboLinePreviewMode->setCurrentIndex(index);
    }
    {
        const QSignalBlocker blocker(_comboInterpolationMode);
        const int mode = settings.value(QStringLiteral("manual_add_interpolation_mode"),
                                        static_cast<int>(ManualAddTool::InterpolationMode::ThinPlateSpline)).toInt();
        const int index = std::max(0, _comboInterpolationMode->findData(mode));
        _comboInterpolationMode->setCurrentIndex(index);
    }
    {
        const QSignalBlocker blocker(_chkIncludeTouchedValidBorder);
        _chkIncludeTouchedValidBorder->setChecked(settings.value(QStringLiteral("manual_add_include_touched_valid_border"), true).toBool());
    }
    {
        const QSignalBlocker blocker(_chkAllowBoundarySmoothing);
        _chkAllowBoundarySmoothing->setChecked(settings.value(QStringLiteral("manual_add_allow_boundary_smoothing"), false).toBool());
    }
    {
        const QSignalBlocker blocker(_chkFollowSheet);
        _chkFollowSheet->setChecked(settings.value(QStringLiteral("manual_add_follow_sheet"), true).toBool());
    }
    setSpin(_spinNormalStep, settings.value(QStringLiteral("manual_add_normal_step"), 0.0).toDouble());
    {
        const QSignalBlocker modeBlocker(_comboWalkMode), methodBlocker(_comboIntegrator);
        _comboWalkMode->setCurrentIndex(std::clamp(settings.value(QStringLiteral("manual_add_walk_mode"), 0).toInt(), 0, 1));
        _comboIntegrator->setCurrentIndex(std::clamp(settings.value(QStringLiteral("manual_add_integrator"), 0).toInt(), 0, 2));
        _comboIntegrator->setEnabled(_comboWalkMode->currentIndex() == 1);
    }
    _restoringSettings = false;
}

void SegmentationManualAddPanel::syncUiState(bool editingEnabled, bool manualAddActive)
{
    setEnabled(editingEnabled);
    if (_btnRecompute) {
        _btnRecompute->setEnabled(editingEnabled && manualAddActive);
    }
    if (_btnApplyExit) {
        _btnApplyExit->setEnabled(editingEnabled && manualAddActive);
    }
    if (_btnCancel) {
        _btnCancel->setEnabled(editingEnabled && manualAddActive);
    }
    if (_btnClearPending) {
        _btnClearPending->setEnabled(editingEnabled && manualAddActive);
    }
    // Extending enters Manual Add on its own, so it only needs editing.
    for (auto* button : {_btnExtendUp, _btnExtendDown, _btnExtendLeft, _btnExtendRight}) {
        if (button) {
            button->setEnabled(editingEnabled);
        }
    }
}

void SegmentationManualAddPanel::writeSetting(const QString& key, const QVariant& value)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(_settingsGroup);
    settings.setValue(key, value);
    settings.endGroup();
}

void SegmentationManualAddPanel::persistFromUi()
{
    if (_restoringSettings) {
        return;
    }
    writeSetting(QStringLiteral("manual_add_max_preview_span"), _spinMaxPreviewSpan->value());
    writeSetting(QStringLiteral("manual_add_boundary_band"), _spinBoundaryBand->value());
    writeSetting(QStringLiteral("manual_add_regularization"), _spinRegularization->value());
    writeSetting(QStringLiteral("manual_add_sample_cap"), _spinSampleCap->value());
    writeSetting(QStringLiteral("manual_add_preview_throttle_ms"), _spinPreviewThrottle->value());
    writeSetting(QStringLiteral("manual_grow_workers"), _spinGrowWorkers->value());
    writeSetting(QStringLiteral("manual_add_extend_cells"), _spinExtendCells->value());
    writeSetting(QStringLiteral("manual_add_tint_opacity_percent"), _spinTintOpacity->value());
    writeSetting(QStringLiteral("manual_add_plane_constraint_radius"), _spinPlaneConstraintRadius->value());
    writeSetting(QStringLiteral("manual_add_plane_constraint_replacement_radius"), _spinPlaneConstraintReplacementRadius->value());
    writeSetting(QStringLiteral("manual_add_line_preview_mode"), _comboLinePreviewMode->currentData().toInt());
    writeSetting(QStringLiteral("manual_add_interpolation_mode"), _comboInterpolationMode->currentData().toInt());
    writeSetting(QStringLiteral("manual_add_include_touched_valid_border"), _chkIncludeTouchedValidBorder->isChecked());
    writeSetting(QStringLiteral("manual_add_allow_boundary_smoothing"), _chkAllowBoundarySmoothing->isChecked());
    writeSetting(QStringLiteral("manual_add_follow_sheet"), _chkFollowSheet->isChecked());
    writeSetting(QStringLiteral("manual_add_normal_step"), _spinNormalStep->value());
    writeSetting(QStringLiteral("manual_add_walk_mode"), _comboWalkMode->currentIndex());
    writeSetting(QStringLiteral("manual_add_integrator"), _comboIntegrator->currentIndex());
}
