#include "SegmentationManualAddPanel.hpp"

#include "VCSettings.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
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

    auto* group = new QGroupBox(tr("Manual Add"), this);
    auto* layout = new QVBoxLayout(group);
    auto* form = new QFormLayout();

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
    connect(_btnClearPending, &QPushButton::clicked, this, &SegmentationManualAddPanel::clearPendingRequested);
    connect(_btnRecompute, &QPushButton::clicked, this, &SegmentationManualAddPanel::recomputeRequested);
    connect(_btnApplyExit, &QPushButton::clicked, this, &SegmentationManualAddPanel::applyExitRequested);
    connect(_btnCancel, &QPushButton::clicked, this, &SegmentationManualAddPanel::cancelRequested);
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
    return cfg;
}

ManualAddTool::LinePreviewMode SegmentationManualAddPanel::cycleLinePreviewMode()
{
    const int nextIndex = (_comboLinePreviewMode->currentIndex() + 1) % _comboLinePreviewMode->count();
    _comboLinePreviewMode->setCurrentIndex(nextIndex);
    return static_cast<ManualAddTool::LinePreviewMode>(_comboLinePreviewMode->currentData().toInt());
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
    writeSetting(QStringLiteral("manual_add_tint_opacity_percent"), _spinTintOpacity->value());
    writeSetting(QStringLiteral("manual_add_plane_constraint_radius"), _spinPlaneConstraintRadius->value());
    writeSetting(QStringLiteral("manual_add_plane_constraint_replacement_radius"), _spinPlaneConstraintReplacementRadius->value());
    writeSetting(QStringLiteral("manual_add_line_preview_mode"), _comboLinePreviewMode->currentData().toInt());
    writeSetting(QStringLiteral("manual_add_interpolation_mode"), _comboInterpolationMode->currentData().toInt());
    writeSetting(QStringLiteral("manual_add_include_touched_valid_border"), _chkIncludeTouchedValidBorder->isChecked());
    writeSetting(QStringLiteral("manual_add_allow_boundary_smoothing"), _chkAllowBoundarySmoothing->isChecked());
}
