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

#include <algorithm>

#include <cmath>

SegmentationWidget::SegmentationWidget(QWidget* parent)
    : QWidget(parent)
    , _chkEditing(nullptr)
    , _editingStatus(nullptr)
    , _spinDownsample(nullptr)
    , _spinRadius(nullptr)
    , _spinSigma(nullptr)
    , _spinHoleRadius(nullptr)
    , _spinHoleIterations(nullptr)
    , _btnApply(nullptr)
    , _btnReset(nullptr)
    , _btnStopTools(nullptr)
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

    // Parameter controls
    auto* samplingGroup = new QGroupBox(tr("Sampling"), this);
    auto* samplingLayout = new QVBoxLayout(samplingGroup);

    auto* downsampleLayout = new QHBoxLayout();
    auto* downsampleLabel = new QLabel(tr("Downsample factor:"), samplingGroup);
    _spinDownsample = new QSpinBox(samplingGroup);
    _spinDownsample->setRange(2, 64);
    _spinDownsample->setSingleStep(2);
    _spinDownsample->setValue(_downsample);
    _spinDownsample->setToolTip(tr("Controls how densely surface control points are sampled"));
    downsampleLayout->addWidget(downsampleLabel);
    downsampleLayout->addWidget(_spinDownsample);
    downsampleLayout->addStretch();
    samplingLayout->addLayout(downsampleLayout);

    layout->addWidget(samplingGroup);

    auto* influenceGroup = new QGroupBox(tr("Influence"), this);
    auto* influenceLayout = new QVBoxLayout(influenceGroup);

    auto* modeLayout = new QHBoxLayout();
    auto* modeLabel = new QLabel(tr("Falloff mode:"), influenceGroup);
    _comboInfluenceMode = new QComboBox(influenceGroup);
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
    auto* rowColLabel = new QLabel(tr("Row/Col preference:"), influenceGroup);
    _comboRowColMode = new QComboBox(influenceGroup);
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
    auto* radiusLabel = new QLabel(tr("Radius:"), influenceGroup);
    _spinRadius = new QSpinBox(influenceGroup);
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
    auto* sigmaLabel = new QLabel(tr("Strength (sigma):"), influenceGroup);
    _spinSigma = new QDoubleSpinBox(influenceGroup);
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

    layout->addWidget(influenceGroup);

    auto* holeGroup = new QGroupBox(tr("Hole Filling"), this);
    auto* holeLayout = new QVBoxLayout(holeGroup);

    auto* holeRadiusLayout = new QHBoxLayout();
    auto* holeRadiusLabel = new QLabel(tr("Search radius:"), holeGroup);
    _spinHoleRadius = new QSpinBox(holeGroup);
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
    auto* holeIterationsLabel = new QLabel(tr("Relax iterations:"), holeGroup);
    _spinHoleIterations = new QSpinBox(holeGroup);
    _spinHoleIterations->setRange(1, 200);
    _spinHoleIterations->setSingleStep(1);
    _spinHoleIterations->setValue(_holeSmoothIterations);
    _spinHoleIterations->setToolTip(tr("Number of smoothing passes applied to the filled patch"));
    holeIterationsLayout->addWidget(holeIterationsLabel);
    holeIterationsLayout->addWidget(_spinHoleIterations);
    holeIterationsLayout->addStretch();
    holeLayout->addLayout(holeIterationsLayout);

    layout->addWidget(holeGroup);

    auto* handleDisplayGroup = new QGroupBox(tr("Handle Display"), this);
    auto* handleDisplayLayout = new QVBoxLayout(handleDisplayGroup);

    _chkHandlesAlwaysVisible = new QCheckBox(tr("Show all handles"), handleDisplayGroup);
    _chkHandlesAlwaysVisible->setChecked(_handlesAlwaysVisible);
    _chkHandlesAlwaysVisible->setToolTip(tr("When unchecked, only handles within the specified world distance from the cursor are shown"));
    handleDisplayLayout->addWidget(_chkHandlesAlwaysVisible);

    auto* handleDistanceLayout = new QHBoxLayout();
    auto* handleDistanceLabel = new QLabel(tr("Display distance:"), handleDisplayGroup);
    _spinHandleDisplayDistance = new QDoubleSpinBox(handleDisplayGroup);
    _spinHandleDisplayDistance->setRange(1.0, 500.0);
    _spinHandleDisplayDistance->setSingleStep(1.0);
    _spinHandleDisplayDistance->setDecimals(1);
    _spinHandleDisplayDistance->setValue(static_cast<double>(_handleDisplayDistance));
    _spinHandleDisplayDistance->setToolTip(tr("Maximum world-space distance from the cursor used to show nearby handles"));
    handleDistanceLayout->addWidget(handleDistanceLabel);
    handleDistanceLayout->addWidget(_spinHandleDisplayDistance);
    handleDistanceLayout->addStretch();
    handleDisplayLayout->addLayout(handleDistanceLayout);

    layout->addWidget(handleDisplayGroup);

    auto* actionsLayout = new QHBoxLayout();
    _btnApply = new QPushButton(tr("Apply"), this);
    _btnApply->setDefault(true);
    _btnReset = new QPushButton(tr("Reset"), this);
    actionsLayout->addWidget(_btnApply);
    actionsLayout->addWidget(_btnReset);
    layout->addLayout(actionsLayout);

    _btnStopTools = new QPushButton(tr("Stop tools"), this);
    layout->addWidget(_btnStopTools);

    layout->addStretch();

    restoreSettings();

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
    if (_spinHoleRadius) {
        _spinHoleRadius->setEnabled(_editingEnabled);
    }
    if (_spinHoleIterations) {
        _spinHoleIterations->setEnabled(_editingEnabled);
    }
    if (_chkHandlesAlwaysVisible) {
        _chkHandlesAlwaysVisible->setEnabled(true);
    }
    if (_spinHandleDisplayDistance) {
        _spinHandleDisplayDistance->setEnabled(_editingEnabled && !_handlesAlwaysVisible);
    }
    if (_btnApply) {
        _btnApply->setEnabled(_editingEnabled && _hasPendingChanges);
    }
    if (_btnReset) {
        _btnReset->setEnabled(_editingEnabled && _hasPendingChanges);
    }
}

void SegmentationWidget::restoreSettings()
{
    QSettings settings("VC.ini", QSettings::IniFormat);

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

    const int storedRowCol = settings.value(QStringLiteral("segmentation_edit/row_col_mode"), static_cast<int>(_rowColMode)).toInt();
    const int clampedRowCol = std::clamp(storedRowCol,
                                         static_cast<int>(SegmentationRowColMode::RowOnly),
                                         static_cast<int>(SegmentationRowColMode::Dynamic));
    setRowColMode(static_cast<SegmentationRowColMode>(clampedRowCol));

    const int storedHoleRadius = settings.value(QStringLiteral("segmentation_edit/hole_search_radius"), _holeSearchRadius).toInt();
    setHoleSearchRadius(std::clamp(storedHoleRadius, 1, 64));

    const int storedHoleIterations = settings.value(QStringLiteral("segmentation_edit/hole_smooth_iterations"), _holeSmoothIterations).toInt();
    setHoleSmoothIterations(std::clamp(storedHoleIterations, 1, 200));

    const bool storedHandlesAlways = settings.value(QStringLiteral("segmentation_edit/handles_always_visible"), _handlesAlwaysVisible).toBool();
    setHandlesAlwaysVisible(storedHandlesAlways);

    const double storedHandleDistance = settings.value(QStringLiteral("segmentation_edit/handle_display_distance"), static_cast<double>(_handleDisplayDistance)).toDouble();
    setHandleDisplayDistance(static_cast<float>(std::clamp(storedHandleDistance, 1.0, 500.0)));
}

void SegmentationWidget::writeSetting(const QString& key, const QVariant& value)
{
    QSettings settings("VC.ini", QSettings::IniFormat);
    settings.setValue(QStringLiteral("segmentation_edit/%1").arg(key), value);
}
