#include "SegmentationWidget.hpp"

#include <QCheckBox>
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
}

void SegmentationWidget::writeSetting(const QString& key, const QVariant& value)
{
    QSettings settings("VC.ini", QSettings::IniFormat);
    settings.setValue(QStringLiteral("segmentation_edit/%1").arg(key), value);
}
