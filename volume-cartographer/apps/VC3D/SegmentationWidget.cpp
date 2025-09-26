#include "SegmentationWidget.hpp"

#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QSignalBlocker>

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
    _spinRadius = new QDoubleSpinBox(influenceGroup);
    _spinRadius->setDecimals(1);
    _spinRadius->setRange(1.0, 250.0);
    _spinRadius->setSingleStep(1.0);
    _spinRadius->setValue(static_cast<double>(_radius));
    _spinRadius->setSuffix(tr(" units"));
    _spinRadius->setToolTip(tr("Controls how far edits propagate on the surface"));
    radiusLayout->addWidget(radiusLabel);
    radiusLayout->addWidget(_spinRadius);
    radiusLayout->addStretch();
    influenceLayout->addLayout(radiusLayout);

    auto* sigmaLayout = new QHBoxLayout();
    auto* sigmaLabel = new QLabel(tr("Falloff sigma:"), influenceGroup);
    _spinSigma = new QDoubleSpinBox(influenceGroup);
    _spinSigma->setDecimals(1);
    _spinSigma->setRange(0.1, 250.0);
    _spinSigma->setSingleStep(0.5);
    _spinSigma->setValue(static_cast<double>(_sigma));
    _spinSigma->setSuffix(tr(" units"));
    _spinSigma->setToolTip(tr("Controls the Gaussian falloff strength (larger values give softer influence)"));
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
        emit downsampleChanged(value);
    });

    connect(_spinRadius, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        float radius = static_cast<float>(value);
        if (std::fabs(radius - _radius) < 1e-4f) {
            return;
        }
        _radius = radius;
        emit radiusChanged(radius);
    });

    connect(_spinSigma, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        float sigma = static_cast<float>(value);
        if (std::fabs(sigma - _sigma) < 1e-4f) {
            return;
        }
        _sigma = sigma;
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
    if (value == _downsample) {
        return;
    }
    _downsample = value;
    const QSignalBlocker blocker(_spinDownsample);
    _spinDownsample->setValue(value);
}

void SegmentationWidget::setRadius(float value)
{
    if (std::fabs(value - _radius) < 1e-4f) {
        return;
    }
    _radius = value;
    const QSignalBlocker blocker(_spinRadius);
    _spinRadius->setValue(static_cast<double>(value));
}

void SegmentationWidget::setSigma(float value)
{
    if (std::fabs(value - _sigma) < 1e-4f) {
        return;
    }
    _sigma = value;
    const QSignalBlocker blocker(_spinSigma);
    _spinSigma->setValue(static_cast<double>(value));
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
