#include "SegmentationEditingPanel.hpp"

#include "VCSettings.hpp"
#include "elements/CollapsibleSettingsGroup.hpp"

#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <utility>

namespace
{
constexpr float kFloatEpsilon = 1e-4f;
constexpr float kAlphaOpacityScale = 255.0f;

bool nearlyEqual(float lhs, float rhs)
{
    return std::fabs(lhs - rhs) < kFloatEpsilon;
}

float displayOpacityToNormalized(double displayValue)
{
    return static_cast<float>(displayValue / kAlphaOpacityScale);
}

double normalizedOpacityToDisplay(float normalizedValue)
{
    return static_cast<double>(normalizedValue * kAlphaOpacityScale);
}

AlphaPushPullConfig sanitizeAlphaConfig(const AlphaPushPullConfig& config)
{
    AlphaPushPullConfig sanitized = config;

    sanitized.start = std::clamp(sanitized.start, -128.0f, 128.0f);
    sanitized.stop = std::clamp(sanitized.stop, -128.0f, 128.0f);
    if (sanitized.start > sanitized.stop) {
        std::swap(sanitized.start, sanitized.stop);
    }

    const float minStep = 0.05f;
    const float maxStep = 20.0f;
    const float magnitude = std::clamp(std::fabs(sanitized.step), minStep, maxStep);
    sanitized.step = (sanitized.step < 0.0f) ? -magnitude : magnitude;

    sanitized.low = std::clamp(sanitized.low, 0.0f, 1.0f);
    sanitized.high = std::clamp(sanitized.high, 0.0f, 1.0f);
    if (sanitized.high <= sanitized.low + 0.01f) {
        sanitized.high = std::min(1.0f, sanitized.low + 0.05f);
    }

    sanitized.borderOffset = std::clamp(sanitized.borderOffset, -20.0f, 20.0f);
    sanitized.blurRadius = std::clamp(sanitized.blurRadius, 0, 15);
    sanitized.perVertexLimit = std::clamp(sanitized.perVertexLimit, 0.0f, 128.0f);

    return sanitized;
}
} // namespace

SegmentationEditingPanel::SegmentationEditingPanel(const QString& settingsGroup,
                                                   QWidget* parent)
    : QWidget(parent)
    , _settingsGroup(settingsGroup)
{
    auto* panelLayout = new QVBoxLayout(this);
    panelLayout->setContentsMargins(0, 0, 0, 0);
    panelLayout->setSpacing(12);

    auto* hoverRow = new QHBoxLayout();
    hoverRow->addSpacing(4);
    _chkShowHoverMarker = new QCheckBox(tr("Show hover marker"), this);
    _chkShowHoverMarker->setToolTip(tr("Toggle the hover indicator in the segmentation viewer. "
                                       "Disabling this hides the preview marker and defers grid lookups "
                                       "until you drag or use push/pull."));
    hoverRow->addWidget(_chkShowHoverMarker);
    hoverRow->addStretch(1);
    panelLayout->addLayout(hoverRow);

    _groupEditing = new CollapsibleSettingsGroup(tr("Editing"), this);
    auto* falloffLayout = _groupEditing->contentLayout();
    auto* falloffParent = _groupEditing->contentWidget();

    auto createToolGroup = [&](const QString& title,
                               QDoubleSpinBox*& radiusSpin,
                               QDoubleSpinBox*& sigmaSpin) {
        auto* group = new CollapsibleSettingsGroup(title, _groupEditing);
        radiusSpin = group->addDoubleSpinBox(tr("Radius"), 0.25, 128.0, 0.25);
        sigmaSpin = group->addDoubleSpinBox(tr("Sigma"), 0.05, 64.0, 0.1);
        return group;
    };

    _groupDrag = createToolGroup(tr("Drag Brush"), _spinDragRadius, _spinDragSigma);
    _groupLine = createToolGroup(tr("Line Brush (S)"), _spinLineRadius, _spinLineSigma);

    _groupPushPull = new CollapsibleSettingsGroup(tr("Push/Pull (A / D, Ctrl for alpha)"), _groupEditing);
    auto* pushGrid = new QGridLayout();
    pushGrid->setContentsMargins(0, 0, 0, 0);
    pushGrid->setHorizontalSpacing(12);
    pushGrid->setVerticalSpacing(8);
    _groupPushPull->contentLayout()->addLayout(pushGrid);

    auto* pushParent = _groupPushPull->contentWidget();

    auto* ppRadiusLabel = new QLabel(tr("Radius"), pushParent);
    _spinPushPullRadius = new QDoubleSpinBox(pushParent);
    _spinPushPullRadius->setDecimals(2);
    _spinPushPullRadius->setRange(0.25, 128.0);
    _spinPushPullRadius->setSingleStep(0.25);
    pushGrid->addWidget(ppRadiusLabel, 0, 0);
    pushGrid->addWidget(_spinPushPullRadius, 0, 1);

    auto* ppSigmaLabel = new QLabel(tr("Sigma"), pushParent);
    _spinPushPullSigma = new QDoubleSpinBox(pushParent);
    _spinPushPullSigma->setDecimals(2);
    _spinPushPullSigma->setRange(0.05, 64.0);
    _spinPushPullSigma->setSingleStep(0.1);
    pushGrid->addWidget(ppSigmaLabel, 0, 2);
    pushGrid->addWidget(_spinPushPullSigma, 0, 3);

    auto* pushPullLabel = new QLabel(tr("Step"), pushParent);
    _spinPushPullStep = new QDoubleSpinBox(pushParent);
    _spinPushPullStep->setDecimals(2);
    _spinPushPullStep->setRange(0.05, 40.0);
    _spinPushPullStep->setSingleStep(0.05);
    pushGrid->addWidget(pushPullLabel, 1, 0);
    pushGrid->addWidget(_spinPushPullStep, 1, 1);

    _lblAlphaInfo = new QLabel(tr("Hold Ctrl with A/D to sample alpha while pushing or pulling."), pushParent);
    _lblAlphaInfo->setWordWrap(true);
    _lblAlphaInfo->setToolTip(tr("Hold Ctrl when starting push/pull to stop at the configured alpha thresholds."));
    pushGrid->addWidget(_lblAlphaInfo, 2, 0, 1, 4);

    _alphaPushPullPanel = new QWidget(pushParent);
    auto* alphaGrid = new QGridLayout(_alphaPushPullPanel);
    alphaGrid->setContentsMargins(0, 0, 0, 0);
    alphaGrid->setHorizontalSpacing(12);
    alphaGrid->setVerticalSpacing(6);

    auto addAlphaWidget = [&](const QString& labelText, QWidget* widget, int row, int column, const QString& tooltip) {
        auto* label = new QLabel(labelText, _alphaPushPullPanel);
        label->setToolTip(tooltip);
        widget->setToolTip(tooltip);
        const int columnBase = column * 2;
        alphaGrid->addWidget(label, row, columnBase);
        alphaGrid->addWidget(widget, row, columnBase + 1);
    };

    auto addAlphaControl = [&](const QString& labelText,
                               QDoubleSpinBox*& target,
                               double min,
                               double max,
                               double step,
                               int row,
                               int column,
                               const QString& tooltip) {
        auto* spin = new QDoubleSpinBox(_alphaPushPullPanel);
        spin->setDecimals(2);
        spin->setRange(min, max);
        spin->setSingleStep(step);
        target = spin;
        addAlphaWidget(labelText, spin, row, column, tooltip);
    };

    auto addAlphaIntControl = [&](const QString& labelText,
                                  QSpinBox*& target,
                                  int min,
                                  int max,
                                  int step,
                                  int row,
                                  int column,
                                  const QString& tooltip) {
        auto* spin = new QSpinBox(_alphaPushPullPanel);
        spin->setRange(min, max);
        spin->setSingleStep(step);
        target = spin;
        addAlphaWidget(labelText, spin, row, column, tooltip);
    };

    int alphaRow = 0;
    addAlphaControl(tr("Start"), _spinAlphaStart, -64.0, 64.0, 0.5, alphaRow, 0,
                    tr("Beginning distance (along the brush normal) where alpha sampling starts."));
    addAlphaControl(tr("Stop"), _spinAlphaStop, -64.0, 64.0, 0.5, alphaRow++, 1,
                    tr("Ending distance for alpha sampling; the search stops once this depth is reached."));
    addAlphaControl(tr("Sample step"), _spinAlphaStep, 0.05, 20.0, 0.05, alphaRow, 0,
                    tr("Spacing between alpha samples inside the start/stop range; smaller steps follow fine features."));
    addAlphaControl(tr("Border offset"), _spinAlphaBorder, -20.0, 20.0, 0.1, alphaRow++, 1,
                    tr("Extra offset applied after the alpha front is located, keeping a safety margin."));
    addAlphaControl(tr("Opacity low"), _spinAlphaLow, 0.0, 255.0, 1.0, alphaRow, 0,
                    tr("Lower bound of the opacity window; voxels below this behave as transparent."));
    addAlphaControl(tr("Opacity high"), _spinAlphaHigh, 0.0, 255.0, 1.0, alphaRow++, 1,
                    tr("Upper bound of the opacity window; voxels above this are fully opaque."));

    const QString blurTooltip = tr("Gaussian blur radius for each sampled slice; higher values smooth noisy volumes before thresholding.");
    addAlphaIntControl(tr("Blur radius"), _spinAlphaBlurRadius, 0, 15, 1, alphaRow++, 0, blurTooltip);

    _chkAlphaPerVertex = new QCheckBox(tr("Independent per-vertex stops"), _alphaPushPullPanel);
    _chkAlphaPerVertex->setToolTip(tr("Move every vertex within the brush independently to the alpha threshold without Gaussian weighting."));
    alphaGrid->addWidget(_chkAlphaPerVertex, alphaRow++, 0, 1, 4);

    const QString perVertexLimitTip = tr("Maximum additional distance (world units) a vertex may exceed relative to the smallest movement in the brush when independent stops are enabled.");
    addAlphaControl(tr("Per-vertex limit"), _spinAlphaPerVertexLimit, 0.0, 128.0, 0.25, alphaRow++, 0, perVertexLimitTip);

    alphaGrid->setColumnStretch(1, 1);
    alphaGrid->setColumnStretch(3, 1);

    pushGrid->addWidget(_alphaPushPullPanel, 3, 0, 1, 4);

    pushGrid->setColumnStretch(1, 1);
    pushGrid->setColumnStretch(3, 1);

    auto setGroupTooltips = [](QWidget* group, QDoubleSpinBox* radiusSpin, QDoubleSpinBox* sigmaSpin, const QString& radiusTip, const QString& sigmaTip) {
        if (group) {
            group->setToolTip(radiusTip + QLatin1Char('\n') + sigmaTip);
        }
        if (radiusSpin) {
            radiusSpin->setToolTip(radiusTip);
        }
        if (sigmaSpin) {
            sigmaSpin->setToolTip(sigmaTip);
        }
    };

    setGroupTooltips(_groupDrag,
                     _spinDragRadius,
                     _spinDragSigma,
                     tr("Brush radius in grid steps for drag edits."),
                     tr("Gaussian falloff sigma for drag edits."));
    setGroupTooltips(_groupLine,
                     _spinLineRadius,
                     _spinLineSigma,
                     tr("Brush radius in grid steps for line drags."),
                     tr("Gaussian falloff sigma for line drags."));
    setGroupTooltips(_groupPushPull,
                     _spinPushPullRadius,
                     _spinPushPullSigma,
                     tr("Radius in grid steps that participates in push/pull."),
                     tr("Gaussian falloff sigma for push/pull."));
    if (_spinPushPullStep) {
        _spinPushPullStep->setToolTip(tr("Baseline step size (in world units) for classic push/pull when alpha mode is disabled."));
    }

    auto* brushToolsRow = new QHBoxLayout();
    brushToolsRow->setSpacing(12);
    brushToolsRow->addWidget(_groupDrag, 1);
    brushToolsRow->addWidget(_groupLine, 1);
    falloffLayout->addLayout(brushToolsRow);

    auto* pushPullRow = new QHBoxLayout();
    pushPullRow->setSpacing(12);
    pushPullRow->addWidget(_groupPushPull, 1);
    falloffLayout->addLayout(pushPullRow);

    auto* smoothingRow = new QHBoxLayout();
    auto* smoothStrengthLabel = new QLabel(tr("Smoothing strength"), falloffParent);
    _spinSmoothStrength = new QDoubleSpinBox(falloffParent);
    _spinSmoothStrength->setDecimals(2);
    _spinSmoothStrength->setToolTip(tr("Blend edits toward neighboring vertices; higher values smooth more."));
    _spinSmoothStrength->setRange(0.0, 1.0);
    _spinSmoothStrength->setSingleStep(0.05);
    smoothingRow->addWidget(smoothStrengthLabel);
    smoothingRow->addWidget(_spinSmoothStrength);
    smoothingRow->addSpacing(12);
    auto* smoothIterationsLabel = new QLabel(tr("Iterations"), falloffParent);
    _spinSmoothIterations = new QSpinBox(falloffParent);
    _spinSmoothIterations->setRange(1, 25);
    _spinSmoothIterations->setToolTip(tr("Number of smoothing passes applied after growth."));
    _spinSmoothIterations->setSingleStep(1);
    smoothingRow->addWidget(smoothIterationsLabel);
    smoothingRow->addWidget(_spinSmoothIterations);
    smoothingRow->addStretch(1);
    falloffLayout->addLayout(smoothingRow);

    panelLayout->addWidget(_groupEditing);

    auto* buttons = new QHBoxLayout();
    _btnApply = new QPushButton(tr("Apply"), this);
    _btnApply->setToolTip(tr("Commit pending edits to the segmentation."));
    _btnReset = new QPushButton(tr("Reset"), this);
    _btnReset->setToolTip(tr("Discard pending edits and reload the segmentation state."));
    _btnStop = new QPushButton(tr("Stop tools"), this);
    _btnStop->setToolTip(tr("Exit the active editing tool and return to selection."));
    buttons->addWidget(_btnApply);
    buttons->addWidget(_btnReset);
    buttons->addWidget(_btnStop);
    panelLayout->addLayout(buttons);

    // --- Signal wiring ---

    connect(_chkShowHoverMarker, &QCheckBox::toggled, this, [this](bool enabled) {
        setShowHoverMarker(enabled);
    });

    connect(_spinDragRadius, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setDragRadius(static_cast<float>(value));
        emit dragRadiusChanged(_dragRadiusSteps);
    });

    connect(_spinDragSigma, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setDragSigma(static_cast<float>(value));
        emit dragSigmaChanged(_dragSigmaSteps);
    });

    connect(_spinLineRadius, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setLineRadius(static_cast<float>(value));
        emit lineRadiusChanged(_lineRadiusSteps);
    });

    connect(_spinLineSigma, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setLineSigma(static_cast<float>(value));
        emit lineSigmaChanged(_lineSigmaSteps);
    });

    connect(_spinPushPullRadius, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setPushPullRadius(static_cast<float>(value));
        emit pushPullRadiusChanged(_pushPullRadiusSteps);
    });

    connect(_spinPushPullSigma, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setPushPullSigma(static_cast<float>(value));
        emit pushPullSigmaChanged(_pushPullSigmaSteps);
    });

    connect(_spinPushPullStep, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setPushPullStep(static_cast<float>(value));
        emit pushPullStepChanged(_pushPullStep);
    });

    auto onAlphaValueChanged = [this](auto updater) {
        AlphaPushPullConfig config = _alphaPushPullConfig;
        updater(config);
        applyAlphaPushPullConfig(config, true);
    };

    connect(_spinAlphaStart, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, onAlphaValueChanged](double value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.start = static_cast<float>(value);
        });
    });
    connect(_spinAlphaStop, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, onAlphaValueChanged](double value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.stop = static_cast<float>(value);
        });
    });
    connect(_spinAlphaStep, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, onAlphaValueChanged](double value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.step = static_cast<float>(value);
        });
    });
    connect(_spinAlphaLow, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, onAlphaValueChanged](double value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.low = displayOpacityToNormalized(value);
        });
    });
    connect(_spinAlphaHigh, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, onAlphaValueChanged](double value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.high = displayOpacityToNormalized(value);
        });
    });
    connect(_spinAlphaBorder, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, onAlphaValueChanged](double value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.borderOffset = static_cast<float>(value);
        });
    });
    connect(_spinAlphaBlurRadius, QOverload<int>::of(&QSpinBox::valueChanged), this, [this, onAlphaValueChanged](int value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.blurRadius = value;
        });
    });
    connect(_spinAlphaPerVertexLimit, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, onAlphaValueChanged](double value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.perVertexLimit = static_cast<float>(value);
        });
    });
    connect(_chkAlphaPerVertex, &QCheckBox::toggled, this, [this, onAlphaValueChanged](bool checked) {
        onAlphaValueChanged([checked](AlphaPushPullConfig& cfg) {
            cfg.perVertex = checked;
        });
    });

    connect(_spinSmoothStrength, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setSmoothingStrength(static_cast<float>(value));
        emit smoothingStrengthChanged(_smoothStrength);
    });

    connect(_spinSmoothIterations, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        setSmoothingIterations(value);
        emit smoothingIterationsChanged(_smoothIterations);
    });

    connect(_btnApply, &QPushButton::clicked, this, &SegmentationEditingPanel::applyRequested);
    connect(_btnReset, &QPushButton::clicked, this, &SegmentationEditingPanel::resetRequested);
    connect(_btnStop, &QPushButton::clicked, this, &SegmentationEditingPanel::stopToolsRequested);

    // Remember group expand/collapse state
    auto rememberGroupState = [this](CollapsibleSettingsGroup* group, const QString& key) {
        if (!group) {
            return;
        }
        connect(group, &CollapsibleSettingsGroup::toggled, this, [this, key](bool expanded) {
            if (_restoringSettings) {
                return;
            }
            writeSetting(key, expanded);
        });
    };

    rememberGroupState(_groupEditing, QStringLiteral("group_editing_expanded"));
    rememberGroupState(_groupDrag, QStringLiteral("group_drag_expanded"));
    rememberGroupState(_groupLine, QStringLiteral("group_line_expanded"));
    rememberGroupState(_groupPushPull, QStringLiteral("group_push_pull_expanded"));
}

void SegmentationEditingPanel::writeSetting(const QString& key, const QVariant& value)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(_settingsGroup);
    settings.setValue(key, value);
    settings.endGroup();
}

void SegmentationEditingPanel::setShowHoverMarker(bool enabled)
{
    if (_showHoverMarker == enabled) {
        return;
    }
    _showHoverMarker = enabled;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("show_hover_marker"), _showHoverMarker);
        emit hoverMarkerToggled(_showHoverMarker);
    }
    if (_chkShowHoverMarker) {
        const QSignalBlocker blocker(_chkShowHoverMarker);
        _chkShowHoverMarker->setChecked(_showHoverMarker);
    }
}

void SegmentationEditingPanel::setDragRadius(float value)
{
    const float clamped = std::clamp(value, 0.25f, 128.0f);
    if (std::fabs(clamped - _dragRadiusSteps) < kFloatEpsilon) {
        return;
    }
    _dragRadiusSteps = clamped;
    writeSetting(QStringLiteral("drag_radius_steps"), _dragRadiusSteps);
    if (_spinDragRadius) {
        const QSignalBlocker blocker(_spinDragRadius);
        _spinDragRadius->setValue(static_cast<double>(_dragRadiusSteps));
    }
}

void SegmentationEditingPanel::setDragSigma(float value)
{
    const float clamped = std::clamp(value, 0.05f, 64.0f);
    if (std::fabs(clamped - _dragSigmaSteps) < kFloatEpsilon) {
        return;
    }
    _dragSigmaSteps = clamped;
    writeSetting(QStringLiteral("drag_sigma_steps"), _dragSigmaSteps);
    if (_spinDragSigma) {
        const QSignalBlocker blocker(_spinDragSigma);
        _spinDragSigma->setValue(static_cast<double>(_dragSigmaSteps));
    }
}

void SegmentationEditingPanel::setLineRadius(float value)
{
    const float clamped = std::clamp(value, 0.25f, 128.0f);
    if (std::fabs(clamped - _lineRadiusSteps) < kFloatEpsilon) {
        return;
    }
    _lineRadiusSteps = clamped;
    writeSetting(QStringLiteral("line_radius_steps"), _lineRadiusSteps);
    if (_spinLineRadius) {
        const QSignalBlocker blocker(_spinLineRadius);
        _spinLineRadius->setValue(static_cast<double>(_lineRadiusSteps));
    }
}

void SegmentationEditingPanel::setLineSigma(float value)
{
    const float clamped = std::clamp(value, 0.05f, 64.0f);
    if (std::fabs(clamped - _lineSigmaSteps) < kFloatEpsilon) {
        return;
    }
    _lineSigmaSteps = clamped;
    writeSetting(QStringLiteral("line_sigma_steps"), _lineSigmaSteps);
    if (_spinLineSigma) {
        const QSignalBlocker blocker(_spinLineSigma);
        _spinLineSigma->setValue(static_cast<double>(_lineSigmaSteps));
    }
}

void SegmentationEditingPanel::setPushPullRadius(float value)
{
    const float clamped = std::clamp(value, 0.25f, 128.0f);
    if (std::fabs(clamped - _pushPullRadiusSteps) < kFloatEpsilon) {
        return;
    }
    _pushPullRadiusSteps = clamped;
    writeSetting(QStringLiteral("push_pull_radius_steps"), _pushPullRadiusSteps);
    if (_spinPushPullRadius) {
        const QSignalBlocker blocker(_spinPushPullRadius);
        _spinPushPullRadius->setValue(static_cast<double>(_pushPullRadiusSteps));
    }
}

void SegmentationEditingPanel::setPushPullSigma(float value)
{
    const float clamped = std::clamp(value, 0.05f, 64.0f);
    if (std::fabs(clamped - _pushPullSigmaSteps) < kFloatEpsilon) {
        return;
    }
    _pushPullSigmaSteps = clamped;
    writeSetting(QStringLiteral("push_pull_sigma_steps"), _pushPullSigmaSteps);
    if (_spinPushPullSigma) {
        const QSignalBlocker blocker(_spinPushPullSigma);
        _spinPushPullSigma->setValue(static_cast<double>(_pushPullSigmaSteps));
    }
}

void SegmentationEditingPanel::setPushPullStep(float value)
{
    const float clamped = std::clamp(value, 0.05f, 40.0f);
    if (std::fabs(clamped - _pushPullStep) < kFloatEpsilon) {
        return;
    }
    _pushPullStep = clamped;
    writeSetting(QStringLiteral("push_pull_step"), _pushPullStep);
    if (_spinPushPullStep) {
        const QSignalBlocker blocker(_spinPushPullStep);
        _spinPushPullStep->setValue(static_cast<double>(_pushPullStep));
    }
}

void SegmentationEditingPanel::setAlphaPushPullConfig(const AlphaPushPullConfig& config)
{
    applyAlphaPushPullConfig(config, false);
}

void SegmentationEditingPanel::applyAlphaPushPullConfig(const AlphaPushPullConfig& config,
                                                        bool emitSignal,
                                                        bool persist)
{
    AlphaPushPullConfig sanitized = sanitizeAlphaConfig(config);

    const bool changed = !nearlyEqual(sanitized.start, _alphaPushPullConfig.start) ||
                         !nearlyEqual(sanitized.stop, _alphaPushPullConfig.stop) ||
                         !nearlyEqual(sanitized.step, _alphaPushPullConfig.step) ||
                         !nearlyEqual(sanitized.low, _alphaPushPullConfig.low) ||
                         !nearlyEqual(sanitized.high, _alphaPushPullConfig.high) ||
                         !nearlyEqual(sanitized.borderOffset, _alphaPushPullConfig.borderOffset) ||
                         sanitized.blurRadius != _alphaPushPullConfig.blurRadius ||
                         !nearlyEqual(sanitized.perVertexLimit, _alphaPushPullConfig.perVertexLimit) ||
                         sanitized.perVertex != _alphaPushPullConfig.perVertex;

    if (changed) {
        _alphaPushPullConfig = sanitized;
        if (persist) {
            writeSetting(QStringLiteral("push_pull_alpha_start"), _alphaPushPullConfig.start);
            writeSetting(QStringLiteral("push_pull_alpha_stop"), _alphaPushPullConfig.stop);
            writeSetting(QStringLiteral("push_pull_alpha_step"), _alphaPushPullConfig.step);
            writeSetting(QStringLiteral("push_pull_alpha_low"), _alphaPushPullConfig.low);
            writeSetting(QStringLiteral("push_pull_alpha_high"), _alphaPushPullConfig.high);
            writeSetting(QStringLiteral("push_pull_alpha_border"), _alphaPushPullConfig.borderOffset);
            writeSetting(QStringLiteral("push_pull_alpha_radius"), _alphaPushPullConfig.blurRadius);
            writeSetting(QStringLiteral("push_pull_alpha_limit"), _alphaPushPullConfig.perVertexLimit);
            writeSetting(QStringLiteral("push_pull_alpha_per_vertex"), _alphaPushPullConfig.perVertex);
        }
    }

    // Update UI — always push state to widgets regardless of changed flag
    if (_spinAlphaStart) {
        const QSignalBlocker blocker(_spinAlphaStart);
        _spinAlphaStart->setValue(static_cast<double>(_alphaPushPullConfig.start));
    }
    if (_spinAlphaStop) {
        const QSignalBlocker blocker(_spinAlphaStop);
        _spinAlphaStop->setValue(static_cast<double>(_alphaPushPullConfig.stop));
    }
    if (_spinAlphaStep) {
        const QSignalBlocker blocker(_spinAlphaStep);
        _spinAlphaStep->setValue(static_cast<double>(_alphaPushPullConfig.step));
    }
    if (_spinAlphaLow) {
        const QSignalBlocker blocker(_spinAlphaLow);
        _spinAlphaLow->setValue(normalizedOpacityToDisplay(_alphaPushPullConfig.low));
    }
    if (_spinAlphaHigh) {
        const QSignalBlocker blocker(_spinAlphaHigh);
        _spinAlphaHigh->setValue(normalizedOpacityToDisplay(_alphaPushPullConfig.high));
    }
    if (_spinAlphaBorder) {
        const QSignalBlocker blocker(_spinAlphaBorder);
        _spinAlphaBorder->setValue(static_cast<double>(_alphaPushPullConfig.borderOffset));
    }
    if (_spinAlphaBlurRadius) {
        const QSignalBlocker blocker(_spinAlphaBlurRadius);
        _spinAlphaBlurRadius->setValue(_alphaPushPullConfig.blurRadius);
    }
    if (_spinAlphaPerVertexLimit) {
        const QSignalBlocker blocker(_spinAlphaPerVertexLimit);
        _spinAlphaPerVertexLimit->setValue(static_cast<double>(_alphaPushPullConfig.perVertexLimit));
    }
    if (_chkAlphaPerVertex) {
        const QSignalBlocker blocker(_chkAlphaPerVertex);
        _chkAlphaPerVertex->setChecked(_alphaPushPullConfig.perVertex);
    }

    if (emitSignal && changed) {
        emit alphaPushPullConfigChanged();
    }
}

void SegmentationEditingPanel::setSmoothingStrength(float value)
{
    const float clamped = std::clamp(value, 0.0f, 1.0f);
    if (std::fabs(clamped - _smoothStrength) < kFloatEpsilon) {
        return;
    }
    _smoothStrength = clamped;
    writeSetting(QStringLiteral("smooth_strength"), _smoothStrength);
    if (_spinSmoothStrength) {
        const QSignalBlocker blocker(_spinSmoothStrength);
        _spinSmoothStrength->setValue(static_cast<double>(_smoothStrength));
    }
}

void SegmentationEditingPanel::setSmoothingIterations(int value)
{
    const int clamped = std::clamp(value, 1, 25);
    if (_smoothIterations == clamped) {
        return;
    }
    _smoothIterations = clamped;
    writeSetting(QStringLiteral("smooth_iterations"), _smoothIterations);
    if (_spinSmoothIterations) {
        const QSignalBlocker blocker(_spinSmoothIterations);
        _spinSmoothIterations->setValue(_smoothIterations);
    }
}

void SegmentationEditingPanel::restoreSettings(QSettings& settings)
{
    using namespace vc3d::settings;
    _restoringSettings = true;

    if (settings.contains(segmentation::DRAG_RADIUS_STEPS)) {
        _dragRadiusSteps = settings.value(segmentation::DRAG_RADIUS_STEPS, _dragRadiusSteps).toFloat();
    } else {
        _dragRadiusSteps = settings.value(segmentation::RADIUS_STEPS, _dragRadiusSteps).toFloat();
    }

    if (settings.contains(segmentation::DRAG_SIGMA_STEPS)) {
        _dragSigmaSteps = settings.value(segmentation::DRAG_SIGMA_STEPS, _dragSigmaSteps).toFloat();
    } else {
        _dragSigmaSteps = settings.value(segmentation::SIGMA_STEPS, _dragSigmaSteps).toFloat();
    }

    _lineRadiusSteps = settings.value(segmentation::LINE_RADIUS_STEPS, _dragRadiusSteps).toFloat();
    _lineSigmaSteps = settings.value(segmentation::LINE_SIGMA_STEPS, _dragSigmaSteps).toFloat();

    _pushPullRadiusSteps = settings.value(segmentation::PUSH_PULL_RADIUS_STEPS, _dragRadiusSteps).toFloat();
    _pushPullSigmaSteps = settings.value(segmentation::PUSH_PULL_SIGMA_STEPS, _dragSigmaSteps).toFloat();
    _showHoverMarker = settings.value(segmentation::SHOW_HOVER_MARKER, _showHoverMarker).toBool();

    _dragRadiusSteps = std::clamp(_dragRadiusSteps, 0.25f, 128.0f);
    _dragSigmaSteps = std::clamp(_dragSigmaSteps, 0.05f, 64.0f);
    _lineRadiusSteps = std::clamp(_lineRadiusSteps, 0.25f, 128.0f);
    _lineSigmaSteps = std::clamp(_lineSigmaSteps, 0.05f, 64.0f);
    _pushPullRadiusSteps = std::clamp(_pushPullRadiusSteps, 0.25f, 128.0f);
    _pushPullSigmaSteps = std::clamp(_pushPullSigmaSteps, 0.05f, 64.0f);

    _pushPullStep = settings.value(segmentation::PUSH_PULL_STEP, _pushPullStep).toFloat();
    _pushPullStep = std::clamp(_pushPullStep, 0.05f, 40.0f);

    AlphaPushPullConfig storedAlpha = _alphaPushPullConfig;
    storedAlpha.start = settings.value(segmentation::PUSH_PULL_ALPHA_START, storedAlpha.start).toFloat();
    storedAlpha.stop = settings.value(segmentation::PUSH_PULL_ALPHA_STOP, storedAlpha.stop).toFloat();
    storedAlpha.step = settings.value(segmentation::PUSH_PULL_ALPHA_STEP, storedAlpha.step).toFloat();
    storedAlpha.low = settings.value(segmentation::PUSH_PULL_ALPHA_LOW, storedAlpha.low).toFloat();
    storedAlpha.high = settings.value(segmentation::PUSH_PULL_ALPHA_HIGH, storedAlpha.high).toFloat();
    storedAlpha.borderOffset = settings.value(segmentation::PUSH_PULL_ALPHA_BORDER, storedAlpha.borderOffset).toFloat();
    storedAlpha.blurRadius = settings.value(segmentation::PUSH_PULL_ALPHA_RADIUS, storedAlpha.blurRadius).toInt();
    storedAlpha.perVertexLimit = settings.value(segmentation::PUSH_PULL_ALPHA_LIMIT, storedAlpha.perVertexLimit).toFloat();
    storedAlpha.perVertex = settings.value(segmentation::PUSH_PULL_ALPHA_PER_VERTEX, storedAlpha.perVertex).toBool();
    applyAlphaPushPullConfig(storedAlpha, false, false);

    _smoothStrength = settings.value(segmentation::SMOOTH_STRENGTH, _smoothStrength).toFloat();
    _smoothIterations = settings.value(segmentation::SMOOTH_ITERATIONS, _smoothIterations).toInt();
    _smoothStrength = std::clamp(_smoothStrength, 0.0f, 1.0f);
    _smoothIterations = std::clamp(_smoothIterations, 1, 25);

    // Restore group expand/collapse state
    const bool editingExpanded = settings.value(segmentation::GROUP_EDITING_EXPANDED, segmentation::GROUP_EDITING_EXPANDED_DEFAULT).toBool();
    const bool dragExpanded = settings.value(segmentation::GROUP_DRAG_EXPANDED, segmentation::GROUP_DRAG_EXPANDED_DEFAULT).toBool();
    const bool lineExpanded = settings.value(segmentation::GROUP_LINE_EXPANDED, segmentation::GROUP_LINE_EXPANDED_DEFAULT).toBool();
    const bool pushPullExpanded = settings.value(segmentation::GROUP_PUSH_PULL_EXPANDED, segmentation::GROUP_PUSH_PULL_EXPANDED_DEFAULT).toBool();

    if (_groupEditing) {
        _groupEditing->setExpanded(editingExpanded);
    }
    if (_groupDrag) {
        _groupDrag->setExpanded(dragExpanded);
    }
    if (_groupLine) {
        _groupLine->setExpanded(lineExpanded);
    }
    if (_groupPushPull) {
        _groupPushPull->setExpanded(pushPullExpanded);
    }

    _restoringSettings = false;
}

void SegmentationEditingPanel::syncUiState(bool editingEnabled, bool growthInProgress)
{
    if (_chkShowHoverMarker) {
        const QSignalBlocker blocker(_chkShowHoverMarker);
        _chkShowHoverMarker->setChecked(_showHoverMarker);
    }

    const bool editingActive = editingEnabled && !growthInProgress;

    auto updateSpin = [&](QDoubleSpinBox* spin, float value) {
        if (!spin) {
            return;
        }
        const QSignalBlocker blocker(spin);
        spin->setValue(static_cast<double>(value));
        spin->setEnabled(editingActive);
    };

    updateSpin(_spinDragRadius, _dragRadiusSteps);
    updateSpin(_spinDragSigma, _dragSigmaSteps);
    updateSpin(_spinLineRadius, _lineRadiusSteps);
    updateSpin(_spinLineSigma, _lineSigmaSteps);
    updateSpin(_spinPushPullRadius, _pushPullRadiusSteps);
    updateSpin(_spinPushPullSigma, _pushPullSigmaSteps);

    if (_groupDrag) {
        _groupDrag->setEnabled(editingActive);
    }
    if (_groupLine) {
        _groupLine->setEnabled(editingActive);
    }
    if (_groupPushPull) {
        _groupPushPull->setEnabled(editingActive);
    }

    if (_spinPushPullStep) {
        const QSignalBlocker blocker(_spinPushPullStep);
        _spinPushPullStep->setValue(static_cast<double>(_pushPullStep));
        _spinPushPullStep->setEnabled(editingActive);
    }

    if (_lblAlphaInfo) {
        _lblAlphaInfo->setEnabled(editingActive);
    }

    auto updateAlphaSpin = [&](QDoubleSpinBox* spin, float value, bool opacitySpin = false) {
        if (!spin) {
            return;
        }
        const QSignalBlocker blocker(spin);
        if (opacitySpin) {
            spin->setValue(normalizedOpacityToDisplay(value));
        } else {
            spin->setValue(static_cast<double>(value));
        }
        spin->setEnabled(editingActive);
    };

    updateAlphaSpin(_spinAlphaStart, _alphaPushPullConfig.start);
    updateAlphaSpin(_spinAlphaStop, _alphaPushPullConfig.stop);
    updateAlphaSpin(_spinAlphaStep, _alphaPushPullConfig.step);
    updateAlphaSpin(_spinAlphaLow, _alphaPushPullConfig.low, true);
    updateAlphaSpin(_spinAlphaHigh, _alphaPushPullConfig.high, true);
    updateAlphaSpin(_spinAlphaBorder, _alphaPushPullConfig.borderOffset);

    if (_spinAlphaBlurRadius) {
        const QSignalBlocker blocker(_spinAlphaBlurRadius);
        _spinAlphaBlurRadius->setValue(_alphaPushPullConfig.blurRadius);
        _spinAlphaBlurRadius->setEnabled(editingActive);
    }
    updateAlphaSpin(_spinAlphaPerVertexLimit, _alphaPushPullConfig.perVertexLimit);
    if (_chkAlphaPerVertex) {
        const QSignalBlocker blocker(_chkAlphaPerVertex);
        _chkAlphaPerVertex->setChecked(_alphaPushPullConfig.perVertex);
        _chkAlphaPerVertex->setEnabled(editingActive);
    }
    if (_alphaPushPullPanel) {
        _alphaPushPullPanel->setEnabled(editingActive);
    }

    if (_spinSmoothStrength) {
        const QSignalBlocker blocker(_spinSmoothStrength);
        _spinSmoothStrength->setValue(static_cast<double>(_smoothStrength));
        _spinSmoothStrength->setEnabled(editingActive);
    }
    if (_spinSmoothIterations) {
        const QSignalBlocker blocker(_spinSmoothIterations);
        _spinSmoothIterations->setValue(_smoothIterations);
        _spinSmoothIterations->setEnabled(editingActive);
    }
}
