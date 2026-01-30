#include "SegmentationEditingPanel.hpp"

#include "elements/CollapsibleSettingsGroup.hpp"

#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>

SegmentationEditingPanel::SegmentationEditingPanel(QWidget* parent)
    : QWidget(parent)
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
}
