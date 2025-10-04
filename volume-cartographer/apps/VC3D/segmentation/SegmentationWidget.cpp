#include "SegmentationWidget.hpp"

#include "elements/CollapsibleSettingsGroup.hpp"

#include <QAbstractItemView>
#include <QByteArray>
#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QGroupBox>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QListWidgetItem>
#include <QLoggingCategory>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollBar>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QToolButton>
#include <QVariant>
#include <QVBoxLayout>
#include <QHBoxLayout>

#include <algorithm>
#include <cmath>
#include <exception>

#include <nlohmann/json.hpp>

namespace
{
Q_LOGGING_CATEGORY(lcSegWidget, "vc.segmentation.widget")

constexpr int kGrowDirUpBit = 1 << 0;
constexpr int kGrowDirDownBit = 1 << 1;
constexpr int kGrowDirLeftBit = 1 << 2;
constexpr int kGrowDirRightBit = 1 << 3;
constexpr int kGrowDirAllMask = kGrowDirUpBit | kGrowDirDownBit | kGrowDirLeftBit | kGrowDirRightBit;
constexpr int kCompactDirectionFieldRowLimit = 3;

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

QString settingsGroup()
{
    return QStringLiteral("segmentation_edit");
}
}

QString SegmentationWidget::determineDefaultVolumeId(const QVector<QPair<QString, QString>>& volumes,
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

SegmentationWidget::SegmentationWidget(QWidget* parent)
    : QWidget(parent)
{
    _growthDirectionMask = kGrowDirAllMask;
    buildUi();
    restoreSettings();
    syncUiState();
}

void SegmentationWidget::buildUi()
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(8, 8, 8, 8);
    layout->setSpacing(12);

    auto* editingRow = new QHBoxLayout();
    _chkEditing = new QCheckBox(tr("Enable editing"), this);
    _chkEditing->setToolTip(tr("Start or stop segmentation editing so brush tools can modify surfaces."));
    _lblStatus = new QLabel(this);
    _lblStatus->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    editingRow->addWidget(_chkEditing);
    editingRow->addSpacing(8);
    editingRow->addWidget(_lblStatus, 1);
    layout->addLayout(editingRow);

    auto* brushRow = new QHBoxLayout();
    brushRow->addSpacing(4);
    _chkEraseBrush = new QCheckBox(tr("Invalidation brush (Shift)"), this);
    _chkEraseBrush->setToolTip(tr("Hold Shift to temporarily switch to the invalidate brush while editing."));
    _chkEraseBrush->setEnabled(false);
    brushRow->addWidget(_chkEraseBrush);
    brushRow->addStretch(1);
    layout->addLayout(brushRow);

    _groupGrowth = new QGroupBox(tr("Surface Growth"), this);
    auto* growthLayout = new QVBoxLayout(_groupGrowth);

    auto* dirRow = new QHBoxLayout();
    auto* stepsLabel = new QLabel(tr("Steps:"), _groupGrowth);
    _spinGrowthSteps = new QSpinBox(_groupGrowth);
    _spinGrowthSteps->setRange(1, 1024);
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

    _btnGrow = new QPushButton(tr("Grow"), _groupGrowth);
    _btnGrow->setToolTip(tr("Run surface growth using the configured steps and directions."));
    growthLayout->addWidget(_btnGrow);

    auto* volumeRow = new QHBoxLayout();
    auto* volumeLabel = new QLabel(tr("Volume:"), _groupGrowth);
    _comboVolumes = new QComboBox(_groupGrowth);
    _comboVolumes->setEnabled(false);
    _comboVolumes->setToolTip(tr("Select which volume provides source data for segmentation growth."));
    volumeRow->addWidget(volumeLabel);
    volumeRow->addWidget(_comboVolumes, 1);
    growthLayout->addLayout(volumeRow);

    _groupGrowth->setLayout(growthLayout);
    layout->addWidget(_groupGrowth);

    _lblNormalGrid = new QLabel(this);
    _lblNormalGrid->setTextFormat(Qt::RichText);
    _lblNormalGrid->setToolTip(tr("Shows whether precomputed normal grids are available for push/pull tools."));
    _lblNormalGrid->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    layout->addWidget(_lblNormalGrid);

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
    _spinPushPullStep->setRange(0.05, 10.0);
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

    layout->addWidget(_groupEditing);

    _groupDirectionField = new CollapsibleSettingsGroup(tr("Direction Fields"), this);

    auto* directionParent = _groupDirectionField->contentWidget();

    _groupDirectionField->addRow(tr("Zarr folder:"), [&](QHBoxLayout* row) {
        _directionFieldPathEdit = new QLineEdit(directionParent);
        _directionFieldPathEdit->setToolTip(tr("Filesystem path to the direction field zarr folder."));
        _directionFieldBrowseButton = new QToolButton(directionParent);
        _directionFieldBrowseButton->setText(QStringLiteral("..."));
        _directionFieldBrowseButton->setToolTip(tr("Browse for a direction field dataset on disk."));
        row->addWidget(_directionFieldPathEdit, 1);
        row->addWidget(_directionFieldBrowseButton);
    }, tr("Filesystem path to the direction field zarr folder."));

    _groupDirectionField->addRow(tr("Orientation:"), [&](QHBoxLayout* row) {
        _comboDirectionFieldOrientation = new QComboBox(directionParent);
        _comboDirectionFieldOrientation->setToolTip(tr("Select which axis the direction field describes."));
        _comboDirectionFieldOrientation->addItem(tr("Normal"), static_cast<int>(SegmentationDirectionFieldOrientation::Normal));
        _comboDirectionFieldOrientation->addItem(tr("Horizontal"), static_cast<int>(SegmentationDirectionFieldOrientation::Horizontal));
        _comboDirectionFieldOrientation->addItem(tr("Vertical"), static_cast<int>(SegmentationDirectionFieldOrientation::Vertical));
        row->addWidget(_comboDirectionFieldOrientation);
        row->addSpacing(12);

        auto* scaleLabel = new QLabel(tr("Scale level:"), directionParent);
        _comboDirectionFieldScale = new QComboBox(directionParent);
        _comboDirectionFieldScale->setToolTip(tr("Choose the multiscale level sampled from the direction field."));
        for (int scale = 0; scale <= 5; ++scale) {
            _comboDirectionFieldScale->addItem(QString::number(scale), scale);
        }
        row->addWidget(scaleLabel);
        row->addWidget(_comboDirectionFieldScale);
        row->addSpacing(12);

        auto* weightLabel = new QLabel(tr("Weight:"), directionParent);
        _spinDirectionFieldWeight = new QDoubleSpinBox(directionParent);
        _spinDirectionFieldWeight->setDecimals(2);
        _spinDirectionFieldWeight->setToolTip(tr("Relative influence of this direction field during growth."));
        _spinDirectionFieldWeight->setRange(0.0, 10.0);
        _spinDirectionFieldWeight->setSingleStep(0.1);
        row->addWidget(weightLabel);
        row->addWidget(_spinDirectionFieldWeight);
        row->addStretch(1);
    });

    _groupDirectionField->addRow(QString(), [&](QHBoxLayout* row) {
        _directionFieldAddButton = new QPushButton(tr("Add"), directionParent);
        _directionFieldAddButton->setToolTip(tr("Save the current direction field parameters to the list."));
        _directionFieldRemoveButton = new QPushButton(tr("Remove"), directionParent);
        _directionFieldRemoveButton->setToolTip(tr("Delete the selected direction field entry."));
        _directionFieldRemoveButton->setEnabled(false);
        row->addWidget(_directionFieldAddButton);
        row->addWidget(_directionFieldRemoveButton);
        row->addStretch(1);
    });

    _directionFieldList = new QListWidget(directionParent);
    _directionFieldList->setToolTip(tr("Direction field configurations applied during growth."));
    _directionFieldList->setSelectionMode(QAbstractItemView::SingleSelection);
    _groupDirectionField->addFullWidthWidget(_directionFieldList);

    layout->addWidget(_groupDirectionField);

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
    rememberGroupState(_groupDirectionField, QStringLiteral("group_direction_field_expanded"));

    _groupCorrections = new QGroupBox(tr("Corrections"), this);
    auto* correctionsLayout = new QVBoxLayout(_groupCorrections);

    auto* correctionsComboRow = new QHBoxLayout();
    auto* correctionsLabel = new QLabel(tr("Active set:"), _groupCorrections);
    _comboCorrections = new QComboBox(_groupCorrections);
    _comboCorrections->setEnabled(false);
    _comboCorrections->setToolTip(tr("Choose an existing correction set to apply."));
    correctionsComboRow->addWidget(correctionsLabel);
    correctionsComboRow->addStretch(1);
    correctionsComboRow->addWidget(_comboCorrections, 1);
    correctionsLayout->addLayout(correctionsComboRow);

    _btnCorrectionsNew = new QPushButton(tr("New correction set"), _groupCorrections);
    _btnCorrectionsNew->setToolTip(tr("Create a new, empty correction set for this segmentation."));
    correctionsLayout->addWidget(_btnCorrectionsNew);

    _chkCorrectionsAnnotate = new QCheckBox(tr("Annotate corrections"), _groupCorrections);
    _chkCorrectionsAnnotate->setToolTip(tr("Toggle annotation overlay while reviewing corrections."));
    correctionsLayout->addWidget(_chkCorrectionsAnnotate);

    _groupCorrections->setLayout(correctionsLayout);
    layout->addWidget(_groupCorrections);

    _groupCustomParams = new QGroupBox(tr("Custom Params"), this);
    auto* customParamsLayout = new QVBoxLayout(_groupCustomParams);

    auto* customParamsDescription = new QLabel(
        tr("Additional JSON fields merge into the tracer params. Leave empty for defaults."), _groupCustomParams);
    customParamsDescription->setWordWrap(true);
    customParamsLayout->addWidget(customParamsDescription);

    _editCustomParams = new QPlainTextEdit(_groupCustomParams);
    _editCustomParams->setToolTip(tr("Optional JSON that merges into tracer parameters before growth."));
    _editCustomParams->setPlaceholderText(QStringLiteral("{\n    \"example_param\": 1\n}"));
    _editCustomParams->setTabChangesFocus(true);
    customParamsLayout->addWidget(_editCustomParams);

    _lblCustomParamsStatus = new QLabel(_groupCustomParams);
    _lblCustomParamsStatus->setWordWrap(true);
    _lblCustomParamsStatus->setVisible(false);
    _lblCustomParamsStatus->setStyleSheet(QStringLiteral("color: #c0392b;"));
    customParamsLayout->addWidget(_lblCustomParamsStatus);

    _groupCustomParams->setLayout(customParamsLayout);
    layout->addWidget(_groupCustomParams);

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
    layout->addLayout(buttons);

    layout->addStretch(1);

    connect(_chkEditing, &QCheckBox::toggled, this, [this](bool enabled) {
        updateEditingState(enabled, true);
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
        value = std::clamp(value, 1, 1024);
        if (_growthSteps == value) {
            return;
        }
        _growthSteps = value;
        writeSetting(QStringLiteral("growth_steps"), _growthSteps);
    });

    const auto triggerConfiguredGrowth = [this]() {
        const auto allowed = allowedGrowthDirections();
        auto direction = SegmentationGrowthDirection::All;
        if (allowed.size() == 1) {
            direction = allowed.front();
        }
        triggerGrowthRequest(direction, _growthSteps);
    };

    connect(_btnGrow, &QPushButton::clicked, this, triggerConfiguredGrowth);

    connect(_comboVolumes, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0) {
            return;
        }
        const QString volumeId = _comboVolumes->itemData(index).toString();
        if (volumeId.isEmpty() || volumeId == _activeVolumeId) {
            return;
        }
        _activeVolumeId = volumeId;
        emit volumeSelectionChanged(volumeId);
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

    connect(_directionFieldPathEdit, &QLineEdit::textChanged, this, [this](const QString& text) {
        _directionFieldPath = text.trimmed();
        if (!_updatingDirectionFieldForm) {
            applyDirectionFieldDraftToSelection(_directionFieldList ? _directionFieldList->currentRow() : -1);
        }
    });

    connect(_directionFieldBrowseButton, &QToolButton::clicked, this, [this]() {
        const QString initial = _directionFieldPath.isEmpty() ? QDir::homePath() : _directionFieldPath;
        const QString dir = QFileDialog::getExistingDirectory(this, tr("Select direction field"), initial);
        if (dir.isEmpty()) {
            return;
        }
        _directionFieldPath = dir;
        _directionFieldPathEdit->setText(dir);
    });

    connect(_comboDirectionFieldOrientation, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        _directionFieldOrientation = segmentationDirectionFieldOrientationFromInt(
            _comboDirectionFieldOrientation->itemData(index).toInt());
        if (!_updatingDirectionFieldForm) {
            applyDirectionFieldDraftToSelection(_directionFieldList ? _directionFieldList->currentRow() : -1);
        }
    });

    connect(_comboDirectionFieldScale, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        _directionFieldScale = _comboDirectionFieldScale->itemData(index).toInt();
        if (!_updatingDirectionFieldForm) {
            applyDirectionFieldDraftToSelection(_directionFieldList ? _directionFieldList->currentRow() : -1);
        }
    });

    connect(_spinDirectionFieldWeight, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        _directionFieldWeight = value;
        if (!_updatingDirectionFieldForm) {
            applyDirectionFieldDraftToSelection(_directionFieldList ? _directionFieldList->currentRow() : -1);
        }
    });

    connect(_directionFieldAddButton, &QPushButton::clicked, this, [this]() {
        auto config = buildDirectionFieldDraft();
        if (!config.isValid()) {
            qCInfo(lcSegWidget) << "Ignoring direction field add; path empty";
            return;
        }
        _directionFields.push_back(std::move(config));
        refreshDirectionFieldList();
        persistDirectionFields();
    });

    connect(_directionFieldRemoveButton, &QPushButton::clicked, this, [this]() {
        const int row = _directionFieldList ? _directionFieldList->currentRow() : -1;
        if (row < 0 || row >= static_cast<int>(_directionFields.size())) {
            return;
        }
        _directionFields.erase(_directionFields.begin() + row);
        refreshDirectionFieldList();
        persistDirectionFields();
    });

    connect(_directionFieldList, &QListWidget::currentRowChanged, this, [this](int row) {
        updateDirectionFieldFormFromSelection(row);
        if (_directionFieldRemoveButton) {
            _directionFieldRemoveButton->setEnabled(_editingEnabled && row >= 0);
        }
    });

    connect(_comboCorrections, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0) {
            emit correctionsCollectionSelected(0);
            return;
        }
        const QVariant data = _comboCorrections->itemData(index);
        emit correctionsCollectionSelected(data.toULongLong());
    });

    connect(_btnCorrectionsNew, &QPushButton::clicked, this, [this]() {
        emit correctionsCreateRequested();
    });

    connect(_editCustomParams, &QPlainTextEdit::textChanged, this, [this]() {
        handleCustomParamsEdited();
    });

    connect(_chkCorrectionsAnnotate, &QCheckBox::toggled, this, [this](bool enabled) {
        emit correctionsAnnotateToggled(enabled);
    });

    connect(_chkCorrectionsUseZRange, &QCheckBox::toggled, this, [this](bool enabled) {
        _correctionsZRangeEnabled = enabled;
        writeSetting(QStringLiteral("corrections_z_range_enabled"), _correctionsZRangeEnabled);
        updateGrowthUiState();
        emit correctionsZRangeChanged(enabled, _correctionsZMin, _correctionsZMax);
    });

    connect(_spinCorrectionsZMin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (_correctionsZMin == value) {
            return;
        }
        _correctionsZMin = value;
        writeSetting(QStringLiteral("corrections_z_min"), _correctionsZMin);
        if (_correctionsZRangeEnabled) {
            emit correctionsZRangeChanged(true, _correctionsZMin, _correctionsZMax);
        }
    });

    connect(_spinCorrectionsZMax, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (_correctionsZMax == value) {
            return;
        }
        _correctionsZMax = value;
        writeSetting(QStringLiteral("corrections_z_max"), _correctionsZMax);
        if (_correctionsZRangeEnabled) {
            emit correctionsZRangeChanged(true, _correctionsZMin, _correctionsZMax);
        }
    });

    connect(_btnApply, &QPushButton::clicked, this, &SegmentationWidget::applyRequested);
    connect(_btnReset, &QPushButton::clicked, this, &SegmentationWidget::resetRequested);
    connect(_btnStop, &QPushButton::clicked, this, &SegmentationWidget::stopToolsRequested);
}

void SegmentationWidget::syncUiState()
{
    if (_chkEditing) {
        const QSignalBlocker blocker(_chkEditing);
        _chkEditing->setChecked(_editingEnabled);
    }

    if (_lblStatus) {
        if (_editingEnabled) {
            _lblStatus->setText(_pending ? tr("Editing enabled – pending changes")
                                         : tr("Editing enabled"));
        } else {
            _lblStatus->setText(tr("Editing disabled"));
        }
    }

    if (_chkEraseBrush) {
        const QSignalBlocker blocker(_chkEraseBrush);
        _chkEraseBrush->setChecked(_eraseBrushActive);
        _chkEraseBrush->setEnabled(_editingEnabled);
    }

    const bool editingActive = _editingEnabled && !_growthInProgress;

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

    if (_editCustomParams) {
        if (_editCustomParams->toPlainText() != _customParamsText) {
            const QSignalBlocker blocker(_editCustomParams);
            _editCustomParams->setPlainText(_customParamsText);
        }
    }
    updateCustomParamsStatus();

    if (_spinGrowthSteps) {
        const QSignalBlocker blocker(_spinGrowthSteps);
        _spinGrowthSteps->setValue(_growthSteps);
    }

    applyGrowthDirectionMaskToUi();
    refreshDirectionFieldList();

    if (_directionFieldPathEdit) {
        const QSignalBlocker blocker(_directionFieldPathEdit);
        _directionFieldPathEdit->setText(_directionFieldPath);
    }
    if (_comboDirectionFieldOrientation) {
        const QSignalBlocker blocker(_comboDirectionFieldOrientation);
        int idx = _comboDirectionFieldOrientation->findData(static_cast<int>(_directionFieldOrientation));
        if (idx >= 0) {
            _comboDirectionFieldOrientation->setCurrentIndex(idx);
        }
    }
    if (_comboDirectionFieldScale) {
        const QSignalBlocker blocker(_comboDirectionFieldScale);
        int idx = _comboDirectionFieldScale->findData(_directionFieldScale);
        if (idx >= 0) {
            _comboDirectionFieldScale->setCurrentIndex(idx);
        }
    }
    if (_spinDirectionFieldWeight) {
        const QSignalBlocker blocker(_spinDirectionFieldWeight);
        _spinDirectionFieldWeight->setValue(_directionFieldWeight);
    }

    if (_comboCorrections) {
        const QSignalBlocker blocker(_comboCorrections);
        _comboCorrections->setEnabled(_correctionsEnabled && !_growthInProgress && _comboCorrections->count() > 0);
    }
    if (_chkCorrectionsAnnotate) {
        const QSignalBlocker blocker(_chkCorrectionsAnnotate);
        _chkCorrectionsAnnotate->setChecked(_correctionsAnnotateChecked);
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

    if (_lblNormalGrid) {
        const QString icon = _normalGridAvailable
            ? QStringLiteral("<span style=\"color:#2e7d32; font-size:16px;\">&#10003;</span>")
            : QStringLiteral("<span style=\"color:#c62828; font-size:16px;\">&#10007;</span>");
        const bool hasExplicitLocation = !_normalGridDisplayPath.isEmpty() && _normalGridDisplayPath != _normalGridHint;
        QString message;
        if (hasExplicitLocation) {
            message = _normalGridAvailable
                ? tr("Normal grids found at %1").arg(_normalGridDisplayPath)
                : tr("Normal grids not found at %1").arg(_normalGridDisplayPath);
        } else {
            message = _normalGridAvailable ? tr("Normal grids found.") : tr("Normal grids not found.");
            if (!_normalGridHint.isEmpty()) {
                message.append(QStringLiteral(" ("));
                message.append(_normalGridHint);
                message.append(QLatin1Char(')'));
            }
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

    updateGrowthUiState();
}

void SegmentationWidget::restoreSettings()
{
    QSettings settings(QStringLiteral("VC.ini"), QSettings::IniFormat);
    settings.beginGroup(settingsGroup());

    _restoringSettings = true;

    if (settings.contains(QStringLiteral("drag_radius_steps"))) {
        _dragRadiusSteps = settings.value(QStringLiteral("drag_radius_steps"), _dragRadiusSteps).toFloat();
    } else {
        _dragRadiusSteps = settings.value(QStringLiteral("radius_steps"), _dragRadiusSteps).toFloat();
    }

    if (settings.contains(QStringLiteral("drag_sigma_steps"))) {
        _dragSigmaSteps = settings.value(QStringLiteral("drag_sigma_steps"), _dragSigmaSteps).toFloat();
    } else {
        _dragSigmaSteps = settings.value(QStringLiteral("sigma_steps"), _dragSigmaSteps).toFloat();
    }

    _lineRadiusSteps = settings.value(QStringLiteral("line_radius_steps"), _dragRadiusSteps).toFloat();
    _lineSigmaSteps = settings.value(QStringLiteral("line_sigma_steps"), _dragSigmaSteps).toFloat();

    _pushPullRadiusSteps = settings.value(QStringLiteral("push_pull_radius_steps"), _dragRadiusSteps).toFloat();
    _pushPullSigmaSteps = settings.value(QStringLiteral("push_pull_sigma_steps"), _dragSigmaSteps).toFloat();

    _dragRadiusSteps = std::clamp(_dragRadiusSteps, 0.25f, 128.0f);
    _dragSigmaSteps = std::clamp(_dragSigmaSteps, 0.05f, 64.0f);
    _lineRadiusSteps = std::clamp(_lineRadiusSteps, 0.25f, 128.0f);
    _lineSigmaSteps = std::clamp(_lineSigmaSteps, 0.05f, 64.0f);
    _pushPullRadiusSteps = std::clamp(_pushPullRadiusSteps, 0.25f, 128.0f);
    _pushPullSigmaSteps = std::clamp(_pushPullSigmaSteps, 0.05f, 64.0f);

    _pushPullStep = settings.value(QStringLiteral("push_pull_step"), _pushPullStep).toFloat();
    _pushPullStep = std::clamp(_pushPullStep, 0.05f, 10.0f);

    AlphaPushPullConfig storedAlpha = _alphaPushPullConfig;
    storedAlpha.start = settings.value(QStringLiteral("push_pull_alpha_start"), storedAlpha.start).toFloat();
    storedAlpha.stop = settings.value(QStringLiteral("push_pull_alpha_stop"), storedAlpha.stop).toFloat();
    storedAlpha.step = settings.value(QStringLiteral("push_pull_alpha_step"), storedAlpha.step).toFloat();
    storedAlpha.low = settings.value(QStringLiteral("push_pull_alpha_low"), storedAlpha.low).toFloat();
    storedAlpha.high = settings.value(QStringLiteral("push_pull_alpha_high"), storedAlpha.high).toFloat();
    storedAlpha.borderOffset = settings.value(QStringLiteral("push_pull_alpha_border"), storedAlpha.borderOffset).toFloat();
    storedAlpha.blurRadius = settings.value(QStringLiteral("push_pull_alpha_radius"), storedAlpha.blurRadius).toInt();
    storedAlpha.perVertexLimit = settings.value(QStringLiteral("push_pull_alpha_limit"), storedAlpha.perVertexLimit).toFloat();
    storedAlpha.perVertex = settings.value(QStringLiteral("push_pull_alpha_per_vertex"), storedAlpha.perVertex).toBool();
    applyAlphaPushPullConfig(storedAlpha, false, false);
    _smoothStrength = settings.value(QStringLiteral("smooth_strength"), _smoothStrength).toFloat();
    _smoothIterations = settings.value(QStringLiteral("smooth_iterations"), _smoothIterations).toInt();
    _smoothStrength = std::clamp(_smoothStrength, 0.0f, 1.0f);
    _smoothIterations = std::clamp(_smoothIterations, 1, 25);
    _growthMethod = segmentationGrowthMethodFromInt(
        settings.value(QStringLiteral("growth_method"), static_cast<int>(_growthMethod)).toInt());
    _growthSteps = settings.value(QStringLiteral("growth_steps"), _growthSteps).toInt();
    _growthSteps = std::clamp(_growthSteps, 1, 1024);
    _growthDirectionMask = normalizeGrowthDirectionMask(
        settings.value(QStringLiteral("growth_direction_mask"), kGrowDirAllMask).toInt());

    QVariantList serialized = settings.value(QStringLiteral("direction_fields"), QVariantList{}).toList();
    _directionFields.clear();
    for (const QVariant& entry : serialized) {
        const QVariantMap map = entry.toMap();
        SegmentationDirectionFieldConfig config;
        config.path = map.value(QStringLiteral("path")).toString();
        config.orientation = segmentationDirectionFieldOrientationFromInt(
            map.value(QStringLiteral("orientation"), 0).toInt());
        config.scale = map.value(QStringLiteral("scale"), 0).toInt();
        config.weight = map.value(QStringLiteral("weight"), 1.0).toDouble();
        if (config.isValid()) {
            _directionFields.push_back(std::move(config));
        }
    }

    _correctionsEnabled = settings.value(QStringLiteral("corrections_enabled"), false).toBool();
    _correctionsZRangeEnabled = settings.value(QStringLiteral("corrections_z_range_enabled"), false).toBool();
    _correctionsZMin = settings.value(QStringLiteral("corrections_z_min"), 0).toInt();
   _correctionsZMax = settings.value(QStringLiteral("corrections_z_max"), _correctionsZMin).toInt();
    if (_correctionsZMax < _correctionsZMin) {
        _correctionsZMax = _correctionsZMin;
    }

    _customParamsText = settings.value(QStringLiteral("custom_params_text"), QString()).toString();
    validateCustomParamsText();

    const bool editingExpanded = settings.value(QStringLiteral("group_editing_expanded"), true).toBool();
    const bool dragExpanded = settings.value(QStringLiteral("group_drag_expanded"), true).toBool();
    const bool lineExpanded = settings.value(QStringLiteral("group_line_expanded"), true).toBool();
    const bool pushPullExpanded = settings.value(QStringLiteral("group_push_pull_expanded"), true).toBool();
    const bool directionExpanded = settings.value(QStringLiteral("group_direction_field_expanded"), true).toBool();

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
    if (_groupDirectionField) {
        _groupDirectionField->setExpanded(directionExpanded);
    }

    settings.endGroup();
    _restoringSettings = false;
}

void SegmentationWidget::writeSetting(const QString& key, const QVariant& value)
{
    QSettings settings(QStringLiteral("VC.ini"), QSettings::IniFormat);
    settings.beginGroup(settingsGroup());
    settings.setValue(key, value);
    settings.endGroup();
}

void SegmentationWidget::updateEditingState(bool enabled, bool notifyListeners)
{
    if (_editingEnabled == enabled) {
        return;
    }

    _editingEnabled = enabled;
    if (!_editingEnabled && _eraseBrushActive) {
        _eraseBrushActive = false;
    }
    syncUiState();

    if (notifyListeners) {
        emit editingModeChanged(_editingEnabled);
    }
}

void SegmentationWidget::setEraseBrushActive(bool active)
{
    const bool sanitized = _editingEnabled && active;
    if (_eraseBrushActive == sanitized) {
        return;
    }
    _eraseBrushActive = sanitized;
    syncUiState();
}

void SegmentationWidget::setPendingChanges(bool pending)
{
    if (_pending == pending) {
        return;
    }
    _pending = pending;
    syncUiState();
}

void SegmentationWidget::setEditingEnabled(bool enabled)
{
    updateEditingState(enabled, false);
}

void SegmentationWidget::setDragRadius(float value)
{
    const float clamped = std::clamp(value, 0.25f, 128.0f);
    if (std::fabs(clamped - _dragRadiusSteps) < 1e-4f) {
        return;
    }
    _dragRadiusSteps = clamped;
    writeSetting(QStringLiteral("drag_radius_steps"), _dragRadiusSteps);
    if (_spinDragRadius) {
        const QSignalBlocker blocker(_spinDragRadius);
        _spinDragRadius->setValue(static_cast<double>(_dragRadiusSteps));
    }
}

void SegmentationWidget::setDragSigma(float value)
{
    const float clamped = std::clamp(value, 0.05f, 64.0f);
    if (std::fabs(clamped - _dragSigmaSteps) < 1e-4f) {
        return;
    }
    _dragSigmaSteps = clamped;
    writeSetting(QStringLiteral("drag_sigma_steps"), _dragSigmaSteps);
    if (_spinDragSigma) {
        const QSignalBlocker blocker(_spinDragSigma);
        _spinDragSigma->setValue(static_cast<double>(_dragSigmaSteps));
    }
}

void SegmentationWidget::setLineRadius(float value)
{
    const float clamped = std::clamp(value, 0.25f, 128.0f);
    if (std::fabs(clamped - _lineRadiusSteps) < 1e-4f) {
        return;
    }
    _lineRadiusSteps = clamped;
    writeSetting(QStringLiteral("line_radius_steps"), _lineRadiusSteps);
    if (_spinLineRadius) {
        const QSignalBlocker blocker(_spinLineRadius);
        _spinLineRadius->setValue(static_cast<double>(_lineRadiusSteps));
    }
}

void SegmentationWidget::setLineSigma(float value)
{
    const float clamped = std::clamp(value, 0.05f, 64.0f);
    if (std::fabs(clamped - _lineSigmaSteps) < 1e-4f) {
        return;
    }
    _lineSigmaSteps = clamped;
    writeSetting(QStringLiteral("line_sigma_steps"), _lineSigmaSteps);
    if (_spinLineSigma) {
        const QSignalBlocker blocker(_spinLineSigma);
        _spinLineSigma->setValue(static_cast<double>(_lineSigmaSteps));
    }
}

void SegmentationWidget::setPushPullRadius(float value)
{
    const float clamped = std::clamp(value, 0.25f, 128.0f);
    if (std::fabs(clamped - _pushPullRadiusSteps) < 1e-4f) {
        return;
    }
    _pushPullRadiusSteps = clamped;
    writeSetting(QStringLiteral("push_pull_radius_steps"), _pushPullRadiusSteps);
    if (_spinPushPullRadius) {
        const QSignalBlocker blocker(_spinPushPullRadius);
        _spinPushPullRadius->setValue(static_cast<double>(_pushPullRadiusSteps));
    }
}

void SegmentationWidget::setPushPullSigma(float value)
{
    const float clamped = std::clamp(value, 0.05f, 64.0f);
    if (std::fabs(clamped - _pushPullSigmaSteps) < 1e-4f) {
        return;
    }
    _pushPullSigmaSteps = clamped;
    writeSetting(QStringLiteral("push_pull_sigma_steps"), _pushPullSigmaSteps);
    if (_spinPushPullSigma) {
        const QSignalBlocker blocker(_spinPushPullSigma);
        _spinPushPullSigma->setValue(static_cast<double>(_pushPullSigmaSteps));
    }
}

void SegmentationWidget::setPushPullStep(float value)
{
    const float clamped = std::clamp(value, 0.05f, 10.0f);
    if (std::fabs(clamped - _pushPullStep) < 1e-4f) {
        return;
    }
    _pushPullStep = clamped;
    writeSetting(QStringLiteral("push_pull_step"), _pushPullStep);
    if (_spinPushPullStep) {
        const QSignalBlocker blocker(_spinPushPullStep);
        _spinPushPullStep->setValue(static_cast<double>(_pushPullStep));
    }
}

AlphaPushPullConfig SegmentationWidget::alphaPushPullConfig() const
{
    return _alphaPushPullConfig;
}

void SegmentationWidget::setAlphaPushPullConfig(const AlphaPushPullConfig& config)
{
    applyAlphaPushPullConfig(config, false);
}

void SegmentationWidget::applyAlphaPushPullConfig(const AlphaPushPullConfig& config,
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

    const bool editingActive = _editingEnabled && !_growthInProgress;

    if (_spinAlphaStart) {
        const QSignalBlocker blocker(_spinAlphaStart);
        _spinAlphaStart->setValue(static_cast<double>(_alphaPushPullConfig.start));
        _spinAlphaStart->setEnabled(editingActive);
    }
    if (_spinAlphaStop) {
        const QSignalBlocker blocker(_spinAlphaStop);
        _spinAlphaStop->setValue(static_cast<double>(_alphaPushPullConfig.stop));
        _spinAlphaStop->setEnabled(editingActive);
    }
    if (_spinAlphaStep) {
        const QSignalBlocker blocker(_spinAlphaStep);
        _spinAlphaStep->setValue(static_cast<double>(_alphaPushPullConfig.step));
        _spinAlphaStep->setEnabled(editingActive);
    }
    if (_spinAlphaLow) {
        const QSignalBlocker blocker(_spinAlphaLow);
        _spinAlphaLow->setValue(normalizedOpacityToDisplay(_alphaPushPullConfig.low));
        _spinAlphaLow->setEnabled(editingActive);
    }
    if (_spinAlphaHigh) {
        const QSignalBlocker blocker(_spinAlphaHigh);
        _spinAlphaHigh->setValue(normalizedOpacityToDisplay(_alphaPushPullConfig.high));
        _spinAlphaHigh->setEnabled(editingActive);
    }
    if (_spinAlphaBorder) {
        const QSignalBlocker blocker(_spinAlphaBorder);
        _spinAlphaBorder->setValue(static_cast<double>(_alphaPushPullConfig.borderOffset));
        _spinAlphaBorder->setEnabled(editingActive);
    }
    if (_spinAlphaBlurRadius) {
        const QSignalBlocker blocker(_spinAlphaBlurRadius);
        _spinAlphaBlurRadius->setValue(_alphaPushPullConfig.blurRadius);
        _spinAlphaBlurRadius->setEnabled(editingActive);
    }
    if (_spinAlphaPerVertexLimit) {
        const QSignalBlocker blocker(_spinAlphaPerVertexLimit);
        _spinAlphaPerVertexLimit->setValue(static_cast<double>(_alphaPushPullConfig.perVertexLimit));
        _spinAlphaPerVertexLimit->setEnabled(editingActive);
    }
    if (_chkAlphaPerVertex) {
        const QSignalBlocker blocker(_chkAlphaPerVertex);
        _chkAlphaPerVertex->setChecked(_alphaPushPullConfig.perVertex);
        _chkAlphaPerVertex->setEnabled(editingActive);
    }
    if (_alphaPushPullPanel) {
        _alphaPushPullPanel->setEnabled(editingActive);
    }

    if (emitSignal && changed) {
        emit alphaPushPullConfigChanged();
    }
}

void SegmentationWidget::setSmoothingStrength(float value)
{
    const float clamped = std::clamp(value, 0.0f, 1.0f);
    if (std::fabs(clamped - _smoothStrength) < 1e-4f) {
        return;
    }
    _smoothStrength = clamped;
    writeSetting(QStringLiteral("smooth_strength"), _smoothStrength);
    if (_spinSmoothStrength) {
        const QSignalBlocker blocker(_spinSmoothStrength);
        _spinSmoothStrength->setValue(static_cast<double>(_smoothStrength));
    }
}

void SegmentationWidget::setSmoothingIterations(int value)
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

void SegmentationWidget::handleCustomParamsEdited()
{
    if (!_editCustomParams) {
        return;
    }
    _customParamsText = _editCustomParams->toPlainText();
    writeSetting(QStringLiteral("custom_params_text"), _customParamsText);
    validateCustomParamsText();
    updateCustomParamsStatus();
}

void SegmentationWidget::validateCustomParamsText()
{
    QString error;
    parseCustomParams(&error);
    _customParamsError = error;
}

void SegmentationWidget::updateCustomParamsStatus()
{
    if (!_lblCustomParamsStatus) {
        return;
    }
    if (_customParamsError.isEmpty()) {
        _lblCustomParamsStatus->clear();
        _lblCustomParamsStatus->setVisible(false);
        return;
    }
    _lblCustomParamsStatus->setText(_customParamsError);
    _lblCustomParamsStatus->setVisible(true);
}

std::optional<nlohmann::json> SegmentationWidget::parseCustomParams(QString* error) const
{
    if (error) {
        error->clear();
    }

    const QString trimmed = _customParamsText.trimmed();
    if (trimmed.isEmpty()) {
        return std::nullopt;
    }

    try {
        const QByteArray utf8 = trimmed.toUtf8();
        nlohmann::json parsed = nlohmann::json::parse(utf8.constData(), utf8.constData() + utf8.size());
        if (!parsed.is_object()) {
            if (error) {
                *error = tr("Custom params must be a JSON object.");
            }
            return std::nullopt;
        }
        return parsed;
    } catch (const nlohmann::json::parse_error& ex) {
        if (error) {
            *error = tr("Custom params JSON parse error (byte %1): %2")
                         .arg(static_cast<qulonglong>(ex.byte))
                         .arg(QString::fromStdString(ex.what()));
        }
    } catch (const std::exception& ex) {
        if (error) {
            *error = tr("Custom params JSON parse error: %1")
                         .arg(QString::fromStdString(ex.what()));
        }
    } catch (...) {
        if (error) {
            *error = tr("Custom params JSON parse error: unknown error");
        }
    }

    return std::nullopt;
}

std::optional<nlohmann::json> SegmentationWidget::customParamsJson() const
{
    QString error;
    auto parsed = parseCustomParams(&error);
    if (!error.isEmpty()) {
        return std::nullopt;
    }
    return parsed;
}

void SegmentationWidget::setGrowthMethod(SegmentationGrowthMethod method)
{
    if (_growthMethod == method) {
        return;
    }
    _growthMethod = method;
    writeSetting(QStringLiteral("growth_method"), static_cast<int>(_growthMethod));
    syncUiState();
    emit growthMethodChanged(_growthMethod);
}

void SegmentationWidget::setGrowthInProgress(bool running)
{
    if (_growthInProgress == running) {
        return;
    }
    _growthInProgress = running;
    updateGrowthUiState();
}

void SegmentationWidget::setNormalGridAvailable(bool available)
{
    _normalGridAvailable = available;
    syncUiState();
}

void SegmentationWidget::setNormalGridPathHint(const QString& hint)
{
    _normalGridHint = hint;
    QString display = hint.trimmed();
    const int colonIndex = display.indexOf(QLatin1Char(':'));
    if (colonIndex >= 0 && colonIndex + 1 < display.size()) {
        display = display.mid(colonIndex + 1).trimmed();
    }
    _normalGridDisplayPath = display;
    syncUiState();
}

void SegmentationWidget::setVolumePackagePath(const QString& path)
{
    _volumePackagePath = path;
    syncUiState();
}

void SegmentationWidget::setAvailableVolumes(const QVector<QPair<QString, QString>>& volumes,
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

void SegmentationWidget::setActiveVolume(const QString& volumeId)
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

void SegmentationWidget::setCorrectionsEnabled(bool enabled)
{
    if (_correctionsEnabled == enabled) {
        return;
    }
    _correctionsEnabled = enabled;
    writeSetting(QStringLiteral("corrections_enabled"), _correctionsEnabled);
    if (!enabled) {
        _correctionsAnnotateChecked = false;
        if (_chkCorrectionsAnnotate) {
            const QSignalBlocker blocker(_chkCorrectionsAnnotate);
            _chkCorrectionsAnnotate->setChecked(false);
        }
    }
    updateGrowthUiState();
}

void SegmentationWidget::setCorrectionsAnnotateChecked(bool enabled)
{
    _correctionsAnnotateChecked = enabled;
    if (_chkCorrectionsAnnotate) {
        const QSignalBlocker blocker(_chkCorrectionsAnnotate);
        _chkCorrectionsAnnotate->setChecked(enabled);
    }
    updateGrowthUiState();
}

void SegmentationWidget::setCorrectionCollections(const QVector<QPair<uint64_t, QString>>& collections,
                                                  std::optional<uint64_t> activeId)
{
    if (!_comboCorrections) {
        return;
    }
    const QSignalBlocker blocker(_comboCorrections);
    _comboCorrections->clear();
    for (const auto& pair : collections) {
        _comboCorrections->addItem(pair.second, QVariant::fromValue(static_cast<qulonglong>(pair.first)));
    }
    if (activeId) {
        int idx = _comboCorrections->findData(QVariant::fromValue(static_cast<qulonglong>(*activeId)));
        if (idx >= 0) {
            _comboCorrections->setCurrentIndex(idx);
        }
    } else {
        _comboCorrections->setCurrentIndex(-1);
    }
    _comboCorrections->setEnabled(_correctionsEnabled && !_growthInProgress && _comboCorrections->count() > 0);
}

std::optional<std::pair<int, int>> SegmentationWidget::correctionsZRange() const
{
    if (!_correctionsZRangeEnabled) {
        return std::nullopt;
    }
    return std::make_pair(_correctionsZMin, _correctionsZMax);
}

std::vector<SegmentationGrowthDirection> SegmentationWidget::allowedGrowthDirections() const
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

std::vector<SegmentationDirectionFieldConfig> SegmentationWidget::directionFieldConfigs() const
{
    std::vector<SegmentationDirectionFieldConfig> configs;
    configs.reserve(_directionFields.size());
    for (const auto& config : _directionFields) {
        if (config.isValid()) {
            configs.push_back(config);
        }
    }
    return configs;
}

SegmentationDirectionFieldConfig SegmentationWidget::buildDirectionFieldDraft() const
{
    SegmentationDirectionFieldConfig config;
    config.path = _directionFieldPath.trimmed();
    config.orientation = _directionFieldOrientation;
    config.scale = std::clamp(_directionFieldScale, 0, 5);
    config.weight = std::clamp(_directionFieldWeight, 0.0, 10.0);
    return config;
}

void SegmentationWidget::refreshDirectionFieldList()
{
    if (!_directionFieldList) {
        return;
    }
    const QSignalBlocker blocker(_directionFieldList);
    const int previousRow = _directionFieldList->currentRow();
    _directionFieldList->clear();

    for (const auto& config : _directionFields) {
        QString orientationLabel = segmentationDirectionFieldOrientationKey(config.orientation);
        const QString weightText = QString::number(std::clamp(config.weight, 0.0, 10.0), 'f', 2);
        const QString itemText = tr("%1 — %2 (scale %3, weight %4)")
                                     .arg(config.path,
                                          orientationLabel,
                                          QString::number(std::clamp(config.scale, 0, 5)),
                                          weightText);
        auto* item = new QListWidgetItem(itemText, _directionFieldList);
        item->setToolTip(config.path);
    }

    if (!_directionFields.empty()) {
        const int clampedRow = std::clamp(previousRow, 0, static_cast<int>(_directionFields.size()) - 1);
        _directionFieldList->setCurrentRow(clampedRow);
    }
    if (_directionFieldRemoveButton) {
        _directionFieldRemoveButton->setEnabled(_editingEnabled && !_directionFields.empty() && _directionFieldList->currentRow() >= 0);
    }

    updateDirectionFieldFormFromSelection(_directionFieldList->currentRow());
    updateDirectionFieldListGeometry();
}

void SegmentationWidget::updateDirectionFieldFormFromSelection(int row)
{
    const bool previousUpdating = _updatingDirectionFieldForm;
    _updatingDirectionFieldForm = true;

    if (row >= 0 && row < static_cast<int>(_directionFields.size())) {
        const auto& config = _directionFields[static_cast<std::size_t>(row)];
        _directionFieldPath = config.path;
        _directionFieldOrientation = config.orientation;
        _directionFieldScale = config.scale;
        _directionFieldWeight = config.weight;
    }

    if (_directionFieldPathEdit) {
        const QSignalBlocker blocker(_directionFieldPathEdit);
        _directionFieldPathEdit->setText(_directionFieldPath);
    }
    if (_comboDirectionFieldOrientation) {
        const QSignalBlocker blocker(_comboDirectionFieldOrientation);
        int idx = _comboDirectionFieldOrientation->findData(static_cast<int>(_directionFieldOrientation));
        if (idx >= 0) {
            _comboDirectionFieldOrientation->setCurrentIndex(idx);
        }
    }
    if (_comboDirectionFieldScale) {
        const QSignalBlocker blocker(_comboDirectionFieldScale);
        int idx = _comboDirectionFieldScale->findData(_directionFieldScale);
        if (idx >= 0) {
            _comboDirectionFieldScale->setCurrentIndex(idx);
        }
    }
    if (_spinDirectionFieldWeight) {
        const QSignalBlocker blocker(_spinDirectionFieldWeight);
        _spinDirectionFieldWeight->setValue(_directionFieldWeight);
    }

    _updatingDirectionFieldForm = previousUpdating;
}

void SegmentationWidget::applyDirectionFieldDraftToSelection(int row)
{
    if (row < 0 || row >= static_cast<int>(_directionFields.size())) {
        return;
    }

    auto config = buildDirectionFieldDraft();
    if (!config.isValid()) {
        return;
    }

    auto& target = _directionFields[static_cast<std::size_t>(row)];
    if (target.path == config.path &&
        target.orientation == config.orientation &&
        target.scale == config.scale &&
        std::abs(target.weight - config.weight) < 1e-4) {
        return;
    }

    target = std::move(config);
    updateDirectionFieldListItem(row);
    persistDirectionFields();
}

void SegmentationWidget::updateDirectionFieldListItem(int row)
{
    if (!_directionFieldList) {
        return;
    }
    if (row < 0 || row >= _directionFieldList->count()) {
        return;
    }
    if (row >= static_cast<int>(_directionFields.size())) {
        return;
    }

    const auto& config = _directionFields[static_cast<std::size_t>(row)];
    QString orientationLabel = segmentationDirectionFieldOrientationKey(config.orientation);
    const QString weightText = QString::number(std::clamp(config.weight, 0.0, 10.0), 'f', 2);
    const QString itemText = tr("%1 — %2 (scale %3, weight %4)")
                                 .arg(config.path,
                                      orientationLabel,
                                      QString::number(std::clamp(config.scale, 0, 5)),
                                      weightText);

    if (auto* item = _directionFieldList->item(row)) {
        item->setText(itemText);
        item->setToolTip(config.path);
    }
}

void SegmentationWidget::updateDirectionFieldListGeometry()
{
    if (!_directionFieldList) {
        return;
    }

    auto policy = _directionFieldList->sizePolicy();
    const int itemCount = _directionFieldList->count();

    if (itemCount <= kCompactDirectionFieldRowLimit) {
        const int sampleRowHeight = _directionFieldList->sizeHintForRow(0);
        const int rowHeight = sampleRowHeight > 0 ? sampleRowHeight : _directionFieldList->fontMetrics().height() + 8;
        const int visibleRows = std::max(1, itemCount);
        const int frameHeight = 2 * _directionFieldList->frameWidth();
        const auto* hScroll = _directionFieldList->horizontalScrollBar();
        const int scrollHeight = (hScroll && hScroll->isVisible()) ? hScroll->sizeHint().height() : 0;
        const int targetHeight = rowHeight * visibleRows + frameHeight + scrollHeight;

        policy.setVerticalPolicy(QSizePolicy::Fixed);
        policy.setVerticalStretch(0);
        _directionFieldList->setSizePolicy(policy);
        _directionFieldList->setMinimumHeight(targetHeight);
        _directionFieldList->setMaximumHeight(targetHeight);
    } else {
        policy.setVerticalPolicy(QSizePolicy::Expanding);
        policy.setVerticalStretch(1);
        _directionFieldList->setSizePolicy(policy);
        _directionFieldList->setMinimumHeight(0);
        _directionFieldList->setMaximumHeight(QWIDGETSIZE_MAX);
    }

    _directionFieldList->updateGeometry();
}

void SegmentationWidget::persistDirectionFields()
{
    QVariantList serialized;
    serialized.reserve(static_cast<int>(_directionFields.size()));
    for (const auto& config : _directionFields) {
        QVariantMap map;
        map.insert(QStringLiteral("path"), config.path);
        map.insert(QStringLiteral("orientation"), static_cast<int>(config.orientation));
        map.insert(QStringLiteral("scale"), std::clamp(config.scale, 0, 5));
        map.insert(QStringLiteral("weight"), std::clamp(config.weight, 0.0, 10.0));
        serialized.push_back(map);
    }
    writeSetting(QStringLiteral("direction_fields"), serialized);
}

void SegmentationWidget::setGrowthDirectionMask(int mask)
{
    mask = normalizeGrowthDirectionMask(mask);
    if (_growthDirectionMask == mask) {
        return;
    }
    _growthDirectionMask = mask;
    writeSetting(QStringLiteral("growth_direction_mask"), _growthDirectionMask);
    applyGrowthDirectionMaskToUi();
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
        mask = kGrowDirAllMask;
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

void SegmentationWidget::updateGrowthUiState()
{
    const bool enableGrowth = _editingEnabled && !_growthInProgress;
    if (_spinGrowthSteps) {
        _spinGrowthSteps->setEnabled(enableGrowth);
    }
    if (_btnGrow) {
        _btnGrow->setEnabled(enableGrowth);
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
    if (_directionFieldAddButton) {
        _directionFieldAddButton->setEnabled(_editingEnabled);
    }
    if (_directionFieldRemoveButton) {
        const bool hasSelection = _directionFieldList && _directionFieldList->currentRow() >= 0;
        _directionFieldRemoveButton->setEnabled(_editingEnabled && hasSelection);
    }
    if (_directionFieldList) {
        _directionFieldList->setEnabled(_editingEnabled);
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
    if (_groupCustomParams) {
        _groupCustomParams->setEnabled(_editingEnabled);
    }

    const bool allowCorrections = _editingEnabled && _correctionsEnabled && !_growthInProgress;
    if (_groupCorrections) {
        _groupCorrections->setEnabled(allowCorrections);
    }
    if (_comboCorrections) {
        const QSignalBlocker blocker(_comboCorrections);
        _comboCorrections->setEnabled(allowCorrections && _comboCorrections->count() > 0);
    }
    if (_btnCorrectionsNew) {
        _btnCorrectionsNew->setEnabled(_editingEnabled && !_growthInProgress);
    }
    if (_chkCorrectionsAnnotate) {
        _chkCorrectionsAnnotate->setEnabled(allowCorrections);
    }
}

void SegmentationWidget::triggerGrowthRequest(SegmentationGrowthDirection direction, int steps)
{
    if (!_editingEnabled || _growthInProgress) {
        return;
    }

    const int clampedSteps = std::clamp(steps, 1, 1024);
    qCInfo(lcSegWidget) << "Grow request" << segmentationGrowthMethodToString(_growthMethod)
                        << segmentationGrowthDirectionToString(direction) << "steps" << clampedSteps;
    emit growSurfaceRequested(_growthMethod, direction, clampedSteps);
}

int SegmentationWidget::normalizeGrowthDirectionMask(int mask)
{
    mask &= kGrowDirAllMask;
    if (mask == 0) {
        // If no directions are selected, enable all directions by default.
        // This ensures that growth is not unintentionally disabled.
        mask = kGrowDirAllMask;
    }
    return mask;
}
