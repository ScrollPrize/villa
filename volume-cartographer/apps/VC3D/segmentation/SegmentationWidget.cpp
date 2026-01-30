#include "SegmentationWidget.hpp"
#include "SegmentationCommon.hpp"

#include "elements/CollapsibleSettingsGroup.hpp"
#include "NeuralTraceServiceManager.hpp"
#include "tools/SegmentationEditingPanel.hpp"
#include "tools/SegmentationGrowthPanel.hpp"
#include "tools/SegmentationHeaderRow.hpp"
#include "tools/SegmentationCorrectionsPanel.hpp"
#include "tools/SegmentationCustomParamsPanel.hpp"
#include "tools/SegmentationApprovalMaskPanel.hpp"
#include "tools/SegmentationCellReoptPanel.hpp"
#include "tools/SegmentationNeuralTracerPanel.hpp"
#include "tools/SegmentationDirectionFieldPanel.hpp"
#include "VCSettings.hpp"

#include <QAbstractItemView>
#include <QApplication>
#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QEvent>
#include <QGroupBox>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QLoggingCategory>
#include <QMouseEvent>
#include <QPushButton>
#include <QRegularExpression>
#include <QSettings>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QVariant>
#include <QVBoxLayout>
#include <QHBoxLayout>

#include <algorithm>
#include <cmath>

#include <nlohmann/json.hpp>

Q_LOGGING_CATEGORY(lcSegWidget, "vc.segmentation.widget")

namespace
{
constexpr int kGrowDirUpBit = 1 << 0;
constexpr int kGrowDirDownBit = 1 << 1;
constexpr int kGrowDirLeftBit = 1 << 2;
constexpr int kGrowDirRightBit = 1 << 3;
constexpr int kGrowDirAllMask = kGrowDirUpBit | kGrowDirDownBit | kGrowDirLeftBit | kGrowDirRightBit;

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

void SegmentationWidget::applyGrowthSteps(int steps, bool persist, bool fromUi)
{
    const int minimum = (_growthMethod == SegmentationGrowthMethod::Corrections) ? 0 : 1;
    const int clamped = std::clamp(steps, minimum, 1024);

    QSpinBox* growthStepsSpin = _growthPanel ? _growthPanel->growthStepsSpin() : nullptr;
    if ((!fromUi || clamped != steps) && growthStepsSpin) {
        QSignalBlocker blocker(growthStepsSpin);
        growthStepsSpin->setValue(clamped);
    }

    if (clamped > 0) {
        _tracerGrowthSteps = std::max(1, clamped);
    }

    _growthSteps = clamped;

    if (persist) {
        writeSetting(QStringLiteral("growth_steps"), _growthSteps);
        writeSetting(QStringLiteral("growth_steps_tracer"), _tracerGrowthSteps);
    }
}

void SegmentationWidget::setGrowthSteps(int steps, bool persist)
{
    applyGrowthSteps(steps, persist, false);
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

    _headerRow = new SegmentationHeaderRow(this);
    layout->addWidget(_headerRow);

    _growthPanel = new SegmentationGrowthPanel(_growthKeybindsEnabled, this);
    layout->addWidget(_growthPanel);

    _editingPanel = new SegmentationEditingPanel(QStringLiteral("segmentation_edit"), this);
    layout->addWidget(_editingPanel);

    _approvalMaskPanel = new SegmentationApprovalMaskPanel(QStringLiteral("segmentation_edit"), this);
    layout->addWidget(_approvalMaskPanel);

    _cellReoptPanel = new SegmentationCellReoptPanel(QStringLiteral("segmentation_edit"), this);
    layout->addWidget(_cellReoptPanel);

    _directionFieldPanel = new SegmentationDirectionFieldPanel(QStringLiteral("segmentation_edit"), this);
    layout->addWidget(_directionFieldPanel);

    _neuralTracerPanel = new SegmentationNeuralTracerPanel(QStringLiteral("segmentation_edit"), this);
    layout->addWidget(_neuralTracerPanel);

    _correctionsPanel = new SegmentationCorrectionsPanel(QStringLiteral("segmentation_edit"), this);
    layout->addWidget(_correctionsPanel);

    _customParamsPanel = new SegmentationCustomParamsPanel(QStringLiteral("segmentation_edit"), this);
    layout->addWidget(_customParamsPanel);

    layout->addStretch(1);

    connect(_headerRow, &SegmentationHeaderRow::editingToggled, this, [this](bool enabled) {
        updateEditingState(enabled, true);
    });

    // Forward editing panel signals
    connect(_editingPanel, &SegmentationEditingPanel::dragRadiusChanged,
            this, &SegmentationWidget::dragRadiusChanged);
    connect(_editingPanel, &SegmentationEditingPanel::dragSigmaChanged,
            this, &SegmentationWidget::dragSigmaChanged);
    connect(_editingPanel, &SegmentationEditingPanel::lineRadiusChanged,
            this, &SegmentationWidget::lineRadiusChanged);
    connect(_editingPanel, &SegmentationEditingPanel::lineSigmaChanged,
            this, &SegmentationWidget::lineSigmaChanged);
    connect(_editingPanel, &SegmentationEditingPanel::pushPullRadiusChanged,
            this, &SegmentationWidget::pushPullRadiusChanged);
    connect(_editingPanel, &SegmentationEditingPanel::pushPullSigmaChanged,
            this, &SegmentationWidget::pushPullSigmaChanged);
    connect(_editingPanel, &SegmentationEditingPanel::pushPullStepChanged,
            this, &SegmentationWidget::pushPullStepChanged);
    connect(_editingPanel, &SegmentationEditingPanel::alphaPushPullConfigChanged,
            this, &SegmentationWidget::alphaPushPullConfigChanged);
    connect(_editingPanel, &SegmentationEditingPanel::smoothingStrengthChanged,
            this, &SegmentationWidget::smoothingStrengthChanged);
    connect(_editingPanel, &SegmentationEditingPanel::smoothingIterationsChanged,
            this, &SegmentationWidget::smoothingIterationsChanged);
    connect(_editingPanel, &SegmentationEditingPanel::hoverMarkerToggled,
            this, &SegmentationWidget::hoverMarkerToggled);
    connect(_editingPanel, &SegmentationEditingPanel::applyRequested,
            this, &SegmentationWidget::applyRequested);
    connect(_editingPanel, &SegmentationEditingPanel::resetRequested,
            this, &SegmentationWidget::resetRequested);
    connect(_editingPanel, &SegmentationEditingPanel::stopToolsRequested,
            this, &SegmentationWidget::stopToolsRequested);

    // Forward approval mask panel signals
    connect(_approvalMaskPanel, &SegmentationApprovalMaskPanel::showApprovalMaskChanged,
            this, &SegmentationWidget::showApprovalMaskChanged);
    connect(_approvalMaskPanel, &SegmentationApprovalMaskPanel::editApprovedMaskChanged,
            this, &SegmentationWidget::editApprovedMaskChanged);
    connect(_approvalMaskPanel, &SegmentationApprovalMaskPanel::editUnapprovedMaskChanged,
            this, &SegmentationWidget::editUnapprovedMaskChanged);
    connect(_approvalMaskPanel, &SegmentationApprovalMaskPanel::autoApproveEditsChanged,
            this, &SegmentationWidget::autoApproveEditsChanged);
    connect(_approvalMaskPanel, &SegmentationApprovalMaskPanel::approvalBrushRadiusChanged,
            this, &SegmentationWidget::approvalBrushRadiusChanged);
    connect(_approvalMaskPanel, &SegmentationApprovalMaskPanel::approvalBrushDepthChanged,
            this, &SegmentationWidget::approvalBrushDepthChanged);
    connect(_approvalMaskPanel, &SegmentationApprovalMaskPanel::approvalMaskOpacityChanged,
            this, &SegmentationWidget::approvalMaskOpacityChanged);
    connect(_approvalMaskPanel, &SegmentationApprovalMaskPanel::approvalBrushColorChanged,
            this, &SegmentationWidget::approvalBrushColorChanged);
    connect(_approvalMaskPanel, &SegmentationApprovalMaskPanel::approvalStrokesUndoRequested,
            this, &SegmentationWidget::approvalStrokesUndoRequested);

    // Forward cell reopt panel signals
    connect(_cellReoptPanel, &SegmentationCellReoptPanel::cellReoptModeChanged,
            this, &SegmentationWidget::cellReoptModeChanged);
    connect(_cellReoptPanel, &SegmentationCellReoptPanel::cellReoptMaxStepsChanged,
            this, &SegmentationWidget::cellReoptMaxStepsChanged);
    connect(_cellReoptPanel, &SegmentationCellReoptPanel::cellReoptMaxPointsChanged,
            this, &SegmentationWidget::cellReoptMaxPointsChanged);
    connect(_cellReoptPanel, &SegmentationCellReoptPanel::cellReoptMinSpacingChanged,
            this, &SegmentationWidget::cellReoptMinSpacingChanged);
    connect(_cellReoptPanel, &SegmentationCellReoptPanel::cellReoptPerimeterOffsetChanged,
            this, &SegmentationWidget::cellReoptPerimeterOffsetChanged);
    connect(_cellReoptPanel, &SegmentationCellReoptPanel::cellReoptGrowthRequested,
            this, &SegmentationWidget::cellReoptGrowthRequested);

    auto connectDirectionCheckbox = [this](QCheckBox* box) {
        if (!box) {
            return;
        }
        connect(box, &QCheckBox::toggled, this, [this, box](bool) {
            updateGrowthDirectionMaskFromUi(box);
        });
    };
    QCheckBox* growthDirUp = _growthPanel ? _growthPanel->growthDirUpCheck() : nullptr;
    QCheckBox* growthDirDown = _growthPanel ? _growthPanel->growthDirDownCheck() : nullptr;
    QCheckBox* growthDirLeft = _growthPanel ? _growthPanel->growthDirLeftCheck() : nullptr;
    QCheckBox* growthDirRight = _growthPanel ? _growthPanel->growthDirRightCheck() : nullptr;
    connectDirectionCheckbox(growthDirUp);
    connectDirectionCheckbox(growthDirDown);
    connectDirectionCheckbox(growthDirLeft);
    connectDirectionCheckbox(growthDirRight);

    if (QCheckBox* growthKeybinds = _growthPanel ? _growthPanel->growthKeybindsCheck() : nullptr) {
        connect(growthKeybinds, &QCheckBox::toggled, this, [this](bool checked) {
            _growthKeybindsEnabled = checked;
            writeSetting(QStringLiteral("growth_keybinds_enabled"), _growthKeybindsEnabled);
        });
    }

    if (QSpinBox* growthStepsSpin = _growthPanel ? _growthPanel->growthStepsSpin() : nullptr) {
        connect(growthStepsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
                [this](int value) { applyGrowthSteps(value, true, true); });
    }

    if (QComboBox* growthMethodCombo = _growthPanel ? _growthPanel->growthMethodCombo() : nullptr) {
        connect(growthMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
                [this, growthMethodCombo](int index) {
                    const auto method = static_cast<SegmentationGrowthMethod>(
                        growthMethodCombo->itemData(index).toInt());
                    setGrowthMethod(method);
                });
    }

    if (QSpinBox* extrapPointsSpinControl = _growthPanel ? _growthPanel->extrapolationPointsSpin() : nullptr) {
        connect(extrapPointsSpinControl, QOverload<int>::of(&QSpinBox::valueChanged), this,
                [this](int value) {
                    _extrapolationPointCount = std::clamp(value, 3, 20);
                    writeSetting(QStringLiteral("extrapolation_point_count"), _extrapolationPointCount);
                });
    }

    QComboBox* extrapTypeCombo = _growthPanel ? _growthPanel->extrapolationTypeCombo() : nullptr;
    QWidget* sdtParamsContainer = _growthPanel ? _growthPanel->sdtParamsContainer() : nullptr;
    QWidget* skeletonParamsContainer = _growthPanel ? _growthPanel->skeletonParamsContainer() : nullptr;
    QLabel* extrapPointsLabel = _growthPanel ? _growthPanel->extrapolationPointsLabel() : nullptr;
    QSpinBox* extrapPointsSpin = _growthPanel ? _growthPanel->extrapolationPointsSpin() : nullptr;
    if (extrapTypeCombo) {
        connect(extrapTypeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
                [this, extrapTypeCombo, sdtParamsContainer, skeletonParamsContainer, extrapPointsLabel, extrapPointsSpin](int index) {
                    _extrapolationType = extrapolationTypeFromInt(
                        extrapTypeCombo->itemData(index).toInt());
                    writeSetting(QStringLiteral("extrapolation_type"), static_cast<int>(_extrapolationType));
                    // Show SDT params only when Extrapolation method AND Linear+Fit type
                    if (sdtParamsContainer) {
                        sdtParamsContainer->setVisible(
                            _growthMethod == SegmentationGrowthMethod::Extrapolation &&
                            _extrapolationType == ExtrapolationType::LinearFit);
                    }
                    // Show skeleton params only when Extrapolation method AND SkeletonPath type
                    if (skeletonParamsContainer) {
                        skeletonParamsContainer->setVisible(
                            _growthMethod == SegmentationGrowthMethod::Extrapolation &&
                            _extrapolationType == ExtrapolationType::SkeletonPath);
                    }
                    // Hide fit points label and spinbox for SkeletonPath (it doesn't use polynomial fitting)
                    bool showFitPoints = _extrapolationType != ExtrapolationType::SkeletonPath;
                    if (extrapPointsLabel) {
                        extrapPointsLabel->setVisible(showFitPoints);
                    }
                    if (extrapPointsSpin) {
                        extrapPointsSpin->setVisible(showFitPoints);
                    }
                });
    }

    // SDT/Newton refinement parameter connections
    if (QSpinBox* sdtMaxStepsSpin = _growthPanel ? _growthPanel->sdtMaxStepsSpin() : nullptr) {
        connect(sdtMaxStepsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            _sdtMaxSteps = std::clamp(value, 1, 10);
            writeSetting(QStringLiteral("sdt_max_steps"), _sdtMaxSteps);
        });
    }
    if (QDoubleSpinBox* sdtStepSizeSpin = _growthPanel ? _growthPanel->sdtStepSizeSpin() : nullptr) {
        connect(sdtStepSizeSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
            _sdtStepSize = std::clamp(static_cast<float>(value), 0.1f, 2.0f);
            writeSetting(QStringLiteral("sdt_step_size"), static_cast<double>(_sdtStepSize));
        });
    }
    if (QDoubleSpinBox* sdtConvergenceSpin = _growthPanel ? _growthPanel->sdtConvergenceSpin() : nullptr) {
        connect(sdtConvergenceSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
            _sdtConvergence = std::clamp(static_cast<float>(value), 0.1f, 2.0f);
            writeSetting(QStringLiteral("sdt_convergence"), static_cast<double>(_sdtConvergence));
        });
    }
    if (QSpinBox* sdtChunkSpin = _growthPanel ? _growthPanel->sdtChunkSizeSpin() : nullptr) {
        connect(sdtChunkSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            _sdtChunkSize = std::clamp(value, 32, 256);
            writeSetting(QStringLiteral("sdt_chunk_size"), _sdtChunkSize);
        });
    }

    // Skeleton path parameter connections
    if (QComboBox* skeletonConnCombo = _growthPanel ? _growthPanel->skeletonConnectivityCombo() : nullptr) {
        connect(skeletonConnCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this, skeletonConnCombo](int index) {
            _skeletonConnectivity = skeletonConnCombo->itemData(index).toInt();
            writeSetting(QStringLiteral("skeleton_connectivity"), _skeletonConnectivity);
        });
    }
    if (QComboBox* skeletonSliceCombo = _growthPanel ? _growthPanel->skeletonSliceOrientationCombo() : nullptr) {
        connect(skeletonSliceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this, skeletonSliceCombo](int index) {
            _skeletonSliceOrientation = skeletonSliceCombo->itemData(index).toInt();
            writeSetting(QStringLiteral("skeleton_slice_orientation"), _skeletonSliceOrientation);
        });
    }
    if (QSpinBox* skeletonChunkSpin = _growthPanel ? _growthPanel->skeletonChunkSizeSpin() : nullptr) {
        connect(skeletonChunkSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            _skeletonChunkSize = std::clamp(value, 32, 256);
            writeSetting(QStringLiteral("skeleton_chunk_size"), _skeletonChunkSize);
        });
    }
    if (QSpinBox* skeletonSearchSpin = _growthPanel ? _growthPanel->skeletonSearchRadiusSpin() : nullptr) {
        connect(skeletonSearchSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            _skeletonSearchRadius = std::clamp(value, 1, 100);
            writeSetting(QStringLiteral("skeleton_search_radius"), _skeletonSearchRadius);
        });
    }

    const auto triggerConfiguredGrowth = [this]() {
        const auto allowed = allowedGrowthDirections();
        auto direction = SegmentationGrowthDirection::All;
        if (allowed.size() == 1) {
            direction = allowed.front();
        }
        triggerGrowthRequest(direction, _growthSteps, false);
    };

    if (QPushButton* growButton = _growthPanel ? _growthPanel->growButton() : nullptr) {
        connect(growButton, &QPushButton::clicked, this, triggerConfiguredGrowth);
    }
    if (QPushButton* inpaintButton = _growthPanel ? _growthPanel->inpaintButton() : nullptr) {
        connect(inpaintButton, &QPushButton::clicked, this, [this]() {
            triggerGrowthRequest(SegmentationGrowthDirection::All, 0, true);
        });
    }

    if (QComboBox* volumesCombo = _growthPanel ? _growthPanel->volumesCombo() : nullptr) {
        connect(volumesCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this, volumesCombo](int index) {
            if (index < 0) {
                return;
            }
            const QString volumeId = volumesCombo->itemData(index).toString();
            if (volumeId.isEmpty() || volumeId == _activeVolumeId) {
                return;
            }
            _activeVolumeId = volumeId;
            emit volumeSelectionChanged(volumeId);
        });
    }

    // Forward corrections panel signals
    connect(_correctionsPanel, &SegmentationCorrectionsPanel::correctionsCreateRequested,
            this, &SegmentationWidget::correctionsCreateRequested);
    connect(_correctionsPanel, &SegmentationCorrectionsPanel::correctionsCollectionSelected,
            this, &SegmentationWidget::correctionsCollectionSelected);
    connect(_correctionsPanel, &SegmentationCorrectionsPanel::correctionsAnnotateToggled,
            this, &SegmentationWidget::correctionsAnnotateToggled);

    if (QComboBox* normal3dCombo = _growthPanel ? _growthPanel->normal3dCombo() : nullptr) {
        connect(normal3dCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this, normal3dCombo](int idx) {
            if (_restoringSettings) {
                return;
            }
            if (idx < 0) {
                return;
            }
            const QString path = normal3dCombo->itemData(idx).toString();
            if (path.isEmpty() || path == _normal3dSelectedPath) {
                return;
            }
            _normal3dSelectedPath = path;
            writeSetting(QStringLiteral("normal3d_selected_path"), _normal3dSelectedPath);
            updateNormal3dUi();
        });
    }

    if (QCheckBox* correctionsZRange = _growthPanel ? _growthPanel->correctionsZRangeCheck() : nullptr) {
        connect(correctionsZRange, &QCheckBox::toggled, this, [this](bool enabled) {
            _correctionsZRangeEnabled = enabled;
            writeSetting(QStringLiteral("corrections_z_range_enabled"), _correctionsZRangeEnabled);
            updateGrowthUiState();
            emit correctionsZRangeChanged(enabled, _correctionsZMin, _correctionsZMax);
        });
    }

    if (QSpinBox* correctionsZMinSpin = _growthPanel ? _growthPanel->correctionsZMinSpin() : nullptr) {
        connect(correctionsZMinSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            if (_correctionsZMin == value) {
                return;
            }
            _correctionsZMin = value;
            writeSetting(QStringLiteral("corrections_z_min"), _correctionsZMin);
            if (_correctionsZRangeEnabled) {
                emit correctionsZRangeChanged(true, _correctionsZMin, _correctionsZMax);
            }
        });
    }

    if (QSpinBox* correctionsZMaxSpin = _growthPanel ? _growthPanel->correctionsZMaxSpin() : nullptr) {
        connect(correctionsZMaxSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            if (_correctionsZMax == value) {
                return;
            }
            _correctionsZMax = value;
            writeSetting(QStringLiteral("corrections_z_max"), _correctionsZMax);
            if (_correctionsZRangeEnabled) {
                emit correctionsZRangeChanged(true, _correctionsZMin, _correctionsZMax);
            }
        });
    }


    // Forward neural tracer panel signals
    connect(_neuralTracerPanel, &SegmentationNeuralTracerPanel::neuralTracerEnabledChanged,
            this, &SegmentationWidget::neuralTracerEnabledChanged);
    connect(_neuralTracerPanel, &SegmentationNeuralTracerPanel::neuralTracerStatusMessage,
            this, &SegmentationWidget::neuralTracerStatusMessage);
}

void SegmentationWidget::syncUiState()
{
    if (_headerRow) {
        _headerRow->setEditingChecked(_editingEnabled);
        if (_editingEnabled) {
            _headerRow->setStatusText(_pending ? tr("Editing enabled – pending changes")
                                               : tr("Editing enabled"));
        } else {
            _headerRow->setStatusText(tr("Editing disabled"));
        }
    }

    _editingPanel->syncUiState(_editingEnabled, _growthInProgress);

    _customParamsPanel->syncUiState(_editingEnabled);

    QSpinBox* growthStepsSpin = _growthPanel ? _growthPanel->growthStepsSpin() : nullptr;
    QComboBox* growthMethodCombo = _growthPanel ? _growthPanel->growthMethodCombo() : nullptr;
    QWidget* extrapOptionsPanel = _growthPanel ? _growthPanel->extrapolationOptionsPanel() : nullptr;
    QSpinBox* extrapPointsSpin = _growthPanel ? _growthPanel->extrapolationPointsSpin() : nullptr;
    QComboBox* extrapTypeCombo = _growthPanel ? _growthPanel->extrapolationTypeCombo() : nullptr;
    QWidget* sdtParamsContainer = _growthPanel ? _growthPanel->sdtParamsContainer() : nullptr;
    QSpinBox* sdtMaxStepsSpin = _growthPanel ? _growthPanel->sdtMaxStepsSpin() : nullptr;
    QDoubleSpinBox* sdtStepSizeSpin = _growthPanel ? _growthPanel->sdtStepSizeSpin() : nullptr;
    QDoubleSpinBox* sdtConvergenceSpin = _growthPanel ? _growthPanel->sdtConvergenceSpin() : nullptr;
    QSpinBox* sdtChunkSpin = _growthPanel ? _growthPanel->sdtChunkSizeSpin() : nullptr;
    QWidget* skeletonParamsContainer = _growthPanel ? _growthPanel->skeletonParamsContainer() : nullptr;
    QComboBox* skeletonConnectivityCombo = _growthPanel ? _growthPanel->skeletonConnectivityCombo() : nullptr;
    QComboBox* skeletonSliceCombo = _growthPanel ? _growthPanel->skeletonSliceOrientationCombo() : nullptr;
    QSpinBox* skeletonChunkSpin = _growthPanel ? _growthPanel->skeletonChunkSizeSpin() : nullptr;
    QSpinBox* skeletonSearchSpin = _growthPanel ? _growthPanel->skeletonSearchRadiusSpin() : nullptr;
    QLabel* extrapPointsLabel = _growthPanel ? _growthPanel->extrapolationPointsLabel() : nullptr;

    if (growthStepsSpin) {
        const QSignalBlocker blocker(growthStepsSpin);
        growthStepsSpin->setValue(_growthSteps);
    }

    if (growthMethodCombo) {
        const QSignalBlocker blocker(growthMethodCombo);
        int idx = growthMethodCombo->findData(static_cast<int>(_growthMethod));
        if (idx >= 0) {
            growthMethodCombo->setCurrentIndex(idx);
        }
    }

    if (extrapOptionsPanel) {
        extrapOptionsPanel->setVisible(_growthMethod == SegmentationGrowthMethod::Extrapolation);
    }

    if (extrapPointsSpin) {
        const QSignalBlocker blocker(extrapPointsSpin);
        extrapPointsSpin->setValue(_extrapolationPointCount);
    }

    if (extrapTypeCombo) {
        const QSignalBlocker blocker(extrapTypeCombo);
        int idx = extrapTypeCombo->findData(static_cast<int>(_extrapolationType));
        if (idx >= 0) {
            extrapTypeCombo->setCurrentIndex(idx);
        }
    }

    // SDT params container visibility: only show when Extrapolation method AND Linear+Fit type
    if (sdtParamsContainer) {
        sdtParamsContainer->setVisible(
            _growthMethod == SegmentationGrowthMethod::Extrapolation &&
            _extrapolationType == ExtrapolationType::LinearFit);
    }
    if (sdtMaxStepsSpin) {
        const QSignalBlocker blocker(sdtMaxStepsSpin);
        sdtMaxStepsSpin->setValue(_sdtMaxSteps);
    }
    if (sdtStepSizeSpin) {
        const QSignalBlocker blocker(sdtStepSizeSpin);
        sdtStepSizeSpin->setValue(static_cast<double>(_sdtStepSize));
    }
    if (sdtConvergenceSpin) {
        const QSignalBlocker blocker(sdtConvergenceSpin);
        sdtConvergenceSpin->setValue(static_cast<double>(_sdtConvergence));
    }
    if (sdtChunkSpin) {
        const QSignalBlocker blocker(sdtChunkSpin);
        sdtChunkSpin->setValue(_sdtChunkSize);
    }

    // Skeleton params container visibility: only show when Extrapolation method AND SkeletonPath type
    if (skeletonParamsContainer) {
        skeletonParamsContainer->setVisible(
            _growthMethod == SegmentationGrowthMethod::Extrapolation &&
            _extrapolationType == ExtrapolationType::SkeletonPath);
    }
    if (skeletonConnectivityCombo) {
        const QSignalBlocker blocker(skeletonConnectivityCombo);
        int idx = skeletonConnectivityCombo->findData(_skeletonConnectivity);
        if (idx >= 0) {
            skeletonConnectivityCombo->setCurrentIndex(idx);
        }
    }
    if (skeletonSliceCombo) {
        const QSignalBlocker blocker(skeletonSliceCombo);
        int idx = skeletonSliceCombo->findData(_skeletonSliceOrientation);
        if (idx >= 0) {
            skeletonSliceCombo->setCurrentIndex(idx);
        }
    }
    if (skeletonChunkSpin) {
        const QSignalBlocker blocker(skeletonChunkSpin);
        skeletonChunkSpin->setValue(_skeletonChunkSize);
    }
    if (skeletonSearchSpin) {
        const QSignalBlocker blocker(skeletonSearchSpin);
        skeletonSearchSpin->setValue(_skeletonSearchRadius);
    }
    // Hide fit points label and spinbox for SkeletonPath (it doesn't use polynomial fitting)
    bool showFitPoints = _extrapolationType != ExtrapolationType::SkeletonPath;
    if (extrapPointsLabel) {
        extrapPointsLabel->setVisible(showFitPoints);
    }
    if (extrapPointsSpin) {
        extrapPointsSpin->setVisible(showFitPoints);
    }

    applyGrowthDirectionMaskToUi();
    if (QCheckBox* growthKeybinds = _growthPanel ? _growthPanel->growthKeybindsCheck() : nullptr) {
        const QSignalBlocker blocker(growthKeybinds);
        growthKeybinds->setChecked(_growthKeybindsEnabled);
    }
    _directionFieldPanel->syncUiState(_editingEnabled);

    _correctionsPanel->syncUiState(_editingEnabled, _growthInProgress);
    if (QCheckBox* correctionsZRange = _growthPanel ? _growthPanel->correctionsZRangeCheck() : nullptr) {
        const QSignalBlocker blocker(correctionsZRange);
        correctionsZRange->setChecked(_correctionsZRangeEnabled);
    }
    if (QSpinBox* correctionsZMinSpin = _growthPanel ? _growthPanel->correctionsZMinSpin() : nullptr) {
        const QSignalBlocker blocker(correctionsZMinSpin);
        correctionsZMinSpin->setValue(_correctionsZMin);
    }
    if (QSpinBox* correctionsZMaxSpin = _growthPanel ? _growthPanel->correctionsZMaxSpin() : nullptr) {
        const QSignalBlocker blocker(correctionsZMaxSpin);
        correctionsZMaxSpin->setValue(_correctionsZMax);
    }

    if (QLabel* normalGridLabel = _growthPanel ? _growthPanel->normalGridLabel() : nullptr) {
        const QString icon = _normalGridAvailable
            ? QStringLiteral("<span style=\"color:#2e7d32; font-size:16px;\">&#10003;</span>")
            : QStringLiteral("<span style=\"color:#c62828; font-size:16px;\">&#10007;</span>");
        const bool hasExplicitLocation = !_normalGridDisplayPath.isEmpty() && _normalGridDisplayPath != _normalGridHint;
        QString message;
        message = _normalGridAvailable ? tr("Normal grids found.") : tr("Normal grids not found.");
        if (!_normalGridHint.isEmpty()) {
            message.append(QStringLiteral(" ("));
            message.append(_normalGridHint);
            message.append(QLatin1Char(')'));
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

        normalGridLabel->setText(icon + QStringLiteral("&nbsp;") + message);
        normalGridLabel->setToolTip(tooltip);
        normalGridLabel->setAccessibleDescription(message);
    }

    if (QLineEdit* normalGridPathEdit = _growthPanel ? _growthPanel->normalGridPathEdit() : nullptr) {
        const bool show = _normalGridAvailable && !_normalGridPath.isEmpty();
        normalGridPathEdit->setVisible(show);
        normalGridPathEdit->setText(_normalGridPath);
        normalGridPathEdit->setToolTip(_normalGridPath);
    }

    updateNormal3dUi();

    _approvalMaskPanel->syncUiState();

    _cellReoptPanel->syncUiState(_approvalMaskPanel->showApprovalMask(), _growthInProgress);

    updateGrowthUiState();
}

void SegmentationWidget::updateNormal3dUi()
{
    QLabel* normal3dLabel = _growthPanel ? _growthPanel->normal3dLabel() : nullptr;
    if (!normal3dLabel) {
        return;
    }

    const int count = _normal3dCandidates.size();
    const bool hasAny = count > 0;

    // Keep selection valid.
    if (hasAny) {
        if (_normal3dSelectedPath.isEmpty() || !_normal3dCandidates.contains(_normal3dSelectedPath)) {
            _normal3dSelectedPath = _normal3dCandidates.front();
        }
    } else {
        _normal3dSelectedPath.clear();
    }

    const bool showCombo = count > 1;
    if (QComboBox* normal3dCombo = _growthPanel ? _growthPanel->normal3dCombo() : nullptr) {
        normal3dCombo->setVisible(showCombo);
        normal3dCombo->setEnabled(_editingEnabled && hasAny);
        if (showCombo) {
            const QSignalBlocker blocker(normal3dCombo);
            normal3dCombo->clear();
            for (const QString& p : _normal3dCandidates) {
                normal3dCombo->addItem(p, p);
            }
            const int idx = normal3dCombo->findData(_normal3dSelectedPath);
            if (idx >= 0) {
                normal3dCombo->setCurrentIndex(idx);
            }
        }
    }

    const QString icon = hasAny
        ? QStringLiteral("<span style=\"color:#2e7d32; font-size:16px;\">&#10003;</span>")
        : QStringLiteral("<span style=\"color:#c62828; font-size:16px;\">&#10007;</span>");

    QString message;
    if (!hasAny) {
        message = tr("Normal3D volume not found.");
    } else if (count == 1) {
        message = tr("Normal3D volume found.");
    } else {
        message = tr("Normal3D volumes found (%1). Select one:").arg(count);
    }

    QString tooltip = message;
    if (!_normal3dHint.isEmpty()) {
        tooltip.append(QStringLiteral("\n"));
        tooltip.append(_normal3dHint);
    }
    if (!_volumePackagePath.isEmpty()) {
        tooltip.append(QStringLiteral("\n"));
        tooltip.append(tr("Volume package: %1").arg(_volumePackagePath));
    }

    normal3dLabel->setText(icon + QStringLiteral("&nbsp;") + message);
    normal3dLabel->setToolTip(tooltip);
    normal3dLabel->setAccessibleDescription(message);

    if (QLineEdit* normal3dPathEdit = _growthPanel ? _growthPanel->normal3dPathEdit() : nullptr) {
        const bool show = hasAny && !showCombo;
        normal3dPathEdit->setVisible(show);
        normal3dPathEdit->setText(_normal3dSelectedPath);
        normal3dPathEdit->setToolTip(_normal3dSelectedPath);
    }
}

void SegmentationWidget::setNormal3dZarrCandidates(const QStringList& candidates, const QString& hint)
{
    _normal3dCandidates = candidates;
    _normal3dHint = hint;
    syncUiState();
}

void SegmentationWidget::restoreSettings()
{
    using namespace vc3d::settings;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(settingsGroup());

    _restoringSettings = true;

    _editingPanel->restoreSettings(settings);
    _growthMethod = segmentationGrowthMethodFromInt(
        settings.value(segmentation::GROWTH_METHOD, static_cast<int>(_growthMethod)).toInt());
    _extrapolationPointCount = settings.value(QStringLiteral("extrapolation_point_count"), _extrapolationPointCount).toInt();
    _extrapolationPointCount = std::clamp(_extrapolationPointCount, 3, 20);
    _extrapolationType = extrapolationTypeFromInt(
        settings.value(QStringLiteral("extrapolation_type"), static_cast<int>(_extrapolationType)).toInt());

    // Restore SDT/Newton refinement parameters
    _sdtMaxSteps = std::clamp(settings.value(QStringLiteral("sdt_max_steps"), _sdtMaxSteps).toInt(), 1, 10);
    _sdtStepSize = std::clamp(settings.value(QStringLiteral("sdt_step_size"), static_cast<double>(_sdtStepSize)).toFloat(), 0.1f, 2.0f);
    _sdtConvergence = std::clamp(settings.value(QStringLiteral("sdt_convergence"), static_cast<double>(_sdtConvergence)).toFloat(), 0.1f, 2.0f);
    _sdtChunkSize = std::clamp(settings.value(QStringLiteral("sdt_chunk_size"), _sdtChunkSize).toInt(), 32, 256);

    // Restore skeleton path parameters
    int storedConnectivity = settings.value(QStringLiteral("skeleton_connectivity"), _skeletonConnectivity).toInt();
    if (storedConnectivity == 6 || storedConnectivity == 18 || storedConnectivity == 26) {
        _skeletonConnectivity = storedConnectivity;
    }
    _skeletonSliceOrientation = std::clamp(settings.value(QStringLiteral("skeleton_slice_orientation"), _skeletonSliceOrientation).toInt(), 0, 1);
    _skeletonChunkSize = std::clamp(settings.value(QStringLiteral("skeleton_chunk_size"), _skeletonChunkSize).toInt(), 32, 256);
    _skeletonSearchRadius = std::clamp(settings.value(QStringLiteral("skeleton_search_radius"), _skeletonSearchRadius).toInt(), 1, 100);

    int storedGrowthSteps = settings.value(segmentation::GROWTH_STEPS, _growthSteps).toInt();
    storedGrowthSteps = std::clamp(storedGrowthSteps, 0, 1024);
    _tracerGrowthSteps = settings
                             .value(QStringLiteral("growth_steps_tracer"),
                                    std::max(1, storedGrowthSteps))
                             .toInt();
    _tracerGrowthSteps = std::clamp(_tracerGrowthSteps, 1, 1024);
    applyGrowthSteps(storedGrowthSteps, false, false);
    _growthDirectionMask = normalizeGrowthDirectionMask(
        settings.value(segmentation::GROWTH_DIRECTION_MASK, kGrowDirAllMask).toInt());
    _growthKeybindsEnabled = settings.value(segmentation::GROWTH_KEYBINDS_ENABLED,
                                            segmentation::GROWTH_KEYBINDS_ENABLED_DEFAULT).toBool();

    _directionFieldPanel->restoreSettings(settings);

    _correctionsPanel->restoreSettings(settings);
    _correctionsZRangeEnabled = settings.value(segmentation::CORRECTIONS_Z_RANGE_ENABLED, segmentation::CORRECTIONS_Z_RANGE_ENABLED_DEFAULT).toBool();
    _correctionsZMin = settings.value(segmentation::CORRECTIONS_Z_MIN, segmentation::CORRECTIONS_Z_MIN_DEFAULT).toInt();
   _correctionsZMax = settings.value(segmentation::CORRECTIONS_Z_MAX, _correctionsZMin).toInt();
    if (_correctionsZMax < _correctionsZMin) {
        _correctionsZMax = _correctionsZMin;
    }

    _customParamsPanel->restoreSettings(settings);

    _normal3dSelectedPath = settings.value(QStringLiteral("normal3d_selected_path"), QString()).toString();

    _approvalMaskPanel->restoreSettings(settings);

    _neuralTracerPanel->restoreSettings(settings);

    _cellReoptPanel->restoreSettings(settings);

    settings.endGroup();
    _restoringSettings = false;
}

void SegmentationWidget::writeSetting(const QString& key, const QVariant& value)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
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
    syncUiState();

    if (notifyListeners) {
        emit editingModeChanged(_editingEnabled);
    }
}

// --- Editing panel delegations ---

float SegmentationWidget::dragRadius() const { return _editingPanel->dragRadius(); }
float SegmentationWidget::dragSigma() const { return _editingPanel->dragSigma(); }
float SegmentationWidget::lineRadius() const { return _editingPanel->lineRadius(); }
float SegmentationWidget::lineSigma() const { return _editingPanel->lineSigma(); }
float SegmentationWidget::pushPullRadius() const { return _editingPanel->pushPullRadius(); }
float SegmentationWidget::pushPullSigma() const { return _editingPanel->pushPullSigma(); }
float SegmentationWidget::pushPullStep() const { return _editingPanel->pushPullStep(); }
AlphaPushPullConfig SegmentationWidget::alphaPushPullConfig() const { return _editingPanel->alphaPushPullConfig(); }
float SegmentationWidget::smoothingStrength() const { return _editingPanel->smoothingStrength(); }
int SegmentationWidget::smoothingIterations() const { return _editingPanel->smoothingIterations(); }
bool SegmentationWidget::showHoverMarker() const { return _editingPanel->showHoverMarker(); }

void SegmentationWidget::setDragRadius(float value) { _editingPanel->setDragRadius(value); }
void SegmentationWidget::setDragSigma(float value) { _editingPanel->setDragSigma(value); }
void SegmentationWidget::setLineRadius(float value) { _editingPanel->setLineRadius(value); }
void SegmentationWidget::setLineSigma(float value) { _editingPanel->setLineSigma(value); }
void SegmentationWidget::setPushPullRadius(float value) { _editingPanel->setPushPullRadius(value); }
void SegmentationWidget::setPushPullSigma(float value) { _editingPanel->setPushPullSigma(value); }
void SegmentationWidget::setPushPullStep(float value) { _editingPanel->setPushPullStep(value); }
void SegmentationWidget::setAlphaPushPullConfig(const AlphaPushPullConfig& config) { _editingPanel->setAlphaPushPullConfig(config); }
void SegmentationWidget::setSmoothingStrength(float value) { _editingPanel->setSmoothingStrength(value); }
void SegmentationWidget::setSmoothingIterations(int value) { _editingPanel->setSmoothingIterations(value); }
void SegmentationWidget::setShowHoverMarker(bool enabled) { _editingPanel->setShowHoverMarker(enabled); }

// --- Approval mask delegations ---

bool SegmentationWidget::showApprovalMask() const { return _approvalMaskPanel->showApprovalMask(); }
bool SegmentationWidget::editApprovedMask() const { return _approvalMaskPanel->editApprovedMask(); }
bool SegmentationWidget::editUnapprovedMask() const { return _approvalMaskPanel->editUnapprovedMask(); }
bool SegmentationWidget::autoApproveEdits() const { return _approvalMaskPanel->autoApproveEdits(); }
float SegmentationWidget::approvalBrushRadius() const { return _approvalMaskPanel->approvalBrushRadius(); }
float SegmentationWidget::approvalBrushDepth() const { return _approvalMaskPanel->approvalBrushDepth(); }
int SegmentationWidget::approvalMaskOpacity() const { return _approvalMaskPanel->approvalMaskOpacity(); }
QColor SegmentationWidget::approvalBrushColor() const { return _approvalMaskPanel->approvalBrushColor(); }

void SegmentationWidget::setShowApprovalMask(bool enabled) { _approvalMaskPanel->setShowApprovalMask(enabled); syncUiState(); }
void SegmentationWidget::setEditApprovedMask(bool enabled) { _approvalMaskPanel->setEditApprovedMask(enabled); }
void SegmentationWidget::setEditUnapprovedMask(bool enabled) { _approvalMaskPanel->setEditUnapprovedMask(enabled); }
void SegmentationWidget::setAutoApproveEdits(bool enabled) { _approvalMaskPanel->setAutoApproveEdits(enabled); }
void SegmentationWidget::setApprovalBrushRadius(float radius) { _approvalMaskPanel->setApprovalBrushRadius(radius); }
void SegmentationWidget::setApprovalBrushDepth(float depth) { _approvalMaskPanel->setApprovalBrushDepth(depth); }
void SegmentationWidget::setApprovalMaskOpacity(int opacity) { _approvalMaskPanel->setApprovalMaskOpacity(opacity); }
void SegmentationWidget::setApprovalBrushColor(const QColor& color) { _approvalMaskPanel->setApprovalBrushColor(color); }

// --- Cell reoptimization delegations ---

bool SegmentationWidget::cellReoptMode() const { return _cellReoptPanel->cellReoptMode(); }
int SegmentationWidget::cellReoptMaxSteps() const { return _cellReoptPanel->cellReoptMaxSteps(); }
int SegmentationWidget::cellReoptMaxPoints() const { return _cellReoptPanel->cellReoptMaxPoints(); }
float SegmentationWidget::cellReoptMinSpacing() const { return _cellReoptPanel->cellReoptMinSpacing(); }
float SegmentationWidget::cellReoptPerimeterOffset() const { return _cellReoptPanel->cellReoptPerimeterOffset(); }

void SegmentationWidget::setCellReoptMode(bool enabled) { _cellReoptPanel->setCellReoptMode(enabled); syncUiState(); }
void SegmentationWidget::setCellReoptCollections(const QVector<QPair<uint64_t, QString>>& collections) { _cellReoptPanel->setCellReoptCollections(collections); syncUiState(); }

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

void SegmentationWidget::setGrowthMethod(SegmentationGrowthMethod method)
{
    if (_growthMethod == method) {
        return;
    }
    const int currentSteps = _growthSteps;
    if (method == SegmentationGrowthMethod::Corrections) {
        _tracerGrowthSteps = (currentSteps > 0) ? currentSteps : std::max(1, _tracerGrowthSteps);
    }
    _growthMethod = method;
    int targetSteps = currentSteps;
    if (method == SegmentationGrowthMethod::Corrections) {
        targetSteps = 0;
    } else {
        targetSteps = (currentSteps < 1) ? std::max(1, _tracerGrowthSteps) : std::max(1, currentSteps);
    }
    applyGrowthSteps(targetSteps, true, false);
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

void SegmentationWidget::setNormalGridPath(const QString& path)
{
    _normalGridPath = path.trimmed();
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
    if (QComboBox* volumesCombo = _growthPanel ? _growthPanel->volumesCombo() : nullptr) {
        const QSignalBlocker blocker(volumesCombo);
        volumesCombo->clear();
        for (const auto& entry : _volumeEntries) {
            const QString& id = entry.first;
            const QString& label = entry.second.isEmpty() ? id : entry.second;
            volumesCombo->addItem(label, id);
        }
        int idx = volumesCombo->findData(_activeVolumeId);
        if (idx < 0 && !_volumeEntries.isEmpty()) {
            _activeVolumeId = volumesCombo->itemData(0).toString();
            idx = 0;
        }
        if (idx >= 0) {
            volumesCombo->setCurrentIndex(idx);
        }
        volumesCombo->setEnabled(!_volumeEntries.isEmpty());
    }
}

void SegmentationWidget::setActiveVolume(const QString& volumeId)
{
    if (_activeVolumeId == volumeId) {
        return;
    }
    _activeVolumeId = volumeId;
    if (QComboBox* volumesCombo = _growthPanel ? _growthPanel->volumesCombo() : nullptr) {
        const QSignalBlocker blocker(volumesCombo);
        int idx = volumesCombo->findData(_activeVolumeId);
        if (idx >= 0) {
            volumesCombo->setCurrentIndex(idx);
        }
    }
}

void SegmentationWidget::setCorrectionsEnabled(bool enabled) { _correctionsPanel->setCorrectionsEnabled(enabled); updateGrowthUiState(); }
void SegmentationWidget::setCorrectionsAnnotateChecked(bool enabled) { _correctionsPanel->setCorrectionsAnnotateChecked(enabled); updateGrowthUiState(); }

void SegmentationWidget::setCorrectionCollections(const QVector<QPair<uint64_t, QString>>& collections,
                                                  std::optional<uint64_t> activeId)
{
    _correctionsPanel->setCorrectionCollections(collections, activeId);
    _correctionsPanel->syncUiState(_editingEnabled, _growthInProgress);
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
    return _directionFieldPanel->directionFieldConfigs();
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
    QCheckBox* growthDirUp = _growthPanel ? _growthPanel->growthDirUpCheck() : nullptr;
    QCheckBox* growthDirDown = _growthPanel ? _growthPanel->growthDirDownCheck() : nullptr;
    QCheckBox* growthDirLeft = _growthPanel ? _growthPanel->growthDirLeftCheck() : nullptr;
    QCheckBox* growthDirRight = _growthPanel ? _growthPanel->growthDirRightCheck() : nullptr;
    if (growthDirUp && growthDirUp->isChecked()) {
        mask |= kGrowDirUpBit;
    }
    if (growthDirDown && growthDirDown->isChecked()) {
        mask |= kGrowDirDownBit;
    }
    if (growthDirLeft && growthDirLeft->isChecked()) {
        mask |= kGrowDirLeftBit;
    }
    if (growthDirRight && growthDirRight->isChecked()) {
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
    if (QCheckBox* growthDirUp = _growthPanel ? _growthPanel->growthDirUpCheck() : nullptr) {
        const QSignalBlocker blocker(growthDirUp);
        growthDirUp->setChecked((_growthDirectionMask & kGrowDirUpBit) != 0);
    }
    if (QCheckBox* growthDirDown = _growthPanel ? _growthPanel->growthDirDownCheck() : nullptr) {
        const QSignalBlocker blocker(growthDirDown);
        growthDirDown->setChecked((_growthDirectionMask & kGrowDirDownBit) != 0);
    }
    if (QCheckBox* growthDirLeft = _growthPanel ? _growthPanel->growthDirLeftCheck() : nullptr) {
        const QSignalBlocker blocker(growthDirLeft);
        growthDirLeft->setChecked((_growthDirectionMask & kGrowDirLeftBit) != 0);
    }
    if (QCheckBox* growthDirRight = _growthPanel ? _growthPanel->growthDirRightCheck() : nullptr) {
        const QSignalBlocker blocker(growthDirRight);
        growthDirRight->setChecked((_growthDirectionMask & kGrowDirRightBit) != 0);
    }
}

void SegmentationWidget::updateGrowthUiState()
{
    const bool enableGrowth = _editingEnabled && !_growthInProgress;
    if (QSpinBox* growthStepsSpin = _growthPanel ? _growthPanel->growthStepsSpin() : nullptr) {
        growthStepsSpin->setEnabled(enableGrowth);
    }
    if (QPushButton* growButton = _growthPanel ? _growthPanel->growButton() : nullptr) {
        growButton->setEnabled(enableGrowth);
    }
    if (QPushButton* inpaintButton = _growthPanel ? _growthPanel->inpaintButton() : nullptr) {
        inpaintButton->setEnabled(enableGrowth);
    }
    const bool enableDirCheckbox = enableGrowth;
    if (QCheckBox* growthDirUp = _growthPanel ? _growthPanel->growthDirUpCheck() : nullptr) {
        growthDirUp->setEnabled(enableDirCheckbox);
    }
    if (QCheckBox* growthDirDown = _growthPanel ? _growthPanel->growthDirDownCheck() : nullptr) {
        growthDirDown->setEnabled(enableDirCheckbox);
    }
    if (QCheckBox* growthDirLeft = _growthPanel ? _growthPanel->growthDirLeftCheck() : nullptr) {
        growthDirLeft->setEnabled(enableDirCheckbox);
    }
    if (QCheckBox* growthDirRight = _growthPanel ? _growthPanel->growthDirRightCheck() : nullptr) {
        growthDirRight->setEnabled(enableDirCheckbox);
    }
    _directionFieldPanel->syncUiState(_editingEnabled);

    const bool allowZRange = _editingEnabled && !_growthInProgress;
    if (QCheckBox* correctionsZRange = _growthPanel ? _growthPanel->correctionsZRangeCheck() : nullptr) {
        correctionsZRange->setEnabled(allowZRange);
    }
    if (QSpinBox* correctionsZMinSpin = _growthPanel ? _growthPanel->correctionsZMinSpin() : nullptr) {
        correctionsZMinSpin->setEnabled(allowZRange && _correctionsZRangeEnabled);
    }
    if (QSpinBox* correctionsZMaxSpin = _growthPanel ? _growthPanel->correctionsZMaxSpin() : nullptr) {
        correctionsZMaxSpin->setEnabled(allowZRange && _correctionsZRangeEnabled);
    }
    _customParamsPanel->syncUiState(_editingEnabled);

    _correctionsPanel->syncUiState(_editingEnabled, _growthInProgress);
}

void SegmentationWidget::triggerGrowthRequest(SegmentationGrowthDirection direction,
                                              int steps,
                                              bool inpaintOnly)
{
    if (!_editingEnabled || _growthInProgress) {
        return;
    }

    const SegmentationGrowthMethod method = inpaintOnly
        ? SegmentationGrowthMethod::Tracer
        : _growthMethod;

    const bool allowZeroSteps = inpaintOnly || method == SegmentationGrowthMethod::Corrections;
    const int minSteps = allowZeroSteps ? 0 : 1;
    const int clampedSteps = std::clamp(steps, minSteps, 1024);
    const int finalSteps = clampedSteps;

    qCInfo(lcSegWidget) << "Grow request" << segmentationGrowthMethodToString(method)
                        << segmentationGrowthDirectionToString(direction)
                        << "steps" << finalSteps
                        << "inpaintOnly" << inpaintOnly;
    emit growSurfaceRequested(method, direction, finalSteps, inpaintOnly);
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

// --- Custom params delegations ---

QString SegmentationWidget::customParamsText() const { return _customParamsPanel->customParamsText(); }
QString SegmentationWidget::customParamsProfile() const { return _customParamsPanel->customParamsProfile(); }
bool SegmentationWidget::customParamsValid() const { return _customParamsPanel->customParamsValid(); }
QString SegmentationWidget::customParamsError() const { return _customParamsPanel->customParamsError(); }
std::optional<nlohmann::json> SegmentationWidget::customParamsJson() const { return _customParamsPanel->customParamsJson(); }

// --- Neural tracer delegations ---

bool SegmentationWidget::neuralTracerEnabled() const { return _neuralTracerPanel->neuralTracerEnabled(); }
QString SegmentationWidget::neuralCheckpointPath() const { return _neuralTracerPanel->neuralCheckpointPath(); }
QString SegmentationWidget::neuralPythonPath() const { return _neuralTracerPanel->neuralPythonPath(); }
QString SegmentationWidget::volumeZarrPath() const { return _neuralTracerPanel->volumeZarrPath(); }
int SegmentationWidget::neuralVolumeScale() const { return _neuralTracerPanel->neuralVolumeScale(); }
int SegmentationWidget::neuralBatchSize() const { return _neuralTracerPanel->neuralBatchSize(); }

void SegmentationWidget::setNeuralTracerEnabled(bool enabled) { _neuralTracerPanel->setNeuralTracerEnabled(enabled); }
void SegmentationWidget::setNeuralCheckpointPath(const QString& path) { _neuralTracerPanel->setNeuralCheckpointPath(path); }
void SegmentationWidget::setNeuralPythonPath(const QString& path) { _neuralTracerPanel->setNeuralPythonPath(path); }
void SegmentationWidget::setNeuralVolumeScale(int scale) { _neuralTracerPanel->setNeuralVolumeScale(scale); }
void SegmentationWidget::setNeuralBatchSize(int size) { _neuralTracerPanel->setNeuralBatchSize(size); }
void SegmentationWidget::setVolumeZarrPath(const QString& path) { _neuralTracerPanel->setVolumeZarrPath(path); }
