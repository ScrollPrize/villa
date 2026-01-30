#include "SegmentationWidget.hpp"

#include "elements/CollapsibleSettingsGroup.hpp"
#include "elements/JsonProfileEditor.hpp"
#include "elements/JsonProfilePresets.hpp"
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
#include <QByteArray>
#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QEvent>
#include <QFileDialog>
#include <QGroupBox>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QListWidgetItem>
#include <QLoggingCategory>
#include <QMouseEvent>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollBar>
#include <QSettings>
#include <QSignalBlocker>
#include <QSlider>
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

    _editingPanel = new SegmentationEditingPanel(this);
    layout->addWidget(_editingPanel);

    _approvalMaskPanel = new SegmentationApprovalMaskPanel(this);
    layout->addWidget(_approvalMaskPanel);

    _cellReoptPanel = new SegmentationCellReoptPanel(this);
    layout->addWidget(_cellReoptPanel);

    _directionFieldPanel = new SegmentationDirectionFieldPanel(this);
    layout->addWidget(_directionFieldPanel);

    _neuralTracerPanel = new SegmentationNeuralTracerPanel(this);
    layout->addWidget(_neuralTracerPanel);

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

    rememberGroupState(_editingPanel->editingGroup(), QStringLiteral("group_editing_expanded"));
    rememberGroupState(_editingPanel->dragGroup(), QStringLiteral("group_drag_expanded"));
    rememberGroupState(_editingPanel->lineGroup(), QStringLiteral("group_line_expanded"));
    rememberGroupState(_editingPanel->pushPullGroup(), QStringLiteral("group_push_pull_expanded"));
    rememberGroupState(_directionFieldPanel->directionFieldGroup(), QStringLiteral("group_direction_field_expanded"));
    rememberGroupState(_neuralTracerPanel->neuralTracerGroup(), QStringLiteral("group_neural_tracer_expanded"));

    _correctionsPanel = new SegmentationCorrectionsPanel(this);
    layout->addWidget(_correctionsPanel);

    _customParamsPanel = new SegmentationCustomParamsPanel(this);
    _customParamsEditor = _customParamsPanel->editor();
    layout->addWidget(_customParamsPanel);

    layout->addStretch(1);

    connect(_headerRow, &SegmentationHeaderRow::editingToggled, this, [this](bool enabled) {
        updateEditingState(enabled, true);
    });
    connect(_editingPanel->showHoverMarkerCheck(), &QCheckBox::toggled, this, [this](bool enabled) {
        setShowHoverMarker(enabled);
    });

    // Approval mask signal connections
    connect(_approvalMaskPanel->showCheck(), &QCheckBox::toggled, this, [this](bool enabled) {
        setShowApprovalMask(enabled);
        // If show is being unchecked and edit modes are active, turn them off
        if (!enabled) {
            if (_editApprovedMask) {
                setEditApprovedMask(false);
            }
            if (_editUnapprovedMask) {
                setEditUnapprovedMask(false);
            }
        }
    });

    connect(_approvalMaskPanel->editApprovedCheck(), &QCheckBox::toggled, this, [this](bool enabled) {
        setEditApprovedMask(enabled);
    });

    connect(_approvalMaskPanel->editUnapprovedCheck(), &QCheckBox::toggled, this, [this](bool enabled) {
        setEditUnapprovedMask(enabled);
    });

    connect(_approvalMaskPanel->autoApproveCheck(), &QCheckBox::toggled, this, [this](bool enabled) {
        setAutoApproveEdits(enabled);
    });

    connect(_approvalMaskPanel->brushRadiusSpin(), QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setApprovalBrushRadius(static_cast<float>(value));
    });

    connect(_approvalMaskPanel->brushDepthSpin(), QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setApprovalBrushDepth(static_cast<float>(value));
    });

    connect(_approvalMaskPanel->opacitySlider(), &QSlider::valueChanged, this, [this](int value) {
        setApprovalMaskOpacity(value);
    });

    connect(_approvalMaskPanel->colorButton(), &QPushButton::clicked, this, [this]() {
        QColor newColor = QColorDialog::getColor(_approvalBrushColor, this, tr("Choose Approval Mask Color"));
        if (newColor.isValid()) {
            setApprovalBrushColor(newColor);
        }
    });

    connect(_approvalMaskPanel->undoButton(), &QPushButton::clicked, this, &SegmentationWidget::approvalStrokesUndoRequested);

    // Cell reoptimization signal connections
    connect(_cellReoptPanel->modeCheck(), &QCheckBox::toggled, this, [this](bool enabled) {
        setCellReoptMode(enabled);
    });

    connect(_cellReoptPanel->maxStepsSpin(), QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (_cellReoptMaxSteps != value) {
            _cellReoptMaxSteps = value;
            if (!_restoringSettings) {
                writeSetting(QStringLiteral("cell_reopt_max_steps"), value);
                emit cellReoptMaxStepsChanged(value);
            }
        }
    });

    connect(_cellReoptPanel->maxPointsSpin(), QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (_cellReoptMaxPoints != value) {
            _cellReoptMaxPoints = value;
            if (!_restoringSettings) {
                writeSetting(QStringLiteral("cell_reopt_max_points"), value);
                emit cellReoptMaxPointsChanged(value);
            }
        }
    });

    connect(_cellReoptPanel->minSpacingSpin(), QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        float floatVal = static_cast<float>(value);
        if (_cellReoptMinSpacing != floatVal) {
            _cellReoptMinSpacing = floatVal;
            if (!_restoringSettings) {
                writeSetting(QStringLiteral("cell_reopt_min_spacing"), value);
                emit cellReoptMinSpacingChanged(floatVal);
            }
        }
    });

    connect(_cellReoptPanel->perimeterOffsetSpin(), QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        float floatVal = static_cast<float>(value);
        if (_cellReoptPerimeterOffset != floatVal) {
            _cellReoptPerimeterOffset = floatVal;
            if (!_restoringSettings) {
                writeSetting(QStringLiteral("cell_reopt_perimeter_offset"), value);
                emit cellReoptPerimeterOffsetChanged(floatVal);
            }
        }
    });

    connect(_cellReoptPanel->runButton(), &QPushButton::clicked, this, [this]() {
        uint64_t collectionId = 0;
        auto* combo = _cellReoptPanel->collectionCombo();
        if (combo && combo->currentIndex() >= 0) {
            collectionId = combo->currentData().toULongLong();
        }
        emit cellReoptGrowthRequested(collectionId);
    });

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

    connect(_editingPanel->dragRadiusSpin(), QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setDragRadius(static_cast<float>(value));
        emit dragRadiusChanged(_dragRadiusSteps);
    });

    connect(_editingPanel->dragSigmaSpin(), QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setDragSigma(static_cast<float>(value));
        emit dragSigmaChanged(_dragSigmaSteps);
    });

    connect(_editingPanel->lineRadiusSpin(), QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setLineRadius(static_cast<float>(value));
        emit lineRadiusChanged(_lineRadiusSteps);
    });

    connect(_editingPanel->lineSigmaSpin(), QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setLineSigma(static_cast<float>(value));
        emit lineSigmaChanged(_lineSigmaSteps);
    });

    connect(_editingPanel->pushPullRadiusSpin(), QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setPushPullRadius(static_cast<float>(value));
        emit pushPullRadiusChanged(_pushPullRadiusSteps);
    });

    connect(_editingPanel->pushPullSigmaSpin(), QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setPushPullSigma(static_cast<float>(value));
        emit pushPullSigmaChanged(_pushPullSigmaSteps);
    });

    connect(_editingPanel->pushPullStepSpin(), QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setPushPullStep(static_cast<float>(value));
        emit pushPullStepChanged(_pushPullStep);
    });

    auto onAlphaValueChanged = [this](auto updater) {
        AlphaPushPullConfig config = _alphaPushPullConfig;
        updater(config);
        applyAlphaPushPullConfig(config, true);
    };

    connect(_editingPanel->alphaStartSpin(), QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, onAlphaValueChanged](double value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.start = static_cast<float>(value);
        });
    });
    connect(_editingPanel->alphaStopSpin(), QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, onAlphaValueChanged](double value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.stop = static_cast<float>(value);
        });
    });
    connect(_editingPanel->alphaStepSpin(), QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, onAlphaValueChanged](double value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.step = static_cast<float>(value);
        });
    });
    connect(_editingPanel->alphaLowSpin(), QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, onAlphaValueChanged](double value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.low = displayOpacityToNormalized(value);
        });
    });
    connect(_editingPanel->alphaHighSpin(), QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, onAlphaValueChanged](double value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.high = displayOpacityToNormalized(value);
        });
    });
    connect(_editingPanel->alphaBorderSpin(), QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, onAlphaValueChanged](double value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.borderOffset = static_cast<float>(value);
        });
    });
    connect(_editingPanel->alphaBlurRadiusSpin(), QOverload<int>::of(&QSpinBox::valueChanged), this, [this, onAlphaValueChanged](int value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.blurRadius = value;
        });
    });
    connect(_editingPanel->alphaPerVertexLimitSpin(), QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, onAlphaValueChanged](double value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.perVertexLimit = static_cast<float>(value);
        });
    });
    connect(_editingPanel->alphaPerVertexCheck(), &QCheckBox::toggled, this, [this, onAlphaValueChanged](bool checked) {
        onAlphaValueChanged([checked](AlphaPushPullConfig& cfg) {
            cfg.perVertex = checked;
        });
    });

    connect(_editingPanel->smoothStrengthSpin(), QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setSmoothingStrength(static_cast<float>(value));
        emit smoothingStrengthChanged(_smoothStrength);
    });

    connect(_editingPanel->smoothIterationsSpin(), QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        setSmoothingIterations(value);
        emit smoothingIterationsChanged(_smoothIterations);
    });

    connect(_directionFieldPanel->pathEdit(), &QLineEdit::textChanged, this, [this](const QString& text) {
        _directionFieldPath = text.trimmed();
        if (!_updatingDirectionFieldForm) {
            auto* list = _directionFieldPanel->listWidget();
            applyDirectionFieldDraftToSelection(list ? list->currentRow() : -1);
        }
    });

    connect(_directionFieldPanel->browseButton(), &QToolButton::clicked, this, [this]() {
        const QString initial = _directionFieldPath.isEmpty() ? QDir::homePath() : _directionFieldPath;
        const QString dir = QFileDialog::getExistingDirectory(this, tr("Select direction field"), initial);
        if (dir.isEmpty()) {
            return;
        }
        _directionFieldPath = dir;
        _directionFieldPanel->pathEdit()->setText(dir);
    });

    connect(_directionFieldPanel->orientationCombo(), QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        _directionFieldOrientation = segmentationDirectionFieldOrientationFromInt(
            _directionFieldPanel->orientationCombo()->itemData(index).toInt());
        if (!_updatingDirectionFieldForm) {
            auto* list = _directionFieldPanel->listWidget();
            applyDirectionFieldDraftToSelection(list ? list->currentRow() : -1);
        }
    });

    connect(_directionFieldPanel->scaleCombo(), QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        _directionFieldScale = _directionFieldPanel->scaleCombo()->itemData(index).toInt();
        if (!_updatingDirectionFieldForm) {
            auto* list = _directionFieldPanel->listWidget();
            applyDirectionFieldDraftToSelection(list ? list->currentRow() : -1);
        }
    });

    connect(_directionFieldPanel->weightSpin(), QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        _directionFieldWeight = value;
        if (!_updatingDirectionFieldForm) {
            auto* list = _directionFieldPanel->listWidget();
            applyDirectionFieldDraftToSelection(list ? list->currentRow() : -1);
        }
    });

    connect(_directionFieldPanel->addButton(), &QPushButton::clicked, this, [this]() {
        auto config = buildDirectionFieldDraft();
        if (!config.isValid()) {
            qCInfo(lcSegWidget) << "Ignoring direction field add; path empty";
            return;
        }
        _directionFields.push_back(std::move(config));
        refreshDirectionFieldList();
        persistDirectionFields();
        clearDirectionFieldForm();
    });

    connect(_directionFieldPanel->removeButton(), &QPushButton::clicked, this, [this]() {
        auto* list = _directionFieldPanel->listWidget();
        const int row = list ? list->currentRow() : -1;
        if (row < 0 || row >= static_cast<int>(_directionFields.size())) {
            return;
        }
        _directionFields.erase(_directionFields.begin() + row);
        refreshDirectionFieldList();
        persistDirectionFields();
    });

    connect(_directionFieldPanel->clearButton(), &QPushButton::clicked, this, [this]() {
        clearDirectionFieldForm();
    });

    connect(_directionFieldPanel->listWidget(), &QListWidget::currentRowChanged, this, [this](int row) {
        updateDirectionFieldFormFromSelection(row);
        if (_directionFieldPanel->removeButton()) {
            _directionFieldPanel->removeButton()->setEnabled(_editingEnabled && row >= 0);
        }
    });

    connect(_correctionsPanel->correctionsCombo(), QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0) {
            emit correctionsCollectionSelected(0);
            return;
        }
        const QVariant data = _correctionsPanel->correctionsCombo()->itemData(index);
        emit correctionsCollectionSelected(data.toULongLong());
    });

    connect(_correctionsPanel->correctionsNewButton(), &QPushButton::clicked, this, [this]() {
        emit correctionsCreateRequested();
    });

    connect(_customParamsEditor, &JsonProfileEditor::textChanged, this, [this]() {
        handleCustomParamsEdited();
    });

    connect(_customParamsEditor, &JsonProfileEditor::profileChanged, this, [this](const QString& profile) {
        if (_restoringSettings) {
            return;
        }
        applyCustomParamsProfile(profile, /*persist=*/true, /*fromUi=*/true);
    });

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

    connect(_correctionsPanel->correctionsAnnotateCheck(), &QCheckBox::toggled, this, [this](bool enabled) {
        emit correctionsAnnotateToggled(enabled);
    });

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

    connect(_editingPanel->applyButton(), &QPushButton::clicked, this, &SegmentationWidget::applyRequested);
    connect(_editingPanel->resetButton(), &QPushButton::clicked, this, &SegmentationWidget::resetRequested);
    connect(_editingPanel->stopButton(), &QPushButton::clicked, this, &SegmentationWidget::stopToolsRequested);

    // Neural tracer connections
    connect(_neuralTracerPanel->enabledCheck(), &QCheckBox::toggled, this, [this](bool enabled) {
        setNeuralTracerEnabled(enabled);
    });

    connect(_neuralTracerPanel->checkpointEdit(), &QLineEdit::textChanged, this, [this](const QString& text) {
        _neuralCheckpointPath = text.trimmed();
        writeSetting(QStringLiteral("neural_checkpoint_path"), _neuralCheckpointPath);
    });

    connect(_neuralTracerPanel->checkpointBrowse(), &QToolButton::clicked, this, [this]() {
        const QString initial = _neuralCheckpointPath.isEmpty() ? QDir::homePath() : _neuralCheckpointPath;
        const QString file = QFileDialog::getOpenFileName(this, tr("Select neural tracer checkpoint"),
                                                          initial, tr("PyTorch Checkpoint (*.pt *.pth);;All Files (*)"));
        if (!file.isEmpty()) {
            _neuralCheckpointPath = file;
            _neuralTracerPanel->checkpointEdit()->setText(file);
        }
    });

    connect(_neuralTracerPanel->pythonEdit(), &QLineEdit::textChanged, this, [this](const QString& text) {
        _neuralPythonPath = text.trimmed();
        writeSetting(QStringLiteral("neural_python_path"), _neuralPythonPath);
    });

    connect(_neuralTracerPanel->pythonBrowse(), &QToolButton::clicked, this, [this]() {
        const QString initial = _neuralPythonPath.isEmpty() ? QDir::homePath() : QFileInfo(_neuralPythonPath).absolutePath();
        const QString file = QFileDialog::getOpenFileName(this, tr("Select Python executable"),
                                                          initial, tr("All Files (*)"));
        if (!file.isEmpty()) {
            _neuralPythonPath = file;
            _neuralTracerPanel->pythonEdit()->setText(file);
        }
    });

    connect(_neuralTracerPanel->volumeScaleCombo(), QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        _neuralVolumeScale = _neuralTracerPanel->volumeScaleCombo()->itemData(index).toInt();
        writeSetting(QStringLiteral("neural_volume_scale"), _neuralVolumeScale);
    });

    connect(_neuralTracerPanel->batchSizeSpin(), QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        _neuralBatchSize = value;
        writeSetting(QStringLiteral("neural_batch_size"), _neuralBatchSize);
    });

    // Connect to service manager signals
    auto& serviceManager = NeuralTraceServiceManager::instance();
    connect(&serviceManager, &NeuralTraceServiceManager::statusMessage, this, [this](const QString& message) {
        if (auto* lbl = _neuralTracerPanel->statusLabel()) {
            lbl->setText(message);
            lbl->setVisible(true);
            lbl->setStyleSheet(QString());
        }
        emit neuralTracerStatusMessage(message);
    });
    connect(&serviceManager, &NeuralTraceServiceManager::serviceStarted, this, [this]() {
        if (auto* lbl = _neuralTracerPanel->statusLabel()) {
            lbl->setText(tr("Service running"));
            lbl->setStyleSheet(QStringLiteral("color: #27ae60;"));
        }
    });
    connect(&serviceManager, &NeuralTraceServiceManager::serviceStopped, this, [this]() {
        if (auto* lbl = _neuralTracerPanel->statusLabel()) {
            lbl->setText(tr("Service stopped"));
            lbl->setStyleSheet(QString());
        }
    });
    connect(&serviceManager, &NeuralTraceServiceManager::serviceError, this, [this](const QString& error) {
        if (auto* lbl = _neuralTracerPanel->statusLabel()) {
            lbl->setText(tr("Error: %1").arg(error));
            lbl->setStyleSheet(QStringLiteral("color: #c0392b;"));
            lbl->setVisible(true);
        }
    });
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

    if (_editingPanel->showHoverMarkerCheck()) {
        const QSignalBlocker blocker(_editingPanel->showHoverMarkerCheck());
        _editingPanel->showHoverMarkerCheck()->setChecked(_showHoverMarker);
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

    updateSpin(_editingPanel->dragRadiusSpin(), _dragRadiusSteps);
    updateSpin(_editingPanel->dragSigmaSpin(), _dragSigmaSteps);
    updateSpin(_editingPanel->lineRadiusSpin(), _lineRadiusSteps);
    updateSpin(_editingPanel->lineSigmaSpin(), _lineSigmaSteps);
    updateSpin(_editingPanel->pushPullRadiusSpin(), _pushPullRadiusSteps);
    updateSpin(_editingPanel->pushPullSigmaSpin(), _pushPullSigmaSteps);

    if (_editingPanel->dragGroup()) {
        _editingPanel->dragGroup()->setEnabled(editingActive);
    }
    if (_editingPanel->lineGroup()) {
        _editingPanel->lineGroup()->setEnabled(editingActive);
    }
    if (_editingPanel->pushPullGroup()) {
        _editingPanel->pushPullGroup()->setEnabled(editingActive);
    }

    if (_editingPanel->pushPullStepSpin()) {
        const QSignalBlocker blocker(_editingPanel->pushPullStepSpin());
        _editingPanel->pushPullStepSpin()->setValue(static_cast<double>(_pushPullStep));
        _editingPanel->pushPullStepSpin()->setEnabled(editingActive);
    }

    if (_editingPanel->alphaInfoLabel()) {
        _editingPanel->alphaInfoLabel()->setEnabled(editingActive);
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

    updateAlphaSpin(_editingPanel->alphaStartSpin(), _alphaPushPullConfig.start);
    updateAlphaSpin(_editingPanel->alphaStopSpin(), _alphaPushPullConfig.stop);
    updateAlphaSpin(_editingPanel->alphaStepSpin(), _alphaPushPullConfig.step);
    updateAlphaSpin(_editingPanel->alphaLowSpin(), _alphaPushPullConfig.low, true);
    updateAlphaSpin(_editingPanel->alphaHighSpin(), _alphaPushPullConfig.high, true);
    updateAlphaSpin(_editingPanel->alphaBorderSpin(), _alphaPushPullConfig.borderOffset);

    if (_editingPanel->alphaBlurRadiusSpin()) {
        const QSignalBlocker blocker(_editingPanel->alphaBlurRadiusSpin());
        _editingPanel->alphaBlurRadiusSpin()->setValue(_alphaPushPullConfig.blurRadius);
        _editingPanel->alphaBlurRadiusSpin()->setEnabled(editingActive);
    }
    updateAlphaSpin(_editingPanel->alphaPerVertexLimitSpin(), _alphaPushPullConfig.perVertexLimit);
    if (_editingPanel->alphaPerVertexCheck()) {
        const QSignalBlocker blocker(_editingPanel->alphaPerVertexCheck());
        _editingPanel->alphaPerVertexCheck()->setChecked(_alphaPushPullConfig.perVertex);
        _editingPanel->alphaPerVertexCheck()->setEnabled(editingActive);
    }
    if (_editingPanel->alphaPushPullPanel()) {
        _editingPanel->alphaPushPullPanel()->setEnabled(editingActive);
    }

    if (_editingPanel->smoothStrengthSpin()) {
        const QSignalBlocker blocker(_editingPanel->smoothStrengthSpin());
        _editingPanel->smoothStrengthSpin()->setValue(static_cast<double>(_smoothStrength));
        _editingPanel->smoothStrengthSpin()->setEnabled(editingActive);
    }
    if (_editingPanel->smoothIterationsSpin()) {
        const QSignalBlocker blocker(_editingPanel->smoothIterationsSpin());
        _editingPanel->smoothIterationsSpin()->setValue(_smoothIterations);
        _editingPanel->smoothIterationsSpin()->setEnabled(editingActive);
    }

    if (_customParamsEditor) {
        if (_customParamsEditor->customText() != _customParamsText) {
            _customParamsEditor->setCustomText(_customParamsText);
        }
        if (_customParamsEditor->profile() != _customParamsProfile) {
            const QSignalBlocker blocker(_customParamsEditor);
            _customParamsEditor->setProfile(_customParamsProfile, false);
        }
    }
    validateCustomParamsText();

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
    refreshDirectionFieldList();

    if (auto* pathEdit = _directionFieldPanel->pathEdit()) {
        const QSignalBlocker blocker(pathEdit);
        pathEdit->setText(_directionFieldPath);
    }
    if (auto* orientCombo = _directionFieldPanel->orientationCombo()) {
        const QSignalBlocker blocker(orientCombo);
        int idx = orientCombo->findData(static_cast<int>(_directionFieldOrientation));
        if (idx >= 0) {
            orientCombo->setCurrentIndex(idx);
        }
    }
    if (auto* scaleCombo = _directionFieldPanel->scaleCombo()) {
        const QSignalBlocker blocker(scaleCombo);
        int idx = scaleCombo->findData(_directionFieldScale);
        if (idx >= 0) {
            scaleCombo->setCurrentIndex(idx);
        }
    }
    if (auto* weightSpin = _directionFieldPanel->weightSpin()) {
        const QSignalBlocker blocker(weightSpin);
        weightSpin->setValue(_directionFieldWeight);
    }

    if (auto* combo = _correctionsPanel->correctionsCombo()) {
        const QSignalBlocker blocker(combo);
        combo->setEnabled(_correctionsEnabled && !_growthInProgress && combo->count() > 0);
    }
    if (auto* chk = _correctionsPanel->correctionsAnnotateCheck()) {
        const QSignalBlocker blocker(chk);
        chk->setChecked(_correctionsAnnotateChecked);
    }
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

    // Approval mask checkboxes
    if (auto* chk = _approvalMaskPanel->showCheck()) {
        const QSignalBlocker blocker(chk);
        chk->setChecked(_showApprovalMask);
    }
    if (auto* chk = _approvalMaskPanel->editApprovedCheck()) {
        const QSignalBlocker blocker(chk);
        chk->setChecked(_editApprovedMask);
        // Edit checkboxes only enabled when show is checked
        chk->setEnabled(_showApprovalMask);
    }
    if (auto* chk = _approvalMaskPanel->editUnapprovedCheck()) {
        const QSignalBlocker blocker(chk);
        chk->setChecked(_editUnapprovedMask);
        // Edit checkboxes only enabled when show is checked
        chk->setEnabled(_showApprovalMask);
    }
    if (auto* chk = _approvalMaskPanel->autoApproveCheck()) {
        const QSignalBlocker blocker(chk);
        chk->setChecked(_autoApproveEdits);
    }
    if (auto* slider = _approvalMaskPanel->opacitySlider()) {
        const QSignalBlocker blocker(slider);
        slider->setValue(_approvalMaskOpacity);
    }
    if (auto* lbl = _approvalMaskPanel->opacityLabel()) {
        lbl->setText(QString::number(_approvalMaskOpacity) + QStringLiteral("%"));
    }

    // Cell reoptimization UI state
    if (auto* chk = _cellReoptPanel->modeCheck()) {
        const QSignalBlocker blocker(chk);
        chk->setChecked(_cellReoptMode);
        // Only enabled when approval mask is visible
        chk->setEnabled(_showApprovalMask);
    }
    if (auto* spin = _cellReoptPanel->maxStepsSpin()) {
        const QSignalBlocker blocker(spin);
        spin->setValue(_cellReoptMaxSteps);
        spin->setEnabled(_cellReoptMode);
    }
    if (auto* spin = _cellReoptPanel->maxPointsSpin()) {
        const QSignalBlocker blocker(spin);
        spin->setValue(_cellReoptMaxPoints);
        spin->setEnabled(_cellReoptMode);
    }
    if (auto* spin = _cellReoptPanel->minSpacingSpin()) {
        const QSignalBlocker blocker(spin);
        spin->setValue(static_cast<double>(_cellReoptMinSpacing));
        spin->setEnabled(_cellReoptMode);
    }
    if (auto* spin = _cellReoptPanel->perimeterOffsetSpin()) {
        const QSignalBlocker blocker(spin);
        spin->setValue(static_cast<double>(_cellReoptPerimeterOffset));
        spin->setEnabled(_cellReoptMode);
    }
    if (auto* combo = _cellReoptPanel->collectionCombo()) {
        combo->setEnabled(_cellReoptMode);
    }
    if (auto* btn = _cellReoptPanel->runButton()) {
        auto* combo = _cellReoptPanel->collectionCombo();
        const bool hasCollection = combo && combo->count() > 0;
        btn->setEnabled(_cellReoptMode && !_growthInProgress && hasCollection);
    }

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

QString SegmentationWidget::paramsTextForProfile(const QString& profile) const
{
    if (profile == QStringLiteral("custom")) {
        return _customParamsText;
    }
    if (profile == QStringLiteral("default")) {
        // Empty => use GrowPatch defaults.
        return QString();
    }
    return vc3d::json_profiles::tracerParamProfileJson(profile);
}

void SegmentationWidget::applyCustomParamsProfile(const QString& profile, bool persist, bool fromUi)
{
    const QString normalized = vc3d::json_profiles::isTracerParamProfileId(profile)
        ? profile
        : QStringLiteral("custom");

    _customParamsProfile = normalized;
    if (persist) {
        writeSetting(QStringLiteral("custom_params_profile"), _customParamsProfile);
    }

    if (_customParamsEditor && !fromUi) {
        const QSignalBlocker blocker(_customParamsEditor);
        _customParamsEditor->setProfile(_customParamsProfile, false);
    }

    validateCustomParamsText();
    syncUiState();
}

void SegmentationWidget::restoreSettings()
{
    using namespace vc3d::settings;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(settingsGroup());

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

    QVariantList serialized = settings.value(segmentation::DIRECTION_FIELDS, QVariantList{}).toList();
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

    _correctionsEnabled = settings.value(segmentation::CORRECTIONS_ENABLED, segmentation::CORRECTIONS_ENABLED_DEFAULT).toBool();
    _correctionsZRangeEnabled = settings.value(segmentation::CORRECTIONS_Z_RANGE_ENABLED, segmentation::CORRECTIONS_Z_RANGE_ENABLED_DEFAULT).toBool();
    _correctionsZMin = settings.value(segmentation::CORRECTIONS_Z_MIN, segmentation::CORRECTIONS_Z_MIN_DEFAULT).toInt();
   _correctionsZMax = settings.value(segmentation::CORRECTIONS_Z_MAX, _correctionsZMin).toInt();
    if (_correctionsZMax < _correctionsZMin) {
        _correctionsZMax = _correctionsZMin;
    }

    _customParamsText = settings.value(segmentation::CUSTOM_PARAMS_TEXT, QString()).toString();
    _customParamsProfile = settings.value(QStringLiteral("custom_params_profile"), _customParamsProfile).toString();
    if (!vc3d::json_profiles::isTracerParamProfileId(_customParamsProfile)) {
        _customParamsProfile = QStringLiteral("custom");
    }
    if (_customParamsEditor) {
        _customParamsEditor->setCustomText(_customParamsText);
    }

    _normal3dSelectedPath = settings.value(QStringLiteral("normal3d_selected_path"), QString()).toString();

    // Apply profile behavior after restoring.
    applyCustomParamsProfile(_customParamsProfile, /*persist=*/false, /*fromUi=*/false);

    _approvalBrushRadius = settings.value(segmentation::APPROVAL_BRUSH_RADIUS, _approvalBrushRadius).toFloat();
    _approvalBrushRadius = std::clamp(_approvalBrushRadius, 1.0f, 1000.0f);
    _approvalBrushDepth = settings.value(segmentation::APPROVAL_BRUSH_DEPTH, _approvalBrushDepth).toFloat();
    _approvalBrushDepth = std::clamp(_approvalBrushDepth, 1.0f, 500.0f);
    // Don't restore approval mask show/edit states - user must explicitly enable each session

    _approvalMaskOpacity = settings.value(segmentation::APPROVAL_MASK_OPACITY, _approvalMaskOpacity).toInt();
    _approvalMaskOpacity = std::clamp(_approvalMaskOpacity, 0, 100);
    const QString colorName = settings.value(segmentation::APPROVAL_BRUSH_COLOR, _approvalBrushColor.name()).toString();
    if (QColor::isValidColorName(colorName)) {
        _approvalBrushColor = QColor::fromString(colorName);
    }
    _showApprovalMask = settings.value(segmentation::SHOW_APPROVAL_MASK, _showApprovalMask).toBool();
    _autoApproveEdits = settings.value(segmentation::APPROVAL_AUTO_APPROVE_EDITS, _autoApproveEdits).toBool();
    // Don't restore edit states - user must explicitly enable editing each session

    // Neural tracer settings
    _neuralTracerEnabled = settings.value(QStringLiteral("neural_tracer_enabled"), false).toBool();
    _neuralCheckpointPath = settings.value(QStringLiteral("neural_checkpoint_path"), QString()).toString();
    _neuralPythonPath = settings.value(QStringLiteral("neural_python_path"), QString()).toString();
    _neuralVolumeScale = settings.value(QStringLiteral("neural_volume_scale"), 0).toInt();
    _neuralVolumeScale = std::clamp(_neuralVolumeScale, 0, 5);
    _neuralBatchSize = settings.value(QStringLiteral("neural_batch_size"), 4).toInt();
    _neuralBatchSize = std::clamp(_neuralBatchSize, 1, 64);
  
    // Cell reoptimization settings
    _cellReoptMaxSteps = settings.value(QStringLiteral("cell_reopt_max_steps"), _cellReoptMaxSteps).toInt();
    _cellReoptMaxSteps = std::clamp(_cellReoptMaxSteps, 10, 10000);
    _cellReoptMaxPoints = settings.value(QStringLiteral("cell_reopt_max_points"), _cellReoptMaxPoints).toInt();
    _cellReoptMaxPoints = std::clamp(_cellReoptMaxPoints, 3, 200);
    _cellReoptMinSpacing = settings.value(QStringLiteral("cell_reopt_min_spacing"), static_cast<double>(_cellReoptMinSpacing)).toFloat();
    _cellReoptMinSpacing = std::clamp(_cellReoptMinSpacing, 1.0f, 50.0f);
    _cellReoptPerimeterOffset = settings.value(QStringLiteral("cell_reopt_perimeter_offset"), static_cast<double>(_cellReoptPerimeterOffset)).toFloat();
    _cellReoptPerimeterOffset = std::clamp(_cellReoptPerimeterOffset, -50.0f, 50.0f);
    // Don't restore cell reopt mode - user must explicitly enable each session

    const bool editingExpanded = settings.value(segmentation::GROUP_EDITING_EXPANDED, segmentation::GROUP_EDITING_EXPANDED_DEFAULT).toBool();
    const bool dragExpanded = settings.value(segmentation::GROUP_DRAG_EXPANDED, segmentation::GROUP_DRAG_EXPANDED_DEFAULT).toBool();
    const bool lineExpanded = settings.value(segmentation::GROUP_LINE_EXPANDED, segmentation::GROUP_LINE_EXPANDED_DEFAULT).toBool();
    const bool pushPullExpanded = settings.value(segmentation::GROUP_PUSH_PULL_EXPANDED, segmentation::GROUP_PUSH_PULL_EXPANDED_DEFAULT).toBool();
    const bool directionExpanded = settings.value(segmentation::GROUP_DIRECTION_FIELD_EXPANDED, segmentation::GROUP_DIRECTION_FIELD_EXPANDED_DEFAULT).toBool();

    if (_editingPanel->editingGroup()) {
        _editingPanel->editingGroup()->setExpanded(editingExpanded);
    }
    if (_editingPanel->dragGroup()) {
        _editingPanel->dragGroup()->setExpanded(dragExpanded);
    }
    if (_editingPanel->lineGroup()) {
        _editingPanel->lineGroup()->setExpanded(lineExpanded);
    }
    if (_editingPanel->pushPullGroup()) {
        _editingPanel->pushPullGroup()->setExpanded(pushPullExpanded);
    }
    if (auto* dirGroup = _directionFieldPanel->directionFieldGroup()) {
        dirGroup->setExpanded(directionExpanded);
    }

    const bool neuralExpanded = settings.value(QStringLiteral("group_neural_tracer_expanded"), false).toBool();
    if (auto* neuralGroup = _neuralTracerPanel->neuralTracerGroup()) {
        neuralGroup->setExpanded(neuralExpanded);
    }

    // Sync neural tracer UI
    if (auto* chk = _neuralTracerPanel->enabledCheck()) {
        const QSignalBlocker blocker(chk);
        chk->setChecked(_neuralTracerEnabled);
    }
    if (auto* edit = _neuralTracerPanel->checkpointEdit()) {
        const QSignalBlocker blocker(edit);
        edit->setText(_neuralCheckpointPath);
    }
    if (auto* edit = _neuralTracerPanel->pythonEdit()) {
        const QSignalBlocker blocker(edit);
        edit->setText(_neuralPythonPath);
    }
    if (auto* combo = _neuralTracerPanel->volumeScaleCombo()) {
        const QSignalBlocker blocker(combo);
        int idx = combo->findData(_neuralVolumeScale);
        if (idx >= 0) {
            combo->setCurrentIndex(idx);
        }
    }
    if (auto* spin = _neuralTracerPanel->batchSizeSpin()) {
        const QSignalBlocker blocker(spin);
        spin->setValue(_neuralBatchSize);
    }

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

void SegmentationWidget::setShowHoverMarker(bool enabled)
{
    if (_showHoverMarker == enabled) {
        return;
    }
    _showHoverMarker = enabled;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("show_hover_marker"), _showHoverMarker);
        emit hoverMarkerToggled(_showHoverMarker);
    }
    if (_editingPanel->showHoverMarkerCheck()) {
        const QSignalBlocker blocker(_editingPanel->showHoverMarkerCheck());
        _editingPanel->showHoverMarkerCheck()->setChecked(_showHoverMarker);
    }
}

void SegmentationWidget::setShowApprovalMask(bool enabled)
{
    if (_showApprovalMask == enabled) {
        return;
    }
    _showApprovalMask = enabled;
    qInfo() << "SegmentationWidget: Show approval mask changed to:" << enabled;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("show_approval_mask"), _showApprovalMask);
        qInfo() << "  Emitting showApprovalMaskChanged signal";
        emit showApprovalMaskChanged(_showApprovalMask);
    }
    if (auto* chk = _approvalMaskPanel->showCheck()) {
        const QSignalBlocker blocker(chk);
        chk->setChecked(_showApprovalMask);
    }
    syncUiState();
}

void SegmentationWidget::setEditApprovedMask(bool enabled)
{
    if (_editApprovedMask == enabled) {
        return;
    }
    _editApprovedMask = enabled;
    qInfo() << "SegmentationWidget: Edit approved mask changed to:" << enabled;

    // Mutual exclusion: if enabling approved, disable unapproved
    if (enabled && _editUnapprovedMask) {
        setEditUnapprovedMask(false);
    }

    if (!_restoringSettings) {
        writeSetting(QStringLiteral("edit_approved_mask"), _editApprovedMask);
        qInfo() << "  Emitting editApprovedMaskChanged signal";
        emit editApprovedMaskChanged(_editApprovedMask);
    }
    if (auto* chk = _approvalMaskPanel->editApprovedCheck()) {
        const QSignalBlocker blocker(chk);
        chk->setChecked(_editApprovedMask);
    }
    syncUiState();
}

void SegmentationWidget::setEditUnapprovedMask(bool enabled)
{
    if (_editUnapprovedMask == enabled) {
        return;
    }
    _editUnapprovedMask = enabled;
    qInfo() << "SegmentationWidget: Edit unapproved mask changed to:" << enabled;

    // Mutual exclusion: if enabling unapproved, disable approved
    if (enabled && _editApprovedMask) {
        setEditApprovedMask(false);
    }

    if (!_restoringSettings) {
        writeSetting(QStringLiteral("edit_unapproved_mask"), _editUnapprovedMask);
        qInfo() << "  Emitting editUnapprovedMaskChanged signal";
        emit editUnapprovedMaskChanged(_editUnapprovedMask);
    }
    if (auto* chk = _approvalMaskPanel->editUnapprovedCheck()) {
        const QSignalBlocker blocker(chk);
        chk->setChecked(_editUnapprovedMask);
    }
    syncUiState();
}

void SegmentationWidget::setAutoApproveEdits(bool enabled)
{
    if (_autoApproveEdits == enabled) {
        return;
    }
    _autoApproveEdits = enabled;
    qInfo() << "SegmentationWidget: Auto-approve edits changed to:" << enabled;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("approval_auto_approve_edits"), _autoApproveEdits);
        emit autoApproveEditsChanged(_autoApproveEdits);
    }
    if (auto* chk = _approvalMaskPanel->autoApproveCheck()) {
        const QSignalBlocker blocker(chk);
        chk->setChecked(_autoApproveEdits);
    }
}

void SegmentationWidget::setApprovalBrushRadius(float radius)
{
    const float sanitized = std::clamp(radius, 1.0f, 1000.0f);
    if (std::abs(_approvalBrushRadius - sanitized) < 1e-4f) {
        return;
    }
    _approvalBrushRadius = sanitized;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("approval_brush_radius"), _approvalBrushRadius);
        emit approvalBrushRadiusChanged(_approvalBrushRadius);
    }
    if (auto* spin = _approvalMaskPanel->brushRadiusSpin()) {
        const QSignalBlocker blocker(spin);
        spin->setValue(static_cast<double>(_approvalBrushRadius));
    }
}

void SegmentationWidget::setApprovalBrushDepth(float depth)
{
    const float sanitized = std::clamp(depth, 1.0f, 500.0f);
    if (std::abs(_approvalBrushDepth - sanitized) < 1e-4f) {
        return;
    }
    _approvalBrushDepth = sanitized;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("approval_brush_depth"), _approvalBrushDepth);
        emit approvalBrushDepthChanged(_approvalBrushDepth);
    }
    if (auto* spin = _approvalMaskPanel->brushDepthSpin()) {
        const QSignalBlocker blocker(spin);
        spin->setValue(static_cast<double>(_approvalBrushDepth));
    }
}

void SegmentationWidget::setApprovalMaskOpacity(int opacity)
{
    const int sanitized = std::clamp(opacity, 0, 100);
    if (_approvalMaskOpacity == sanitized) {
        return;
    }
    _approvalMaskOpacity = sanitized;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("approval_mask_opacity"), _approvalMaskOpacity);
        emit approvalMaskOpacityChanged(_approvalMaskOpacity);
    }
    if (auto* slider = _approvalMaskPanel->opacitySlider()) {
        const QSignalBlocker blocker(slider);
        slider->setValue(_approvalMaskOpacity);
    }
    if (auto* lbl = _approvalMaskPanel->opacityLabel()) {
        lbl->setText(QString::number(_approvalMaskOpacity) + QStringLiteral("%"));
    }
}

void SegmentationWidget::setApprovalBrushColor(const QColor& color)
{
    if (!color.isValid() || _approvalBrushColor == color) {
        return;
    }
    _approvalBrushColor = color;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("approval_brush_color"), _approvalBrushColor.name());
        emit approvalBrushColorChanged(_approvalBrushColor);
    }
    if (auto* btn = _approvalMaskPanel->colorButton()) {
        btn->setStyleSheet(
            QStringLiteral("background-color: %1; border: 1px solid #888;").arg(_approvalBrushColor.name()));
    }
}

void SegmentationWidget::setCellReoptMode(bool enabled)
{
    if (_cellReoptMode == enabled) {
        return;
    }
    _cellReoptMode = enabled;
    qInfo() << "SegmentationWidget: Cell reoptimization mode changed to:" << enabled;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("cell_reopt_mode"), _cellReoptMode);
        emit cellReoptModeChanged(_cellReoptMode);
    }
    if (auto* chk = _cellReoptPanel->modeCheck()) {
        const QSignalBlocker blocker(chk);
        chk->setChecked(_cellReoptMode);
    }
    syncUiState();
}

void SegmentationWidget::setCellReoptCollections(const QVector<QPair<uint64_t, QString>>& collections)
{
    auto* combo = _cellReoptPanel->collectionCombo();
    if (!combo) {
        return;
    }

    // Remember current selection
    uint64_t currentId = 0;
    if (combo->currentIndex() >= 0) {
        currentId = combo->currentData().toULongLong();
    }

    const QSignalBlocker blocker(combo);
    combo->clear();

    int indexToSelect = -1;
    for (int i = 0; i < collections.size(); ++i) {
        const auto& [id, name] = collections[i];
        combo->addItem(name, QVariant::fromValue(id));
        if (id == currentId) {
            indexToSelect = i;
        }
    }

    // Restore selection if possible, otherwise select first item
    if (indexToSelect >= 0) {
        combo->setCurrentIndex(indexToSelect);
    } else if (combo->count() > 0) {
        combo->setCurrentIndex(0);
    }

    // Update run button state - need a collection selected to run
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
    if (_editingPanel->dragRadiusSpin()) {
        const QSignalBlocker blocker(_editingPanel->dragRadiusSpin());
        _editingPanel->dragRadiusSpin()->setValue(static_cast<double>(_dragRadiusSteps));
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
    if (_editingPanel->dragSigmaSpin()) {
        const QSignalBlocker blocker(_editingPanel->dragSigmaSpin());
        _editingPanel->dragSigmaSpin()->setValue(static_cast<double>(_dragSigmaSteps));
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
    if (_editingPanel->lineRadiusSpin()) {
        const QSignalBlocker blocker(_editingPanel->lineRadiusSpin());
        _editingPanel->lineRadiusSpin()->setValue(static_cast<double>(_lineRadiusSteps));
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
    if (_editingPanel->lineSigmaSpin()) {
        const QSignalBlocker blocker(_editingPanel->lineSigmaSpin());
        _editingPanel->lineSigmaSpin()->setValue(static_cast<double>(_lineSigmaSteps));
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
    if (_editingPanel->pushPullRadiusSpin()) {
        const QSignalBlocker blocker(_editingPanel->pushPullRadiusSpin());
        _editingPanel->pushPullRadiusSpin()->setValue(static_cast<double>(_pushPullRadiusSteps));
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
    if (_editingPanel->pushPullSigmaSpin()) {
        const QSignalBlocker blocker(_editingPanel->pushPullSigmaSpin());
        _editingPanel->pushPullSigmaSpin()->setValue(static_cast<double>(_pushPullSigmaSteps));
    }
}

void SegmentationWidget::setPushPullStep(float value)
{
    const float clamped = std::clamp(value, 0.05f, 40.0f);
    if (std::fabs(clamped - _pushPullStep) < 1e-4f) {
        return;
    }
    _pushPullStep = clamped;
    writeSetting(QStringLiteral("push_pull_step"), _pushPullStep);
    if (_editingPanel->pushPullStepSpin()) {
        const QSignalBlocker blocker(_editingPanel->pushPullStepSpin());
        _editingPanel->pushPullStepSpin()->setValue(static_cast<double>(_pushPullStep));
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

    if (_editingPanel->alphaStartSpin()) {
        const QSignalBlocker blocker(_editingPanel->alphaStartSpin());
        _editingPanel->alphaStartSpin()->setValue(static_cast<double>(_alphaPushPullConfig.start));
        _editingPanel->alphaStartSpin()->setEnabled(editingActive);
    }
    if (_editingPanel->alphaStopSpin()) {
        const QSignalBlocker blocker(_editingPanel->alphaStopSpin());
        _editingPanel->alphaStopSpin()->setValue(static_cast<double>(_alphaPushPullConfig.stop));
        _editingPanel->alphaStopSpin()->setEnabled(editingActive);
    }
    if (_editingPanel->alphaStepSpin()) {
        const QSignalBlocker blocker(_editingPanel->alphaStepSpin());
        _editingPanel->alphaStepSpin()->setValue(static_cast<double>(_alphaPushPullConfig.step));
        _editingPanel->alphaStepSpin()->setEnabled(editingActive);
    }
    if (_editingPanel->alphaLowSpin()) {
        const QSignalBlocker blocker(_editingPanel->alphaLowSpin());
        _editingPanel->alphaLowSpin()->setValue(normalizedOpacityToDisplay(_alphaPushPullConfig.low));
        _editingPanel->alphaLowSpin()->setEnabled(editingActive);
    }
    if (_editingPanel->alphaHighSpin()) {
        const QSignalBlocker blocker(_editingPanel->alphaHighSpin());
        _editingPanel->alphaHighSpin()->setValue(normalizedOpacityToDisplay(_alphaPushPullConfig.high));
        _editingPanel->alphaHighSpin()->setEnabled(editingActive);
    }
    if (_editingPanel->alphaBorderSpin()) {
        const QSignalBlocker blocker(_editingPanel->alphaBorderSpin());
        _editingPanel->alphaBorderSpin()->setValue(static_cast<double>(_alphaPushPullConfig.borderOffset));
        _editingPanel->alphaBorderSpin()->setEnabled(editingActive);
    }
    if (_editingPanel->alphaBlurRadiusSpin()) {
        const QSignalBlocker blocker(_editingPanel->alphaBlurRadiusSpin());
        _editingPanel->alphaBlurRadiusSpin()->setValue(_alphaPushPullConfig.blurRadius);
        _editingPanel->alphaBlurRadiusSpin()->setEnabled(editingActive);
    }
    if (_editingPanel->alphaPerVertexLimitSpin()) {
        const QSignalBlocker blocker(_editingPanel->alphaPerVertexLimitSpin());
        _editingPanel->alphaPerVertexLimitSpin()->setValue(static_cast<double>(_alphaPushPullConfig.perVertexLimit));
        _editingPanel->alphaPerVertexLimitSpin()->setEnabled(editingActive);
    }
    if (_editingPanel->alphaPerVertexCheck()) {
        const QSignalBlocker blocker(_editingPanel->alphaPerVertexCheck());
        _editingPanel->alphaPerVertexCheck()->setChecked(_alphaPushPullConfig.perVertex);
        _editingPanel->alphaPerVertexCheck()->setEnabled(editingActive);
    }
    if (_editingPanel->alphaPushPullPanel()) {
        _editingPanel->alphaPushPullPanel()->setEnabled(editingActive);
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
    if (_editingPanel->smoothStrengthSpin()) {
        const QSignalBlocker blocker(_editingPanel->smoothStrengthSpin());
        _editingPanel->smoothStrengthSpin()->setValue(static_cast<double>(_smoothStrength));
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
    if (_editingPanel->smoothIterationsSpin()) {
        const QSignalBlocker blocker(_editingPanel->smoothIterationsSpin());
        _editingPanel->smoothIterationsSpin()->setValue(_smoothIterations);
    }
}

void SegmentationWidget::handleCustomParamsEdited()
{
    if (!_customParamsEditor) {
        return;
    }

    // Edits only allowed in custom profile (UI should already be read-only otherwise).
    if (_customParamsProfile != QStringLiteral("custom")) {
        return;
    }

    _customParamsText = _customParamsEditor->customText();
    writeSetting(QStringLiteral("custom_params_text"), _customParamsText);
    validateCustomParamsText();
}

void SegmentationWidget::validateCustomParamsText()
{
    if (_customParamsEditor) {
        _customParamsError = _customParamsEditor->errorText();
        return;
    }

    QString error;
    parseCustomParams(&error);
    _customParamsError = error;
}

std::optional<nlohmann::json> SegmentationWidget::parseCustomParams(QString* error) const
{
    if (error) {
        error->clear();
    }

    const QString trimmed = paramsTextForProfile(_customParamsProfile).trimmed();
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

void SegmentationWidget::setCorrectionsEnabled(bool enabled)
{
    if (_correctionsEnabled == enabled) {
        return;
    }
    _correctionsEnabled = enabled;
    writeSetting(QStringLiteral("corrections_enabled"), _correctionsEnabled);
    if (!enabled) {
        _correctionsAnnotateChecked = false;
        if (auto* chk = _correctionsPanel->correctionsAnnotateCheck()) {
            const QSignalBlocker blocker(chk);
            chk->setChecked(false);
        }
    }
    updateGrowthUiState();
}

void SegmentationWidget::setCorrectionsAnnotateChecked(bool enabled)
{
    _correctionsAnnotateChecked = enabled;
    if (auto* chk = _correctionsPanel->correctionsAnnotateCheck()) {
        const QSignalBlocker blocker(chk);
        chk->setChecked(enabled);
    }
    updateGrowthUiState();
}

void SegmentationWidget::setCorrectionCollections(const QVector<QPair<uint64_t, QString>>& collections,
                                                  std::optional<uint64_t> activeId)
{
    auto* combo = _correctionsPanel->correctionsCombo();
    if (!combo) {
        return;
    }
    const QSignalBlocker blocker(combo);
    combo->clear();
    for (const auto& pair : collections) {
        combo->addItem(pair.second, QVariant::fromValue(static_cast<qulonglong>(pair.first)));
    }
    if (activeId) {
        int idx = combo->findData(QVariant::fromValue(static_cast<qulonglong>(*activeId)));
        if (idx >= 0) {
            combo->setCurrentIndex(idx);
        }
    } else {
        combo->setCurrentIndex(-1);
    }
    combo->setEnabled(_correctionsEnabled && !_growthInProgress && combo->count() > 0);
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
    auto* list = _directionFieldPanel->listWidget();
    if (!list) {
        return;
    }
    const QSignalBlocker blocker(list);
    const int previousRow = list->currentRow();
    list->clear();

    for (const auto& config : _directionFields) {
        QString orientationLabel = segmentationDirectionFieldOrientationKey(config.orientation);
        const QString weightText = QString::number(std::clamp(config.weight, 0.0, 10.0), 'f', 2);
        const QString itemText = tr("%1 — %2 (scale %3, weight %4)")
                                     .arg(config.path,
                                          orientationLabel,
                                          QString::number(std::clamp(config.scale, 0, 5)),
                                          weightText);
        auto* item = new QListWidgetItem(itemText, list);
        item->setToolTip(config.path);
    }

    if (!_directionFields.empty()) {
        const int clampedRow = std::clamp(previousRow, 0, static_cast<int>(_directionFields.size()) - 1);
        list->setCurrentRow(clampedRow);
    }
    if (auto* removeBtn = _directionFieldPanel->removeButton()) {
        removeBtn->setEnabled(_editingEnabled && !_directionFields.empty() && list->currentRow() >= 0);
    }

    updateDirectionFieldFormFromSelection(list->currentRow());
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

    if (auto* pathEdit = _directionFieldPanel->pathEdit()) {
        const QSignalBlocker blocker(pathEdit);
        pathEdit->setText(_directionFieldPath);
    }
    if (auto* orientCombo = _directionFieldPanel->orientationCombo()) {
        const QSignalBlocker blocker(orientCombo);
        int idx = orientCombo->findData(static_cast<int>(_directionFieldOrientation));
        if (idx >= 0) {
            orientCombo->setCurrentIndex(idx);
        }
    }
    if (auto* scaleCombo = _directionFieldPanel->scaleCombo()) {
        const QSignalBlocker blocker(scaleCombo);
        int idx = scaleCombo->findData(_directionFieldScale);
        if (idx >= 0) {
            scaleCombo->setCurrentIndex(idx);
        }
    }
    if (auto* weightSpin = _directionFieldPanel->weightSpin()) {
        const QSignalBlocker blocker(weightSpin);
        weightSpin->setValue(_directionFieldWeight);
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
    auto* list = _directionFieldPanel->listWidget();
    if (!list) {
        return;
    }
    if (row < 0 || row >= list->count()) {
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

    if (auto* item = list->item(row)) {
        item->setText(itemText);
        item->setToolTip(config.path);
    }
}

void SegmentationWidget::updateDirectionFieldListGeometry()
{
    auto* list = _directionFieldPanel->listWidget();
    if (!list) {
        return;
    }

    auto policy = list->sizePolicy();
    const int itemCount = list->count();

    if (itemCount <= kCompactDirectionFieldRowLimit) {
        const int sampleRowHeight = list->sizeHintForRow(0);
        const int rowHeight = sampleRowHeight > 0 ? sampleRowHeight : list->fontMetrics().height() + 8;
        const int visibleRows = std::max(1, itemCount);
        const int frameHeight = 2 * list->frameWidth();
        const auto* hScroll = list->horizontalScrollBar();
        const int scrollHeight = (hScroll && hScroll->isVisible()) ? hScroll->sizeHint().height() : 0;
        const int targetHeight = rowHeight * visibleRows + frameHeight + scrollHeight;

        policy.setVerticalPolicy(QSizePolicy::Fixed);
        policy.setVerticalStretch(0);
        list->setSizePolicy(policy);
        list->setMinimumHeight(targetHeight);
        list->setMaximumHeight(targetHeight);
    } else {
        policy.setVerticalPolicy(QSizePolicy::Expanding);
        policy.setVerticalStretch(1);
        list->setSizePolicy(policy);
        list->setMinimumHeight(0);
        list->setMaximumHeight(QWIDGETSIZE_MAX);
    }

    list->updateGeometry();
}

void SegmentationWidget::clearDirectionFieldForm()
{
    // Clear the list selection
    if (auto* list = _directionFieldPanel->listWidget()) {
        list->setCurrentRow(-1);
    }

    // Reset member variables to defaults
    _directionFieldPath.clear();
    _directionFieldOrientation = SegmentationDirectionFieldOrientation::Normal;
    _directionFieldScale = 0;
    _directionFieldWeight = 1.0;

    // Update the form fields to reflect the cleared state
    const bool previousUpdating = _updatingDirectionFieldForm;
    _updatingDirectionFieldForm = true;

    if (auto* pathEdit = _directionFieldPanel->pathEdit()) {
        pathEdit->clear();
    }
    if (auto* orientCombo = _directionFieldPanel->orientationCombo()) {
        int idx = orientCombo->findData(static_cast<int>(SegmentationDirectionFieldOrientation::Normal));
        if (idx >= 0) {
            orientCombo->setCurrentIndex(idx);
        }
    }
    if (auto* scaleCombo = _directionFieldPanel->scaleCombo()) {
        int idx = scaleCombo->findData(0);
        if (idx >= 0) {
            scaleCombo->setCurrentIndex(idx);
        }
    }
    if (auto* weightSpin = _directionFieldPanel->weightSpin()) {
        weightSpin->setValue(1.0);
    }

    _updatingDirectionFieldForm = previousUpdating;

    // Update button states
    if (auto* removeBtn = _directionFieldPanel->removeButton()) {
        removeBtn->setEnabled(false);
    }
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
    if (auto* addBtn = _directionFieldPanel->addButton()) {
        addBtn->setEnabled(_editingEnabled);
    }
    if (auto* removeBtn = _directionFieldPanel->removeButton()) {
        auto* list = _directionFieldPanel->listWidget();
        const bool hasSelection = list && list->currentRow() >= 0;
        removeBtn->setEnabled(_editingEnabled && hasSelection);
    }
    if (auto* list = _directionFieldPanel->listWidget()) {
        list->setEnabled(_editingEnabled);
    }

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
    if (_customParamsEditor) {
        _customParamsEditor->setEnabled(_editingEnabled);
    }

    const bool allowCorrections = _editingEnabled && _correctionsEnabled && !_growthInProgress;
    if (auto* group = _correctionsPanel->correctionsGroup()) {
        group->setEnabled(allowCorrections);
    }
    if (auto* combo = _correctionsPanel->correctionsCombo()) {
        const QSignalBlocker blocker(combo);
        combo->setEnabled(allowCorrections && combo->count() > 0);
    }
    if (auto* btn = _correctionsPanel->correctionsNewButton()) {
        btn->setEnabled(_editingEnabled && !_growthInProgress);
    }
    if (auto* chk = _correctionsPanel->correctionsAnnotateCheck()) {
        chk->setEnabled(allowCorrections);
    }
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

void SegmentationWidget::setNeuralTracerEnabled(bool enabled)
{
    if (_neuralTracerEnabled == enabled) {
        return;
    }
    _neuralTracerEnabled = enabled;
    writeSetting(QStringLiteral("neural_tracer_enabled"), _neuralTracerEnabled);

    if (auto* chk = _neuralTracerPanel->enabledCheck()) {
        const QSignalBlocker blocker(chk);
        chk->setChecked(enabled);
    }

    emit neuralTracerEnabledChanged(enabled);
}

void SegmentationWidget::setNeuralCheckpointPath(const QString& path)
{
    if (_neuralCheckpointPath == path) {
        return;
    }
    _neuralCheckpointPath = path;
    writeSetting(QStringLiteral("neural_checkpoint_path"), _neuralCheckpointPath);

    if (auto* edit = _neuralTracerPanel->checkpointEdit()) {
        const QSignalBlocker blocker(edit);
        edit->setText(path);
    }
}

void SegmentationWidget::setNeuralPythonPath(const QString& path)
{
    if (_neuralPythonPath == path) {
        return;
    }
    _neuralPythonPath = path;
    writeSetting(QStringLiteral("neural_python_path"), _neuralPythonPath);

    if (auto* edit = _neuralTracerPanel->pythonEdit()) {
        const QSignalBlocker blocker(edit);
        edit->setText(path);
    }
}

void SegmentationWidget::setNeuralVolumeScale(int scale)
{
    scale = std::clamp(scale, 0, 5);
    if (_neuralVolumeScale == scale) {
        return;
    }
    _neuralVolumeScale = scale;
    writeSetting(QStringLiteral("neural_volume_scale"), _neuralVolumeScale);

    if (auto* combo = _neuralTracerPanel->volumeScaleCombo()) {
        const QSignalBlocker blocker(combo);
        int idx = combo->findData(scale);
        if (idx >= 0) {
            combo->setCurrentIndex(idx);
        }
    }
}

void SegmentationWidget::setNeuralBatchSize(int size)
{
    size = std::clamp(size, 1, 64);
    if (_neuralBatchSize == size) {
        return;
    }
    _neuralBatchSize = size;
    writeSetting(QStringLiteral("neural_batch_size"), _neuralBatchSize);

    if (auto* spin = _neuralTracerPanel->batchSizeSpin()) {
        const QSignalBlocker blocker(spin);
        spin->setValue(size);
    }
}

void SegmentationWidget::setVolumeZarrPath(const QString& path)
{
    _volumeZarrPath = path;
}
