#include "SegmentationGrowthPanel.hpp"

#include "VCSettings.hpp"
#include "elements/VolumeSelector.hpp"
#include "segmentation/SegmentationCommon.hpp"
#include "segmentation/growth/SegmentationGrowth.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QRegularExpression>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QVariant>

#include <algorithm>
#include <cmath>
#include <filesystem>

namespace {
constexpr int kGrowDirUpBit = 1 << 0;
constexpr int kGrowDirDownBit = 1 << 1;
constexpr int kGrowDirLeftBit = 1 << 2;
constexpr int kGrowDirRightBit = 1 << 3;
constexpr int kGrowDirAllMask = kGrowDirUpBit | kGrowDirDownBit | kGrowDirLeftBit | kGrowDirRightBit;
constexpr int kDefaultGrowthMaxSteps = 1024;

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
} // namespace

SegmentationGrowthPanel::SegmentationGrowthPanel(const QString& settingsGroup, QWidget* parent)
    : QWidget(parent)
    , _settingsGroup(settingsGroup)
{
    _growthDirectionMask = kGrowDirAllMask;

    auto* panelLayout = new QVBoxLayout(this);
    panelLayout->setContentsMargins(0, 0, 0, 0);
    panelLayout->setSpacing(12);

    _groupGrowth = new QGroupBox(tr("Surface Growth"), this);
    auto* growthLayout = new QVBoxLayout(_groupGrowth);

    // Method selection row
    auto* methodRow = new QHBoxLayout();
    auto* methodLabel = new QLabel(tr("Method:"), _groupGrowth);
    _comboGrowthMethod = new QComboBox(_groupGrowth);
    _comboGrowthMethod->addItem(tr("Tracer"), static_cast<int>(SegmentationGrowthMethod::Tracer));
    _comboGrowthMethod->addItem(tr("Manual Add"), static_cast<int>(SegmentationGrowthMethod::ManualAdd));
    _comboGrowthMethod->setToolTip(tr("Select the growth algorithm:\n"
                                      "- Tracer: Neural-guided growth using volume data\n"
                                      "- Manual Add: Interactive invalid-region bridging"));
    methodRow->addWidget(methodLabel);
    methodRow->addWidget(_comboGrowthMethod);
    methodRow->addStretch(1);
    growthLayout->addLayout(methodRow);

    auto* dirRow = new QHBoxLayout();
    auto* stepsLabel = new QLabel(tr("Steps:"), _groupGrowth);
    _spinGrowthSteps = new QSpinBox(_groupGrowth);
    _spinGrowthSteps->setRange(0, kDefaultGrowthMaxSteps);
    _spinGrowthSteps->setSingleStep(1);
    _spinGrowthSteps->setToolTip(tr("Number of iterations to run when growing the segmentation."));
    dirRow->addWidget(stepsLabel);
    dirRow->addWidget(_spinGrowthSteps);
    dirRow->addSpacing(16);

    auto* scaleLabel = new QLabel(tr("Growth scale:"), _groupGrowth);
    _spinGrowthScale = new QSpinBox(_groupGrowth);
    _spinGrowthScale->setRange(0, 5);
    _spinGrowthScale->setSingleStep(1);
    _spinGrowthScale->setToolTip(tr("Tracer working scale. 0 uses the original tifxyz grid; higher values grow on coarser grids and return to the original scale."));
    dirRow->addWidget(scaleLabel);
    dirRow->addWidget(_spinGrowthScale);
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

    auto* keybindsRow = new QHBoxLayout();
    _chkGrowthKeybindsEnabled = new QCheckBox(tr("Enable growth keybinds (1-6, Shift+5)"), _groupGrowth);
    _chkGrowthKeybindsEnabled->setToolTip(tr("When enabled, keys 1-6 trigger growth in different directions and Shift+5 triggers fill growth."));
    keybindsRow->addWidget(_chkGrowthKeybindsEnabled);
    keybindsRow->addStretch(1);
    growthLayout->addLayout(keybindsRow);

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

    auto* growButtonsRow = new QHBoxLayout();
    _btnGrow = new QPushButton(tr("Grow"), _groupGrowth);
    _btnGrow->setToolTip(tr("Run surface growth using the configured steps and directions."));
    growButtonsRow->addWidget(_btnGrow);

    _btnInpaint = new QPushButton(tr("Inpaint"), _groupGrowth);
    _btnInpaint->setToolTip(tr("Resume the current surface and run tracer inpainting without additional growth."));
    growButtonsRow->addWidget(_btnInpaint);
    growButtonsRow->addStretch(1);
    growthLayout->addLayout(growButtonsRow);

    auto* volumeRow = new QHBoxLayout();
    auto* volumeLabel = new QLabel(tr("Volume:"), _groupGrowth);
    auto* volumeSelector = new VolumeSelector(_groupGrowth);
    volumeSelector->setLabelVisible(false);
    _comboVolumes = volumeSelector->comboBox();
    _comboVolumes->setEnabled(false);
    _comboVolumes->setToolTip(tr("Select which volume provides source data for segmentation growth."));
    volumeRow->addWidget(volumeLabel);
    volumeRow->addWidget(volumeSelector, 1);
    growthLayout->addLayout(volumeRow);

    _groupGrowth->setLayout(growthLayout);
    panelLayout->addWidget(_groupGrowth);

    {
        auto* normalGridRow = new QHBoxLayout();
        _lblNormalGrid = new QLabel(this);
        _lblNormalGrid->setTextFormat(Qt::RichText);
        _lblNormalGrid->setToolTip(tr("Shows whether precomputed normal grids are available for tracer growth."));
        _lblNormalGrid->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        normalGridRow->addWidget(_lblNormalGrid, 0);

        _editNormalGridPath = new QLineEdit(this);
        _editNormalGridPath->setReadOnly(true);
        _editNormalGridPath->setClearButtonEnabled(false);
        _editNormalGridPath->setVisible(false);
        normalGridRow->addWidget(_editNormalGridPath, 1);

        panelLayout->addLayout(normalGridRow);
    }

    // --- Signal wiring ---

    auto connectDirectionCheckbox = [this](QCheckBox* box) {
        if (!box) return;
        connect(box, &QCheckBox::toggled, this, [this, box](bool) {
            updateGrowthDirectionMaskFromUi(box);
        });
    };
    connectDirectionCheckbox(_chkGrowthDirUp);
    connectDirectionCheckbox(_chkGrowthDirDown);
    connectDirectionCheckbox(_chkGrowthDirLeft);
    connectDirectionCheckbox(_chkGrowthDirRight);

    connect(_chkGrowthKeybindsEnabled, &QCheckBox::toggled, this, [this](bool checked) {
        _growthKeybindsEnabled = checked;
        writeSetting(vc3d::settings::segmentation::GROWTH_KEYBINDS_ENABLED, _growthKeybindsEnabled);
    });

    connect(_spinGrowthSteps, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int value) { applyGrowthSteps(value, true, true); });

    connect(_spinGrowthScale, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (_restoringSettings) return;
        const int scale = std::clamp(value, 0, 5);
        if (_growthScale == scale) return;
        _growthScale = scale;
        writeSetting(QStringLiteral("growth_scale"), _growthScale);
    });

    connect(_comboGrowthMethod, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int index) {
                const auto method = static_cast<SegmentationGrowthMethod>(
                    _comboGrowthMethod->itemData(index).toInt());
                setGrowthMethod(method);
            });

    // Grow/inpaint buttons
    connect(_btnGrow, &QPushButton::clicked, this, [this]() {
        const auto allowed = allowedGrowthDirections();
        auto direction = SegmentationGrowthDirection::All;
        if (allowed.size() == 1) {
            direction = allowed.front();
        }
        triggerGrowthRequest(direction, _growthSteps, false);
    });
    connect(_btnInpaint, &QPushButton::clicked, this, [this]() {
        triggerGrowthRequest(SegmentationGrowthDirection::All, 0, true);
    });

    // Volume combo
    connect(_comboVolumes, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0) return;
        const QString volumeId = _comboVolumes->itemData(index).toString();
        if (volumeId.isEmpty() || volumeId == _activeVolumeId) return;
        _activeVolumeId = volumeId;
        emit volumeSelectionChanged(volumeId);
    });

    // Z-range connections
    connect(_chkCorrectionsUseZRange, &QCheckBox::toggled, this, [this](bool enabled) {
        _correctionsZRangeEnabled = enabled;
        writeSetting(vc3d::settings::segmentation::CORRECTIONS_Z_RANGE_ENABLED, _correctionsZRangeEnabled);
        updateGrowthUiState();
        emit correctionsZRangeChanged(enabled, _correctionsZMin, _correctionsZMax);
    });
    connect(_spinCorrectionsZMin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (_correctionsZMin == value) return;
        _correctionsZMin = value;
        writeSetting(vc3d::settings::segmentation::CORRECTIONS_Z_MIN, _correctionsZMin);
        if (_correctionsZRangeEnabled) {
            emit correctionsZRangeChanged(true, _correctionsZMin, _correctionsZMax);
        }
    });
    connect(_spinCorrectionsZMax, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (_correctionsZMax == value) return;
        _correctionsZMax = value;
        writeSetting(vc3d::settings::segmentation::CORRECTIONS_Z_MAX, _correctionsZMax);
        if (_correctionsZRangeEnabled) {
            emit correctionsZRangeChanged(true, _correctionsZMin, _correctionsZMax);
        }
    });
}

void SegmentationGrowthPanel::writeSetting(const QString& key, const QVariant& value)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(_settingsGroup);
    settings.setValue(key, value);
    settings.endGroup();
}

QString SegmentationGrowthPanel::determineDefaultVolumeId(const QVector<QPair<QString, QString>>& volumes,
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

void SegmentationGrowthPanel::applyGrowthSteps(int steps, bool persist, bool fromUi)
{
    const int minimum = (_growthMethod == SegmentationGrowthMethod::Corrections) ? 0 : 1;
    const int maximum = kDefaultGrowthMaxSteps;
    const int clamped = std::clamp(steps, minimum, maximum);

    if (_spinGrowthSteps && _spinGrowthSteps->maximum() != maximum) {
        QSignalBlocker blocker(_spinGrowthSteps);
        _spinGrowthSteps->setMaximum(maximum);
    }

    if ((!fromUi || clamped != steps) && _spinGrowthSteps) {
        QSignalBlocker blocker(_spinGrowthSteps);
        _spinGrowthSteps->setValue(clamped);
    }

    if (clamped > 0) {
        _tracerGrowthSteps = std::max(1, std::min(clamped, kDefaultGrowthMaxSteps));
    }

    _growthSteps = clamped;

    if (persist) {
        writeSetting(vc3d::settings::segmentation::GROWTH_STEPS, _growthSteps);
        writeSetting(QStringLiteral("growth_steps_tracer"), _tracerGrowthSteps);
    }
}

void SegmentationGrowthPanel::setGrowthSteps(int steps, bool persist)
{
    applyGrowthSteps(steps, persist, false);
}

void SegmentationGrowthPanel::setGrowthMethod(SegmentationGrowthMethod method)
{
    if (_growthMethod == method) {
        return;
    }
    const int currentSteps = _growthSteps;
    const SegmentationGrowthMethod previousMethod = _growthMethod;
    if (previousMethod != SegmentationGrowthMethod::Corrections) {
        _tracerGrowthSteps = (currentSteps > 0)
            ? std::min(currentSteps, kDefaultGrowthMaxSteps)
            : std::max(1, _tracerGrowthSteps);
    }
    _growthMethod = method;
    int targetSteps = std::max(1, _tracerGrowthSteps);
    if (method == SegmentationGrowthMethod::Corrections) {
        targetSteps = 0;
    }
    applyGrowthSteps(targetSteps, true, false);
    writeSetting(vc3d::settings::segmentation::GROWTH_METHOD, static_cast<int>(_growthMethod));
    syncUiState(_editingEnabled, _growthInProgress);
    emit growthMethodChanged(_growthMethod);
}

void SegmentationGrowthPanel::setGrowthInProgress(bool running)
{
    if (_growthInProgress == running) {
        return;
    }
    _growthInProgress = running;
    updateGrowthUiState();
}

void SegmentationGrowthPanel::setManualAddUiActive(bool active)
{
    if (_manualAddUiActive == active) {
        return;
    }
    _manualAddUiActive = active;
    updateGrowthUiState();
}

void SegmentationGrowthPanel::setNormalGridAvailable(bool available)
{
    _normalGridAvailable = available;
    syncUiState(_editingEnabled, _growthInProgress);
}

void SegmentationGrowthPanel::setNormalGridPathHint(const QString& hint)
{
    _normalGridHint = hint;
    QString display = hint.trimmed();
    const int colonIndex = display.indexOf(QLatin1Char(':'));
    if (colonIndex >= 0 && colonIndex + 1 < display.size()) {
        display = display.mid(colonIndex + 1).trimmed();
    }
    _normalGridDisplayPath = display;
    syncUiState(_editingEnabled, _growthInProgress);
}

void SegmentationGrowthPanel::setNormalGridPath(const QString& path)
{
    _normalGridPath = path.trimmed();
    syncUiState(_editingEnabled, _growthInProgress);
}

void SegmentationGrowthPanel::setVolumePackagePath(const QString& path)
{
    _volumePackagePath = path;
    syncUiState(_editingEnabled, _growthInProgress);
}

void SegmentationGrowthPanel::setAvailableVolumes(const QVector<QPair<QString, QString>>& volumes,
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

void SegmentationGrowthPanel::setActiveVolume(const QString& volumeId)
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

std::vector<SegmentationGrowthDirection> SegmentationGrowthPanel::allowedGrowthDirections() const
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

std::optional<std::pair<int, int>> SegmentationGrowthPanel::correctionsZRange() const
{
    if (!_correctionsZRangeEnabled) {
        return std::nullopt;
    }
    return std::make_pair(_correctionsZMin, _correctionsZMax);
}

void SegmentationGrowthPanel::setGrowthDirectionMask(int mask)
{
    mask = normalizeGrowthDirectionMask(mask);
    if (_growthDirectionMask == mask) {
        return;
    }
    _growthDirectionMask = mask;
    writeSetting(vc3d::settings::segmentation::GROWTH_DIRECTION_MASK, _growthDirectionMask);
    applyGrowthDirectionMaskToUi();
}

void SegmentationGrowthPanel::updateGrowthDirectionMaskFromUi(QCheckBox* changedCheckbox)
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

void SegmentationGrowthPanel::applyGrowthDirectionMaskToUi()
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

int SegmentationGrowthPanel::normalizeGrowthDirectionMask(int mask)
{
    mask &= kGrowDirAllMask;
    if (mask == 0) {
        // If no directions are selected, enable all directions by default.
        // This ensures that growth is not unintentionally disabled.
        mask = kGrowDirAllMask;
    }
    return mask;
}

void SegmentationGrowthPanel::updateGrowthUiState()
{
    const bool manual = _manualAddUiActive || _growthMethod == SegmentationGrowthMethod::ManualAdd;
    const bool enableGrowth = _editingEnabled && !_growthInProgress && !manual;
    if (_spinGrowthSteps) {
        _spinGrowthSteps->setVisible(!manual);
    }
    if (_spinGrowthScale) {
        _spinGrowthScale->setVisible(!manual);
    }
    if (_btnGrow) {
        _btnGrow->setVisible(!manual);
    }
    if (_btnInpaint) {
        _btnInpaint->setVisible(!manual);
    }
    if (_chkGrowthDirUp) {
        _chkGrowthDirUp->setVisible(!manual);
    }
    if (_chkGrowthDirDown) {
        _chkGrowthDirDown->setVisible(!manual);
    }
    if (_chkGrowthDirLeft) {
        _chkGrowthDirLeft->setVisible(!manual);
    }
    if (_chkGrowthDirRight) {
        _chkGrowthDirRight->setVisible(!manual);
    }
    if (_chkGrowthKeybindsEnabled) {
        _chkGrowthKeybindsEnabled->setVisible(!manual);
    }
    if (_chkCorrectionsUseZRange) {
        _chkCorrectionsUseZRange->setVisible(!manual);
    }
    if (_spinCorrectionsZMin) {
        _spinCorrectionsZMin->setVisible(!manual);
    }
    if (_spinCorrectionsZMax) {
        _spinCorrectionsZMax->setVisible(!manual);
    }
    if (_comboVolumes) {
        if (auto* parent = _comboVolumes->parentWidget()) {
            parent->setVisible(!manual);
        }
    }
    if (_lblNormalGrid) {
        _lblNormalGrid->setVisible(!manual);
    }
    if (_editNormalGridPath) {
        _editNormalGridPath->setVisible(!manual && _normalGridAvailable && !_normalGridPath.isEmpty());
    }
    if (_spinGrowthSteps) {
        _spinGrowthSteps->setEnabled(enableGrowth);
    }
    if (_spinGrowthScale) {
        _spinGrowthScale->setEnabled(enableGrowth);
    }
    if (_btnGrow) {
        _btnGrow->setEnabled(enableGrowth);
    }
    if (_btnInpaint) {
        _btnInpaint->setEnabled(enableGrowth);
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
}

void SegmentationGrowthPanel::triggerGrowthRequest(SegmentationGrowthDirection direction,
                                                    int steps,
                                                    bool inpaintOnly)
{
    if (!_editingEnabled || _growthInProgress) {
        return;
    }
    if (_growthMethod == SegmentationGrowthMethod::ManualAdd) {
        return;
    }

    const SegmentationGrowthMethod method = inpaintOnly
        ? SegmentationGrowthMethod::Tracer
        : _growthMethod;

    const bool allowZeroSteps = inpaintOnly || method == SegmentationGrowthMethod::Corrections;
    const int minSteps = allowZeroSteps ? 0 : 1;
    const int clampedSteps = std::clamp(steps, minSteps, kDefaultGrowthMaxSteps);
    const int finalSteps = clampedSteps;

    qCInfo(lcSegWidget) << "Grow request" << segmentationGrowthMethodToString(method)
                        << segmentationGrowthDirectionToString(direction)
                        << "steps" << finalSteps
                        << "inpaintOnly" << inpaintOnly;
    emit growSurfaceRequested(method, direction, finalSteps, inpaintOnly);
}

void SegmentationGrowthPanel::restoreSettings(QSettings& settings)
{
    using namespace vc3d::settings;
    _restoringSettings = true;

    _growthMethod = segmentationGrowthMethodFromInt(
        settings.value(segmentation::GROWTH_METHOD, static_cast<int>(_growthMethod)).toInt());

    int storedGrowthSteps = settings.value(segmentation::GROWTH_STEPS, _growthSteps).toInt();
    storedGrowthSteps = std::clamp(storedGrowthSteps, 0, 1024);
    _tracerGrowthSteps = settings
                             .value(QStringLiteral("growth_steps_tracer"),
                                    std::max(1, storedGrowthSteps))
                             .toInt();
    _tracerGrowthSteps = std::clamp(_tracerGrowthSteps, 1, kDefaultGrowthMaxSteps);
    applyGrowthSteps(storedGrowthSteps, false, false);
    _growthDirectionMask = normalizeGrowthDirectionMask(
        settings.value(segmentation::GROWTH_DIRECTION_MASK, kGrowDirAllMask).toInt());
    _growthKeybindsEnabled = settings.value(segmentation::GROWTH_KEYBINDS_ENABLED,
                                            segmentation::GROWTH_KEYBINDS_ENABLED_DEFAULT).toBool();

    _correctionsZRangeEnabled = settings.value(segmentation::CORRECTIONS_Z_RANGE_ENABLED, segmentation::CORRECTIONS_Z_RANGE_ENABLED_DEFAULT).toBool();
    _correctionsZMin = settings.value(segmentation::CORRECTIONS_Z_MIN, segmentation::CORRECTIONS_Z_MIN_DEFAULT).toInt();
    _correctionsZMax = settings.value(segmentation::CORRECTIONS_Z_MAX, _correctionsZMin).toInt();
    if (_correctionsZMax < _correctionsZMin) {
        _correctionsZMax = _correctionsZMin;
    }

    _growthScale = std::clamp(settings.value(QStringLiteral("growth_scale"), _growthScale).toInt(), 0, 5);

    _restoringSettings = false;
}

void SegmentationGrowthPanel::syncUiState(bool editingEnabled, bool growthInProgress)
{
    _editingEnabled = editingEnabled;
    _growthInProgress = growthInProgress;

    if (_spinGrowthSteps) {
        const QSignalBlocker blocker(_spinGrowthSteps);
        _spinGrowthSteps->setMaximum(kDefaultGrowthMaxSteps);
        _spinGrowthSteps->setValue(_growthSteps);
    }
    if (_spinGrowthScale) {
        const QSignalBlocker blocker(_spinGrowthScale);
        _spinGrowthScale->setValue(_growthScale);
    }

    if (_comboGrowthMethod) {
        const QSignalBlocker blocker(_comboGrowthMethod);
        int idx = _comboGrowthMethod->findData(static_cast<int>(_growthMethod));
        if (idx >= 0) {
            _comboGrowthMethod->setCurrentIndex(idx);
        }
    }

    applyGrowthDirectionMaskToUi();
    if (_chkGrowthKeybindsEnabled) {
        const QSignalBlocker blocker(_chkGrowthKeybindsEnabled);
        _chkGrowthKeybindsEnabled->setChecked(_growthKeybindsEnabled);
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

    // Normal grid UI
    if (_lblNormalGrid) {
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

        _lblNormalGrid->setText(icon + QStringLiteral("&nbsp;") + message);
        _lblNormalGrid->setToolTip(tooltip);
        _lblNormalGrid->setAccessibleDescription(message);
    }

    if (_editNormalGridPath) {
        const bool show = _normalGridAvailable && !_normalGridPath.isEmpty();
        _editNormalGridPath->setVisible(show);
        _editNormalGridPath->setText(_normalGridPath);
        _editNormalGridPath->setToolTip(_normalGridPath);
    }

    updateGrowthUiState();
}
