#include "SegmentationCellReoptPanel.hpp"

#include "VCSettings.hpp"
#include "elements/CollapsibleSettingsGroup.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QVariant>
#include <QVBoxLayout>

#include <algorithm>

SegmentationCellReoptPanel::SegmentationCellReoptPanel(const QString& settingsGroup,
                                                       QWidget* parent)
    : QWidget(parent)
    , _settingsGroup(settingsGroup)
{
    auto* panelLayout = new QVBoxLayout(this);
    panelLayout->setContentsMargins(0, 0, 0, 0);
    panelLayout->setSpacing(0);

    _groupCellReopt = new CollapsibleSettingsGroup(tr("Cell Reoptimization"), this);
    auto* cellReoptLayout = _groupCellReopt->contentLayout();
    auto* cellReoptParent = _groupCellReopt->contentWidget();

    // Enable mode checkbox
    _chkCellReoptMode = new QCheckBox(tr("Enable Cell Reoptimization"), cellReoptParent);
    _chkCellReoptMode->setToolTip(tr("Click on unapproved regions to flood fill and place correction points.\n"
                                      "Requires approval mask to be visible."));
    cellReoptLayout->addWidget(_chkCellReoptMode);

    // Max flood cells
    auto* maxFloodRow = new QHBoxLayout();
    maxFloodRow->setSpacing(8);
    auto* maxFloodLabel = new QLabel(tr("Max Flood Cells:"), cellReoptParent);
    _spinCellReoptMaxSteps = new QSpinBox(cellReoptParent);
    _spinCellReoptMaxSteps->setRange(10, 10000);
    _spinCellReoptMaxSteps->setToolTip(tr("Maximum number of cells to include in the flood fill."));
    maxFloodRow->addWidget(maxFloodLabel);
    maxFloodRow->addWidget(_spinCellReoptMaxSteps);
    maxFloodRow->addStretch(1);
    cellReoptLayout->addLayout(maxFloodRow);

    // Max correction points
    auto* maxPointsRow = new QHBoxLayout();
    maxPointsRow->setSpacing(8);
    auto* maxPointsLabel = new QLabel(tr("Max Points:"), cellReoptParent);
    _spinCellReoptMaxPoints = new QSpinBox(cellReoptParent);
    _spinCellReoptMaxPoints->setRange(3, 200);
    _spinCellReoptMaxPoints->setToolTip(tr("Maximum number of correction points to place on the boundary."));
    maxPointsRow->addWidget(maxPointsLabel);
    maxPointsRow->addWidget(_spinCellReoptMaxPoints);
    maxPointsRow->addStretch(1);
    cellReoptLayout->addLayout(maxPointsRow);

    // Min point spacing
    auto* minSpacingRow = new QHBoxLayout();
    minSpacingRow->setSpacing(8);
    auto* minSpacingLabel = new QLabel(tr("Min Spacing:"), cellReoptParent);
    _spinCellReoptMinSpacing = new QDoubleSpinBox(cellReoptParent);
    _spinCellReoptMinSpacing->setRange(1.0, 50.0);
    _spinCellReoptMinSpacing->setSuffix(tr(" grid"));
    _spinCellReoptMinSpacing->setToolTip(tr("Minimum spacing between correction points (grid steps)."));
    minSpacingRow->addWidget(minSpacingLabel);
    minSpacingRow->addWidget(_spinCellReoptMinSpacing);
    minSpacingRow->addStretch(1);
    cellReoptLayout->addLayout(minSpacingRow);

    // Perimeter offset
    auto* perimeterOffsetRow = new QHBoxLayout();
    perimeterOffsetRow->setSpacing(8);
    auto* perimeterOffsetLabel = new QLabel(tr("Perimeter Offset:"), cellReoptParent);
    _spinCellReoptPerimeterOffset = new QDoubleSpinBox(cellReoptParent);
    _spinCellReoptPerimeterOffset->setRange(-50.0, 50.0);
    _spinCellReoptPerimeterOffset->setSuffix(tr(" grid"));
    _spinCellReoptPerimeterOffset->setToolTip(tr("Offset to expand (+) or shrink (-) the traced perimeter from center of mass."));
    perimeterOffsetRow->addWidget(perimeterOffsetLabel);
    perimeterOffsetRow->addWidget(_spinCellReoptPerimeterOffset);
    perimeterOffsetRow->addStretch(1);
    cellReoptLayout->addLayout(perimeterOffsetRow);

    // Collection selector
    auto* collectionRow = new QHBoxLayout();
    collectionRow->setSpacing(8);
    auto* collectionLabel = new QLabel(tr("Collection:"), cellReoptParent);
    _comboCellReoptCollection = new QComboBox(cellReoptParent);
    _comboCellReoptCollection->setToolTip(tr("Select which correction point collection to use for reoptimization."));
    _comboCellReoptCollection->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    collectionRow->addWidget(collectionLabel);
    collectionRow->addWidget(_comboCellReoptCollection, 1);
    cellReoptLayout->addLayout(collectionRow);

    // Run reoptimization button
    auto* runButtonRow = new QHBoxLayout();
    runButtonRow->setSpacing(8);
    _btnCellReoptRun = new QPushButton(tr("Run Reoptimization"), cellReoptParent);
    _btnCellReoptRun->setToolTip(tr("Trigger reoptimization using the selected correction point collection."));
    runButtonRow->addWidget(_btnCellReoptRun);
    runButtonRow->addStretch(1);
    cellReoptLayout->addLayout(runButtonRow);

    panelLayout->addWidget(_groupCellReopt);

    // --- Signal wiring (moved from SegmentationWidget::buildUi) ---

    connect(_chkCellReoptMode, &QCheckBox::toggled, this, [this](bool enabled) {
        setCellReoptMode(enabled);
    });

    connect(_spinCellReoptMaxSteps, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (_cellReoptMaxSteps != value) {
            _cellReoptMaxSteps = value;
            if (!_restoringSettings) {
                writeSetting(QStringLiteral("cell_reopt_max_steps"), value);
                emit cellReoptMaxStepsChanged(value);
            }
        }
    });

    connect(_spinCellReoptMaxPoints, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (_cellReoptMaxPoints != value) {
            _cellReoptMaxPoints = value;
            if (!_restoringSettings) {
                writeSetting(QStringLiteral("cell_reopt_max_points"), value);
                emit cellReoptMaxPointsChanged(value);
            }
        }
    });

    connect(_spinCellReoptMinSpacing, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        float floatVal = static_cast<float>(value);
        if (_cellReoptMinSpacing != floatVal) {
            _cellReoptMinSpacing = floatVal;
            if (!_restoringSettings) {
                writeSetting(QStringLiteral("cell_reopt_min_spacing"), value);
                emit cellReoptMinSpacingChanged(floatVal);
            }
        }
    });

    connect(_spinCellReoptPerimeterOffset, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        float floatVal = static_cast<float>(value);
        if (_cellReoptPerimeterOffset != floatVal) {
            _cellReoptPerimeterOffset = floatVal;
            if (!_restoringSettings) {
                writeSetting(QStringLiteral("cell_reopt_perimeter_offset"), value);
                emit cellReoptPerimeterOffsetChanged(floatVal);
            }
        }
    });

    connect(_btnCellReoptRun, &QPushButton::clicked, this, [this]() {
        uint64_t collectionId = 0;
        if (_comboCellReoptCollection && _comboCellReoptCollection->currentIndex() >= 0) {
            collectionId = _comboCellReoptCollection->currentData().toULongLong();
        }
        emit cellReoptGrowthRequested(collectionId);
    });
}

void SegmentationCellReoptPanel::writeSetting(const QString& key, const QVariant& value)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(_settingsGroup);
    settings.setValue(key, value);
    settings.endGroup();
}

void SegmentationCellReoptPanel::setCellReoptMode(bool enabled)
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
    if (_chkCellReoptMode) {
        const QSignalBlocker blocker(_chkCellReoptMode);
        _chkCellReoptMode->setChecked(_cellReoptMode);
    }
}

void SegmentationCellReoptPanel::setCellReoptCollections(const QVector<QPair<uint64_t, QString>>& collections)
{
    if (!_comboCellReoptCollection) {
        return;
    }

    // Remember current selection
    uint64_t currentId = 0;
    if (_comboCellReoptCollection->currentIndex() >= 0) {
        currentId = _comboCellReoptCollection->currentData().toULongLong();
    }

    const QSignalBlocker blocker(_comboCellReoptCollection);
    _comboCellReoptCollection->clear();

    int indexToSelect = -1;
    for (int i = 0; i < collections.size(); ++i) {
        const auto& [id, name] = collections[i];
        _comboCellReoptCollection->addItem(name, QVariant::fromValue(id));
        if (id == currentId) {
            indexToSelect = i;
        }
    }

    // Restore selection if possible, otherwise select first item
    if (indexToSelect >= 0) {
        _comboCellReoptCollection->setCurrentIndex(indexToSelect);
    } else if (_comboCellReoptCollection->count() > 0) {
        _comboCellReoptCollection->setCurrentIndex(0);
    }
}

void SegmentationCellReoptPanel::restoreSettings(QSettings& settings)
{
    _restoringSettings = true;

    _cellReoptMaxSteps = settings.value(QStringLiteral("cell_reopt_max_steps"), _cellReoptMaxSteps).toInt();
    _cellReoptMaxSteps = std::clamp(_cellReoptMaxSteps, 10, 10000);
    _cellReoptMaxPoints = settings.value(QStringLiteral("cell_reopt_max_points"), _cellReoptMaxPoints).toInt();
    _cellReoptMaxPoints = std::clamp(_cellReoptMaxPoints, 3, 200);
    _cellReoptMinSpacing = settings.value(QStringLiteral("cell_reopt_min_spacing"), static_cast<double>(_cellReoptMinSpacing)).toFloat();
    _cellReoptMinSpacing = std::clamp(_cellReoptMinSpacing, 1.0f, 50.0f);
    _cellReoptPerimeterOffset = settings.value(QStringLiteral("cell_reopt_perimeter_offset"), static_cast<double>(_cellReoptPerimeterOffset)).toFloat();
    _cellReoptPerimeterOffset = std::clamp(_cellReoptPerimeterOffset, -50.0f, 50.0f);
    // Don't restore cell reopt mode - user must explicitly enable each session

    _restoringSettings = false;
}

void SegmentationCellReoptPanel::syncUiState(bool showApprovalMask, bool growthInProgress)
{
    if (_chkCellReoptMode) {
        const QSignalBlocker blocker(_chkCellReoptMode);
        _chkCellReoptMode->setChecked(_cellReoptMode);
        // Only enabled when approval mask is visible
        _chkCellReoptMode->setEnabled(showApprovalMask);
    }
    if (_spinCellReoptMaxSteps) {
        const QSignalBlocker blocker(_spinCellReoptMaxSteps);
        _spinCellReoptMaxSteps->setValue(_cellReoptMaxSteps);
        _spinCellReoptMaxSteps->setEnabled(_cellReoptMode);
    }
    if (_spinCellReoptMaxPoints) {
        const QSignalBlocker blocker(_spinCellReoptMaxPoints);
        _spinCellReoptMaxPoints->setValue(_cellReoptMaxPoints);
        _spinCellReoptMaxPoints->setEnabled(_cellReoptMode);
    }
    if (_spinCellReoptMinSpacing) {
        const QSignalBlocker blocker(_spinCellReoptMinSpacing);
        _spinCellReoptMinSpacing->setValue(static_cast<double>(_cellReoptMinSpacing));
        _spinCellReoptMinSpacing->setEnabled(_cellReoptMode);
    }
    if (_spinCellReoptPerimeterOffset) {
        const QSignalBlocker blocker(_spinCellReoptPerimeterOffset);
        _spinCellReoptPerimeterOffset->setValue(static_cast<double>(_cellReoptPerimeterOffset));
        _spinCellReoptPerimeterOffset->setEnabled(_cellReoptMode);
    }
    if (_comboCellReoptCollection) {
        _comboCellReoptCollection->setEnabled(_cellReoptMode);
    }
    if (_btnCellReoptRun) {
        const bool hasCollection = _comboCellReoptCollection && _comboCellReoptCollection->count() > 0;
        _btnCellReoptRun->setEnabled(_cellReoptMode && !growthInProgress && hasCollection);
    }
}
