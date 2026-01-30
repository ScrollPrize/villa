#include "SegmentationCellReoptPanel.hpp"

#include "elements/CollapsibleSettingsGroup.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>

SegmentationCellReoptPanel::SegmentationCellReoptPanel(QWidget* parent)
    : QWidget(parent)
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
}
