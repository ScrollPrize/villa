#include "SegmentationApprovalMaskPanel.hpp"

#include "elements/CollapsibleSettingsGroup.hpp"

#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QVBoxLayout>

SegmentationApprovalMaskPanel::SegmentationApprovalMaskPanel(QWidget* parent)
    : QWidget(parent)
{
    auto* panelLayout = new QVBoxLayout(this);
    panelLayout->setContentsMargins(0, 0, 0, 0);
    panelLayout->setSpacing(0);

    _groupApprovalMask = new CollapsibleSettingsGroup(tr("Approval Mask"), this);
    auto* approvalLayout = _groupApprovalMask->contentLayout();
    auto* approvalParent = _groupApprovalMask->contentWidget();

    // Show approval mask checkbox
    _chkShowApprovalMask = new QCheckBox(tr("Show Approval Mask"), approvalParent);
    _chkShowApprovalMask->setToolTip(tr("Display the approval mask overlay on the surface."));
    approvalLayout->addWidget(_chkShowApprovalMask);

    // Edit checkboxes row - mutually exclusive approve/unapprove modes
    auto* editRow = new QHBoxLayout();
    editRow->setSpacing(8);

    _chkEditApprovedMask = new QCheckBox(tr("Edit Approved (B)"), approvalParent);
    _chkEditApprovedMask->setToolTip(tr("Paint regions as approved. Saves to disk when toggled off."));
    _chkEditApprovedMask->setEnabled(false);

    _chkEditUnapprovedMask = new QCheckBox(tr("Edit Unapproved (N)"), approvalParent);
    _chkEditUnapprovedMask->setToolTip(tr("Paint regions as unapproved. Saves to disk when toggled off."));
    _chkEditUnapprovedMask->setEnabled(false);

    editRow->addWidget(_chkEditApprovedMask);
    editRow->addWidget(_chkEditUnapprovedMask);
    editRow->addStretch(1);
    approvalLayout->addLayout(editRow);

    // Auto-approve edits checkbox
    _chkAutoApproveEdits = new QCheckBox(tr("Auto-Approve Edits"), approvalParent);
    _chkAutoApproveEdits->setToolTip(tr("Automatically add edited surface regions to the approval mask."));
    approvalLayout->addWidget(_chkAutoApproveEdits);

    // Cylinder brush controls: radius and depth
    auto* approvalBrushRow = new QHBoxLayout();
    approvalBrushRow->setSpacing(8);

    auto* brushRadiusLabel = new QLabel(tr("Radius:"), approvalParent);
    _spinApprovalBrushRadius = new QDoubleSpinBox(approvalParent);
    _spinApprovalBrushRadius->setDecimals(0);
    _spinApprovalBrushRadius->setRange(1.0, 1000.0);
    _spinApprovalBrushRadius->setSingleStep(10.0);
    _spinApprovalBrushRadius->setToolTip(tr("Cylinder radius: circle size in plane views, rectangle width in flattened view (native voxels)."));
    approvalBrushRow->addWidget(brushRadiusLabel);
    approvalBrushRow->addWidget(_spinApprovalBrushRadius);

    auto* brushDepthLabel = new QLabel(tr("Depth:"), approvalParent);
    _spinApprovalBrushDepth = new QDoubleSpinBox(approvalParent);
    _spinApprovalBrushDepth->setDecimals(0);
    _spinApprovalBrushDepth->setRange(1.0, 500.0);
    _spinApprovalBrushDepth->setSingleStep(5.0);
    _spinApprovalBrushDepth->setToolTip(tr("Cylinder depth: rectangle height in flattened view, painting thickness from plane views (native voxels)."));
    approvalBrushRow->addWidget(brushDepthLabel);
    approvalBrushRow->addWidget(_spinApprovalBrushDepth);
    approvalBrushRow->addStretch(1);
    approvalLayout->addLayout(approvalBrushRow);

    // Opacity slider row
    auto* opacityRow = new QHBoxLayout();
    opacityRow->setSpacing(8);

    auto* opacityLabel = new QLabel(tr("Opacity:"), approvalParent);
    _sliderApprovalMaskOpacity = new QSlider(Qt::Horizontal, approvalParent);
    _sliderApprovalMaskOpacity->setRange(0, 100);
    _sliderApprovalMaskOpacity->setToolTip(tr("Mask overlay transparency (0 = transparent, 100 = opaque)."));

    _lblApprovalMaskOpacity = new QLabel(approvalParent);
    _lblApprovalMaskOpacity->setMinimumWidth(35);

    opacityRow->addWidget(opacityLabel);
    opacityRow->addWidget(_sliderApprovalMaskOpacity, 1);
    opacityRow->addWidget(_lblApprovalMaskOpacity);
    approvalLayout->addLayout(opacityRow);

    // Color picker row
    auto* colorRow = new QHBoxLayout();
    colorRow->setSpacing(8);

    auto* colorLabel = new QLabel(tr("Brush Color:"), approvalParent);
    _btnApprovalColor = new QPushButton(approvalParent);
    _btnApprovalColor->setFixedSize(60, 24);
    _btnApprovalColor->setToolTip(tr("Click to choose the color for approval mask painting."));

    colorRow->addWidget(colorLabel);
    colorRow->addWidget(_btnApprovalColor);
    colorRow->addStretch(1);
    approvalLayout->addLayout(colorRow);

    // Undo button
    auto* buttonRow = new QHBoxLayout();
    buttonRow->setSpacing(8);
    _btnUndoApprovalStroke = new QPushButton(tr("Undo (Ctrl+B)"), approvalParent);
    _btnUndoApprovalStroke->setToolTip(tr("Undo the last approval mask brush stroke."));
    buttonRow->addWidget(_btnUndoApprovalStroke);
    buttonRow->addStretch(1);
    approvalLayout->addLayout(buttonRow);

    panelLayout->addWidget(_groupApprovalMask);
}
