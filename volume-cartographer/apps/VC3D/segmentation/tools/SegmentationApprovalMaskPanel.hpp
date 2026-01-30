#pragma once

#include <QWidget>

class QCheckBox;
class QDoubleSpinBox;
class QLabel;
class QPushButton;
class QSlider;
class CollapsibleSettingsGroup;

class SegmentationApprovalMaskPanel : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationApprovalMaskPanel(QWidget* parent = nullptr);

    CollapsibleSettingsGroup* approvalMaskGroup() const { return _groupApprovalMask; }
    QCheckBox* showCheck() const { return _chkShowApprovalMask; }
    QCheckBox* editApprovedCheck() const { return _chkEditApprovedMask; }
    QCheckBox* editUnapprovedCheck() const { return _chkEditUnapprovedMask; }
    QCheckBox* autoApproveCheck() const { return _chkAutoApproveEdits; }
    QDoubleSpinBox* brushRadiusSpin() const { return _spinApprovalBrushRadius; }
    QDoubleSpinBox* brushDepthSpin() const { return _spinApprovalBrushDepth; }
    QSlider* opacitySlider() const { return _sliderApprovalMaskOpacity; }
    QLabel* opacityLabel() const { return _lblApprovalMaskOpacity; }
    QPushButton* colorButton() const { return _btnApprovalColor; }
    QPushButton* undoButton() const { return _btnUndoApprovalStroke; }

private:
    CollapsibleSettingsGroup* _groupApprovalMask{nullptr};
    QCheckBox* _chkShowApprovalMask{nullptr};
    QCheckBox* _chkEditApprovedMask{nullptr};
    QCheckBox* _chkEditUnapprovedMask{nullptr};
    QCheckBox* _chkAutoApproveEdits{nullptr};
    QDoubleSpinBox* _spinApprovalBrushRadius{nullptr};
    QDoubleSpinBox* _spinApprovalBrushDepth{nullptr};
    QSlider* _sliderApprovalMaskOpacity{nullptr};
    QLabel* _lblApprovalMaskOpacity{nullptr};
    QPushButton* _btnApprovalColor{nullptr};
    QPushButton* _btnUndoApprovalStroke{nullptr};
};
