#pragma once

#include <QWidget>

class QCheckBox;
class QDoubleSpinBox;
class QLabel;
class QPushButton;
class QSpinBox;
class CollapsibleSettingsGroup;

class SegmentationEditingPanel : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationEditingPanel(QWidget* parent = nullptr);

    // Group accessors
    CollapsibleSettingsGroup* editingGroup() const { return _groupEditing; }
    CollapsibleSettingsGroup* dragGroup() const { return _groupDrag; }
    CollapsibleSettingsGroup* lineGroup() const { return _groupLine; }
    CollapsibleSettingsGroup* pushPullGroup() const { return _groupPushPull; }

    // Drag
    QDoubleSpinBox* dragRadiusSpin() const { return _spinDragRadius; }
    QDoubleSpinBox* dragSigmaSpin() const { return _spinDragSigma; }

    // Line
    QDoubleSpinBox* lineRadiusSpin() const { return _spinLineRadius; }
    QDoubleSpinBox* lineSigmaSpin() const { return _spinLineSigma; }

    // Push/Pull
    QDoubleSpinBox* pushPullRadiusSpin() const { return _spinPushPullRadius; }
    QDoubleSpinBox* pushPullSigmaSpin() const { return _spinPushPullSigma; }
    QDoubleSpinBox* pushPullStepSpin() const { return _spinPushPullStep; }

    // Alpha config
    QWidget* alphaPushPullPanel() const { return _alphaPushPullPanel; }
    QLabel* alphaInfoLabel() const { return _lblAlphaInfo; }
    QCheckBox* alphaPerVertexCheck() const { return _chkAlphaPerVertex; }
    QDoubleSpinBox* alphaStartSpin() const { return _spinAlphaStart; }
    QDoubleSpinBox* alphaStopSpin() const { return _spinAlphaStop; }
    QDoubleSpinBox* alphaStepSpin() const { return _spinAlphaStep; }
    QDoubleSpinBox* alphaLowSpin() const { return _spinAlphaLow; }
    QDoubleSpinBox* alphaHighSpin() const { return _spinAlphaHigh; }
    QDoubleSpinBox* alphaBorderSpin() const { return _spinAlphaBorder; }
    QSpinBox* alphaBlurRadiusSpin() const { return _spinAlphaBlurRadius; }
    QDoubleSpinBox* alphaPerVertexLimitSpin() const { return _spinAlphaPerVertexLimit; }

    // Smoothing
    QDoubleSpinBox* smoothStrengthSpin() const { return _spinSmoothStrength; }
    QSpinBox* smoothIterationsSpin() const { return _spinSmoothIterations; }

    // Buttons
    QPushButton* applyButton() const { return _btnApply; }
    QPushButton* resetButton() const { return _btnReset; }
    QPushButton* stopButton() const { return _btnStop; }

    // Hover marker
    QCheckBox* showHoverMarkerCheck() const { return _chkShowHoverMarker; }

private:
    CollapsibleSettingsGroup* _groupEditing{nullptr};
    CollapsibleSettingsGroup* _groupDrag{nullptr};
    CollapsibleSettingsGroup* _groupLine{nullptr};
    CollapsibleSettingsGroup* _groupPushPull{nullptr};

    QDoubleSpinBox* _spinDragRadius{nullptr};
    QDoubleSpinBox* _spinDragSigma{nullptr};
    QDoubleSpinBox* _spinLineRadius{nullptr};
    QDoubleSpinBox* _spinLineSigma{nullptr};
    QDoubleSpinBox* _spinPushPullRadius{nullptr};
    QDoubleSpinBox* _spinPushPullSigma{nullptr};
    QDoubleSpinBox* _spinPushPullStep{nullptr};
    QWidget* _alphaPushPullPanel{nullptr};
    QCheckBox* _chkAlphaPerVertex{nullptr};
    QDoubleSpinBox* _spinAlphaStart{nullptr};
    QDoubleSpinBox* _spinAlphaStop{nullptr};
    QDoubleSpinBox* _spinAlphaStep{nullptr};
    QDoubleSpinBox* _spinAlphaLow{nullptr};
    QDoubleSpinBox* _spinAlphaHigh{nullptr};
    QDoubleSpinBox* _spinAlphaBorder{nullptr};
    QSpinBox* _spinAlphaBlurRadius{nullptr};
    QDoubleSpinBox* _spinAlphaPerVertexLimit{nullptr};
    QLabel* _lblAlphaInfo{nullptr};
    QDoubleSpinBox* _spinSmoothStrength{nullptr};
    QSpinBox* _spinSmoothIterations{nullptr};
    QPushButton* _btnApply{nullptr};
    QPushButton* _btnReset{nullptr};
    QPushButton* _btnStop{nullptr};
    QCheckBox* _chkShowHoverMarker{nullptr};
};
