#pragma once

#include <QWidget>

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QSpinBox;
class QToolButton;
class CollapsibleSettingsGroup;

class SegmentationNeuralTracerPanel : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationNeuralTracerPanel(QWidget* parent = nullptr);

    CollapsibleSettingsGroup* neuralTracerGroup() const { return _groupNeuralTracer; }
    QCheckBox* enabledCheck() const { return _chkNeuralTracerEnabled; }
    QLineEdit* checkpointEdit() const { return _neuralCheckpointEdit; }
    QToolButton* checkpointBrowse() const { return _neuralCheckpointBrowse; }
    QLineEdit* pythonEdit() const { return _neuralPythonEdit; }
    QToolButton* pythonBrowse() const { return _neuralPythonBrowse; }
    QComboBox* volumeScaleCombo() const { return _comboNeuralVolumeScale; }
    QSpinBox* batchSizeSpin() const { return _spinNeuralBatchSize; }
    QLabel* statusLabel() const { return _lblNeuralTracerStatus; }

private:
    CollapsibleSettingsGroup* _groupNeuralTracer{nullptr};
    QCheckBox* _chkNeuralTracerEnabled{nullptr};
    QLineEdit* _neuralCheckpointEdit{nullptr};
    QToolButton* _neuralCheckpointBrowse{nullptr};
    QLineEdit* _neuralPythonEdit{nullptr};
    QToolButton* _neuralPythonBrowse{nullptr};
    QComboBox* _comboNeuralVolumeScale{nullptr};
    QSpinBox* _spinNeuralBatchSize{nullptr};
    QLabel* _lblNeuralTracerStatus{nullptr};
};
