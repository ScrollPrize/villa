#include "SegmentationNeuralTracerPanel.hpp"

#include "elements/CollapsibleSettingsGroup.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QSpinBox>
#include <QToolButton>
#include <QVBoxLayout>

SegmentationNeuralTracerPanel::SegmentationNeuralTracerPanel(QWidget* parent)
    : QWidget(parent)
{
    auto* panelLayout = new QVBoxLayout(this);
    panelLayout->setContentsMargins(0, 0, 0, 0);
    panelLayout->setSpacing(0);

    _groupNeuralTracer = new CollapsibleSettingsGroup(tr("Neural Tracer"), this);
    auto* neuralParent = _groupNeuralTracer->contentWidget();

    _chkNeuralTracerEnabled = new QCheckBox(tr("Enable neural tracer"), neuralParent);
    _chkNeuralTracerEnabled->setToolTip(tr("Use neural network-based tracing instead of the default tracer. "
                                           "Requires a trained model checkpoint."));
    _groupNeuralTracer->contentLayout()->addWidget(_chkNeuralTracerEnabled);

    _groupNeuralTracer->addRow(tr("Checkpoint:"), [&](QHBoxLayout* row) {
        _neuralCheckpointEdit = new QLineEdit(neuralParent);
        _neuralCheckpointEdit->setPlaceholderText(tr("Path to model checkpoint (.pt)"));
        _neuralCheckpointEdit->setToolTip(tr("Path to the trained neural network checkpoint file."));
        _neuralCheckpointBrowse = new QToolButton(neuralParent);
        _neuralCheckpointBrowse->setText(QStringLiteral("..."));
        _neuralCheckpointBrowse->setToolTip(tr("Browse for checkpoint file."));
        row->addWidget(_neuralCheckpointEdit, 1);
        row->addWidget(_neuralCheckpointBrowse);
    }, tr("Path to the trained neural network checkpoint file."));

    _groupNeuralTracer->addRow(tr("Python:"), [&](QHBoxLayout* row) {
        _neuralPythonEdit = new QLineEdit(neuralParent);
        _neuralPythonEdit->setPlaceholderText(tr("Path to Python executable (leave empty for auto-detect)"));
        _neuralPythonEdit->setToolTip(tr("Path to the Python executable with torch installed (e.g. ~/miniconda3/bin/python). "
                                         "Leave empty to auto-detect."));
        _neuralPythonBrowse = new QToolButton(neuralParent);
        _neuralPythonBrowse->setText(QStringLiteral("..."));
        _neuralPythonBrowse->setToolTip(tr("Browse for Python executable."));
        row->addWidget(_neuralPythonEdit, 1);
        row->addWidget(_neuralPythonBrowse);
    }, tr("Python executable with torch installed."));

    _groupNeuralTracer->addRow(tr("Volume scale:"), [&](QHBoxLayout* row) {
        _comboNeuralVolumeScale = new QComboBox(neuralParent);
        _comboNeuralVolumeScale->setToolTip(tr("OME-Zarr scale level to use for neural tracing (0 = full resolution)."));
        for (int scale = 0; scale <= 5; ++scale) {
            _comboNeuralVolumeScale->addItem(QString::number(scale), scale);
        }
        row->addWidget(_comboNeuralVolumeScale);

        auto* batchLabel = new QLabel(tr("Batch size:"), neuralParent);
        _spinNeuralBatchSize = new QSpinBox(neuralParent);
        _spinNeuralBatchSize->setRange(1, 64);
        _spinNeuralBatchSize->setToolTip(tr("Number of points to process in parallel (higher = faster but more memory)."));
        row->addSpacing(12);
        row->addWidget(batchLabel);
        row->addWidget(_spinNeuralBatchSize);
        row->addStretch(1);
    });

    _lblNeuralTracerStatus = new QLabel(neuralParent);
    _lblNeuralTracerStatus->setWordWrap(true);
    _lblNeuralTracerStatus->setVisible(false);
    _groupNeuralTracer->contentLayout()->addWidget(_lblNeuralTracerStatus);

    panelLayout->addWidget(_groupNeuralTracer);
}
