#include "SegmentationNeuralTracerPanel.hpp"

#include "NeuralTraceServiceManager.hpp"
#include "VCSettings.hpp"
#include "elements/CollapsibleSettingsGroup.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QToolButton>
#include <QVBoxLayout>

#include <algorithm>

SegmentationNeuralTracerPanel::SegmentationNeuralTracerPanel(const QString& settingsGroup,
                                                             QWidget* parent)
    : QWidget(parent)
    , _settingsGroup(settingsGroup)
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

    // --- Signal wiring (moved from SegmentationWidget::buildUi) ---

    connect(_chkNeuralTracerEnabled, &QCheckBox::toggled, this, [this](bool enabled) {
        setNeuralTracerEnabled(enabled);
    });

    connect(_neuralCheckpointEdit, &QLineEdit::textChanged, this, [this](const QString& text) {
        _neuralCheckpointPath = text.trimmed();
        writeSetting(QStringLiteral("neural_checkpoint_path"), _neuralCheckpointPath);
    });

    connect(_neuralCheckpointBrowse, &QToolButton::clicked, this, [this]() {
        const QString initial = _neuralCheckpointPath.isEmpty() ? QDir::homePath() : _neuralCheckpointPath;
        const QString file = QFileDialog::getOpenFileName(this, tr("Select neural tracer checkpoint"),
                                                          initial, tr("PyTorch Checkpoint (*.pt *.pth);;All Files (*)"));
        if (!file.isEmpty()) {
            _neuralCheckpointPath = file;
            _neuralCheckpointEdit->setText(file);
        }
    });

    connect(_neuralPythonEdit, &QLineEdit::textChanged, this, [this](const QString& text) {
        _neuralPythonPath = text.trimmed();
        writeSetting(QStringLiteral("neural_python_path"), _neuralPythonPath);
    });

    connect(_neuralPythonBrowse, &QToolButton::clicked, this, [this]() {
        const QString initial = _neuralPythonPath.isEmpty() ? QDir::homePath() : QFileInfo(_neuralPythonPath).absolutePath();
        const QString file = QFileDialog::getOpenFileName(this, tr("Select Python executable"),
                                                          initial, tr("All Files (*)"));
        if (!file.isEmpty()) {
            _neuralPythonPath = file;
            _neuralPythonEdit->setText(file);
        }
    });

    connect(_comboNeuralVolumeScale, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        _neuralVolumeScale = _comboNeuralVolumeScale->itemData(index).toInt();
        writeSetting(QStringLiteral("neural_volume_scale"), _neuralVolumeScale);
    });

    connect(_spinNeuralBatchSize, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        _neuralBatchSize = value;
        writeSetting(QStringLiteral("neural_batch_size"), _neuralBatchSize);
    });

    // Connect to service manager signals
    auto& serviceManager = NeuralTraceServiceManager::instance();
    connect(&serviceManager, &NeuralTraceServiceManager::statusMessage, this, [this](const QString& message) {
        if (_lblNeuralTracerStatus) {
            _lblNeuralTracerStatus->setText(message);
            _lblNeuralTracerStatus->setVisible(true);
            _lblNeuralTracerStatus->setStyleSheet(QString());
        }
        emit neuralTracerStatusMessage(message);
    });
    connect(&serviceManager, &NeuralTraceServiceManager::serviceStarted, this, [this]() {
        if (_lblNeuralTracerStatus) {
            _lblNeuralTracerStatus->setText(tr("Service running"));
            _lblNeuralTracerStatus->setStyleSheet(QStringLiteral("color: #27ae60;"));
        }
    });
    connect(&serviceManager, &NeuralTraceServiceManager::serviceStopped, this, [this]() {
        if (_lblNeuralTracerStatus) {
            _lblNeuralTracerStatus->setText(tr("Service stopped"));
            _lblNeuralTracerStatus->setStyleSheet(QString());
        }
    });
    connect(&serviceManager, &NeuralTraceServiceManager::serviceError, this, [this](const QString& error) {
        if (_lblNeuralTracerStatus) {
            _lblNeuralTracerStatus->setText(tr("Error: %1").arg(error));
            _lblNeuralTracerStatus->setStyleSheet(QStringLiteral("color: #c0392b;"));
            _lblNeuralTracerStatus->setVisible(true);
        }
    });
}

void SegmentationNeuralTracerPanel::writeSetting(const QString& key, const QVariant& value)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(_settingsGroup);
    settings.setValue(key, value);
    settings.endGroup();
}

void SegmentationNeuralTracerPanel::setNeuralTracerEnabled(bool enabled)
{
    if (_neuralTracerEnabled == enabled) {
        return;
    }
    _neuralTracerEnabled = enabled;
    writeSetting(QStringLiteral("neural_tracer_enabled"), _neuralTracerEnabled);

    if (_chkNeuralTracerEnabled) {
        const QSignalBlocker blocker(_chkNeuralTracerEnabled);
        _chkNeuralTracerEnabled->setChecked(enabled);
    }

    emit neuralTracerEnabledChanged(enabled);
}

void SegmentationNeuralTracerPanel::setNeuralCheckpointPath(const QString& path)
{
    if (_neuralCheckpointPath == path) {
        return;
    }
    _neuralCheckpointPath = path;
    writeSetting(QStringLiteral("neural_checkpoint_path"), _neuralCheckpointPath);

    if (_neuralCheckpointEdit) {
        const QSignalBlocker blocker(_neuralCheckpointEdit);
        _neuralCheckpointEdit->setText(path);
    }
}

void SegmentationNeuralTracerPanel::setNeuralPythonPath(const QString& path)
{
    if (_neuralPythonPath == path) {
        return;
    }
    _neuralPythonPath = path;
    writeSetting(QStringLiteral("neural_python_path"), _neuralPythonPath);

    if (_neuralPythonEdit) {
        const QSignalBlocker blocker(_neuralPythonEdit);
        _neuralPythonEdit->setText(path);
    }
}

void SegmentationNeuralTracerPanel::setNeuralVolumeScale(int scale)
{
    scale = std::clamp(scale, 0, 5);
    if (_neuralVolumeScale == scale) {
        return;
    }
    _neuralVolumeScale = scale;
    writeSetting(QStringLiteral("neural_volume_scale"), _neuralVolumeScale);

    if (_comboNeuralVolumeScale) {
        const QSignalBlocker blocker(_comboNeuralVolumeScale);
        int idx = _comboNeuralVolumeScale->findData(scale);
        if (idx >= 0) {
            _comboNeuralVolumeScale->setCurrentIndex(idx);
        }
    }
}

void SegmentationNeuralTracerPanel::setNeuralBatchSize(int size)
{
    size = std::clamp(size, 1, 64);
    if (_neuralBatchSize == size) {
        return;
    }
    _neuralBatchSize = size;
    writeSetting(QStringLiteral("neural_batch_size"), _neuralBatchSize);

    if (_spinNeuralBatchSize) {
        const QSignalBlocker blocker(_spinNeuralBatchSize);
        _spinNeuralBatchSize->setValue(size);
    }
}

void SegmentationNeuralTracerPanel::setVolumeZarrPath(const QString& path)
{
    _volumeZarrPath = path;
}

void SegmentationNeuralTracerPanel::restoreSettings(QSettings& settings)
{
    _restoringSettings = true;

    _neuralTracerEnabled = settings.value(QStringLiteral("neural_tracer_enabled"), false).toBool();
    _neuralCheckpointPath = settings.value(QStringLiteral("neural_checkpoint_path"), QString()).toString();
    _neuralPythonPath = settings.value(QStringLiteral("neural_python_path"), QString()).toString();
    _neuralVolumeScale = settings.value(QStringLiteral("neural_volume_scale"), 0).toInt();
    _neuralVolumeScale = std::clamp(_neuralVolumeScale, 0, 5);
    _neuralBatchSize = settings.value(QStringLiteral("neural_batch_size"), 4).toInt();
    _neuralBatchSize = std::clamp(_neuralBatchSize, 1, 64);

    // Restore group expansion state
    const bool neuralExpanded = settings.value(QStringLiteral("group_neural_tracer_expanded"), false).toBool();
    if (_groupNeuralTracer) {
        _groupNeuralTracer->setExpanded(neuralExpanded);
    }

    _restoringSettings = false;
}

void SegmentationNeuralTracerPanel::syncUiState()
{
    if (_chkNeuralTracerEnabled) {
        const QSignalBlocker blocker(_chkNeuralTracerEnabled);
        _chkNeuralTracerEnabled->setChecked(_neuralTracerEnabled);
    }
    if (_neuralCheckpointEdit) {
        const QSignalBlocker blocker(_neuralCheckpointEdit);
        _neuralCheckpointEdit->setText(_neuralCheckpointPath);
    }
    if (_neuralPythonEdit) {
        const QSignalBlocker blocker(_neuralPythonEdit);
        _neuralPythonEdit->setText(_neuralPythonPath);
    }
    if (_comboNeuralVolumeScale) {
        const QSignalBlocker blocker(_comboNeuralVolumeScale);
        int idx = _comboNeuralVolumeScale->findData(_neuralVolumeScale);
        if (idx >= 0) {
            _comboNeuralVolumeScale->setCurrentIndex(idx);
        }
    }
    if (_spinNeuralBatchSize) {
        const QSignalBlocker blocker(_spinNeuralBatchSize);
        _spinNeuralBatchSize->setValue(_neuralBatchSize);
    }
}
