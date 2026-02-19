#include "SegmentationNeuralTracerPanel.hpp"

#include "NeuralTraceServiceManager.hpp"
#include "VCSettings.hpp"
#include "elements/CollapsibleSettingsGroup.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QToolButton>
#include <QVBoxLayout>

#include <algorithm>
#include <array>

namespace {
const QString kDenseLatestSentinel = QStringLiteral("extrap_displacement_latest");
const QString kCopyLatestSentinel = QStringLiteral("copy_displacement_latest");
const QString kDensePresetSettingKey = QStringLiteral("neural_dense_checkpoint_preset");
const QString kCopyPresetSettingKey = QStringLiteral("neural_copy_checkpoint_preset");
const QString kDensePresetLatest = QStringLiteral("latest");
const QString kCopyPresetLatest = QStringLiteral("latest");
const QString kDensePresetCustom = QStringLiteral("custom");
const QString kCopyPresetCustom = QStringLiteral("custom");
const QString kDenseTtaModeSettingKey = QStringLiteral("neural_dense_tta_mode");
const QString kDenseTtaMergeMethodSettingKey = QStringLiteral("neural_dense_tta_merge_method");
const QString kDenseTtaOutlierDropThreshSettingKey = QStringLiteral("neural_dense_tta_outlier_drop_thresh");
const QString kDenseTtaMergeMethodDefault = QStringLiteral("vector_geomedian");
constexpr double kDenseTtaOutlierDropThreshDefault = 1.25;

QString normalizeDenseTtaMergeMethod(const QString& method)
{
    const QString normalized = method.trimmed().toLower();
    static const std::array<const char*, 5> kSupportedMethods = {
        "median",
        "mean",
        "trimmed_mean",
        "vector_medoid",
        "vector_geomedian",
    };
    for (const char* candidate : kSupportedMethods) {
        if (normalized == QLatin1String(candidate)) {
            return normalized;
        }
    }
    return kDenseTtaMergeMethodDefault;
}
} // namespace

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

    _btnCopyWithNt = new QPushButton(tr("Copy with NT"), neuralParent);
    _btnCopyWithNt->setToolTip(tr("Run displacement copy inference and create front/back output segments."));
    _groupNeuralTracer->contentLayout()->addWidget(_btnCopyWithNt);

    _groupNeuralTracer->addRow(tr("Model type:"), [&](QHBoxLayout* row) {
        _comboNeuralModelType = new QComboBox(neuralParent);
        _comboNeuralModelType->addItem(tr("Heatmap"), static_cast<int>(NeuralTracerModelType::Heatmap));
        _comboNeuralModelType->addItem(tr("Extrapolation growth"), static_cast<int>(NeuralTracerModelType::DenseDisplacement));
        _comboNeuralModelType->addItem(tr("Displacement Copy"), static_cast<int>(NeuralTracerModelType::DisplacementCopy));
        _comboNeuralModelType->setToolTip(tr("Select which neural tracing model path to use."));
        row->addWidget(_comboNeuralModelType);
        row->addStretch(1);
    });

    _groupNeuralTracer->addRow(tr("Output mode:"), [&](QHBoxLayout* row) {
        _comboNeuralOutputMode = new QComboBox(neuralParent);
        _comboNeuralOutputMode->addItem(tr("Overwrite current segment"),
                                        static_cast<int>(NeuralTracerOutputMode::OverwriteCurrentSegment));
        _comboNeuralOutputMode->addItem(tr("Create new segment"),
                                        static_cast<int>(NeuralTracerOutputMode::CreateNewSegment));
        _comboNeuralOutputMode->setToolTip(tr("Choose whether dense displacement updates the current segment or creates a new one."));
        row->addWidget(_comboNeuralOutputMode);
        row->addStretch(1);
    });

    _groupNeuralTracer->addRow(tr("TTA Type:"), [&](QHBoxLayout* row) {
        _comboDenseTtaMode = new QComboBox(neuralParent);
        _comboDenseTtaMode->addItem(tr("Mirror TTA"), static_cast<int>(DenseTtaMode::Mirror));
        _comboDenseTtaMode->addItem(tr("Rotate3 TTA"), static_cast<int>(DenseTtaMode::Rotate3));
        _comboDenseTtaMode->addItem(tr("None"), static_cast<int>(DenseTtaMode::None));
        _comboDenseTtaMode->setToolTip(
            tr("Dense displacement test-time augmentation mode. Mirror is default."));
        row->addWidget(_comboDenseTtaMode);
        row->addStretch(1);
    });

    _groupNeuralTracer->addRow(tr("TTA merge:"), [&](QHBoxLayout* row) {
        _comboDenseTtaMergeMethod = new QComboBox(neuralParent);
        _comboDenseTtaMergeMethod->addItem(tr("Vector geomedian"), QStringLiteral("vector_geomedian"));
        _comboDenseTtaMergeMethod->addItem(tr("Vector medoid"), QStringLiteral("vector_medoid"));
        _comboDenseTtaMergeMethod->addItem(tr("Median"), QStringLiteral("median"));
        _comboDenseTtaMergeMethod->addItem(tr("Trimmed mean"), QStringLiteral("trimmed_mean"));
        _comboDenseTtaMergeMethod->addItem(tr("Mean"), QStringLiteral("mean"));
        _comboDenseTtaMergeMethod->setToolTip(
            tr("How to merge TTA displacement predictions."));
        row->addWidget(_comboDenseTtaMergeMethod);

        auto* threshLabel = new QLabel(tr("Outlier thresh:"), neuralParent);
        _spinDenseTtaOutlierDropThresh = new QDoubleSpinBox(neuralParent);
        _spinDenseTtaOutlierDropThresh->setDecimals(2);
        _spinDenseTtaOutlierDropThresh->setRange(0.01, 100.0);
        _spinDenseTtaOutlierDropThresh->setSingleStep(0.05);
        _spinDenseTtaOutlierDropThresh->setToolTip(
            tr("Outlier threshold multiplier for dropping inconsistent TTA variants."));
        row->addSpacing(12);
        row->addWidget(threshLabel);
        row->addWidget(_spinDenseTtaOutlierDropThresh);
        row->addStretch(1);
    });

    _groupNeuralTracer->addRow(tr("Checkpoint path:"), [&](QHBoxLayout* row) {
        _comboDenseCheckpointPreset = new QComboBox(neuralParent);
        _comboDenseCheckpointPreset->addItem(
            tr("Dense Displacement Latest"),
            static_cast<int>(DenseCheckpointPreset::DenseLatest));
        _comboDenseCheckpointPreset->addItem(
            tr("Custom path"),
            static_cast<int>(DenseCheckpointPreset::CustomPath));
        _comboDenseCheckpointPreset->setToolTip(
            tr("Choose a built-in dense displacement checkpoint preset or provide a custom checkpoint file path."));
        row->addWidget(_comboDenseCheckpointPreset);
        row->addStretch(1);
    });

    _groupNeuralTracer->addRow(tr("Copy checkpoint path:"), [&](QHBoxLayout* row) {
        _comboCopyCheckpointPreset = new QComboBox(neuralParent);
        _comboCopyCheckpointPreset->addItem(
            tr("Copy Displacement Latest"),
            static_cast<int>(CopyCheckpointPreset::CopyLatest));
        _comboCopyCheckpointPreset->addItem(
            tr("Custom path"),
            static_cast<int>(CopyCheckpointPreset::CustomPath));
        _comboCopyCheckpointPreset->setToolTip(
            tr("Choose a built-in displacement copy checkpoint preset or provide a custom checkpoint file path."));
        row->addWidget(_comboCopyCheckpointPreset);
        row->addStretch(1);
    });

    _groupNeuralTracer->addRow(tr("Checkpoint:"), [&](QHBoxLayout* row) {
        _neuralCheckpointEdit = new QLineEdit(neuralParent);
        _neuralCheckpointEdit->setPlaceholderText(tr("Path to model checkpoint (.pt)"));
        _neuralCheckpointEdit->setToolTip(tr("Checkpoint path used for Heatmap and for Extrapolation growth when Checkpoint path is set to Custom path."));
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

    connect(_comboNeuralModelType, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        auto type = static_cast<NeuralTracerModelType>(_comboNeuralModelType->itemData(index).toInt());
        setNeuralModelType(type);
    });

    connect(_comboNeuralOutputMode, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        auto mode = static_cast<NeuralTracerOutputMode>(_comboNeuralOutputMode->itemData(index).toInt());
        setNeuralOutputMode(mode);
    });

    connect(_comboDenseTtaMode, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        auto mode = static_cast<DenseTtaMode>(_comboDenseTtaMode->itemData(index).toInt());
        setDenseTtaMode(mode);
    });

    connect(_comboDenseTtaMergeMethod, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        setDenseTtaMergeMethod(_comboDenseTtaMergeMethod->itemData(index).toString());
    });

    connect(_spinDenseTtaOutlierDropThresh, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setDenseTtaOutlierDropThresh(value);
    });

    connect(_comboDenseCheckpointPreset, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        _denseCheckpointPreset = static_cast<DenseCheckpointPreset>(_comboDenseCheckpointPreset->itemData(index).toInt());
        writeSetting(kDensePresetSettingKey,
                     _denseCheckpointPreset == DenseCheckpointPreset::DenseLatest
                         ? kDensePresetLatest
                         : kDensePresetCustom);
        updateDenseUiState();
    });

    connect(_comboCopyCheckpointPreset, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        _copyCheckpointPreset = static_cast<CopyCheckpointPreset>(_comboCopyCheckpointPreset->itemData(index).toInt());
        writeSetting(kCopyPresetSettingKey,
                     _copyCheckpointPreset == CopyCheckpointPreset::CopyLatest
                         ? kCopyPresetLatest
                         : kCopyPresetCustom);
        updateDenseUiState();
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

    connect(_groupNeuralTracer, &CollapsibleSettingsGroup::toggled, this, [this](bool expanded) {
        if (_restoringSettings) {
            return;
        }
        writeSetting(vc3d::settings::segmentation::GROUP_NEURAL_TRACER_EXPANDED, expanded);
    });

    connect(_btnCopyWithNt, &QPushButton::clicked, this, [this]() {
        emit copyWithNtRequested();
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

    updateDenseUiState();
}

void SegmentationNeuralTracerPanel::writeSetting(const QString& key, const QVariant& value)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(_settingsGroup);
    settings.setValue(key, value);
    settings.endGroup();
}

QString SegmentationNeuralTracerPanel::denseCheckpointPath() const
{
    if (_denseCheckpointPreset == DenseCheckpointPreset::DenseLatest) {
        return kDenseLatestSentinel;
    }
    return _neuralCheckpointPath.trimmed();
}

QString SegmentationNeuralTracerPanel::copyCheckpointPath() const
{
    if (_copyCheckpointPreset == CopyCheckpointPreset::CopyLatest) {
        return kCopyLatestSentinel;
    }
    return _neuralCheckpointPath.trimmed();
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

    updateDenseUiState();
    emit neuralTracerEnabledChanged(enabled);
}

void SegmentationNeuralTracerPanel::setNeuralCheckpointPath(const QString& path)
{
    const QString trimmed = path.trimmed();
    if (_neuralCheckpointPath == trimmed) {
        return;
    }
    _neuralCheckpointPath = trimmed;
    writeSetting(QStringLiteral("neural_checkpoint_path"), _neuralCheckpointPath);

    if (_neuralCheckpointEdit) {
        const QSignalBlocker blocker(_neuralCheckpointEdit);
        _neuralCheckpointEdit->setText(trimmed);
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

void SegmentationNeuralTracerPanel::setNeuralModelType(NeuralTracerModelType type)
{
    if (_neuralModelType == type) {
        return;
    }
    _neuralModelType = type;
    writeSetting(QStringLiteral("neural_model_type"), static_cast<int>(_neuralModelType));

    if (_comboNeuralModelType) {
        const QSignalBlocker blocker(_comboNeuralModelType);
        int idx = _comboNeuralModelType->findData(static_cast<int>(_neuralModelType));
        if (idx >= 0) {
            _comboNeuralModelType->setCurrentIndex(idx);
        }
    }
    updateDenseUiState();
}

void SegmentationNeuralTracerPanel::setNeuralOutputMode(NeuralTracerOutputMode mode)
{
    if (_neuralOutputMode == mode) {
        return;
    }
    _neuralOutputMode = mode;
    writeSetting(QStringLiteral("neural_output_mode"), static_cast<int>(_neuralOutputMode));

    if (_comboNeuralOutputMode) {
        const QSignalBlocker blocker(_comboNeuralOutputMode);
        int idx = _comboNeuralOutputMode->findData(static_cast<int>(_neuralOutputMode));
        if (idx >= 0) {
            _comboNeuralOutputMode->setCurrentIndex(idx);
        }
    }
}

void SegmentationNeuralTracerPanel::setDenseTtaMode(DenseTtaMode mode)
{
    if (_denseTtaMode == mode) {
        return;
    }
    _denseTtaMode = mode;
    writeSetting(kDenseTtaModeSettingKey, static_cast<int>(_denseTtaMode));

    if (_comboDenseTtaMode) {
        const QSignalBlocker blocker(_comboDenseTtaMode);
        int idx = _comboDenseTtaMode->findData(static_cast<int>(_denseTtaMode));
        if (idx >= 0) {
            _comboDenseTtaMode->setCurrentIndex(idx);
        }
    }
}

void SegmentationNeuralTracerPanel::setDenseTtaMergeMethod(const QString& method)
{
    const QString normalized = normalizeDenseTtaMergeMethod(method);
    if (_denseTtaMergeMethod == normalized) {
        return;
    }
    _denseTtaMergeMethod = normalized;
    writeSetting(kDenseTtaMergeMethodSettingKey, _denseTtaMergeMethod);

    if (_comboDenseTtaMergeMethod) {
        const QSignalBlocker blocker(_comboDenseTtaMergeMethod);
        const int idx = _comboDenseTtaMergeMethod->findData(_denseTtaMergeMethod);
        if (idx >= 0) {
            _comboDenseTtaMergeMethod->setCurrentIndex(idx);
        }
    }
}

void SegmentationNeuralTracerPanel::setDenseTtaOutlierDropThresh(double threshold)
{
    const double sanitized = std::max(0.01, threshold);
    if (qFuzzyCompare(_denseTtaOutlierDropThresh + 1.0, sanitized + 1.0)) {
        return;
    }
    _denseTtaOutlierDropThresh = sanitized;
    writeSetting(kDenseTtaOutlierDropThreshSettingKey, _denseTtaOutlierDropThresh);

    if (_spinDenseTtaOutlierDropThresh) {
        const QSignalBlocker blocker(_spinDenseTtaOutlierDropThresh);
        _spinDenseTtaOutlierDropThresh->setValue(_denseTtaOutlierDropThresh);
    }
}

void SegmentationNeuralTracerPanel::setDenseCheckpointPath(const QString& path)
{
    const QString trimmed = path.trimmed();
    const DenseCheckpointPreset nextPreset =
        (trimmed == kDenseLatestSentinel) ? DenseCheckpointPreset::DenseLatest : DenseCheckpointPreset::CustomPath;
    if (_denseCheckpointPreset != nextPreset) {
        _denseCheckpointPreset = nextPreset;
        writeSetting(kDensePresetSettingKey,
                     _denseCheckpointPreset == DenseCheckpointPreset::DenseLatest
                         ? kDensePresetLatest
                         : kDensePresetCustom);
        if (_comboDenseCheckpointPreset) {
            const QSignalBlocker blocker(_comboDenseCheckpointPreset);
            const int idx = _comboDenseCheckpointPreset->findData(static_cast<int>(_denseCheckpointPreset));
            if (idx >= 0) {
                _comboDenseCheckpointPreset->setCurrentIndex(idx);
            }
        }
    }
    if (nextPreset == DenseCheckpointPreset::CustomPath) {
        setNeuralCheckpointPath(trimmed);
    }
    updateDenseUiState();
}

void SegmentationNeuralTracerPanel::setCopyCheckpointPath(const QString& path)
{
    const QString trimmed = path.trimmed();
    const CopyCheckpointPreset nextPreset =
        (trimmed == kCopyLatestSentinel) ? CopyCheckpointPreset::CopyLatest : CopyCheckpointPreset::CustomPath;
    if (_copyCheckpointPreset != nextPreset) {
        _copyCheckpointPreset = nextPreset;
        writeSetting(kCopyPresetSettingKey,
                     _copyCheckpointPreset == CopyCheckpointPreset::CopyLatest
                         ? kCopyPresetLatest
                         : kCopyPresetCustom);
        if (_comboCopyCheckpointPreset) {
            const QSignalBlocker blocker(_comboCopyCheckpointPreset);
            const int idx = _comboCopyCheckpointPreset->findData(static_cast<int>(_copyCheckpointPreset));
            if (idx >= 0) {
                _comboCopyCheckpointPreset->setCurrentIndex(idx);
            }
        }
    }
    if (nextPreset == CopyCheckpointPreset::CustomPath) {
        setNeuralCheckpointPath(trimmed);
    }
    updateDenseUiState();
}

void SegmentationNeuralTracerPanel::setVolumeZarrPath(const QString& path)
{
    _volumeZarrPath = path;
}

void SegmentationNeuralTracerPanel::restoreSettings(QSettings& settings)
{
    using namespace vc3d::settings;

    _restoringSettings = true;

    _neuralTracerEnabled = settings.value(QStringLiteral("neural_tracer_enabled"), false).toBool();
    _neuralCheckpointPath = settings.value(QStringLiteral("neural_checkpoint_path"), QString()).toString();
    _neuralPythonPath = settings.value(QStringLiteral("neural_python_path"), QString()).toString();
    _neuralVolumeScale = settings.value(QStringLiteral("neural_volume_scale"), 0).toInt();
    _neuralVolumeScale = std::clamp(_neuralVolumeScale, 0, 5);
    _neuralBatchSize = settings.value(QStringLiteral("neural_batch_size"), 4).toInt();
    _neuralBatchSize = std::clamp(_neuralBatchSize, 1, 64);
    const int modelType = settings.value(QStringLiteral("neural_model_type"),
                                         static_cast<int>(NeuralTracerModelType::Heatmap)).toInt();
    if (modelType == static_cast<int>(NeuralTracerModelType::DenseDisplacement)) {
        _neuralModelType = NeuralTracerModelType::DenseDisplacement;
    } else if (modelType == static_cast<int>(NeuralTracerModelType::DisplacementCopy)) {
        _neuralModelType = NeuralTracerModelType::DisplacementCopy;
    } else {
        _neuralModelType = NeuralTracerModelType::Heatmap;
    }
    const int outputMode = settings.value(QStringLiteral("neural_output_mode"),
                                          static_cast<int>(NeuralTracerOutputMode::OverwriteCurrentSegment)).toInt();
    _neuralOutputMode = outputMode == static_cast<int>(NeuralTracerOutputMode::CreateNewSegment)
        ? NeuralTracerOutputMode::CreateNewSegment
        : NeuralTracerOutputMode::OverwriteCurrentSegment;
    const int ttaModeValue = settings.value(kDenseTtaModeSettingKey,
                                            static_cast<int>(DenseTtaMode::Mirror)).toInt();
    if (ttaModeValue == static_cast<int>(DenseTtaMode::Rotate3)) {
        _denseTtaMode = DenseTtaMode::Rotate3;
    } else if (ttaModeValue == static_cast<int>(DenseTtaMode::None)) {
        _denseTtaMode = DenseTtaMode::None;
    } else {
        _denseTtaMode = DenseTtaMode::Mirror;
    }
    _denseTtaMergeMethod = normalizeDenseTtaMergeMethod(
        settings.value(kDenseTtaMergeMethodSettingKey, kDenseTtaMergeMethodDefault).toString());
    _denseTtaOutlierDropThresh = settings.value(
        kDenseTtaOutlierDropThreshSettingKey,
        kDenseTtaOutlierDropThreshDefault).toDouble();
    if (_denseTtaOutlierDropThresh <= 0.0) {
        _denseTtaOutlierDropThresh = kDenseTtaOutlierDropThreshDefault;
    }
    const QString presetValue = settings.value(kDensePresetSettingKey, kDensePresetLatest).toString();
    if (presetValue == kDensePresetCustom) {
        _denseCheckpointPreset = DenseCheckpointPreset::CustomPath;
    } else {
        _denseCheckpointPreset = DenseCheckpointPreset::DenseLatest;
    }
    const QString copyPresetValue = settings.value(kCopyPresetSettingKey, kCopyPresetLatest).toString();
    if (copyPresetValue == kCopyPresetCustom) {
        _copyCheckpointPreset = CopyCheckpointPreset::CustomPath;
    } else {
        _copyCheckpointPreset = CopyCheckpointPreset::CopyLatest;
    }

    // Restore group expansion state
    const bool neuralExpanded = settings.value(segmentation::GROUP_NEURAL_TRACER_EXPANDED,
                                               segmentation::GROUP_NEURAL_TRACER_EXPANDED_DEFAULT).toBool();
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
    if (_comboNeuralModelType) {
        const QSignalBlocker blocker(_comboNeuralModelType);
        int idx = _comboNeuralModelType->findData(static_cast<int>(_neuralModelType));
        if (idx >= 0) {
            _comboNeuralModelType->setCurrentIndex(idx);
        }
    }
    if (_comboNeuralOutputMode) {
        const QSignalBlocker blocker(_comboNeuralOutputMode);
        int idx = _comboNeuralOutputMode->findData(static_cast<int>(_neuralOutputMode));
        if (idx >= 0) {
            _comboNeuralOutputMode->setCurrentIndex(idx);
        }
    }
    if (_comboDenseTtaMode) {
        const QSignalBlocker blocker(_comboDenseTtaMode);
        int idx = _comboDenseTtaMode->findData(static_cast<int>(_denseTtaMode));
        if (idx >= 0) {
            _comboDenseTtaMode->setCurrentIndex(idx);
        }
    }
    if (_comboDenseTtaMergeMethod) {
        const QSignalBlocker blocker(_comboDenseTtaMergeMethod);
        int idx = _comboDenseTtaMergeMethod->findData(_denseTtaMergeMethod);
        if (idx >= 0) {
            _comboDenseTtaMergeMethod->setCurrentIndex(idx);
        }
    }
    if (_spinDenseTtaOutlierDropThresh) {
        const QSignalBlocker blocker(_spinDenseTtaOutlierDropThresh);
        _spinDenseTtaOutlierDropThresh->setValue(_denseTtaOutlierDropThresh);
    }
    if (_comboDenseCheckpointPreset) {
        const QSignalBlocker blocker(_comboDenseCheckpointPreset);
        int idx = _comboDenseCheckpointPreset->findData(static_cast<int>(_denseCheckpointPreset));
        if (idx >= 0) {
            _comboDenseCheckpointPreset->setCurrentIndex(idx);
        }
    }
    if (_comboCopyCheckpointPreset) {
        const QSignalBlocker blocker(_comboCopyCheckpointPreset);
        int idx = _comboCopyCheckpointPreset->findData(static_cast<int>(_copyCheckpointPreset));
        if (idx >= 0) {
            _comboCopyCheckpointPreset->setCurrentIndex(idx);
        }
    }
    updateDenseUiState();
}

void SegmentationNeuralTracerPanel::updateDenseUiState()
{
    const bool denseMode = _neuralModelType == NeuralTracerModelType::DenseDisplacement;
    const bool copyMode = _neuralModelType == NeuralTracerModelType::DisplacementCopy;
    const bool displacementMode = denseMode || copyMode;
    const bool useDenseLatestCheckpoint = denseMode && _denseCheckpointPreset == DenseCheckpointPreset::DenseLatest;
    const bool useCopyLatestCheckpoint = copyMode && _copyCheckpointPreset == CopyCheckpointPreset::CopyLatest;
    const bool usingLatestCheckpoint = useDenseLatestCheckpoint || useCopyLatestCheckpoint;
    if (_neuralCheckpointEdit) {
        _neuralCheckpointEdit->setEnabled(!usingLatestCheckpoint);
    }
    if (_neuralCheckpointBrowse) {
        _neuralCheckpointBrowse->setEnabled(!usingLatestCheckpoint);
    }
    if (_comboDenseCheckpointPreset) {
        _comboDenseCheckpointPreset->setEnabled(denseMode);
        _comboDenseCheckpointPreset->setVisible(denseMode);
    }
    if (_comboCopyCheckpointPreset) {
        _comboCopyCheckpointPreset->setEnabled(copyMode);
        _comboCopyCheckpointPreset->setVisible(copyMode);
    }
    if (_comboDenseTtaMode) {
        _comboDenseTtaMode->setEnabled(displacementMode);
        _comboDenseTtaMode->setVisible(displacementMode);
    }
    if (_comboDenseTtaMergeMethod) {
        _comboDenseTtaMergeMethod->setEnabled(displacementMode);
        _comboDenseTtaMergeMethod->setVisible(displacementMode);
    }
    if (_spinDenseTtaOutlierDropThresh) {
        _spinDenseTtaOutlierDropThresh->setEnabled(displacementMode);
        _spinDenseTtaOutlierDropThresh->setVisible(displacementMode);
    }
    if (_comboNeuralOutputMode) {
        _comboNeuralOutputMode->setEnabled(denseMode);
    }
    if (_btnCopyWithNt) {
        _btnCopyWithNt->setEnabled(_neuralTracerEnabled && copyMode);
        _btnCopyWithNt->setVisible(copyMode);
    }
}
