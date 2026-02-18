#include "SegmentationFitOptimizerPanel.hpp"

#include "FitServiceManager.hpp"
#include "VCSettings.hpp"
#include "elements/CollapsibleSettingsGroup.hpp"

#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSettings>
#include <QSignalBlocker>
#include <QToolButton>
#include <QVBoxLayout>

#include <nlohmann/json.hpp>

static const char* kDefaultConfig = R"({
    "base": {
        "dir_v": 10.0,
        "dir_conn": 1.0,
        "step": 100.0,
        "smooth_x": 0.0,
        "smooth_y": 0.01,
        "gradmag": 1.0,
        "data": 0.4,
        "data_plain": 0.2,
        "z_straight": 10.0
    },
    "stages": [
        {
            "name": "reopt",
            "steps": 2000,
            "lr": [0.1, 0.01, 0.001],
            "params": ["mesh_ms", "conn_offset_ms", "amp", "bias"],
            "min_scaledown": 0
        }
    ]
})";

SegmentationFitOptimizerPanel::SegmentationFitOptimizerPanel(
    const QString& settingsGroup, QWidget* parent)
    : QWidget(parent)
    , _settingsGroup(settingsGroup)
{
    auto* panelLayout = new QVBoxLayout(this);
    panelLayout->setContentsMargins(0, 0, 0, 0);
    panelLayout->setSpacing(0);

    _group = new CollapsibleSettingsGroup(tr("Fit Optimizer"), this);
    auto* content = _group->contentWidget();

    // -- Python executable --
    _group->addRow(tr("Python:"), [&](QHBoxLayout* row) {
        _pythonEdit = new QLineEdit(content);
        _pythonEdit->setPlaceholderText(
            tr("Path to Python (leave empty for auto-detect)"));
        _pythonBrowse = new QToolButton(content);
        _pythonBrowse->setText(QStringLiteral("..."));
        row->addWidget(_pythonEdit, 1);
        row->addWidget(_pythonBrowse);
    }, tr("Python executable with torch installed."));

    // -- Model checkpoint --
    _group->addRow(tr("Model:"), [&](QHBoxLayout* row) {
        _modelEdit = new QLineEdit(content);
        _modelEdit->setPlaceholderText(
            tr("Path to fit model (.pt) — auto-detected from segment"));
        _modelBrowse = new QToolButton(content);
        _modelBrowse->setText(QStringLiteral("..."));
        row->addWidget(_modelEdit, 1);
        row->addWidget(_modelBrowse);
    }, tr("Fit model checkpoint (.pt). Auto-populated from segment's model.pt symlink."));

    // -- Output directory --
    _group->addRow(tr("Output:"), [&](QHBoxLayout* row) {
        _outputEdit = new QLineEdit(content);
        _outputEdit->setPlaceholderText(
            tr("Output directory for tifxyz segments"));
        _outputBrowse = new QToolButton(content);
        _outputBrowse->setText(QStringLiteral("..."));
        row->addWidget(_outputEdit, 1);
        row->addWidget(_outputBrowse);
    }, tr("Directory where re-exported tifxyz segments will be written."));

    // -- JSON config editor --
    auto* configLabel = new QLabel(tr("Optimizer config (JSON):"), content);
    _group->contentLayout()->addWidget(configLabel);

    _configEdit = new QPlainTextEdit(content);
    _configEdit->setPlaceholderText(tr("Paste optimizer JSON config here"));
    _configEdit->setTabStopDistance(20);
    _configEdit->setMinimumHeight(200);
    _configEdit->setLineWrapMode(QPlainTextEdit::NoWrap);
    QFont monoFont("monospace");
    monoFont.setStyleHint(QFont::Monospace);
    monoFont.setPointSize(9);
    _configEdit->setFont(monoFont);
    _group->contentLayout()->addWidget(_configEdit);

    _configStatus = new QLabel(content);
    _configStatus->setWordWrap(true);
    _configStatus->setVisible(false);
    _group->contentLayout()->addWidget(_configStatus);

    // -- Run / Stop / Stop Service buttons --
    auto* btnRow = new QHBoxLayout();
    _runBtn = new QPushButton(tr("Run Optimization"), content);
    _stopBtn = new QPushButton(tr("Stop"), content);
    _stopBtn->setEnabled(false);
    _stopServiceBtn = new QPushButton(tr("Stop Service"), content);
    _stopServiceBtn->setEnabled(false);
    btnRow->addWidget(_runBtn);
    btnRow->addWidget(_stopBtn);
    btnRow->addWidget(_stopServiceBtn);
    btnRow->addStretch(1);
    _group->contentLayout()->addLayout(btnRow);

    // -- Progress label --
    _progressLabel = new QLabel(content);
    _progressLabel->setWordWrap(true);
    _progressLabel->setVisible(false);
    _group->contentLayout()->addWidget(_progressLabel);

    panelLayout->addWidget(_group);

    // -----------------------------------------------------------------------
    // Signal wiring
    // -----------------------------------------------------------------------

    // Python path
    connect(_pythonEdit, &QLineEdit::textChanged, this, [this](const QString& text) {
        _fitPythonPath = text.trimmed();
        writeSetting(QStringLiteral("fit_python_path"), _fitPythonPath);
    });
    connect(_pythonBrowse, &QToolButton::clicked, this, [this]() {
        QString initial = _fitPythonPath.isEmpty()
            ? QDir::homePath() : QFileInfo(_fitPythonPath).absolutePath();
        QString file = QFileDialog::getOpenFileName(
            this, tr("Select Python executable"), initial, tr("All Files (*)"));
        if (!file.isEmpty()) {
            _fitPythonPath = file;
            _pythonEdit->setText(file);
        }
    });

    // Model path
    connect(_modelEdit, &QLineEdit::textChanged, this, [this](const QString& text) {
        _fitModelPath = text.trimmed();
        writeSetting(QStringLiteral("fit_model_path"), _fitModelPath);
    });
    connect(_modelBrowse, &QToolButton::clicked, this, [this]() {
        QString initial = _fitModelPath.isEmpty()
            ? QDir::homePath() : QFileInfo(_fitModelPath).absolutePath();
        QString file = QFileDialog::getOpenFileName(
            this, tr("Select fit model checkpoint"), initial,
            tr("PyTorch Checkpoint (*.pt *.pth);;All Files (*)"));
        if (!file.isEmpty()) {
            _fitModelPath = file;
            _modelEdit->setText(file);
        }
    });

    // Output dir
    connect(_outputEdit, &QLineEdit::textChanged, this, [this](const QString& text) {
        _fitOutputDir = text.trimmed();
        writeSetting(QStringLiteral("fit_output_dir"), _fitOutputDir);
    });
    connect(_outputBrowse, &QToolButton::clicked, this, [this]() {
        QString initial = _fitOutputDir.isEmpty()
            ? QDir::homePath() : _fitOutputDir;
        QString dir = QFileDialog::getExistingDirectory(
            this, tr("Select output directory"), initial);
        if (!dir.isEmpty()) {
            _fitOutputDir = dir;
            _outputEdit->setText(dir);
        }
    });

    // Config editor
    connect(_configEdit, &QPlainTextEdit::textChanged, this, [this]() {
        if (_restoringSettings) return;
        _fitConfigText = _configEdit->toPlainText();
        writeSetting(QStringLiteral("fit_config_text"), _fitConfigText);
        validateConfigText();
    });

    // Run button
    connect(_runBtn, &QPushButton::clicked, this, [this]() {
        validateConfigText();
        if (!_configError.isEmpty()) {
            _progressLabel->setText(tr("Fix JSON errors before running."));
            _progressLabel->setStyleSheet(QStringLiteral("color: #c0392b;"));
            _progressLabel->setVisible(true);
            return;
        }
        emit fitOptimizeRequested();
    });

    // Stop button
    connect(_stopBtn, &QPushButton::clicked, this, [this]() {
        emit fitStopRequested();
    });

    // Stop Service button — kills the Python process entirely
    connect(_stopServiceBtn, &QPushButton::clicked, this, []() {
        FitServiceManager::instance().stopService();
    });

    // Connect to service manager signals
    auto& mgr = FitServiceManager::instance();
    connect(&mgr, &FitServiceManager::statusMessage, this, [this](const QString& msg) {
        if (_progressLabel) {
            _progressLabel->setText(msg);
            _progressLabel->setStyleSheet(QString());
            _progressLabel->setVisible(true);
        }
        emit fitStatusMessage(msg);
    });
    connect(&mgr, &FitServiceManager::serviceStarted, this, [this]() {
        if (_progressLabel) {
            _progressLabel->setText(tr("Service running"));
            _progressLabel->setStyleSheet(QStringLiteral("color: #27ae60;"));
        }
        if (_stopServiceBtn) _stopServiceBtn->setEnabled(true);
    });
    connect(&mgr, &FitServiceManager::serviceStopped, this, [this]() {
        if (_progressLabel) {
            _progressLabel->setText(tr("Service stopped"));
            _progressLabel->setStyleSheet(QString());
        }
        if (_stopBtn) _stopBtn->setEnabled(false);
        if (_stopServiceBtn) _stopServiceBtn->setEnabled(false);
        if (_runBtn) _runBtn->setEnabled(true);
    });
    connect(&mgr, &FitServiceManager::serviceError, this, [this](const QString& err) {
        if (_progressLabel) {
            _progressLabel->setText(tr("Error: %1").arg(err));
            _progressLabel->setStyleSheet(QStringLiteral("color: #c0392b;"));
            _progressLabel->setVisible(true);
        }
    });
    connect(&mgr, &FitServiceManager::optimizationStarted, this, [this]() {
        if (_stopBtn) _stopBtn->setEnabled(true);
        if (_runBtn) _runBtn->setEnabled(false);
        if (_progressLabel) {
            _progressLabel->setText(tr("Optimization started..."));
            _progressLabel->setStyleSheet(QString());
            _progressLabel->setVisible(true);
        }
    });
    connect(&mgr, &FitServiceManager::optimizationProgress, this,
            [this](const QString& stage, int step, int /*total*/, double loss) {
        if (_progressLabel) {
            _progressLabel->setText(
                tr("Stage: %1  |  Step: %2  |  Loss: %3")
                    .arg(stage)
                    .arg(step)
                    .arg(loss, 0, 'g', 5));
            _progressLabel->setStyleSheet(QString());
            _progressLabel->setVisible(true);
        }
    });
    connect(&mgr, &FitServiceManager::optimizationFinished, this,
            [this](const QString& outputDir) {
        if (_stopBtn) _stopBtn->setEnabled(false);
        if (_runBtn) _runBtn->setEnabled(true);
        if (_progressLabel) {
            _progressLabel->setText(tr("Optimization finished. Output: %1").arg(outputDir));
            _progressLabel->setStyleSheet(QStringLiteral("color: #27ae60;"));
            _progressLabel->setVisible(true);
        }
    });
    connect(&mgr, &FitServiceManager::optimizationError, this,
            [this](const QString& err) {
        if (_stopBtn) _stopBtn->setEnabled(false);
        if (_runBtn) _runBtn->setEnabled(true);
        if (_progressLabel) {
            _progressLabel->setText(tr("Optimization error: %1").arg(err));
            _progressLabel->setStyleSheet(QStringLiteral("color: #c0392b;"));
            _progressLabel->setVisible(true);
        }
    });
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

void SegmentationFitOptimizerPanel::writeSetting(const QString& key, const QVariant& value)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(_settingsGroup);
    settings.setValue(key, value);
    settings.endGroup();
}

void SegmentationFitOptimizerPanel::restoreSettings(QSettings& settings)
{
    _restoringSettings = true;

    _fitPythonPath = settings.value(QStringLiteral("fit_python_path"), QString()).toString();
    _fitModelPath = settings.value(QStringLiteral("fit_model_path"), QString()).toString();
    _fitOutputDir = settings.value(QStringLiteral("fit_output_dir"), QString()).toString();
    _fitConfigText = settings.value(QStringLiteral("fit_config_text"), QString()).toString();

    if (_fitConfigText.trimmed().isEmpty()) {
        _fitConfigText = QString::fromUtf8(kDefaultConfig);
    }

    const bool expanded = settings.value(
        QStringLiteral("group_fit_optimizer_expanded"), false).toBool();
    if (_group) {
        _group->setExpanded(expanded);
    }

    _restoringSettings = false;
}

void SegmentationFitOptimizerPanel::syncUiState(bool /*editingEnabled*/, bool optimizing)
{
    if (_pythonEdit) {
        const QSignalBlocker b(_pythonEdit);
        _pythonEdit->setText(_fitPythonPath);
    }
    if (_modelEdit) {
        const QSignalBlocker b(_modelEdit);
        _modelEdit->setText(_fitModelPath);
    }
    if (_outputEdit) {
        const QSignalBlocker b(_outputEdit);
        _outputEdit->setText(_fitOutputDir);
    }
    if (_configEdit && _configEdit->toPlainText() != _fitConfigText) {
        const QSignalBlocker b(_configEdit);
        _configEdit->setPlainText(_fitConfigText);
    }

    if (_runBtn) _runBtn->setEnabled(!optimizing);
    if (_stopBtn) _stopBtn->setEnabled(optimizing);

    validateConfigText();
}

// ---------------------------------------------------------------------------
// Setters
// ---------------------------------------------------------------------------

void SegmentationFitOptimizerPanel::setFitPythonPath(const QString& path)
{
    if (_fitPythonPath == path) return;
    _fitPythonPath = path;
    writeSetting(QStringLiteral("fit_python_path"), _fitPythonPath);
    if (_pythonEdit) {
        const QSignalBlocker b(_pythonEdit);
        _pythonEdit->setText(path);
    }
}

void SegmentationFitOptimizerPanel::setFitModelPath(const QString& path)
{
    if (_fitModelPath == path) return;
    _fitModelPath = path;
    writeSetting(QStringLiteral("fit_model_path"), _fitModelPath);
    if (_modelEdit) {
        const QSignalBlocker b(_modelEdit);
        _modelEdit->setText(path);
    }
}

void SegmentationFitOptimizerPanel::setFitOutputDir(const QString& path)
{
    if (_fitOutputDir == path) return;
    _fitOutputDir = path;
    writeSetting(QStringLiteral("fit_output_dir"), _fitOutputDir);
    if (_outputEdit) {
        const QSignalBlocker b(_outputEdit);
        _outputEdit->setText(path);
    }
}

// ---------------------------------------------------------------------------
// Config JSON
// ---------------------------------------------------------------------------

void SegmentationFitOptimizerPanel::validateConfigText()
{
    _configError.clear();

    QString trimmed = _fitConfigText.trimmed();
    if (trimmed.isEmpty()) {
        if (_configStatus) _configStatus->setVisible(false);
        return;
    }

    try {
        QByteArray utf8 = trimmed.toUtf8();
        nlohmann::json parsed = nlohmann::json::parse(
            utf8.constData(), utf8.constData() + utf8.size());
        if (!parsed.is_object()) {
            _configError = tr("Config must be a JSON object.");
        }
    } catch (const nlohmann::json::parse_error& ex) {
        _configError = tr("JSON parse error (byte %1): %2")
                           .arg(static_cast<qulonglong>(ex.byte))
                           .arg(QString::fromStdString(ex.what()));
    } catch (const std::exception& ex) {
        _configError = tr("JSON error: %1").arg(QString::fromStdString(ex.what()));
    }

    if (_configStatus) {
        if (_configError.isEmpty()) {
            _configStatus->setText(tr("JSON valid"));
            _configStatus->setStyleSheet(QStringLiteral("color: #27ae60;"));
        } else {
            _configStatus->setText(_configError);
            _configStatus->setStyleSheet(QStringLiteral("color: #c0392b;"));
        }
        _configStatus->setVisible(true);
    }
}

std::optional<nlohmann::json> SegmentationFitOptimizerPanel::fitConfigJson() const
{
    QString trimmed = _fitConfigText.trimmed();
    if (trimmed.isEmpty()) return std::nullopt;

    try {
        QByteArray utf8 = trimmed.toUtf8();
        nlohmann::json parsed = nlohmann::json::parse(
            utf8.constData(), utf8.constData() + utf8.size());
        if (parsed.is_object()) return parsed;
    } catch (...) {}

    return std::nullopt;
}
