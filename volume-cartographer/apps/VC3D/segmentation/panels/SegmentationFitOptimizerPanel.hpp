#pragma once

#include <QWidget>

#include <optional>

#include <nlohmann/json_fwd.hpp>

class CollapsibleSettingsGroup;
class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QPushButton;
class QSettings;
class QToolButton;

/**
 * Segmentation sidebar panel for the 2D fit optimizer.
 *
 * Displays:
 *   - Python executable path
 *   - Model checkpoint path (auto-populated from segment's model.pt symlink)
 *   - Editable JSON config for the optimizer (base weights, stages, args)
 *   - Run / Stop buttons
 *   - Progress status label
 */
class SegmentationFitOptimizerPanel : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationFitOptimizerPanel(const QString& settingsGroup,
                                            QWidget* parent = nullptr);

    // Getters
    [[nodiscard]] QString fitPythonPath() const { return _fitPythonPath; }
    [[nodiscard]] QString fitModelPath() const { return _fitModelPath; }
    [[nodiscard]] QString fitOutputDir() const { return _fitOutputDir; }
    [[nodiscard]] QString fitConfigText() const { return _fitConfigText; }
    [[nodiscard]] std::optional<nlohmann::json> fitConfigJson() const;

    // Setters
    void setFitPythonPath(const QString& path);
    void setFitModelPath(const QString& path);
    void setFitOutputDir(const QString& path);

    void restoreSettings(QSettings& settings);
    void syncUiState(bool editingEnabled, bool optimizing);

signals:
    void fitOptimizeRequested();
    void fitStopRequested();
    void fitStatusMessage(const QString& message);

private:
    void writeSetting(const QString& key, const QVariant& value);
    void validateConfigText();

    CollapsibleSettingsGroup* _group{nullptr};

    QLineEdit* _pythonEdit{nullptr};
    QToolButton* _pythonBrowse{nullptr};
    QLineEdit* _modelEdit{nullptr};
    QToolButton* _modelBrowse{nullptr};
    QLineEdit* _outputEdit{nullptr};
    QToolButton* _outputBrowse{nullptr};
    QPlainTextEdit* _configEdit{nullptr};
    QLabel* _configStatus{nullptr};
    QPushButton* _runBtn{nullptr};
    QPushButton* _stopBtn{nullptr};
    QLabel* _progressLabel{nullptr};

    QString _fitPythonPath;
    QString _fitModelPath;
    QString _fitOutputDir;
    QString _fitConfigText;
    QString _configError;

    bool _restoringSettings{false};
    const QString _settingsGroup;
};
