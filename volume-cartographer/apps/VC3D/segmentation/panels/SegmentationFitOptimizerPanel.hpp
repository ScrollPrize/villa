#pragma once

#include <QWidget>

#include <optional>

#include <nlohmann/json_fwd.hpp>

class CollapsibleSettingsGroup;
class QComboBox;
class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QProgressBar;
class QPushButton;
class QSettings;
class QStackedWidget;
class QToolButton;
class QWidget;

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
    [[nodiscard]] QString fitDataInputPath() const { return _fitDataInputPath; }
    [[nodiscard]] QString fitOutputDir() const { return _fitOutputDir; }
    [[nodiscard]] QString fitConfigText() const { return _fitConfigText; }
    [[nodiscard]] std::optional<nlohmann::json> fitConfigJson() const;

    // Setters
    void setFitPythonPath(const QString& path);
    void setFitModelPath(const QString& path);
    void setFitDataInputPath(const QString& path);
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
    void loadProfile(int index);
    void onConnectionModeChanged(int index);
    void refreshDiscoveredServices();
    void onDiscoveredServiceSelected(int index);
    void updateConnectionWidgets();

    CollapsibleSettingsGroup* _group{nullptr};

    // Connection mode
    QComboBox* _connectionCombo{nullptr};
    QWidget* _internalWidget{nullptr};   // contains python path row
    QWidget* _externalWidget{nullptr};   // contains discovery + host/port

    // External service widgets
    QComboBox* _discoveryCombo{nullptr};
    QToolButton* _refreshBtn{nullptr};
    QLineEdit* _hostEdit{nullptr};
    QLineEdit* _portEdit{nullptr};

    // Data input with dataset combo support
    QComboBox* _datasetCombo{nullptr};
    QStackedWidget* _dataInputStack{nullptr};

    QComboBox* _profileCombo{nullptr};
    QLineEdit* _pythonEdit{nullptr};
    QToolButton* _pythonBrowse{nullptr};
    QLineEdit* _modelEdit{nullptr};
    QToolButton* _modelBrowse{nullptr};
    QLineEdit* _dataInputEdit{nullptr};
    QToolButton* _dataInputBrowse{nullptr};
    QLineEdit* _outputEdit{nullptr};
    QToolButton* _outputBrowse{nullptr};
    QPlainTextEdit* _configEdit{nullptr};
    QLabel* _configStatus{nullptr};
    QPushButton* _runBtn{nullptr};
    QPushButton* _stopBtn{nullptr};
    QPushButton* _stopServiceBtn{nullptr};
    QProgressBar* _progressBar{nullptr};
    QLabel* _progressLabel{nullptr};

    QString _fitPythonPath;
    QString _fitModelPath;
    QString _fitDataInputPath;
    QString _fitOutputDir;
    QString _fitConfigText;
    QString _configError;

    int _connectionMode{0};  // 0=internal, 1=external
    QString _externalHost{"127.0.0.1"};
    int _externalPort{9999};

    bool _restoringSettings{false};
    const QString _settingsGroup;
};
