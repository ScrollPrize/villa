#pragma once

#include <QWidget>

#include <optional>

#include <nlohmann/json_fwd.hpp>

class CollapsibleSettingsGroup;
class QComboBox;
class QLabel;
class QLineEdit;
class QProgressBar;
class QPushButton;
class QSettings;
class QSpinBox;
class QStackedWidget;
class QToolButton;
class QWidget;

/**
 * Segmentation sidebar panel for the 2D lasagna.
 *
 * Sections:
 *   - Connection  (expandable) — connection mode + data input
 *   - New Model   (button + expandable settings)
 *   - Re-optimize (button + expandable settings)
 *   - Shared: stop buttons, progress bar/label
 */
class SegmentationLasagnaPanel : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationLasagnaPanel(const QString& settingsGroup,
                                            QWidget* parent = nullptr);

    /** 0 = Re-optimize, 1 = New Model */
    enum LasagnaMode { ReOptimize = 0, NewModel = 1 };

    // Getters
    [[nodiscard]] QString lasagnaDataInputPath() const { return _lasagnaDataInputPath; }
    /** Reads the selected config JSON file from disk and returns its contents. */
    [[nodiscard]] QString lasagnaConfigText() const;
    [[nodiscard]] std::optional<nlohmann::json> lasagnaConfigJson() const;
    [[nodiscard]] LasagnaMode lasagnaMode() const { return static_cast<LasagnaMode>(_lasagnaMode); }
    [[nodiscard]] int newModelWidth() const;
    [[nodiscard]] int newModelHeight() const;
    [[nodiscard]] int newModelDepth() const;
    [[nodiscard]] QString seedPointText() const;
    [[nodiscard]] QString newModelOutputName() const;

    // Setters
    void setLasagnaDataInputPath(const QString& path);

    void restoreSettings(QSettings& settings);
    void syncUiState(bool editingEnabled, bool optimizing);

public slots:
    void setSeedFromFocus(int x, int y, int z);

signals:
    void lasagnaOptimizeRequested();
    void lasagnaStopRequested();
    void lasagnaStatusMessage(const QString& message);
    void seedFromFocusRequested();

private:
    void writeSetting(const QString& key, const QVariant& value);
    void populateConfigCombo(QComboBox* combo, const QString& dir,
                             const QString& selectName, QString& outPath);
    void onConnectionModeChanged(int index);
    void refreshDiscoveredServices();
    void onDiscoveredServiceSelected(int index);
    void updateConnectionWidgets();
    void triggerOptimization();

    // -- Sections --
    CollapsibleSettingsGroup* _connectionGroup{nullptr};
    CollapsibleSettingsGroup* _newModelGroup{nullptr};
    CollapsibleSettingsGroup* _reoptGroup{nullptr};

    // Connection mode
    QComboBox* _connectionCombo{nullptr};
    QWidget* _externalWidget{nullptr};   // contains discovery + host/port

    // External service widgets
    QComboBox* _discoveryCombo{nullptr};
    QToolButton* _refreshBtn{nullptr};
    QWidget* _hostPortWidget{nullptr};   // host/port row (hidden when discovered)
    QLineEdit* _hostEdit{nullptr};
    QLineEdit* _portEdit{nullptr};

    // Data input with dataset combo support
    QComboBox* _datasetCombo{nullptr};
    QStackedWidget* _dataInputStack{nullptr};
    QLineEdit* _dataInputEdit{nullptr};
    QToolButton* _dataInputBrowse{nullptr};

    // New model settings
    QSpinBox* _widthSpin{nullptr};
    QSpinBox* _heightSpin{nullptr};
    QSpinBox* _depthSpin{nullptr};
    QLineEdit* _seedEdit{nullptr};
    QPushButton* _seedFromFocusBtn{nullptr};
    QLineEdit* _outputNameEdit{nullptr};

    // Config combos (one per section)
    QComboBox* _newModelConfigCombo{nullptr};
    QToolButton* _newModelConfigBrowse{nullptr};
    QComboBox* _reoptConfigCombo{nullptr};
    QToolButton* _reoptConfigBrowse{nullptr};

    // Action buttons
    QPushButton* _newModelBtn{nullptr};
    QPushButton* _reoptBtn{nullptr};
    QPushButton* _stopBtn{nullptr};
    QPushButton* _stopServiceBtn{nullptr};

    QProgressBar* _progressBar{nullptr};
    QLabel* _progressLabel{nullptr};

    QString _lasagnaDataInputPath;
    QString _newModelConfigFilePath;
    QString _reoptConfigFilePath;

    int _lasagnaMode{0};         // 0=re-optimize, 1=new model
    int _connectionMode{0};  // 0=internal, 1=external
    QString _externalHost{"127.0.0.1"};
    int _externalPort{9999};

    bool _restoringSettings{false};
    const QString _settingsGroup;
};
