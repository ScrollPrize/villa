#pragma once

#include <QSet>
#include <QMetaObject>
#include <QStringList>
#include <QWidget>

#include <optional>

#include <opencv2/core.hpp>

#include "utils/Json.hpp"

class CollapsibleSettingsGroup;
class CState;
class LasagnaBatchWindow;
class QComboBox;
class QLabel;
class QLineEdit;
class QProgressBar;
class QPushButton;
class QStatusBar;
class QSettings;
class QDoubleSpinBox;
class QSpinBox;
class QStandardItemModel;
class QStackedWidget;
class QTableView;
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

    enum LasagnaMode { ReOptimize = 0, NewModel = 1, Offset = 3, Atlas = 4 };

    // Getters
    [[nodiscard]] QString lasagnaDataInputPath() const { return _lasagnaDataInputPath; }
    /** Reads the selected config JSON file from disk and returns its contents. */
    [[nodiscard]] QString lasagnaConfigText() const;
    [[nodiscard]] utils::Json lasagnaConfigJson() const;
    [[nodiscard]] LasagnaMode lasagnaMode() const { return static_cast<LasagnaMode>(_lasagnaMode); }
    [[nodiscard]] double newModelWidth() const;
    [[nodiscard]] QString newModelWidthUnit() const;
    [[nodiscard]] int newModelHeight() const;
    [[nodiscard]] int newModelWindings() const;
    [[nodiscard]] QString seedPointText() const;
    [[nodiscard]] QString newModelOutputName() const;
    [[nodiscard]] double offsetValue() const;

    // Setters
    void setLasagnaDataInputPath(const QString& path);
    void setSelectedAtlasPath(const QString& path);
    void setState(CState* state);

    void restoreSettings(QSettings& settings);
    void syncUiState(bool editingEnabled, bool optimizing);
    QWidget* createCompactView(QWidget* parent = nullptr);
    void repeatLastLasagnaAction();
    void startOptimization(CState* state, QStatusBar* statusBar);
    void startOptimizationAtSeed(CState* state,
                                 QStatusBar* statusBar,
                                 LasagnaMode mode,
                                 const QString& configPath,
                                 int seedX,
                                 int seedY,
                                 int seedZ);
    /// Starts optimization without dialogs and reports failures through
    /// errorMessage. Returns true when the optimization was submitted.
    ///   configPath: empty => selectedLasagnaConfigPathForMode(mode)
    ///   seed:       nullopt => panel seed / focus (no seed override)
    ///   atlasPath:  empty => panel selection; applied via setSelectedAtlasPath
    bool startOptimizationHeadless(CState* state,
                                   LasagnaMode mode,
                                   const QString& configPath,
                                   std::optional<cv::Vec3i> seed,
                                   const QString& atlasPath,
                                   QString* errorMessage = nullptr);
    /// Repeats the last action without entering the interactive signal chain.
    bool repeatLastLasagnaActionHeadless(CState* state, QString* errorMessage = nullptr);
    [[nodiscard]] QString selectedLasagnaConfigPathForMode(LasagnaMode mode) const;
    [[nodiscard]] QStringList lasagnaConfigPathsForMode(LasagnaMode mode) const;
    [[nodiscard]] QString selectedAtlasPath() const { return _atlasDirPath; }
    [[nodiscard]] QString selectedAtlasConfigPath() const { return _atlasConfigFilePath; }

public slots:
    void setSeedFromFocus(int x, int y, int z);

signals:
    void lasagnaOptimizeRequested();
    void lasagnaStopRequested();
    void lasagnaStatusMessage(const QString& message);
    void seedFromFocusRequested();
    void openLasagnaWorkspaceRequested();
    void lasagnaOutputActivated(const QString& outputName);

private:
    enum class Presentation {
        Interactive,
        Silent
    };

    struct SubmissionResult {
        bool submitted{false};
        QString error;

        static SubmissionResult success()
        {
            return {.submitted = true};
        }

        static SubmissionResult failure(const QString& error)
        {
            return {.submitted = false, .error = error};
        }
    };

    void writeSetting(const QString& key, const QVariant& value);
    void populateConfigCombo(QComboBox* combo, const QString& dir,
                             const QString& selectName, QString& outPath,
                             bool growOnly = false);
    void onConnectionModeChanged(int index);
    void refreshDiscoveredServices();
    void onDiscoveredServiceSelected(int index);
    void updateConnectionWidgets();
    void triggerOptimization();
    void launchLasagnaMode(LasagnaMode mode);
    void launchAtlasOptimization();
    [[nodiscard]] SubmissionResult startAtlasOptimization(
        CState* state,
        QStatusBar* statusBar,
        Presentation presentation = Presentation::Interactive);
    void syncCompactConfigCombos();
    void syncCompactStatusFromFull();
    void updateCompactLinkedSurfaceTable(const QStringList& names);
    void updateLinkedSurfaceTables();
    [[nodiscard]] QStringList currentLinkedSurfaceNames() const;
    void showLasagnaConfigError(const QString& message,
                                QStatusBar* statusBar,
                                int timeoutMs,
                                Presentation presentation = Presentation::Interactive);
    [[nodiscard]] bool validateLasagnaConfigPath(const QString& configPath,
                                                 QStatusBar* statusBar,
                                                 Presentation presentation =
                                                     Presentation::Interactive);
    [[nodiscard]] bool validateAtlasDirPath(const QString& atlasDir,
                                            QStatusBar* statusBar,
                                            Presentation presentation =
                                                Presentation::Interactive);
    void populateAtlasCombo(const QString& volpkgRoot, const QString& selectPath);
    void refreshAtlasComboFromState();
    [[nodiscard]] SubmissionResult startOptimizationWithOverrides(
        CState* state,
        QStatusBar* statusBar,
        int modeOverride,
        const QString& configPathOverride,
        bool hasSeedOverride,
        int seedX,
        int seedY,
        int seedZ,
        Presentation presentation = Presentation::Interactive);

    // -- Sections --
    CollapsibleSettingsGroup* _connectionGroup{nullptr};
    CollapsibleSettingsGroup* _newModelGroup{nullptr};
    CollapsibleSettingsGroup* _reoptGroup{nullptr};
    CollapsibleSettingsGroup* _offsetGroup{nullptr};
    CollapsibleSettingsGroup* _atlasGroup{nullptr};

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
    QDoubleSpinBox* _widthSpin{nullptr};
    QComboBox* _widthUnitCombo{nullptr};
    QSpinBox* _heightSpin{nullptr};
    QSpinBox* _windingsSpin{nullptr};
    QLineEdit* _seedEdit{nullptr};
    QPushButton* _seedFromFocusBtn{nullptr};
    QLineEdit* _outputNameEdit{nullptr};

    // Config combos (one per section)
    QComboBox* _newModelConfigCombo{nullptr};
    QToolButton* _newModelConfigBrowse{nullptr};
    QComboBox* _reoptConfigCombo{nullptr};
    QToolButton* _reoptConfigBrowse{nullptr};
    QComboBox* _offsetConfigCombo{nullptr};
    QToolButton* _offsetConfigBrowse{nullptr};
    QDoubleSpinBox* _offsetValueSpin{nullptr};
    QComboBox* _atlasCombo{nullptr};
    QComboBox* _atlasConfigCombo{nullptr};
    QToolButton* _atlasConfigBrowse{nullptr};

    // Action buttons
    QPushButton* _newModelBtn{nullptr};
    QPushButton* _reoptBtn{nullptr};
    QPushButton* _offsetBtn{nullptr};
    QPushButton* _atlasBtn{nullptr};
    QPushButton* _stopBtn{nullptr};
    QPushButton* _stopServiceBtn{nullptr};

    QProgressBar* _progressBar{nullptr};
    QLabel* _progressLabel{nullptr};
    LasagnaBatchWindow* _batchWindow{nullptr};

    QWidget* _compactView{nullptr};
    QComboBox* _compactNewModelConfigCombo{nullptr};
    QComboBox* _compactReoptConfigCombo{nullptr};
    QComboBox* _compactAtlasCombo{nullptr};
    QComboBox* _compactAtlasConfigCombo{nullptr};
    QPushButton* _compactNewModelBtn{nullptr};
    QPushButton* _compactReoptBtn{nullptr};
    QPushButton* _compactAtlasBtn{nullptr};
    QPushButton* _compactStopBtn{nullptr};
    QPushButton* _compactStopServiceBtn{nullptr};
    QProgressBar* _compactProgressBar{nullptr};
    QLabel* _compactProgressLabel{nullptr};
    QTableView* _compactLinkedSurfaceTable{nullptr};
    QStandardItemModel* _compactLinkedSurfaceModel{nullptr};

    QString _lasagnaDataInputPath;
    QString _newModelConfigFilePath;
    QString _reoptConfigFilePath;
    QString _offsetConfigFilePath;
    QString _atlasConfigFilePath;
    QString _atlasDirPath;

    int _lasagnaMode{0};         // 0=re-optimize, 1=new model, 2=expand, 3=offset
    LasagnaMode _lastLasagnaMode{LasagnaMode::ReOptimize};

    int _connectionMode{0};  // 0=internal, 1=external
    QString _externalHost{"127.0.0.1"};
    int _externalPort{9999};

    bool _restoringSettings{false};
    const QString _settingsGroup;
    CState* _state{nullptr};
    QMetaObject::Connection _stateVpkgChangedConnection;
    QMetaObject::Connection _stateSurfaceChangedConnection;
    QSet<QString> _submittedOutputNames;
};
