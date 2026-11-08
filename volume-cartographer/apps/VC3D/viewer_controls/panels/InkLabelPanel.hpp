#pragma once

#include <QPointer>
#include <QWidget>

class CollapsibleSettingsGroup;
class InkLabelController;
class QCheckBox;
class QComboBox;
class QLabel;
class QSpinBox;

namespace vc3d::ink_labels {
struct ImportOutcome;
struct ImportOptions;
}
class QPushButton;
class QSlider;
class QTimer;

// The ink label section of the Ink Detection dock: label mode, class, tool,
// display and file controls for the InkLabelController. Every control is
// Qt::NoFocus, so a click leaves keyboard focus in the view.
class InkLabelPanel : public QWidget
{
    Q_OBJECT

public:
    explicit InkLabelPanel(InkLabelController* labels, QWidget* parent = nullptr);

private:
    void refreshSession();
    void refreshTools();
    void refreshDisplay();
    void refreshHistory();
    void refreshCounts();
    void pushDisplay();
    void refreshAssist();
    void pushAssist();
    void refreshVersions();
    void chooseImport(bool zarrFolder);
    void onImportFinished(const QString& path, const vc3d::ink_labels::ImportOutcome& outcome,
                          const vc3d::ink_labels::ImportOptions& options);

    QPointer<InkLabelController> _labels;
    CollapsibleSettingsGroup* _labelsGroup{nullptr};
    CollapsibleSettingsGroup* _assistGroup{nullptr};
    CollapsibleSettingsGroup* _filesGroup{nullptr};
    QPushButton* _modeButton{nullptr};
    QLabel* _folderLabel{nullptr};
    QLabel* _statusLabel{nullptr};
    QLabel* _bannerLabel{nullptr};
    QPushButton* _inkButton{nullptr};
    QPushButton* _backgroundButton{nullptr};
    QPushButton* _validationButton{nullptr};
    QPushButton* _eraserButton{nullptr};
    QCheckBox* _keepsInkCheck{nullptr};
    QPushButton* _brushButton{nullptr};
    QPushButton* _polygonButton{nullptr};
    QPushButton* _fillButton{nullptr};
    QSlider* _radiusSlider{nullptr};
    QLabel* _radiusLabel{nullptr};
    QCheckBox* _showCheck{nullptr};
    QCheckBox* _showInkCheck{nullptr};
    QCheckBox* _showSupervisionCheck{nullptr};
    QCheckBox* _showValidationCheck{nullptr};
    QSlider* _opacitySlider{nullptr};
    QPushButton* _undoButton{nullptr};
    QPushButton* _redoButton{nullptr};
    QPushButton* _saveButton{nullptr};
    QPushButton* _saveVersionButton{nullptr};
    QPushButton* _importButton{nullptr};
    QPushButton* _commandsButton{nullptr};
    QPushButton* _newVersionButton{nullptr};
    QComboBox* _versionCombo{nullptr};
    QLabel* _countsLabel{nullptr};
    QTimer* _countsTimer{nullptr};
    // Assist section.
    QComboBox* _guideCombo{nullptr};
    QComboBox* _sideCombo{nullptr};
    QSlider* _thresholdSlider{nullptr};
    QLabel* _thresholdLabel{nullptr};
    QComboBox* _connectivityCombo{nullptr};
    QCheckBox* _unlabeledOnlyCheck{nullptr};
    QComboBox* _medianCombo{nullptr};
    QSpinBox* _minAreaSpin{nullptr};
    QPushButton* _suggestButton{nullptr};
    QPushButton* _pickAllButton{nullptr};
    QPushButton* _acceptButton{nullptr};
    QPushButton* _discardButton{nullptr};
    QLabel* _assistLabel{nullptr};
    QSpinBox* _cleanupSpin{nullptr};
    QList<QPushButton*> _cleanupButtons;
};
