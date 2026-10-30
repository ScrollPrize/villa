#pragma once

#include "segmentation/tools/ManualAddTool.hpp"

#include <QWidget>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QPushButton;
class QSettings;
class QSpinBox;

class SegmentationManualAddPanel : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationManualAddPanel(const QString& settingsGroup, QWidget* parent = nullptr);

    [[nodiscard]] ManualAddTool::Config config() const;
    [[nodiscard]] int extendCells() const;
    ManualAddTool::LinePreviewMode cycleLinePreviewMode();
    // Direct changes emit configChanged through the same path as combo edits.
    ManualAddTool::LinePreviewMode setLinePreviewMode(ManualAddTool::LinePreviewMode mode);
    ManualAddTool::InterpolationMode setInterpolationMode(ManualAddTool::InterpolationMode mode);
    void restoreSettings(QSettings& settings);
    void syncUiState(bool editingEnabled, bool manualAddActive);

signals:
    void configChanged();
    void clearPendingRequested();
    void recomputeRequested();
    void applyExitRequested();
    void cancelRequested();
    void extendRequested(int side, int cells);

private:
    void writeSetting(const QString& key, const QVariant& value);
    void persistFromUi();

    QString _settingsGroup;
    bool _restoringSettings{false};

    QSpinBox* _spinMaxPreviewSpan{nullptr};
    QSpinBox* _spinBoundaryBand{nullptr};
    QDoubleSpinBox* _spinRegularization{nullptr};
    QSpinBox* _spinSampleCap{nullptr};
    QSpinBox* _spinPreviewThrottle{nullptr};
    QSpinBox* _spinExtendCells{nullptr};
    QSpinBox* _spinTintOpacity{nullptr};
    QDoubleSpinBox* _spinPlaneConstraintRadius{nullptr};
    QDoubleSpinBox* _spinPlaneConstraintReplacementRadius{nullptr};
    QComboBox* _comboLinePreviewMode{nullptr};
    QComboBox* _comboInterpolationMode{nullptr};
    QCheckBox* _chkIncludeTouchedValidBorder{nullptr};
    QCheckBox* _chkAllowBoundarySmoothing{nullptr};
    QCheckBox* _chkFollowSheet{nullptr};
    QPushButton* _btnClearPending{nullptr};
    QPushButton* _btnRecompute{nullptr};
    QPushButton* _btnApplyExit{nullptr};
    QPushButton* _btnCancel{nullptr};
    QPushButton* _btnExtendUp{nullptr};
    QPushButton* _btnExtendDown{nullptr};
    QPushButton* _btnExtendLeft{nullptr};
    QPushButton* _btnExtendRight{nullptr};
};
