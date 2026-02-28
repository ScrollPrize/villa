#pragma once

#include "segmentation/SegmentationCommon.hpp"

#include <QColor>
#include <QPair>
#include <QString>
#include <QVector>
#include <QWidget>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QPushButton;
class QSettings;
class QSlider;
class CollapsibleSettingsGroup;

class SegmentationApprovalMaskPanel : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationApprovalMaskPanel(const QString& settingsGroup,
                                           QWidget* parent = nullptr);

    // Getters
    [[nodiscard]] bool showApprovalMask() const { return _showApprovalMask; }
    [[nodiscard]] bool editApprovedMask() const { return _editApprovedMask; }
    [[nodiscard]] bool editUnapprovedMask() const { return _editUnapprovedMask; }
    [[nodiscard]] bool autoApproveEdits() const { return _autoApprovalEnabled; }
    [[nodiscard]] bool autoApprovalEnabled() const { return _autoApprovalEnabled; }
    [[nodiscard]] float autoApprovalRadius() const { return _autoApprovalRadius; }
    [[nodiscard]] float autoApprovalThreshold() const { return _autoApprovalThreshold; }
    [[nodiscard]] float autoApprovalMaxDistance() const { return _autoApprovalMaxDistance; }

    [[nodiscard]] float approvalBrushRadius() const { return _approvalBrushRadius; }
    [[nodiscard]] float approvalBrushDepth() const { return _approvalBrushDepth; }
    [[nodiscard]] ApprovalBrushShape approvalBrushShape() const { return _approvalBrushShape; }
    [[nodiscard]] int approvalMaskOpacity() const { return _approvalMaskOpacity; }
    [[nodiscard]] QColor approvalBrushColor() const { return _approvalBrushColor; }
    [[nodiscard]] QString selectedApprovalMaskId() const { return _selectedApprovalMaskId; }

    // Setters
    void setShowApprovalMask(bool enabled);
    void setEditApprovedMask(bool enabled);
    void setEditUnapprovedMask(bool enabled);
    void setAutoApproveEdits(bool enabled);
    void setAutoApprovalEnabled(bool enabled);
    void setAutoApprovalRadius(float radius);
    void setAutoApprovalThreshold(float threshold);
    void setAutoApprovalMaxDistance(float distance);

    void setApprovalBrushRadius(float radius);
    void setApprovalBrushDepth(float depth);
    void setApprovalBrushShape(ApprovalBrushShape shape);
    void setApprovalMaskOpacity(int opacity);
    void setApprovalBrushColor(const QColor& color);
    void setApprovalMaskOptions(const QVector<QPair<QString, QString>>& options);
    void setSelectedApprovalMaskId(const QString& maskId);

    void restoreSettings(QSettings& settings);
    void syncUiState();

signals:
    void showApprovalMaskChanged(bool enabled);
    void editApprovedMaskChanged(bool enabled);
    void editUnapprovedMaskChanged(bool enabled);
    void autoApproveEditsChanged(bool enabled);
    void autoApprovalEnabledChanged(bool enabled);
    void autoApprovalRadiusChanged(float radius);
    void autoApprovalThresholdChanged(float threshold);
    void autoApprovalMaxDistanceChanged(float distance);

    void approvalBrushRadiusChanged(float radius);
    void approvalBrushDepthChanged(float depth);
    void approvalBrushShapeChanged(ApprovalBrushShape shape);
    void approvalMaskOpacityChanged(int opacity);
    void approvalBrushColorChanged(QColor color);
    void approvalStrokesUndoRequested();
    void approvalMaskSelected(const QString& maskId);
    void approvalMaskCreateRequested(const QString& displayName);
    void copyMaskedForwardRequested();
    void copyMaskedBackwardRequested();
    void growAllInMaskedRequested();

private:
    void writeSetting(const QString& key, const QVariant& value);

    CollapsibleSettingsGroup* _groupApprovalMask{nullptr};
    QCheckBox* _chkShowApprovalMask{nullptr};
    QCheckBox* _chkEditApprovedMask{nullptr};
    QCheckBox* _chkEditUnapprovedMask{nullptr};
    QCheckBox* _chkAutoApproveEdits{nullptr};
    QDoubleSpinBox* _spinAutoApprovalRadius{nullptr};
    QDoubleSpinBox* _spinAutoApprovalThreshold{nullptr};
    QDoubleSpinBox* _spinAutoApprovalMaxDistance{nullptr};

    QDoubleSpinBox* _spinApprovalBrushRadius{nullptr};
    QDoubleSpinBox* _spinApprovalBrushDepth{nullptr};
    QComboBox* _comboApprovalBrushShape{nullptr};
    QSlider* _sliderApprovalMaskOpacity{nullptr};
    QLabel* _lblApprovalMaskOpacity{nullptr};
    QPushButton* _btnApprovalColor{nullptr};
    QPushButton* _btnUndoApprovalStroke{nullptr};
    QComboBox* _comboApprovalMask{nullptr};
    QPushButton* _btnNewApprovalMask{nullptr};
    QPushButton* _btnCopyMaskedForward{nullptr};
    QPushButton* _btnCopyMaskedBackward{nullptr};
    QPushButton* _btnGrowAllInMasked{nullptr};

    bool _showApprovalMask{false};
    bool _editApprovedMask{false};
    bool _editUnapprovedMask{false};
    bool _autoApprovalEnabled{true};
    float _autoApprovalRadius{0.5f};
    float _autoApprovalThreshold{0.0f};
    float _autoApprovalMaxDistance{0.0f};

    float _approvalBrushRadius{50.0f};
    float _approvalBrushDepth{15.0f};
    ApprovalBrushShape _approvalBrushShape{ApprovalBrushShape::Rectangle};
    int _approvalMaskOpacity{50};
    QColor _approvalBrushColor{0, 255, 0};
    QString _selectedApprovalMaskId{QStringLiteral("default")};
    QVector<QPair<QString, QString>> _approvalMaskOptions{
        {QStringLiteral("default"), QStringLiteral("approval")}
    };

    bool _restoringSettings{false};
    const QString _settingsGroup;
};
