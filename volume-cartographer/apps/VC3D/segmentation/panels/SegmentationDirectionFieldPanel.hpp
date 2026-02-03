#pragma once

#include "segmentation/SegmentationCommon.hpp"
#include "segmentation/growth/SegmentationGrowth.hpp"

#include <QString>
#include <QWidget>

#include <vector>

class QComboBox;
class QDoubleSpinBox;
class QLineEdit;
class QListWidget;
class QPushButton;
class QSettings;
class QToolButton;
class CollapsibleSettingsGroup;

class SegmentationDirectionFieldPanel : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationDirectionFieldPanel(const QString& settingsGroup,
                                             QWidget* parent = nullptr);

    // Getters
    [[nodiscard]] std::vector<SegmentationDirectionFieldConfig> directionFieldConfigs() const;

    // Group accessor (for expand/collapse state persistence by coordinator)
    CollapsibleSettingsGroup* directionFieldGroup() const { return _groupDirectionField; }

    void restoreSettings(QSettings& settings);
    void syncUiState(bool editingEnabled);

signals:
    void directionFieldConfigsChanged();

private:
    void writeSetting(const QString& key, const QVariant& value);
    void refreshDirectionFieldList();
    void persistDirectionFields();
    [[nodiscard]] SegmentationDirectionFieldConfig buildDirectionFieldDraft() const;
    void updateDirectionFieldFormFromSelection(int row);
    void applyDirectionFieldDraftToSelection(int row);
    void updateDirectionFieldListItem(int row);
    void updateDirectionFieldListGeometry();
    void clearDirectionFieldForm();

    CollapsibleSettingsGroup* _groupDirectionField{nullptr};
    QLineEdit* _directionFieldPathEdit{nullptr};
    QToolButton* _directionFieldBrowseButton{nullptr};
    QComboBox* _comboDirectionFieldOrientation{nullptr};
    QComboBox* _comboDirectionFieldScale{nullptr};
    QDoubleSpinBox* _spinDirectionFieldWeight{nullptr};
    QPushButton* _directionFieldAddButton{nullptr};
    QPushButton* _directionFieldRemoveButton{nullptr};
    QPushButton* _directionFieldClearButton{nullptr};
    QListWidget* _directionFieldList{nullptr};

    QString _directionFieldPath;
    SegmentationDirectionFieldOrientation _directionFieldOrientation{SegmentationDirectionFieldOrientation::Normal};
    int _directionFieldScale{0};
    double _directionFieldWeight{1.0};
    std::vector<SegmentationDirectionFieldConfig> _directionFields;
    bool _updatingDirectionFieldForm{false};
    bool _editingEnabled{false};

    bool _restoringSettings{false};
    const QString _settingsGroup;
};
