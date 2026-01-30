#pragma once

#include "segmentation/SegmentationCommon.hpp"

#include <QString>
#include <QVector>
#include <QWidget>

#include <cstdint>
#include <utility>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QPushButton;
class QSettings;
class QSpinBox;
class CollapsibleSettingsGroup;

class SegmentationCellReoptPanel : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationCellReoptPanel(const QString& settingsGroup,
                                        QWidget* parent = nullptr);

    // Getters
    [[nodiscard]] bool cellReoptMode() const { return _cellReoptMode; }
    [[nodiscard]] int cellReoptMaxSteps() const { return _cellReoptMaxSteps; }
    [[nodiscard]] int cellReoptMaxPoints() const { return _cellReoptMaxPoints; }
    [[nodiscard]] float cellReoptMinSpacing() const { return _cellReoptMinSpacing; }
    [[nodiscard]] float cellReoptPerimeterOffset() const { return _cellReoptPerimeterOffset; }

    // Setters
    void setCellReoptMode(bool enabled);
    void setCellReoptCollections(const QVector<QPair<uint64_t, QString>>& collections);

    void restoreSettings(QSettings& settings);
    void syncUiState(bool showApprovalMask, bool growthInProgress);

signals:
    void cellReoptModeChanged(bool enabled);
    void cellReoptMaxStepsChanged(int steps);
    void cellReoptMaxPointsChanged(int points);
    void cellReoptMinSpacingChanged(float spacing);
    void cellReoptPerimeterOffsetChanged(float offset);
    void cellReoptGrowthRequested(uint64_t collectionId);

private:
    void writeSetting(const QString& key, const QVariant& value);

    CollapsibleSettingsGroup* _groupCellReopt{nullptr};
    QCheckBox* _chkCellReoptMode{nullptr};
    QSpinBox* _spinCellReoptMaxSteps{nullptr};
    QSpinBox* _spinCellReoptMaxPoints{nullptr};
    QDoubleSpinBox* _spinCellReoptMinSpacing{nullptr};
    QDoubleSpinBox* _spinCellReoptPerimeterOffset{nullptr};
    QComboBox* _comboCellReoptCollection{nullptr};
    QPushButton* _btnCellReoptRun{nullptr};

    bool _cellReoptMode{false};
    int _cellReoptMaxSteps{500};
    int _cellReoptMaxPoints{50};
    float _cellReoptMinSpacing{5.0f};
    float _cellReoptPerimeterOffset{0.0f};

    bool _restoringSettings{false};
    const QString _settingsGroup;
};
