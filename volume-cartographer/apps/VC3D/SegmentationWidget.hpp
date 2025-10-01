#pragma once

#include <QVector>
#include <QWidget>

#include <optional>
#include <utility>
#include <vector>

#include "SegmentationGrowth.hpp"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QGroupBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QPushButton;
class QSpinBox;
class QToolButton;

class SegmentationWidget : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationWidget(QWidget* parent = nullptr);

    [[nodiscard]] bool isEditingEnabled() const { return _editingEnabled; }
    [[nodiscard]] float radius() const { return _radiusSteps; }
    [[nodiscard]] float sigma() const { return _sigmaSteps; }
    [[nodiscard]] SegmentationGrowthMethod growthMethod() const { return _growthMethod; }
    [[nodiscard]] int growthSteps() const { return _growthSteps; }

    void setPendingChanges(bool pending);
    void setEditingEnabled(bool enabled);
    void setRadius(float value);
    void setSigma(float value);
    void setGrowthMethod(SegmentationGrowthMethod method);
    void setGrowthInProgress(bool running);
    void setEraseBrushActive(bool active);

    void setNormalGridAvailable(bool available);
    void setNormalGridPathHint(const QString& hint);

    void setVolumePackagePath(const QString& path);
    void setAvailableVolumes(const QVector<QPair<QString, QString>>& volumes,
                             const QString& activeId);
    void setActiveVolume(const QString& volumeId);

    void setCorrectionsEnabled(bool enabled);
    void setCorrectionsAnnotateChecked(bool enabled);
    void setCorrectionCollections(const QVector<QPair<uint64_t, QString>>& collections,
                                   std::optional<uint64_t> activeId);
    [[nodiscard]] std::optional<std::pair<int, int>> correctionsZRange() const;

    [[nodiscard]] std::vector<SegmentationGrowthDirection> allowedGrowthDirections() const;
    [[nodiscard]] std::vector<SegmentationDirectionFieldConfig> directionFieldConfigs() const;

signals:
    void editingModeChanged(bool enabled);
    void radiusChanged(float value);
    void sigmaChanged(float value);
    void growthMethodChanged(SegmentationGrowthMethod method);
    void growSurfaceRequested(SegmentationGrowthMethod method,
                              SegmentationGrowthDirection direction,
                              int steps);
    void applyRequested();
    void resetRequested();
    void stopToolsRequested();
    void volumeSelectionChanged(const QString& volumeId);
    void correctionsCreateRequested();
    void correctionsCollectionSelected(uint64_t collectionId);
    void correctionsAnnotateToggled(bool enabled);
    void correctionsZRangeChanged(bool enabled, int zMin, int zMax);

private:
    void buildUi();
    void syncUiState();
    void restoreSettings();
    void writeSetting(const QString& key, const QVariant& value);
    void updateEditingState(bool enabled, bool notifyListeners);

    void refreshDirectionFieldList();
    void persistDirectionFields();
    SegmentationDirectionFieldConfig buildDirectionFieldDraft() const;
    void updateDirectionFieldFormFromSelection(int row);
    void applyDirectionFieldDraftToSelection(int row);
    void updateDirectionFieldListItem(int row);
    void setGrowthDirectionMask(int mask);
    void updateGrowthDirectionMaskFromUi(QCheckBox* changedCheckbox);
    void applyGrowthDirectionMaskToUi();
    void updateGrowthUiState();
    static int normalizeGrowthDirectionMask(int mask);

    bool _editingEnabled{false};
    bool _pending{false};
    bool _growthInProgress{false};
    bool _eraseBrushActive{false};
    float _radiusSteps{3.0f};
    float _sigmaSteps{1.5f};

    bool _normalGridAvailable{false};
    QString _normalGridHint;
    QString _normalGridDisplayPath;
    QString _volumePackagePath;
    QVector<QPair<QString, QString>> _volumeEntries;
    QString _activeVolumeId;

    SegmentationGrowthMethod _growthMethod{SegmentationGrowthMethod::Corrections};
    int _growthSteps{5};
    int _growthDirectionMask{0};

    QString _directionFieldPath;
    SegmentationDirectionFieldOrientation _directionFieldOrientation{SegmentationDirectionFieldOrientation::Normal};
    int _directionFieldScale{0};
    double _directionFieldWeight{1.0};
    std::vector<SegmentationDirectionFieldConfig> _directionFields;
    bool _updatingDirectionFieldForm{false};

    QCheckBox* _chkEditing{nullptr};
    QLabel* _lblStatus{nullptr};
    QGroupBox* _groupGrowth{nullptr};
    QSpinBox* _spinGrowthSteps{nullptr};
    QPushButton* _btnGrow{nullptr};
    QCheckBox* _chkGrowthDirUp{nullptr};
    QCheckBox* _chkGrowthDirDown{nullptr};
    QCheckBox* _chkGrowthDirLeft{nullptr};
    QCheckBox* _chkGrowthDirRight{nullptr};
    QComboBox* _comboVolumes{nullptr};
    QLabel* _lblNormalGrid{nullptr};

    QGroupBox* _groupDirectionField{nullptr};
    QLineEdit* _directionFieldPathEdit{nullptr};
    QToolButton* _directionFieldBrowseButton{nullptr};
    QComboBox* _comboDirectionFieldOrientation{nullptr};
    QComboBox* _comboDirectionFieldScale{nullptr};
    QDoubleSpinBox* _spinDirectionFieldWeight{nullptr};
    QPushButton* _directionFieldAddButton{nullptr};
    QPushButton* _directionFieldRemoveButton{nullptr};
    QListWidget* _directionFieldList{nullptr};

    QGroupBox* _groupCorrections{nullptr};
    QComboBox* _comboCorrections{nullptr};
    QPushButton* _btnCorrectionsNew{nullptr};
    QCheckBox* _chkCorrectionsAnnotate{nullptr};
    QCheckBox* _chkCorrectionsUseZRange{nullptr};
    QSpinBox* _spinCorrectionsZMin{nullptr};
    QSpinBox* _spinCorrectionsZMax{nullptr};

    QDoubleSpinBox* _spinRadius{nullptr};
    QDoubleSpinBox* _spinSigma{nullptr};
    QPushButton* _btnApply{nullptr};
    QPushButton* _btnReset{nullptr};
    QPushButton* _btnStop{nullptr};
    QCheckBox* _chkEraseBrush{nullptr};

    bool _correctionsEnabled{false};
    bool _correctionsZRangeEnabled{false};
    int _correctionsZMin{0};
    int _correctionsZMax{0};
    bool _correctionsAnnotateChecked{false};
};
