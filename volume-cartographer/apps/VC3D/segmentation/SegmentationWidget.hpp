#pragma once

#include <QVector>
#include <QWidget>

#include <optional>
#include <utility>
#include <vector>

#include "SegmentationPushPullConfig.hpp"

#include <nlohmann/json_fwd.hpp>

#include "SegmentationGrowth.hpp"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QGroupBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QPlainTextEdit;
class QPushButton;
class QSpinBox;
class QToolButton;
class CollapsibleSettingsGroup;

class SegmentationWidget : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationWidget(QWidget* parent = nullptr);

    [[nodiscard]] bool isEditingEnabled() const { return _editingEnabled; }
    [[nodiscard]] float dragRadius() const { return _dragRadiusSteps; }
    [[nodiscard]] float dragSigma() const { return _dragSigmaSteps; }
    [[nodiscard]] float lineRadius() const { return _lineRadiusSteps; }
    [[nodiscard]] float lineSigma() const { return _lineSigmaSteps; }
    [[nodiscard]] float pushPullRadius() const { return _pushPullRadiusSteps; }
    [[nodiscard]] float pushPullSigma() const { return _pushPullSigmaSteps; }
    [[nodiscard]] float pushPullStep() const { return _pushPullStep; }
    [[nodiscard]] AlphaPushPullConfig alphaPushPullConfig() const;
    [[nodiscard]] float smoothingStrength() const { return _smoothStrength; }
    [[nodiscard]] int smoothingIterations() const { return _smoothIterations; }
    [[nodiscard]] SegmentationGrowthMethod growthMethod() const { return _growthMethod; }
    [[nodiscard]] int growthSteps() const { return _growthSteps; }
    [[nodiscard]] QString customParamsText() const { return _customParamsText; }
    [[nodiscard]] bool customParamsValid() const { return _customParamsError.isEmpty(); }
    [[nodiscard]] QString customParamsError() const { return _customParamsError; }
    [[nodiscard]] std::optional<nlohmann::json> customParamsJson() const;

    void setPendingChanges(bool pending);
    void setEditingEnabled(bool enabled);
    void setDragRadius(float value);
    void setDragSigma(float value);
    void setLineRadius(float value);
    void setLineSigma(float value);
    void setPushPullRadius(float value);
    void setPushPullSigma(float value);
    void setPushPullStep(float value);
    void setAlphaPushPullConfig(const AlphaPushPullConfig& config);
    void setSmoothingStrength(float value);
    void setSmoothingIterations(int value);
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
    void setGrowthSteps(int steps, bool persist = true);
    [[nodiscard]] std::optional<std::pair<int, int>> correctionsZRange() const;

    [[nodiscard]] std::vector<SegmentationGrowthDirection> allowedGrowthDirections() const;
    [[nodiscard]] std::vector<SegmentationDirectionFieldConfig> directionFieldConfigs() const;

signals:
    void editingModeChanged(bool enabled);
    void dragRadiusChanged(float value);
    void dragSigmaChanged(float value);
    void lineRadiusChanged(float value);
    void lineSigmaChanged(float value);
    void pushPullRadiusChanged(float value);
    void pushPullSigmaChanged(float value);
    void growthMethodChanged(SegmentationGrowthMethod method);
    void pushPullStepChanged(float value);
    void alphaPushPullConfigChanged();
    void smoothingStrengthChanged(float value);
    void smoothingIterationsChanged(int value);
    void growSurfaceRequested(SegmentationGrowthMethod method,
                              SegmentationGrowthDirection direction,
                              int steps,
                              bool inpaintOnly);
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
    void updateDirectionFieldListGeometry();
    void clearDirectionFieldForm();
    [[nodiscard]] QString determineDefaultVolumeId(const QVector<QPair<QString, QString>>& volumes,
                                                   const QString& requestedId) const;
    void applyGrowthSteps(int steps, bool persist, bool fromUi);
    void setGrowthDirectionMask(int mask);
    void updateGrowthDirectionMaskFromUi(QCheckBox* changedCheckbox);
    void applyGrowthDirectionMaskToUi();
    void updateGrowthUiState();
    static int normalizeGrowthDirectionMask(int mask);
    void handleCustomParamsEdited();
    void validateCustomParamsText();
    void updateCustomParamsStatus();
    std::optional<nlohmann::json> parseCustomParams(QString* error) const;
    void triggerGrowthRequest(SegmentationGrowthDirection direction, int steps, bool inpaintOnly);
    void applyAlphaPushPullConfig(const AlphaPushPullConfig& config, bool emitSignal, bool persist = true);

    bool _editingEnabled{false};
    bool _pending{false};
    bool _growthInProgress{false};
    bool _eraseBrushActive{false};
    float _dragRadiusSteps{5.75f};
    float _dragSigmaSteps{2.0f};
    float _lineRadiusSteps{5.75f};
    float _lineSigmaSteps{2.0f};
    float _pushPullRadiusSteps{5.75f};
    float _pushPullSigmaSteps{2.0f};
    float _pushPullStep{4.0f};
    AlphaPushPullConfig _alphaPushPullConfig{};
    float _smoothStrength{0.4f};
    int _smoothIterations{2};

    bool _normalGridAvailable{false};
    QString _normalGridHint;
    QString _normalGridDisplayPath;
    QString _volumePackagePath;
    QVector<QPair<QString, QString>> _volumeEntries;
    QString _activeVolumeId;

    SegmentationGrowthMethod _growthMethod{SegmentationGrowthMethod::Corrections};
    int _growthSteps{5};
    int _tracerGrowthSteps{5};
    int _growthDirectionMask{0};

    QString _directionFieldPath;
    SegmentationDirectionFieldOrientation _directionFieldOrientation{SegmentationDirectionFieldOrientation::Normal};
    int _directionFieldScale{0};
    double _directionFieldWeight{1.0};
    std::vector<SegmentationDirectionFieldConfig> _directionFields;
    bool _updatingDirectionFieldForm{false};
    bool _restoringSettings{false};

    QCheckBox* _chkEditing{nullptr};
    QLabel* _lblStatus{nullptr};
    QGroupBox* _groupGrowth{nullptr};
    QSpinBox* _spinGrowthSteps{nullptr};
    QPushButton* _btnGrow{nullptr};
    QPushButton* _btnInpaint{nullptr};
    QCheckBox* _chkGrowthDirUp{nullptr};
    QCheckBox* _chkGrowthDirDown{nullptr};
    QCheckBox* _chkGrowthDirLeft{nullptr};
    QCheckBox* _chkGrowthDirRight{nullptr};
    QComboBox* _comboVolumes{nullptr};
    QLabel* _lblNormalGrid{nullptr};
    QLabel* _lblAlphaInfo{nullptr};

    CollapsibleSettingsGroup* _groupEditing{nullptr};
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

    QGroupBox* _groupCorrections{nullptr};
    QComboBox* _comboCorrections{nullptr};
    QPushButton* _btnCorrectionsNew{nullptr};
    QCheckBox* _chkCorrectionsAnnotate{nullptr};
    QCheckBox* _chkCorrectionsUseZRange{nullptr};
    QSpinBox* _spinCorrectionsZMin{nullptr};
    QSpinBox* _spinCorrectionsZMax{nullptr};

    CollapsibleSettingsGroup* _groupDrag{nullptr};
    CollapsibleSettingsGroup* _groupLine{nullptr};
    CollapsibleSettingsGroup* _groupPushPull{nullptr};

    QDoubleSpinBox* _spinDragRadius{nullptr};
    QDoubleSpinBox* _spinDragSigma{nullptr};
    QDoubleSpinBox* _spinLineRadius{nullptr};
    QDoubleSpinBox* _spinLineSigma{nullptr};
    QDoubleSpinBox* _spinPushPullRadius{nullptr};
    QDoubleSpinBox* _spinPushPullSigma{nullptr};
    QDoubleSpinBox* _spinPushPullStep{nullptr};
    QWidget* _alphaPushPullPanel{nullptr};
    QCheckBox* _chkAlphaPerVertex{nullptr};
    QDoubleSpinBox* _spinAlphaStart{nullptr};
    QDoubleSpinBox* _spinAlphaStop{nullptr};
    QDoubleSpinBox* _spinAlphaStep{nullptr};
    QDoubleSpinBox* _spinAlphaLow{nullptr};
    QDoubleSpinBox* _spinAlphaHigh{nullptr};
    QDoubleSpinBox* _spinAlphaBorder{nullptr};
    QSpinBox* _spinAlphaBlurRadius{nullptr};
    QDoubleSpinBox* _spinAlphaPerVertexLimit{nullptr};
    QDoubleSpinBox* _spinSmoothStrength{nullptr};
    QSpinBox* _spinSmoothIterations{nullptr};
    QPushButton* _btnApply{nullptr};
    QPushButton* _btnReset{nullptr};
    QPushButton* _btnStop{nullptr};
    QCheckBox* _chkEraseBrush{nullptr};

    QGroupBox* _groupCustomParams{nullptr};
    QPlainTextEdit* _editCustomParams{nullptr};
    QLabel* _lblCustomParamsStatus{nullptr};
    QString _customParamsText;
    QString _customParamsError;

    bool _correctionsEnabled{false};
    bool _correctionsZRangeEnabled{false};
    int _correctionsZMin{0};
    int _correctionsZMax{0};
    bool _correctionsAnnotateChecked{false};
};
