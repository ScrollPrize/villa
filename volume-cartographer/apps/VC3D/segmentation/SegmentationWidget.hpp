#pragma once

#include <QColor>
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
class QSlider;
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
    [[nodiscard]] int extrapolationPointCount() const { return _extrapolationPointCount; }
    [[nodiscard]] ExtrapolationType extrapolationType() const { return _extrapolationType; }
    [[nodiscard]] int sdtMaxSteps() const { return _sdtMaxSteps; }
    [[nodiscard]] float sdtStepSize() const { return _sdtStepSize; }
    [[nodiscard]] float sdtConvergence() const { return _sdtConvergence; }
    [[nodiscard]] int sdtChunkSize() const { return _sdtChunkSize; }
    [[nodiscard]] int skeletonConnectivity() const { return _skeletonConnectivity; }
    [[nodiscard]] int skeletonSliceOrientation() const { return _skeletonSliceOrientation; }
    [[nodiscard]] int skeletonChunkSize() const { return _skeletonChunkSize; }
    [[nodiscard]] int skeletonSearchRadius() const { return _skeletonSearchRadius; }
    [[nodiscard]] QString customParamsText() const { return _customParamsText; }
    [[nodiscard]] QString customParamsProfile() const { return _customParamsProfile; }
    [[nodiscard]] bool customParamsValid() const { return _customParamsError.isEmpty(); }
    [[nodiscard]] QString customParamsError() const { return _customParamsError; }
    [[nodiscard]] std::optional<nlohmann::json> customParamsJson() const;
    [[nodiscard]] bool showHoverMarker() const { return _showHoverMarker; }
    [[nodiscard]] bool growthKeybindsEnabled() const { return _growthKeybindsEnabled; }

    [[nodiscard]] QString normal3dZarrPath() const { return _normal3dSelectedPath; }
    // Neural tracer getters
    [[nodiscard]] bool neuralTracerEnabled() const { return _neuralTracerEnabled; }
    [[nodiscard]] QString neuralCheckpointPath() const { return _neuralCheckpointPath; }
    [[nodiscard]] QString neuralPythonPath() const { return _neuralPythonPath; }
    [[nodiscard]] QString volumeZarrPath() const { return _volumeZarrPath; }
    [[nodiscard]] int neuralVolumeScale() const { return _neuralVolumeScale; }
    [[nodiscard]] int neuralBatchSize() const { return _neuralBatchSize; }

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
    void setShowHoverMarker(bool enabled);

    void setNormalGridAvailable(bool available);
    void setNormalGridPathHint(const QString& hint);
    void setNormalGridPath(const QString& path);

    void setNormal3dZarrCandidates(const QStringList& candidates, const QString& hint);

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

    // Approval mask getters
    [[nodiscard]] bool showApprovalMask() const { return _showApprovalMask; }
    [[nodiscard]] bool editApprovedMask() const { return _editApprovedMask; }
    [[nodiscard]] bool editUnapprovedMask() const { return _editUnapprovedMask; }
    [[nodiscard]] float approvalBrushRadius() const { return _approvalBrushRadius; }
    [[nodiscard]] float approvalBrushDepth() const { return _approvalBrushDepth; }
    [[nodiscard]] int approvalMaskOpacity() const { return _approvalMaskOpacity; }
    [[nodiscard]] QColor approvalBrushColor() const { return _approvalBrushColor; }

    // Approval mask setters
    void setShowApprovalMask(bool enabled);
    void setEditApprovedMask(bool enabled);
    void setEditUnapprovedMask(bool enabled);
    void setApprovalBrushRadius(float radius);
    void setApprovalBrushDepth(float depth);
    void setApprovalMaskOpacity(int opacity);
    void setApprovalBrushColor(const QColor& color);

    // Neural tracer setters
    void setNeuralTracerEnabled(bool enabled);
    void setNeuralCheckpointPath(const QString& path);
    void setNeuralPythonPath(const QString& path);
    void setNeuralVolumeScale(int scale);
    void setNeuralBatchSize(int size);

    /**
     * Set the volume zarr path for neural tracing.
     * This is typically set automatically when the volume changes.
     */
    void setVolumeZarrPath(const QString& path);

    // Cell reoptimization getters
    [[nodiscard]] bool cellReoptMode() const { return _cellReoptMode; }
    [[nodiscard]] int cellReoptMaxSteps() const { return _cellReoptMaxSteps; }
    [[nodiscard]] int cellReoptMaxPoints() const { return _cellReoptMaxPoints; }
    [[nodiscard]] float cellReoptMinSpacing() const { return _cellReoptMinSpacing; }
    [[nodiscard]] float cellReoptPerimeterOffset() const { return _cellReoptPerimeterOffset; }

    // Cell reoptimization setters
    void setCellReoptMode(bool enabled);
    void setCellReoptCollections(const QVector<QPair<uint64_t, QString>>& collections);

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
    void hoverMarkerToggled(bool enabled);
    void showApprovalMaskChanged(bool enabled);
    void editApprovedMaskChanged(bool enabled);
    void editUnapprovedMaskChanged(bool enabled);
    void approvalBrushRadiusChanged(float radius);
    void approvalBrushDepthChanged(float depth);
    void approvalMaskOpacityChanged(int opacity);
    void approvalBrushColorChanged(QColor color);
    void approvalStrokesUndoRequested();

    // Neural tracer signals
    void neuralTracerEnabledChanged(bool enabled);
    void neuralTracerServiceRequested(const QString& checkpointPath,
                                      const QString& volumeZarr,
                                      int volumeScale);
    void neuralTracerStatusMessage(const QString& message);

    // Cell reoptimization signals
    void cellReoptModeChanged(bool enabled);
    void cellReoptMaxStepsChanged(int steps);
    void cellReoptMaxPointsChanged(int points);
    void cellReoptMinSpacingChanged(float spacing);
    void cellReoptPerimeterOffsetChanged(float offset);
    void cellReoptGrowthRequested(uint64_t collectionId);

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
    void applyCustomParamsProfile(const QString& profile, bool persist, bool fromUi);
    [[nodiscard]] QString paramsTextForProfile(const QString& profile) const;
    void triggerGrowthRequest(SegmentationGrowthDirection direction, int steps, bool inpaintOnly);
    void applyAlphaPushPullConfig(const AlphaPushPullConfig& config, bool emitSignal, bool persist = true);

    void updateNormal3dUi();

    bool _editingEnabled{false};
    bool _pending{false};
    bool _growthInProgress{false};
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
    bool _showHoverMarker{true};

    bool _normalGridAvailable{false};
    QString _normalGridHint;
    QString _normalGridDisplayPath;
    QString _normalGridPath;

    QStringList _normal3dCandidates;
    QString _normal3dHint;
    QString _normal3dSelectedPath;
    QString _volumePackagePath;
    QVector<QPair<QString, QString>> _volumeEntries;
    QString _activeVolumeId;

    SegmentationGrowthMethod _growthMethod{SegmentationGrowthMethod::Corrections};
    int _growthSteps{5};
    int _tracerGrowthSteps{5};
    int _growthDirectionMask{0};
    bool _growthKeybindsEnabled{true};
    int _extrapolationPointCount{7};
    ExtrapolationType _extrapolationType{ExtrapolationType::Linear};

    // SDT/Newton refinement parameters for Linear+Fit
    int _sdtMaxSteps{5};
    float _sdtStepSize{0.8f};
    float _sdtConvergence{0.5f};
    int _sdtChunkSize{128};

    // Skeleton path parameters
    int _skeletonConnectivity{26};  // 6, 18, or 26
    int _skeletonSliceOrientation{0};  // 0=X, 1=Y for up/down growth
    int _skeletonChunkSize{128};
    int _skeletonSearchRadius{5};  // 1-100 pixels

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
    QComboBox* _comboGrowthMethod{nullptr};
    QWidget* _extrapolationOptionsPanel{nullptr};
    QLabel* _lblExtrapolationPoints{nullptr};
    QSpinBox* _spinExtrapolationPoints{nullptr};
    QComboBox* _comboExtrapolationType{nullptr};
    QWidget* _sdtParamsContainer{nullptr};
    QSpinBox* _spinSDTMaxSteps{nullptr};
    QDoubleSpinBox* _spinSDTStepSize{nullptr};
    QDoubleSpinBox* _spinSDTConvergence{nullptr};
    QSpinBox* _spinSDTChunkSize{nullptr};
    QWidget* _skeletonParamsContainer{nullptr};
    QComboBox* _comboSkeletonConnectivity{nullptr};
    QComboBox* _comboSkeletonSliceOrientation{nullptr};
    QSpinBox* _spinSkeletonChunkSize{nullptr};
    QSpinBox* _spinSkeletonSearchRadius{nullptr};
    QPushButton* _btnGrow{nullptr};
    QPushButton* _btnInpaint{nullptr};
    QCheckBox* _chkGrowthDirUp{nullptr};
    QCheckBox* _chkGrowthDirDown{nullptr};
    QCheckBox* _chkGrowthDirLeft{nullptr};
    QCheckBox* _chkGrowthDirRight{nullptr};
    QCheckBox* _chkGrowthKeybindsEnabled{nullptr};
    QComboBox* _comboVolumes{nullptr};
    QLabel* _lblNormalGrid{nullptr};
    QLineEdit* _editNormalGridPath{nullptr};
    QLabel* _lblNormal3d{nullptr};
    QComboBox* _comboNormal3d{nullptr};
    QLineEdit* _editNormal3dPath{nullptr};
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
    QCheckBox* _chkShowHoverMarker{nullptr};

    QGroupBox* _groupCustomParams{nullptr};
    QComboBox* _comboCustomParamsProfile{nullptr};
    QPlainTextEdit* _editCustomParams{nullptr};
    QLabel* _lblCustomParamsStatus{nullptr};
    QString _customParamsText;
    QString _customParamsError;
    QString _customParamsProfile{QStringLiteral("custom")};
    bool _updatingCustomParamsProgrammatically{false};

    bool _correctionsEnabled{false};
    bool _correctionsZRangeEnabled{false};
    int _correctionsZMin{0};
    int _correctionsZMax{0};
    bool _correctionsAnnotateChecked{false};

    // Approval mask state and UI
    // Cylinder brush model: radius defines circle in plane views, depth defines cylinder height
    bool _showApprovalMask{false};
    bool _editApprovedMask{false};    // Editing in approve mode (mutually exclusive with unapprove)
    bool _editUnapprovedMask{false};  // Editing in unapprove mode (mutually exclusive with approve)
    float _approvalBrushRadius{50.0f};     // Cylinder radius (circle in plane views, rect width in flattened)
    float _approvalBrushDepth{15.0f};      // Cylinder depth (rect height in flattened view)
    int _approvalMaskOpacity{50};          // Mask overlay opacity (0-100, default 50%)
    QColor _approvalBrushColor{0, 255, 0}; // RGB color for approval painting (default pure green)
    CollapsibleSettingsGroup* _groupApprovalMask{nullptr};
    QCheckBox* _chkShowApprovalMask{nullptr};
    QCheckBox* _chkEditApprovedMask{nullptr};
    QCheckBox* _chkEditUnapprovedMask{nullptr};
    QDoubleSpinBox* _spinApprovalBrushRadius{nullptr};
    QDoubleSpinBox* _spinApprovalBrushDepth{nullptr};
    QSlider* _sliderApprovalMaskOpacity{nullptr};
    QLabel* _lblApprovalMaskOpacity{nullptr};
    QPushButton* _btnApprovalColor{nullptr};
    QPushButton* _btnUndoApprovalStroke{nullptr};

    // Neural tracer state
    bool _neuralTracerEnabled{false};
    QString _neuralCheckpointPath;
    QString _neuralPythonPath;
    QString _volumeZarrPath;
    int _neuralVolumeScale{0};
    int _neuralBatchSize{4};

    // Neural tracer UI
    CollapsibleSettingsGroup* _groupNeuralTracer{nullptr};
    QCheckBox* _chkNeuralTracerEnabled{nullptr};
    QLineEdit* _neuralCheckpointEdit{nullptr};
    QToolButton* _neuralCheckpointBrowse{nullptr};
    QLineEdit* _neuralPythonEdit{nullptr};
    QToolButton* _neuralPythonBrowse{nullptr};
    QComboBox* _comboNeuralVolumeScale{nullptr};
    QSpinBox* _spinNeuralBatchSize{nullptr};
    QLabel* _lblNeuralTracerStatus{nullptr};

    // Cell reoptimization state and UI
    bool _cellReoptMode{false};
    int _cellReoptMaxSteps{500};
    int _cellReoptMaxPoints{50};
    float _cellReoptMinSpacing{5.0f};
    float _cellReoptPerimeterOffset{0.0f};
    CollapsibleSettingsGroup* _groupCellReopt{nullptr};
    QCheckBox* _chkCellReoptMode{nullptr};
    QSpinBox* _spinCellReoptMaxSteps{nullptr};
    QSpinBox* _spinCellReoptMaxPoints{nullptr};
    QDoubleSpinBox* _spinCellReoptMinSpacing{nullptr};
    QDoubleSpinBox* _spinCellReoptPerimeterOffset{nullptr};
    QComboBox* _comboCellReoptCollection{nullptr};
    QPushButton* _btnCellReoptRun{nullptr};
};
