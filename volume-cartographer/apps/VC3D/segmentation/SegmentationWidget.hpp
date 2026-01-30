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
class JsonProfileEditor;
class SegmentationHeaderRow;
class SegmentationEditingPanel;
class SegmentationGrowthPanel;
class SegmentationCorrectionsPanel;
class SegmentationCustomParamsPanel;
class SegmentationApprovalMaskPanel;
class SegmentationCellReoptPanel;
class SegmentationNeuralTracerPanel;
class SegmentationDirectionFieldPanel;

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
    [[nodiscard]] QString customParamsText() const { return paramsTextForProfile(_customParamsProfile); }
    [[nodiscard]] QString customParamsProfile() const { return _customParamsProfile; }
    [[nodiscard]] bool customParamsValid() const { return _customParamsError.isEmpty(); }
    [[nodiscard]] QString customParamsError() const { return _customParamsError; }
    [[nodiscard]] std::optional<nlohmann::json> customParamsJson() const;
    [[nodiscard]] bool showHoverMarker() const { return _showHoverMarker; }
    [[nodiscard]] bool growthKeybindsEnabled() const { return _growthKeybindsEnabled; }

    [[nodiscard]] QString normal3dZarrPath() const { return _normal3dSelectedPath; }
    // Neural tracer getters — delegated to panel
    [[nodiscard]] bool neuralTracerEnabled() const;
    [[nodiscard]] QString neuralCheckpointPath() const;
    [[nodiscard]] QString neuralPythonPath() const;
    [[nodiscard]] QString volumeZarrPath() const;
    [[nodiscard]] int neuralVolumeScale() const;
    [[nodiscard]] int neuralBatchSize() const;

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

    // Approval mask getters — delegated to panel
    [[nodiscard]] bool showApprovalMask() const;
    [[nodiscard]] bool editApprovedMask() const;
    [[nodiscard]] bool editUnapprovedMask() const;
    [[nodiscard]] bool autoApproveEdits() const;
    [[nodiscard]] float approvalBrushRadius() const;
    [[nodiscard]] float approvalBrushDepth() const;
    [[nodiscard]] int approvalMaskOpacity() const;
    [[nodiscard]] QColor approvalBrushColor() const;

    // Approval mask setters
    void setShowApprovalMask(bool enabled);
    void setEditApprovedMask(bool enabled);
    void setEditUnapprovedMask(bool enabled);
    void setAutoApproveEdits(bool enabled);
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

    // Cell reoptimization getters — delegated to panel
    [[nodiscard]] bool cellReoptMode() const;
    [[nodiscard]] int cellReoptMaxSteps() const;
    [[nodiscard]] int cellReoptMaxPoints() const;
    [[nodiscard]] float cellReoptMinSpacing() const;
    [[nodiscard]] float cellReoptPerimeterOffset() const;

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
    void autoApproveEditsChanged(bool enabled);
    void approvalBrushRadiusChanged(float radius);
    void approvalBrushDepthChanged(float depth);
    void approvalMaskOpacityChanged(int opacity);
    void approvalBrushColorChanged(QColor color);
    void approvalStrokesUndoRequested();

    // Neural tracer signals
    void neuralTracerEnabledChanged(bool enabled);
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

    SegmentationHeaderRow* _headerRow{nullptr};
    SegmentationGrowthPanel* _growthPanel{nullptr};
    SegmentationEditingPanel* _editingPanel{nullptr};
    SegmentationCorrectionsPanel* _correctionsPanel{nullptr};
    SegmentationCustomParamsPanel* _customParamsPanel{nullptr};
    SegmentationApprovalMaskPanel* _approvalMaskPanel{nullptr};
    SegmentationCellReoptPanel* _cellReoptPanel{nullptr};
    SegmentationNeuralTracerPanel* _neuralTracerPanel{nullptr};
    SegmentationDirectionFieldPanel* _directionFieldPanel{nullptr};

    JsonProfileEditor* _customParamsEditor{nullptr};
    QString _customParamsText;
    QString _customParamsError;
    QString _customParamsProfile{QStringLiteral("custom")};

    bool _correctionsEnabled{false};
    bool _correctionsZRangeEnabled{false};
    int _correctionsZMin{0};
    int _correctionsZMax{0};
    bool _correctionsAnnotateChecked{false};

};
