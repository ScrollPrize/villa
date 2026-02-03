#pragma once

#include <QColor>
#include <QVector>
#include <QWidget>

#include <optional>
#include <utility>
#include <vector>

#include "SegmentationPushPullConfig.hpp"

#include <nlohmann/json_fwd.hpp>

#include "growth/SegmentationGrowth.hpp"

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
    [[nodiscard]] float dragRadius() const;
    [[nodiscard]] float dragSigma() const;
    [[nodiscard]] float lineRadius() const;
    [[nodiscard]] float lineSigma() const;
    [[nodiscard]] float pushPullRadius() const;
    [[nodiscard]] float pushPullSigma() const;
    [[nodiscard]] float pushPullStep() const;
    [[nodiscard]] AlphaPushPullConfig alphaPushPullConfig() const;
    [[nodiscard]] float smoothingStrength() const;
    [[nodiscard]] int smoothingIterations() const;
    [[nodiscard]] SegmentationGrowthMethod growthMethod() const;
    [[nodiscard]] int growthSteps() const;
    [[nodiscard]] int extrapolationPointCount() const;
    [[nodiscard]] ExtrapolationType extrapolationType() const;
    [[nodiscard]] int sdtMaxSteps() const;
    [[nodiscard]] float sdtStepSize() const;
    [[nodiscard]] float sdtConvergence() const;
    [[nodiscard]] int sdtChunkSize() const;
    [[nodiscard]] int skeletonConnectivity() const;
    [[nodiscard]] int skeletonSliceOrientation() const;
    [[nodiscard]] int skeletonChunkSize() const;
    [[nodiscard]] int skeletonSearchRadius() const;
    [[nodiscard]] QString customParamsText() const;
    [[nodiscard]] QString customParamsProfile() const;
    [[nodiscard]] bool customParamsValid() const;
    [[nodiscard]] QString customParamsError() const;
    [[nodiscard]] std::optional<nlohmann::json> customParamsJson() const;
    [[nodiscard]] bool showHoverMarker() const;
    [[nodiscard]] bool growthKeybindsEnabled() const;
    [[nodiscard]] QString normal3dZarrPath() const;
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
    void setEraseBrushActive(bool active);

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

    bool _editingEnabled{false};
    bool _pending{false};
    bool _growthInProgress{false};
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

};
