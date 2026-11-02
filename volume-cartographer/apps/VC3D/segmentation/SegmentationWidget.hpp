#pragma once

#include <QColor>
#include <QVector>
#include <QWidget>

#include <optional>
#include <utility>
#include <vector>

#include "SegmentationCommon.hpp"
#include "SegmentationPushPullConfig.hpp"

#include <nlohmann/json.hpp>

#include "growth/SegmentationGrowth.hpp"
#include "tools/ManualAddTool.hpp"

class SegmentationHeaderRow;
class SegmentationEditingPanel;
class SegmentationGrowthPanel;
class SegmentationCorrectionsPanel;
class SegmentationCustomParamsPanel;
class SegmentationNeuralTracerPanel;
class SegmentationLasagnaPanel;
class SegmentationManualAddPanel;

class SegmentationWidget : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationWidget(QWidget* parent = nullptr);

    [[nodiscard]] bool isEditingEnabled() const { return _editingEnabled; }
    [[nodiscard]] bool drawMaskEnabled() const { return _drawMaskEnabled; }
    [[nodiscard]] float dragRadius() const;
    [[nodiscard]] float dragSigma() const;
    [[nodiscard]] float pushPullRadius() const;
    [[nodiscard]] float pushPullSigma() const;
    [[nodiscard]] float pushPullStep() const;
    [[nodiscard]] AlphaPushPullConfig alphaPushPullConfig() const;
    [[nodiscard]] float editScale() const;
    [[nodiscard]] float smoothingStrength() const;
    [[nodiscard]] int smoothingIterations() const;
    [[nodiscard]] SegmentationGrowthMethod growthMethod() const;
    [[nodiscard]] SegmentationGrowthMethod lastNonManualGrowthMethod() const { return _lastNonManualGrowthMethod; }
    [[nodiscard]] int growthSteps() const;
    [[nodiscard]] QString customParamsText() const;
    [[nodiscard]] QString customParamsProfile() const;
    [[nodiscard]] bool customParamsValid() const;
    [[nodiscard]] QString customParamsError() const;
    [[nodiscard]] nlohmann::json customParamsJson() const;
    [[nodiscard]] bool showHoverMarker() const;
    [[nodiscard]] bool growthKeybindsEnabled() const;
    [[nodiscard]] int growthScale() const;
    [[nodiscard]] ManualAddTool::Config manualAddConfig() const;
    // Neural tracer getters — delegated to panel
    [[nodiscard]] bool neuralTracerEnabled() const;
    [[nodiscard]] QString neuralCheckpointPath() const;
    [[nodiscard]] QString neuralPythonPath() const;
    [[nodiscard]] QString volumeZarrPath() const;
    [[nodiscard]] int neuralVolumeScale() const;
    [[nodiscard]] int neuralBatchSize() const;
    [[nodiscard]] NeuralTracerModelType neuralModelType() const;
    [[nodiscard]] TtaMode ttaMode() const;
    [[nodiscard]] QString ttaMergeMethod() const;
    [[nodiscard]] double ttaOutlierDropThresh() const;
    [[nodiscard]] QString copyCheckpointPath() const;

    void setPendingChanges(bool pending);
    void setEditingEnabled(bool enabled);
    void setDrawMaskEnabled(bool enabled);
    void setDragRadius(float value);
    void setDragSigma(float value);
    void setPushPullRadius(float value);
    void setPushPullSigma(float value);
    void setPushPullStep(float value);
    void setAlphaPushPullConfig(const AlphaPushPullConfig& config);
    void setEditScale(float value);
    void setSmoothingStrength(float value);
    void setSmoothingIterations(int value);
    void setGrowthMethod(SegmentationGrowthMethod method);
    void setGrowthInProgress(bool running);
    void setShowHoverMarker(bool enabled);
    void setManualAddActive(bool active);
    ManualAddTool::LinePreviewMode cycleManualAddLinePreviewMode();
    // Route direct changes through the panel so manualAddConfigChanged reaches
    // the module just as it does for a combo-box edit.
    ManualAddTool::LinePreviewMode setManualAddLinePreviewMode(ManualAddTool::LinePreviewMode mode);
    ManualAddTool::InterpolationMode setManualAddInterpolationMode(ManualAddTool::InterpolationMode mode);

    void setNormalGridAvailable(bool available);
    void setNormalGridPathHint(const QString& hint);
    void setNormalGridPath(const QString& path);


    void setVolumePackagePath(const QString& path);
    void setAvailableVolumes(const QVector<QPair<QString, QString>>& volumes,
                             const QString& activeId);
    void setActiveVolume(const QString& volumeId);

    void setCorrectionsEnabled(bool enabled);
    void setCorrectionCollections(const QVector<QPair<uint64_t, QString>>& collections,
                                   std::optional<uint64_t> activeId);
    void setGrowthSteps(int steps, bool persist = true);
    [[nodiscard]] std::optional<std::pair<int, int>> correctionsZRange() const;

    [[nodiscard]] std::vector<SegmentationGrowthDirection> allowedGrowthDirections() const;

    // "Draw mask" brush radius in voxels.
    [[nodiscard]] float maskBrushRadius() const { return _maskBrushRadius; }

    // Neural tracer setters
    void setNeuralTracerEnabled(bool enabled);
    void setNeuralCheckpointPath(const QString& path);
    void setNeuralPythonPath(const QString& path);
    void setNeuralVolumeScale(int scale);
    void setNeuralBatchSize(int size);
    void setNeuralModelType(NeuralTracerModelType type);
    void setTtaMode(TtaMode mode);
    void setTtaMergeMethod(const QString& method);
    void setTtaOutlierDropThresh(double threshold);
    void setCopyCheckpointPath(const QString& path);

    /**
     * Set the volume zarr path for neural tracing.
     * This is typically set automatically when the volume changes.
     */
    void setVolumeZarrPath(const QString& path);

    /** Returns the lasagna panel widget (for hosting in a separate dock). */
    [[nodiscard]] SegmentationLasagnaPanel* lasagnaPanel() const { return _lasagnaPanel; }

    // Lasagna getters — delegated to panel
    [[nodiscard]] QString lasagnaDataInputPath() const;
    [[nodiscard]] QString lasagnaConfigText() const;
    [[nodiscard]] int lasagnaMode() const;
    [[nodiscard]] int newModelWidth() const;
    [[nodiscard]] int newModelHeight() const;
    [[nodiscard]] int newModelWindings() const;
    [[nodiscard]] QString seedPointText() const;
    [[nodiscard]] QString newModelOutputName() const;
    [[nodiscard]] double offsetValue() const;

    // Lasagna setters
    void setLasagnaDataInputPath(const QString& path);
    void setSeedFromFocus(int x, int y, int z);

    void setAnnotateChecked(bool checked);

signals:
    void annotateToggled(bool enabled);
    void editingModeChanged(bool enabled);
    void drawMaskChanged(bool enabled);
    void dragRadiusChanged(float value);
    void dragSigmaChanged(float value);
    void pushPullRadiusChanged(float value);
    void pushPullSigmaChanged(float value);
    void growthMethodChanged(SegmentationGrowthMethod method);
    void pushPullStepChanged(float value);
    void alphaPushPullConfigChanged();
    void editScaleChanged(float value);
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
    void correctionsCollectionSelectionCleared();
    void correctionsZRangeChanged(bool enabled, int zMin, int zMax);
    void hoverMarkerToggled(bool enabled);
    void maskBrushRadiusChanged(float radius);

    // Neural tracer signals
    void neuralTracerEnabledChanged(bool enabled);
    void neuralTracerStatusMessage(const QString& message);
    void copyWithNtRequested();

    // Lasagna signals
    void lasagnaOptimizeRequested();
    void lasagnaStopRequested();
    void lasagnaStatusMessage(const QString& message);
    void seedFromFocusRequested();

    void manualAddConfigChanged();
    void manualAddClearPendingRequested();
    void manualAddRecomputeRequested();
    void manualAddApplyExitRequested();
    void manualAddCancelRequested();

private:
    void buildUi();
    void syncUiState();
    void restoreSettings();
    void writeSetting(const QString& key, const QVariant& value);
    void updateEditingState(bool enabled, bool notifyListeners);
    void noteGrowthMethod(SegmentationGrowthMethod method);

    bool _editingEnabled{false};
    bool _drawMaskEnabled{false};
    float _maskBrushRadius{50.0f};
    bool _pending{false};
    bool _growthInProgress{false};
    bool _restoringSettings{false};

    SegmentationHeaderRow* _headerRow{nullptr};
    SegmentationGrowthPanel* _growthPanel{nullptr};
    SegmentationEditingPanel* _editingPanel{nullptr};
    SegmentationCorrectionsPanel* _correctionsPanel{nullptr};
    SegmentationCustomParamsPanel* _customParamsPanel{nullptr};
    SegmentationNeuralTracerPanel* _neuralTracerPanel{nullptr};
    SegmentationLasagnaPanel* _lasagnaPanel{nullptr};
    SegmentationManualAddPanel* _manualAddPanel{nullptr};
    bool _manualAddActive{false};
    SegmentationGrowthMethod _lastNonManualGrowthMethod{SegmentationGrowthMethod::Tracer};

};
