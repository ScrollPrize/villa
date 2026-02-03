#include "SegmentationWidget.hpp"
#include "SegmentationCommon.hpp"

#include "panels/SegmentationEditingPanel.hpp"
#include "panels/SegmentationGrowthPanel.hpp"
#include "panels/SegmentationHeaderRow.hpp"
#include "panels/SegmentationCorrectionsPanel.hpp"
#include "panels/SegmentationCustomParamsPanel.hpp"
#include "panels/SegmentationApprovalMaskPanel.hpp"
#include "panels/SegmentationCellReoptPanel.hpp"
#include "panels/SegmentationNeuralTracerPanel.hpp"
#include "panels/SegmentationDirectionFieldPanel.hpp"
#include "VCSettings.hpp"

#include <QSettings>
#include <QVBoxLayout>
#include <QVariant>

#include <nlohmann/json.hpp>

Q_LOGGING_CATEGORY(lcSegWidget, "vc.segmentation.widget")

SegmentationWidget::SegmentationWidget(QWidget* parent)
    : QWidget(parent)
{
    buildUi();
    restoreSettings();
    syncUiState();
}

void SegmentationWidget::buildUi()
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(8, 8, 8, 8);
    layout->setSpacing(12);

    _headerRow = new SegmentationHeaderRow(this);
    layout->addWidget(_headerRow);

    _growthPanel = new SegmentationGrowthPanel(QStringLiteral("segmentation_edit"), this);
    layout->addWidget(_growthPanel);

    _editingPanel = new SegmentationEditingPanel(QStringLiteral("segmentation_edit"), this);
    layout->addWidget(_editingPanel);

    _approvalMaskPanel = new SegmentationApprovalMaskPanel(QStringLiteral("segmentation_edit"), this);
    layout->addWidget(_approvalMaskPanel);

    _cellReoptPanel = new SegmentationCellReoptPanel(QStringLiteral("segmentation_edit"), this);
    layout->addWidget(_cellReoptPanel);

    _directionFieldPanel = new SegmentationDirectionFieldPanel(QStringLiteral("segmentation_edit"), this);
    layout->addWidget(_directionFieldPanel);

    _neuralTracerPanel = new SegmentationNeuralTracerPanel(QStringLiteral("segmentation_edit"), this);
    layout->addWidget(_neuralTracerPanel);

    _correctionsPanel = new SegmentationCorrectionsPanel(QStringLiteral("segmentation_edit"), this);
    layout->addWidget(_correctionsPanel);

    _customParamsPanel = new SegmentationCustomParamsPanel(QStringLiteral("segmentation_edit"), this);
    layout->addWidget(_customParamsPanel);

    layout->addStretch(1);

    connect(_headerRow, &SegmentationHeaderRow::editingToggled, this, [this](bool enabled) {
        updateEditingState(enabled, true);
    });

    // Forward editing panel signals
    connect(_editingPanel, &SegmentationEditingPanel::dragRadiusChanged,
            this, &SegmentationWidget::dragRadiusChanged);
    connect(_editingPanel, &SegmentationEditingPanel::dragSigmaChanged,
            this, &SegmentationWidget::dragSigmaChanged);
    connect(_editingPanel, &SegmentationEditingPanel::lineRadiusChanged,
            this, &SegmentationWidget::lineRadiusChanged);
    connect(_editingPanel, &SegmentationEditingPanel::lineSigmaChanged,
            this, &SegmentationWidget::lineSigmaChanged);
    connect(_editingPanel, &SegmentationEditingPanel::pushPullRadiusChanged,
            this, &SegmentationWidget::pushPullRadiusChanged);
    connect(_editingPanel, &SegmentationEditingPanel::pushPullSigmaChanged,
            this, &SegmentationWidget::pushPullSigmaChanged);
    connect(_editingPanel, &SegmentationEditingPanel::pushPullStepChanged,
            this, &SegmentationWidget::pushPullStepChanged);
    connect(_editingPanel, &SegmentationEditingPanel::alphaPushPullConfigChanged,
            this, &SegmentationWidget::alphaPushPullConfigChanged);
    connect(_editingPanel, &SegmentationEditingPanel::smoothingStrengthChanged,
            this, &SegmentationWidget::smoothingStrengthChanged);
    connect(_editingPanel, &SegmentationEditingPanel::smoothingIterationsChanged,
            this, &SegmentationWidget::smoothingIterationsChanged);
    connect(_editingPanel, &SegmentationEditingPanel::hoverMarkerToggled,
            this, &SegmentationWidget::hoverMarkerToggled);
    connect(_editingPanel, &SegmentationEditingPanel::applyRequested,
            this, &SegmentationWidget::applyRequested);
    connect(_editingPanel, &SegmentationEditingPanel::resetRequested,
            this, &SegmentationWidget::resetRequested);
    connect(_editingPanel, &SegmentationEditingPanel::stopToolsRequested,
            this, &SegmentationWidget::stopToolsRequested);

    // Forward approval mask panel signals
    connect(_approvalMaskPanel, &SegmentationApprovalMaskPanel::showApprovalMaskChanged,
            this, &SegmentationWidget::showApprovalMaskChanged);
    connect(_approvalMaskPanel, &SegmentationApprovalMaskPanel::showApprovalMaskChanged,
            this, &SegmentationWidget::syncUiState);
    connect(_approvalMaskPanel, &SegmentationApprovalMaskPanel::editApprovedMaskChanged,
            this, &SegmentationWidget::editApprovedMaskChanged);
    connect(_approvalMaskPanel, &SegmentationApprovalMaskPanel::editUnapprovedMaskChanged,
            this, &SegmentationWidget::editUnapprovedMaskChanged);
    connect(_approvalMaskPanel, &SegmentationApprovalMaskPanel::autoApproveEditsChanged,
            this, &SegmentationWidget::autoApproveEditsChanged);
    connect(_approvalMaskPanel, &SegmentationApprovalMaskPanel::approvalBrushRadiusChanged,
            this, &SegmentationWidget::approvalBrushRadiusChanged);
    connect(_approvalMaskPanel, &SegmentationApprovalMaskPanel::approvalBrushDepthChanged,
            this, &SegmentationWidget::approvalBrushDepthChanged);
    connect(_approvalMaskPanel, &SegmentationApprovalMaskPanel::approvalMaskOpacityChanged,
            this, &SegmentationWidget::approvalMaskOpacityChanged);
    connect(_approvalMaskPanel, &SegmentationApprovalMaskPanel::approvalBrushColorChanged,
            this, &SegmentationWidget::approvalBrushColorChanged);
    connect(_approvalMaskPanel, &SegmentationApprovalMaskPanel::approvalStrokesUndoRequested,
            this, &SegmentationWidget::approvalStrokesUndoRequested);

    // Forward cell reopt panel signals
    connect(_cellReoptPanel, &SegmentationCellReoptPanel::cellReoptModeChanged,
            this, &SegmentationWidget::cellReoptModeChanged);
    connect(_cellReoptPanel, &SegmentationCellReoptPanel::cellReoptMaxStepsChanged,
            this, &SegmentationWidget::cellReoptMaxStepsChanged);
    connect(_cellReoptPanel, &SegmentationCellReoptPanel::cellReoptMaxPointsChanged,
            this, &SegmentationWidget::cellReoptMaxPointsChanged);
    connect(_cellReoptPanel, &SegmentationCellReoptPanel::cellReoptMinSpacingChanged,
            this, &SegmentationWidget::cellReoptMinSpacingChanged);
    connect(_cellReoptPanel, &SegmentationCellReoptPanel::cellReoptPerimeterOffsetChanged,
            this, &SegmentationWidget::cellReoptPerimeterOffsetChanged);
    connect(_cellReoptPanel, &SegmentationCellReoptPanel::cellReoptGrowthRequested,
            this, &SegmentationWidget::cellReoptGrowthRequested);

    // Forward growth panel signals
    connect(_growthPanel, &SegmentationGrowthPanel::growSurfaceRequested,
            this, &SegmentationWidget::growSurfaceRequested);
    connect(_growthPanel, &SegmentationGrowthPanel::growthMethodChanged,
            this, &SegmentationWidget::growthMethodChanged);
    connect(_growthPanel, &SegmentationGrowthPanel::volumeSelectionChanged,
            this, &SegmentationWidget::volumeSelectionChanged);
    connect(_growthPanel, &SegmentationGrowthPanel::correctionsZRangeChanged,
            this, &SegmentationWidget::correctionsZRangeChanged);

    // Forward corrections panel signals
    connect(_correctionsPanel, &SegmentationCorrectionsPanel::correctionsCreateRequested,
            this, &SegmentationWidget::correctionsCreateRequested);
    connect(_correctionsPanel, &SegmentationCorrectionsPanel::correctionsCollectionSelected,
            this, &SegmentationWidget::correctionsCollectionSelected);
    connect(_correctionsPanel, &SegmentationCorrectionsPanel::correctionsAnnotateToggled,
            this, &SegmentationWidget::correctionsAnnotateToggled);

    // Forward neural tracer panel signals
    connect(_neuralTracerPanel, &SegmentationNeuralTracerPanel::neuralTracerEnabledChanged,
            this, &SegmentationWidget::neuralTracerEnabledChanged);
    connect(_neuralTracerPanel, &SegmentationNeuralTracerPanel::neuralTracerStatusMessage,
            this, &SegmentationWidget::neuralTracerStatusMessage);
}

void SegmentationWidget::syncUiState()
{
    if (_headerRow) {
        _headerRow->setEditingChecked(_editingEnabled);
        if (_editingEnabled) {
            _headerRow->setStatusText(_pending ? tr("Editing enabled – pending changes")
                                               : tr("Editing enabled"));
        } else {
            _headerRow->setStatusText(tr("Editing disabled"));
        }
    }

    _growthPanel->syncUiState(_editingEnabled, _growthInProgress);
    _editingPanel->syncUiState(_editingEnabled, _growthInProgress);
    _customParamsPanel->syncUiState(_editingEnabled);
    _directionFieldPanel->syncUiState(_editingEnabled);
    _correctionsPanel->syncUiState(_editingEnabled, _growthInProgress);
    _approvalMaskPanel->syncUiState();
    _cellReoptPanel->syncUiState(_approvalMaskPanel->showApprovalMask(), _growthInProgress);
}

void SegmentationWidget::restoreSettings()
{
    using namespace vc3d::settings;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(QStringLiteral("segmentation_edit"));

    _restoringSettings = true;

    _editingPanel->restoreSettings(settings);
    _growthPanel->restoreSettings(settings);
    _directionFieldPanel->restoreSettings(settings);
    _correctionsPanel->restoreSettings(settings);
    _customParamsPanel->restoreSettings(settings);
    _approvalMaskPanel->restoreSettings(settings);
    _neuralTracerPanel->restoreSettings(settings);
    _cellReoptPanel->restoreSettings(settings);

    settings.endGroup();
    _restoringSettings = false;
}

void SegmentationWidget::writeSetting(const QString& key, const QVariant& value)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(QStringLiteral("segmentation_edit"));
    settings.setValue(key, value);
    settings.endGroup();
}

void SegmentationWidget::updateEditingState(bool enabled, bool notifyListeners)
{
    if (_editingEnabled == enabled) {
        return;
    }

    _editingEnabled = enabled;
    syncUiState();

    if (notifyListeners) {
        emit editingModeChanged(_editingEnabled);
    }
}

// --- Editing panel delegations ---

float SegmentationWidget::dragRadius() const { return _editingPanel->dragRadius(); }
float SegmentationWidget::dragSigma() const { return _editingPanel->dragSigma(); }
float SegmentationWidget::lineRadius() const { return _editingPanel->lineRadius(); }
float SegmentationWidget::lineSigma() const { return _editingPanel->lineSigma(); }
float SegmentationWidget::pushPullRadius() const { return _editingPanel->pushPullRadius(); }
float SegmentationWidget::pushPullSigma() const { return _editingPanel->pushPullSigma(); }
float SegmentationWidget::pushPullStep() const { return _editingPanel->pushPullStep(); }
AlphaPushPullConfig SegmentationWidget::alphaPushPullConfig() const { return _editingPanel->alphaPushPullConfig(); }
float SegmentationWidget::smoothingStrength() const { return _editingPanel->smoothingStrength(); }
int SegmentationWidget::smoothingIterations() const { return _editingPanel->smoothingIterations(); }
bool SegmentationWidget::showHoverMarker() const { return _editingPanel->showHoverMarker(); }

void SegmentationWidget::setDragRadius(float value) { _editingPanel->setDragRadius(value); }
void SegmentationWidget::setDragSigma(float value) { _editingPanel->setDragSigma(value); }
void SegmentationWidget::setLineRadius(float value) { _editingPanel->setLineRadius(value); }
void SegmentationWidget::setLineSigma(float value) { _editingPanel->setLineSigma(value); }
void SegmentationWidget::setPushPullRadius(float value) { _editingPanel->setPushPullRadius(value); }
void SegmentationWidget::setPushPullSigma(float value) { _editingPanel->setPushPullSigma(value); }
void SegmentationWidget::setPushPullStep(float value) { _editingPanel->setPushPullStep(value); }
void SegmentationWidget::setAlphaPushPullConfig(const AlphaPushPullConfig& config) { _editingPanel->setAlphaPushPullConfig(config); }
void SegmentationWidget::setSmoothingStrength(float value) { _editingPanel->setSmoothingStrength(value); }
void SegmentationWidget::setSmoothingIterations(int value) { _editingPanel->setSmoothingIterations(value); }
void SegmentationWidget::setShowHoverMarker(bool enabled) { _editingPanel->setShowHoverMarker(enabled); }

// --- Approval mask delegations ---

bool SegmentationWidget::showApprovalMask() const { return _approvalMaskPanel->showApprovalMask(); }
bool SegmentationWidget::editApprovedMask() const { return _approvalMaskPanel->editApprovedMask(); }
bool SegmentationWidget::editUnapprovedMask() const { return _approvalMaskPanel->editUnapprovedMask(); }
bool SegmentationWidget::autoApproveEdits() const { return _approvalMaskPanel->autoApproveEdits(); }
float SegmentationWidget::approvalBrushRadius() const { return _approvalMaskPanel->approvalBrushRadius(); }
float SegmentationWidget::approvalBrushDepth() const { return _approvalMaskPanel->approvalBrushDepth(); }
int SegmentationWidget::approvalMaskOpacity() const { return _approvalMaskPanel->approvalMaskOpacity(); }
QColor SegmentationWidget::approvalBrushColor() const { return _approvalMaskPanel->approvalBrushColor(); }

void SegmentationWidget::setShowApprovalMask(bool enabled) { _approvalMaskPanel->setShowApprovalMask(enabled); syncUiState(); }
void SegmentationWidget::setEditApprovedMask(bool enabled) { _approvalMaskPanel->setEditApprovedMask(enabled); }
void SegmentationWidget::setEditUnapprovedMask(bool enabled) { _approvalMaskPanel->setEditUnapprovedMask(enabled); }
void SegmentationWidget::setAutoApproveEdits(bool enabled) { _approvalMaskPanel->setAutoApproveEdits(enabled); }
void SegmentationWidget::setApprovalBrushRadius(float radius) { _approvalMaskPanel->setApprovalBrushRadius(radius); }
void SegmentationWidget::setApprovalBrushDepth(float depth) { _approvalMaskPanel->setApprovalBrushDepth(depth); }
void SegmentationWidget::setApprovalMaskOpacity(int opacity) { _approvalMaskPanel->setApprovalMaskOpacity(opacity); }
void SegmentationWidget::setApprovalBrushColor(const QColor& color) { _approvalMaskPanel->setApprovalBrushColor(color); }

// --- Cell reoptimization delegations ---

bool SegmentationWidget::cellReoptMode() const { return _cellReoptPanel->cellReoptMode(); }
int SegmentationWidget::cellReoptMaxSteps() const { return _cellReoptPanel->cellReoptMaxSteps(); }
int SegmentationWidget::cellReoptMaxPoints() const { return _cellReoptPanel->cellReoptMaxPoints(); }
float SegmentationWidget::cellReoptMinSpacing() const { return _cellReoptPanel->cellReoptMinSpacing(); }
float SegmentationWidget::cellReoptPerimeterOffset() const { return _cellReoptPanel->cellReoptPerimeterOffset(); }

void SegmentationWidget::setCellReoptMode(bool enabled) { _cellReoptPanel->setCellReoptMode(enabled); syncUiState(); }
void SegmentationWidget::setCellReoptCollections(const QVector<QPair<uint64_t, QString>>& collections) { _cellReoptPanel->setCellReoptCollections(collections); syncUiState(); }

void SegmentationWidget::setPendingChanges(bool pending)
{
    if (_pending == pending) {
        return;
    }
    _pending = pending;
    syncUiState();
}

void SegmentationWidget::setEditingEnabled(bool enabled)
{
    updateEditingState(enabled, false);
}

// --- Growth panel delegations ---

SegmentationGrowthMethod SegmentationWidget::growthMethod() const { return _growthPanel->growthMethod(); }
int SegmentationWidget::growthSteps() const { return _growthPanel->growthSteps(); }
int SegmentationWidget::extrapolationPointCount() const { return _growthPanel->extrapolationPointCount(); }
ExtrapolationType SegmentationWidget::extrapolationType() const { return _growthPanel->extrapolationType(); }
int SegmentationWidget::sdtMaxSteps() const { return _growthPanel->sdtMaxSteps(); }
float SegmentationWidget::sdtStepSize() const { return _growthPanel->sdtStepSize(); }
float SegmentationWidget::sdtConvergence() const { return _growthPanel->sdtConvergence(); }
int SegmentationWidget::sdtChunkSize() const { return _growthPanel->sdtChunkSize(); }
int SegmentationWidget::skeletonConnectivity() const { return _growthPanel->skeletonConnectivity(); }
int SegmentationWidget::skeletonSliceOrientation() const { return _growthPanel->skeletonSliceOrientation(); }
int SegmentationWidget::skeletonChunkSize() const { return _growthPanel->skeletonChunkSize(); }
int SegmentationWidget::skeletonSearchRadius() const { return _growthPanel->skeletonSearchRadius(); }
bool SegmentationWidget::growthKeybindsEnabled() const { return _growthPanel->growthKeybindsEnabled(); }
QString SegmentationWidget::normal3dZarrPath() const { return _growthPanel->normal3dZarrPath(); }
std::vector<SegmentationGrowthDirection> SegmentationWidget::allowedGrowthDirections() const { return _growthPanel->allowedGrowthDirections(); }
std::optional<std::pair<int, int>> SegmentationWidget::correctionsZRange() const { return _growthPanel->correctionsZRange(); }

void SegmentationWidget::setGrowthMethod(SegmentationGrowthMethod method) { _growthPanel->setGrowthMethod(method); }
void SegmentationWidget::setGrowthSteps(int steps, bool persist) { _growthPanel->setGrowthSteps(steps, persist); }
void SegmentationWidget::setGrowthInProgress(bool running)
{
    if (_growthInProgress == running) {
        return;
    }
    _growthInProgress = running;
    _growthPanel->setGrowthInProgress(running);
}
void SegmentationWidget::setNormalGridAvailable(bool available) { _growthPanel->setNormalGridAvailable(available); }
void SegmentationWidget::setNormalGridPathHint(const QString& hint) { _growthPanel->setNormalGridPathHint(hint); }
void SegmentationWidget::setNormalGridPath(const QString& path) { _growthPanel->setNormalGridPath(path); }
void SegmentationWidget::setNormal3dZarrCandidates(const QStringList& candidates, const QString& hint) { _growthPanel->setNormal3dZarrCandidates(candidates, hint); }
void SegmentationWidget::setVolumePackagePath(const QString& path) { _growthPanel->setVolumePackagePath(path); }
void SegmentationWidget::setAvailableVolumes(const QVector<QPair<QString, QString>>& volumes, const QString& activeId) { _growthPanel->setAvailableVolumes(volumes, activeId); }
void SegmentationWidget::setActiveVolume(const QString& volumeId) { _growthPanel->setActiveVolume(volumeId); }

// --- Corrections delegations ---

void SegmentationWidget::setCorrectionsEnabled(bool enabled) { _correctionsPanel->setCorrectionsEnabled(enabled); }
void SegmentationWidget::setCorrectionsAnnotateChecked(bool enabled) { _correctionsPanel->setCorrectionsAnnotateChecked(enabled); }

void SegmentationWidget::setCorrectionCollections(const QVector<QPair<uint64_t, QString>>& collections,
                                                  std::optional<uint64_t> activeId)
{
    _correctionsPanel->setCorrectionCollections(collections, activeId);
    _correctionsPanel->syncUiState(_editingEnabled, _growthInProgress);
}

// --- Direction field delegations ---

std::vector<SegmentationDirectionFieldConfig> SegmentationWidget::directionFieldConfigs() const
{
    return _directionFieldPanel->directionFieldConfigs();
}

// --- Custom params delegations ---

QString SegmentationWidget::customParamsText() const { return _customParamsPanel->customParamsText(); }
QString SegmentationWidget::customParamsProfile() const { return _customParamsPanel->customParamsProfile(); }
bool SegmentationWidget::customParamsValid() const { return _customParamsPanel->customParamsValid(); }
QString SegmentationWidget::customParamsError() const { return _customParamsPanel->customParamsError(); }
std::optional<nlohmann::json> SegmentationWidget::customParamsJson() const { return _customParamsPanel->customParamsJson(); }

// --- Neural tracer delegations ---

bool SegmentationWidget::neuralTracerEnabled() const { return _neuralTracerPanel->neuralTracerEnabled(); }
QString SegmentationWidget::neuralCheckpointPath() const { return _neuralTracerPanel->neuralCheckpointPath(); }
QString SegmentationWidget::neuralPythonPath() const { return _neuralTracerPanel->neuralPythonPath(); }
QString SegmentationWidget::volumeZarrPath() const { return _neuralTracerPanel->volumeZarrPath(); }
int SegmentationWidget::neuralVolumeScale() const { return _neuralTracerPanel->neuralVolumeScale(); }
int SegmentationWidget::neuralBatchSize() const { return _neuralTracerPanel->neuralBatchSize(); }

void SegmentationWidget::setNeuralTracerEnabled(bool enabled) { _neuralTracerPanel->setNeuralTracerEnabled(enabled); }
void SegmentationWidget::setNeuralCheckpointPath(const QString& path) { _neuralTracerPanel->setNeuralCheckpointPath(path); }
void SegmentationWidget::setNeuralPythonPath(const QString& path) { _neuralTracerPanel->setNeuralPythonPath(path); }
void SegmentationWidget::setNeuralVolumeScale(int scale) { _neuralTracerPanel->setNeuralVolumeScale(scale); }
void SegmentationWidget::setNeuralBatchSize(int size) { _neuralTracerPanel->setNeuralBatchSize(size); }
void SegmentationWidget::setVolumeZarrPath(const QString& path) { _neuralTracerPanel->setVolumeZarrPath(path); }

void SegmentationWidget::setEraseBrushActive(bool /*active*/) {}
