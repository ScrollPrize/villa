#pragma once

#include "LineAnnotationController.hpp"

#include <QMainWindow>
#include <QFutureWatcher>
#include <QHash>
#include <QImage>
#include <QJsonObject>
#include <QColor>
#include <QSet>
#include <QStringList>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <functional>

class AxisAlignedSliceController;
class CState;
class ConsoleOutputWidget;
class QDialog;
class QComboBox;
class QKeyEvent;
class QuadSurface;
class SpiralPanel;
class SpiralServiceManager;
class ViewerManager;
class ViewerSplitGrid;
class VolumePkg;
class Volume;
class SpiralOverlayController;
class SpiralMinimap;
class SpiralBrushController;
class SpiralLineDraftOverlay;
class SegmentationOverlayController;
class PointsOverlayController;
class VCCollection;
class VolumeViewerBase;

class SpiralWorkspace : public QMainWindow
{
    Q_OBJECT
public:
    explicit SpiralWorkspace(CState* mainState, QWidget* parent = nullptr);
    ~SpiralWorkspace() override;

    ViewerManager* viewerManager() const { return _viewerManager.get(); }
    ViewerSplitGrid* viewerGrid() const { return _grid; }
    QComboBox* volumeSelectionControl() const;
    void synchronizeVolume(const std::shared_ptr<Volume>& volume,
                           const std::optional<cv::Matx44d>& navigationTransform = std::nullopt);

    // Cross-panel entry points for "Add to current spiral fit".
    bool hasActiveSpiralSession() const;
    void addPatchToCurrentFit(const QString& tifxyzDirectory,
                              const std::shared_ptr<QuadSurface>& surface = {});
    void addFiberToCurrentFit(uint64_t fiberId, const QString& fiberJsonPath);
    void noteTrackedFiberSaved(uint64_t fiberId, uint64_t generation,
                               const QString& fiberJsonPath);
    void requestSessionExit(std::function<void()> continuation);
    bool hasPendingBrushWork() const;
    void setLineAnnotationController(LineAnnotationController* controller);
    [[nodiscard]] bool isFlattenedViewer(const VolumeViewerBase* viewer) const;
    [[nodiscard]] QString lineAnnotationDraftUnavailableReason() const;
    [[nodiscard]] std::optional<double> fiberBaseToPreviewFactor() const {
        return _fiberBaseToPreviewFactor;
    }
    void startLineAnnotationDraft();

signals:
    void spiralSessionActiveChanged(bool active);
    void fiberBaseToPreviewFactorChanged(double factor, bool valid);

protected:
    void keyPressEvent(QKeyEvent* event) override;

private:
    struct PreviewComponent {
        int rowBegin = 0;
        int rowEnd = 0;
        int columnBegin = 0;
        int columnEnd = 0;
        int winding = 0;
    };
    struct PreviewLoadResult {
        std::shared_ptr<QuadSurface> surface;
        QString surfaceId;
        std::vector<PreviewComponent> components;
        cv::Mat_<int32_t> windingIds;
        std::optional<std::array<std::size_t, 3>> baseShapeZYX;
        QString error;
        struct LossMap {
            QString name;
            QString relativePath;
            QString imagePath;
            double weight = 0.0;
            double p50 = 0.0;
            double p95 = 0.0;
            double maximum = 0.0;
            double displayMaximum = 0.0;
            qint64 sampleCount = 0;
            qint64 eligibleSampleCount = 0;
            qint64 projectedSampleCount = 0;
            qint64 offSurfaceSampleCount = 0;
            qint64 omittedSampleCount = 0;
            qint64 supportedPixels = 0;
        };
        std::vector<LossMap> lossMaps;
        QString runDiffImagePath;
    };
    struct PreviewDisplaySelection {
        cv::Rect region;
        int minimumWinding = 0;
        int maximumWinding = -1;
        QString registrationId;
    };
    struct InputSurfaceEntry {
        QString category;
        QString id;
        QString sourceId;
        std::shared_ptr<QuadSurface> surface;
    };
    struct InputSurfaceLoadResult {
        std::vector<InputSurfaceEntry> surfaces;
        QStringList warnings;
    };

    QString mapServicePath(const QString& servicePath) const;
    // Loss-map entries of one manifest, resolved against the artifact
    // directory holding them. Shared by the surface manifest (older services
    // still carry the overlays there) and the diagnostics manifest.
    static std::vector<PreviewLoadResult::LossMap> parseLossMaps(
        const QJsonObject& manifest, const QString& artifactRoot);
    void loadPreview(const QString& manifestPath, qint64 generation);
    void installPreview(const PreviewLoadResult& result, qint64 generation);
    // Adopt the loss overlays published for the installed preview. They are a
    // separate artifact that lands after the surface, so this only extends
    // what is already displayed - it never reloads the surface.
    void installPreviewDiagnostics(const QString& manifestPath,
                                   qint64 generation);
    void installSameWindingArtifact(const QString& manifestPath,
                                    const QJsonObject& artifactRef);
    void refreshSameWindingOverlay();
    void applyPreviewWindingRange(bool preserveFocus);
    void loadRunDiff();
    void updateRunDiffOverlay();
    void updateLossMapOverlay();
    void updateWindingTransitionOverlay();
    void updateWindingMinimap();
    void updateMinimapViewIndicator();
    void panFlattenedViewerToColumn(float column);
    std::optional<PreviewDisplaySelection> displayedPreviewSelection() const;
    void installPreviewAliasWhenIndexed(const std::shared_ptr<QuadSurface>& preview,
                                        const QString& registrationId,
                                        qint64 generation, quint64 revision,
                                        bool preserveFocus, int attempt);
    void loadInputSurfaces(const QJsonObject& paths, quint64 generation);
    void installInputSurfaces(const InputSurfaceLoadResult& result, quint64 generation);
    void registerPendingPatchSurface(const QString& inputId,
                                     const std::shared_ptr<QuadSurface>& surface,
                                     const std::optional<QColor>& color = std::nullopt);
    void finalizeBrushPaint();
    void finalizeLineAnnotationDraft();
    void cancelLineAnnotationDraft();
    void undoLineAnnotationDraftPoint();
    void appendLineAnnotationDraftPoint(const QPointF& scenePoint,
                                        Qt::KeyboardModifiers modifiers);
    [[nodiscard]] std::optional<LineAnnotationController::ResolvedFiberOptimizationInputs>
        resolveLineAnnotationInputs(QString* errorMessage) const;
    [[nodiscard]] QStringList fallbackFiberManifests() const;
    [[nodiscard]] std::optional<std::array<std::size_t, 3>>
        resolveFiberBaseShape(QString* errorMessage) const;
    void updatePreviewCoordinateScale();
    void submitReadyDrafts(bool commitAfterAdd);
    void maybeCommitForPendingExit();
    QString provisionalBrushRoot() const;
    void discardBrushWork();
    void setSurfaceCategoryVisible(const QString& category, bool visible);
    void updatePendingPatchIds(const QJsonObject& status);
    void uploadNewestFiberRevision(uint64_t fiberId);
    void updateSurfaceIntersections();
    void ensureInitialFocus();
    void initializePreviewFocus();
    void mirrorFocusToMainWorkspace(const cv::Vec3f& position);

    CState* _mainState = nullptr;
    CState* _state = nullptr;
    std::unique_ptr<ViewerManager> _viewerManager;
    std::unique_ptr<AxisAlignedSliceController> _slices;
    std::unique_ptr<SpiralOverlayController> _overlay;
    std::unique_ptr<SpiralBrushController> _brush;
    std::unique_ptr<SpiralLineDraftOverlay> _lineDraftOverlay;
    std::unique_ptr<VCCollection> _sameWindingCollection;
    std::unique_ptr<PointsOverlayController> _sameWindingOverlay;
    LineAnnotationController* _lineAnnotationController = nullptr;
    std::unique_ptr<SegmentationOverlayController> _surfaceOverlapOverlay;
    SpiralServiceManager* _service = nullptr;
    SpiralPanel* _panel = nullptr;
    ConsoleOutputWidget* _pythonOutput = nullptr;
    QDialog* _pythonOutputDialog = nullptr;
    ViewerSplitGrid* _grid = nullptr;
    VolumeViewerBase* _flattenedViewer = nullptr;
    SpiralMinimap* _windingMinimap = nullptr;
    qint64 _requestedPreviewGeneration = -1;
    QJsonObject _sessionPaths;
    QJsonObject _sessionRunConfig;
    QString _externalFiberSource;
    struct LineAnnotationDraft {
        std::shared_ptr<QuadSurface> surface;
        std::vector<QPointF> surfacePoints;
        std::shared_ptr<std::atomic_bool> saveAllowed;
        bool optimizing = false;
    };
    std::optional<LineAnnotationDraft> _lineAnnotationDraft;
    QHash<QString, QStringList> _surfaceCategoryIds;
    QHash<QString, QString> _surfaceSourceIds;
    QHash<QString, bool> _surfaceCategoryVisible;
    QSet<QString> _pendingPatchIds;
    QSet<QString> _visibleUncommittedPointCollectionIds;
    std::map<std::string, std::size_t> _surfaceOverlayColorAssignments;
    std::map<std::string, cv::Vec3b> _surfaceOverlayColors;
    std::size_t _nextSurfaceOverlayColorIndex = 0;
    quint64 _inputSurfaceGeneration = 0;
    std::shared_ptr<QuadSurface> _previewSource;
    QString _previewSourceId;
    std::optional<std::array<std::size_t, 3>> _previewBaseShapeZYX;
    std::optional<std::array<std::size_t, 3>> _sameWindingBaseShapeZYX;
    QString _sameWindingManifestPath;
    std::optional<std::array<std::size_t, 3>> _fiberBaseShapeZYX;
    std::optional<double> _previewToFiberBaseScale;
    std::optional<double> _fiberBaseToPreviewFactor;
    QString _previewCoordinateError;
    std::vector<PreviewComponent> _previewComponents;
    cv::Mat_<int32_t> _previewWindingIds;
    QString _previewRunDiffImagePath;
    std::shared_ptr<QuadSurface> _currentPreview;
    QString _currentPreviewRegistrationId;
    QImage _previewRunDiffImage;
    QHash<QString, PreviewLoadResult::LossMap> _previewLossMaps;
    QString _selectedLossMap;
    QString _loadedLossMap;
    QSet<QString> _fetchingLossMaps;
    QImage _loadedLossMapImage;
    qreal _lossMapOpacity = 0.8;
    quint64 _previewDisplayRevision = 0;
    quint64 _runDiffRequestRevision = 0;
    int _minimumDisplayedWinding = 10;
    int _maximumDisplayedWinding = 130;
    bool _outputVisible = true;
    bool _showSurfaceIntersections = true;
    bool _showSurfaceOverlap = true;
    bool _pendingPatchesOnly = false;
    bool _runDiffVisible = false;
    bool _windingTransitionsVisible = true;
    bool _sameWindingPclsVisible = false;
    // True while the focus is the automatic volume-center default (no user
    // interaction and no preview yet); the first preview may then retarget it.
    bool _focusIsAutoDefault = false;
    bool _shuttingDown = false;
    struct PendingBrushPatch {
        QString path;
        QColor color;
        std::shared_ptr<QuadSurface> surface;
    };
    QHash<QString, PendingBrushPatch> _pendingBrushPatches;
    QHash<QString, QString> _brushProvisionalPaths;
    QSet<QString> _unverifiedBrushIds;
    QHash<QString, QString> _pendingPointCollectionPaths;
    QHash<QString, QString> _pointCollectionProvisionalPaths;
    QSet<QString> _uncommittedPointCollectionIds;
    std::function<void()> _pendingExitAction;
    bool _commitAfterBrushUploads = false;
    struct TrackedFiber {
        QString inputId;
        QString path;
        QString revision;
        QString snapshotPath;
        uint64_t latestGeneration = 0;
        uint64_t sentGeneration = 0;
        uint64_t inFlightGeneration = 0;
        bool added = false;
        bool uploadInFlight = false;
    };
    QHash<uint64_t, TrackedFiber> _trackedFibers;
    QHash<QString, QString> _residentFiberRevisions;
};
