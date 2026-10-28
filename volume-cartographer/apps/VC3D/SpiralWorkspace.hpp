#pragma once

#include "LineAnnotationController.hpp"
#include "SpiralPclRole.hpp"
#include "overlays/SpiralFiberOverlayController.hpp"

#include <QMainWindow>
#include <QFutureWatcher>
#include <QHash>
#include <QImage>
#include <QJsonObject>
#include <QColor>
#include <QSet>
#include <QStringList>
#include <QTemporaryDir>
#include <nlohmann/json_fwd.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vc_reader/affine.hpp>
#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <functional>

namespace vc3d::spiral {
struct FitInputs;
struct FitInputChanges;
struct FitSurface;
}

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
class QTimer;
class StoreSurfaceOverlayController;
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
                           const std::optional<vcr::catalog::Affine>& navigationTransform = std::nullopt);

    // Cross-panel entry points for "Add to current spiral fit".
    void addPatchToCurrentFit(const QString& tifxyzDirectory,
                              const std::shared_ptr<QuadSurface>& surface = {});
    void addFibersToCurrentFit(const std::vector<uint64_t>& fiberIds);
    void requestSessionExit(std::function<void()> continuation);
    bool hasPendingBrushWork() const;
    void setLineAnnotationController(LineAnnotationController* controller);
    [[nodiscard]] bool isFlattenedViewer(const VolumeViewerBase* viewer) const;
    // Why a draft cannot start now, ignoring the object store, which the caller
    // offers to create or attach.
    [[nodiscard]] QString lineAnnotationDraftUnavailableReason() const;
    void startLineAnnotationDraft();

signals:
    void spiralSessionActiveChanged(bool active);
    void patchEditorRequested(const QString& id, const QString& path);
    // Clone a hub (the Spiral service's) to a local replica and attach it.
    void hubReplicaRequested(const QString& hubAddress);

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
        // The surface categories this result replaces; others are kept. With
        // none, the surfaces are added or replaced by ID and `removedIds`
        // removed, leaving every other surface as it is.
        QStringList categories;
        QStringList removedIds;
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
    // The fit's inputs as the attached store holds them now, with how they
    // differ from what the session's input manifest says the fit read
    // (vc3d::spiral::readFitInputs).
    void loadFitInputs();
    void installFitInputs(const std::shared_ptr<vc3d::spiral::FitInputs>& inputs,
                             quint64 generation);
    void clearFitInputs(const QString& reason);
    // Re-read only the fitted objects and links store commits changed, and
    // replace only their overlays (vc3d::spiral::readFitInputChanges).
    void loadFitInputChanges();
    void installFitInputChanges(vc3d::spiral::FitInputChanges changes);
    InputSurfaceEntry fitSurfaceEntry(const vc3d::spiral::FitSurface& entry) const;
    void rebuildFitPclDocuments();
    void rebuildFitFibers();
    void showFitInputProblems(const vc3d::spiral::FitInputs& inputs);
    // Queue the fitted objects and links a store commit changed for
    // loadFitInputChanges; count other inputs a rebuild's selection may include.
    void noteStoreChanges(const vc::objects::StoreSessionEvent& event);
    void updateInputFreshness();
    void refreshPclOverlay(vc3d::spiral::PclRole role);
    void refreshFiberOverlay();
    void refreshFitOverlays();
    // Reference-volume voxels -> the preview's volume frame; nullopt (with
    // the reason) when either side lacks or disagrees on a coordinate domain.
    std::optional<double> fitInputScale(QString* error) const;
    void applyPreviewWindingRange(bool preserveFocus);
    void loadRunDiff();
    void updateRunDiffOverlay();
    void updateLossMapOverlay();
    void updateWindingTransitionOverlay();
    void updateWindingMinimap();
    void updateMinimapViewIndicator();
    void panFlattenedViewerToColumn(float column);
    std::optional<PreviewDisplaySelection> displayedPreviewSelection() const;
    void syncSurfaceDrafts(const std::shared_ptr<QuadSurface>& edited = nullptr);
    void clearSurfaceDrafts();
    void installPreviewAlias(const std::shared_ptr<QuadSurface>& preview, const QString& registrationId,
                             qint64 generation, quint64 revision, bool preserveFocus);
    void loadInputSurfaces(const QJsonObject& paths, quint64 generation);
    void installInputSurfaces(const InputSurfaceLoadResult& result, quint64 generation);
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
    // Save the brush's ready drafts into the attached store (store-backed
    // edits); ``updateAfter`` then runs Update Fit Inputs.
    void submitReadyDrafts(bool updateAfter);
    void discardBrushWork();
    void setSurfaceCategoryVisible(const QString& category, bool visible);
    void pruneBrushPreviewSurfaces();
    QSet<QString> _retainedBrushPreviewIds;
    // Brush patches saved to the store, by gesture id (their object and the
    // revision the save wrote, which their next edit starts from), and
    // drafts saved but not yet part of the fit (dropped once an input update
    // includes them).
    struct SavedPatch {
        std::string uuid, revision;
    };
    QHash<QString, SavedPatch> _brushPatches;
    QStringList _savedDraftIds;
    QStringList _savedPatchIds;
    // Those saved when the input update in flight began.
    QStringList _updatingDraftIds;
    QStringList _updatingPatchIds;
    void updateSurfaceIntersections();
    void ensureInitialFocus();
    void initializePreviewFocus();
    void mirrorFocusToMainWorkspace(const cv::Vec3f& position);

    CState* _mainState = nullptr;
    CState* _state = nullptr;
    std::unique_ptr<ViewerManager> _viewerManager;
    std::unique_ptr<AxisAlignedSliceController> _slices;
    std::unique_ptr<SpiralOverlayController> _overlay;
    std::unique_ptr<StoreSurfaceOverlayController> _storeSurfaceOverlay;
    // Service previews, shells and local brush surfaces are session-only
    // drafts for store queries. Fitted surfaces retain their stored UUIDs
    // and are queried directly, without overriding another editor's draft.
    std::map<std::string, std::weak_ptr<QuadSurface>> _surfaceDrafts;
    std::unique_ptr<SpiralBrushController> _brush;
    std::unique_ptr<SpiralLineDraftOverlay> _lineDraftOverlay;
    // The display-only overlay of the fit's point collections of one role
    // (same-winding, or relative-winding with the absolute ones), read from
    // the attached store as it is now.
    struct PclOverlayState {
        std::unique_ptr<VCCollection> collection;
        std::unique_ptr<PointsOverlayController> overlay;
        // The fit's collections as a point-collections document, and that
        // document on disk for VCCollection::loadFromJSON; empty when none.
        QJsonObject document;
        QString documentPath;
        bool visible = false;
    };
    std::array<PclOverlayState, vc3d::spiral::kEditablePclRoles.size()> _pclOverlays;
    PclOverlayState& pclOverlay(vc3d::spiral::PclRole role)
    {
        return _pclOverlays[vc3d::spiral::pclRoleIndex(role)];
    }
    // The fit's fibers as the store holds them now, in reference-volume voxels.
    std::unique_ptr<SpiralFiberOverlayController> _fiberOverlay;
    std::vector<SpiralFiberOverlayController::Fiber> _fitFibers;
    bool _fiberOverlayVisible = false;
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
    std::map<std::string, std::size_t> _surfaceOverlayColorAssignments;
    std::map<std::string, cv::Vec3b> _surfaceOverlayColors;
    std::size_t _nextSurfaceOverlayColorIndex = 0;
    quint64 _inputSurfaceGeneration = 0;
    // The session's input manifest and what the attached store holds of it.
    QString _inputManifestPath;
    QString _inputDigest;
    quint64 _fitInputGeneration = 0;
    std::shared_ptr<vc3d::spiral::FitInputs> _fitInputs;
    // Record UUIDs of the fitted objects and relationships, and of the other
    // surfaces, collections and fibers changed since the fit's inputs loaded.
    QSet<QString> _fitRecords;
    QSet<QString> _changedOtherInputs;
    QTimer* _fitInputReload = nullptr;
    // The manifest being displayed, and the fitted objects and links changed
    // by commits not yet read.
    std::shared_ptr<const nlohmann::json> _fitDocument;
    std::set<std::string> _pendingObjectChanges;
    std::set<std::string> _pendingLinkChanges;
    bool _fitChangesReading = false;
    // The generation of the full read _fitInputs holds.
    quint64 _installedFitInputGeneration = 0;
    QTemporaryDir _fitPclDocuments;
    std::shared_ptr<QuadSurface> _previewSource;
    QString _previewSourceId;
    std::optional<std::array<std::size_t, 3>> _previewBaseShapeZYX;
    std::optional<std::array<std::size_t, 3>> _fiberBaseShapeZYX;
    std::optional<double> _previewToFiberBaseScale;
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
    bool _runDiffVisible = false;
    bool _windingTransitionsVisible = true;
    // True while the focus is the automatic volume-center default (no user
    // interaction and no preview yet); the first preview may then retarget it.
    bool _focusIsAutoDefault = false;
    bool _shuttingDown = false;
};
