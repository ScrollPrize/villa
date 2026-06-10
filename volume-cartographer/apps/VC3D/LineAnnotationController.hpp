#pragma once

#include <QObject>
#include <QPointF>
#include <QPointer>
#include <QString>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "LineAnnotationFiberClassification.hpp"
#include "vc/atlas/FiberIntersections.hpp"
#include "vc/lasagna/LineOptimizer.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"

class CState;
class FiberSliceOverlayController;
class LineAnnotationDialog;
class QMdiArea;
class QEvent;
class QPoint;
class Surface;
class SurfacePanelController;
class ViewerManager;
class VolumePkg;
class QWidget;

class LineAnnotationController : public QObject
{
    Q_OBJECT

public:
    enum class InitialDirectionMode {
        Sideways,
        ZInOut,
    };

    struct OptimizationTaskResult {
        bool ok = false;
        std::filesystem::path manifestPath;
        cv::Vec3d seedPoint{0.0, 0.0, 0.0};
        std::vector<vc::lasagna::LineControlPoint> controlPoints;
        cv::Vec3d sourceSliceNormal{0.0, 0.0, 1.0};
        InitialDirectionMode initialDirectionMode = InitialDirectionMode::Sideways;
        vc::lasagna::LineOptimizationResult result;
        std::string error;
        std::string eventName;
    };

    struct FiberSummary {
        uint64_t id = 0;
        std::string name;
        int controlPointCount = 0;
        int linePointCount = 0;
        double lengthVx = 0.0;
        double hvZDistance = 0.0;
        double hvFiberLength = 0.0;
        double horizontalScore = 0.0;
        double verticalScore = 0.0;
        double automaticCertainty = 0.0;
        std::string automaticHvTag;
        std::string manualHvTag;
        std::vector<std::string> tags;
    };

    struct FiberSnapshotWithPath {
        std::filesystem::path fiberPath;
        vc::atlas::FiberPolyline fiber;
    };

    using DatasetPicker =
        std::function<std::optional<std::string>(QWidget*, const std::filesystem::path&)>;
    using OptimizationTaskFactory =
        std::function<OptimizationTaskResult(std::filesystem::path,
                                             std::vector<vc::lasagna::LineControlPoint>,
                                             std::vector<cv::Vec3d>,
                                             cv::Vec3d,
                                             InitialDirectionMode,
                                             bool,
                                             int,
                                             int)>;

    LineAnnotationController(CState* state,
                             ViewerManager* viewerManager,
                             QWidget* parentWidget,
                             QObject* parent = nullptr);
    ~LineAnnotationController() override;

    bool canLaunchFromViewer(const CChunkedVolumeViewer* viewer) const;
    void launchFromViewer(CChunkedVolumeViewer* viewer, const QPointF& scenePoint);
    void launchFromViewerAtPoint(CChunkedVolumeViewer* viewer, const QPointF& scenePoint);
    void openFiber(uint64_t fiberId);
    void deleteFiber(uint64_t fiberId);
    void deleteFibers(std::vector<uint64_t> fiberIds);
    void renameFiberFile(uint64_t fiberId);
    void setFiberManualHvTag(uint64_t fiberId, const QString& tag);
    void setFiberTag(uint64_t fiberId, const QString& tag, bool enabled);
    void recalculateFiberHvClassification(uint64_t fiberId);
    void recalculateAllFiberHvClassifications();
    void createAtlasFromFiber(uint64_t fiberId);
    void showFiberSlice(uint64_t fiberId, QMdiArea* targetArea);
    void showIntersectionInspection(const vc::atlas::FiberIntersectionResult& result,
                                    QMdiArea* targetArea,
                                    std::optional<std::filesystem::path> atlasDir = std::nullopt);
    void saveOpenFibers();
    void closeFiberWindowForSurface(const std::string& surfaceName);
    bool showGeneratedControlPointContextMenu(CChunkedVolumeViewer* viewer,
                                              const QPointF& scenePoint,
                                              const QPoint& globalPos);
    [[nodiscard]] std::vector<FiberSummary> fiberSummaries() const;
    [[nodiscard]] std::vector<std::string> knownFiberTags() const;
    [[nodiscard]] std::vector<vc::atlas::FiberPolyline> fiberSnapshots() const;
    [[nodiscard]] std::vector<vc::atlas::FiberPolyline> fiberSnapshotsFromStorage() const;
    [[nodiscard]] std::vector<FiberSnapshotWithPath> fiberSnapshotsFromStorageWithPaths() const;

    void setDatasetPickerForTesting(DatasetPicker picker);
    void setOptimizationTaskFactoryForTesting(OptimizationTaskFactory factory);
    void setSurfacePanel(SurfacePanelController* panel);

signals:
    void fibersChanged(std::vector<LineAnnotationController::FiberSummary> fibers);
    void fiberSaved(uint64_t fiberId, uint64_t generation);
    void fibersDeleted(std::vector<uint64_t> fiberIds);
    void atlasCreated(std::filesystem::path atlasDir);

private slots:
    void onSurfaceChanged(std::string name, std::shared_ptr<Surface> surf, bool isEditUpdate = false);
    void onVolumePackageChanged(std::shared_ptr<VolumePkg> pkg);

private:
    enum class SourceKind {
        Plane,
        Segmentation,
    };

    struct LineAnnotationSession;
    struct IntersectionInspectionSession;
    struct StoredFiber {
        uint64_t id = 0;
        std::string username;
        std::string startedAt;
        uint64_t sequence = 0;
        std::string fileName;
        uint64_t generation = 1;
        std::vector<cv::Vec3d> controlPoints;
        std::vector<cv::Vec3d> linePoints;
        vc3d::line_annotation::FiberHvClassification hvClassification;
        std::string manualHvTag;
        std::vector<std::string> tags;
        bool needsSave = false;
    };

    struct PaneRecord {
        int id = 0;
        SourceKind sourceKind = SourceKind::Plane;
        std::string surfaceName;
        QPointer<LineAnnotationDialog> dialog;
        std::shared_ptr<LineAnnotationSession> session;
    };

    std::string nextSurfaceName();
    void cleanupSurfaceName(const std::string& surfaceName);
    void launchSession(SourceKind sourceKind,
                       const std::string& surfaceName,
                       std::shared_ptr<Surface> sourceSurface,
                       const CChunkedVolumeViewer::CameraState& camera,
                       cv::Vec3d sourceSliceNormal,
                       std::shared_ptr<LineAnnotationSession> session,
                       bool deferShowUntilGenerated = false);
    void handleLineSeed(const std::string& surfaceName,
                        cv::Vec3f volumePoint,
                        InitialDirectionMode directionMode);
    void handleGeneratedControlPoint(const std::string& surfaceName,
                                     cv::Vec3f volumePoint,
                                     double linePosition);
    void handleGeneratedControlPointDelete(const std::string& surfaceName,
                                           double linePosition,
                                           cv::Vec3f volumePoint);
    bool ensureDatasetForSession(LineAnnotationSession& session);
    void startOptimization(LineAnnotationSession& session,
                           bool fullOptimization = false,
                           int activeStart = -1,
                           int activeEnd = -1);
    void finishOptimization(const std::string& surfaceName);
    bool materializeGeneratedViews(LineAnnotationSession& session);
    bool materializeGeneratedViews(LineAnnotationSession& session,
                                   const std::string& surfacePrefix);
    void handleShowAsMesh(const std::string& surfaceName);
    [[nodiscard]] std::filesystem::path resolveMeshExportPathsDir() const;
    [[nodiscard]] std::filesystem::path nextMeshExportPath(const std::filesystem::path& pathsDir,
                                                           const std::string& stem) const;
    [[nodiscard]] std::vector<std::filesystem::path> saveGeneratedQuadMeshes(LineAnnotationSession& session);
    [[nodiscard]] PaneRecord* paneForSurface(const std::string& surfaceName);
    [[nodiscard]] const PaneRecord* paneForSurface(const std::string& surfaceName) const;
    [[nodiscard]] std::optional<std::string> pickDataset(QWidget* parent,
                                                          const std::filesystem::path& startDir) const;
    [[nodiscard]] OptimizationTaskResult runOptimizationTask(std::filesystem::path manifestPath,
                                                             std::vector<vc::lasagna::LineControlPoint> controlPoints,
                                                             std::vector<cv::Vec3d> initialLinePoints,
                                                             cv::Vec3d sourceSliceNormal,
                                                             InitialDirectionMode directionMode,
                                                             bool fullOptimization = false,
                                                             int activeStart = -1,
                                                             int activeEnd = -1) const;
    void loadFibersForCurrentPackage();
    void emitFiberSummaries();
    void addKnownFiberTags(const std::vector<std::string>& tags);
    [[nodiscard]] std::filesystem::path fibersDir() const;
    [[nodiscard]] std::filesystem::path fiberPath(uint64_t fiberId) const;
    [[nodiscard]] std::filesystem::path fiberPath(const StoredFiber& fiber) const;
    [[nodiscard]] uint64_t nextFiberId() const;
    [[nodiscard]] uint64_t nextFiberSequenceForUsername(const std::string& username) const;
    [[nodiscard]] std::string currentFiberUsername() const;
    [[nodiscard]] static std::string currentFiberDateTimeString();
    void ensureSessionFiberIdentity(LineAnnotationSession& session);
    [[nodiscard]] static double lineLengthVx(const std::vector<cv::Vec3d>& points);
    [[nodiscard]] static vc::lasagna::LineModel lineModelFromPoints(
        const std::vector<cv::Vec3d>& points,
        const vc::lasagna::NormalSampler* normalSampler);
    [[nodiscard]] static vc::lasagna::LineModel syntheticLineModelFromPoints(
        const std::vector<cv::Vec3d>& points);
    void saveSessionAsFiber(LineAnnotationSession& session);
    void saveFiber(const StoredFiber& fiber) const;
    [[nodiscard]] std::optional<StoredFiber> loadFiberFile(const std::filesystem::path& path) const;
    void showError(const QString& message) const;
    void cleanupIntersectionInspectionSurfaces();
    void rebuildIntersectionInspection();
    bool updateIntersectionFollowSlice(bool sourceSideFlag,
                                       double linePosition,
                                       const char* reason);
    void toggleIntersectionFollowSlice(bool sourceSideFlag);
    bool handleIntersectionFollowKeyPress(int key, Qt::KeyboardModifiers modifiers);
    bool eventFilter(QObject* watched, QEvent* event) override;
    void refreshIntersectionInspectionAfterEdit(uint64_t editedFiberId,
                                                double oldSourceArclength,
                                                double oldTargetArclength);
    bool acceptIntersectionSameWindingChoice();
    [[nodiscard]] std::shared_ptr<LineAnnotationSession> makeIntersectionLineSession(
        const StoredFiber& fiber,
        double focusLinePosition,
        const cv::Vec3d& sourceSliceNormal,
        const std::string& surfaceName,
        std::function<void()> onOptimizationSucceeded);

    CState* _state = nullptr;
    ViewerManager* _viewerManager = nullptr;
    SurfacePanelController* _surfacePanel = nullptr;
    QPointer<QWidget> _parentWidget;
    int _nextPaneId = 1;
    std::vector<PaneRecord> _panes;
    std::vector<StoredFiber> _fibers;
    std::vector<std::string> _knownFiberTags;
    std::unique_ptr<IntersectionInspectionSession> _intersectionInspection;
    std::unique_ptr<FiberSliceOverlayController> _fiberSliceOverlay;
    DatasetPicker _datasetPicker;
    OptimizationTaskFactory _optimizationTaskFactory;
};
