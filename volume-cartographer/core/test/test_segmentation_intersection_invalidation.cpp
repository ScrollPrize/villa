#include <QSignalSpy>
#include <QTest>

#include "overlays/SegmentationIntersectionInvalidation.hpp"

#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/Surface.hpp"

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace {

class DummySurface final : public Surface {
public:
    void move(cv::Vec3f&, const cv::Vec3f&) override {}
    bool valid(const cv::Vec3f&, const cv::Vec3f& = {0, 0, 0}) override { return true; }
    cv::Vec3f loc(const cv::Vec3f& ptr, const cv::Vec3f& = {0, 0, 0}) override { return ptr; }
    cv::Vec3f coord(const cv::Vec3f& ptr, const cv::Vec3f& = {0, 0, 0}) const override { return ptr; }
    cv::Vec3f normal(const cv::Vec3f&, const cv::Vec3f& = {0, 0, 0}) override { return {0, 0, 1}; }
    float pointTo(cv::Vec3f&, const cv::Vec3f&, float, int, SurfacePatchIndex*, PointIndex*) override { return 0.0f; }
    void gen(cv::Mat_<cv::Vec3f>*, cv::Mat_<cv::Vec3f>*, cv::Size, const cv::Vec3f&, float, const cv::Vec3f&) const override {}
};

class FakeViewer final : public QObject, public VolumeViewerBase {
    Q_OBJECT

public:
    explicit FakeViewer(Surface* surface)
        : surface_(surface)
    {
    }

    QPointF volumeToScene(const cv::Vec3f&) override { return {}; }
    cv::Vec3f sceneToVolume(const QPointF&) const override { return {}; }
    cv::Vec2f sceneToSurfaceCoords(const QPointF&) const override { return {}; }
    QPointF surfaceCoordsToScene(float, float) const override { return {}; }
    QPointF lastScenePosition() const override { return {}; }
    void setLinkedCursorVolumePoint(const std::optional<cv::Vec3f>&) override {}

    void setSurface(const std::string&) override {}
    void setIntersects(const std::set<std::string>&) override {}
    void renderVisible(bool, const char*, std::source_location) override {}
    void requestRender(const char*, std::source_location) override {}
    void invalidateVis() override {}
    void centerOnVolumePoint(const cv::Vec3f&, bool) override {}
    void centerOnSurfacePoint(const cv::Vec2f&, bool) override {}
    void adjustZoomByFactor(float) override {}
    void adjustSurfaceOffset(float) override {}
    void resetSurfaceOffsets() override {}
    void fitSurfaceInView() override {}

    Surface* currentSurface() const override { return surface_; }
    std::string surfName() const override { return "fake"; }
    std::shared_ptr<Volume> currentVolume() const override { return {}; }
    VCCollection* pointCollection() const override { return nullptr; }

    float getCurrentScale() const override { return 1.0f; }
    float dsScale() const override { return 1.0f; }
    float normalOffset() const override { return 0.0f; }
    int datasetScaleIndex() const override { return 0; }
    float datasetScaleFactor() const override { return 1.0f; }

    bool isShowDirectionHints() const override { return false; }
    bool isShowSurfaceNormals() const override { return false; }
    float normalArrowLengthScale() const override { return 1.0f; }
    int normalMaxArrows() const override { return 0; }
    void setNormalArrowLengthScale(float) override {}
    void setNormalMaxArrows(int) override {}

    const CompositeRenderSettings& compositeRenderSettings() const override { return compositeSettings_; }
    bool isCompositeEnabled() const override { return false; }
    bool isPlaneCompositeEnabled() const override { return false; }
    void setCompositeRenderSettings(const CompositeRenderSettings& settings) override { compositeSettings_ = settings; }
    void setVolumeWindow(float, float) override {}
    void setBaseColormap(const std::string&) override {}
    void setResetViewOnSurfaceChange(bool) override {}
    void setPlaneIntersectionLinesVisible(bool) override {}
    void setShowDirectionHints(bool) override {}
    void setShowSurfaceNormals(bool) override {}
    void setSegmentationEditActive(bool) override {}
    void setSegmentationIntersectionDeferral(bool) override {}
    void setSegmentationCursorMirroring(bool) override {}
    void setOverlayVolume(std::shared_ptr<Volume>) override {}
    void setOverlayOpacity(float) override {}
    void setOverlayColormap(const std::string&) override {}
    void setOverlayThreshold(float) override {}
    void setOverlayWindow(float, float) override {}
    void reloadPerfSettings() override {}

    uint64_t highlightedPointId() const override { return 0; }
    uint64_t selectedPointId() const override { return 0; }
    uint64_t selectedCollectionId() const override { return 0; }
    bool isPointDragActive() const override { return false; }
    bool isSameWrapAnnotationModeEnabled() const override { return false; }
    double sameWrapAnnotationPolylineOpacity() const override { return 1.0; }
    const std::vector<ViewerOverlayControllerBase::PathPrimitive>& drawingPaths() const override { return paths_; }

    void setOverlayGroup(const std::string&, const std::vector<QGraphicsItem*>&) override {}
    void clearOverlayGroup(const std::string&) override {}
    void clearAllOverlayGroups() override {}

    std::vector<std::pair<QRectF, QColor>> selections() const override { return {}; }
    std::optional<QRectF> activeBBoxSceneRect() const override { return std::nullopt; }
    void setBBoxMode(bool) override {}
    QuadSurface* makeBBoxFilteredSurfaceFromSceneRect(const QRectF&) override { return nullptr; }
    void clearSelections() override {}

    void renderIntersections(const char*, std::source_location) override
    {
        ++renderCount;
        emit rendered();
    }

    void scheduleIntersectionRender(const char*, std::source_location) override
    {
        ++scheduleRequestCount;
        if (intersectionScheduled_) {
            return;
        }
        intersectionScheduled_ = true;
        ++scheduleStartCount;
        emit scheduleStarted();
    }

    void invalidateIntersect(const std::string& name = "") override
    {
        ++invalidateCount;
        lastInvalidatedName = name;
    }

    float intersectionOpacity() const override { return 1.0f; }
    float intersectionThickness() const override { return 1.0f; }
    int surfacePatchSamplingStride() const override { return 1; }
    void setIntersectionOpacity(float) override {}
    void setIntersectionThickness(float) override {}
    void setHighlightedSurfaceIds(const std::vector<std::string>&) override {}
    void setSurfacePatchSamplingStride(int) override {}

    bool surfaceOverlayEnabled() const override { return false; }
    const std::map<std::string, cv::Vec3b>& surfaceOverlays() const override { return overlays_; }
    float surfaceOverlapThreshold() const override { return 0.0f; }
    void setSurfaceOverlayEnabled(bool) override {}
    void setSurfaceOverlays(const std::map<std::string, cv::Vec3b>& overlays) override { overlays_ = overlays; }
    void setSurfaceOverlapThreshold(float) override {}

    const ActiveSegmentationHandle& activeSegmentationHandle() const override { return activeSegmentation_; }
    CVolumeViewerView* graphicsView() const override { return nullptr; }
    QObject* asQObject() override { return this; }
    QMetaObject::Connection connectOverlaysUpdated(QObject*, const std::function<void()>&) override { return {}; }

    int invalidateCount{0};
    int scheduleRequestCount{0};
    int scheduleStartCount{0};
    int renderCount{0};
    std::string lastInvalidatedName;

signals:
    void scheduleStarted();
    void rendered();

private:
    Surface* surface_{nullptr};
    bool intersectionScheduled_{false};
    CompositeRenderSettings compositeSettings_;
    std::vector<ViewerOverlayControllerBase::PathPrimitive> paths_;
    std::map<std::string, cv::Vec3b> overlays_;
    ActiveSegmentationHandle activeSegmentation_;
};

} // namespace

class SegmentationIntersectionInvalidationTest final : public QObject {
    Q_OBJECT

private slots:
    void repeatedAutoApprovalPlaneInvalidationsAreDebounced()
    {
        PlaneSurface plane;
        FakeViewer viewer(&plane);
        std::vector<VolumeViewerBase*> viewers{&viewer};
        QSignalSpy scheduleStarted(&viewer, &FakeViewer::scheduleStarted);
        QSignalSpy rendered(&viewer, &FakeViewer::rendered);

        vc3d::segmentation::invalidateApprovalPlaneIntersections(
            viewers, vc3d::segmentation::ApprovalIntersectionRefresh::Deferred);
        vc3d::segmentation::invalidateApprovalPlaneIntersections(
            viewers, vc3d::segmentation::ApprovalIntersectionRefresh::Deferred);
        vc3d::segmentation::invalidateApprovalPlaneIntersections(
            viewers, vc3d::segmentation::ApprovalIntersectionRefresh::Deferred);

        QCOMPARE(viewer.invalidateCount, 3);
        QCOMPARE(QString::fromStdString(viewer.lastInvalidatedName), QString("segmentation"));
        QCOMPARE(viewer.scheduleRequestCount, 3);
        QCOMPARE(viewer.scheduleStartCount, 1);
        QTRY_COMPARE(scheduleStarted.count(), 1);
        QCOMPARE(viewer.renderCount, 0);
        QCOMPARE(rendered.count(), 0);
    }

    void manualPlaneInvalidationsRenderSynchronously()
    {
        PlaneSurface plane;
        FakeViewer viewer(&plane);
        std::vector<VolumeViewerBase*> viewers{&viewer};
        QSignalSpy scheduleStarted(&viewer, &FakeViewer::scheduleStarted);
        QSignalSpy rendered(&viewer, &FakeViewer::rendered);

        vc3d::segmentation::invalidateApprovalPlaneIntersections(
            viewers, vc3d::segmentation::ApprovalIntersectionRefresh::Immediate);
        vc3d::segmentation::invalidateApprovalPlaneIntersections(
            viewers, vc3d::segmentation::ApprovalIntersectionRefresh::Immediate);

        QCOMPARE(viewer.invalidateCount, 2);
        QCOMPARE(QString::fromStdString(viewer.lastInvalidatedName), QString("segmentation"));
        QCOMPARE(viewer.scheduleRequestCount, 0);
        QCOMPARE(viewer.scheduleStartCount, 0);
        QCOMPARE(scheduleStarted.count(), 0);
        QCOMPARE(viewer.renderCount, 2);
        QCOMPARE(rendered.count(), 2);
    }

    void nonPlaneViewersAreIgnored()
    {
        DummySurface surface;
        FakeViewer viewer(&surface);
        std::vector<VolumeViewerBase*> viewers{nullptr, &viewer};
        QSignalSpy scheduleStarted(&viewer, &FakeViewer::scheduleStarted);
        QSignalSpy rendered(&viewer, &FakeViewer::rendered);

        vc3d::segmentation::invalidateApprovalPlaneIntersections(
            viewers, vc3d::segmentation::ApprovalIntersectionRefresh::Deferred);

        QCOMPARE(viewer.invalidateCount, 0);
        QCOMPARE(viewer.scheduleRequestCount, 0);
        QCOMPARE(viewer.scheduleStartCount, 0);
        QCOMPARE(viewer.renderCount, 0);
        QCOMPARE(scheduleStarted.count(), 0);
        QCOMPARE(rendered.count(), 0);
    }
};

QTEST_APPLESS_MAIN(SegmentationIntersectionInvalidationTest)

#include "test_segmentation_intersection_invalidation.moc"
