#include <QFile>
#include <QGraphicsPathItem>
#include <QGraphicsScene>
#include <QIODevice>
#include <QTemporaryDir>
#include <QTest>
#include <QTreeWidget>

#include "AtlasControlPointsDock.hpp"
#include "ViewerManager.hpp"
#include "overlays/AtlasControlPointsOverlayController.hpp"
#include "overlays/AtlasOverlayController.hpp"
#include "overlays/FiberOverlayController.hpp"
#include "overlays/ViewerOverlayControllerBase.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"

#include <map>
#include <cmath>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

struct OffscreenQtPlatformGuard {
    OffscreenQtPlatformGuard()
    {
        if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
            qputenv("QT_QPA_PLATFORM", "offscreen");
        }
    }
};

const OffscreenQtPlatformGuard kOffscreenQtPlatformGuard;

class FakeViewer final : public QObject, public VolumeViewerBase {
    Q_OBJECT

public:
    FakeViewer()
    {
        view_.setScene(&scene_);
    }

    QPointF volumeToScene(const cv::Vec3f& point) override
    {
        return {point[0] * surfaceScale_ + surfaceOffset_.x(),
                point[1] * surfaceScale_ + surfaceOffset_.y()};
    }
    cv::Vec3f sceneToVolume(const QPointF&) const override { return {}; }
    cv::Vec2f sceneToSurfaceCoords(const QPointF&) const override { return {}; }
    QPointF surfaceCoordsToScene(float surfX, float surfY) const override
    {
        return {surfX * surfaceScale_ + surfaceOffset_.x(),
                surfY * surfaceScale_ + surfaceOffset_.y()};
    }
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

    Surface* currentSurface() const override { return surface_.get(); }
    std::string surfName() const override { return surfName_; }
    void setSurfName(std::string name) { surfName_ = std::move(name); }
    void setCurrentSurface(std::shared_ptr<Surface> surface) { surface_ = std::move(surface); }
    void setSurfaceSceneTransform(qreal scale, QPointF offset)
    {
        surfaceScale_ = scale;
        surfaceOffset_ = offset;
    }
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
    void setOverlaySamplingMethod(vc::Sampling) override {}
    void setOverlayThreshold(float) override {}
    void setOverlayWindow(float, float) override {}
    void setOverlayMaxDisplayedResolution(int) override {}
    void setOverlayComposite(const OverlayCompositeSettings&) override {}
    void reloadPerfSettings() override {}

    uint64_t highlightedPointId() const override { return 0; }
    uint64_t selectedPointId() const override { return 0; }
    uint64_t selectedCollectionId() const override { return 0; }
    bool isPointDragActive() const override { return false; }
    bool isSameWrapAnnotationModeEnabled() const override { return false; }
    double sameWrapAnnotationPolylineOpacity() const override { return 1.0; }
    const std::vector<ViewerOverlayControllerBase::PathPrimitive>& drawingPaths() const override { return paths_; }

    void setOverlayGroup(const std::string& key, const std::vector<QGraphicsItem*>& items) override
    {
        clearOverlayGroup(key);
        overlayGroups_[key] = items;
    }

    void clearOverlayGroup(const std::string& key) override
    {
        auto it = overlayGroups_.find(key);
        if (it == overlayGroups_.end()) {
            return;
        }
        for (auto* item : it->second) {
            delete item;
        }
        overlayGroups_.erase(it);
    }

    void clearAllOverlayGroups() override
    {
        auto keys = std::vector<std::string>{};
        for (const auto& [key, _] : overlayGroups_) {
            keys.push_back(key);
        }
        for (const auto& key : keys) {
            clearOverlayGroup(key);
        }
    }

    std::vector<std::pair<QRectF, QColor>> selections() const override { return {}; }
    std::optional<QRectF> activeBBoxSceneRect() const override { return std::nullopt; }
    void setBBoxMode(bool) override {}
    QuadSurface* makeBBoxFilteredSurfaceFromSceneRect(const QRectF&) override { return nullptr; }
    void clearSelections() override {}

    void renderIntersections(const char*, std::source_location) override {}
    void scheduleIntersectionRender(const char*, std::source_location) override {}
    void invalidateIntersect(const std::string& = "") override {}
    float intersectionOpacity() const override { return 1.0f; }
    float intersectionThickness() const override { return 0.0f; }
    int surfacePatchSamplingStride() const override { return 1; }
    void setIntersectionOpacity(float) override {}
    void setIntersectionThickness(float) override {}
    void setHighlightedSurfaceIds(const std::vector<std::string>&) override {}
    void setSurfacePatchSamplingStride(int) override {}

    bool surfaceOverlayEnabled() const override { return false; }
    const std::map<std::string, cv::Vec3b>& surfaceOverlays() const override { return surfaceOverlays_; }
    float surfaceOverlapThreshold() const override { return 0.0f; }
    void setSurfaceOverlayEnabled(bool) override {}
    void setSurfaceOverlays(const std::map<std::string, cv::Vec3b>& overlays) override { surfaceOverlays_ = overlays; }
    void setSurfaceOverlapThreshold(float) override {}

    const ActiveSegmentationHandle& activeSegmentationHandle() const override { return activeSegmentation_; }
    CVolumeViewerView* graphicsView() const override { return const_cast<CVolumeViewerView*>(&view_); }
    QObject* asQObject() override { return this; }
    QMetaObject::Connection connectOverlaysUpdated(QObject*, const std::function<void()>&) override { return {}; }

    const QGraphicsScene& scene() const { return scene_; }

private:
    CVolumeViewerView view_;
    QGraphicsScene scene_;
    std::shared_ptr<Surface> surface_;
    qreal surfaceScale_{10.0};
    QPointF surfaceOffset_{1.0, 2.0};
    CompositeRenderSettings compositeSettings_;
    std::vector<ViewerOverlayControllerBase::PathPrimitive> paths_;
    std::map<std::string, cv::Vec3b> surfaceOverlays_;
    ActiveSegmentationHandle activeSegmentation_;
    std::map<std::string, std::vector<QGraphicsItem*>> overlayGroups_;
    std::string surfName_{"fake"};
};

class ChainTestController final : public ViewerOverlayControllerBase {
public:
    ChainTestController() : ViewerOverlayControllerBase("chain_test") {}
    using ViewerOverlayControllerBase::OverlayBuilder;
    using ViewerOverlayControllerBase::polylineBreakDistance;
    using ViewerOverlayControllerBase::addBrokenLineStrips;
    using ViewerOverlayControllerBase::renderPointChain;

protected:
    void collectPrimitives(VolumeViewerBase*, OverlayBuilder&) override {}
};

} // namespace

class ViewerOverlaySurfacePrimitivesTest final : public QObject {
    Q_OBJECT

private slots:
    void fiberStylesUseDistinctMatchingColors()
    {
        const QColor first = FiberOverlayController::fiberColor(1);
        const QColor second = FiberOverlayController::fiberColor(2);
        QVERIFY(first.isValid());
        QVERIFY(second.isValid());
        QVERIFY(first != second);

        const auto style = FiberOverlayController::fiberStyle(first, 25.0f);
        QCOMPARE(style.color, first);
        QCOMPARE(style.pointBorderColor, first);
        QCOMPARE(style.distanceTolerance, 25.0f);
    }

    void fiberOverlayRetainsGraphicsItemsAcrossRefreshes()
    {
        FakeViewer viewer;
        viewer.graphicsView()->resize(800, 600);
        viewer.setSurfaceSceneTransform(1.0, QPointF{});
        viewer.setCurrentSurface(std::make_shared<PlaneSurface>(
            cv::Vec3f(0.0f, 0.0f, 0.0f), cv::Vec3f(0.0f, 0.0f, 1.0f)));

        FiberOverlayController controller;
        controller.attachViewer(&viewer);
        controller.setChains({
            FiberOverlayController::Chain{1, {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}},
            FiberOverlayController::Chain{2, {{0, 2, 0}, {1, 2, 0}, {2, 2, 0}}},
        });
        controller.setVisible(true);

        const auto initialItems = viewer.scene().items();
        QCOMPARE(initialItems.size(), 2);
        const std::set<QGraphicsItem*> initialSet(initialItems.begin(), initialItems.end());
        const QRectF initialBounds = viewer.scene().itemsBoundingRect();

        controller.refreshViewer(&viewer);
        const auto refreshedItems = viewer.scene().items();
        const std::set<QGraphicsItem*> refreshedSet(refreshedItems.begin(), refreshedItems.end());
        QCOMPARE(refreshedItems.size(), 2);
        QVERIFY(refreshedSet == initialSet);

        // Camera changes update the retained batches in place rather than
        // replacing their QGraphicsItems.
        viewer.setSurfaceSceneTransform(2.0, QPointF(20.0, 30.0));
        controller.refreshViewer(&viewer);
        const auto movedItems = viewer.scene().items();
        const std::set<QGraphicsItem*> movedSet(movedItems.begin(), movedItems.end());
        QCOMPARE(movedItems.size(), 2);
        QVERIFY(movedSet == initialSet);
        QVERIFY(viewer.scene().itemsBoundingRect() != initialBounds);

        controller.setVisible(false);
        QCOMPARE(viewer.scene().items().size(), 0);
    }

    void surfacePrimitivesUseViewerSurfaceTransform()
    {
        FakeViewer viewer;

        ViewerOverlayControllerBase::OverlayStyle style;
        style.penColor = Qt::white;
        style.brushColor = Qt::white;
        style.penWidth = 1.0;

        ViewerOverlayControllerBase::applyPrimitives(&viewer, "surface_test", {
            ViewerOverlayControllerBase::SurfaceLineStripPrimitive{
                {cv::Vec2f(1.0f, 2.0f), cv::Vec2f(3.0f, 4.0f)},
                false,
                style,
            },
            ViewerOverlayControllerBase::SurfacePointPrimitive{
                cv::Vec2f(5.0f, 6.0f),
                2.0,
                style,
            },
        });

        const auto items = viewer.scene().items();
        QCOMPARE(items.size(), 2);

        bool sawLine = false;
        bool sawPoint = false;
        for (auto* item : items) {
            const QRectF rect = item->sceneBoundingRect();
            if (rect.width() > 10.0 && rect.height() > 10.0) {
                sawLine = true;
                QVERIFY(std::abs(rect.center().x() - 21.0) < 0.75);
                QVERIFY(std::abs(rect.center().y() - 32.0) < 0.75);
            } else {
                sawPoint = true;
                QVERIFY(std::abs(rect.center().x() - 51.0) < 0.75);
                QVERIFY(std::abs(rect.center().y() - 62.0) < 0.75);
            }
        }
        QVERIFY(sawLine);
        QVERIFY(sawPoint);
    }

    void atlasOverlayControllerEmitsLineAndAnchorPoints()
    {
        FakeViewer viewer;

        cv::Mat_<cv::Vec3f> points(4, 6);
        points.setTo(cv::Vec3f(0.0f, 0.0f, 0.0f));
        auto surface = std::make_shared<QuadSurface>(points, cv::Vec2f(1.0f, 1.0f));

        vc::atlas::Atlas atlas;
        vc::atlas::FiberMapping mapping;
        mapping.lineAnchors.push_back({0, {}, 40.0, 1.0, 0.0});
        mapping.lineAnchors.push_back({1, {}, 2.0, 1.0, 0.0});
        mapping.lineAnchors.push_back({2, {}, 3.0, 2.0, 0.0});
        mapping.lineAnchors.push_back({3, {}, 50.0, 2.0, 0.0});
        mapping.controlAnchors.push_back({0, {}, 2.0, 1.0, 0.0});
        mapping.controlAnchors.push_back({1, {}, 3.0, 2.0, 0.0});
        atlas.fibers.push_back(std::move(mapping));

        vc::atlas::AtlasDisplayRange range;
        range.baseColumns = points.cols;
        range.unwrapCount = 1;

        AtlasOverlayController controller;
        controller.attachViewer(&viewer);
        controller.setAtlas(atlas, surface, range);

        QCOMPARE(viewer.scene().items().size(), 2);
        const auto initialBounds = controller.surfaceBounds();
        QVERIFY(initialBounds.has_value());
        QVERIFY(std::isfinite(initialBounds->right()));

        vc::atlas::FiberMapping previewMapping;
        previewMapping.lineAnchors.push_back({0, {}, 2.0, 1.0, 0.0});
        previewMapping.lineAnchors.push_back({1, {}, 3.0, 2.0, 0.0});
        previewMapping.controlAnchors.push_back({0, {}, 2.0, 1.0, 0.0});
        vc::atlas::FiberMapping shiftedPreviewMapping = previewMapping;
        shiftedPreviewMapping.windingOffset = 1;

        controller.setSearchPreviewCandidates({
            AtlasOverlayController::SearchPreviewCandidate{0, cv::Vec2f(2.5f, 1.5f)},
            AtlasOverlayController::SearchPreviewCandidate{1, cv::Vec2f(3.5f, 2.5f)},
        });
        controller.setSearchPreviewHover(0);
        controller.setSearchPreviewSelection({1});
        controller.setSearchPreviewFiber({0, previewMapping});
        controller.setSearchPreviewFiber({1, shiftedPreviewMapping});

        // Saved atlas line + saved control point group + two cross markers
        // (two line strips each) + two line-only preview mappings.
        QCOMPARE(viewer.scene().items().size(), 8);

        controller.clearSearchPreviews();
        QCOMPARE(viewer.scene().items().size(), 2);

        vc::atlas::Atlas updatedAtlas;
        vc::atlas::FiberMapping updatedMapping;
        updatedMapping.lineAnchors.push_back({0, {}, 1.0, 1.0, 0.0});
        updatedMapping.lineAnchors.push_back({1, {}, 2.0, 2.0, 0.0});
        updatedAtlas.fibers.push_back(std::move(updatedMapping));
        controller.setAtlas(updatedAtlas, surface, range);
        controller.refreshViewer(&viewer);

        QCOMPARE(viewer.scene().items().size(), 1);
    }

    void atlasControlPointsOverlayEmitsLinePointsAndSelection()
    {
        FakeViewer viewer;
        viewer.setSurfName("segmentation");
        cv::Mat_<cv::Vec3f> points(4, 6);
        points.setTo(cv::Vec3f(0.0f, 0.0f, 0.0f));
        viewer.setCurrentSurface(std::make_shared<QuadSurface>(points, cv::Vec2f(2.0f, 3.0f)));

        AtlasControlPointsOverlayController controller;
        controller.attachViewer(&viewer);

        AtlasControlPointResult a;
        a.fiberId = QStringLiteral("fiber_a");
        a.sourceIndex = 0;
        a.controlIndex = 0;
        a.valid = true;
        a.modelH = 3.0f;
        a.modelW = 4.0f;
        a.snapValid = true;
        a.snapTargetXyz = cv::Vec3f(9.0f, 8.0f, 7.0f);
        a.snapModelH = 3.0f;
        a.snapModelW = 5.0f;

        AtlasControlPointResult b = a;
        b.sourceIndex = 1;
        b.controlIndex = 1;
        b.modelH = 3.0f;
        b.modelW = 5.0f;
        b.snapTargetXyz = cv::Vec3f(10.0f, 8.0f, 7.0f);
        b.snapModelW = 6.0f;

        AtlasControlPointResult invalid = a;
        invalid.controlIndex = 2;
        invalid.valid = false;
        invalid.snapValid = false;

        controller.setResults({a, invalid});
        QCOMPARE(viewer.scene().items().size(), 0);

        controller.setOverlayEnabled(true);
        QCOMPARE(viewer.scene().items().size(), 1);
        const QRectF pointBounds = viewer.scene().items().front()->sceneBoundingRect();
        QCOMPARE(pointBounds.center().x(), 11.0);
        QVERIFY(std::abs(pointBounds.center().y() - (2.0 + (1.0 / 3.0) * 10.0)) < 1.0e-5);

        controller.setResults({a, b, invalid});
        controller.setOverlayEnabled(true);
        QVERIFY(viewer.scene().items().size() >= 2);

        controller.setSelectedPoint(QStringLiteral("fiber_a"), 1);
        QVERIFY(viewer.scene().items().size() >= 2);
    }

    void atlasControlPointsOverlayShowsSnapTargetsOnPlaneViewers()
    {
        FakeViewer viewer;
        viewer.setSurfName("xy plane");

        AtlasControlPointsOverlayController controller;
        controller.attachViewer(&viewer);

        AtlasControlPointResult point;
        point.fiberId = QStringLiteral("fiber_a");
        point.controlIndex = 0;
        point.valid = true;
        point.snapValid = true;
        point.snapTargetXyz = cv::Vec3f(9.0f, 8.0f, 7.0f);
        point.modelH = 3.0f;
        point.modelW = 4.0f;

        controller.setResults({point});
        controller.setOverlayEnabled(true);
        QCOMPARE(viewer.scene().items().size(), 1);

        const QRectF bounds = viewer.scene().items().front()->sceneBoundingRect();
        QVERIFY(std::abs(bounds.center().x() - 91.0) < 0.75);
        QVERIFY(std::abs(bounds.center().y() - 82.0) < 0.75);

        controller.setSelectedPoint(QStringLiteral("fiber_a"), 0);
        QCOMPARE(viewer.scene().items().size(), 1);
    }

    void atlasControlPointsDockLoadsGroupedRows()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString path = dir.filePath(QStringLiteral("atlas_control_points_results.json"));
        QFile file(path);
        QVERIFY(file.open(QIODevice::WriteOnly));
        file.write(R"json(
{
  "format": "lasagna_atlas_control_points_results",
  "version": 1,
  "records": [
    {
      "fiber_id": "fiber_a",
      "object_id": "fiber_a",
      "source_index": 4,
      "control_index": 0,
      "valid": true,
      "distance": 1.25,
      "signed_delta": -0.5,
      "snap_status": "valid fw",
      "snap_valid": true,
      "snap_direction": "fw",
      "snap_signed_delta": 0.25,
      "snap_target_xyz": [9, 8, 7],
      "snap_mesh_xyz": [10, 9, 8],
      "snap_model_h": 11,
      "snap_model_w": 12,
      "target_xyz": [1, 2, 3],
      "mesh_xyz": [2, 3, 4],
      "model_h": 7,
      "model_w": 8
    }
  ]
}
)json");
        file.close();

        AtlasControlPointsDock dock;
        dock.loadResults(std::filesystem::path(path.toStdString()));

        QVERIFY(!dock.overlayChecked());
        QCOMPARE(dock.results().size(), size_t{1});
        auto* tree = dock.findChild<QTreeWidget*>(QStringLiteral("atlasControlResultsTree"));
        QVERIFY(tree != nullptr);
        QCOMPARE(tree->topLevelItemCount(), 1);
        QCOMPARE(tree->topLevelItem(0)->childCount(), 1);
        QCOMPARE(tree->topLevelItem(0)->text(0), QStringLiteral("fiber_a"));
        QCOMPARE(tree->columnCount(), 7);
        QCOMPARE(tree->topLevelItem(0)->child(0)->text(1), QStringLiteral("yes"));
        QCOMPARE(tree->topLevelItem(0)->child(0)->text(2), QStringLiteral("valid fw"));
        QCOMPARE(tree->headerItem()->text(3), QStringLiteral("Snap Delta"));
        QCOMPARE(tree->headerItem()->text(4), QStringLiteral("Snap XYZ"));
        QCOMPARE(tree->topLevelItem(0)->child(0)->text(3), QStringLiteral("0.2500"));
        QCOMPARE(tree->topLevelItem(0)->child(0)->text(4), QStringLiteral("9.0, 8.0, 7.0"));
        QCOMPARE(dock.results().front().snapMeshXyz[0], 10.0f);
        QCOMPARE(dock.results().front().snapMeshXyz[1], 9.0f);
        QCOMPARE(dock.results().front().snapMeshXyz[2], 8.0f);
        QCOMPARE(dock.results().front().snapModelH, 11.0f);
        QCOMPARE(dock.results().front().snapModelW, 12.0f);
    }
    void polylineBreakDistanceUsesMedianSpacing()
    {
        std::vector<cv::Vec3f> positions;
        for (int i = 0; i < 6; ++i) {
            positions.emplace_back(10.0f * i, 0.0f, 0.0f);
        }
        QCOMPARE(ChainTestController::polylineBreakDistance(positions), 40.0f);

        positions.resize(3);
        QVERIFY(std::isinf(ChainTestController::polylineBreakDistance(positions)));
    }

    void brokenLineStripsSplitOnSkippedPointsAndGaps()
    {
        FakeViewer viewer;
        ChainTestController::OverlayBuilder builder(&viewer);

        ViewerOverlayControllerBase::FilteredPoints filtered;
        filtered.volumePoints = {{0, 0, 0}, {10, 0, 0}, {20, 0, 0},
                                 {40, 0, 0}, {50, 0, 0}, {200, 0, 0}};
        // Source index 3 was filtered out, so the strip must break between the
        // third and fourth kept points; the jump to x=200 exceeds the segment
        // limit, so the last point must not be joined either.
        filtered.sourceIndices = {0, 1, 2, 4, 5, 6};
        for (const auto& point : filtered.volumePoints) {
            filtered.scenePoints.emplace_back(point[0], point[1]);
        }

        ViewerOverlayControllerBase::OverlayStyle style;
        ChainTestController::addBrokenLineStrips(builder, filtered, 60.0f, style);

        const auto primitives = builder.takePrimitives();
        QCOMPARE(primitives.size(), std::size_t{2});
        const auto* first =
            std::get_if<ViewerOverlayControllerBase::LineStripPrimitive>(&primitives[0]);
        const auto* second =
            std::get_if<ViewerOverlayControllerBase::LineStripPrimitive>(&primitives[1]);
        QVERIFY(first != nullptr);
        QVERIFY(second != nullptr);
        QCOMPARE(first->points.size(), std::size_t{3});
        QCOMPARE(first->points.back().x(), 20.0);
        QCOMPARE(second->points.size(), std::size_t{2});
        QCOMPARE(second->points.front().x(), 40.0);
        QCOMPARE(second->points.back().x(), 50.0);
    }

    void brokenLineStripsSplitOnSceneSpaceJumps()
    {
        FakeViewer viewer;
        ChainTestController::OverlayBuilder builder(&viewer);

        // Four points evenly spaced in volume space, but the projection of
        // the third lands far away in scene space (an invalid-region / wrap
        // discontinuity): the strip must break around that segment even
        // though every volume-space gap is identical.
        ViewerOverlayControllerBase::FilteredPoints filtered;
        filtered.volumePoints = {{0, 0, 0}, {10, 0, 0}, {20, 0, 0}, {30, 0, 0}};
        filtered.sourceIndices = {0, 1, 2, 3};
        filtered.scenePoints = {{0.0, 0.0}, {10.0, 0.0}, {900.0, 0.0}, {910.0, 0.0}};

        ViewerOverlayControllerBase::OverlayStyle style;
        ChainTestController::addBrokenLineStrips(
            builder, filtered, std::numeric_limits<float>::infinity(), style, 4.0);

        const auto primitives = builder.takePrimitives();
        QCOMPARE(primitives.size(), std::size_t{2});
        const auto* first =
            std::get_if<ViewerOverlayControllerBase::LineStripPrimitive>(&primitives[0]);
        const auto* second =
            std::get_if<ViewerOverlayControllerBase::LineStripPrimitive>(&primitives[1]);
        QVERIFY(first != nullptr);
        QVERIFY(second != nullptr);
        QCOMPARE(first->points.size(), std::size_t{2});
        QCOMPARE(first->points.back().x(), 10.0);
        QCOMPARE(second->points.size(), std::size_t{2});
        QCOMPARE(second->points.front().x(), 900.0);
    }

    void renderPointChainEmitsDotsAndOrderedPolyline()
    {
        FakeViewer viewer;
        viewer.graphicsView()->resize(800, 600);

        ChainTestController controller;
        ChainTestController::OverlayBuilder builder(&viewer);
        const std::vector<cv::Vec3f> chain = {
            {1.0f, 1.0f, 0.0f}, {2.0f, 1.0f, 0.0f}, {3.0f, 1.0f, 0.0f}, {4.0f, 1.0f, 0.0f}};
        ViewerOverlayControllerBase::PointChainStyle style;
        // FakeViewer::volumeToScene applies a 10x mapping while
        // getCurrentScale() reports 1.0, so widen the discontinuity guard
        // beyond that mismatch to keep this a pure connectivity test.
        style.sceneJumpRatio = 20.0f;
        controller.renderPointChain(&viewer, builder, chain, style);

        const auto primitives = builder.takePrimitives();
        std::size_t strips = 0;
        std::size_t dots = 0;
        for (const auto& primitive : primitives) {
            if (const auto* strip =
                    std::get_if<ViewerOverlayControllerBase::LineStripPrimitive>(&primitive)) {
                ++strips;
                QCOMPARE(strip->points.size(), chain.size());
                for (std::size_t i = 1; i < strip->points.size(); ++i) {
                    QVERIFY(strip->points[i - 1].x() < strip->points[i].x());
                }
            } else if (std::holds_alternative<ViewerOverlayControllerBase::PointPrimitive>(primitive)) {
                ++dots;
            }
        }
        QCOMPARE(strips, std::size_t{1});
        QCOMPARE(dots, chain.size());
    }

    void renderPointChainKeepsTopologyStableAcrossCameraChanges()
    {
        FakeViewer viewer;
        viewer.graphicsView()->resize(320, 240);
        auto plane = std::make_shared<PlaneSurface>(cv::Vec3f(0.0f, 0.0f, 0.0f),
                                                    cv::Vec3f(0.0f, 0.0f, 1.0f));
        viewer.setCurrentSurface(plane);

        ChainTestController controller;
        const std::vector<cv::Vec3f> chain = {
            {-100.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {100.0f, 1.0f, 0.0f}};
        ViewerOverlayControllerBase::PointChainStyle style;
        style.distanceTolerance = 4.0f;
        style.sceneJumpRatio = 20.0f;

        ChainTestController::OverlayBuilder initialBuilder(&viewer);
        controller.renderPointChain(&viewer, initialBuilder, chain, style);
        const auto initial = initialBuilder.takePrimitives();
        const auto initialLine = std::find_if(initial.begin(), initial.end(), [](const auto& primitive) {
            return std::holds_alternative<ViewerOverlayControllerBase::LineStripPrimitive>(primitive);
        });
        QVERIFY(initialLine != initial.end());
        const auto* initialStrip =
            std::get_if<ViewerOverlayControllerBase::LineStripPrimitive>(&*initialLine);
        QVERIFY(initialStrip != nullptr);
        QCOMPARE(initialStrip->points.size(), chain.size());
        QCOMPARE(initialStrip->points.front().x(), -999.0);
        QCOMPARE(initialStrip->points.back().x(), 1001.0);

        // A pan/zoom changes only the final surface-to-scene transform. The
        // complete source chain, including off-screen endpoints, must remain.
        viewer.setSurfaceSceneTransform(5.0, QPointF(20.0, 30.0));
        ChainTestController::OverlayBuilder movedBuilder(&viewer);
        controller.renderPointChain(&viewer, movedBuilder, chain, style);
        const auto moved = movedBuilder.takePrimitives();
        const auto movedLine = std::find_if(moved.begin(), moved.end(), [](const auto& primitive) {
            return std::holds_alternative<ViewerOverlayControllerBase::LineStripPrimitive>(primitive);
        });
        QVERIFY(movedLine != moved.end());
        const auto* movedStrip =
            std::get_if<ViewerOverlayControllerBase::LineStripPrimitive>(&*movedLine);
        QVERIFY(movedStrip != nullptr);
        QCOMPARE(movedStrip->points.size(), chain.size());
        QCOMPARE(movedStrip->points.front().x(), -480.0);
        QCOMPARE(movedStrip->points.back().x(), 520.0);

        // Plane movement retains the Surface pointer, so its geometry
        // signature must still invalidate the cached projection.
        plane->setOrigin(cv::Vec3f(0.0f, 0.0f, 100.0f));
        ChainTestController::OverlayBuilder movedPlaneBuilder(&viewer);
        controller.renderPointChain(&viewer, movedPlaneBuilder, chain, style);
        QVERIFY(movedPlaneBuilder.takePrimitives().empty());
    }

    void renderPointChainDoesNotInventControlPointAtQuadBoundary()
    {
        FakeViewer viewer;
        viewer.graphicsView()->resize(800, 600);
        viewer.setSurfaceSceneTransform(1.0, QPointF{});

        cv::Mat_<cv::Vec3f> points(40, 30);
        for (int row = 0; row < points.rows; ++row) {
            for (int col = 0; col < points.cols; ++col) {
                points(row, col) = cv::Vec3f(static_cast<float>(col),
                                              static_cast<float>(row),
                                              0.0f);
            }
        }
        auto surface = std::make_shared<QuadSurface>(points, cv::Vec2f(1.0f, 1.0f));
        viewer.setCurrentSurface(surface);

        ViewerManager manager(nullptr, nullptr);
        manager.surfacePatchIndex()->rebuild({surface}, 0.0f, false);
        ChainTestController controller;
        controller.bindToViewerManager(&manager);
        ChainTestController::OverlayBuilder builder(&viewer);
        // The first control point is coplanar but outside the finite segment.
        // Its nearest mesh point is on the left edge and remains useful as a
        // line endpoint, but it must not become a displayed control point.
        const std::vector<cv::Vec3f> chain = {
            {-2.0f, 20.0f, 0.0f}, {5.0f, 20.0f, 0.0f}};
        ViewerOverlayControllerBase::PointChainStyle style;
        // A wide tolerance used to accept the surface center as a good-enough
        // iterative-search seed, making this artifact distance-dependent.
        style.distanceTolerance = 20.0f;
        style.sceneJumpRatio = 20.0f;
        controller.renderPointChain(&viewer, builder, chain, style);

        const auto primitives = builder.takePrimitives();
        std::size_t strips = 0;
        std::size_t dots = 0;
        for (const auto& primitive : primitives) {
            if (const auto* strip =
                    std::get_if<ViewerOverlayControllerBase::LineStripPrimitive>(&primitive)) {
                ++strips;
                QCOMPARE(strip->points.size(), std::size_t{2});
                QVERIFY(std::fabs(strip->points.front().x() + surface->center()[0]) < 0.1);
            } else if (std::holds_alternative<ViewerOverlayControllerBase::PointPrimitive>(primitive)) {
                ++dots;
            }
        }
        QCOMPARE(strips, std::size_t{1});
        QCOMPARE(dots, std::size_t{1});

        ChainTestController::OverlayBuilder realEdgePointBuilder(&viewer);
        const std::vector<cv::Vec3f> realEdgePointChain = {
            {0.0f, 20.0f, 0.0f}, {5.0f, 20.0f, 2.0f}};
        controller.renderPointChain(&viewer, realEdgePointBuilder, realEdgePointChain, style);
        const auto realEdgePointPrimitives = realEdgePointBuilder.takePrimitives();
        const auto realEdgePointDots = std::count_if(
            realEdgePointPrimitives.begin(), realEdgePointPrimitives.end(), [](const auto& primitive) {
                return std::holds_alternative<ViewerOverlayControllerBase::PointPrimitive>(primitive);
            });
        // A control point genuinely on the segment boundary remains visible.
        QCOMPARE(realEdgePointDots, std::ptrdiff_t{2});
    }

    void renderPointChainCullsPointsOutsideVolumeBounds()
    {
        FakeViewer viewer;
        viewer.graphicsView()->resize(800, 600);

        ChainTestController controller;
        ChainTestController::OverlayBuilder builder(&viewer);
        const std::vector<cv::Vec3f> chain = {
            {1.0f, 1.0f, 0.0f}, {2.0f, 1.0f, 0.0f}, {3.0f, 1.0f, 0.0f},
            {4.0f, 1.0f, 0.0f}, {5.0f, 1.0f, 0.0f}};
        ViewerOverlayControllerBase::PointChainStyle style;
        style.sceneJumpRatio = 20.0f;
        // The box excludes the middle point, so its dot must disappear and
        // the polyline must break rather than bridge the culled point.
        ViewerOverlayControllerBase::VolumeBounds bounds;
        bounds.lo = {0.0f, 0.0f, -1.0f};
        bounds.hi = {10.0f, 10.0f, 1.0f};
        controller.renderPointChain(&viewer, builder, chain, style, bounds);
        const auto allPrimitives = builder.takePrimitives();
        QCOMPARE(allPrimitives.size(), std::size_t{1 + chain.size()});

        ChainTestController::OverlayBuilder culledBuilder(&viewer);
        std::vector<cv::Vec3f> gappedChain = chain;
        gappedChain[2][2] = 5.0f;  // move the middle point outside the z-bounds
        controller.renderPointChain(&viewer, culledBuilder, gappedChain, style, bounds);
        const auto primitives = culledBuilder.takePrimitives();
        std::size_t strips = 0;
        std::size_t dots = 0;
        for (const auto& primitive : primitives) {
            if (const auto* strip =
                    std::get_if<ViewerOverlayControllerBase::LineStripPrimitive>(&primitive)) {
                ++strips;
                QCOMPARE(strip->points.size(), std::size_t{2});
            } else if (std::holds_alternative<ViewerOverlayControllerBase::PointPrimitive>(primitive)) {
                ++dots;
            }
        }
        QCOMPARE(strips, std::size_t{2});
        QCOMPARE(dots, std::size_t{4});
    }
};

QTEST_MAIN(ViewerOverlaySurfacePrimitivesTest)
#include "test_viewer_overlay_surface_primitives.moc"
