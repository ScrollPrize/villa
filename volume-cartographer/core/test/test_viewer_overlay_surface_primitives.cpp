#include <QGraphicsPathItem>
#include <QGraphicsScene>
#include <QTest>

#include "overlays/AtlasOverlayController.hpp"
#include "overlays/ViewerOverlayControllerBase.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
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

    QPointF volumeToScene(const cv::Vec3f&) override { return {}; }
    cv::Vec3f sceneToVolume(const QPointF&) const override { return {}; }
    cv::Vec2f sceneToSurfaceCoords(const QPointF&) const override { return {}; }
    QPointF surfaceCoordsToScene(float surfX, float surfY) const override
    {
        return {surfX * 10.0 + 1.0, surfY * 10.0 + 2.0};
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

    Surface* currentSurface() const override { return nullptr; }
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
    CompositeRenderSettings compositeSettings_;
    std::vector<ViewerOverlayControllerBase::PathPrimitive> paths_;
    std::map<std::string, cv::Vec3b> surfaceOverlays_;
    ActiveSegmentationHandle activeSegmentation_;
    std::map<std::string, std::vector<QGraphicsItem*>> overlayGroups_;
};

} // namespace

class ViewerOverlaySurfacePrimitivesTest final : public QObject {
    Q_OBJECT

private slots:
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
        mapping.lineAnchors.push_back({0, {}, 1.0, 1.0, 0.0});
        mapping.lineAnchors.push_back({1, {}, 2.0, 1.0, 0.0});
        mapping.lineAnchors.push_back({2, {}, 3.0, 2.0, 0.0});
        mapping.controlAnchors.push_back({0, {}, 5.0, 3.0, 0.0});
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
        QVERIFY(initialBounds->right() < 5.0);

        vc::atlas::Atlas updatedAtlas;
        vc::atlas::FiberMapping updatedMapping;
        updatedMapping.lineAnchors.push_back({0, {}, 1.0, 1.0, 0.0});
        updatedMapping.lineAnchors.push_back({1, {}, 2.0, 2.0, 0.0});
        updatedAtlas.fibers.push_back(std::move(updatedMapping));
        controller.setAtlas(updatedAtlas, surface, range);
        controller.refreshViewer(&viewer);

        QCOMPARE(viewer.scene().items().size(), 1);
    }
};

QTEST_MAIN(ViewerOverlaySurfacePrimitivesTest)
#include "test_viewer_overlay_surface_primitives.moc"
