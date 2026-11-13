// The live ink preview's controller (apps/VC3D/overlays/InkPreviewOverlayController):
// the canvas rectangle of a known camera, the placement of a reply after a pan, the
// requests it does not start, the key, and one round trip against an in-test server
// that checks the layers VC3D samples from a synthetic volume.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include "CState.hpp"
#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "overlays/InkDetectionOverlayController.hpp"
#include "overlays/InkPreviewOverlayController.hpp"
#include "overlays/ViewerOverlayControllerBase.hpp"
#include "support/LoopbackHttpServer.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"
#include "viewer_controls/panels/ViewerInkDetectionPanel.hpp"

#include <nlohmann/json.hpp>
#include <vc_reader/zarr_write.hpp>

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QSettings>
#include <QSlider>
#include <QElapsedTimer>
#include <QFrame>
#include <QGraphicsItem>
#include <QGraphicsScene>
#include <QThread>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <variant>
#include <vector>

namespace fs = std::filesystem;

namespace {

// A segmentation viewer whose camera maps surface coordinates linearly to the scene:
// scene = surface * scale + offset, over a 200 x 100 view of scene (0, 0) to (200, 100).
class FakeViewer final : public QObject, public VolumeViewerBase {
public:
    FakeViewer()
    {
        view_.setScene(&scene_);
        view_.setFrameShape(QFrame::NoFrame);
        view_.setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        view_.setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        view_.setAlignment(Qt::AlignLeft | Qt::AlignTop);
        view_.setSceneRect(0, 0, 200, 100);
        view_.resize(200, 100);
        view_.show();
        QApplication::processEvents();
    }

    void setCamera(qreal scale, QPointF offset)
    {
        scale_ = scale;
        offset_ = offset;
    }

    std::optional<SurfaceProjection> projectVolumePoint(const cv::Vec3f&, float) const override { return std::nullopt; }
    QPointF surfaceProjectionToScene(const SurfaceProjection& p) const override { return surfaceCoordsToScene(p.u, p.v); }
    SurfaceProjectionContext surfaceProjectionContext() const override { return {}; }
    QPointF volumeToScene(const cv::Vec3f&) override { return {}; }
    cv::Vec3f sceneToVolume(const QPointF&) const override { return {}; }
    cv::Vec2f sceneToSurfaceCoords(const QPointF& scene) const override
    {
        return {float((scene.x() - offset_.x()) / scale_), float((scene.y() - offset_.y()) / scale_)};
    }
    QPointF surfaceCoordsToScene(float surfX, float surfY) const override
    {
        return {surfX * scale_ + offset_.x(), surfY * scale_ + offset_.y()};
    }
    QPointF lastScenePosition() const override { return {}; }
    void setLinkedCursorVolumePoint(const std::optional<cv::Vec3f>&) override {}

    void setSurface(const std::string&) override {}
    void setIntersects(const std::set<std::string>&) override {}
    void renderVisible(const char*, std::source_location) override {}
    void requestRender(const char*, std::source_location) override {}
    void invalidateVis() override {}
    void centerOnVolumePoint(const cv::Vec3f&) override {}
    void centerOnSurfacePoint(const cv::Vec2f&) override {}
    void adjustZoomByFactor(float) override {}
    void adjustSurfaceOffset(float) override {}
    void resetSurfaceOffsets() override {}
    void fitSurfaceInView() override {}

    Surface* currentSurface() const override { return surface_.get(); }
    std::string surfName() const override { return surfName_; }
    void setSurfName(std::string name) { surfName_ = std::move(name); }
    void setCurrentSurface(std::shared_ptr<Surface> surface) { surface_ = std::move(surface); }
    std::shared_ptr<Volume> currentVolume() const override { return volume_; }
    void setCurrentVolume(std::shared_ptr<Volume> volume) { volume_ = std::move(volume); }
    VCCollection* pointCollection() const override { return nullptr; }

    float getCurrentScale() const override { return float(scale_); }
    float normalOffset() const override { return 0.0f; }

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
    void setResetViewOnSurfaceChange(bool) override {}
    void setPlaneIntersectionLinesVisible(bool) override {}
    void setShowSurfaceNormals(bool) override {}
    void setSegmentationCursorMirroring(bool) override {}
    void setOverlayVolume(std::shared_ptr<Volume>) override {}
    void setOverlayOpacity(float) override {}
    void setOverlayColormap(const std::string&) override {}
    void setOverlaySamplingMethod(int) override {}
    void setOverlayIntensityAsAlpha(bool) override {}
    void setOverlayWindow(float, float) override {}
    void setOverlayMaxDisplayedResolution(int) override {}
    void setOverlayComposite(const OverlayCompositeSettings&) override {}
    void reloadPerfSettings() override {}

    std::optional<vc::PointRef> highlightedPoint() const override { return std::nullopt; }
    std::optional<vc::PointRef> selectedPoint() const override { return std::nullopt; }
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

    std::vector<std::pair<QRectF, QColor>> selections() const override { return {}; }
    std::optional<QRectF> activeBBoxSceneRect() const override { return std::nullopt; }
    void setBBoxMode(bool) override {}
    QuadSurface* makeBBoxFilteredSurfaceFromSceneRect(const QRectF&) override { return nullptr; }
    void clearSelections() override {}

    const std::set<std::string>& intersects() const override { return intersects_; }
    bool planeIntersectionLinesVisible() const override { return true; }
    float intersectionOpacity() const override { return 1.0f; }
    float intersectionThickness() const override { return 0.0f; }
    void setIntersectionOpacity(float) override {}
    void setIntersectionThickness(float) override {}

    bool surfaceOverlayEnabled() const override { return false; }
    const std::map<std::string, cv::Vec3b>& surfaceOverlays() const override { return surfaceOverlays_; }
    std::uint64_t surfaceOverlaysRevision() const override { return 0; }
    float surfaceOverlapThreshold() const override { return 0.0f; }
    void setSurfaceOverlayEnabled(bool) override {}
    void setSurfaceOverlays(const std::map<std::string, cv::Vec3b>&) override {}

    CVolumeViewerView* graphicsView() const override { return const_cast<CVolumeViewerView*>(&view_); }
    QObject* asQObject() override { return this; }
    QMetaObject::Connection connectOverlaysUpdated(QObject*, const std::function<void()>&) override { return {}; }

private:
    QGraphicsScene scene_;
    CVolumeViewerView view_;
    std::shared_ptr<Surface> surface_;
    std::shared_ptr<Volume> volume_;
    qreal scale_{2.0};
    QPointF offset_{0.0, 0.0};
    CompositeRenderSettings compositeSettings_;
    std::vector<ViewerOverlayControllerBase::PathPrimitive> paths_;
    std::map<std::string, cv::Vec3b> surfaceOverlays_;
    std::map<std::string, std::vector<QGraphicsItem*>> overlayGroups_;
    std::string surfName_{"segmentation"};
    std::set<std::string> intersects_;
};

class ProbePreviewController final : public InkPreviewOverlayController {
public:
    using InkPreviewOverlayController::InkPreviewOverlayController;

    bool enabledFor(VolumeViewerBase* viewer) const { return isOverlayEnabledFor(viewer); }

    std::vector<ViewerOverlayControllerBase::ImagePrimitive> images(VolumeViewerBase* viewer)
    {
        OverlayBuilder builder(viewer);
        collectPrimitives(viewer, builder);
        std::vector<ImagePrimitive> out;
        for (auto& primitive : builder.takePrimitives()) {
            if (auto* image = std::get_if<ImagePrimitive>(&primitive)) {
                out.push_back(*image);
            }
        }
        return out;
    }
};

bool waitFor(const std::function<bool()>& done, int ms = 10000)
{
    QElapsedTimer timer;
    timer.start();
    while (!done() && timer.elapsed() < ms) {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
        QThread::msleep(1);
    }
    return done();
}

constexpr int kDepth = 5;
constexpr int kVolumeZ = 32;
constexpr int kVolumeY = 128;
constexpr int kVolumeX = 224;
constexpr int kSurfaceZ = 16;

// A uint8 volume whose voxel at depth z reads 10 z, so a layer's value names its depth.
struct VolumeFixture {
    fs::path root;
    std::shared_ptr<Volume> volume;

    VolumeFixture()
    {
        std::mt19937_64 rng(std::random_device{}());
        root = fs::temp_directory_path() / ("vc_ink_preview_overlay_" + std::to_string(rng()));
        const fs::path zarr = root / "ct.zarr";
        const vcr::zarr::CreateOptions options{
            .shape = {kVolumeZ, kVolumeY, kVolumeX}, .chunks = {16, 64, 64}, .compressor = "", .overwrite = true};
        auto writer = vcr::zarr::Writer::create(zarr / "0", options);
        std::vector<std::uint8_t> voxels(std::size_t(kVolumeZ) * kVolumeY * kVolumeX);
        for (int z = 0; z < kVolumeZ; ++z) {
            std::fill_n(voxels.begin() + std::ptrdiff_t(z) * kVolumeY * kVolumeX, kVolumeY * kVolumeX,
                        std::uint8_t(10 * z));
        }
        writer.writeRegion({0, 0, 0}, {kVolumeZ, kVolumeY, kVolumeX}, voxels.data());
        std::ofstream(zarr / "meta.json") << nlohmann::json{{"type", "vol"}, {"uuid", "preview-volume"},
                                                            {"name", "preview-volume"}, {"format", "zarr"},
                                                            {"width", kVolumeX}, {"height", kVolumeY},
                                                            {"slices", kVolumeZ}, {"voxelsize", 9.0}}
                                                 .dump();
        volume = Volume::New(zarr);
    }
    ~VolumeFixture()
    {
        std::error_code ec;
        fs::remove_all(root, ec);
    }
};

// A flat 21 x 11 grid at scale 0.1 (10 voxels per grid step) at depth kSurfaceZ: its
// canvas at level 0 is 210 x 110 pixels, one pixel per surface unit.
std::shared_ptr<QuadSurface> makeSurface(const std::string& id)
{
    cv::Mat_<cv::Vec3f> points(11, 21);
    for (int r = 0; r < points.rows; ++r) {
        for (int c = 0; c < points.cols; ++c) {
            points(r, c) = cv::Vec3f(2.0f + 10.0f * c, 2.0f + 10.0f * r, float(kSurfaceZ));
        }
    }
    auto surface = std::make_shared<QuadSurface>(points, cv::Vec2f(0.1f, 0.1f));
    surface->id = id;
    return surface;
}

InkPreviewClient::Health testHealth(qint64 maxViewPx = 16777216)
{
    InkPreviewClient::Health health;
    health.model = QStringLiteral("test-model");
    health.backend = QStringLiteral("fake");
    health.patch = 64;
    health.stride = 32;
    health.depth = kDepth;
    health.voxelUm = 9.0;
    health.maxViewPx = maxViewPx;
    return health;
}

struct Scene {
    VolumeFixture data;
    CState state;
    FakeViewer viewer;
    std::shared_ptr<QuadSurface> surface = makeSurface("seg-1");

    Scene()
    {
        viewer.setCurrentSurface(surface);
        viewer.setCurrentVolume(data.volume);
        state.setSurface("segmentation", surface, false, false);
    }
};

} // namespace

TEST_CASE("the visible canvas rectangle of a known camera")
{
    Scene scene;
    ProbePreviewController preview(&scene.state);
    preview.setHealthForTesting(testHealth());
    preview.setLevelsForTesting("preview-volume", {{0, 1.0f, 9.0}});

    const auto canvas = preview.canvasForTesting(&scene.viewer);
    REQUIRE(canvas.has_value());
    CHECK(canvas->size == cv::Size(210, 110));
    // Scene (0, 0)..(200, 100) at 2 scene px per surface unit is surface (0, 0)..(100, 50);
    // surface (0, 0) is grid (10.5, 5.5), canvas (105, 55).
    const auto rect = preview.visibleRectForTesting(&scene.viewer);
    REQUIRE(rect.has_value());
    CHECK(*rect == QRect(105, 55, 100, 50));
    // Half a pixel more on each side touches one more pixel there.
    scene.viewer.setCamera(2.0, QPointF(1.0, 1.0));
    CHECK(*preview.visibleRectForTesting(&scene.viewer) == QRect(104, 54, 101, 51));

    // A view half off the surface is clipped to the canvas.
    scene.viewer.setCamera(1.0, QPointF(-60.0, -30.0));
    const auto clipped = preview.visibleRectForTesting(&scene.viewer);
    REQUIRE(clipped.has_value());
    CHECK(*clipped == QRect(165, 85, 45, 25));
}

TEST_CASE("the shown reply stays on its canvas rectangle when the view pans")
{
    Scene scene;
    ProbePreviewController preview(&scene.state);
    preview.setHealthForTesting(testHealth());
    preview.setLevelsForTesting("preview-volume", {{0, 1.0f, 9.0}});
    preview.setEnabled(true);
    REQUIRE(preview.enabledFor(&scene.viewer));

    cv::Mat_<uint8_t> pixels(50, 100, uint8_t(200));
    preview.showForTesting(&scene.viewer, QRect(105, 55, 100, 50), pixels);
    auto images = preview.images(&scene.viewer);
    REQUIRE(images.size() == 1);
    CHECK(images[0].offset.x() == doctest::Approx(0.0));
    CHECK(images[0].offset.y() == doctest::Approx(0.0));
    CHECK(images[0].transform.m11() == doctest::Approx(2.0));
    CHECK(images[0].transform.m22() == doctest::Approx(2.0));
    CHECK(images[0].opacity == doctest::Approx(0.7));
    CHECK(images[0].image.size() == QSize(100, 50));
    // The fixed mapping: byte 200 draws gray 200 at alpha 200.
    CHECK(qRed(images[0].image.pixel(0, 0)) == 200);
    CHECK(qAlpha(images[0].image.pixel(0, 0)) == 200);

    // Pan by (-40, -20) scene px: the image moves with the surface.
    scene.viewer.setCamera(2.0, QPointF(-40.0, -20.0));
    images = preview.images(&scene.viewer);
    REQUIRE(images.size() == 1);
    CHECK(images[0].offset.x() == doctest::Approx(-40.0));
    CHECK(images[0].offset.y() == doctest::Approx(-20.0));

    // Zoom out by 2: the image keeps its surface position at half the scale.
    scene.viewer.setCamera(1.0, QPointF(0.0, 0.0));
    images = preview.images(&scene.viewer);
    REQUIRE(images.size() == 1);
    CHECK(images[0].offset.x() == doctest::Approx(0.0));
    CHECK(images[0].transform.m11() == doctest::Approx(1.0));

    // The reversed view does not show a forward reply.
    preview.setReversed(true);
    CHECK(preview.images(&scene.viewer).empty());
}

TEST_CASE("no request when the preview is off, outside the segmentation view, or over the view budget")
{
    Scene scene;
    ProbePreviewController preview(&scene.state);
    preview.setHealthForTesting(testHealth());
    preview.setLevelsForTesting("preview-volume", {{0, 1.0f, 9.0}});

    // Off by default: nothing is drawn, nothing starts, nothing waits.
    CHECK(!preview.enabled());
    CHECK(!preview.enabledFor(&scene.viewer));
    CHECK(preview.images(&scene.viewer).empty());
    preview.settleForTesting();
    CHECK(preview.requestsStartedForTesting() == 0);
    CHECK(!preview.settlePendingForTesting());

    // Only the segmentation view.
    preview.setEnabled(true);
    scene.viewer.setSurfName("xy plane");
    CHECK(!preview.enabledFor(&scene.viewer));
    scene.viewer.setSurfName("segmentation");
    CHECK(preview.enabledFor(&scene.viewer));

    // A server that computes at most 1000 px: the 100 x 50 view is over the budget.
    preview.setHealthForTesting(testHealth(1000));
    preview.images(&scene.viewer);
    preview.settleForTesting();
    CHECK(preview.requestsStartedForTesting() == 0);
    CHECK(preview.statusText().startsWith(QStringLiteral("Zoom in for the live preview")));
    CHECK(preview.statusIsError() == false);

    // Opacity 0 (the Space key) starts nothing either.
    preview.setHealthForTesting(testHealth());
    preview.toggleVisibility();
    CHECK(preview.opacity() == 0);
    CHECK(!preview.enabledFor(&scene.viewer));
    preview.settleForTesting();
    CHECK(preview.requestsStartedForTesting() == 0);
    preview.toggleVisibility();
    CHECK(preview.opacity() == 70);
}

TEST_CASE("the key names the surface, its edits, the volume, the level and the layers")
{
    Scene scene;
    ProbePreviewController preview(&scene.state);
    preview.setHealthForTesting(testHealth());
    preview.setLevelsForTesting("preview-volume", {{0, 1.0f, 9.0}, {1, 0.5f, 18.0}});

    const QString key = preview.keyForTesting(&scene.viewer);
    REQUIRE(!key.isEmpty());
    CHECK(key.size() <= 200);
    CHECK(key.contains(QStringLiteral(":s=seg-1:")));
    CHECK(key.contains(QStringLiteral(":v=preview-volume:")));
    CHECK(key.contains(QStringLiteral(":l=0:")));
    CHECK(key.contains(QStringLiteral(":o=-2..2")));

    // An edit of the segmentation surface changes the key; the same state again does not.
    CHECK(preview.keyForTesting(&scene.viewer) == key);
    scene.state.setSurface("segmentation", scene.surface, false, true);
    const QString edited = preview.keyForTesting(&scene.viewer);
    CHECK(edited != key);
    // Another surface's change does not.
    scene.state.setSurface("other", makeSurface("seg-2"), false, false);
    CHECK(preview.keyForTesting(&scene.viewer) == edited);

    // Auto picks the level nearest the model's voxel size; the key and canvas follow it.
    auto coarse = testHealth();
    coarse.voxelUm = 17.0;
    preview.setHealthForTesting(coarse);
    CHECK(preview.resolvedLevel() == 1);
    CHECK(preview.keyForTesting(&scene.viewer).contains(QStringLiteral(":l=1:")));
    CHECK(preview.canvasForTesting(&scene.viewer)->size == cv::Size(105, 55));
    // Unknown voxel size: level 0.
    coarse.voxelUm.reset();
    preview.setHealthForTesting(coarse);
    CHECK(preview.resolvedLevel() == 0);
    // A chosen level wins.
    preview.setLevelChoice(1);
    CHECK(preview.resolvedLevel() == 1);
    preview.setLevelChoice(InkPreviewOverlayController::kAutoLevel);

    // An even depth: offsets -D/2 .. D/2 - 1.
    auto even = testHealth();
    even.depth = 4;
    preview.setHealthForTesting(even);
    CHECK(preview.keyForTesting(&scene.viewer).contains(QStringLiteral(":o=-2..1")));
}

TEST_CASE("a round trip samples the layers the server asks for and draws its reply")
{
    using vc::test::LoopbackHttpServer;
    Scene scene;

    std::mutex mutex;
    std::string uploadError = "no upload";
    std::vector<int> centerColumn;  // the uploaded layers at the need's center pixel
    LoopbackHttpServer server;
    server.setHandler([&](const LoopbackHttpServer::Request& request) -> std::optional<LoopbackHttpServer::Response> {
        LoopbackHttpServer::Response response;
        if (request.path == "/health") {
            response.body = R"({"service": "ink-preview", "api": 1, "backend": "fake", "model": "test-model",)"
                            R"( "patch": 64, "stride": 32, "depth": 5, "voxel_um": 9.0, "max_view_px": 16777216})";
            return response;
        }
        const int w = std::stoi(request.query.at("w"));
        const int h = std::stoi(request.query.at("h"));
        if (request.body.empty()) {
            response.status = 409;
            response.body = nlohmann::json{{"need", {{"x", std::stoi(request.query.at("x"))},
                                                     {"y", std::stoi(request.query.at("y"))},
                                                     {"w", w}, {"h", h}}},
                                           {"depth", kDepth}, {"missing", 1}}
                                .dump();
            return response;
        }
        {
            std::lock_guard lock(mutex);
            const int sw = std::stoi(request.query.at("sw"));
            const int sh = std::stoi(request.query.at("sh"));
            if (request.query.at("dtype") != "uint8" || request.body.size() != std::size_t(kDepth) * sw * sh) {
                uploadError = "upload of " + std::to_string(request.body.size()) + " bytes";
            } else {
                uploadError.clear();
                // The pixel of canvas (150, 80) inside the uploaded rectangle.
                const int x = 150 - std::stoi(request.query.at("sx"));
                const int y = 80 - std::stoi(request.query.at("sy"));
                for (int layer = 0; layer < kDepth; ++layer) {
                    centerColumn.push_back(
                        std::uint8_t(request.body[std::size_t(layer) * sw * sh + std::size_t(y) * sw + x]));
                }
            }
        }
        response.body.assign(std::size_t(w) * h, char(128));
        response.headers = {{"X-Shape", std::to_string(h) + "," + std::to_string(w)},
                            {"X-Backend", "fake"},
                            {"X-Tiles", "4,0"}};
        return response;
    });

    ProbePreviewController preview(&scene.state);
    preview.clientForTesting()->setServerUrl(QUrl(QString::fromStdString(server.url())));
    preview.setEnabled(true);
    REQUIRE(preview.enabledFor(&scene.viewer));
    REQUIRE(waitFor([&] { return preview.health().has_value(); }));

    // The view is noticed by the overlay rebuild; the settle timer starts the work: the
    // volume's levels are read first, then the request goes out.
    preview.images(&scene.viewer);
    REQUIRE(waitFor([&] { return preview.requestsStartedForTesting() == 1; }));
    CHECK(preview.levels().size() == 1);
    CHECK(preview.resolvedLevel() == 0);
    // The view (105, 55, 100, 50) plus a 64 px margin, clipped to the 210 x 110 canvas.
    CHECK(preview.requestedRectForTesting() == QRect(41, 0, 169, 110));

    REQUIRE_MESSAGE(waitFor([&] { return preview.statusText().startsWith(QStringLiteral("Model output")); }),
                    preview.statusText().toStdString());
    CHECK(!preview.statusIsError());
    {
        std::lock_guard lock(mutex);
        CHECK(uploadError.empty());
        // Layer i sits i - 2 voxels along the normal from depth 16: values 10 z.
        CHECK(centerColumn == std::vector<int>{140, 150, 160, 170, 180});
    }

    const auto images = preview.images(&scene.viewer);
    REQUIRE(images.size() == 1);
    CHECK(images[0].image.size() == QSize(169, 110));
    // Canvas (41, 0) is surface (-64, -55): scene (-128, -110).
    CHECK(images[0].offset.x() == doctest::Approx(-128.0));
    CHECK(images[0].offset.y() == doctest::Approx(-110.0));

    // A pan inside the held image asks nothing new.
    preview.settleForTesting();
    CHECK(preview.requestsStartedForTesting() == 1);
    CHECK(!preview.settlePendingForTesting());
    scene.viewer.setCamera(2.0, QPointF(-20.0, 0.0));
    preview.images(&scene.viewer);
    CHECK(preview.settlePendingForTesting());
    preview.settleForTesting();
    CHECK(preview.requestsStartedForTesting() == 1);

    // Turning the preview off drops the image and the status.
    preview.setEnabled(false);
    CHECK(preview.images(&scene.viewer).empty());
    CHECK(preview.statusText().isEmpty());
}

TEST_CASE("the Ink Detection dock's live preview group drives the controller")
{
    VolumeFixture data;
    const fs::path config = data.root / "config";
    fs::create_directories(config);
    qputenv("VC3D_CONFIG_DIR", QByteArray::fromStdString(config.string()));

    CState state;
    InkDetectionOverlayController detections(&state);
    ProbePreviewController preview(&state);
    preview.setHealthForTesting(testHealth());
    ViewerManager manager(&state, nullptr);
    manager.setInkDetectionOverlay(&detections);
    manager.setInkPreviewOverlay(&preview);
    ViewerInkDetectionPanel panel(&manager);

    auto* group = panel.findChild<QGroupBox*>();
    REQUIRE(group != nullptr);
    CHECK(group->title() == QStringLiteral("Live preview"));
    QCheckBox* enable = nullptr;
    QCheckBox* reversed = nullptr;
    for (auto* box : group->findChildren<QCheckBox*>()) {
        (box->text() == QStringLiteral("Reverse layers") ? reversed : enable) = box;
    }
    REQUIRE(enable != nullptr);
    REQUIRE(reversed != nullptr);
    CHECK(reversed->toolTip().contains(QStringLiteral("control view")));
    auto* server = group->findChild<QLineEdit*>();
    REQUIRE(server != nullptr);
    auto* levels = group->findChild<QComboBox*>();
    REQUIRE(levels != nullptr);
    QSlider* opacity = panel.findChild<QSlider*>();
    REQUIRE(opacity != nullptr);
    const auto combos = panel.findChildren<QComboBox*>();
    REQUIRE(combos.size() == 3);  // detection, colormap, preview level
    QComboBox* detection = combos[0];
    QComboBox* colormap = combos[1];

    // Off by default: the rows the published detections use are as without the preview.
    CHECK(!enable->isChecked());
    CHECK(!preview.enabled());
    CHECK(group->isEnabled());
    CHECK(server->text() == QStringLiteral("http://127.0.0.1:8765"));
    CHECK(preview.serverUrl() == QUrl(QStringLiteral("http://127.0.0.1:8765")));
    CHECK(!detection->isEnabled());  // no published detections in this project
    CHECK(!colormap->isEnabled());
    CHECK(!opacity->isEnabled());
    CHECK(levels->count() == 1);
    CHECK(levels->itemText(0) == QStringLiteral("Auto"));

    // On: the shared opacity and colormap rows serve the preview.
    enable->setChecked(true);
    CHECK(preview.enabled());
    CHECK(opacity->isEnabled());
    CHECK(colormap->isEnabled());
    CHECK(!detection->isEnabled());
    opacity->setValue(40);
    CHECK(preview.opacity() == 40);
    preview.toggleVisibility();
    CHECK(opacity->value() == 0);
    preview.toggleVisibility();
    CHECK(opacity->value() == 40);
    reversed->setChecked(true);
    CHECK(preview.reversed());

    // The address is saved as typed and applied when editing finishes.
    server->setText(QStringLiteral("http://127.0.0.1:18766"));
    {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        CHECK(settings.value(QStringLiteral("ink_preview/server_url")).toString() ==
              QStringLiteral("http://127.0.0.1:18766"));
    }
    CHECK(preview.serverUrl() == QUrl(QStringLiteral("http://127.0.0.1:8765")));
    emit server->editingFinished();
    CHECK(preview.serverUrl() == QUrl(QStringLiteral("http://127.0.0.1:18766")));
    preview.setEnabled(false);
    preview.clientForTesting()->cancel();

    // The level list follows the volume's levels and the choice.
    preview.setLevelsForTesting("preview-volume", {{0, 1.0f, 9.0}, {1, 0.5f, 18.0}});
    preview.setLevelChoice(1);
    REQUIRE(levels->count() == 3);
    CHECK(levels->currentIndex() == 2);
    CHECK(levels->itemText(2).startsWith(QStringLiteral("1 (18")));
    levels->setCurrentIndex(0);
    CHECK(preview.levelChoice() == InkPreviewOverlayController::kAutoLevel);
    CHECK(levels->itemText(0) == QStringLiteral("Auto (level 0)"));

    // Turning the preview off from the controller unchecks the box and restores the rows.
    preview.setHealthForTesting(testHealth());
    preview.setEnabled(true);
    CHECK(enable->isChecked());
    preview.setEnabled(false);
    CHECK(!enable->isChecked());
    CHECK(!opacity->isEnabled());
    CHECK(!colormap->isEnabled());
    qunsetenv("VC3D_CONFIG_DIR");
}

// Against a real server, when INK_PREVIEW_TEST_SERVER names one; skipped otherwise.
TEST_CASE("the preview of a running ink preview server")
{
    const QString address = qEnvironmentVariable("INK_PREVIEW_TEST_SERVER");
    if (address.isEmpty()) {
        MESSAGE("skipped: set INK_PREVIEW_TEST_SERVER to a running ink preview server's address");
        return;
    }
    Scene scene;
    ProbePreviewController preview(&scene.state);
    preview.clientForTesting()->setServerUrl(QUrl(address));
    preview.setEnabled(true);
    if (!waitFor([&] { return preview.health().has_value() || preview.statusIsError(); }) || !preview.health()) {
        MESSAGE("skipped: " << preview.statusText().toStdString());
        return;
    }
    // The synthetic volume holds the layers of models up to 31 deep around depth 16.
    REQUIRE(preview.health()->depth <= 31);
    preview.images(&scene.viewer);
    REQUIRE_MESSAGE(waitFor([&] { return preview.statusText().startsWith(QStringLiteral("Model output")); }, 60000),
                    preview.statusText().toStdString());
    MESSAGE(preview.statusText().toStdString());
    const auto images = preview.images(&scene.viewer);
    REQUIRE(images.size() == 1);
    CHECK(images[0].image.size() == QSize(169, 110));
}

int main(int argc, char** argv)
{
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }
    QApplication app(argc, argv);
    return ::doctest::detail::runAll();
}
