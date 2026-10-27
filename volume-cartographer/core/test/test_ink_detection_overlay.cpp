#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "CState.hpp"
#include "SurfaceCatalog.hpp"
#include "overlays/InkDetectionOverlayController.hpp"
#include "vc/core/types/Volume.hpp"
#include "support/TestZarrVolume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <QCoreApplication>
#include <QElapsedTimer>
#include <QImage>
#include <QThread>

#include <nlohmann/json.hpp>
#include <vc_object_store/Uuid.hpp>

#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>

namespace fs = std::filesystem;

namespace {

fs::path tempRoot(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto path = fs::temp_directory_path() /
                ("vc_ink_detection_overlay_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(path);
    return path;
}

void writeFile(const fs::path& path, const std::string& body)
{
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << body;
    REQUIRE(out.good());
}

void writeGrayImage(const fs::path& path)
{
    fs::create_directories(path.parent_path());
    cv::Mat_<uint8_t> image(2, 3);
    image(0, 0) = 10;
    image(0, 1) = 20;
    image(0, 2) = 30;
    image(1, 0) = 40;
    image(1, 1) = 50;
    image(1, 2) = 60;
    REQUIRE(cv::imwrite(path.string(), image));
}

std::shared_ptr<QuadSurface> makeSurface(std::string id)
{
    cv::Mat_<cv::Vec3f> points(3, 3);
    points.setTo(cv::Vec3f(0.0f, 0.0f, 0.0f));
    auto surface = std::make_shared<QuadSurface>(points, cv::Vec2f(1.0f, 1.0f));
    surface->id = std::move(id);
    return surface;
}

// An Open Data sample store with one catalog surface published with an ink
// detection, listed by a catalog for the catalog volume.
struct CatalogFixture {
    fs::path root;
    CState state;
    std::shared_ptr<Volume> volume;
    std::unique_ptr<SurfaceCatalog> catalog;
    std::string surfaceUuid;
    fs::path inkFile;

    explicit CatalogFixture(const std::string& tag, const std::string& inkName)
        : root(tempRoot(tag))
    {
        namespace vo = vc::objects;
        vc::test::ZarrVolumeSpec options;
        options.shapeZYX = {4, 4, 4};
        options.chunkShapeZYX = {4, 4, 4};
        options.uuid = "catalog-volume";
        volume = vc::test::createZarrVolume(root / "ct.zarr", options);
        const auto storePath = root / "sample.vcobj";
        vo::OpenOptions storeOptions;
        storeOptions.author = "ink-test";
        {
            auto db = vo::ObjectStore::create(storePath, storeOptions);
            surfaceUuid = vo::util::newUuid();
            auto txn = db.write({{surfaceUuid, std::nullopt}});
            vo::ObjectRecord record;
            record.uuid = surfaceUuid;
            record.kind = "surface";
            record.name = "segment";
            record.meta = nlohmann::json{
                {"coordinates", {{"volume_uuid", "catalog-volume"}, {"scale_factor", 1}}},
                {"catalog", {{"sample", "sample"}, {"segment", "segment"}, {"files", {{
                    {"name", inkName},
                    {"label", "model-a"},
                    {"url", "https://example.test/" + inkName},
                }}}}}}.dump();
            db.upsertObject(txn, record);
            txn.commit();
        }
        const auto attachment = state.objectStoreSession()->attachFetched(storePath, storeOptions,
            [](vo::ObjectStore&, const std::string&, const vo::ObjectStoreSession::FetchWrite&) {
                throw std::runtime_error("offline");
            });
        inkFile = state.objectStoreSession()->artifactDirectory(attachment, surfaceUuid) / inkName;
        catalog = std::make_unique<SurfaceCatalog>(state.objectStoreSession());
        catalog->setSource(std::nullopt, volume);
        catalog->setCatalogSources({attachment});
        QElapsedTimer timer;
        timer.start();
        while ((catalog->loading() || catalog->rows().empty()) && timer.elapsed() < 10000) {
            QCoreApplication::processEvents();
            QThread::msleep(1);
        }
        REQUIRE(catalog->rows().size() == 1);
    }
    ~CatalogFixture()
    {
        catalog.reset();
        std::error_code ec;
        fs::remove_all(root, ec);
    }
};

} // namespace

TEST_CASE("CState batches surface catalog mutations into one notification")
{
    int changedCount = 0;
    int willDeleteCount = 0;
    std::string changedName;
    CState state;
    auto first = makeSurface("first");
    auto second = makeSurface("second");

    const auto changedConnection = QObject::connect(
        &state, &CState::surfaceChanged,
        [&changedCount, &changedName](std::string name, std::shared_ptr<Surface>, bool) {
            ++changedCount;
            changedName = std::move(name);
        });
    const auto deleteConnection = QObject::connect(
        &state, &CState::surfaceWillBeDeleted,
        [&willDeleteCount](std::string, std::shared_ptr<Surface>) {
            ++willDeleteCount;
        });

    state.setSurfacesBatch({
        {"first", first},
        {"second", second},
    });

    CHECK(changedCount == 1);
    CHECK(changedName.empty());
    CHECK(willDeleteCount == 0);
    CHECK(state.surface("first") == first);
    CHECK(state.surface("second") == second);

    state.setSurfacesBatch({
        {"first", nullptr},
        {"second", first},
    });

    CHECK(changedCount == 2);
    CHECK(changedName.empty());
    CHECK(willDeleteCount == 0);
    CHECK(state.surface("first") == nullptr);
    CHECK(state.surface("second") == first);

    QObject::disconnect(changedConnection);
    QObject::disconnect(deleteConnection);
}

TEST_CASE("CState::closeAll clears the project's surfaces and keeps the view planes")
{
    for (const bool withProject : {false, true}) {
        CState state;
        if (withProject) state.setVpkg(VolumePkg::newEmpty());
        state.setSurface("segmentation", makeSurface("segmentation"));
        state.setSurface("other", makeSurface("other"));
        const auto xy = state.surface("xy plane");
        REQUIRE(xy);

        state.closeAll();

        CHECK(state.surface("segmentation") == nullptr);
        CHECK(state.surface("other") == nullptr);
        CHECK(state.surface("xy plane") == xy);
        CHECK(state.surface("xz plane"));
        CHECK(state.surface("yz plane"));
    }
}

TEST_CASE("ink detection options follow the active catalog surface")
{
    int argc = 0;
    QCoreApplication app(argc, nullptr);
    CatalogFixture fixture("options", "prediction.jpg");
    writeFile(fixture.inkFile, "not-empty");

    auto source = makeSurface(fixture.surfaceUuid);
    fixture.state.setSurface("segmentation", source, false, false);

    InkDetectionOverlayController overlay(&fixture.state);
    overlay.setSurfaceCatalog(fixture.catalog.get());
    REQUIRE(overlay.options().size() == 1);
    CHECK(overlay.options().front().localPath == fixture.inkFile);
    CHECK(overlay.options().front().name == "prediction.jpg");

    // A preview of the same surface keeps them; another surface has none.
    fixture.state.setSurface("segmentation", makeSurface(fixture.surfaceUuid), false, false);
    CHECK(overlay.options().size() == 1);
    fixture.state.setSurface("segmentation", makeSurface({}), false, false);
    CHECK(overlay.options().empty());
}

TEST_CASE("ink detection overlay flips the displayed image")
{
    int argc = 0;
    QCoreApplication app(argc, nullptr);
    CatalogFixture fixture("flip", "prediction.png");
    writeGrayImage(fixture.inkFile);

    fixture.state.setSurface("segmentation", makeSurface(fixture.surfaceUuid), false, false);

    InkDetectionOverlayController overlay(&fixture.state);
    overlay.setSurfaceCatalog(fixture.catalog.get());
    REQUIRE(overlay.options().size() == 1);
    overlay.setSelectedPath(overlay.options().front().localPath);

    REQUIRE(!overlay.selectedImageForTesting().isNull());
    CHECK(qRed(overlay.selectedImageForTesting().pixel(0, 0)) == 10);
    CHECK(qAlpha(overlay.selectedImageForTesting().pixel(0, 0)) == 10);

    overlay.setHorizontalFlip(true);
    CHECK(overlay.horizontalFlip());
    CHECK(qRed(overlay.selectedImageForTesting().pixel(0, 0)) == 30);
    CHECK(qRed(overlay.selectedImageForTesting().pixel(2, 0)) == 10);

    overlay.setVerticalFlip(true);
    CHECK(overlay.verticalFlip());
    CHECK(qRed(overlay.selectedImageForTesting().pixel(0, 0)) == 60);
    CHECK(qRed(overlay.selectedImageForTesting().pixel(2, 1)) == 10);

    overlay.setHorizontalFlip(false);
    CHECK(!overlay.horizontalFlip());
    CHECK(qRed(overlay.selectedImageForTesting().pixel(0, 0)) == 40);
    CHECK(qRed(overlay.selectedImageForTesting().pixel(2, 1)) == 30);
}
