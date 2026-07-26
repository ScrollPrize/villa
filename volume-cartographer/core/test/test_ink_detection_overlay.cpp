#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "CState.hpp"
#include "overlays/InkDetectionOverlayController.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <QImage>

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

} // namespace

TEST_CASE("CState batches surface catalog mutations into one notification")
{
    int changedCount = 0;
    int willDeleteCount = 0;
    std::string changedName;
    CState state(0);
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

TEST_CASE("ink detection options survive transformed segmentation previews with source id")
{
    const std::string segmentId = "20260623144957-w053-058";
    const std::string segmentLongId = "PHercParis4-" + segmentId;

    const fs::path root = tempRoot("preview_id");
    const fs::path segmentDir = root / "segment";
    writeFile(segmentDir / "ink-detections" / "prediction.jpg", "not-empty");
    writeFile(segmentDir / "ink-detections.json",
              R"([{
                "label":"model-a",
                "sample_id":"PHercParis4",
                "segment_id":")" + segmentId + R"(",
                "segment_long_id":")" + segmentLongId + R"(",
                "artifact_type":"ink_detection",
                "resolved_http_url":"https://example.test/prediction.jpg",
                "local_file":"ink-detections/prediction.jpg"
              }])");

    auto package = VolumePkg::newEmpty();
    REQUIRE(package->addSegmentsEntry(segmentDir.string()));

    CState state(0);
    state.setVpkg(package);

    auto source = makeSurface(segmentLongId);
    state.setActiveSurface(segmentLongId, source);
    state.setSurface("segmentation", source, false, false);

    InkDetectionOverlayController overlay(&state);
    REQUIRE(overlay.options().size() == 1);
    CHECK(overlay.options().front().segmentLongId == segmentLongId);

    auto previewWithSourceId = makeSurface(segmentLongId);
    previewWithSourceId->path.clear();
    state.setSurface("segmentation", previewWithSourceId, false, false);
    CHECK(overlay.options().size() == 1);

    auto anonymousPreview = makeSurface({});
    state.setSurface("segmentation", anonymousPreview, false, false);
    CHECK(overlay.options().empty());

    fs::remove_all(root);
}

TEST_CASE("ink detection overlay flips the displayed image")
{
    const std::string segmentId = "20260630120000-w001-002";
    const std::string segmentLongId = "PHercParis4-" + segmentId;

    const fs::path root = tempRoot("flip");
    const fs::path segmentDir = root / "segment";
    writeGrayImage(segmentDir / "ink-detections" / "prediction.png");
    writeFile(segmentDir / "ink-detections.json",
              R"([{
                "label":"model-a",
                "sample_id":"PHercParis4",
                "segment_id":")" + segmentId + R"(",
                "segment_long_id":")" + segmentLongId + R"(",
                "artifact_type":"ink_detection",
                "resolved_http_url":"https://example.test/prediction.png",
                "local_file":"ink-detections/prediction.png"
              }])");

    auto package = VolumePkg::newEmpty();
    REQUIRE(package->addSegmentsEntry(segmentDir.string()));

    CState state(0);
    state.setVpkg(package);

    auto source = makeSurface(segmentLongId);
    state.setActiveSurface(segmentLongId, source);
    state.setSurface("segmentation", source, false, false);

    InkDetectionOverlayController overlay(&state);
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

    fs::remove_all(root);
}
