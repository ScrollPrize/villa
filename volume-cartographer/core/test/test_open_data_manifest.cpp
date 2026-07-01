#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "OpenDataManifest.hpp"
#include "OpenDataSampleProject.hpp"
#include "OpenDataSegmentCache.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <utility>
#include <unistd.h>

using namespace vc3d::opendata;

namespace {

constexpr const char* kFixture = R"({
  "metadata": {
    "samples": {
      "PHerc0139": {
        "sample": {
          "type": "scroll",
          "description": "Example sample",
          "unknown_sample_field": 7,
          "data": [{
            "type": "photo",
            "origins": [{
              "path": "PHerc0139/photos/PHerc0139_photo.jpg",
              "access_roots": [{
                "type": "s3",
                "url": "s3://vesuvius-challenge-open-data/",
                "usage": "public-read"
              }]
            }]
          }]
        },
        "scans": {
          "20240101000000": {
            "suffix": "scan-a",
            "created_at": "2024-01-01T00:00:00Z",
            "properties": {"operator": "vc"}
          }
        },
        "volumes": {
          "vol1": {
            "scan_id": "20240101000000",
            "suffix": "surface-volume",
            "pixel_size_um": 7.91,
            "energy_kev": 88.0,
            "detector_distance_mm": 12.5,
            "data_format": "zarr",
            "data": [{
              "type": "zarr",
              "properties": {"representation": "normal3d"},
              "origins": [{
                "path": "PHerc0139/volumes/vol1.zarr",
                "access_roots": [
                  {"type": "s3", "url": "s3://private-root/", "usage": "internal"},
                  {"type": "s3", "url": "s3://vesuvius-challenge-open-data/", "usage": "public-read"}
                ]
              }]
            }]
          }
        },
        "segments": {
          "20260311000000": {
            "long_id": "PHerc0139-20260311000000",
            "suffix": "seg-a",
            "original_volume_id": "vol1",
            "width": 123,
            "height": 456,
            "created_at": "2026-03-11T00:00:00Z",
            "creation": {
              "derived_from": {
                "type": "volume",
                "id": "vol1"
              }
            },
            "properties": {
              "original_volume_downscale": 2
            },
            "data": [
              {
                "type": "tifxyz",
                "properties": {"extra": true},
                "origins": [{
                  "path": "PHerc0139/segments/20260311000000/tifxyz",
                  "access_roots": [{
                    "type": "s3",
                    "url": "s3://vesuvius-challenge-open-data/",
                    "usage": "public-read"
                  }]
                }]
              },
              {
                "type": "ink_detection",
                "origins": [{
                  "path": "PHerc0139/segments/20260311000000/ink.zarr",
                  "access_roots": [{
                    "type": "https",
                    "url": "https://example.test/data/",
                    "usage": "public-read"
                  }]
                }]
              },
              {
                "type": "layers_zarr",
                "origins": []
              }
            ]
          }
        }
      }
    },
    "models": {
      "model-a": {"kind": "ink"}
    }
  }
})";

void writeFile(const std::filesystem::path& path, const std::string& body)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << body;
    REQUIRE(out.good());
}

void copyFixtureFile(const std::filesystem::path& source,
                     const std::filesystem::path& target)
{
    std::filesystem::create_directories(target.parent_path());
    std::filesystem::copy_file(
        source,
        target,
        std::filesystem::copy_options::overwrite_existing);
}

cv::Size tifxyzGridSize(const std::filesystem::path& segmentDir)
{
    auto surface = load_quad_from_tifxyz(segmentDir.string());
    REQUIRE(surface != nullptr);
    const auto* points = surface->rawPointsPtr();
    REQUIRE(points != nullptr);
    return points->size();
}

} // namespace

TEST_CASE("OpenDataManifest parses samples and computes summary counts")
{
    const auto manifest = parseOpenDataManifest(kFixture, "fixture://metadata.json");

    CHECK(manifest.manifestUrl == "fixture://metadata.json");
    REQUIRE(manifest.samples.size() == 1);
    const auto* sample = manifest.findSample("PHerc0139");
    REQUIRE(sample != nullptr);
    CHECK(sample->type == "scroll");
    CHECK(sample->description == "Example sample");
    CHECK(sample->scanCount() == 1);
    CHECK(sample->volumeCount() == 1);
    CHECK(sample->segmentCount() == 1);
    CHECK(sample->tifxyzSegmentCount() == 1);
    CHECK(sample->inkDetectionSegmentCount() == 1);
    CHECK(sample->properties.at("unknown_sample_field").get<int>() == 7);
    REQUIRE(sample->artifacts.size() == 1);
    const auto* photo = preferredPhotoArtifact(*sample);
    REQUIRE(photo != nullptr);
    CHECK(photo->resolvedUrl ==
          "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/photos/PHerc0139_photo.jpg");

    REQUIRE(manifest.findModel("model-a") != nullptr);
}

TEST_CASE("OpenDataManifest parses volumes and segment artifact availability")
{
    const auto manifest = parseOpenDataManifest(kFixture);
    const auto& sample = *manifest.findSample("PHerc0139");

    REQUIRE(sample.volumes.size() == 1);
    const auto& volume = sample.volumes.front();
    CHECK(volume.id == "vol1");
    CHECK(volume.scanId == "20240101000000");
    REQUIRE(volume.pixelSizeUm.has_value());
    CHECK(*volume.pixelSizeUm == doctest::Approx(7.91));
    REQUIRE(volume.energyKeV.has_value());
    CHECK(*volume.energyKeV == doctest::Approx(88.0));
    REQUIRE(volume.detectorDistanceMm.has_value());
    CHECK(*volume.detectorDistanceMm == doctest::Approx(12.5));

    const auto* volumeArtifact = preferredVolumeArtifact(volume);
    REQUIRE(volumeArtifact != nullptr);
    CHECK(volumeArtifact->sourcePath ==
          "s3://vesuvius-challenge-open-data/PHerc0139/volumes/vol1.zarr");
    CHECK(volumeArtifact->resolvedUrl ==
          "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/volumes/vol1.zarr");

    REQUIRE(sample.segments.size() == 1);
    const auto& segment = sample.segments.front();
    CHECK(segment.id == "20260311000000");
    CHECK(segment.longId == "PHerc0139-20260311000000");
    CHECK(segment.originalVolumeId == "vol1");
    REQUIRE(segment.width.has_value());
    CHECK(*segment.width == 123);
    REQUIRE(segment.height.has_value());
    CHECK(*segment.height == 456);
    CHECK(segment.hasTifxyz());
    CHECK(segment.hasInkDetection());
    CHECK(segment.hasLayersZarr());
}

TEST_CASE("OpenDataManifest resolves public origins with the website rewrite table")
{
    CHECK(resolveOpenDataUrl("s3://vesuvius-challenge-open-data/a/b") ==
          "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/a/b");
    CHECK(resolveOpenDataUrl("s3://vesuvius-challenge/full-scrolls/x") ==
          "https://data.aws.ash2txt.org/samples/full-scrolls/x");
    CHECK(joinOpenDataUrl("s3://vesuvius-challenge-open-data/", "/a/b") ==
          "s3://vesuvius-challenge-open-data/a/b");

    const auto manifest = parseOpenDataManifest(kFixture);
    const auto& segment = manifest.findSample("PHerc0139")->segments.front();
    const auto* tifxyz = findArtifact(segment.artifacts, "tifxyz");
    REQUIRE(tifxyz != nullptr);
    CHECK(tifxyz->resolvedUrl ==
          "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/segments/20260311000000/tifxyz");

    const auto* ink = findArtifact(segment.artifacts, "ink_detection");
    REQUIRE(ink != nullptr);
    CHECK(ink->resolvedUrl == "https://example.test/data/PHerc0139/segments/20260311000000/ink.zarr");
}

TEST_CASE("OpenDataSampleProject attaches all supported zarr artifacts for a catalog volume")
{
    OpenDataSample sample;
    sample.id = "PHerc0139";

    OpenDataVolume volume;
    volume.id = "scan-volume";
    volume.dataFormat = "zarr";

    auto artifact = [](std::string type, std::string url) {
        OpenDataArtifact out;
        out.type = std::move(type);
        out.resolvedUrl = std::move(url);
        return out;
    };

    volume.artifacts.push_back(artifact("ome-zarr", "http://127.0.0.1:9/base.zarr"));
    volume.artifacts.push_back(artifact("surface-prediction-zarr", "http://127.0.0.1:9/surface.zarr"));
    volume.artifacts.push_back(artifact("ink-detection-3d-zarr", "http://127.0.0.1:9/ink.zarr"));
    volume.artifacts.push_back(artifact("obj", "http://127.0.0.1:9/mesh.obj"));
    sample.volumes.push_back(std::move(volume));

    auto pkg = VolumePkg::newEmpty();
    const auto result = attachOpenDataSampleVolumes(*pkg, sample);

    CHECK(result.supportedVolumes == 3);
    CHECK(result.attachedVolumeEntries == 3);
    CHECK(result.skippedVolumes == 0);
    REQUIRE(pkg->volumeEntries().size() == 3);
    CHECK(pkg->volumeEntries()[0].location == "http://127.0.0.1:9/base.zarr");
    CHECK(pkg->volumeEntries()[1].location == "http://127.0.0.1:9/surface.zarr");
    CHECK(pkg->volumeEntries()[2].location == "http://127.0.0.1:9/ink.zarr");

    CHECK(std::find(pkg->volumeEntries()[1].tags.begin(),
                    pkg->volumeEntries()[1].tags.end(),
                    "prediction") != pkg->volumeEntries()[1].tags.end());
    CHECK(std::find(pkg->volumeEntries()[1].tags.begin(),
                    pkg->volumeEntries()[1].tags.end(),
                    "surface-prediction") != pkg->volumeEntries()[1].tags.end());
    CHECK(std::find(pkg->volumeEntries()[2].tags.begin(),
                    pkg->volumeEntries()[2].tags.end(),
                    "ink-detection-3d") != pkg->volumeEntries()[2].tags.end());
    CHECK(std::find(pkg->volumeEntries()[0].tags.begin(),
                    pkg->volumeEntries()[0].tags.end(),
                    "vc-open-data-volume-id:scan-volume") != pkg->volumeEntries()[0].tags.end());
}


TEST_CASE("OpenDataSegmentCache discovers cached ink detection overlays")
{
    const auto cacheRoot = std::filesystem::temp_directory_path() /
                           ("vc_open_data_ink_detection_test_" + std::to_string(getpid()));
    std::filesystem::remove_all(cacheRoot);
    const auto segmentDir = cacheRoot / "segment";
    writeFile(segmentDir / "catalog-origin.json",
              R"({"sample_id":"PHerc0139","segment_id":"20260311000000"})");
    writeFile(segmentDir / "ink-detections" / "prediction.jpg", "not-empty");
    writeFile(segmentDir / "ink-detections.json",
              R"([{
                "label":"model-a",
                "sample_id":"PHerc0139",
                "segment_id":"20260311000000",
                "segment_long_id":"PHerc0139-20260311000000",
                "artifact_type":"ink_detection",
                "resolved_http_url":"https://example.test/prediction.jpg",
                "local_file":"ink-detections/prediction.jpg"
              }])");

    const auto detections = cachedInkDetectionsForSegmentDirectory(segmentDir);
    REQUIRE(detections.size() == 1);
    CHECK(detections.front().label == "model-a");
    CHECK(detections.front().sampleId == "PHerc0139");
    CHECK(detections.front().segmentId == "20260311000000");
    CHECK(detections.front().segmentLongId == "PHerc0139-20260311000000");
    CHECK(detections.front().artifactType == "ink_detection");
    CHECK(detections.front().sourceUrl == "https://example.test/prediction.jpg");
    CHECK(detections.front().localPath == segmentDir / "ink-detections" / "prediction.jpg");

    std::filesystem::remove_all(cacheRoot);
}

TEST_CASE("OpenDataSampleProject attaches cached tifxyz segments")
{
    const auto manifest = parseOpenDataManifest(kFixture);
    const auto& sample = *manifest.findSample("PHerc0139");

    const auto cacheRoot = std::filesystem::temp_directory_path() /
                           ("vc_open_data_sample_project_test_" + std::to_string(getpid()));
    std::filesystem::remove_all(cacheRoot);

    const auto segmentDir = cacheRoot / "open_data" / "segments" / "PHerc0139" / "20260311000000";
    const auto fixtureSegment = std::filesystem::path(VC_TEST_FIXTURES_DIR) /
                                "segments" / "20241113070770";
    const cv::Size fixtureGridSize = tifxyzGridSize(fixtureSegment);
    writeFile(segmentDir / "meta.json",
              R"({"type":"seg","uuid":"out","name":"seg-a","format":"tifxyz","scale":[1,1]})");
    copyFixtureFile(fixtureSegment / "x.tif", segmentDir / "x.tif");
    copyFixtureFile(fixtureSegment / "y.tif", segmentDir / "y.tif");
    copyFixtureFile(fixtureSegment / "z.tif", segmentDir / "z.tif");
    writeFile(segmentDir / "mask.tif", "optional");
    writeFile(segmentDir / "overlapping.json", "{}");

    auto pkg = VolumePkg::newEmpty();
    OpenDataSampleProjectResult result;
    std::vector<OpenDataSampleDownloadProgress> progressEvents;
    attachOpenDataSampleSegments(
        *pkg,
        sample,
        cacheRoot,
        result,
        [&](const OpenDataSampleDownloadProgress& progress) {
            progressEvents.push_back(progress);
        });

    CHECK(result.supportedTifxyzSegments == 1);
    CHECK(result.cachedTifxyzSegments == 1);
    CHECK(result.attachedSegmentEntries == 1);
    REQUIRE(pkg->segmentEntries().size() == 1);
    CHECK(pkg->hasSegmentations());
    const auto ids = pkg->segmentationIDs();
    REQUIRE(ids.size() == 1);
    CHECK(ids.front() == "PHerc0139-20260311000000");
    REQUIRE(!progressEvents.empty());
    CHECK(progressEvents.back().status == "finished");
    CHECK(progressEvents.back().completedFiles == 0);
    CHECK(progressEvents.back().totalFiles == 6);
    CHECK(progressEvents.back().completedSegments == 1);
    std::ifstream metaIn(segmentDir / "meta.json", std::ios::binary);
    REQUIRE(metaIn.good());
    const auto meta = nlohmann::json::parse(metaIn);
    CHECK(meta.at("vc_open_data_segment_id").get<std::string>() == "20260311000000");
    CHECK(meta.at("vc_open_data_segment_long_id").get<std::string>() == "PHerc0139-20260311000000");
    CHECK(meta.at("vc_open_data_original_volume_id").get<std::string>() == "vol1");
    CHECK(meta.at("vc_open_data_derived_volume_id").get<std::string>() == "vol1");
    CHECK(meta.at("vc_open_data_original_volume_downscale").get<double>() == doctest::Approx(2.0));
    CHECK(meta.at("vc_open_data_coordinates_scaled_to_original_volume").get<double>() == doctest::Approx(2.0));

    const cv::Size cachedGridSize = tifxyzGridSize(segmentDir);
    CHECK(cachedGridSize.width == fixtureGridSize.width * 2);
    CHECK(cachedGridSize.height == fixtureGridSize.height * 2);

    std::ifstream originIn(segmentDir / "catalog-origin.json", std::ios::binary);
    REQUIRE(originIn.good());
    const auto origin = nlohmann::json::parse(originIn);
    CHECK(origin.at("cache_state").get<std::string>() == "current");
    CHECK(origin.at("sample_id").get<std::string>() == "PHerc0139");
    CHECK(origin.at("segment_id").get<std::string>() == "20260311000000");

    std::filesystem::remove_all(cacheRoot);
}

TEST_CASE("OpenDataSampleProject saves and reuses cached volpkg json")
{
    OpenDataSample sample;
    sample.id = "Sample With Spaces";

    const auto cacheRoot = std::filesystem::temp_directory_path() /
                           ("vc_open_data_cached_project_test_" + std::to_string(getpid()));
    std::filesystem::remove_all(cacheRoot);

    OpenDataSampleProjectResult firstResult;
    auto first = createOpenDataSampleProject(sample, cacheRoot, &firstResult);
    const auto projectPath = cacheRoot / "open_data" / "projects" /
                             "Sample_With_Spaces.volpkg.json";
    CHECK(std::filesystem::is_regular_file(projectPath));
    CHECK(first->path() == projectPath);

    first->setSelectedLasagnaDataset("cached-marker");

    OpenDataSampleProjectResult secondResult;
    auto second = createOpenDataSampleProject(sample, cacheRoot, &secondResult);
    CHECK(second->path() == projectPath);
    CHECK(second->selectedLasagnaDataset() == "cached-marker");
    CHECK(std::any_of(secondResult.messages.begin(),
                      secondResult.messages.end(),
                      [](const std::string& message) {
                          return message.find("Loaded cached sample project") != std::string::npos;
                      }));

    std::filesystem::remove_all(cacheRoot);
}
