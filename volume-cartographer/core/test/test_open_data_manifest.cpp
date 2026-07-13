#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "OpenDataManifest.hpp"
#include "OpenDataNormalGrids.hpp"
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

std::string fixtureWithSegmentArtifactType(const char* type)
{
    auto fixture = nlohmann::json::parse(kFixture);
    fixture["metadata"]["samples"]["PHerc0139"]["segments"]["20260311000000"]["data"][0]["type"] = type;
    return fixture.dump();
}

} // namespace

TEST_CASE("OpenDataManifest strictly parses coordinate-bearing artifact parameters")
{
    const auto manifest = parseOpenDataManifest(R"({
      "metadata":{"samples":{"sample":{"volumes":{"volume":{"data":[
        {"type":"surface-prediction-zarr","parameters":{"level":2}},
        {"type":"normal-grids","parameters":{"level":2.5}},
        {"type":"surface-prediction-zarr"},
        {"type":"tifxyz-transformed","parameters":{"target_volume":"target"}}
      ]}}}}}
    })");
    REQUIRE(manifest.samples.size() == 1);
    REQUIRE(manifest.samples[0].volumes.size() == 1);
    const auto& artifacts = manifest.samples[0].volumes[0].artifacts;
    REQUIRE(artifacts.size() == 4);
    CHECK(artifacts[0].levelParameterPresent);
    CHECK(artifacts[0].sourceCoordinateLevel == 2);
    CHECK(artifacts[0].parameters.at("level") == 2);
    CHECK(artifacts[1].levelParameterPresent);
    CHECK_FALSE(artifacts[1].sourceCoordinateLevel.has_value());
    CHECK_FALSE(artifacts[2].levelParameterPresent);
    CHECK_FALSE(artifacts[2].sourceCoordinateLevel.has_value());
    CHECK(artifacts[3].targetVolumeId == "target");
}

TEST_CASE("OpenDataNormalGrids preserves distinct level artifacts and cache identities")
{
    const auto manifest = parseOpenDataManifest(R"({
      "metadata":{"samples":{"sample":{"volumes":{"volume":{"data":[
        {"type":"normal-grids","parameters":{"level":0},"origins":[{
          "path":"grids/L0","access_roots":[{"type":"https","url":"https://example.test/","usage":"public-read"}]}]},
        {"type":"normal-grids","parameters":{"level":2},"origins":[{
          "path":"grids/L2","access_roots":[{"type":"https","url":"https://example.test/","usage":"public-read"}]}]},
        {"type":"normal-grids","parameters":{"level":2.5},"origins":[{
          "path":"grids/invalid","access_roots":[{"type":"https","url":"https://example.test/","usage":"public-read"}]}]}
      ]}}}}}
    })");
    const auto& sample = manifest.samples.front();
    const auto infos = normalGridsArtifacts(sample.id, sample.volumes.front());
    REQUIRE(infos.size() == 2);
    CHECK(infos[0].sourceCoordinateLevel == 0);
    CHECK(infos[1].sourceCoordinateLevel == 2);

    const auto root = std::filesystem::path("/tmp/open-data-normal-grids-test");
    const auto first = normalGridsCacheDir(root, infos[0].sampleId, infos[0].volumeId,
                                           infos[0].sourceCoordinateLevel, infos[0].url);
    const auto second = normalGridsCacheDir(root, infos[1].sampleId, infos[1].volumeId,
                                            infos[1].sourceCoordinateLevel, infos[1].url);
    CHECK(first != second);
    CHECK(first.string().find("L0-") != std::string::npos);
    CHECK(second.string().find("L2-") != std::string::npos);
}

TEST_CASE("OpenDataSegmentCache selects one canonical source and published targets")
{
    OpenDataSample sample;
    sample.id = "sample";
    OpenDataVolume source;
    source.id = "source";
    sample.volumes.push_back(source);
    OpenDataVolume target;
    target.id = "target";
    sample.volumes.push_back(target);

    OpenDataSegment segment;
    segment.id = "segment";
    segment.longId = "sample-segment";
    segment.originalVolumeId = "source";
    segment.properties["original_volume_downscale"] = 4;
    OpenDataArtifact authored;
    authored.type = "tifxyz-flattened";
    authored.resolvedUrl = "https://example.test/authored";
    segment.artifacts.push_back(authored);
    OpenDataArtifact normalized = authored;
    normalized.type = "tifxyz-normalized";
    normalized.resolvedUrl = "https://example.test/normalized";
    segment.artifacts.push_back(normalized);
    OpenDataArtifact publishedA;
    publishedA.type = "tifxyz-transformed";
    publishedA.resolvedUrl = "https://example.test/published-a";
    publishedA.targetVolumeId = "target";
    segment.artifacts.push_back(publishedA);
    OpenDataArtifact publishedUnknown = publishedA;
    publishedUnknown.resolvedUrl = "https://example.test/published-unknown";
    publishedUnknown.targetVolumeId = "missing";
    segment.artifacts.push_back(publishedUnknown);

    const auto representations =
        classifyOpenDataSegmentRepresentations(sample, segment);
    REQUIRE(representations.size() == 2);
    CHECK(representations[0].kind == OpenDataSegmentRepresentationKind::DerivedNative);
    CHECK(representations[0].canonicalSource);
    CHECK(representations[0].sourceCoordinateLevel == 0);
    CHECK(representations[0].coordinateSpace == "sample/source@L0");
    CHECK(representations[0].artifact == &segment.artifacts[0]);
    CHECK(representations[1].kind ==
          OpenDataSegmentRepresentationKind::PublishedTransformed);
    CHECK_FALSE(representations[1].canonicalSource);
    CHECK(representations[1].coordinateSpace == "sample/target@L0");
    CHECK(representations[0].representationId !=
          representations[1].representationId);

    const auto root = std::filesystem::path("/tmp/segment-representations");
    CHECK(openDataSegmentRepresentationCacheRoot(root, sample, representations[0]) !=
          openDataSegmentRepresentationCacheRoot(root, sample, representations[1]));
    CHECK(openDataSegmentRepresentationCacheDirectory(
              root, sample, segment, representations[0]) !=
          openDataSegmentRepresentationCacheDirectory(
              root, sample, segment, representations[1]));

    OpenDataArtifact publishedSource = publishedA;
    publishedSource.resolvedUrl = "https://example.test/published-source";
    publishedSource.targetVolumeId = "source";
    segment.artifacts.push_back(publishedSource);

    const auto withPublishedSource =
        classifyOpenDataSegmentRepresentations(sample, segment);
    REQUIRE(withPublishedSource.size() == 2);
    CHECK(withPublishedSource[0].kind ==
          OpenDataSegmentRepresentationKind::PublishedTransformed);
    CHECK(withPublishedSource[0].canonicalSource);
    CHECK(withPublishedSource[0].coordinateSpace == "sample/source@L0");
    CHECK(withPublishedSource[0].artifact == &segment.artifacts.back());
    CHECK(std::none_of(
        withPublishedSource.begin(), withPublishedSource.end(),
        [](const auto& representation) {
            return representation.kind ==
                   OpenDataSegmentRepresentationKind::DerivedNative;
        }));
}

TEST_CASE("OpenDataSegmentCache prepares lazy source and generated placeholders")
{
    auto manifest = parseOpenDataManifest(kFixture);
    auto sample = *manifest.findSample("PHerc0139");
    sample.segments.front().createdAt = "2026-06-02T20:44:01.955138Z";
    OpenDataVolume targetVolume;
    targetVolume.id = "vol2";
    sample.volumes.push_back(targetVolume);
    sample.properties["properties"]["volume_transforms"] = nlohmann::json::array({
        {
            {"from_volume_id", "vol1"},
            {"transforms", nlohmann::json::array({
                {
                    {"to_volume_id", "vol2"},
                    {"matrix", nlohmann::json::array({
                        nlohmann::json::array({1.0, 0.0, 0.0, 10.0}),
                        nlohmann::json::array({0.0, 1.0, 0.0, 20.0}),
                        nlohmann::json::array({0.0, 0.0, 1.0, 30.0})
                    })}
                }
            })}
        }
    });

    const auto cacheRoot = std::filesystem::temp_directory_path() /
        ("vc_open_data_lazy_placeholder_test_" + std::to_string(getpid()));
    std::filesystem::remove_all(cacheRoot);
    const auto previousAutosaveRoot = VolumePkg::autosaveRoot();
    VolumePkg::setAutosaveRoot(cacheRoot / "autosave");
    auto pkg = VolumePkg::newEmpty();

    const auto result = reconcileOpenDataSampleSegments(
        *pkg, sample, cacheRoot, {}, false);
    const auto sourceDir = openDataCanonicalSegmentCacheDirectory(
        cacheRoot, sample, sample.segments.front());
    const auto generatedDir = openDataTransformedSegmentCacheDirectory(
        cacheRoot, sample, sample.segments.front(), "vol2");

    CHECK(result.cachedTifxyzSegments == 0);
    CHECK(result.transformedTifxyzSegments == 0);
    CHECK(result.attachedSegmentEntries == 2);
    CHECK(isOpenDataSegmentPlaceholder(sourceDir));
    CHECK(isOpenDataSegmentPlaceholder(generatedDir));
    CHECK(std::filesystem::is_regular_file(sourceDir / "meta.json"));
    {
        std::ifstream metaIn(sourceDir / "meta.json", std::ios::binary);
        REQUIRE(metaIn.good());
        const auto meta = nlohmann::json::parse(metaIn);
        CHECK(meta.at("date_last_modified") == "20260602204401");
        CHECK(meta.at("vc_open_data_creation_date") ==
              "2026-06-02T20:44:01.955138Z");
    }
    CHECK_FALSE(std::filesystem::exists(sourceDir / "x.tif"));
    CHECK_FALSE(std::filesystem::exists(generatedDir / "x.tif"));
    CHECK(pkg->segmentEntries().size() == 2);
    CHECK(pkg->segmentationIDs().size() == 1);
    std::set<std::string> folderNames;
    for (const auto& entry : pkg->segmentEntries()) {
        folderNames.insert(
            std::filesystem::path(entry.location).filename().string());
        CHECK(std::find(entry.tags.begin(), entry.tags.end(),
                        "vc-open-data-segment-aggregate") !=
              entry.tags.end());
    }
    CHECK(folderNames == std::set<std::string>{"vol1", "vol2"});

    pkg.reset();
    VolumePkg::setAutosaveRoot(previousAutosaveRoot);
    std::filesystem::remove_all(cacheRoot);
}

TEST_CASE("VolumePkg aggregate open-data folders deduplicate segment lineage")
{
    const auto root = std::filesystem::temp_directory_path() /
        ("vc_open_data_aggregate_dedupe_test_" + std::to_string(getpid()));
    std::filesystem::remove_all(root);
    const auto published = root / "vol1" / "published-L0" / "published";
    const auto generated =
        root / "vol1" / "generated-from-source-L0" / "generated";
    writeFile(published / "meta.json", nlohmann::json{
        {"type", "seg"},
        {"uuid", "published-id"},
        {"format", "tifxyz"},
        {"vc_open_data_catalog_segment_lineage_id", "same-lineage"},
        {"vc_open_data_representation", "published-transformed"}
    }.dump());
    writeFile(generated / "meta.json", nlohmann::json{
        {"type", "seg"},
        {"uuid", "generated-id"},
        {"format", "tifxyz"},
        {"vc_open_data_catalog_segment_lineage_id", "same-lineage"},
        {"vc_open_data_representation", "generated-native-transform"}
    }.dump());

    const auto previousAutosaveRoot = VolumePkg::autosaveRoot();
    VolumePkg::setAutosaveRoot(root / "autosave");
    auto pkg = VolumePkg::newEmpty();
    REQUIRE(pkg->addSegmentsEntry(
        (root / "vol1").string(),
        {"open-data", "immutable", "vc-open-data-segment-aggregate"}));
    const auto ids = pkg->segmentationIDs();
    REQUIRE(ids.size() == 1);
    CHECK(ids.front() == "published-id");

    pkg.reset();
    VolumePkg::setAutosaveRoot(previousAutosaveRoot);
    std::filesystem::remove_all(root);
}

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
    CHECK(segment.createdAt == "2026-03-11T00:00:00Z");
    REQUIRE(segment.width.has_value());
    CHECK(*segment.width == 123);
    REQUIRE(segment.height.has_value());
    CHECK(*segment.height == 456);
    CHECK(segment.hasTifxyz());
    CHECK(segment.hasInkDetection());
    CHECK(segment.hasLayersZarr());
}

TEST_CASE("OpenDataManifest reads the live segment creation date shape")
{
    const auto manifest = parseOpenDataManifest(R"({
      "samples":{"sample":{"segments":{"segment":{
        "creation":{"date":"2026-06-02T20:44:01.955138Z"}
      }}}}
    })");

    REQUIRE(manifest.samples.size() == 1);
    REQUIRE(manifest.samples.front().segments.size() == 1);
    CHECK(manifest.samples.front().segments.front().createdAt ==
          "2026-06-02T20:44:01.955138Z");
}

TEST_CASE("OpenDataManifest propagates catalog pixel size from volume properties or referenced scan")
{
    const auto manifest = parseOpenDataManifest(R"({
      "samples": {
        "sample-a": {
          "scans": {
            "scan-a": {
              "properties": {
                "pixel_size_um": 2.403,
                "detector_distance_mm": 220.0
              }
            }
          },
          "volumes": {
            "volume-with-properties": {
              "scan_id": "scan-a",
              "properties": {
                "pixel_size_um": 7.91,
                "detector_distance_mm": 12.5
              },
              "data": [{
                "type": "ome-zarr",
                "origins": [{
                  "path": "sample-a/volumes/volume-with-properties.zarr",
                  "access_roots": [{
                    "type": "s3",
                    "url": "s3://vesuvius-challenge-open-data/",
                    "usage": "public-read"
                  }]
                }]
              }]
            },
            "volume-from-scan": {
              "scan_id": "scan-a",
              "data": [{
                "type": "ome-zarr",
                "origins": [{
                  "path": "sample-a/volumes/volume-from-scan.zarr",
                  "access_roots": [{
                    "type": "s3",
                    "url": "s3://vesuvius-challenge-open-data/",
                    "usage": "public-read"
                  }]
                }]
              }]
            }
          }
        }
      }
    })");

    const auto* sample = manifest.findSample("sample-a");
    REQUIRE(sample != nullptr);
    REQUIRE(sample->scans.size() == 1);
    REQUIRE(sample->scans.front().pixelSizeUm.has_value());
    CHECK(*sample->scans.front().pixelSizeUm == doctest::Approx(2.403));
    REQUIRE(sample->volumes.size() == 2);

    const auto byId = [&](const std::string& id) -> const OpenDataVolume* {
        const auto it = std::find_if(
            sample->volumes.begin(), sample->volumes.end(), [&](const OpenDataVolume& volume) {
                return volume.id == id;
            });
        return it == sample->volumes.end() ? nullptr : &*it;
    };

    const auto* volumeWithProperties = byId("volume-with-properties");
    REQUIRE(volumeWithProperties != nullptr);
    REQUIRE(volumeWithProperties->pixelSizeUm.has_value());
    CHECK(*volumeWithProperties->pixelSizeUm == doctest::Approx(7.91));
    const auto* volumeFromScan = byId("volume-from-scan");
    REQUIRE(volumeFromScan != nullptr);
    REQUIRE(volumeFromScan->pixelSizeUm.has_value());
    CHECK(*volumeFromScan->pixelSizeUm == doctest::Approx(2.403));

    auto pkg = VolumePkg::newEmpty();
    const auto result = attachOpenDataSampleVolumes(*pkg, *sample);
    CHECK(result.supportedVolumes == 2);
    REQUIRE(pkg->volumeEntries().size() == 2);

    const auto hasTag = [](const vc::project::Entry& entry, const std::string& tag) {
        return std::find(entry.tags.begin(), entry.tags.end(), tag) != entry.tags.end();
    };
    const auto entryForVolumeId = [&](const std::string& id) -> const vc::project::Entry* {
        const auto tag = "vc-open-data-volume-id:" + id;
        const auto it = std::find_if(
            pkg->volumeEntries().begin(), pkg->volumeEntries().end(), [&](const vc::project::Entry& entry) {
                return hasTag(entry, tag);
            });
        return it == pkg->volumeEntries().end() ? nullptr : &*it;
    };

    const auto* volumeWithPropertiesEntry = entryForVolumeId("volume-with-properties");
    REQUIRE(volumeWithPropertiesEntry != nullptr);
    CHECK(hasTag(*volumeWithPropertiesEntry, "vc-open-data-voxel-size-um:7.910000"));
    const auto* volumeFromScanEntry = entryForVolumeId("volume-from-scan");
    REQUIRE(volumeFromScanEntry != nullptr);
    CHECK(hasTag(*volumeFromScanEntry, "vc-open-data-voxel-size-um:2.403000"));
}

TEST_CASE("OpenDataManifest parses exact sample-level volume transforms")
{
    const auto manifest = parseOpenDataManifest(R"({
      "metadata": {
        "samples": {
          "sample-a": {
            "sample": {
              "properties": {
                "volume_transforms": [
                  {
                    "from_volume_id": "vol-a",
                    "transforms": [
                      {
                        "to_volume_id": "vol-b",
                        "matrix": [
                          [0.5, 0.0, 0.0, 10.0],
                          [0.0, 0.5, 0.0, 20.0],
                          [0.0, 0.0, 0.5, 30.0]
                        ],
                        "derivation_path": "vol-a-vol-b"
                      }
                    ]
                  }
                ]
              }
            },
            "volumes": {
              "vol-a": {"data_format": "zarr"},
              "vol-b": {"data_format": "zarr"}
            }
          }
        }
      }
    })");

    const auto* sample = manifest.findSample("sample-a");
    REQUIRE(sample != nullptr);
    REQUIRE(sample->volumeTransforms.size() == 1);
    CHECK(sample->volumeTransforms.front().fromVolumeId == "vol-a");
    REQUIRE(sample->volumeTransforms.front().transforms.size() == 1);
    CHECK(sample->volumeTransforms.front().transforms.front().toVolumeId == "vol-b");
    CHECK(sample->volumeTransforms.front().transforms.front().derivationPath == "vol-a-vol-b");

    const auto forward = findSampleVolumeTransform(*sample, "vol-a", "vol-b");
    REQUIRE(forward.has_value());
    CHECK((*forward)(0, 0) == doctest::Approx(0.5));
    CHECK((*forward)(0, 3) == doctest::Approx(10.0));
    CHECK((*forward)(1, 3) == doctest::Approx(20.0));
    CHECK((*forward)(2, 3) == doctest::Approx(30.0));
    CHECK((*forward)(3, 3) == doctest::Approx(1.0));

    CHECK_FALSE(findSampleVolumeTransform(*sample, "vol-b", "vol-a").has_value());
    CHECK_FALSE(findSampleVolumeTransform(*sample, "vol-a", "vol-c").has_value());
    CHECK_FALSE(findSampleVolumeTransform(*sample, "", "vol-b").has_value());
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
    volume.pixelSizeUm = 7.91;

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

    vc::project::LoadOptions loadOptions;
    loadOptions.deferResolution = true;
    auto pkg = VolumePkg::newEmpty(loadOptions);
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
    CHECK(std::find(pkg->volumeEntries()[2].tags.begin(),
                    pkg->volumeEntries()[2].tags.end(),
                    "vc-open-data-coordinate-level-unknown:missing-parameters.level") !=
          pkg->volumeEntries()[2].tags.end());
    CHECK(std::any_of(
        result.messages.begin(), result.messages.end(), [](const std::string& message) {
            return message.find(
                       "Skipped coordinate pairing for scan-volume "
                       "(ink-detection-3d-zarr) against source L0:") !=
                   std::string::npos;
        }));
    CHECK(std::find(pkg->volumeEntries()[0].tags.begin(),
                    pkg->volumeEntries()[0].tags.end(),
                    "vc-open-data-volume-id:scan-volume") != pkg->volumeEntries()[0].tags.end());
    CHECK(std::find(pkg->volumeEntries()[0].tags.begin(),
                    pkg->volumeEntries()[0].tags.end(),
                    "vc-open-data-voxel-size-um:7.910000") != pkg->volumeEntries()[0].tags.end());
    CHECK(std::find(pkg->volumeEntries()[0].tags.begin(),
                    pkg->volumeEntries()[0].tags.end(),
                    "vc-open-data-source-path:http://127.0.0.1:9/base.zarr") !=
          pkg->volumeEntries()[0].tags.end());
    CHECK(std::find(pkg->volumeEntries()[0].tags.begin(),
                    pkg->volumeEntries()[0].tags.end(),
                    "vc-open-data-source-original-resolution:7.910000") !=
          pkg->volumeEntries()[0].tags.end());
}

TEST_CASE("OpenDataSampleProject prefers the volume sourcing the most segments")
{
    OpenDataSample sample;
    sample.id = "PHerc0139";

    auto artifact = [](std::string url) {
        OpenDataArtifact out;
        out.type = "zarr";
        out.resolvedUrl = std::move(url);
        return out;
    };
    auto volume = [&](std::string id) {
        OpenDataVolume out;
        out.id = std::move(id);
        out.dataFormat = "zarr";
        out.artifacts.push_back(artifact("http://127.0.0.1:9/" + out.id + ".zarr"));
        return out;
    };
    auto segment = [](std::string volumeId) {
        OpenDataSegment out;
        out.originalVolumeId = std::move(volumeId);
        return out;
    };

    sample.volumes.push_back(volume("vol-a"));
    sample.volumes.push_back(volume("vol-b"));
    sample.volumes.push_back(volume("vol-c"));
    sample.segments.push_back(segment("vol-a"));
    sample.segments.push_back(segment("vol-b"));
    sample.segments.push_back(segment("vol-b"));
    sample.segments.push_back(segment("vol-c"));

    auto pkg = VolumePkg::newEmpty();
    const auto result = attachOpenDataSampleVolumes(*pkg, sample);

    CHECK(result.supportedVolumes == 3);
    CHECK(result.preferredVolumeId == "vol-b");
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

TEST_CASE("OpenDataSegmentCache reports local freshness against the live manifest")
{
    const auto manifest = parseOpenDataManifest(kFixture);
    const auto& sample = *manifest.findSample("PHerc0139");
    const auto& segment = sample.segments.front();
    const auto* tifxyz = preferredTifxyzArtifact(segment);
    REQUIRE(tifxyz != nullptr);

    const auto cacheRoot = std::filesystem::temp_directory_path() /
                           ("vc_open_data_cache_state_test_" + std::to_string(getpid()));
    std::filesystem::remove_all(cacheRoot);
    const auto segmentDir = openDataSegmentCacheDirectory(cacheRoot, sample, segment);

    CHECK(cacheStateForSegment(cacheRoot, sample, segment) == OpenDataSegmentCacheState::Missing);

    writeFile(segmentDir / "meta.json", "{}");
    CHECK(cacheStateForSegment(cacheRoot, sample, segment) == OpenDataSegmentCacheState::Incomplete);

    writeFile(segmentDir / "x.tif", "x");
    writeFile(segmentDir / "y.tif", "y");
    writeFile(segmentDir / "z.tif", "z");
    writeFile(segmentDir / "catalog-origin.json",
              R"({"sample_id":"PHerc0139","segment_id":"20260311000000","resolved_http_url":"https://stale.example/old"})");
    CHECK(cacheStateForSegment(cacheRoot, sample, segment) == OpenDataSegmentCacheState::Stale);

    writeFile(segmentDir / "catalog-origin.json",
              nlohmann::json{
                  {"sample_id", sample.id},
                  {"segment_id", segment.id},
                  {"resolved_http_url", tifxyz->resolvedUrl},
                  {"cache_state", "current"}
              }.dump());
    CHECK(cacheStateForSegment(cacheRoot, sample, segment) == OpenDataSegmentCacheState::Current);

    writeFile(segmentDir / "catalog-origin.json",
              nlohmann::json{
                  {"sample_id", sample.id},
                  {"segment_id", segment.id},
                  {"resolved_http_url", tifxyz->resolvedUrl},
                  {"cache_state", "orphaned"}
              }.dump());
    CHECK(cacheStateForSegment(cacheRoot, sample, segment) == OpenDataSegmentCacheState::Orphaned);

    std::filesystem::remove_all(cacheRoot);
}

TEST_CASE("OpenDataSampleProject attaches cached tifxyz segments")
{
    const auto manifest = parseOpenDataManifest(kFixture);
    const auto& sample = *manifest.findSample("PHerc0139");

    const auto cacheRoot = std::filesystem::temp_directory_path() /
                           ("vc_open_data_sample_project_test_" + std::to_string(getpid()));
    std::filesystem::remove_all(cacheRoot);

    const auto segmentDir = openDataSegmentCacheDirectory(cacheRoot, sample, sample.segments.front());
    CHECK(segmentDir.parent_path().filename() == "vol1");
    CHECK(segmentDir.parent_path().parent_path().filename() == "PHerc0139");
    const auto fixtureSegment = std::filesystem::path(VC_TEST_FIXTURES_DIR) /
                                "segments" / "20241113070770";
    const auto* tifxyz = preferredTifxyzArtifact(sample.segments.front());
    REQUIRE(tifxyz != nullptr);
    const cv::Size fixtureGridSize = tifxyzGridSize(fixtureSegment);
    writeFile(segmentDir / "meta.json",
              R"({"type":"seg","uuid":"20260311000000.tmp-12345","name":"seg-a","format":"tifxyz","scale":[1,1]})");
    copyFixtureFile(fixtureSegment / "x.tif", segmentDir / "x.tif");
    copyFixtureFile(fixtureSegment / "y.tif", segmentDir / "y.tif");
    copyFixtureFile(fixtureSegment / "z.tif", segmentDir / "z.tif");
    writeFile(segmentDir / "mask.tif", "optional");
    writeFile(segmentDir / "overlapping.json", "{}");
    writeFile(segmentDir / "catalog-origin.json",
              nlohmann::json{
                  {"sample_id", sample.id},
                  {"segment_id", sample.segments.front().id},
                  {"resolved_http_url", tifxyz->resolvedUrl},
                  {"cache_state", "current"}
              }.dump());

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
    const auto representations = classifyOpenDataSegmentRepresentations(
        sample, sample.segments.front());
    REQUIRE(representations.size() == 1);
    CHECK(ids.front() == representations.front().representationId);
    REQUIRE(!progressEvents.empty());
    CHECK(progressEvents.back().status == "placeholders-finished");
    CHECK(progressEvents.back().completedFiles == 0);
    CHECK(progressEvents.back().totalFiles == 0);
    CHECK(progressEvents.back().completedSegments == 1);
    std::ifstream metaIn(segmentDir / "meta.json", std::ios::binary);
    REQUIRE(metaIn.good());
    const auto meta = nlohmann::json::parse(metaIn);
    CHECK(meta.at("uuid").get<std::string>() ==
          representations.front().representationId);
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

    metaIn.close();
    originIn.close();
    pkg.reset();
    std::filesystem::remove_all(cacheRoot);
}

TEST_CASE("OpenDataSegmentCache does not auto-associate transformed artifacts without a target")
{
    const auto manifest = parseOpenDataManifest(
        fixtureWithSegmentArtifactType("tifxyz-transformed"));
    const auto& sample = *manifest.findSample("PHerc0139");

    const auto cacheRoot = std::filesystem::temp_directory_path() /
                           ("vc_open_data_transformed_segment_test_" + std::to_string(getpid()));
    std::filesystem::remove_all(cacheRoot);

    const auto segmentDir = openDataSegmentCacheDirectory(cacheRoot, sample, sample.segments.front());
    CHECK(segmentDir.parent_path().filename() == "vol1");
    CHECK(segmentDir.parent_path().parent_path().filename() == "PHerc0139");
    const auto fixtureSegment = std::filesystem::path(VC_TEST_FIXTURES_DIR) /
                                "segments" / "20241113070770";
    const cv::Size fixtureGridSize = tifxyzGridSize(fixtureSegment);
    writeFile(segmentDir / "meta.json",
              R"({"type":"seg","uuid":"out","name":"seg-a","format":"tifxyz","scale":[1,1]})");
    copyFixtureFile(fixtureSegment / "x.tif", segmentDir / "x.tif");
    copyFixtureFile(fixtureSegment / "y.tif", segmentDir / "y.tif");
    copyFixtureFile(fixtureSegment / "z.tif", segmentDir / "z.tif");

    auto pkg = VolumePkg::newEmpty();
    OpenDataSampleProjectResult result;
    attachOpenDataSampleSegments(*pkg, sample, cacheRoot, result);

    CHECK(result.supportedTifxyzSegments == 1);
    CHECK(result.cachedTifxyzSegments == 0);
    CHECK(result.attachedSegmentEntries == 0);
    CHECK(pkg->segmentEntries().empty());

    std::ifstream metaIn(segmentDir / "meta.json", std::ios::binary);
    REQUIRE(metaIn.good());
    const auto meta = nlohmann::json::parse(metaIn);
    CHECK_FALSE(meta.contains("vc_open_data_original_volume_downscale"));
    CHECK(!meta.contains("vc_open_data_coordinates_scaled_to_original_volume"));

    const cv::Size cachedGridSize = tifxyzGridSize(segmentDir);
    CHECK(cachedGridSize.width == fixtureGridSize.width);
    CHECK(cachedGridSize.height == fixtureGridSize.height);

    metaIn.close();
    pkg.reset();
    std::filesystem::remove_all(cacheRoot);
}

TEST_CASE("OpenDataSegmentCache writes per-volume transformed segment caches")
{
    auto manifest = parseOpenDataManifest(kFixture);
    auto sample = *manifest.findSample("PHerc0139");
    OpenDataVolume targetVolume;
    targetVolume.id = "vol2";
    sample.volumes.push_back(targetVolume);
    sample.properties["properties"]["volume_transforms"] = nlohmann::json::array({
        {
            {"from_volume_id", "vol1"},
            {"transforms", nlohmann::json::array({
                {
                    {"to_volume_id", "vol2"},
                    {"matrix", nlohmann::json::array({
                        nlohmann::json::array({1.0, 0.0, 0.0, 10.0}),
                        nlohmann::json::array({0.0, 1.0, 0.0, 20.0}),
                        nlohmann::json::array({0.0, 0.0, 1.0, 30.0})
                    })}
                }
            })}
        }
    });

    const auto cacheRoot = std::filesystem::temp_directory_path() /
                           ("vc_open_data_volume_transform_test_" + std::to_string(getpid()));
    std::filesystem::remove_all(cacheRoot);

    const auto sourceDir = openDataSegmentCacheDirectory(cacheRoot, sample, sample.segments.front());
    const auto transformedDir = openDataTransformedSegmentCacheDirectory(
        cacheRoot, sample, sample.segments.front(), "vol2");
    CHECK(sourceDir.parent_path().filename() == "vol1");
    CHECK(sourceDir.parent_path().parent_path().filename() == "PHerc0139");
    CHECK(transformedDir.parent_path().filename() == "generated-from-vol1-L0");
    CHECK(transformedDir.parent_path().parent_path().filename() == "vol2");
    const auto fixtureSegment = std::filesystem::path(VC_TEST_FIXTURES_DIR) /
                                "segments" / "20241113070770";
    writeFile(sourceDir / "meta.json",
              R"({"type":"seg","uuid":"20260311000000.tmp-12345","name":"seg-a","format":"tifxyz","scale":[1,1]})");
    copyFixtureFile(fixtureSegment / "x.tif", sourceDir / "x.tif");
    copyFixtureFile(fixtureSegment / "y.tif", sourceDir / "y.tif");
    copyFixtureFile(fixtureSegment / "z.tif", sourceDir / "z.tif");
    writeFile(sourceDir / "catalog-origin.json",
              nlohmann::json{
                  {"sample_id", sample.id},
                  {"segment_id", sample.segments.front().id},
                  {"resolved_http_url", preferredTifxyzArtifact(sample.segments.front())->resolvedUrl},
                  {"cache_state", "current"}
              }.dump());

    auto pkg = VolumePkg::newEmpty();
    OpenDataSampleProjectResult result;
    attachOpenDataSampleSegments(*pkg, sample, cacheRoot, result);

    CHECK(result.cachedTifxyzSegments == 1);
    CHECK(result.transformedTifxyzSegments == 0);
    REQUIRE(std::filesystem::is_regular_file(transformedDir / "meta.json"));
    CHECK(isOpenDataSegmentPlaceholder(transformedDir));
    REQUIRE(pkg->segmentEntries().size() == 2);
    CHECK(std::any_of(pkg->segmentEntries().begin(),
                      pkg->segmentEntries().end(),
                      [](const vc::project::Entry& entry) {
                          return std::find(entry.tags.begin(),
                                           entry.tags.end(),
                                           "vc-open-data-target-volume-id:vol2") != entry.tags.end();
                      }));

    const auto materialized =
        materializeOpenDataSegmentFolder(transformedDir.parent_path());
    REQUIRE(materialized.success);
    CHECK(materialized.materializedSegments == 1);
    CHECK_FALSE(isOpenDataSegmentPlaceholder(transformedDir));
    const auto alreadyMaterialized = materializeOpenDataSegment(transformedDir);
    REQUIRE(alreadyMaterialized.success);
    CHECK(alreadyMaterialized.alreadyMaterialized);

    sample.segments.front().createdAt = "2026-06-23T14:11:35.916443Z";
    const auto refreshed = reconcileOpenDataSampleSegments(
        *pkg, sample, cacheRoot, {}, false);
    CHECK(refreshed.failedTransformedTifxyzSegments == 0);

    std::ifstream metaIn(transformedDir / "meta.json", std::ios::binary);
    REQUIRE(metaIn.good());
    const auto meta = nlohmann::json::parse(metaIn);
    CHECK(meta.at("vc_open_data_transform_source_volume_id").get<std::string>() == "vol1");
    CHECK(meta.at("vc_open_data_transform_target_volume_id").get<std::string>() == "vol2");
    CHECK(meta.at("vc_open_data_volume_transform_matrix")[0][3].get<double>() == doctest::Approx(10.0));
    CHECK(meta.at("date_last_modified") == "20260623141135");
    CHECK(meta.at("vc_open_data_creation_date") ==
          "2026-06-23T14:11:35.916443Z");

    metaIn.close();
    pkg.reset();
    std::filesystem::remove_all(cacheRoot);
}

TEST_CASE("OpenDataSegmentCache prefers published source and target representations")
{
    auto manifest = parseOpenDataManifest(kFixture);
    auto sample = *manifest.findSample("PHerc0139");
    OpenDataVolume targetVolume;
    targetVolume.id = "vol2";
    sample.volumes.push_back(targetVolume);
    sample.properties["properties"]["volume_transforms"] = nlohmann::json::array({
        {
            {"from_volume_id", "vol1"},
            {"transforms", nlohmann::json::array({
                {
                    {"to_volume_id", "vol2"},
                    {"matrix", nlohmann::json::array({
                        nlohmann::json::array({1.0, 0.0, 0.0, 10.0}),
                        nlohmann::json::array({0.0, 1.0, 0.0, 20.0}),
                        nlohmann::json::array({0.0, 0.0, 1.0, 30.0})
                    })}
                }
            })}
        }
    });

    OpenDataArtifact publishedSource;
    publishedSource.type = "tifxyz-transformed";
    publishedSource.resolvedUrl = "https://example.test/published-source";
    publishedSource.targetVolumeId = "vol1";
    sample.segments.front().artifacts.push_back(publishedSource);
    OpenDataArtifact publishedTarget = publishedSource;
    publishedTarget.resolvedUrl = "https://example.test/published-target";
    publishedTarget.targetVolumeId = "vol2";
    sample.segments.front().artifacts.push_back(publishedTarget);

    const auto representations =
        classifyOpenDataSegmentRepresentations(sample, sample.segments.front());
    REQUIRE(representations.size() == 2);
    REQUIRE(representations[0].canonicalSource);
    CHECK(representations[0].coordinateVolumeId == "vol1");
    CHECK(representations[1].coordinateVolumeId == "vol2");

    const auto cacheRoot = std::filesystem::temp_directory_path() /
        ("vc_open_data_published_precedence_test_" + std::to_string(getpid()));
    std::filesystem::remove_all(cacheRoot);
    const auto fixtureSegment = std::filesystem::path(VC_TEST_FIXTURES_DIR) /
                                "segments" / "20241113070770";
    auto populateRepresentation = [&](const OpenDataSegmentRepresentation& representation) {
        const auto dir = openDataSegmentRepresentationCacheDirectory(
            cacheRoot, sample, sample.segments.front(), representation);
        writeFile(dir / "meta.json",
                  R"({"type":"seg","uuid":"cached","name":"seg-a","format":"tifxyz","scale":[1,1]})");
        copyFixtureFile(fixtureSegment / "x.tif", dir / "x.tif");
        copyFixtureFile(fixtureSegment / "y.tif", dir / "y.tif");
        copyFixtureFile(fixtureSegment / "z.tif", dir / "z.tif");
        writeFile(dir / "catalog-origin.json",
                  nlohmann::json{
                      {"sample_id", sample.id},
                      {"segment_id", sample.segments.front().id},
                      {"resolved_http_url", representation.artifact->resolvedUrl},
                      {"cache_state", "current"}
                  }.dump());
        return dir;
    };
    const auto sourcePublishedDir = populateRepresentation(representations[0]);
    const auto targetPublishedDir = populateRepresentation(representations[1]);
    const auto legacySourceDir = openDataSegmentCacheDirectory(
        cacheRoot, sample, sample.segments.front());
    const auto generatedDir = openDataTransformedSegmentCacheDirectory(
        cacheRoot, sample, sample.segments.front(), "vol2");

    const auto previousAutosaveRoot = VolumePkg::autosaveRoot();
    VolumePkg::setAutosaveRoot(cacheRoot / "autosave");
    auto pkg = VolumePkg::newEmpty();
    const auto result = reconcileOpenDataSampleSegments(
        *pkg, sample, cacheRoot, {}, false);

    CHECK(result.cachedTifxyzSegments == 1);
    CHECK(result.transformedTifxyzSegments == 0);
    CHECK(result.failedTifxyzSegments == 0);
    CHECK(result.failedTransformedTifxyzSegments == 0);
    CHECK(std::filesystem::is_regular_file(sourcePublishedDir / "meta.json"));
    CHECK(std::filesystem::is_regular_file(targetPublishedDir / "meta.json"));
    CHECK_FALSE(std::filesystem::exists(legacySourceDir));
    CHECK_FALSE(std::filesystem::exists(generatedDir));
    CHECK(pkg->segmentEntries().size() == 2);

    pkg.reset();
    VolumePkg::setAutosaveRoot(previousAutosaveRoot);
    std::filesystem::remove_all(cacheRoot);
}

TEST_CASE("OpenDataSegmentCache places GrowPatch output under the sample patches folder")
{
    const std::filesystem::path cacheRoot = "/home/test/.VC3D/remote_cache";
    CHECK(openDataPatchesRoot(cacheRoot, "PHerc0175A") ==
          cacheRoot / "open_data" / "segments" / "PHerc0175A" / "patches");
    CHECK(openDataPatchesRoot(cacheRoot, "Sample With Spaces") ==
          cacheRoot / "open_data" / "segments" / "Sample_With_Spaces" / "patches");
}

TEST_CASE("OpenDataSegmentCache counts manually created sample segments")
{
    const auto cacheRoot = std::filesystem::temp_directory_path() /
                           ("vc_open_data_manual_segment_count_test_" + std::to_string(getpid()));
    std::filesystem::remove_all(cacheRoot);

    const auto sampleRoot = cacheRoot / "open_data" / "segments" / "PHerc0175A";
    writeFile(sampleRoot / "patches" / "manual-1" / "meta.json", "{}");
    writeFile(sampleRoot / "my-custom-folder" / "manual-2" / "meta.json", "{}");
    writeFile(sampleRoot / "another" / "nested" / "manual-3" / "meta.json", "{}");
    writeFile(sampleRoot / "volume-1_editable" / "manual-4" / "meta.json", "{}");
    writeFile(sampleRoot / "patches" / "incomplete" / "x.tif", "x");
    writeFile(sampleRoot / "patches" / "backups" / "manual-1" / "meta.json", "{}");
    writeFile(sampleRoot / ".download-in-progress" / "manual-5" / "meta.json", "{}");
    writeFile(sampleRoot / "volume-1" / "catalog-segment" / "meta.json", "{}");
    writeFile(sampleRoot / "volume-1" / "catalog-segment" / "catalog-origin.json", "{}");
    writeFile(sampleRoot / "notes.txt", "not a segment");

    CHECK(manualOpenDataSegmentCount(cacheRoot, "PHerc0175A") == 4);
    CHECK(manualOpenDataSegmentCount(cacheRoot, "PHerc1447") == 0);

    std::filesystem::remove_all(cacheRoot);
}

TEST_CASE("OpenDataSegmentCache names editable copies after the current segments folder")
{
    const std::filesystem::path cacheRoot = "/home/test/.VC3D/remote_cache";
    const auto currentSegmentsFolder = cacheRoot / "open_data" / "segments" /
                                       "PHerc0175A" / "volume-1";
    const auto editableSegmentsFolder = currentSegmentsFolder.parent_path() /
                                        "volume-1_editable";

    CHECK(defaultEditableCopyPathForCatalogSegment(
              currentSegmentsFolder / "segment-42", currentSegmentsFolder) ==
          editableSegmentsFolder / "segment-42");
    CHECK(defaultEditableCopyPathForCatalogSegment(
              currentSegmentsFolder / "segment-99", currentSegmentsFolder) ==
          editableSegmentsFolder / "segment-99");

    const auto spacedFolder = currentSegmentsFolder.parent_path() / "source surfaces";
    CHECK(defaultEditableCopyPathForCatalogSegment(
              spacedFolder / "surface 7", spacedFolder) ==
          spacedFolder.parent_path() / "source surfaces_editable" / "surface 7");

    const auto catalogOverlay = currentSegmentsFolder.parent_path() /
                                "catalog-overlay" / "segment-100";
    CHECK(defaultEditableCopyPathForCatalogSegment(
              catalogOverlay, currentSegmentsFolder) ==
          editableSegmentsFolder / "segment-100");
}

TEST_CASE("OpenDataSegmentCache editable copy is selectable from its new folder")
{
    const auto testRoot = std::filesystem::temp_directory_path() /
                          ("vc_open_data_editable_selection_test_" +
                           std::to_string(getpid()));
    std::filesystem::remove_all(testRoot);

    const auto sourceRoot = testRoot / "catalog-segments";
    const auto sourceSegment = sourceRoot / "20241113070770";
    const auto fixtureSegment = std::filesystem::path(VC_TEST_FIXTURES_DIR) /
                                "segments" / "20241113070770";
    copyFixtureFile(fixtureSegment / "meta.json", sourceSegment / "meta.json");
    copyFixtureFile(fixtureSegment / "x.tif", sourceSegment / "x.tif");
    copyFixtureFile(fixtureSegment / "y.tif", sourceSegment / "y.tif");
    copyFixtureFile(fixtureSegment / "z.tif", sourceSegment / "z.tif");
    writeFile(sourceSegment / "catalog-origin.json", "{}");

    const auto editableSegment =
        defaultEditableCopyPathForCatalogSegment(sourceSegment, sourceRoot);
    const auto editableRoot = editableSegment.parent_path();
    copyCatalogSegmentToEditableDirectory(sourceSegment, editableSegment);

    const auto previousAutosaveRoot = VolumePkg::autosaveRoot();
    VolumePkg::setAutosaveRoot(testRoot / "autosave");
    auto pkg = VolumePkg::newEmpty();
    REQUIRE(pkg->addSegmentsEntry(sourceRoot.string(), {"open-data", "immutable"}));
    REQUIRE(pkg->addSegmentsEntry(editableRoot.string(), {"open-data-editable"}));
    pkg->setOutputSegments(editableRoot.string());

    CHECK(pkg->outputSegmentsPath() == editableRoot);
    CHECK_FALSE(std::filesystem::exists(editableSegment / "catalog-origin.json"));
    const auto surface = pkg->loadSurface("20241113070770");
    REQUIRE(surface);
    CHECK(surface->path == editableSegment);

    pkg.reset();
    VolumePkg::setAutosaveRoot(previousAutosaveRoot);
    std::filesystem::remove_all(testRoot);
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

TEST_CASE("OpenDataSampleProject reuses cached project and attaches existing cached transforms")
{
    const auto manifest = parseOpenDataManifest(kFixture);
    auto sample = *manifest.findSample("PHerc0139");

    const auto cacheRoot = std::filesystem::temp_directory_path() /
                           ("vc_open_data_cached_project_segments_test_" + std::to_string(getpid()));
    std::filesystem::remove_all(cacheRoot);

    const auto segmentDir = openDataSegmentCacheDirectory(
        cacheRoot,
        sample,
        sample.segments.front());
    const auto fixtureSegment = std::filesystem::path(VC_TEST_FIXTURES_DIR) /
                                "segments" / "20241113070770";
    const auto* tifxyz = preferredTifxyzArtifact(sample.segments.front());
    REQUIRE(tifxyz != nullptr);
    writeFile(segmentDir / "meta.json",
              R"({"type":"seg","uuid":"20260311000000.tmp-12345","name":"seg-a","format":"tifxyz","scale":[1,1]})");
    copyFixtureFile(fixtureSegment / "x.tif", segmentDir / "x.tif");
    copyFixtureFile(fixtureSegment / "y.tif", segmentDir / "y.tif");
    copyFixtureFile(fixtureSegment / "z.tif", segmentDir / "z.tif");
    writeFile(segmentDir / "catalog-origin.json",
              nlohmann::json{
                  {"sample_id", sample.id},
                  {"segment_id", sample.segments.front().id},
                  {"resolved_http_url", tifxyz->resolvedUrl},
                  {"cache_state", "current"}
              }.dump());

    OpenDataSampleProjectResult firstResult;
    std::vector<OpenDataSampleDownloadProgress> firstProgressEvents;
    auto first = createOpenDataSampleProject(
        sample,
        cacheRoot,
        &firstResult,
        [&](const OpenDataSampleDownloadProgress& progress) {
            firstProgressEvents.push_back(progress);
        });
    REQUIRE(first);
    CHECK(firstResult.cachedTifxyzSegments == 1);
    CHECK(firstResult.attachedSegmentEntries == 1);
    REQUIRE(!firstProgressEvents.empty());

    OpenDataVolume targetVolume;
    targetVolume.id = "vol2";
    targetVolume.dataFormat = "zarr";
    OpenDataArtifact targetArtifact;
    targetArtifact.type = "zarr";
    targetArtifact.resolvedUrl = "http://127.0.0.1:9/vol2.zarr";
    targetVolume.artifacts.push_back(std::move(targetArtifact));
    sample.volumes.push_back(std::move(targetVolume));
    sample.properties["properties"]["volume_transforms"] = nlohmann::json::array({
        {
            {"from_volume_id", "vol1"},
            {"transforms", nlohmann::json::array({
                {
                    {"to_volume_id", "vol2"},
                    {"matrix", nlohmann::json::array({
                        nlohmann::json::array({1.0, 0.0, 0.0, 10.0}),
                        nlohmann::json::array({0.0, 1.0, 0.0, 20.0}),
                        nlohmann::json::array({0.0, 0.0, 1.0, 30.0})
                    })}
                }
            })}
        }
    });
    const auto transformedDir = openDataTransformedSegmentCacheDirectory(
        cacheRoot,
        sample,
        sample.segments.front(),
        "vol2");
    writeFile(transformedDir / "meta.json",
              R"({"type":"seg","uuid":"PHerc0139-20260311000000","name":"seg-a","format":"tifxyz","scale":[1,1]})");
    copyFixtureFile(fixtureSegment / "x.tif", transformedDir / "x.tif");
    copyFixtureFile(fixtureSegment / "y.tif", transformedDir / "y.tif");
    copyFixtureFile(fixtureSegment / "z.tif", transformedDir / "z.tif");

    OpenDataSampleProjectResult secondResult;
    std::vector<OpenDataSampleDownloadProgress> secondProgressEvents;
    auto second = createOpenDataSampleProject(
        sample,
        cacheRoot,
        &secondResult,
        [&](const OpenDataSampleDownloadProgress& progress) {
            secondProgressEvents.push_back(progress);
    });
    REQUIRE(second);
    CHECK(secondResult.cachedTifxyzSegments == 1);
    CHECK(secondResult.transformedTifxyzSegments == 1);
    CHECK(secondResult.attachedSegmentEntries == 1);
    CHECK_FALSE(secondProgressEvents.empty());
    REQUIRE(second->segmentEntries().size() == 2);
    CHECK(std::any_of(second->segmentEntries().begin(),
                      second->segmentEntries().end(),
                      [](const vc::project::Entry& entry) {
                          return std::find(entry.tags.begin(),
                                           entry.tags.end(),
                                           "vc-open-data-target-volume-id:vol2") != entry.tags.end();
                      }));
    CHECK(std::any_of(secondResult.messages.begin(),
                      secondResult.messages.end(),
                      [](const std::string& message) {
                          return message.find("Loaded cached sample project") != std::string::npos;
                      }));

    std::filesystem::remove_all(cacheRoot);
}
