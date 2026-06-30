#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "OpenDataManifest.hpp"

using namespace vc3d::opendata;

namespace {

constexpr const char* kFixture = R"({
  "metadata": {
    "samples": {
      "PHerc0139": {
        "sample": {
          "type": "scroll",
          "description": "Example sample",
          "unknown_sample_field": 7
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
