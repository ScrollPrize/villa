#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "OpenDataManifest.hpp"
#include "OpenDataCoordinateIdentity.hpp"
#include "OpenDataLasagna.hpp"
#include "OpenDataNormalGrids.hpp"
#include "OpenDataSampleProject.hpp"
#include "ProjectCreationDefaults.hpp"
#include "VolumeDisplayNames.hpp"
#include "OpenDataSurfaceStore.hpp"
#include "VCSettings.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/MemMap.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/lasagna/Dataset.hpp"

#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/objects/ObjectStore.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <utility>

#include <QSettings>

using namespace vc3d::opendata;

TEST_CASE("Overlay coordinate compatibility permits untagged local volumes")
{
    const auto scan = coordinateSpaceTag({
        "vc-open-data-source-coordinate-level:0",
        "vc-open-data-coordinate-space:PHerc0139/scan@L0"});
    const auto fiber = coordinateSpaceTag({"vc-lasagna-group:presence"});
    REQUIRE(scan == "PHerc0139/scan@L0");
    REQUIRE(fiber.empty());
    CHECK(coordinateSpaceTagsCompatible(scan, fiber));
    CHECK(coordinateSpaceTagsCompatible(fiber, scan));
    CHECK(coordinateSpaceTagsCompatible(fiber, fiber));
    CHECK(coordinateSpaceTagsCompatible(scan, scan));
    CHECK_FALSE(coordinateSpaceTagsCompatible(scan, "PHerc0139/scan@L1"));
    CHECK_FALSE(coordinateSpaceTagsCompatible(scan, "PHerc0139/other@L0"));
    CHECK(coordinateSpaceTag({}).empty());
}

TEST_CASE("Volume aliases use semantic tags, not long filenames")
{
    CHECK(vc3d::volumeDisplayAlias({"vc-lasagna-group:presence"}) == "fiber");
    CHECK(vc3d::volumeDisplayAlias({"vc-lasagna-group:nx"}).isEmpty());
    CHECK(vc3d::volumeDisplayAlias({"surface-prediction"}) == "surf");
    CHECK(vc3d::volumeDisplayAlias({"vc-open-data-preferred-source"}) == "scan");
    CHECK(vc3d::volumeDisplayAlias({"vc-open-data-virtual-source"}) == "scan");
    CHECK(vc3d::volumeDisplayAlias({"prediction", "normal3d"}).isEmpty());
    CHECK(vc3d::volumeDisplayAlias({}).isEmpty());
}

namespace {

struct TestAutosaveRoot {
    TestAutosaveRoot()
        : previous(VolumePkg::autosaveRoot())
        , root(std::filesystem::temp_directory_path() /
               ("vc_open_data_manifest_autosave_" +
                std::to_string(vc::memmap::pid())))
    {
        std::filesystem::remove_all(root);
        VolumePkg::setAutosaveRoot(root);
    }

    ~TestAutosaveRoot()
    {
        VolumePkg::setAutosaveRoot(previous);
        std::filesystem::remove_all(root);
    }

    std::filesystem::path previous;
    std::filesystem::path root;
};

TestAutosaveRoot testAutosaveRoot;

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

TEST_CASE("OpenDataManifest parses Lasagna as a distinct typed artifact")
{
    const auto manifest = parseOpenDataManifest(R"({
      "metadata":{"samples":{"sample":{"volumes":{"volume":{
        "properties":{"shape":[100,200,300]},
        "data":[{
          "type":"lasagna",
          "parameters":{"model_id":"model-7","level":2},
          "creation_info":{"lasagna_version":2,"source_to_base":1.0},
          "origins":[{"path":"lasagna/volume/","access_roots":[{
            "type":"https","url":"https://example.test/","usage":"public-read"
          }]}]
        }]
      }}}}}
    })");
    const auto& sample = manifest.samples.front();
    const auto& volume = sample.volumes.front();
    REQUIRE(volume.shapeZYX.has_value());
    CHECK(*volume.shapeZYX == std::array<std::size_t, 3>{100, 200, 300});

    const auto artifacts = lasagnaArtifacts(sample.id, volume);
    REQUIRE(artifacts.size() == 1);
    CHECK(artifacts.front().volumeId == "volume");
    CHECK(artifacts.front().artifactIndex == 0);
    CHECK(artifacts.front().artifactUrl == "https://example.test/lasagna/volume");
    CHECK(artifacts.front().modelId == "model-7");
    CHECK(artifacts.front().sourceCoordinateLevel == 2);
    CHECK(artifacts.front().levelWasExplicit);
    CHECK(artifacts.front().lasagnaVersion == 2);
    CHECK(artifacts.front().sourceToBase == doctest::Approx(1.0));
    CHECK(artifacts.front().baseShapeZYX == volume.shapeZYX);

    const auto inner = vc::lasagna::LasagnaDatasetManifest::parseText(R"({
      "version":2,
      "source_to_base":1.0,
      "base_shape_zyx":[100,200,300],
      "groups":{
        "grad_mag":{"zarr":"grad_mag.zarr","channels":["grad_mag"]},
        "nx":{"zarr":"nx.zarr","channels":["nx"]},
        "ny":{"zarr":"ny.zarr","channels":["ny"]}
      }
    })");
    CHECK(validateOpenDataLasagnaManifest(artifacts.front(), inner) ==
          OpenDataLasagnaDatasetKind::Normal);

    auto missingLevelVolume = volume;
    missingLevelVolume.artifacts.front().levelParameterPresent = false;
    missingLevelVolume.artifacts.front().sourceCoordinateLevel.reset();
    CHECK(lasagnaArtifacts(sample.id, missingLevelVolume).empty());
}

TEST_CASE("Open-data fiber Lasagna accepts authoritative inner metadata")
{
    const auto manifest = parseOpenDataManifest(R"({
      "metadata":{"samples":{"PHercParis4":{"volumes":{"20260411134726":{
        "properties":{"shape":[75784,32693,32693]},
        "data":[{
          "type":"lasagna",
          "parameters":{"model_id":"20260801084232","level":1},
          "creation_info":null,
          "origins":[{"path":"PHercParis4/representations/predictions/fibers/run/",
            "access_roots":[{"type":"s3","url":"s3://vesuvius-challenge-open-data/",
              "usage":"public-read"}]}]
        }]
      }}}}}
    })");
    const auto& volume = manifest.samples.front().volumes.front();
    const auto infos = lasagnaArtifacts("PHercParis4", volume);
    REQUIRE(infos.size() == 1);
    CHECK_FALSE(infos.front().lasagnaVersionPresent);
    CHECK_FALSE(infos.front().sourceToBasePresent);

    const auto inner = vc::lasagna::LasagnaDatasetManifest::parseText(R"({
      "version":2,
      "source_to_base":1.0,
      "base_shape_zyx":[75784,32694,32694],
      "groups":{
        "presence":{"zarr":"presence.zarr/3","scaledown":3,"channels":["presence"]},
        "nx":{"zarr":"nx.zarr/3","scaledown":3,"channels":["nx"]},
        "ny":{"zarr":"ny.zarr/3","scaledown":3,"channels":["ny"]}
      }
    })");
    CHECK(validateOpenDataLasagnaManifest(infos.front(), inner) ==
          OpenDataLasagnaDatasetKind::FiberInference);
}

TEST_CASE("Open-data Lasagna still rejects incompatible inner metadata")
{
    OpenDataLasagnaInfo info;
    info.levelWasExplicit = true;
    info.sourceCoordinateLevel = 1;
    info.baseShapeZYX = std::array<std::size_t, 3>{100, 200, 300};

    const auto wrongShape = vc::lasagna::LasagnaDatasetManifest::parseText(R"({
      "version":2,
      "source_to_base":1.0,
      "base_shape_zyx":[100,202,300],
      "groups":{
        "presence":{"zarr":"presence.zarr","channels":["presence"]},
        "nx":{"zarr":"nx.zarr","channels":["nx"]},
        "ny":{"zarr":"ny.zarr","channels":["ny"]}
      }
    })");
    CHECK_THROWS_WITH_AS(
        validateOpenDataLasagnaManifest(info, wrongShape),
        doctest::Contains("base_shape_zyx"),
        std::runtime_error);

    info.baseShapeZYX = std::array<std::size_t, 3>{100, 202, 300};
    info.lasagnaVersionPresent = true;
    CHECK_THROWS_WITH_AS(
        validateOpenDataLasagnaManifest(info, wrongShape),
        doctest::Contains("lasagna_version"),
        std::runtime_error);
}

TEST_CASE("Open-data Lasagna attachment handles level artifacts independently")
{
    OpenDataSample sample;
    sample.id = "sample";
    OpenDataVolume volume;
    volume.id = "volume";
    for (int level : {1, 3}) {
        OpenDataArtifact artifact;
        artifact.type = "lasagna";
        artifact.resolvedUrl =
            "https://example.test/lasagna/L" + std::to_string(level);
        artifact.levelParameterPresent = true;
        artifact.sourceCoordinateLevel = level;
        volume.artifacts.push_back(std::move(artifact));
    }
    sample.volumes.push_back(std::move(volume));

    const auto infos =
        lasagnaArtifacts(sample.id, sample.volumes.front());
    REQUIRE(infos.size() == 2);
    CHECK(infos[0].artifactIndex == 0);
    CHECK(infos[0].sourceCoordinateLevel == 1);
    CHECK(infos[1].artifactIndex == 1);
    CHECK(infos[1].sourceCoordinateLevel == 3);

    auto pkg = VolumePkg::newEmpty();
    OpenDataResourceSelection selection;
    selection.representations =
        std::vector<OpenDataRepresentationRef>{{0, 1,
                                                OpenDataRepresentationKind::Lasagna}};
    std::vector<std::string> messages;
    CHECK(attachOpenDataLasagna(
              *pkg, sample, {}, &messages, &selection) == 0);
    REQUIRE(messages.size() == 1);
    CHECK(messages.front().find("coordinate level L3") != std::string::npos);
    CHECK(messages.front().find("ambiguous") == std::string::npos);
}

TEST_CASE("OpenDataManifest enumerates derived volume representations")
{
    const auto manifest = parseOpenDataManifest(R"({
      "metadata":{"samples":{"sample":{"volumes":{"volume":{"data":[
        {"type":"zarr"},
        {"type":"normal-grids"},
        {"type":"lasagna"},
        {"type":"surface-prediction-zarr"},
        {"type":"ink_detection_3d_zarr"},
        {"type":"future-pred-zarr"}
      ]}}}}}
    })");
    const auto representations = derivedRepresentations(manifest.samples.front());
    REQUIRE(representations.size() == 5);
    CHECK(representations[0].kind == OpenDataRepresentationKind::NormalGrids);
    CHECK(representations[1].kind == OpenDataRepresentationKind::Lasagna);
    CHECK(representations[2].kind == OpenDataRepresentationKind::Prediction);
    CHECK(representations[3].kind == OpenDataRepresentationKind::Prediction);
    CHECK(representations[4].kind == OpenDataRepresentationKind::Prediction);
    CHECK(representationKindName(representations[0].kind) == "Normal grids");
    CHECK(representationKindName(representations[1].kind) == "Lasagna");
}

TEST_CASE("Open-data Lasagna resolution requires exact parent volume and dyadic level")
{
    auto pkg = VolumePkg::newEmpty();
    REQUIRE(pkg->addLasagnaDatasetEntry(
        "/cache/sample/vol-a/data.lasagna.json",
        {"vc-open-data-lasagna",
         "vc-open-data-sample-id:sample",
         "vc-open-data-volume-id:vol-a",
         "vc-open-data-source-coordinate-level:3",
         "vc-open-data-lasagna-artifact:https://example.test/vol-a"}));

    const auto resolved = resolveLasagnaForCoordinateTags(
        *pkg,
        {"vc-open-data-sample-id:sample",
         "vc-open-data-volume-id:vol-a",
         "vc-open-data-source-coordinate-level:3",
         "vc-open-data-lasagna-artifact:https://example.test/vol-a"});
    REQUIRE(resolved.has_value());
    CHECK(resolved->manifestBacked);
    CHECK(resolved->workingToBaseScale == doctest::Approx(8.0));
    CHECK(resolved->coordinateSpace == "sample/vol-a@L3");
    CHECK(resolved->sourceManifestLocation ==
          "https://example.test/vol-a/data.lasagna.json");

    const auto native = resolveLasagnaForCoordinateTags(
        *pkg,
        {"vc-open-data-sample-id:sample",
         "vc-open-data-volume-id:vol-a",
         "vc-open-data-source-coordinate-level:0",
         "vc-open-data-lasagna-artifact:https://example.test/vol-a"});
    REQUIRE(native.has_value());
    CHECK(native->workingToBaseScale == doctest::Approx(1.0));
    CHECK(native->coordinateSpace == "sample/vol-a@L0");

    CHECK_THROWS(resolveLasagnaForCoordinateTags(
        *pkg,
        {"vc-open-data-sample-id:sample",
         "vc-open-data-volume-id:vol-a",
         "vc-open-data-source-coordinate-level:2",
         "vc-open-data-lasagna-artifact:https://example.test/vol-a"}));
    CHECK_THROWS(resolveLasagnaForCoordinateTags(
        *pkg,
        {"vc-open-data-sample-id:sample",
         "vc-open-data-volume-id:vol-b",
         "vc-open-data-source-coordinate-level:0",
         "vc-open-data-lasagna-artifact:https://example.test/vol-b"}));
    CHECK_THROWS(resolveLasagnaForCoordinateTags(
        *pkg,
        {"vc-open-data-sample-id:sample",
         "vc-open-data-volume-id:vol-a",
         "vc-open-data-source-coordinate-level:6",
         "vc-open-data-lasagna-artifact:https://example.test/vol-a"}));
}

TEST_CASE("Manual project Lasagna resolution materializes a selected remote manifest")
{
    const auto root = std::filesystem::temp_directory_path() /
        ("vc_open_data_remote_manual_" + std::to_string(vc::memmap::pid()));
    std::filesystem::remove_all(root);
    const std::string location =
        "https://example.test/manual/data.lasagna.json";
    int fetches = 0;
    vc::lasagna::LasagnaDatasetOpenOptions openOptions;
    openOptions.remoteCacheRoot = root;
    openOptions.remoteFileFetcher = [&](const std::string&, const std::filesystem::path& temporary) {
        ++fetches;
        std::ofstream(temporary) << R"({"version":2,"groups":{
            "nx":{"zarr":"nx.ome.zarr/4","scaledown":4,"channels":["nx"]}
        }})";
    };
    const auto cached = vc::lasagna::LasagnaDataset::openLocation(
        location, openOptions);

    const QByteArray previousConfigDir = qgetenv("VC3D_CONFIG_DIR");
    qputenv("VC3D_CONFIG_DIR", QByteArray::fromStdString((root / "settings").string()));
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(
        vc3d::settings::viewer::REMOTE_CACHE_DIR,
        QString::fromStdString(root.string()));
    settings.sync();

    auto pkg = VolumePkg::newEmpty();
    REQUIRE(pkg->addLasagnaDatasetEntry(location));
    pkg->setSelectedLasagnaDataset(location);
    const auto resolved = resolveLasagnaForCoordinateTags(*pkg, {});
    REQUIRE(resolved.has_value());
    CHECK(resolved->manifestPath == cached.manifest().manifestPath);
    CHECK(resolved->sourceManifestLocation == location);
    // Alignment metrics must retain this remote dataset, not reopen its cached
    // manifest as a local file and interpret the relative channel as local.
    openOptions.workingToBaseScale = resolved->workingToBaseScale;
    const auto metricsDataset = std::make_shared<vc::lasagna::LasagnaDataset>(
        vc::lasagna::LasagnaDataset::openLocation(
            resolved->sourceManifestLocation, openOptions));
    REQUIRE(metricsDataset->manifest().groups.size() == 1);
    const auto& channel = metricsDataset->manifest().groups.front();
    CHECK(channel.isRemote());
    CHECK(channel.remoteZarrBaseUrl == "https://example.test/manual");
    CHECK(channel.remoteZarrKey == "nx.ome.zarr/4");
    CHECK(metricsDataset->manifest().workingToBaseScale == resolved->workingToBaseScale);
    CHECK(fetches == 1);
    qputenv("VC3D_CONFIG_DIR", previousConfigDir);
    std::filesystem::remove_all(root);
}

TEST_CASE("Open-data Lasagna derives and verifies coordinate scale from volume shape")
{
    const auto root = std::filesystem::temp_directory_path() /
        ("vc_open_data_lasagna_shape_" + std::to_string(vc::memmap::pid()));
    std::filesystem::remove_all(root);
    const auto manifestPath = root / "data.lasagna.json";
    writeFile(manifestPath, R"({
      "version":2,
      "source_to_base":1,
      "base_shape_zyx":[100,200,300],
      "groups":{}
    })");

    auto pkg = VolumePkg::newEmpty();
    REQUIRE(pkg->addLasagnaDatasetEntry(
        manifestPath.string(),
        {"vc-open-data-lasagna",
         "vc-open-data-sample-id:sample",
         "vc-open-data-volume-id:vol-a",
         "vc-open-data-source-coordinate-level:3"}));
    const std::vector<std::string> tags{
        "vc-open-data-sample-id:sample",
        "vc-open-data-volume-id:vol-a",
        "vc-open-data-source-coordinate-level:3"};
    const auto resolved = resolveLasagnaForCoordinateShape(
        *pkg, tags, std::array<int, 3>{13, 25, 38});
    REQUIRE(resolved.has_value());
    CHECK(resolved->workingToBaseScale == doctest::Approx(8.0));

    auto inconsistentTags = tags;
    inconsistentTags.back() = "vc-open-data-source-coordinate-level:2";
    CHECK_THROWS(resolveLasagnaForCoordinateShape(
        *pkg, inconsistentTags, std::array<int, 3>{13, 25, 38}));
    std::filesystem::remove_all(root);
}

TEST_CASE("Open-data Lasagna alone requests its declared rebased source view")
{
    OpenDataSample sample;
    sample.id = "sample";
    OpenDataVolume volume;
    volume.id = "high-resolution";
    volume.suffix = "2.4um";
    volume.pixelSizeUm = 2.4;

    OpenDataArtifact source;
    source.type = "ome-zarr";
    source.resolvedUrl = "http://127.0.0.1:9/source.zarr";
    volume.artifacts.push_back(source);

    OpenDataArtifact lasagna;
    lasagna.type = "lasagna";
    lasagna.resolvedUrl = "http://127.0.0.1:9/lasagna";
    lasagna.levelParameterPresent = true;
    lasagna.sourceCoordinateLevel = 2;
    volume.artifacts.push_back(lasagna);
    sample.volumes.push_back(volume);

    vc::project::LoadOptions options;
    options.deferResolution = true;
    auto pkg = VolumePkg::newEmpty(options);
    const auto result = attachOpenDataSampleVolumes(*pkg, sample);

    CHECK(result.attachedVolumeEntries == 2);
    REQUIRE(pkg->volumeEntries().size() == 2);
    CHECK(pkg->volumeEntries()[0].location ==
          "http://127.0.0.1:9/source.zarr");
    CHECK(pkg->volumeEntries()[1].location ==
          "http://127.0.0.1:9/source.zarr#vc-base-scale=2");
    CHECK(vc::project::usesAnonymousRemoteAuth(pkg->volumeEntries()[0]));
    CHECK(vc::project::usesAnonymousRemoteAuth(pkg->volumeEntries()[1]));
    CHECK(std::find(pkg->volumeEntries()[1].tags.begin(),
                    pkg->volumeEntries()[1].tags.end(),
                    "vc-open-data-coordinate-space:sample/high-resolution@L2") !=
          pkg->volumeEntries()[1].tags.end());
    CHECK(std::find(pkg->volumeEntries()[1].tags.begin(),
                    pkg->volumeEntries()[1].tags.end(),
                    "vc-open-data-lasagna-artifact:http://127.0.0.1:9/lasagna") !=
          pkg->volumeEntries()[1].tags.end());
}

TEST_CASE("Selected Lasagna does not implicitly attach scan views")
{
    OpenDataSample sample;
    sample.id = "sample";
    OpenDataVolume volume;
    volume.id = "v";
    volume.pixelSizeUm = 2.4;
    OpenDataArtifact source, lasagna;
    source.type = "ome-zarr";
    source.resolvedUrl = "http://127.0.0.1:9/source.zarr";
    lasagna.type = "lasagna";
    lasagna.resolvedUrl = "http://127.0.0.1:9/lasagna";
    lasagna.levelParameterPresent = true;
    lasagna.sourceCoordinateLevel = 2;
    volume.artifacts = {source, lasagna};
    sample.volumes = {volume};
    OpenDataResourceSelection selection;
    selection.rawVolumeIds.emplace();
    selection.representations = derivedRepresentations(sample);
    vc::project::LoadOptions options;
    options.deferResolution = true;
    auto pkg = VolumePkg::newDetached(options);
    attachOpenDataSampleVolumes(*pkg, sample, &selection);
    CHECK(pkg->volumeEntries().empty());

    selection.rawVolumeIds->push_back(volume.id);
    auto withScan = VolumePkg::newDetached(options);
    attachOpenDataSampleVolumes(*withScan, sample, &selection);
    REQUIRE(withScan->volumeEntries().size() == 1);
    CHECK(withScan->volumeEntries().front().location == source.resolvedUrl);
    REQUIRE(withScan->addLasagnaDatasetEntry("/cache/selected.lasagna.json", {
        "vc-open-data-lasagna", "vc-open-data-sample-id:sample",
        "vc-open-data-volume-id:v", "vc-open-data-source-coordinate-level:2"}));
    const auto resolved = resolveLasagnaForCoordinateTags(
        *withScan, withScan->volumeEntries().front().tags);
    REQUIRE(resolved.has_value());
    CHECK(resolved->workingToBaseScale == doctest::Approx(1.0));
    CHECK(resolved->manifestPath == std::filesystem::path("/cache/selected.lasagna.json"));
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
    for (const auto& entry : pkg->volumeEntries()) {
        CHECK(vc::project::usesAnonymousRemoteAuth(entry));
    }

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

TEST_CASE("Project creation defaults to the shared autosave directory")
{
    const auto root = std::filesystem::current_path() /
        ("project_defaults_test_" + std::to_string(vc::memmap::pid()));
    const auto previous = VolumePkg::autosaveRoot();
    VolumePkg::setAutosaveRoot(root / "vc3d");
    {
        QSettings settings(QString::fromStdString((root / "settings.ini").string()), QSettings::IniFormat);
        settings.setValue(vc3d::settings::project::DEFAULT_PATH, "different-directory");
        CHECK(vc3d::defaultNewProjectDirectory(settings).toStdString() == (root / "vc3d").string());
        CHECK(std::filesystem::is_directory(root / "vc3d"));
    }
    VolumePkg::setAutosaveRoot(previous);
    std::filesystem::remove_all(root);
}

TEST_CASE("Catalog source selection is independent of derived representations")
{
    OpenDataSample sample;
    sample.id = "selection";
    OpenDataVolume volume;
    volume.id = "v";
    volume.dataFormat = "zarr";
    OpenDataArtifact source, prediction;
    source.type = "ome-zarr";
    source.resolvedUrl = "http://127.0.0.1:9/base.zarr";
    prediction.type = "surface-prediction-zarr";
    prediction.resolvedUrl = "http://127.0.0.1:9/prediction.zarr";
    volume.artifacts = {source, prediction};
    sample.volumes = {volume};
    vc::project::LoadOptions options;
    options.deferResolution = true;
    OpenDataResourceSelection selection;
    selection.rawVolumeIds.emplace();
    selection.representations.emplace();
    auto empty = VolumePkg::newEmpty(options);
    attachOpenDataSampleVolumes(*empty, sample, &selection);
    CHECK(empty->volumeEntries().empty());

    selection.representations = derivedRepresentations(sample);
    auto derived = VolumePkg::newEmpty(options);
    attachOpenDataSampleVolumes(*derived, sample, &selection);
    REQUIRE(derived->volumeEntries().size() == 1);
    CHECK(derived->volumeEntries().front().location == prediction.resolvedUrl);

    selection.rawVolumeIds = std::vector<std::string>{"v"};
    selection.representations->clear();
    auto raw = VolumePkg::newEmpty(options);
    attachOpenDataSampleVolumes(*raw, sample, &selection);
    REQUIRE(raw->volumeEntries().size() == 1);
    CHECK(raw->volumeEntries().front().location == source.resolvedUrl);
}

TEST_CASE("Single catalog prediction attachment retains identity and validates its source level")
{
    OpenDataSample sample;
    sample.id = "sample";
    OpenDataVolume volume;
    volume.id = "scan";
    volume.pixelSizeUm = 2.4;
    OpenDataArtifact source, prediction;
    source.type = "ome-zarr";
    source.resolvedUrl = "http://127.0.0.1:9/source.zarr";
    prediction.type = "surface-prediction-zarr";
    prediction.resolvedUrl = "http://127.0.0.1:9/prediction.zarr";
    prediction.levelParameterPresent = true;
    prediction.sourceCoordinateLevel = 2;
    volume.artifacts = {source, prediction};
    sample.volumes = {volume};
    OpenDataResourceSelection selection;
    selection.rawVolumeIds.emplace();
    selection.representations = std::vector<OpenDataRepresentationRef>{
        {0, 1, OpenDataRepresentationKind::Prediction}};
    vc::project::LoadOptions options;
    options.deferResolution = true;
    auto pkg = VolumePkg::newDetached(options);
    const auto result = attachOpenDataSampleVolumes(*pkg, sample, &selection);
    REQUIRE(pkg->volumeEntries().size() == 1);
    const auto& entry = pkg->volumeEntries().front();
    CHECK(entry.location == prediction.resolvedUrl);
    for (const auto* tag : {"surface-prediction", "vc-open-data-sample-id:sample",
                           "vc-open-data-volume-id:scan"})
        CHECK(std::find(entry.tags.begin(), entry.tags.end(), tag) != entry.tags.end());
    CHECK(vc::project::usesAnonymousRemoteAuth(entry));
    // The source need not be attached, but its descriptor must be checked before
    // assigning a coordinate frame. A failed check is reported, not silently L0.
    CHECK(result.failedVolumes == 1);
    CHECK(std::any_of(result.messages.begin(), result.messages.end(), [](const auto& message) {
        return message.find("against source L2:") != std::string::npos;
    }));
    CHECK(std::none_of(entry.tags.begin(), entry.tags.end(), [](const auto& tag) {
        return tag.starts_with("vc-open-data-coordinate-space:");
    }));
}

TEST_CASE("Selective catalog projects isolate cached projects and individual segments")
{
    auto manifest = parseOpenDataManifest(kFixture);
    auto sample = *manifest.findSample("PHerc0139");
    const auto root = std::filesystem::current_path() /
        ("selective_catalog_test_" + std::to_string(vc::memmap::pid()));
    std::filesystem::remove_all(root);
    const auto oldAutosave = VolumePkg::autosaveRoot();
    VolumePkg::setAutosaveRoot(root / "autosave");
    // Prepare the full catalog cache first, including an unrelated segment.
    auto extra = sample.segments.front();
    extra.id = "unselected-segment";
    extra.longId = "unselected-segment";
    sample.segments.push_back(extra);
    vc::project::LoadOptions options;
    options.deferResolution = true;
    auto full = VolumePkg::newEmpty(options);
    full->setSelectedLasagnaDataset("full-project-marker");
    const auto cachedPath = root / "open_data/projects/PHerc0139.volpkg.json";
    full->save(cachedPath);
    const auto autosaveBefore = [&]() {
        std::ifstream stream(VolumePkg::autosaveRoot() / "current_project.json");
        return nlohmann::json::parse(stream);
    }();

    OpenDataResourceSelection selection;
    selection.rawVolumeIds.emplace();
    selection.representations.emplace();
    selection.segmentIds.emplace();
    OpenDataNewProject destination{root / "empty.volpkg.json", "Empty chosen project"};
    auto empty = createOpenDataSampleProject(sample, root, nullptr, {}, &selection, &destination);
    CHECK(empty->volumeEntries().empty());
    CHECK_FALSE(std::filesystem::exists(surfaceStorePath(root, sample.id)));
    CHECK(empty->selectedLasagnaDataset().empty());
    CHECK(empty->name() == destination.name);
    CHECK(empty->path() == destination.path);
    CHECK_FALSE(std::filesystem::exists(root / "fibers"));

    selection.segmentIds->push_back(sample.segments.front().id);
    destination = {root / "selected.volpkg.json", "Only selected"};
    auto selected = createOpenDataSampleProject(sample, root, nullptr, {}, &selection, &destination);
    CHECK(selected->volumeEntries().empty());
    // Only the selected segment is recorded in the sample's surface store.
    CHECK(surfaceStateForSegment(root, sample, sample.segments.front()) == SurfaceState::Remote);
    CHECK(surfaceStateForSegment(root, sample, extra) == SurfaceState::Missing);
    auto reloaded = VolumePkg::load(destination.path, options);
    CHECK(reloaded->name() == destination.name);
    auto cached = VolumePkg::load(cachedPath, options);
    CHECK(cached->selectedLasagnaDataset() == "full-project-marker");
    OpenDataNewProject invalid{root, "Cannot save onto directory"};
    CHECK_THROWS(createOpenDataSampleProject(sample, root, nullptr, {}, &selection, &invalid));
    std::ifstream autosaveStream(VolumePkg::autosaveRoot() / "current_project.json");
    CHECK(nlohmann::json::parse(autosaveStream) == autosaveBefore);
    autosaveStream.close();
    VolumePkg::setAutosaveRoot(oldAutosave);
    std::filesystem::remove_all(root);
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


TEST_CASE("OpenDataSampleProject saves and reuses cached volpkg json")
{
    OpenDataSample sample;
    sample.id = "Sample With Spaces";

    const auto cacheRoot = std::filesystem::temp_directory_path() /
                           ("vc_open_data_cached_project_test_" + std::to_string(vc::memmap::pid()));
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

namespace {

// A sample with a source volume, a published target and a third volume reached
// by the sample's volume matrix; one segment authored on the source at 2x
// downscale with one ink detection.
OpenDataSample representationSample()
{
    OpenDataSample sample;
    sample.id = "sample";
    for (const char* id : {"source", "target", "third"}) {
        OpenDataVolume volume;
        volume.id = id;
        sample.volumes.push_back(volume);
    }
    sample.properties["volume_transforms"] = nlohmann::json::array({{
        {"from_volume_id", "source"},
        {"transforms", nlohmann::json::array({{
            {"to_volume_id", "third"},
            {"matrix", {{1, 0, 0, 10}, {0, 1, 0, 20}, {0, 0, 1, 30}}},
        }})},
    }});
    OpenDataSegment segment;
    segment.id = "segment";
    segment.longId = "sample-segment";
    segment.suffix = "w001";
    segment.originalVolumeId = "source";
    segment.properties["original_volume_downscale"] = 2;
    OpenDataArtifact authored;
    authored.type = "tifxyz-flattened";
    authored.resolvedUrl = "https://example.test/authored";
    segment.artifacts.push_back(authored);
    OpenDataArtifact published;
    published.type = "tifxyz-transformed";
    published.resolvedUrl = "https://example.test/published";
    published.targetVolumeId = "target";
    segment.artifacts.push_back(published);
    OpenDataArtifact ink;
    ink.type = "ink_detection";
    ink.resolvedUrl = "https://example.test/ink.png";
    segment.artifacts.push_back(ink);
    sample.segments.push_back(segment);
    return sample;
}

std::filesystem::path freshRoot(const std::string& tag)
{
    const auto root = std::filesystem::temp_directory_path() /
        ("vc_open_data_store_" + tag + "_" + std::to_string(vc::memmap::pid()));
    std::filesystem::remove_all(root);
    return root;
}

} // namespace

TEST_CASE("Open Data segments are canonical, published and generated representations")
{
    auto sample = representationSample();
    auto& segment = sample.segments.front();
    const auto representations = classifyOpenDataSegmentRepresentations(sample, segment);
    REQUIRE(representations.size() == 3);
    CHECK(representations[0].kind == OpenDataSegmentRepresentationKind::DerivedNative);
    CHECK(representations[0].canonicalSource);
    CHECK(representations[0].coordinateSpace == "sample/source@L0");
    CHECK(representations[0].artifact == &segment.artifacts[0]);
    CHECK(representations[1].kind == OpenDataSegmentRepresentationKind::PublishedTransformed);
    CHECK(representations[1].coordinateVolumeId == "target");
    CHECK(representations[2].kind == OpenDataSegmentRepresentationKind::Generated);
    CHECK(representations[2].coordinateVolumeId == "third");
    CHECK(representations[2].sourceRepresentationId == representations[0].representationId);
    REQUIRE(representations[2].matrix);
    CHECK((*representations[2].matrix)(0, 3) == doctest::Approx(10));

    // A published representation of the source volume wins over the authored one.
    OpenDataArtifact publishedSource = segment.artifacts[1];
    publishedSource.resolvedUrl = "https://example.test/published-source";
    publishedSource.targetVolumeId = "source";
    segment.artifacts.push_back(publishedSource);
    const auto withPublishedSource = classifyOpenDataSegmentRepresentations(sample, segment);
    REQUIRE_FALSE(withPublishedSource.empty());
    CHECK(withPublishedSource[0].kind == OpenDataSegmentRepresentationKind::PublishedTransformed);
    CHECK(withPublishedSource[0].canonicalSource);
    CHECK(withPublishedSource[0].artifact == &segment.artifacts.back());

    // Artifacts targeting volumes outside the sample are ignored.
    OpenDataArtifact elsewhere = segment.artifacts[1];
    elsewhere.targetVolumeId = "missing";
    segment.artifacts.push_back(elsewhere);
    CHECK(classifyOpenDataSegmentRepresentations(sample, segment).size() == withPublishedSource.size());
}

TEST_CASE("Open Data import records representations as catalog surfaces bound to their volumes")
{
    namespace vo = vc::objects;
    const auto sample = representationSample();
    const auto root = freshRoot("import");
    const auto imported = importSampleSurfaces(root, sample);
    CHECK(imported.representations == 3);
    CHECK(imported.added == 3);
    const auto store = surfaceStorePath(root, "sample");
    REQUIRE(std::filesystem::exists(store));

    const auto representations = classifyOpenDataSegmentRepresentations(sample, sample.segments.front());
    const auto native = surfaceUuidForRepresentation(representations[0].representationId);
    const auto published = surfaceUuidForRepresentation(representations[1].representationId);
    const auto generated = surfaceUuidForRepresentation(representations[2].representationId);
    CHECK(native == surfaceUuidForRepresentation(representations[0].representationId));
    {
        vo::OpenOptions options;
        options.readOnly = true;
        auto db = vo::ObjectStore::open(store, options);
        auto read = db.read();
        const auto nativeRecord = db.objectByUuid(read, native);
        REQUIRE(nativeRecord);
        CHECK(nativeRecord->name == "w001");
        CHECK(nativeRecord->storageKind == "url");
        CHECK(nativeRecord->storageRef == "https://example.test/authored");
        const auto nativeMeta = nlohmann::json::parse(nativeRecord->meta);
        // Authored coordinates are half the source volume's voxels.
        CHECK(vo::geometryScaleToVolume(nativeMeta, "source") == doctest::Approx(2.0));
        CHECK(nativeMeta.at("vc_open_data_coordinate_space") == "sample/source@L0");
        const auto ink = inkDetections(nativeMeta, "/artifacts");
        REQUIRE(ink.size() == 1);
        CHECK(ink.front().url == "https://example.test/ink.png");
        CHECK(ink.front().file.parent_path() == "/artifacts/ink-detections");

        const auto publishedMeta = nlohmann::json::parse(db.objectByUuid(read, published)->meta);
        CHECK(vo::geometryScaleToVolume(publishedMeta, "target") == doctest::Approx(1.0));
        const auto generatedRecord = db.objectByUuid(read, generated);
        REQUIRE(generatedRecord);
        CHECK(generatedRecord->storageKind != "url");
        const auto generate = nlohmann::json::parse(generatedRecord->meta).at("open_data").at("generate");
        CHECK(generate.at("source_surface") == native);
        CHECK(generate.at("target_volume_id") == "third");
        CHECK_FALSE(db.currentGeometry(read, native));
    }
    CHECK(surfaceStateForSegment(root, sample, sample.segments.front()) == SurfaceState::Remote);

    // Re-importing an unchanged manifest writes nothing.
    const auto again = importSampleSurfaces(root, sample);
    CHECK(again.added == 0);
    CHECK(again.updated == 0);
    std::filesystem::remove_all(root);
}

TEST_CASE("Generated Open Data surfaces are computed from their source when first opened")
{
    namespace vo = vc::objects;
    const auto sample = representationSample();
    const auto root = freshRoot("generate");
    (void)importSampleSurfaces(root, sample);
    const auto store = surfaceStorePath(root, "sample");
    const auto representations = classifyOpenDataSegmentRepresentations(sample, sample.segments.front());
    const auto native = surfaceUuidForRepresentation(representations[0].representationId);
    const auto generated = surfaceUuidForRepresentation(representations[2].representationId);

    // The canonical source as the store holds it once fetched (authored at 2x downscale).
    {
        cv::Mat_<cv::Vec3f> points(2, 2);
        points.setTo(cv::Vec3f(1, 2, 3));
        QuadSurface authored(points, cv::Vec2f(1, 1));
        vo::OpenOptions options;
        options.author = "test";
        auto db = vo::ObjectStore::open(store, options);
        std::string revision;
        {
            auto read = db.read();
            revision = db.currentObjectRevision(read, native)->uuid;
        }
        auto txn = db.write({{native, revision}});
        db.putGeometry(txn, native, quad_surface_geometry(authored));
        txn.commit();
    }
    CHECK(surfaceStateForSegment(root, sample, sample.segments.front()) == SurfaceState::Stored);
    CHECK_FALSE(surfaceHasGeometry(store, generated));

    materializeSurface(store, generated);
    REQUIRE(surfaceHasGeometry(store, generated));
    {
        vo::OpenOptions options;
        options.readOnly = true;
        auto db = vo::ObjectStore::open(store, options);
        auto read = db.read();
        const auto surface = load_quad_from_store(db, read, generated);
        // Source L0 voxels (2x the authored coordinates) moved by the matrix.
        CHECK((*surface->rawPointsPtr())(0, 0)[0] == doctest::Approx(2 + 10));
        CHECK((*surface->rawPointsPtr())(0, 0)[2] == doctest::Approx(6 + 30));
        CHECK(vo::geometryScaleToVolume(nlohmann::json::parse(db.objectByUuid(read, generated)->meta), "third") ==
              doctest::Approx(1.0));
    }
    std::filesystem::remove_all(root);
}

TEST_CASE("Open Data volume tags name the catalog volume and level")
{
    const auto volume = catalogVolumeFromTags({"vc-open-data-sample-id:PHerc0139", "vc-open-data-volume-id:vol1",
                                               "vc-open-data-coordinate-space:PHerc0139/vol1@L2"});
    REQUIRE(volume);
    CHECK(volume->sampleId == "PHerc0139");
    CHECK(volume->volumeId == "vol1");
    CHECK(volume->level == 2);
    CHECK_FALSE(catalogVolumeFromTags({"vc-open-data-volume-id:vol1"}));
}
