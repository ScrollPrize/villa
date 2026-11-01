#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <vc_reader/zarr.hpp>
#include "vc/core/util/NormalGridVolume.hpp"

#include "OpenDataManifest.hpp"
#include "OpenDataLasagna.hpp"
#include "OpenDataNormalGrids.hpp"
#include "OpenDataSampleProject.hpp"
#include "ProjectCreationDefaults.hpp"
#include "VCSettings.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "support/LoopbackHttpServer.hpp"
#include "support/TestProject.hpp"
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

namespace {


constexpr const char* kFixture = R"({
  "samples": {
    "PHerc0139": {
      "sample": {
        "type": "scroll",
        "description": "Example sample",
        "unknown_sample_field": 7,
        "data": [
          {
            "type": "photo",
            "origins": [
              {
                "path": "PHerc0139/photos/PHerc0139_photo.jpg",
                "access_roots": [
                  {
                    "type": "s3",
                    "url": "s3://vesuvius-challenge-open-data/",
                    "usage": "public-read"
                  }
                ]
              }
            ]
          }
        ]
      },
      "scans": {
        "20240101000000": {
          "suffix": "scan-a",
          "properties": {
            "operator": "vc"
          },
          "creation": {
            "date": "2024-01-01T00:00:00Z"
          }
        }
      },
      "volumes": {
        "vol1": {
          "scan_id": "20240101000000",
          "suffix": "surface-volume",
          "data": [
            {
              "type": "zarr",
              "properties": {
                "representation": "normal3d"
              },
              "origins": [
                {
                  "path": "PHerc0139/volumes/vol1.zarr",
                  "access_roots": [
                    {
                      "type": "s3",
                      "url": "s3://private-root/",
                      "usage": "internal"
                    },
                    {
                      "type": "s3",
                      "url": "s3://vesuvius-challenge-open-data/",
                      "usage": "public-read"
                    }
                  ]
                }
              ]
            }
          ],
          "properties": {
            "pixel_size_um": 7.91,
            "energy_keV": 88.0,
            "detector_distance_mm": 12.5,
            "data_format": "zarr"
          }
        }
      },
      "segments": {
        "20260311000000": {
          "long_id": "PHerc0139-20260311000000",
          "suffix": "seg-a",
          "original_volume_id": "vol1",
          "creation": {
            "derived_from": {
              "type": "volume",
              "id": "vol1"
            },
            "date": "2026-03-11T00:00:00Z"
          },
          "properties": {
            "original_volume_downscale": 2,
            "width": 123,
            "height": 456
          },
          "data": [
            {
              "type": "tifxyz",
              "properties": {
                "extra": true
              },
              "origins": [
                {
                  "path": "PHerc0139/segments/20260311000000/tifxyz",
                  "access_roots": [
                    {
                      "type": "s3",
                      "url": "s3://vesuvius-challenge-open-data/",
                      "usage": "public-read"
                    }
                  ]
                }
              ]
            },
            {
              "type": "ink_detection",
              "origins": [
                {
                  "path": "PHerc0139/segments/20260311000000/ink.zarr",
                  "access_roots": [
                    {
                      "type": "https",
                      "url": "https://example.test/data/",
                      "usage": "public-read"
                    }
                  ]
                }
              ]
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
    "model-a": {
      "kind": "ink"
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

// A Lasagna cache as prepareOpenDataLasagna leaves it: the manifest and the
// remote marker naming its catalog artifact.
std::filesystem::path cachedLasagna(const std::filesystem::path& dir, const std::string& artifactUrl,
                                    const std::string& manifest = R"({"version":2,"groups":{}})")
{
    writeFile(dir / "data.lasagna.json", manifest);
    writeFile(dir / vc::lasagna::kLasagnaRemoteMarker,
              nlohmann::json{{"artifact_url", artifactUrl}, {"manifest_file", "data.lasagna.json"},
                             {"anonymous", true}}.dump());
    return dir / "data.lasagna.json";
}

// A catalog sample whose volumes publish a source zarr and Lasagna at the
// given levels (JSON body of "volumes").
std::shared_ptr<const vcr::catalog::OpenDataManifest> lasagnaCatalog(const std::string& volumes)
{
    return std::make_shared<const vcr::catalog::OpenDataManifest>(vcr::catalog::parseOpenDataManifest(
        R"({"samples": {"sample": {"volumes": )" + volumes + "}}}"));
}

std::string publishedArtifact(const std::string& type, const std::string& path, std::optional<int> level = std::nullopt)
{
    return R"({"type": ")" + type + '"' + (level ? R"(, "parameters": {"level": )" + std::to_string(*level) + "}" : "") +
           R"(, "origins": [{"path": ")" + path +
           R"(", "access_roots": [{"type": "https", "url": "https://example.test/", "usage": "public-read"}]}]})";
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
    fixture["samples"]["PHerc0139"]["segments"]["20260311000000"]["data"][0]["type"] = type;
    return fixture.dump();
}

} // namespace

TEST_CASE("OpenDataManifest parses Lasagna as a distinct typed artifact")
{
    const auto manifest = vcr::catalog::parseOpenDataManifest(R"({
  "samples": {
    "sample": {
      "volumes": {
        "volume": {
          "properties": {
            "shape": [
              100,
              200,
              300
            ]
          },
          "data": [
            {
              "type": "lasagna",
              "parameters": {
                "model_id": "model-7",
                "level": 2
              },
              "creation_info": {
                "lasagna_version": 2,
                "source_to_base": 1.0
              },
              "origins": [
                {
                  "path": "lasagna/volume/",
                  "access_roots": [
                    {
                      "type": "https",
                      "url": "https://example.test/",
                      "usage": "public-read"
                    }
                  ]
                }
              ]
            }
          ]
        }
      }
    }
  }
})");
    const auto& sample = manifest.samples.front();
    const auto& volume = sample.volumes.front();
    REQUIRE(volume.shapeZYX.has_value());
    CHECK(*volume.shapeZYX == std::array<std::size_t, 3>{100, 200, 300});

    REQUIRE(lasagnaArtifacts(volume) == std::vector<std::size_t>{0});
    const auto& artifact = volume.artifacts.front();
    CHECK(artifact.url() == "https://example.test/lasagna/volume");
    CHECK(artifact.modelId == "model-7");
    CHECK(artifact.sourceCoordinateLevel == 2);
    CHECK(artifact.lasagnaVersion == 2);
    CHECK(*artifact.sourceToBase == doctest::Approx(1.0));

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
    CHECK(validateOpenDataLasagnaManifest(volume, artifact, inner) ==
          OpenDataLasagnaDatasetKind::Normal);

    auto missingLevelVolume = volume;
    missingLevelVolume.artifacts.front().levelParameterPresent = false;
    missingLevelVolume.artifacts.front().sourceCoordinateLevel.reset();
    CHECK(lasagnaArtifacts(missingLevelVolume).empty());
}

TEST_CASE("Open-data fiber Lasagna accepts authoritative inner metadata")
{
    const auto manifest = vcr::catalog::parseOpenDataManifest(R"({
  "samples": {
    "PHercParis4": {
      "volumes": {
        "20260411134726": {
          "properties": {
            "shape": [
              75784,
              32693,
              32693
            ]
          },
          "data": [
            {
              "type": "lasagna",
              "parameters": {
                "model_id": "20260801084232",
                "level": 1
              },
              "creation_info": null,
              "origins": [
                {
                  "path": "PHercParis4/representations/predictions/fibers/run/",
                  "access_roots": [
                    {
                      "type": "s3",
                      "url": "s3://vesuvius-challenge-open-data/",
                      "usage": "public-read"
                    }
                  ]
                }
              ]
            }
          ]
        }
      }
    }
  }
})");
    const auto& volume = manifest.samples.front().volumes.front();
    REQUIRE(lasagnaArtifacts(volume).size() == 1);
    const auto& artifact = volume.artifacts.front();
    CHECK_FALSE(artifact.lasagnaVersionPresent);
    CHECK_FALSE(artifact.sourceToBasePresent);

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
    CHECK(validateOpenDataLasagnaManifest(volume, artifact, inner) ==
          OpenDataLasagnaDatasetKind::FiberInference);
}

TEST_CASE("Open-data Lasagna still rejects incompatible inner metadata")
{
    vcr::catalog::OpenDataVolume volume;
    volume.shapeZYX = std::array<std::size_t, 3>{100, 200, 300};
    vcr::catalog::OpenDataArtifact artifact;
    artifact.levelParameterPresent = true;
    artifact.sourceCoordinateLevel = 1;

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
        validateOpenDataLasagnaManifest(volume, artifact, wrongShape),
        doctest::Contains("base_shape_zyx"),
        std::runtime_error);

    volume.shapeZYX = std::array<std::size_t, 3>{100, 202, 300};
    artifact.lasagnaVersionPresent = true;
    CHECK_THROWS_WITH_AS(
        validateOpenDataLasagnaManifest(volume, artifact, wrongShape),
        doctest::Contains("lasagna_version"),
        std::runtime_error);
}

TEST_CASE("Open-data Lasagna attachment handles level artifacts independently")
{
    vcr::catalog::OpenDataSample sample;
    sample.id = "sample";
    vcr::catalog::OpenDataVolume volume;
    volume.id = "volume";
    for (int level : {1, 3}) {
        vcr::catalog::OpenDataArtifact artifact;
        artifact.type = "lasagna";
        artifact.resolvedUrl =
            "https://example.test/lasagna/L" + std::to_string(level);
        artifact.levelParameterPresent = true;
        artifact.sourceCoordinateLevel = level;
        volume.artifacts.push_back(std::move(artifact));
    }
    sample.volumes.push_back(std::move(volume));

    REQUIRE(lasagnaArtifacts(sample.volumes.front()) == std::vector<std::size_t>{0, 1});

    OpenDataResourceSelection selection;
    selection.representations =
        std::vector<vcr::catalog::OpenDataRepresentationRef>{{0, 1,
                                                vcr::catalog::OpenDataRepresentationKind::Lasagna}};
    const auto preparation = prepareOpenDataLasagna(sample, {}, &selection);
    CHECK(preparation.datasets.empty());
    CHECK(preparation.artifactUrls == std::vector<std::string>{"https://example.test/lasagna/L3"});
    REQUIRE(preparation.messages.size() == 1);
    CHECK(preparation.messages.front().find("coordinate level L3") != std::string::npos);
    CHECK(preparation.messages.front().find("ambiguous") == std::string::npos);
    auto pkg = vc::test::emptyProject();
    CHECK(attachOpenDataLasagna(*pkg, preparation) == 0);
}

TEST_CASE("Open-data Lasagna resolution requires exact parent volume and dyadic level")
{
    const auto root = std::filesystem::temp_directory_path() /
        ("vc_open_data_lasagna_pairing_" + std::to_string(vc::memmap::pid()));
    std::filesystem::remove_all(root);
    vc::project::LoadOptions options;
    options.catalog = lasagnaCatalog(
        R"({"vol-a": {"id": "vol-a", "data": [)" + publishedArtifact("zarr", "vol-a.zarr") + ", " +
        publishedArtifact("lasagna", "vol-a", 3) + R"(]}, "vol-b": {"id": "vol-b", "data": [)" +
        publishedArtifact("zarr", "vol-b.zarr") + ", " + publishedArtifact("lasagna", "vol-b", 0) + "]}}");
    auto pkg = vc::test::emptyProject(options);
    REQUIRE(pkg->addLasagnaDatasetEntry(cachedLasagna(root / "vol-a", "https://example.test/vol-a").string()));
    const auto& catalog = *options.catalog;

    const auto resolved = resolveLasagnaForView(*pkg, catalog.volumeAt("https://example.test/vol-a.zarr", 3));
    REQUIRE(resolved.has_value());
    CHECK(resolved->manifestBacked);
    CHECK(resolved->workingToBaseScale == doctest::Approx(8.0));
    CHECK(resolved->sourceManifestLocation ==
          "https://example.test/vol-a/data.lasagna.json");

    const auto native = resolveLasagnaForView(*pkg, catalog.volumeAt("https://example.test/vol-a.zarr", 0));
    REQUIRE(native.has_value());
    CHECK(native->workingToBaseScale == doctest::Approx(1.0));

    // Another level of the volume, and a volume whose published Lasagna was
    // never cached, have none.
    CHECK_THROWS(resolveLasagnaForView(*pkg, catalog.volumeAt("https://example.test/vol-a.zarr", 2)));
    CHECK_THROWS(resolveLasagnaForView(*pkg, catalog.volumeAt("https://example.test/vol-b.zarr", 0)));
    std::filesystem::remove_all(root);
}

TEST_CASE("Manual project Lasagna resolution materializes a selected remote manifest")
{
    const auto root = std::filesystem::temp_directory_path() /
        ("vc_open_data_remote_manual_" + std::to_string(vc::memmap::pid()));
    std::filesystem::remove_all(root);
    vc::test::LoopbackHttpServer server;
    const auto served = root / "served";
    std::filesystem::create_directories(served / "manual");
    writeFile(served / "manual" / "data.lasagna.json", R"({"version":2,"groups":{
        "nx":{"zarr":"nx.ome.zarr/4","scaledown":4,"channels":["nx"]}
    }})");
    server.serveDirectory(served);
    const std::string location = server.url() + "/manual/data.lasagna.json";
    vc::lasagna::LasagnaDatasetOpenOptions openOptions;
    openOptions.remoteCacheRoot = root;
    const auto cached = vc::lasagna::LasagnaDataset::openLocation(
        location, openOptions);

    const QByteArray previousConfigDir = qgetenv("VC3D_CONFIG_DIR");
    qputenv("VC3D_CONFIG_DIR", QByteArray::fromStdString((root / "settings").string()));
    writeFile(root / "settings" / "cache.json",
              nlohmann::json{{"version",1},{"root",root.string()}}.dump());
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(
        vc3d::settings::viewer::REMOTE_CACHE_DIR,
        QString::fromStdString(root.string()));
    settings.sync();

    auto pkg = vc::test::emptyProject();
    REQUIRE(pkg->addLasagnaDatasetEntry(location));
    pkg->setSelectedLasagnaDataset(location);
    const auto resolved = resolveLasagnaForView(*pkg, std::nullopt);
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
    CHECK(channel.sourceLocation == server.url() + "/manual/nx.ome.zarr/4");
    CHECK(metricsDataset->manifest().workingToBaseScale == resolved->workingToBaseScale);
    CHECK(server.requestCount("/manual/data.lasagna.json") == 1);
    if (previousConfigDir.isNull()) qunsetenv("VC3D_CONFIG_DIR");
    else qputenv("VC3D_CONFIG_DIR", previousConfigDir);
    // resolveLasagnaForView pinned the process-wide remote cache root to
    // `root`; it stays so later cases in this process keep a writable root.
}

TEST_CASE("Open-data Lasagna derives and verifies coordinate scale from volume shape")
{
    const auto root = std::filesystem::temp_directory_path() /
        ("vc_open_data_lasagna_shape_" + std::to_string(vc::memmap::pid()));
    std::filesystem::remove_all(root);
    vc::project::LoadOptions options;
    options.catalog = lasagnaCatalog(R"({"vol-a": {"id": "vol-a", "data": [)" + publishedArtifact("zarr", "vol-a.zarr") +
                                     ", " + publishedArtifact("lasagna", "vol-a", 3) + "]}}");
    auto pkg = vc::test::emptyProject(options);
    REQUIRE(pkg->addLasagnaDatasetEntry(cachedLasagna(root, "https://example.test/vol-a", R"({
      "version":2,
      "source_to_base":1,
      "base_shape_zyx":[100,200,300],
      "groups":{}
    })").string()));
    const auto resolved = resolveLasagnaForView(
        *pkg, options.catalog->volumeAt("https://example.test/vol-a.zarr", 3), std::array<int, 3>{13, 25, 38});
    REQUIRE(resolved.has_value());
    CHECK(resolved->workingToBaseScale == doctest::Approx(8.0));

    // A level-2 volume of that shape is not what the catalog says it is.
    CHECK_THROWS(resolveLasagnaForView(
        *pkg, options.catalog->volumeAt("https://example.test/vol-a.zarr", 2), std::array<int, 3>{13, 25, 38}));
    std::filesystem::remove_all(root);
}

TEST_CASE("Open-data Lasagna attaches only its scan, opened at a working level in VC3D")
{
    vcr::catalog::OpenDataSample sample;
    sample.id = "sample";
    vcr::catalog::OpenDataVolume volume;
    volume.id = "high-resolution";
    volume.suffix = "2.4um";
    volume.pixelSizeUm = 2.4;

    vcr::catalog::OpenDataArtifact source;
    source.type = "ome-zarr";
    source.resolvedUrl = "http://127.0.0.1:9/source.zarr";
    volume.artifacts.push_back(source);

    vcr::catalog::OpenDataArtifact lasagna;
    lasagna.type = "lasagna";
    lasagna.resolvedUrl = "http://127.0.0.1:9/lasagna";
    lasagna.levelParameterPresent = true;
    lasagna.sourceCoordinateLevel = 2;
    volume.artifacts.push_back(lasagna);
    sample.volumes.push_back(volume);

    vc::project::LoadOptions options;
    options.deferResolution = true;
    auto pkg = vc::test::emptyProject(options);
    const auto result = attachOpenDataSampleVolumes(*pkg, sample);

    CHECK(result.attachedVolumeEntries == 1);
    REQUIRE(pkg->volumeEntries().size() == 1);
    CHECK(pkg->volumeEntries()[0].location ==
          "http://127.0.0.1:9/source.zarr");
    CHECK(vc::project::usesAnonymousRemoteAuth(pkg->volumeEntries()[0]));
    // Opened at level 2, the entry still names the scan: its volume at level 2.
    vcr::catalog::OpenDataManifest catalog;
    catalog.samples = {sample};
    const auto view = vc::project::catalogVolumeView(&catalog, "http://127.0.0.1:9/source.zarr#vc-base-scale=2");
    REQUIRE(view);
    CHECK((view->volumeId == "high-resolution" && view->level == 2 && view->source));
    // Attaching the sample again keeps the one entry.
    CHECK(attachOpenDataSampleVolumes(*pkg, sample).attachedVolumeEntries == 0);
    CHECK(pkg->volumeEntries().size() == 1);
}

TEST_CASE("Selected Lasagna does not implicitly attach scan views")
{
    vcr::catalog::OpenDataSample sample;
    sample.id = "sample";
    vcr::catalog::OpenDataVolume volume;
    volume.id = "v";
    volume.pixelSizeUm = 2.4;
    vcr::catalog::OpenDataArtifact source, lasagna;
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
    selection.representations = vcr::catalog::derivedRepresentations(sample);
    vc::project::LoadOptions options;
    options.deferResolution = true;
    auto pkg = vc::test::emptyProject(options);
    attachOpenDataSampleVolumes(*pkg, sample, &selection);
    CHECK(pkg->volumeEntries().empty());

    selection.rawVolumeIds->push_back(volume.id);
    auto withScan = vc::test::emptyProject(options);
    attachOpenDataSampleVolumes(*withScan, sample, &selection);
    REQUIRE(withScan->volumeEntries().size() == 1);
    CHECK(withScan->volumeEntries().front().location == source.resolvedUrl);
    // The L0 scan resolves the level-2 Lasagna directly.
    const auto root = std::filesystem::temp_directory_path() /
        ("vc_open_data_selected_lasagna_" + std::to_string(vc::memmap::pid()));
    std::filesystem::remove_all(root);
    auto catalog = std::make_shared<vcr::catalog::OpenDataManifest>();
    catalog->samples = {sample};
    options.catalog = catalog;
    auto cataloged = vc::test::emptyProject(options);
    const auto manifest = cachedLasagna(root, lasagna.resolvedUrl);
    REQUIRE(cataloged->addLasagnaDatasetEntry(manifest.string()));
    const auto resolved = resolveLasagnaForView(*cataloged, catalog->volumeAt(source.resolvedUrl, 0));
    REQUIRE(resolved.has_value());
    CHECK(resolved->workingToBaseScale == doctest::Approx(1.0));
    CHECK(resolved->manifestPath == manifest);
    std::filesystem::remove_all(root);
}

TEST_CASE("OpenDataNormalGrids preserves distinct level artifacts and cache identities")
{
    const auto manifest = vcr::catalog::parseOpenDataManifest(R"({
  "samples": {
    "sample": {
      "volumes": {
        "volume": {
          "data": [
            {
              "type": "normal-grids",
              "parameters": {
                "level": 0
              },
              "origins": [
                {
                  "path": "grids/L0",
                  "access_roots": [
                    {
                      "type": "https",
                      "url": "https://example.test/",
                      "usage": "public-read"
                    }
                  ]
                }
              ]
            },
            {
              "type": "normal_grids",
              "parameters": {
                "level": 2
              },
              "origins": [
                {
                  "path": "grids/L2",
                  "access_roots": [
                    {
                      "type": "https",
                      "url": "https://example.test/",
                      "usage": "public-read"
                    }
                  ]
                }
              ]
            },
            {
              "type": "normal-grids",
              "parameters": {
                "level": 2.5
              },
              "origins": [
                {
                  "path": "grids/invalid",
                  "access_roots": [
                    {
                      "type": "https",
                      "url": "https://example.test/",
                      "usage": "public-read"
                    }
                  ]
                }
              ]
            }
          ]
        }
      }
    }
  }
})");
    const auto& sample = manifest.samples.front();
    const auto infos = normalGridsArtifacts(sample.id, sample.volumes.front());
    // Any spelling the catalog classifies as normal grids, at its artifact index.
    REQUIRE(infos.size() == 2);
    CHECK(infos[0].sourceCoordinateLevel == 0);
    CHECK(infos[0].artifactIndex == 0);
    CHECK(infos[1].sourceCoordinateLevel == 2);
    CHECK(infos[1].artifactIndex == 1);

    // Each store streams from its own cache entry.
    const auto root = std::filesystem::temp_directory_path() /
        ("vc_open_data_normal_grids_" + std::to_string(vc::memmap::pid()));
    std::filesystem::remove_all(root);
    auto pkg = vc::test::emptyProject();
    CHECK(attachOpenDataNormalGrids(*pkg, sample, root) == 2);
    const auto entries = pkg->normalGridEntries();
    REQUIRE(entries.size() == 2);
    // At the URL's own place under the root, nothing beside it.
    CHECK(entries[0].location == vcr::zarr::decodedCachePath(root, infos[0].url).string());
    CHECK(entries[1].location == vcr::zarr::decodedCachePath(root, infos[1].url).string());

    // The cache state counts the store's own files: grids and metadata, not the markers,
    // not what the file cache has in flight, not markers an older VC3D left.
    {
        const auto dir = vcr::zarr::decodedCachePath(root, infos[0].url);
        std::filesystem::create_directories(dir / "xy");
        std::ofstream(dir / "metadata.json") << "{}";
        std::ofstream(dir / "xy" / "000000.grid") << "0123456789";
        std::ofstream(dir / "xy" / "000001.grid.tmp.123.456") << "abc";
        std::ofstream(dir / "xy" / "000002.grid.missing").flush();
        const auto state = normalGridsCacheState(root, sample.id, sample.volumes.front());
        CHECK(state.hasArtifact);
        CHECK_FALSE(state.complete);
        CHECK(state.cachedFiles == 2);
        CHECK(state.cachedBytes == 12);
    }

    // A store attached from another cache directory gives way to this one.
    const auto other = root / "normal_grids" / "old";
    REQUIRE(vc::core::util::writeNormalGridsRemoteMarker(other, infos[0].url));
    pkg->addNormalGridEntry(other.string());
    REQUIRE(pkg->normalGridEntries().size() == 3);
    CHECK(attachOpenDataNormalGrids(*pkg, sample, root) == 2);
    CHECK(pkg->normalGridEntries().size() == 2);
    std::filesystem::remove_all(root);
}

TEST_CASE("Normal grids download through the file cache: listing, signed objects, resume, completion")
{
    // A signed bucket on a loopback server, reached through the environment's keys and endpoint.
    const auto scratch = std::filesystem::temp_directory_path() /
        ("vc_open_data_grid_download_" + std::to_string(vc::memmap::pid()));
    std::filesystem::remove_all(scratch);
    const auto bucket = scratch / "bucket", root = scratch / "cache";
    for (const char* plane : {"xy", "xz", "yz"}) {
        std::filesystem::create_directories(bucket / "ng" / plane);
        for (int slice = 0; slice < 3; ++slice) {
            char name[32];
            std::snprintf(name, sizeof name, "%06d.grid", slice);
            std::ofstream(bucket / "ng" / plane / name) << plane << slice << std::string(100, 'g');
        }
    }
    std::ofstream(bucket / "ng" / "metadata.json") << R"({"sparse-volume": 1, "spiral-step": 20.0})";
    std::filesystem::create_directories(bucket / "ng" / "empty-dir");
    vc::test::LoopbackHttpServer server;
    server.serveBucket("grids", bucket, true);
    setenv("AWS_ACCESS_KEY_ID", "key", 1);
    setenv("AWS_SECRET_ACCESS_KEY", "secret", 1);
    setenv("AWS_REGION", "us-east-1", 1);
    setenv("AWS_ENDPOINT_URL", server.url().c_str(), 1);
    setenv("VC3D_CONFIG_DIR", (scratch / "config").c_str(), 1);
    std::filesystem::create_directories(scratch / "config");
    std::ofstream(scratch / "config" / "VC3D.ini") << "[perf]\nremote_cache_min_free_gib=0\n";

    OpenDataNormalGridsInfo info;
    info.sampleId = "sample";
    info.volumeId = "volume";
    info.url = "s3://grids/ng";
    const auto finalDir = vcr::zarr::decodedCachePath(root, info.url);
    std::filesystem::create_directories(finalDir / "xy");
    std::ofstream(finalDir / "xy" / "000007.grid.missing").flush();  // an older VC3D's marker

    // Cancelled after the first file: what was fetched stays, nothing is complete.
    std::atomic<bool> cancel{false};
    std::vector<std::string> statuses;
    std::string error;
    auto dir = downloadOpenDataNormalGrids(info, root, 2, [&](const NormalGridsDownloadProgress& p) {
        if (statuses.empty() || statuses.back() != p.status) statuses.push_back(p.status);
        if (p.completedFiles >= 1) cancel.store(true);
    }, &cancel, &error);
    CHECK(dir.empty());
    CHECK(error == "cancelled");
    CHECK(statuses == std::vector<std::string>{"listing", "downloading"});
    CHECK_FALSE(std::filesystem::exists(finalDir / "normal-grids-complete.json"));
    std::size_t fetched = 0;  // grids and metadata.json fetched before the cancel
    for (const auto& entry : std::filesystem::recursive_directory_iterator(finalDir))
        if (entry.is_regular_file() && (entry.path().extension() == ".grid" || entry.path().filename() == "metadata.json")) ++fetched;
    CHECK(fetched >= 1);
    CHECK(fetched < 10);

    // The rerun lists again, fetches only what is missing, clears the old marker and completes.
    const auto requestsBefore = server.requests().size();
    cancel.store(false);
    NormalGridsDownloadProgress last;
    dir = downloadOpenDataNormalGrids(info, root, 4, [&](const NormalGridsDownloadProgress& p) { last = p; }, &cancel, &error);
    CHECK(dir == finalDir);
    CHECK(last.totalFiles == 10);
    CHECK(last.completedFiles == 10);
    CHECK(last.failedFiles == 0);
    std::uint64_t servedBytes = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(bucket / "ng"))
        if (entry.is_regular_file()) servedBytes += entry.file_size();
    CHECK(last.totalBytes == servedBytes);
    CHECK(last.completedBytes == servedBytes);
    CHECK(std::filesystem::is_regular_file(finalDir / "normal-grids-complete.json"));
    CHECK(std::filesystem::is_regular_file(finalDir / "yz" / "000002.grid"));
    CHECK_FALSE(std::filesystem::exists(finalDir / "xy" / "000007.grid.missing"));
    CHECK(vc::core::util::normalGridsRemoteUrl(finalDir) == info.url);
    std::size_t objectGets = 0;
    for (const auto& request : server.requests()) {
        CHECK(request.signedV4());
        if (request.method == "GET" && request.path.ends_with(".grid")) ++objectGets;
    }
    CHECK(objectGets == 9);  // every grid once across both runs
    CHECK(server.requests().size() > requestsBefore);
    const auto state = normalGridsCacheState(root, info.sampleId, vcr::catalog::OpenDataVolume{});
    CHECK_FALSE(state.hasArtifact);  // a volume without the artifact

    // Complete: no request at all.
    const auto requestsAfter = server.requests().size();
    CHECK(downloadOpenDataNormalGrids(info, root, 4, {}, &cancel, &error) == finalDir);
    CHECK(server.requests().size() == requestsAfter);
    for (const char* name : {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "AWS_ENDPOINT_URL"}) unsetenv(name);
    std::filesystem::remove_all(scratch);
}

TEST_CASE("OpenDataManifest propagates catalog pixel size from volume properties or referenced scan")
{
    const auto manifest = vcr::catalog::parseOpenDataManifest(R"({
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

    const auto byId = [&](const std::string& id) -> const vcr::catalog::OpenDataVolume* {
        const auto it = std::find_if(
            sample->volumes.begin(), sample->volumes.end(), [&](const vcr::catalog::OpenDataVolume& volume) {
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

    // Deferred: the catalog's URLs are not opened.
    vc::project::LoadOptions deferred;
    deferred.deferResolution = true;
    auto pkg = vc::test::emptyProject(deferred);
    const auto result = attachOpenDataSampleVolumes(*pkg, *sample);
    CHECK(result.supportedVolumes == 2);
    REQUIRE(pkg->volumeEntries().size() == 2);

    // The catalog names each entry when it opens; its pixel size reaches the volume through
    // vcr::zarr::volumeMetadata, where the store's documents win.
    const auto metadataFor = [&](const std::string& id) {
        for (const auto& entry : pkg->volumeEntries()) {
            const auto metadata = vc::project::catalogVolumeMetadata(&manifest, entry.location);
            if (metadata.contains("uuid") && metadata.at("uuid").get_string() == id) return metadata;
        }
        FAIL("no entry shows " << id);
        return utils::Json::object();
    };
    CHECK_FALSE(metadataFor("volume-with-properties").contains("voxelsize"));
    CHECK(metadataFor("volume-from-scan").at("uuid").get_string() == "volume-from-scan");
}

TEST_CASE("OpenDataSampleProject attaches all supported zarr artifacts for a catalog volume")
{
    vcr::catalog::OpenDataSample sample;
    sample.id = "PHerc0139";

    vcr::catalog::OpenDataVolume volume;
    volume.id = "scan-volume";
    volume.dataFormat = "zarr";
    volume.pixelSizeUm = 7.91;

    auto artifact = [](std::string type, std::string url) {
        vcr::catalog::OpenDataArtifact out;
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
    auto pkg = vc::test::emptyProject(loadOptions);
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

    // The source is the catalog volume; predictions without parameters.level
    // have no catalog frame and are kept unpaired.
    vcr::catalog::OpenDataManifest catalog;
    catalog.samples = {sample};
    const auto source = vc::project::catalogVolumeView(&catalog, pkg->volumeEntries()[0].location);
    REQUIRE(source);
    CHECK((source->volumeId == "scan-volume" && source->level == 0 && source->source));
    CHECK(vc::project::catalogVolumeMetadata(&catalog, pkg->volumeEntries()[0].location).at("uuid").get_string() ==
          "scan-volume");
    CHECK_FALSE(vc::project::catalogVolumeView(&catalog, pkg->volumeEntries()[1].location));
    CHECK_FALSE(vc::project::catalogVolumeView(&catalog, pkg->volumeEntries()[2].location));
    CHECK(std::any_of(
        result.messages.begin(), result.messages.end(), [](const std::string& message) {
            return message.find(
                       "Kept scan-volume (ink-detection-3d-zarr) coordinate-unspecified") !=
                   std::string::npos;
        }));
}

TEST_CASE("Project creation defaults to the chosen directory, else VC3D's projects")
{
    const auto root = std::filesystem::current_path() /
        ("project_defaults_test_" + std::to_string(vc::memmap::pid()));
    // projectsDirectory() creates VC3D's AppData projects directory.
    const QByteArray previousDataHome = qgetenv("XDG_DATA_HOME");
    qputenv("XDG_DATA_HOME", QByteArray::fromStdString((root / "xdg-data").string()));
    {
        QSettings settings(QString::fromStdString((root / "settings.ini").string()), QSettings::IniFormat);
        CHECK(vc3d::defaultNewProjectDirectory(settings) == vc3d::projectsDirectory());
        CHECK(std::filesystem::is_directory(vc3d::projectsDirectory().toStdString()));
        settings.setValue(vc3d::settings::project::DEFAULT_PATH, "chosen-directory");
        CHECK(vc3d::defaultNewProjectDirectory(settings) == QStringLiteral("chosen-directory"));
    }
    if (previousDataHome.isNull()) qunsetenv("XDG_DATA_HOME");
    else qputenv("XDG_DATA_HOME", previousDataHome);
    std::filesystem::remove_all(root);
}

TEST_CASE("Catalog source selection is independent of derived representations")
{
    vcr::catalog::OpenDataSample sample;
    sample.id = "selection";
    vcr::catalog::OpenDataVolume volume;
    volume.id = "v";
    volume.dataFormat = "zarr";
    vcr::catalog::OpenDataArtifact source, prediction;
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
    auto empty = vc::test::emptyProject(options);
    attachOpenDataSampleVolumes(*empty, sample, &selection);
    CHECK(empty->volumeEntries().empty());

    selection.representations = vcr::catalog::derivedRepresentations(sample);
    auto derived = vc::test::emptyProject(options);
    attachOpenDataSampleVolumes(*derived, sample, &selection);
    REQUIRE(derived->volumeEntries().size() == 1);
    CHECK(derived->volumeEntries().front().location == prediction.resolvedUrl);

    selection.rawVolumeIds = std::vector<std::string>{"v"};
    selection.representations->clear();
    auto raw = vc::test::emptyProject(options);
    attachOpenDataSampleVolumes(*raw, sample, &selection);
    REQUIRE(raw->volumeEntries().size() == 1);
    CHECK(raw->volumeEntries().front().location == source.resolvedUrl);
}

TEST_CASE("Single catalog prediction attachment retains identity and validates its source level")
{
    vcr::catalog::OpenDataSample sample;
    sample.id = "sample";
    vcr::catalog::OpenDataVolume volume;
    volume.id = "scan";
    volume.pixelSizeUm = 2.4;
    vcr::catalog::OpenDataArtifact source, prediction;
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
    selection.representations = std::vector<vcr::catalog::OpenDataRepresentationRef>{
        {0, 1, vcr::catalog::OpenDataRepresentationKind::Prediction}};
    vc::project::LoadOptions options;
    options.deferResolution = true;
    auto pkg = vc::test::emptyProject(options);
    const auto result = attachOpenDataSampleVolumes(*pkg, sample, &selection);
    REQUIRE(pkg->volumeEntries().size() == 1);
    const auto& entry = pkg->volumeEntries().front();
    CHECK(entry.location == prediction.resolvedUrl);
    vcr::catalog::OpenDataManifest catalog;
    catalog.samples = {sample};
    const auto view = vc::project::catalogVolumeView(&catalog, entry.location);
    REQUIRE(view);
    CHECK((view->sampleId == "sample" && view->volumeId == "scan" && view->level == 2 && !view->source));
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

TEST_CASE("A sample's project file names its sample and references its store")
{
    auto manifest = vcr::catalog::parseOpenDataManifest(kFixture);
    const auto sample = *manifest.findSample("PHerc0139");
    const auto root = std::filesystem::current_path() /
        ("selective_catalog_test_" + std::to_string(vc::memmap::pid()));
    std::filesystem::remove_all(root);
    vc::objects::OpenOptions options;
    options.author = "vc-test";

    const auto path = root / "empty.vcproj";
    createSampleProjectFiles(path, sample, "Empty chosen project", options);
    CHECK_THROWS(createSampleProjectFiles(path, sample, "again", options));
    auto session = std::make_shared<vc::objects::ObjectStoreSession>();
    auto storePath = path; storePath.replace_extension(".vcobj");
    const auto project = session->attach(storePath, options);
    CHECK(openDataProjectSample(*VolumePkg::open(path, {.deferResolution = true})) == std::optional<std::string>("PHerc0139"));
    OpenDataResourceSelection selection;
    selection.rawVolumeIds.emplace();
    selection.representations.emplace();
    selection.segmentIds.emplace();
    auto empty = createOpenDataSampleProject(session, path, project, sample, root, nullptr, {}, &selection);
    CHECK(empty->volumeEntries().empty());
    CHECK(empty->selectedLasagnaDataset().empty());
    CHECK(empty->name() == "Empty chosen project");
    CHECK(empty->path() == path);
    CHECK_FALSE(std::filesystem::exists(root / "fibers"));

    // Any other store is not a sample's project.
    const auto plain = vc::test::createTestProject();
    CHECK_FALSE(openDataProjectSample(*plain.open()));
    std::filesystem::remove_all(root);
}

TEST_CASE("OpenDataSampleProject prefers the volume sourcing the most segments")
{
    vcr::catalog::OpenDataSample sample;
    sample.id = "PHerc0139";

    auto artifact = [](std::string url) {
        vcr::catalog::OpenDataArtifact out;
        out.type = "zarr";
        out.resolvedUrl = std::move(url);
        return out;
    };
    auto volume = [&](std::string id) {
        vcr::catalog::OpenDataVolume out;
        out.id = std::move(id);
        out.dataFormat = "zarr";
        out.artifacts.push_back(artifact("http://127.0.0.1:9/" + out.id + ".zarr"));
        return out;
    };
    auto segment = [](std::string volumeId) {
        vcr::catalog::OpenDataSegment out;
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

    vc::project::LoadOptions deferred;
    deferred.deferResolution = true;
    auto pkg = vc::test::emptyProject(deferred);
    const auto result = attachOpenDataSampleVolumes(*pkg, sample);

    CHECK(result.supportedVolumes == 3);
    CHECK(result.preferredVolumeId == "vol-b");

    // A segment is sourced by the volume it was derived from, as its surfaces are.
    for (auto& derived : sample.segments)
        if (derived.originalVolumeId == "vol-b")
            derived.creation.derivedFrom = vcr::catalog::OpenDataReference{"volume", "vol-c"};
    auto derivedPkg = vc::test::emptyProject(deferred);
    CHECK(attachOpenDataSampleVolumes(*derivedPkg, sample).preferredVolumeId == "vol-c");
}


TEST_CASE("Reopening a sample's project keeps what the user changed")
{
    vcr::catalog::OpenDataSample sample;
    sample.id = "Sample With Spaces";

    const auto root = std::filesystem::temp_directory_path() /
                      ("vc_open_data_sample_project_test_" + std::to_string(vc::memmap::pid()));
    std::filesystem::remove_all(root);
    const auto path = sampleProjectPath(root, sample);
    CHECK(path == root / "open_data" / "Sample_With_Spaces.vcproj");
    vc::objects::OpenOptions options;
    options.author = "vc-test";
    createSampleProjectFiles(path, sample, sample.id, options);
    auto session = std::make_shared<vc::objects::ObjectStoreSession>();
    auto storePath = path; storePath.replace_extension(".vcobj");
    const auto project = session->attach(storePath, options);

    auto first = createOpenDataSampleProject(session, path, project, sample, root);
    CHECK(first->path() == path);
    first->setSelectedLasagnaDataset("user-marker");
    REQUIRE(first->addVolumeEntry("/user/volume.zarr"));

    auto second = createOpenDataSampleProject(session, path, project, sample, root);
    CHECK(second->selectedLasagnaDataset() == "user-marker");
    CHECK(std::any_of(second->volumeEntries().begin(), second->volumeEntries().end(),
                      [](const auto& entry) { return entry.location == "/user/volume.zarr"; }));
    std::filesystem::remove_all(root);
}
