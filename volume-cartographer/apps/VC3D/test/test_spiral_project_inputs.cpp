// The "project" block a Spiral session request carries: per store volume the
// facts the project knows and that volume's umbilicus, ZYX in its own voxels,
// read from the project's umbilicus store.
#include "SpiralProjectInputs.hpp"

#include "vc/lasagna/Manifest.hpp"

#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <vc_object_store/objects/UmbilicusFile.hpp>

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QJsonArray>
#include <QTemporaryDir>

#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;
using namespace vc::objects;
using vc3d::spiral::ProjectVolumeFacts;

namespace {
void require(bool value, const std::string& message)
{
    if (!value) throw std::runtime_error(message);
}

// A Lasagna dataset of one-channel arrays (version 2) at the base's 2^4.
constexpr const char* kManifest = R"({
    "version": 2, "source_to_base": 1.0, "base_shape_zyx": [400, 256, 192],
    "groups": {
        "nx": {"zarr": "s_nx.ome.zarr/4", "scaledown": 4, "channels": ["nx"]},
        "ny": {"zarr": "s_ny.ome.zarr/4", "scaledown": 4, "channels": ["ny"]},
        "grad_mag": {"zarr": "s_gm.ome.zarr/4", "scaledown": 4, "channels": ["grad_mag"]}}})";

void describesTheVolumesAndTheirUmbilici(const fs::path& directory)
{
    auto session = std::make_shared<ObjectStoreSession>();
    OpenOptions options; options.author = "spiral-project-inputs-test";
    const auto path = directory / "project.vcobj";
    { auto created = ObjectStore::create(path, options); }
    const auto attachment = session->attach(path, options);
    UmbilicusFile file;
    file.points = {{10, 20, 0}, {12, 22, 100}};  // XYZ
    importVolumeUmbilicus(*session, attachment, "scan", file, 1.0, "umbilicus", {{"tool", "test"}});

    ProjectVolumeFacts scan;
    scan.voxelSizeUm = 7.91;
    scan.baseShapeZYX = std::array<std::size_t, 3>{100, 64, 48};
    scan.zTopToBottom = true;
    std::map<std::string, ProjectVolumeFacts> volumes{{"scan", scan}, {"other", {}}};
    const QJsonObject block = vc3d::spiral::projectBlock("PHerc", volumes, std::nullopt, std::nullopt, session.get(),
                                                         attachment);
    require(block.value("name").toString() == "PHerc", "the project's name");
    const QJsonObject entries = block.value("volumes").toObject();
    const QJsonObject entry = entries.value("scan").toObject();
    require(entry.value("voxel_size_um").toDouble() == 7.91, "the voxel size");
    require(entry.value("base_shape_zyx").toArray() == QJsonArray{100, 64, 48}, "the shape, ZYX");
    require(entry.value("z_direction_is_top_to_bottom").toBool(), "a known orientation");
    require(!entry.contains("left_handed_coordinates"), "an unknown fact is left out");
    const QJsonObject umbilicus = entry.value("umbilicus").toObject();
    const auto reference = umbilicus.value("reference").toObject();
    require(!reference.value("object").toString().isEmpty()
                && !reference.value("store").toString().isEmpty()
                && !reference.value("generation").toString().isEmpty()
                && reference.value("version").isDouble(),
            "the umbilicus object and its generation and version read");
    require(umbilicus.value("points_zyx").toArray()
                == QJsonArray{QJsonArray{0, 20, 10}, QJsonArray{100, 22, 12}},
            "the umbilicus points, ZYX");
    require(entries.value("other").toObject().isEmpty(),
            "a volume with no facts and no umbilicus is listed empty");
    require(!vc3d::spiral::projectBlock("PHerc", volumes, std::nullopt, std::nullopt, session.get(), std::nullopt)
                 .value("volumes").toObject().value("scan").toObject().contains("umbilicus"),
            "without an umbilicus store there is no umbilicus");
}

// The fit reads each channel's store at the arrays' name, downsampled
// relative to each volume by the dataset's scale over the volume's.
void describesTheLasagnaDataset()
{
    using vc::lasagna::LasagnaDatasetManifest;
    const auto manifest = LasagnaDatasetManifest::parseText(kManifest, "/data/s.lasagna.json");
    const auto lasagna = vc3d::spiral::projectLasagna(manifest, "/data/s.lasagna.json", "run-1");
    require(lasagna.error.empty(), "a readable dataset: " + lasagna.error);
    require(lasagna.identity == "run-1", "the identity");
    require(lasagna.group == "4", "the arrays' name");
    require(lasagna.stores.at("normal_x") == "/data/s_nx.ome.zarr"
                && lasagna.stores.at("gradient_magnitude") == "/data/s_gm.ome.zarr",
            "each channel's store");

    ProjectVolumeFacts quarter;  // the base at a quarter of its resolution
    quarter.baseShapeZYX = std::array<std::size_t, 3>{100, 64, 48};
    std::map<std::string, ProjectVolumeFacts> volumes{{"quarter", quarter}, {"unknown", {}}};
    const QJsonObject block = vc3d::spiral::projectBlock("PHerc", volumes, lasagna, std::nullopt, nullptr, std::nullopt);
    const QJsonObject entries = block.value("volumes").toObject();
    require(entries.value("quarter").toObject().value("lasagna_scale").toInt() == 4,
            "a 2^4 array of the base is 4 voxels of a quarter-resolution volume");
    require(!entries.value("unknown").toObject().contains("lasagna_scale"),
            "no scale for a volume of unknown shape");
    const QJsonObject value = block.value("lasagna").toObject();
    require(value.value("group").toString() == "4"
                && value.value("identity").toString() == "run-1"
                && value.value("normal_y").toString() == "/data/s_ny.ome.zarr",
            "the block carries the identity, the stores and their array");
    auto remote = lasagna;
    remote.location = "https://data.example/s.lasagna.json";
    remote.identity.clear();
    require(!vc3d::spiral::projectBlock("PHerc", volumes, remote, std::nullopt, nullptr, std::nullopt)
                 .value("lasagna").toObject().contains("identity"),
            "a remote dataset is sent without an identity");

    const auto shared = vc3d::spiral::projectLasagna(LasagnaDatasetManifest::parseText(R"({
        "version": 2, "groups": {"normals": {"zarr": "n.zarr/0", "scaledown": 2, "channels": ["nx", "ny"]}}})",
        "/data/n.lasagna.json"), "/data/n.lasagna.json", "run-2");
    require(shared.error.find("single-channel") != std::string::npos, "a multichannel array is refused");
}
// A dataset's identity is its provenance run_uuid when the record names this
// manifest (or none), else the SHA-256 of the manifest's bytes; spiral-fitting
// (lasagna_source.manifest_identity) computes the same.
void identifiesTheDataset()
{
    const std::string bytes = "{\"version\": 2}";
    const auto digest = vc3d::spiral::lasagnaIdentity(bytes, "s.lasagna.json", "");
    require(digest == "sha256-" + QCryptographicHash::hash(QByteArray::fromStdString(bytes),
                                                          QCryptographicHash::Sha256).toHex().toStdString(),
            "without provenance, the manifest's SHA-256");
    require(vc3d::spiral::lasagnaIdentity(bytes, "s.lasagna.json",
                                          R"({"run_uuid": "5b17ff6c-3c47", "manifest": "s.lasagna.json"})")
                == "5b17ff6c-3c47",
            "the provenance's run_uuid");
    require(vc3d::spiral::lasagnaIdentity(bytes, "s.lasagna.json",
                                          R"({"run_uuid": "5b17ff6c-3c47", "manifest": "other.lasagna.json"})")
                == digest,
            "a record of another manifest is not this one's");
}

// The fiber-inference dataset: its identity, and per volume of known shape
// the dataset's base voxels per volume voxel, the scale its fiber-direction
// samples are extracted at.
void describesTheFiberInferenceDataset()
{
    using vc3d::spiral::ProjectDataset;
    ProjectVolumeFacts quarter;
    quarter.baseShapeZYX = std::array<std::size_t, 3>{100, 64, 48};
    std::map<std::string, ProjectVolumeFacts> volumes{{"quarter", quarter}, {"unknown", {}}};
    ProjectDataset fibers;
    fibers.location = "/data/f.lasagna.json";
    fibers.identity = "run-4";
    fibers.baseShapeZYX = std::array<std::size_t, 3>{400, 256, 192};
    const QJsonObject block = vc3d::spiral::projectBlock("PHerc", volumes, std::nullopt, fibers, nullptr,
                                                         std::nullopt);
    require(block.value("fiber_inference").toObject()
                == QJsonObject{{"location", "/data/f.lasagna.json"}, {"identity", "run-4"}},
            "the dataset and its identity");
    // A local copy of a remote dataset also names the manifest it was copied
    // from, which a service host without the copy reads.
    auto manifest = vc::lasagna::LasagnaDatasetManifest::parseText(kManifest);
    manifest.manifestPath = "/cache/f/f.lasagna.json";
    require(vc3d::spiral::remoteManifestUrl(manifest).empty(), "a local dataset has no remote manifest");
    manifest.remoteSourceBaseLocation = "https://data.example/f";
    fibers.remoteManifest = vc3d::spiral::remoteManifestUrl(manifest);
    require(fibers.remoteManifest == "https://data.example/f/f.lasagna.json",
            "the copy's manifest at its remote base: " + fibers.remoteManifest);
    manifest.manifestIsRemote = true;
    require(vc3d::spiral::remoteManifestUrl(manifest).empty(), "a remote dataset's location is its URL");
    require(vc3d::spiral::projectBlock("PHerc", volumes, std::nullopt, fibers, nullptr, std::nullopt)
                    .value("fiber_inference").toObject().value("remote_manifest").toString()
                == "https://data.example/f/f.lasagna.json",
            "the remote manifest is sent");
    const QJsonObject entries = block.value("volumes").toObject();
    require(entries.value("quarter").toObject().value("fiber_scale").toDouble() == 4,
            "4 base voxels per voxel of a quarter-resolution volume");
    require(!entries.value("unknown").toObject().contains("fiber_scale"), "no scale without a shape");
}

// A remote dataset's stores are its arrays' remote locations; a service on
// this host is given their read-through cache directories instead, and caches
// only what a pack it lacks needs.
void givesALocalServiceTheCachedArrays(const fs::path& directory)
{
    auto manifest = vc::lasagna::LasagnaDatasetManifest::parseText(kManifest);
    const fs::path cacheRoot = directory / "remote_cache" / "s";
    for (auto& group : manifest.groups) {
        group.sourceLocation = "https://data.example/s/" + group.relativeZarrKey;
        group.zarrPath.clear();
        group.remoteCacheRoot = cacheRoot;
    }
    const auto lasagna = vc3d::spiral::projectLasagna(manifest, "/project/s.lasagna.json", "run-4");
    require(lasagna.error.empty() && lasagna.stores.at("normal_x") == "https://data.example/s/s_nx.ome.zarr",
            "a remote dataset's stores are remote locations: " + lasagna.error);
    require(lasagna.remoteGroups.size() == 3, "the remote groups are kept by input");
    const auto cached = vc3d::spiral::withCachedStores(lasagna);
    require(cached.error.empty(), "the cached stores are readable: " + cached.error);
    // Beside vc_reader's cache of the arrays, at their URL's place.
    const fs::path copies = cacheRoot / "lasagna_arrays" / "https" / "data.example" / "s";
    require(cached.stores.at("normal_x") == (copies / "s_nx.ome.zarr").string()
                && cached.stores.at("gradient_magnitude") == (copies / "s_gm.ome.zarr").string()
                && cached.group == "4",
            "each store is its cache directory's");

    const fs::path spiral = directory / "spiral";
    require(vc3d::spiral::lasagnaGroupsToCache(lasagna, spiral).size() == 3,
            "a cache without packs needs every array");
    fs::create_directories(spiral / "lasagna" / "run-4" / "normals_g4");
    std::ofstream(spiral / "lasagna" / "run-4" / "normals_g4" / "meta.json") << "{}";
    const auto needed = vc3d::spiral::lasagnaGroupsToCache(lasagna, spiral);
    require(needed.size() == 1 && needed.front().channels == std::vector<std::string>{"grad_mag"},
            "a packed kind's arrays are not needed");
    require(vc3d::spiral::lasagnaGroupsToCache(lasagna, spiral, std::set<std::string>{"normals"}).empty(),
            "a kind the build does not read is not cached");
    auto remote = lasagna;
    remote.identity.clear();
    require(vc3d::spiral::lasagnaGroupsToCache(remote, spiral).size() == 3,
            "without an identity no pack can be found");
}
} // namespace

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    try {
        QTemporaryDir directory;
        describesTheVolumesAndTheirUmbilici(directory.path().toStdString());
        describesTheLasagnaDataset();
        identifiesTheDataset();
        givesALocalServiceTheCachedArrays(directory.path().toStdString());
        describesTheFiberInferenceDataset();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    std::cout << "spiral project inputs: ok\n";
    return 0;
}
