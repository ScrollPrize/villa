// The "project" block a Spiral session request carries: per store volume the
// facts the project knows and that volume's umbilicus, ZYX in its own voxels,
// read from the project's umbilicus store.
#include "SpiralProjectInputs.hpp"

#include "vc/lasagna/Manifest.hpp"

#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <vc_object_store/objects/UmbilicusFile.hpp>

#include <QCoreApplication>
#include <QJsonArray>
#include <QTemporaryDir>

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
using namespace vc::objects;
using vc3d::spiral::ProjectVolumeFacts;

namespace {
void require(bool value, const std::string& message)
{
    if (!value) throw std::runtime_error(message);
}

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
    const QJsonObject block = vc3d::spiral::projectBlock("PHerc", volumes, std::nullopt, session.get(),
                                                         attachment);
    require(block.value("name").toString() == "PHerc", "the project's name");
    const QJsonObject entries = block.value("volumes").toObject();
    const QJsonObject entry = entries.value("scan").toObject();
    require(entry.value("voxel_size_um").toDouble() == 7.91, "the voxel size");
    require(entry.value("base_shape_zyx").toArray() == QJsonArray{100, 64, 48}, "the shape, ZYX");
    require(entry.value("z_direction_is_top_to_bottom").toBool(), "a known orientation");
    require(!entry.contains("left_handed_coordinates"), "an unknown fact is left out");
    const QJsonObject umbilicus = entry.value("umbilicus").toObject();
    require(!umbilicus.value("uuid").toString().isEmpty()
                && !umbilicus.value("revision").toString().isEmpty(),
            "the umbilicus object and the revision read");
    require(umbilicus.value("points_zyx").toArray()
                == QJsonArray{QJsonArray{0, 20, 10}, QJsonArray{100, 22, 12}},
            "the umbilicus points, ZYX");
    require(entries.value("other").toObject().isEmpty(),
            "a volume with no facts and no umbilicus is listed empty");
    require(!vc3d::spiral::projectBlock("PHerc", volumes, std::nullopt, session.get(), std::nullopt)
                 .value("volumes").toObject().value("scan").toObject().contains("umbilicus"),
            "without an umbilicus store there is no umbilicus");
}

// A Lasagna dataset of one-channel arrays (version 2): the fit reads each
// channel's store at the arrays' name, downsampled relative to each volume by
// the dataset's scale over the volume's.
void describesTheLasagnaDataset()
{
    using vc::lasagna::LasagnaDatasetManifest;
    const auto manifest = LasagnaDatasetManifest::parseText(R"({
        "version": 2, "source_to_base": 1.0, "base_shape_zyx": [400, 256, 192],
        "groups": {
            "nx": {"zarr": "s_nx.ome.zarr/4", "scaledown": 4, "channels": ["nx"]},
            "ny": {"zarr": "s_ny.ome.zarr/4", "scaledown": 4, "channels": ["ny"]},
            "grad_mag": {"zarr": "s_gm.ome.zarr/4", "scaledown": 4, "channels": ["grad_mag"]}}})",
        "/data/s.lasagna.json");
    const auto lasagna = vc3d::spiral::projectLasagna(manifest, "/data/s.lasagna.json");
    require(lasagna.error.empty(), "a readable dataset: " + lasagna.error);
    require(lasagna.group == "4", "the arrays' name");
    require(lasagna.stores.at("normal_x") == "/data/s_nx.ome.zarr"
                && lasagna.stores.at("gradient_magnitude") == "/data/s_gm.ome.zarr",
            "each channel's store");

    ProjectVolumeFacts quarter;  // the base at a quarter of its resolution
    quarter.baseShapeZYX = std::array<std::size_t, 3>{100, 64, 48};
    std::map<std::string, ProjectVolumeFacts> volumes{{"quarter", quarter}, {"unknown", {}}};
    const QJsonObject block = vc3d::spiral::projectBlock("PHerc", volumes, lasagna, nullptr, std::nullopt);
    const QJsonObject entries = block.value("volumes").toObject();
    require(entries.value("quarter").toObject().value("lasagna_scale").toInt() == 4,
            "a 2^4 array of the base is 4 voxels of a quarter-resolution volume");
    require(!entries.value("unknown").toObject().contains("lasagna_scale"),
            "no scale for a volume of unknown shape");
    const QJsonObject value = block.value("lasagna").toObject();
    require(value.value("group").toString() == "4"
                && value.value("normal_y").toString() == "/data/s_ny.ome.zarr",
            "the block carries the stores and their array");

    const auto shared = vc3d::spiral::projectLasagna(LasagnaDatasetManifest::parseText(R"({
        "version": 2, "groups": {"normals": {"zarr": "n.zarr/0", "scaledown": 2, "channels": ["nx", "ny"]}}})",
        "/data/n.lasagna.json"), "/data/n.lasagna.json");
    require(shared.error.find("single-channel") != std::string::npos, "a multichannel array is refused");
    require(vc3d::spiral::projectBlock("PHerc", volumes, shared, nullptr, std::nullopt)
                .value("lasagna").toObject().contains("error"),
            "the block carries why the dataset cannot be read");
}
} // namespace

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    try {
        QTemporaryDir directory;
        describesTheVolumesAndTheirUmbilici(directory.path().toStdString());
        describesTheLasagnaDataset();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    std::cout << "spiral project inputs: ok\n";
    return 0;
}
