// The "project" block a Spiral session request carries: per store volume the
// facts the project knows and that volume's umbilicus, ZYX in its own voxels,
// read from the project's umbilicus store.
#include "SpiralProjectInputs.hpp"

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
    const QJsonObject block = vc3d::spiral::projectBlock("PHerc", volumes, session.get(), attachment);
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
    require(!vc3d::spiral::projectBlock("PHerc", volumes, session.get(), std::nullopt)
                 .value("volumes").toObject().value("scan").toObject().contains("umbilicus"),
            "without an umbilicus store there is no umbilicus");
}
} // namespace

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    try {
        QTemporaryDir directory;
        describesTheVolumesAndTheirUmbilici(directory.path().toStdString());
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    std::cout << "spiral project inputs: ok\n";
    return 0;
}
