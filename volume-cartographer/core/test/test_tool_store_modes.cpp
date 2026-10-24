// Command-line tools in store mode produce what their tifxyz mode produces for
// the same surface. Each tool's binary comes from VC_<TOOL>_BIN.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/Array3D.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include <vc_object_store/Uuid.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

namespace fs = std::filesystem;
using namespace vc::objects;

namespace {
std::string readFile(const fs::path& path)
{
    std::stringstream text;
    text << std::ifstream(path, std::ios::binary).rdbuf();
    return text.str();
}

std::string run(const std::string& command, const fs::path& log)
{
    const int status = std::system((command + " > \"" + log.string() + "\" 2>&1").c_str());
    const auto output = readFile(log);
    INFO(output);
    REQUIRE(status == 0);
    return output;
}

std::string tool(const char* variable)
{
    const char* path = std::getenv(variable);
    REQUIRE(path);
    return std::string("\"") + path + "\"";
}

// A store holding one curved surface, and the same surface as a tifxyz directory.
struct Fixture {
    fs::path dir = fs::temp_directory_path() / ("vc-tool-store-" + util::newUuid());
    fs::path store = dir / "objects.vcobj";
    fs::path tifxyz = dir / "surface.tifxyz";
    std::string uuid = util::newUuid();
    Fixture() {
        fs::create_directories(dir);
        cv::Mat_<cv::Vec3f> points(12, 16);
        for (int r = 0; r < points.rows; ++r)
            for (int c = 0; c < points.cols; ++c)
                points(r, c) = {100.f + 4.f * c, 120.f + 4.f * r, 80.f + 0.05f * float((c - 8) * (c - 8))};
        points(0, 0) = {-1.f, -1.f, -1.f};
        QuadSurface surface(points, {0.25f, 0.25f});
        surface.save(tifxyz, true);
        OpenOptions options; options.author = "tool-store-test";
        auto db = ObjectStore::create(store, options);
        auto txn = db.write();
        ObjectRecord record; record.uuid = uuid; record.kind = "surface"; record.name = "surface";
        record.meta = R"({"coordinates":{"volume_uuid":"vol","scale_factor":1}})";
        db.upsertObject(txn, record);
        db.putGeometry(txn, uuid, quad_surface_geometry(surface));
        txn.commit();
    }
    ~Fixture() { std::error_code ignored; fs::remove_all(dir, ignored); }
    std::string storeArgs() const { return "--store \"" + store.string() + "\" --surface " + uuid; }
};
}

TEST_CASE("vc_tifxyz2obj writes the OBJ into the surface's artifact directory")
{
    Fixture f;
    const auto bin = tool("VC_TIFXYZ2OBJ_BIN");
    run(bin + " \"" + f.tifxyz.string() + "\" \"" + (f.dir / "file.obj").string() + "\" --normalize-uv", f.dir / "file.log");
    const auto output = run(bin + " " + f.storeArgs() + " mesh.obj --normalize-uv", f.dir / "store.log");
    const auto stored = fs::path(f.store.string() + ".artifacts") / f.uuid / "mesh.obj";
    CHECK(output.find("obj: " + stored.string()) != std::string::npos);
    CHECK(readFile(stored) == readFile(f.dir / "file.obj"));
}

TEST_CASE("vc_render_tifxyz renders a stored surface like its tifxyz")
{
    Fixture f;
    Volume::ZarrCreateOptions options;
    options.shapeZYX = {128, 192, 192};
    options.compressor = "none";
    auto volume = Volume::New(f.dir / "volume.zarr", options);
    Array3D<uint8_t> ramp({128, 192, 192}, 0);
    for (size_t z = 0; z < 128; ++z)
        for (size_t y = 0; y < 192; ++y)
            for (size_t x = 0; x < 192; ++x) ramp(z, y, x) = uint8_t((x + 2 * y + 3 * z) % 251);
    volume->writeZYX(ramp, {0, 0, 0});
    const auto bin = tool("VC_RENDER_TIFXYZ_BIN") + " --volume \"" + (f.dir / "volume.zarr").string() +
        "\" --scale 1 --group-idx 0 --num-slices 3";
    run(bin + " --segmentation \"" + f.tifxyz.string() + "\" --tif-output \"" + (f.dir / "file").string() + "\"", f.dir / "file.log");
    run(bin + " " + f.storeArgs() + " --tif-output \"" + (f.dir / "store").string() + "\"", f.dir / "store.log");
    std::vector<fs::path> files;
    for (const auto& entry : fs::recursive_directory_iterator(f.dir / "file"))
        if (entry.is_regular_file()) files.push_back(fs::relative(entry.path(), f.dir / "file"));
    REQUIRE_FALSE(files.empty());
    for (const auto& file : files) CHECK(readFile(f.dir / "store" / file) == readFile(f.dir / "file" / file));
}
