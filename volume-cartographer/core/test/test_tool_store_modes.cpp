// Command-line tools in store mode produce what their tifxyz mode produces for
// the same surface. Each tool's binary comes from VC_<TOOL>_BIN.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <vc_reader/zarr.hpp>
#include <vc_reader/zarr_write.hpp>
#include "vc/core/types/Volume.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include "vc/core/util/QuadSurface.hpp"
#include <vc_object_store/Uuid.hpp>

#include <opencv2/core.hpp>
#include "vc/core/util/ImageMat.hpp"

#include <cmath>
#include <cstdlib>
#include <regex>
#include <tuple>
#include <algorithm>
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
    if (status != 0) MESSAGE(command << "\n" << output);
    REQUIRE(status == 0);
    return output;
}

std::string tool(const char* variable)
{
    const char* path = std::getenv(variable);
    REQUIRE(path);
    return std::string("\"") + path + "\"";
}

// A 128x192x192 uint8 volume with a deterministic ramp, covering the fixture surface.
void writeRampVolume(const fs::path& path)
{
    const std::filesystem::path volumeRoot = path;
    const vcr::zarr::CreateOptions levelZero{.shape = {128, 192, 192}, .chunks = {64, 64, 64}, .compressor = "", .overwrite = true};
    vcr::zarr::Writer::create(volumeRoot / "0", levelZero);
    std::vector<uint8_t> ramp(128 * 192 * 192);
    for (size_t z = 0; z < 128; ++z)
        for (size_t y = 0; y < 192; ++y)
            for (size_t x = 0; x < 192; ++x) ramp[(z * 192 + y) * 192 + x] = uint8_t((x + 2 * y + 3 * z) % 251);
    vcr::zarr::Writer::open(path / "0").writeRegion({0, 0, 0}, {128, 192, 192}, ramp.data());
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
    ~Fixture() { std::error_code ignored; if (!std::getenv("VC_KEEP_FIXTURE")) fs::remove_all(dir, ignored); }
    std::string storeArgs() const { return "--store \"" + store.string() + "\" --surface " + uuid; }
};
}

TEST_CASE("vc_tifxyz2obj writes the OBJ into the surface's artifact directory")
{
    Fixture f;
    const auto bin = tool("VC_TIFXYZ2OBJ_BIN");
    run(bin + " \"" + f.tifxyz.string() + "\" \"" + (f.dir / "file.obj").string() + "\" --normalize-uv", f.dir / "file.log");
    const auto output = run(bin + " " + f.storeArgs() + " --name mesh.obj --normalize-uv", f.dir / "store.log");
    const auto stored = fs::path(f.store.string() + ".artifacts") / f.uuid / "mesh.obj";
    CHECK(output.find("obj: " + stored.string()) != std::string::npos);
    CHECK(readFile(stored) == readFile(f.dir / "file.obj"));
}

TEST_CASE("vc_tifxyz2obj with --volume-uuid exports through the store's view of the surface")
{
    Fixture f;
    // "half" is "vol" at half resolution: the export is in its voxels.
    {
        OpenOptions options; options.author = "tool-store-test";
        auto db = ObjectStore::open(f.store, options);
        auto txn = db.write();
        db.registerVolumeTransform(txn, "vol", "half", VolumeAffine::scale(0.5), "test");
        txn.commit();
    }
    const auto bin = tool("VC_TIFXYZ2OBJ_BIN");
    run(bin + " " + f.storeArgs() + " --name full.obj", f.dir / "full.log");
    run(bin + " " + f.storeArgs() + " --name half.obj --volume-uuid=half", f.dir / "half.log");
    const auto artifacts = fs::path(f.store.string() + ".artifacts") / f.uuid;
    auto firstVertex = [](const fs::path& obj) {
        std::ifstream in(obj);
        for (std::string line; std::getline(in, line);)
            if (line.rfind("v ", 0) == 0) {
                std::istringstream fields(line.substr(2));
                cv::Vec3d v;
                fields >> v[0] >> v[1] >> v[2];
                return v;
            }
        return cv::Vec3d(-1, -1, -1);
    };
    const auto full = firstVertex(artifacts / "full.obj"), half = firstVertex(artifacts / "half.obj");
    for (int i = 0; i < 3; ++i) CHECK(half[i] == doctest::Approx(full[i] / 2));
}

TEST_CASE("vc_render_tifxyz renders a stored surface like its tifxyz")
{
    Fixture f;
    writeRampVolume(f.dir / "volume.zarr");
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

TEST_CASE("vc_render_tifxyz reads a stored surface through a registered volume like --affine on its tifxyz")
{
    Fixture f;
    writeRampVolume(f.dir / "volume.zarr");
    // An 8 degree turn about z with a shift, from "vol" to the rendered volume.
    VolumeAffine registration;
    const double c = std::cos(8 * M_PI / 180), s = std::sin(8 * M_PI / 180);
    registration(0, 0) = c; registration(0, 1) = -s; registration(1, 0) = s; registration(1, 1) = c;
    registration(0, 3) = 12.5; registration(1, 3) = -9.25; registration(2, 3) = 3;
    {
        OpenOptions options; options.author = "tool-store-test";
        auto db = ObjectStore::open(f.store, options);
        const auto uuid = util::newUuid();
        auto txn = db.write({{uuid, std::nullopt}});
        db.registerVolumeTransform(txn, "vol", "rendered", registration, "test registration", uuid);
        txn.commit();
    }
    std::ofstream(f.dir / "transform.json") << nlohmann::json{{"transformation_matrix", registration.json()}}.dump();
    const auto bin = tool("VC_RENDER_TIFXYZ_BIN") + " --volume \"" + (f.dir / "volume.zarr").string() +
        "\" --scale 1 --group-idx 0 --num-slices 5";
    run(bin + " --segmentation \"" + f.tifxyz.string() + "\" --affine \"" + (f.dir / "transform.json").string() +
        "\" --tif-output \"" + (f.dir / "file").string() + "\"", f.dir / "file.log");
    run(bin + " " + f.storeArgs() + " --volume-uuid rendered --tif-output \"" + (f.dir / "store").string() + "\"", f.dir / "store.log");
    // The view maps the grid before sampling and derives normals from it; the
    // file mode maps sampled points and normals. Both are the same affine
    // image of the surface, up to rounding.
    size_t files = 0, pixels = 0, differing = 0;
    double worst = 0;
    for (const auto& entry : fs::directory_iterator(f.dir / "file")) {
        const auto file = vc::core::util::readImageMat(entry.path());
        const auto store = vc::core::util::readImageMat(f.dir / "store" / entry.path().filename());
        REQUIRE(file.size() == store.size());
        cv::Mat diff;
        cv::absdiff(file, store, diff);
        double max = 0;
        cv::minMaxLoc(diff, nullptr, &max);
        worst = std::max(worst, max);
        differing += size_t(cv::countNonZero(diff));
        pixels += diff.total();
        ++files;
    }
    CHECK(files == 5);
    CHECK(worst <= 1);
    CHECK(differing * 100 <= pixels);

    // The registration is the store's: render refuses a second one on top.
    const auto refused = std::system((bin + " " + f.storeArgs() + " --volume-uuid rendered --affine \"" +
        (f.dir / "transform.json").string() + "\" --tif-output \"" + (f.dir / "both").string() + "\" > \"" +
        (f.dir / "both.log").string() + "\" 2>&1").c_str());
    CHECK(refused != 0);
}

namespace {
std::string surfaceUuid(const std::string& output)
{
    std::smatch match;
    REQUIRE(std::regex_search(output, match, std::regex("surface_uuid: ([0-9a-f-]{36})")));
    return match[1];
}
void checkSameSurface(const Fixture& f, const std::string& uuid, const fs::path& tifxyz, const std::string& tool)
{
    OpenOptions options; options.author = "tool-store-test";
    auto db = ObjectStore::open(f.store, options);
    auto txn = db.read();
    const auto stored = load_quad_from_store(db, txn, uuid);
    const auto files = load_quad_from_tifxyz(tifxyz);
    REQUIRE(stored->rawPointsPtr()->size() == files->rawPointsPtr()->size());
    CHECK(cv::norm(*stored->rawPointsPtr(), *files->rawPointsPtr(), cv::NORM_INF) == 0);
    CHECK(stored->_scale == files->_scale);
    const auto meta = nlohmann::json::parse(db.objectByUuid(txn, uuid)->meta);
    CHECK(meta.at("provenance").at("tool") == tool);
    CHECK(meta.at("coordinates").at("volume_uuid") == "vol");
    CHECK(db.objectByUuid(txn, uuid)->tags == std::vector<std::string>{"imported"});
}
}

TEST_CASE("OBJ converters create stored surfaces equal to their tifxyz output")
{
    Fixture f;
    run(tool("VC_TIFXYZ2OBJ_BIN") + " \"" + f.tifxyz.string() + "\" \"" + (f.dir / "mesh.obj").string() + "\"", f.dir / "obj.log");
    const std::string mesh = "\"" + (f.dir / "mesh.obj").string() + "\"";
    const std::string storeOptions = " --store=\"" + f.store.string() + "\" --tag=imported --volume-uuid=vol";

    const auto legacy = tool("VC_OBJ2TIFXYZ_LEGACY_BIN");
    run(legacy + " " + mesh + " \"" + (f.dir / "legacy.tifxyz").string() + "\" 4", f.dir / "legacy-file.log");
    const auto legacyUuid = surfaceUuid(run(legacy + " " + mesh + storeOptions + " 4", f.dir / "legacy-store.log"));
    checkSameSurface(f, legacyUuid, f.dir / "legacy.tifxyz", "vc_obj2tifxyz_legacy");

    const auto modern = tool("VC_OBJ2TIFXYZ_BIN");
    run(modern + " " + mesh + " \"" + (f.dir / "modern.tifxyz").string() + "\" --tifxyz-source=\"" + f.tifxyz.string() + "\"", f.dir / "modern-file.log");
    const auto modernUuid = surfaceUuid(run(modern + " " + mesh + storeOptions + " --source-surface=" + f.uuid, f.dir / "modern-store.log"));
    checkSameSurface(f, modernUuid, f.dir / "modern.tifxyz", "vc_obj2tifxyz");
}

TEST_CASE("vc_merge_patch edits a stored parent like a tifxyz parent")
{
    Fixture f;
    auto sheet = [](int rows, int cols, int row0, int col0, float lift) {
        cv::Mat_<cv::Vec3f> points(rows, cols);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                const float x = 200.f + 2.f * float(col0 + c), y = 200.f + 2.f * float(row0 + r);
                const bool interior = r > 6 && c > 6 && r < rows - 7 && c < cols - 7;
                points(r, c) = {x, y, 100.f + 0.01f * (x - 240.f) * (x - 240.f) + (interior ? lift : 0.f)};
                // A patch has an invalid frame around its valid region.
                if (lift != 0.f && (r < 2 || c < 2 || r >= rows - 2 || c >= cols - 2)) points(r, c) = {-1.f, -1.f, -1.f};
            }
        return QuadSurface(points, {0.5f, 0.5f});
    };
    auto parent = sheet(40, 40, 0, 0, 0.f), child = sheet(24, 24, 8, 8, 1.5f);
    parent.save(f.dir / "parent", true);
    child.save(f.dir / "child", true);
    const auto parentUuid = util::newUuid(), childUuid = util::newUuid();
    {
        OpenOptions options; options.author = "tool-store-test";
        auto db = ObjectStore::open(f.store, options);
        auto txn = db.write();
        for (const auto& [uuid, surface] : {std::pair{parentUuid, &parent}, std::pair{childUuid, &child}}) {
            ObjectRecord record; record.uuid = uuid; record.kind = "surface";
            db.upsertObject(txn, record);
            db.putGeometry(txn, uuid, quad_surface_geometry(*surface));
        }
        txn.commit();
    }
    const auto bin = tool("VC_MERGE_PATCH_BIN") + " --ransac-seed 7 --border-cells 4 --blend-cells 2";
    run(bin + " --parent \"" + (f.dir / "parent").string() + "\" --child \"" + (f.dir / "child").string() + "\"", f.dir / "file.log");
    const auto output = run(bin + " --store \"" + f.store.string() + "\" --surface " + parentUuid + " --surface " + childUuid +
                            " --explicit-roles", f.dir / "store.log");
    CHECK(output.find("geometry_revision: ") != std::string::npos);
    OpenOptions options; options.author = "tool-store-test";
    auto db = ObjectStore::open(f.store, options);
    auto txn = db.read();
    const auto stored = load_quad_from_store(db, txn, parentUuid);
    const auto files = load_quad_from_tifxyz(f.dir / "parent");
    REQUIRE(stored->rawPointsPtr()->size() == files->rawPointsPtr()->size());
    CHECK(cv::norm(*stored->rawPointsPtr(), *files->rawPointsPtr(), cv::NORM_INF) == 0);
    CHECK(cv::norm(*stored->rawPointsPtr(), *parent.rawPointsPtr(), cv::NORM_INF) > 0);  // the patch changed it
    CHECK(fs::exists(fs::path(f.store.string() + ".artifacts") / parentUuid / "merge_patch_summary.json"));
}

TEST_CASE("vc_merge_tifxyz stores the merge of stored surfaces like a tifxyz merge")
{
    Fixture f;
    auto plane = [](int xOffset) {
        cv::Mat_<cv::Vec3f> points(16, 16);
        for (int r = 0; r < 16; ++r)
            for (int c = 0; c < 16; ++c) points(r, c) = {float(c + xOffset), float(r), 100.f};
        return QuadSurface(points, {1.f, 1.f});
    };
    auto a = plane(0), b = plane(8);
    fs::create_directories(f.dir / "paths");
    a.save(f.dir / "paths" / "surface_a", true);
    b.save(f.dir / "paths" / "surface_b", true);
    const auto uuidA = util::newUuid(), uuidB = util::newUuid();
    {
        OpenOptions options; options.author = "tool-store-test";
        auto db = ObjectStore::open(f.store, options);
        auto txn = db.write();
        for (const auto& [uuid, surface, name] : {std::tuple{uuidA, &a, "surface_a"}, std::tuple{uuidB, &b, "surface_b"}}) {
            ObjectRecord record; record.uuid = uuid; record.kind = "surface"; record.name = name;
            record.meta = R"({"coordinates":{"volume_uuid":"vol","scale_factor":1}})";
            db.upsertObject(txn, record);
            db.putGeometry(txn, uuid, quad_surface_geometry(*surface));
        }
        txn.commit();
    }
    std::ofstream(f.dir / "merge.json") << R"({"rows":["surface_a surface_b"]})";
    std::ofstream(f.dir / "merge-store.json") << nlohmann::json{{"rows", {uuidA + " " + uuidB}}}.dump();
    const auto bin = tool("VC_MERGE_TIFXYZ_BIN");
    run(bin + " --merge \"" + (f.dir / "merge.json").string() + "\"", f.dir / "file.log");
    const auto uuid = surfaceUuid(run(bin + " --merge \"" + (f.dir / "merge-store.json").string() + "\" --store \"" +
                                      f.store.string() + "\" --tag merged --volume-uuid vol", f.dir / "store.log"));
    OpenOptions options; options.author = "tool-store-test";
    auto db = ObjectStore::open(f.store, options);
    auto txn = db.read();
    const auto stored = load_quad_from_store(db, txn, uuid);
    const auto files = load_quad_from_tifxyz(f.dir / "paths" / "surface_a_merged");
    REQUIRE(stored->rawPointsPtr()->size() == files->rawPointsPtr()->size());
    CHECK(cv::norm(*stored->rawPointsPtr(), *files->rawPointsPtr(), cv::NORM_INF) == 0);
    const auto object = db.objectByUuid(txn, uuid);
    CHECK(object->name == "surface_a_merged");
    CHECK(object->tags == std::vector<std::string>{"merged"});
    const auto meta = nlohmann::json::parse(object->meta);
    CHECK(meta.at("provenance").at("sources").size() == 2);
    const auto artifacts = fs::path(f.store.string() + ".artifacts") / uuid;
    CHECK(fs::exists(artifacts / "merge.obj"));
    CHECK(fs::exists(artifacts / "merge_summary.json"));
}

TEST_CASE("vc_objrefine refines a stored surface into a new object or in place")
{
    Fixture f;
    writeRampVolume(f.dir / "volume.zarr");
    std::ofstream(f.dir / "params.json") << R"({"refine": true, "start": -3, "stop": 3, "step": 1, "r": 1, "scale_group": "0", "reader_scale": 1})";
    const auto bin = tool("VC_OBJREFINE_BIN");
    const std::string tail = " \"" + (f.dir / "params.json").string() + "\"";
    const std::string volume = " \"" + (f.dir / "volume.zarr").string() + "\" ";
    run(bin + volume + "\"" + f.tifxyz.string() + "\" \"" + (f.dir / "refined").string() + "\"" + tail, f.dir / "file.log");
    const std::string store = bin + " " + f.storeArgs();
    const auto created = surfaceUuid(run(store + " --name refined" + volume + tail, f.dir / "store.log"));
    run(store + volume + tail, f.dir / "inplace.log");
    OpenOptions options; options.author = "tool-store-test";
    auto db = ObjectStore::open(f.store, options);
    auto txn = db.read();
    const auto files = load_quad_from_tifxyz(f.dir / "refined");
    for (const auto& uuid : {created, f.uuid}) {
        const auto stored = load_quad_from_store(db, txn, uuid);
        REQUIRE(stored->rawPointsPtr()->size() == files->rawPointsPtr()->size());
        CHECK(cv::norm(*stored->rawPointsPtr(), *files->rawPointsPtr(), cv::NORM_INF) == 0);
    }
    CHECK(db.objectByUuid(txn, created)->name == "refined");
}

TEST_CASE("vc_straighten stores its result as a new surface")
{
    Fixture f;
    const auto bin = tool("VC_STRAIGHTEN_BIN") + " --trim --trim-max-edge 4.03";
    run(bin + " \"" + f.tifxyz.string() + "\" \"" + (f.dir / "straight").string() + "\"", f.dir / "file.log");
    const auto uuid = surfaceUuid(run(bin + " " + f.storeArgs() + " --tag straight", f.dir / "store.log"));
    OpenOptions options; options.author = "tool-store-test";
    auto db = ObjectStore::open(f.store, options);
    auto txn = db.read();
    const auto stored = load_quad_from_store(db, txn, uuid);
    const auto files = load_quad_from_tifxyz(f.dir / "straight");
    REQUIRE(stored->rawPointsPtr()->size() == files->rawPointsPtr()->size());
    CHECK(cv::norm(*stored->rawPointsPtr(), *files->rawPointsPtr(), cv::NORM_INF) == 0);
    CHECK(stored->countValidPoints() < load_quad_from_store(db, txn, f.uuid)->countValidPoints());  // trimmed
    CHECK(db.objectByUuid(txn, uuid)->name == "surface_straight");
}

TEST_CASE("vc_grow_seg_from_segments grows a stored surface from stored sources")
{
    // The growth itself is not repeatable (unseeded rand() and OpenMP), so this
    // checks what store mode adds: stored sources in, a stored trace out.
    Fixture f;
    writeRampVolume(f.dir / "volume.zarr");
    auto plane = [](int xOffset) {
        cv::Mat_<cv::Vec3f> points(24, 24);
        for (int r = 0; r < 24; ++r)
            for (int c = 0; c < 24; ++c) points(r, c) = {60.f + 2.f * float(c + xOffset), 60.f + 2.f * float(r), 64.f};
        return QuadSurface(points, {0.5f, 0.5f});
    };
    auto a = plane(0), b = plane(12);
    const auto uuidA = util::newUuid(), uuidB = util::newUuid();
    {
        OpenOptions options; options.author = "tool-store-test";
        auto db = ObjectStore::open(f.store, options);
        auto txn = db.write();
        for (const auto& [uuid, surface] : {std::pair{uuidA, &a}, std::pair{uuidB, &b}}) {
            ObjectRecord record; record.uuid = uuid; record.kind = "surface"; record.tags = {"source"};
            db.upsertObject(txn, record);
            db.putGeometry(txn, uuid, quad_surface_geometry(*surface));
        }
        txn.commit();
    }
    std::ofstream(f.dir / "params.json") << R"({"generations": 3})";
    const auto output = run(tool("VC_GROW_SEG_FROM_SEGMENTS_BIN") + " --volume \"" + (f.dir / "volume.zarr").string() +
                            "\" --params \"" + (f.dir / "params.json").string() + "\" --store \"" + f.store.string() +
                            "\" --surface " + uuidA + " --source-tag source --tag trace", f.dir / "store.log");
    const auto uuid = surfaceUuid(output);
    OpenOptions options; options.author = "tool-store-test";
    auto db = ObjectStore::open(f.store, options);
    auto txn = db.read();
    CHECK(db.currentGeometry(txn, uuid).has_value());
    const auto object = db.objectByUuid(txn, uuid);
    CHECK(object->tags == std::vector<std::string>{"trace"});
    CHECK(nlohmann::json::parse(object->meta).at("provenance").at("sources") == nlohmann::json::array({uuidA}));
    // The tracer's debug snapshots land in the output's artifacts, not a target directory.
    CHECK(fs::is_directory(fs::path(f.store.string() + ".artifacts") / uuid / "grow_debug"));
}

TEST_CASE("flatboi flattens a stored surface and a tifxyz directory alike")
{
    const char* flatboi = std::getenv("VC_FLATBOI_BIN");
    if (!flatboi) {
        MESSAGE("flatboi is not built; skipped");
        return;
    }
    Fixture f;
    // One thread: PaStiX's threaded solve is not bitwise repeatable.
    const std::string env = "PASTIX_NUM_THREADS=1 OMP_NUM_THREADS=1 ";
    const auto bin = env + tool("VC_FLATBOI_BIN");
    const std::string store = " " + f.storeArgs();
    const auto uuid = surfaceUuid(run(bin + store + " --tag=flat 10", f.dir / "store.log"));
    const auto coarse = surfaceUuid(run(bin + store + " --keep=50 --name coarse 10", f.dir / "keep.log"));
    run(bin + " \"" + f.tifxyz.string() + "\" --output=\"" + (f.dir / "direct").string() + "\" 10", f.dir / "direct.log");

    OpenOptions options; options.author = "tool-store-test";
    auto db = ObjectStore::open(f.store, options);
    auto txn = db.read();
    const auto stored = load_quad_from_store(db, txn, uuid);
    const auto direct = load_quad_from_tifxyz(f.dir / "direct");
    REQUIRE(stored->countValidPoints() > 0);
    CHECK(cv::norm(*stored->rawPointsPtr(), *direct->rawPointsPtr(), cv::NORM_INF) == 0);
    CHECK(stored->_scale == direct->_scale);
    const auto object = db.objectByUuid(txn, uuid);
    CHECK(object->name == "surface_flatboi");
    CHECK(object->tags == std::vector<std::string>{"flat"});
    const auto meta = nlohmann::json::parse(object->meta);
    CHECK(meta.at("provenance").at("tool") == "flatboi");
    CHECK(meta.at("provenance").at("sources") == nlohmann::json::array({f.uuid}));
    CHECK(meta.at("coordinates").at("volume_uuid") == "vol");
    CHECK(fs::is_regular_file(fs::path(f.store.string() + ".artifacts") / uuid / "slim_energies_flatboi.txt"));

    // Decimated: SLIM on every other row and column, UVs lifted to the full grid.
    CHECK(load_quad_from_store(db, txn, coarse)->countValidPoints() > 0);
    CHECK(db.objectByUuid(txn, coarse)->name == "coarse");
}
