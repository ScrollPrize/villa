// vc_grow_seg_from_seed in store mode: grow a surface into an object store, then
// resume from the stored surface into a new object. With a fixed seed and one
// thread the stored surface equals the tifxyz output of the same run.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/Array3D.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include <vc_object_store/Uuid.hpp>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

namespace fs = std::filesystem;
using namespace vc::objects;

namespace {
std::string run(const std::string& command, const fs::path& log)
{
    const int status = std::system((command + " > \"" + log.string() + "\" 2>&1").c_str());
    std::stringstream output;
    output << std::ifstream(log).rdbuf();
    INFO(output.str());
    REQUIRE(status == 0);
    std::smatch match;
    const auto text = output.str();
    REQUIRE(std::regex_search(text, match, std::regex("surface_uuid: ([0-9a-f-]{36})")));
    return match[1];
}
}

TEST_CASE("vc_grow_seg_from_seed grows into and resumes from an object store")
{
    const char* tool = std::getenv("VC_GROW_SEG_FROM_SEED_BIN");
    REQUIRE(tool);
    const fs::path dir = fs::temp_directory_path() / ("vc-grow-store-" + util::newUuid());
    fs::create_directories(dir);
    {
        Volume::ZarrCreateOptions options;
        options.shapeZYX = {128, 128, 128};
        options.compressor = "none";
        options.uuid = "test-volume";
        options.name = "test-volume";
        auto volume = Volume::New(dir / "volume.zarr", options);
        Array3D<uint8_t> sheet({128, 128, 128}, 0);
        for (size_t z = 60; z <= 68; ++z)
            for (size_t y = 0; y < 128; ++y)
                for (size_t x = 0; x < 128; ++x) sheet(z, y, x) = uint8_t(255 - 20 * std::abs(int(z) - 64));
        volume->writeZYX(sheet, {0, 0, 0});
    }
    OpenOptions options; options.author = "grow-store-test";
    { auto store = ObjectStore::create(dir / "objects.vcobj", options); }
    std::ofstream(dir / "params.json") << nlohmann::json{{"generations", 6}, {"thread_limit", 1}, {"min_area_cm", 0},
                                                          {"cache_root", (dir / "cache").string()}}.dump();
    const std::string base = std::string("\"") + tool + "\" --volume \"" + (dir / "volume.zarr").string() +
        "\" --params \"" + (dir / "params.json").string() + "\" --rng-seed 1 --seed 64 64 64";
    const std::string common = base + " --store \"" + (dir / "objects.vcobj").string() + "\" --volume-uuid test-volume --tag paths";

    const auto grown = run(common, dir / "grow.log");
    const auto resumed = run(common + " --surface " + grown, dir / "resume.log");
    REQUIRE(std::system((base + " --target-dir \"" + (dir / "files").string() + "\" --segment-name grown > \"" +
                         (dir / "files.log").string() + "\" 2>&1").c_str()) == 0);

    auto store = ObjectStore::open(dir / "objects.vcobj", options);
    auto txn = store.read();
    for (const auto& uuid : {grown, resumed}) {
        const auto object = store.objectByUuid(txn, uuid);
        REQUIRE(object.has_value());
        CHECK(object->kind == "surface");
        CHECK(object->tags == std::vector<std::string>{"paths"});
        const auto meta = nlohmann::json::parse(object->meta);
        CHECK(meta.at("coordinates").at("volume_uuid") == "test-volume");
        CHECK(meta.at("provenance").at("tool") == "vc_grow_seg_from_seed");
        CHECK(load_quad_from_store(store, txn, uuid)->countValidPoints() > 4);
    }
    const auto stored = load_quad_from_store(store, txn, grown);
    const auto files = load_quad_from_tifxyz(dir / "files");
    REQUIRE(stored->rawPointsPtr()->size() == files->rawPointsPtr()->size());
    CHECK(cv::norm(*stored->rawPointsPtr(), *files->rawPointsPtr(), cv::NORM_INF) == 0);
    CHECK(nlohmann::json::parse(store.objectByUuid(txn, resumed)->meta).at("provenance").at("sources") ==
          nlohmann::json::array({grown}));
    std::error_code ignored;
    fs::remove_all(dir, ignored);
}
