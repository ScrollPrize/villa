// vc_grow_seg_from_seed in store mode: grow a surface into an object store, then
// resume from the stored surface into a new object. With a fixed seed and one
// thread the stored surface equals the tifxyz output of the same run after
// applying that format's negative-coordinate validity rule. A
// neighbor copy projects and refines in one run, creating one surface.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <vc_object_store/objects/GeometryIngest.hpp>
#include <vc_object_store/objects/StoreReader.hpp>
#include <doctest/doctest.h>

#include <vc_reader/zarr.hpp>
#include <vc_reader/zarr_write.hpp>
#include <vc_reader/tifxyz.hpp>
#include "vc/core/types/Volume.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include "vc/core/util/QuadSurface.hpp"
#include "vc/tracer/Tracer.hpp"
#include <vc_object_store/Uuid.hpp>

#include <opencv2/core.hpp>

#include <cstdlib>
#include <regex>
#include <sstream>
#include <tuple>
#include <utility>

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

// A 128^3 volume with a bright 9-voxel-thick sheet centred on each z in `centres`.
void writeSheets(const fs::path& dir, const std::vector<int>& centres)
{
    fs::create_directories(dir);
    const std::filesystem::path volumeRoot = dir / "volume.zarr";
    const vcr::zarr::CreateOptions levelZero{.shape = {128, 128, 128}, .chunks = {64, 64, 64}, .compressor = "", .overwrite = true};
    vcr::zarr::Writer::create(volumeRoot / "0", levelZero);
    std::ofstream(volumeRoot / "meta.json") << nlohmann::json{{"type", "vol"}, {"uuid", "test-volume"}, {"name", "test-volume"}, {"format", "zarr"}, {"width", 128}, {"height", 128}, {"slices", 128}, {"voxelsize", 1.0}}.dump();
    std::vector<uint8_t> sheet(128 * 128 * 128, 0);
    for (const int centre : centres)
        for (int z = centre - 4; z <= centre + 4; ++z)
            std::fill_n(sheet.begin() + size_t(z) * 128 * 128, 128 * 128, uint8_t(255 - 20 * std::abs(z - centre)));
    vcr::zarr::Writer::open(dir / "volume.zarr" / "0").writeRegion({0, 0, 0}, {128, 128, 128}, sheet.data());
}

// Odd dimensions exercise a coarse grid whose extent is not an exact multiple
// of the scale factor. Empty padding and one excluded, populated cell coexist.
void writeMaskedPlane(const fs::path& path, cv::Point excluded, float hiddenZ = 64.f,
                      cv::Size size = {15, 13})
{
    cv::Mat_<cv::Vec3f> points(size, cv::Vec3f(-1, -1, -1));
    cv::Mat_<uint8_t> mask(points.size(), uint8_t(0));
    cv::Mat_<uint16_t> generations(points.size(), uint16_t(0));
    for (int r = 2; r < points.rows - 2; ++r)
        for (int c = 2; c < points.cols - 2; ++c) {
            points(r, c) = {float(20 + 4 * c), float(20 + 4 * r), 64.f};
            mask(r, c) = 255;
            generations(r, c) = 1;
        }
    points(excluded)[2] = hiddenZ;
    mask(excluded) = 0;
    QuadSurface surface(points, {0.25f, 0.25f});
    surface.setChannel("mask", mask);
    surface.setChannel("generations", generations);
    surface.save(path, true);
}
}

TEST_CASE("vc_grow_seg_from_seed grows into and resumes from an object store")
{
    const char* tool = std::getenv("VC_GROW_SEG_FROM_SEED_BIN");
    REQUIRE(tool);
    const fs::path dir = fs::temp_directory_path() / ("vc-grow-store-" + util::newUuid());
    writeSheets(dir, {64});
    OpenOptions options; options.author = "grow-store-test";
    { auto store = ObjectStore::create(dir / "objects.vcobj", options); }
    std::ofstream(dir / "params.json") << nlohmann::json{{"generations", 6}, {"thread_limit", 1}, {"min_area_cm", 0}}.dump();
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
        CHECK(load_quad_from_store(vc::objects::LocalStoreReader(store, txn), uuid)->countValidPoints() > 4);
    }
    const auto stored = load_quad_from_store(vc::objects::LocalStoreReader(store, txn), grown);
    const auto files = load_quad_from_tifxyz(dir / "files");
    REQUIRE(stored->rawPointsPtr()->size() == files->rawPointsPtr()->size());
    auto inFileFormat = stored->rawPointsPtr()->clone();
    for (auto& point : inFileFormat)
        if (!vcr::tifxyz::vertexValid(point.val)) point = {-1.f, -1.f, -1.f};
    CHECK(cv::norm(inFileFormat, *files->rawPointsPtr(), cv::NORM_INF) == 0);
    CHECK(nlohmann::json::parse(store.objectByUuid(txn, resumed)->meta).at("provenance").at("sources") ==
          nlohmann::json::array({grown}));
    std::error_code ignored;
    fs::remove_all(dir, ignored);
}

TEST_CASE("A neighbor copy projects and refines in one run and creates one surface")
{
    const char* tool = std::getenv("VC_GROW_SEG_FROM_SEED_BIN");
    REQUIRE(tool);
    const fs::path dir = fs::temp_directory_path() / ("vc-grow-neighbor-" + util::newUuid());
    writeSheets(dir, {40, 90});
    OpenOptions options; options.author = "grow-store-test";
    { auto store = ObjectStore::create(dir / "objects.vcobj", options); }
    std::ofstream(dir / "grow.json") << nlohmann::json{{"generations", 6}, {"thread_limit", 1}, {"min_area_cm", 0}}.dump();
    std::ofstream(dir / "copy.json") << nlohmann::json{
        {"mode", "gen_neighbor"}, {"neighbor_dir", "out"}, {"neighbor_max_distance", 80},
        {"refine", {{"max_gen", 1}, {"generations", 1}, {"thread_limit", 1}, {"min_area_cm", 0}}}}.dump();
    const std::string common = std::string("\"") + tool + "\" --volume \"" + (dir / "volume.zarr").string() +
        "\" --store \"" + (dir / "objects.vcobj").string() + "\" --volume-uuid test-volume --tag paths";

    const auto grown = run(common + " --params \"" + (dir / "grow.json").string() + "\" --rng-seed 1 --seed 64 64 40",
                           dir / "grow.log");
    const auto copied = run(common + " --params \"" + (dir / "copy.json").string() + "\" --surface " + grown,
                            dir / "copy.log");

    auto store = ObjectStore::open(dir / "objects.vcobj", options);
    auto txn = store.read();
    CHECK(store.selectObjects(txn, {}).size() == 2);
    const auto meta = nlohmann::json::parse(store.objectByUuid(txn, copied)->meta);
    CHECK(meta.at("provenance").at("sources") == nlohmann::json::array({grown}));
    CHECK(meta.at("vc_gsfs_mode") == "resume");  // the projection was refined in the same run
    // The source's vertices outside this small volume are invalid, which leaves one quad to copy.
    CHECK(load_quad_from_store(vc::objects::LocalStoreReader(store, txn), copied)->countValidPoints() >= 4);
    std::error_code ignored;
    fs::remove_all(dir, ignored);
}

TEST_CASE("A TIFXYZ import resumes with an optional GrowPatch anchor")
{
    bool request_anchor = false;
    bool mask_seed = false;
    SUBCASE("Default resume does not activate the imported anchor") {}
    SUBCASE("Default resume preserves a deliberately masked seed") { mask_seed = true; }
    SUBCASE("An explicit anchor survives growth, saving and reloading") { request_anchor = true; }
    SUBCASE("An explicit anchor rejects a deliberately masked seed") {
        request_anchor = true;
        mask_seed = true;
    }

    const char* tool = std::getenv("VC_GROW_SEG_FROM_SEED_BIN");
    REQUIRE(tool);
    const fs::path dir = fs::temp_directory_path() / ("vc-grow-import-anchor-" + util::newUuid());
    writeSheets(dir, {64});
    const nlohmann::json anchor = {64, 64, 64};
    nlohmann::json params{{"generations", 4}, {"thread_limit", 1}, {"min_area_cm", 0},
                         {"seed_anchor_xyz", anchor}};
    std::ofstream(dir / "grow.json") << params;
    const std::string common = std::string("\"") + tool + "\" --volume \"" + (dir / "volume.zarr").string() +
        "\" --rng-seed 1";
    REQUIRE(std::system((common + " --params \"" + (dir / "grow.json").string() + "\" --seed 64 64 64 --target-dir \"" +
                         (dir / "tifxyz").string() + "\" --segment-name anchored > \"" +
                         (dir / "grow.log").string() + "\" 2>&1").c_str()) == 0);

    cv::Point seed_grid(-1, -1);
    if (mask_seed) {
        auto surface = load_quad_from_tifxyz(dir / "tifxyz");
        auto mask = surface->validMask();
        for (const auto [row, col, point] : surface->validPoints()) {
            if (point == cv::Vec3f(64, 64, 64)) {
                seed_grid = {col, row};
                mask(row, col) = 0;
            }
        }
        REQUIRE(seed_grid.x >= 0);
        surface->setChannel("mask", mask);
        surface->save(dir / "tifxyz", true);
    }

    OpenOptions options; options.author = "grow-store-test";
    const auto dbPath = dir / "objects.vcobj";
    std::string imported;
    int initial_valid = 0;
    {
        auto store = ObjectStore::create(dbPath, options);
        const auto report = ingestGeometrySource(store, "surface", dir / "tifxyz", {},
            R"({"volume_uuid":"test-volume","scale_factor":1})");
        REQUIRE(report.problems.empty());
        REQUIRE(report.uuids.size() == 1);
        imported = report.uuids.front();
        auto txn = store.read();
        const auto meta = nlohmann::json::parse(store.objectByUuid(txn, imported)->meta);
        CHECK_FALSE(meta.contains("seed_anchor"));
        CHECK(meta.at("source").at("seed_anchor").at("xyz") == anchor);
        initial_valid = load_quad_from_store(vc::objects::LocalStoreReader(store, txn), imported)->countValidPoints();
    }

    params["generations"] = 5;
    if (!request_anchor) params.erase("seed_anchor_xyz");
    std::ofstream(dir / "resume.json") << params;
    const std::string resume_command = common + " --params \"" + (dir / "resume.json").string() + "\" --store \"" +
        dbPath.string() + "\" --volume-uuid test-volume --resume-opt skip --surface ";
    if (request_anchor && mask_seed) {
        CHECK(std::system((resume_command + imported + " > \"" + (dir / "resume.log").string() + "\" 2>&1").c_str()) != 0);
        std::stringstream output;
        output << std::ifstream(dir / "resume.log").rdbuf();
        CHECK(output.str().find("seed anchor was lost or moved") != std::string::npos);
    } else {
        const auto resumed = run(resume_command + imported, dir / "resume.log");
        int resumed_valid = 0;
        {
            auto store = ObjectStore::open(dbPath, options);
            auto txn = store.read();
            const auto object = store.objectByUuid(txn, resumed);
            REQUIRE(object.has_value());
            const auto meta = nlohmann::json::parse(object->meta);
            auto surface = load_quad_from_store(vc::objects::LocalStoreReader(store, txn), resumed);
            resumed_valid = surface->countValidPoints();
            CHECK(resumed_valid > initial_valid);
            if (request_anchor) {
                CHECK(meta.at("seed_anchor").at("xyz") == anchor);
            } else {
                CHECK_FALSE(meta.contains("seed_anchor"));
            }
            if (mask_seed) {
                const auto offset = meta.at("grid_offset").get<std::vector<int>>();
                const auto mask = surface->channel("mask", SURF_CHANNEL_NORESIZE);
                REQUIRE_FALSE(mask.empty());
                const cv::Point target(seed_grid.x + offset[0], seed_grid.y + offset[1]);
                CHECK(mask.at<uint8_t>(target) == 0);
                const vc::objects::LocalStoreReader reader(store, txn);
                const auto selected = reader.currentObject(resumed);
                REQUIRE(selected && selected->geometryRevision);
                const auto array = reader.readGeometryRectangle(*selected->geometryRevision, "generations",
                                                                target.y, target.x, 1, 1);
                CHECK(array == std::vector<uint8_t>{1, 0});
            }
        }

        if (request_anchor) {
            // Reopening the just-written native result must retain its opt-in contract.
            params.erase("seed_anchor_xyz");
            std::ofstream(dir / "resume.json") << params;
            CHECK(std::system((resume_command + resumed + " > \"" + (dir / "guard.log").string() + "\" 2>&1").c_str()) != 0);
            std::stringstream output;
            output << std::ifstream(dir / "guard.log").rdbuf();
            CHECK(output.str().find("anchored resume requires seed_anchor_xyz") != std::string::npos);
            params["seed_anchor_xyz"] = anchor;
            params["generations"] = 6;
            std::ofstream(dir / "resume.json") << params;
            const auto reloaded = run(resume_command + resumed, dir / "reload.log");
            auto store = ObjectStore::open(dbPath, options);
            auto txn = store.read();
            CHECK(nlohmann::json::parse(store.objectByUuid(txn, reloaded)->meta).at("seed_anchor").at("xyz") == anchor);
            auto surface = load_quad_from_store(vc::objects::LocalStoreReader(store, txn), reloaded);
            CHECK(surface->countValidPoints() > resumed_valid);
            int seed_points = 0;
            for (const auto [row, col, point] : surface->validPoints())
                if (point == cv::Vec3f(64, 64, 64)) ++seed_points;
            CHECK(seed_points == 1);
        }
    }
    std::error_code ignored;
    fs::remove_all(dir, ignored);
}

TEST_CASE("Coarse interpolation does not use an excluded vertex's hidden coordinates")
{
    const fs::path dir = fs::temp_directory_path() / ("vc-grow-coarse-hidden-" + util::newUuid());
    writeSheets(dir, {64});
    const cv::Point excluded(4, 4);  // sampled by the factor-two working grid
    writeMaskedPlane(dir / "low", excluded, 64.f);
    writeMaskedPlane(dir / "high", excluded, 364.f);
    auto low = load_quad_from_tifxyz(dir / "low");
    auto high = load_quad_from_tifxyz(dir / "high");
    REQUIRE(low->rawPointsPtr()->at<cv::Vec3f>(excluded) == cv::Vec3f(-1, -1, -1));
    REQUIRE(high->rawPointsPtr()->at<cv::Vec3f>(excluded) == cv::Vec3f(-1, -1, -1));
    vcr::zarr::OpenOptions openOptions;
    openOptions.memoryCache = std::make_shared<vcr::MemoryCache>(16 * 1024 * 1024);
    auto volume = vcr::zarr::openPyramidLevel((dir / "volume.zarr").string(), 0, openOptions);
    const nlohmann::json params{{"generations", 1}, {"growth_scale", 1}, {"resume_opt", "skip"},
                                {"disable_grid_expansion", true}, {"thread_limit", 1}, {"min_area_cm", 0}};
    // With no added generation this isolates output interpolation from solving.
    std::unique_ptr<QuadSurface> a(tracer(*volume.array, 1.f, {64, 64, 64}, params, 1.f, low.get()));
    std::unique_ptr<QuadSurface> b(tracer(*volume.array, 1.f, {64, 64, 64}, params, 1.f, high.get()));
    REQUIRE(a);
    REQUIRE(b);
    const auto& aPoints = *std::as_const(*a).rawPointsPtr();
    const auto& bPoints = *std::as_const(*b).rawPointsPtr();
    REQUIRE(aPoints.size() == bPoints.size());
    CHECK(a->meta.at("grid_offset") == b->meta.at("grid_offset"));
    int compared = 0;
    for (int r = 0; r < aPoints.rows; ++r)
        for (int c = 0; c < aPoints.cols; ++c) {
            const bool aValid = vc::geometry::validSurfacePoint(aPoints(r, c));
            const bool bValid = vc::geometry::validSurfacePoint(bPoints(r, c));
            CHECK(aValid == bValid);
            if (aValid && bValid) {
                CHECK(aPoints(r, c) == bPoints(r, c));
                ++compared;
            }
        }
    CHECK(compared > 20);
    const auto offset = a->meta.at("grid_offset").get<std::vector<int>>();
    const cv::Point target = excluded + cv::Point(offset[0], offset[1]);
    REQUIRE(cv::Rect(cv::Point(), aPoints.size()).contains(target));
    CHECK(aPoints(target) == cv::Vec3f(-1, -1, -1));
    CHECK(bPoints(target) == cv::Vec3f(-1, -1, -1));
    std::error_code ignored;
    fs::remove_all(dir, ignored);
}

TEST_CASE("An off-lattice exclusion survives coarse store output and full-resolution resume")
{
    const char* tool = std::getenv("VC_GROW_SEG_FROM_SEED_BIN");
    REQUIRE(tool);
    const fs::path dir = fs::temp_directory_path() / ("vc-grow-coarse-exclusion-" + util::newUuid());
    writeSheets(dir, {64});
    const cv::Point excluded(2, 5);  // left edge, skipped by factor-two sampling
    writeMaskedPlane(dir / "source", excluded);
    OpenOptions options; options.author = "grow-store-test";
    const auto dbPath = dir / "objects.vcobj";
    std::string imported;
    {
        auto store = ObjectStore::create(dbPath, options);
        const auto report = ingestGeometrySource(store, "surface", dir / "source", {},
            R"({"volume_uuid":"test-volume","scale_factor":1})");
        REQUIRE(report.problems.empty());
        REQUIRE(report.uuids.size() == 1);
        imported = report.uuids.front();
    }
    nlohmann::json params{{"generations", 3}, {"growth_scale", 1}, {"growth_directions", {"right"}},
                          {"thread_limit", 1}, {"min_area_cm", 0}};
    std::ofstream(dir / "params.json") << params;
    const std::string common = std::string("\"") + tool + "\" --volume \"" + (dir / "volume.zarr").string() +
        "\" --params \"" + (dir / "params.json").string() + "\" --store \"" + dbPath.string() +
        "\" --volume-uuid test-volume --rng-seed 1 --resume-opt skip --surface ";
    const auto coarse = run(common + imported, dir / "coarse.log");
    cv::Point coarseCell;
    int coarseValid = 0;
    const auto checkExcluded = [&](const std::string& uuid, cv::Point sourceCell) {
        auto store = ObjectStore::open(dbPath, options);
        auto txn = store.read();
        const LocalStoreReader reader(store, txn);
        const auto selected = reader.currentObject(uuid);
        REQUIRE(selected && selected->geometryRevision);
        const auto metadata = nlohmann::json::parse(selected->record.meta);
        const auto offset = metadata.at("grid_offset").get<std::vector<int>>();
        const cv::Point target = sourceCell + cv::Point(offset[0], offset[1]);
        const auto surface = load_quad_from_store(reader, uuid);
        REQUIRE(cv::Rect(cv::Point(), surface->rawPointsPtr()->size()).contains(target));
        CHECK(surface->rawPointsPtr()->at<cv::Vec3f>(target) == cv::Vec3f(-1, -1, -1));
        CHECK(reader.readGeometryRectangle(*selected->geometryRevision, "growth_exclusions", target.y, target.x, 1, 1) ==
              std::vector<uint8_t>{255});
        CHECK(reader.readGeometryRectangle(*selected->geometryRevision, "valid", target.y, target.x, 1, 1) ==
              std::vector<uint8_t>{0});
        return std::pair(target, surface->countValidPoints());
    };
    std::tie(coarseCell, coarseValid) = checkExcluded(coarse, excluded);
    // Reopen at the original resolution, growing back towards the excluded edge.
    params["growth_scale"] = 0;
    params["generations"] = 5;
    params["growth_directions"] = {"left"};
    std::ofstream(dir / "params.json") << params;
    const auto fine = run(common + coarse, dir / "fine.log");
    const auto [fineCell, fineValid] = checkExcluded(fine, coarseCell);
    CHECK(fineValid > coarseValid);
    std::error_code ignored;
    fs::remove_all(dir, ignored);
}

TEST_CASE("Explicit selected repair does not inherit an unselected coarse-cell exclusion")
{
    const fs::path dir = fs::temp_directory_path() / ("vc-grow-selected-repair-" + util::newUuid());
    writeSheets(dir, {64});
    // Both cells occupy one factor-two or factor-four working cell. Only the
    // off-lattice cell is selected; the sampled cell must remain excluded.
    const cv::Point unselected(8, 8);
    const cv::Point selected(9, 9);
    writeMaskedPlane(dir / "source", unselected, 64.f, {21, 21});
    auto source = load_quad_from_tifxyz(dir / "source");
    auto mask = source->vertexMask();
    auto points = source->pointsForStorage();
    mask(selected) = 0;
    QuadSurface input(points, source->scale());
    input.setChannel("mask", mask);
    input.setChannel("generations", source->channel("generations").clone());
    input.save(dir / "source", true);

    vcr::zarr::OpenOptions openOptions;
    openOptions.memoryCache = std::make_shared<vcr::MemoryCache>(16 * 1024 * 1024);
    auto volume = vcr::zarr::openPyramidLevel((dir / "volume.zarr").string(), 0, openOptions);
    cv::Mat_<uint8_t> selection(points.size(), uint8_t(0));
    selection(selected) = 255;
    const auto shiftedCell = [](const QuadSurface& result, cv::Point inputCell) {
        const auto offset = result.meta.at("grid_offset").get<std::vector<int>>();
        return inputCell + cv::Point(offset[0], offset[1]);
    };
    for (const int level : {0, 1, 2}) {
        INFO("growth_scale=" << level);
        nlohmann::json params{{"generations", 3}, {"growth_scale", level}, {"resume_opt", "skip"},
                              {"disable_grid_expansion", true}, {"thread_limit", 1}, {"min_area_cm", 0}};
        // An ordinary allowed-growth mask is a limit, not permission to undo
        // exclusions. Manual Add supplies the explicit repair opt-in below.
        auto constrainedInput = load_quad_from_tifxyz(dir / "source");
        std::unique_ptr<QuadSurface> constrained(tracer(*volume.array, 1.f, {64, 64, 64}, params,
                                                      1.f, constrainedInput.get(), {}, {}, {}, &selection));
        REQUIRE(constrained);
        const cv::Point constrainedSelected = shiftedCell(*constrained, selected);
        REQUIRE(cv::Rect(cv::Point{}, constrained->rawPointsPtr()->size()).contains(constrainedSelected));
        CHECK_FALSE(vc::geometry::validSurfacePoint(constrained->rawPointsPtr()->at<cv::Vec3f>(constrainedSelected)));
        CHECK(constrained->vertexMask()(constrainedSelected) == 0);
        CHECK(constrained->channel("growth_exclusions").at<uint8_t>(constrainedSelected) == 255);

        params["repair_selected_exclusions"] = true;
        auto repairInput = load_quad_from_tifxyz(dir / "source");
        std::unique_ptr<QuadSurface> repaired(tracer(*volume.array, 1.f, {64, 64, 64}, params,
                                                   1.f, repairInput.get(), {}, {}, {}, &selection));
        REQUIRE(repaired);
        const cv::Point repairedSelected = shiftedCell(*repaired, selected);
        const cv::Point repairedUnselected = shiftedCell(*repaired, unselected);
        REQUIRE(cv::Rect(cv::Point{}, repaired->rawPointsPtr()->size()).contains(repairedSelected));
        REQUIRE(cv::Rect(cv::Point{}, repaired->rawPointsPtr()->size()).contains(repairedUnselected));
        CHECK(vc::geometry::validSurfacePoint(repaired->rawPointsPtr()->at<cv::Vec3f>(repairedSelected)));
        CHECK(repaired->vertexMask()(repairedSelected) == 255);
        CHECK(repaired->channel("growth_exclusions").at<uint8_t>(repairedSelected) == 0);
        CHECK_FALSE(vc::geometry::validSurfacePoint(repaired->rawPointsPtr()->at<cv::Vec3f>(repairedUnselected)));
        CHECK(repaired->vertexMask()(repairedUnselected) == 0);
        CHECK(repaired->channel("growth_exclusions").at<uint8_t>(repairedUnselected) == 255);
        CHECK(repaired->pointsForStorage()(repairedUnselected) == points(unselected));
    }
    std::error_code ignored;
    fs::remove_all(dir, ignored);
}

TEST_CASE("Inpaint retains solved coordinates and clears repaired exclusions")
{
    const fs::path dir = fs::temp_directory_path() / ("vc-grow-inpaint-repair-" + util::newUuid());
    writeSheets(dir, {64});
    const cv::Point hole(8, 8);
    // A deliberately incorrect hidden coordinate must not overwrite the
    // geometry reconstructed from the surrounding sheet.
    writeMaskedPlane(dir / "source", hole, 84.f, {21, 21});
    {
        auto initial = load_quad_from_tifxyz(dir / "source");
        auto mask = initial->vertexMask();
        mask(2, 2) = 0;  // boundary indentation, not a closed Inpaint hole
        QuadSurface input(initial->pointsForStorage(), initial->scale());
        input.setChannel("mask", mask);
        input.setChannel("generations", initial->channel("generations").clone());
        input.save(dir / "source", true);
    }
    auto source = load_quad_from_tifxyz(dir / "source");
    vcr::zarr::OpenOptions openOptions;
    openOptions.memoryCache = std::make_shared<vcr::MemoryCache>(16 * 1024 * 1024);
    auto volume = vcr::zarr::openPyramidLevel((dir / "volume.zarr").string(), 0, openOptions);
    const nlohmann::json params{{"generations", 3}, {"inpaint", true}, {"resume_opt", "skip"},
                                {"thread_limit", 1}, {"min_area_cm", 0}};
    std::unique_ptr<QuadSurface> result(tracer(*volume.array, 1.f, {64, 64, 64}, params, 1.f, source.get()));
    REQUIRE(result);
    const auto solved = result->rawPointsPtr()->at<cv::Vec3f>(hole);
    CHECK(vc::geometry::validSurfacePoint(solved));
    CHECK(std::abs(solved[2] - 64.f) < 1.f);
    CHECK(result->vertexMask()(hole) == 255);
    CHECK(result->channel("growth_exclusions").at<uint8_t>(hole) == 0);
    CHECK_FALSE(vc::geometry::validSurfacePoint(result->rawPointsPtr()->at<cv::Vec3f>(2, 2)));
    CHECK(result->vertexMask()(2, 2) == 0);
    CHECK(result->channel("growth_exclusions").at<uint8_t>(2, 2) == 255);
    std::error_code ignored;
    fs::remove_all(dir, ignored);
}

TEST_CASE("Coarse Inpaint leaves unsolved off-lattice exclusions masked")
{
    const fs::path dir = fs::temp_directory_path() / ("vc-grow-inpaint-unsolved-" + util::newUuid());
    writeSheets(dir, {64});
    const cv::Point hole(9, 9);
    writeMaskedPlane(dir / "source", hole, 84.f, {21, 21});
    vcr::zarr::OpenOptions openOptions;
    openOptions.memoryCache = std::make_shared<vcr::MemoryCache>(16 * 1024 * 1024);
    auto volume = vcr::zarr::openPyramidLevel((dir / "volume.zarr").string(), 0, openOptions);
    for (const int level : {1, 2}) {
        INFO("growth_scale=" << level);
        auto source = load_quad_from_tifxyz(dir / "source");
        const auto original = source->pointsForStorage()(hole);
        const nlohmann::json params{{"generations", 3}, {"inpaint", true}, {"growth_scale", level},
                                    {"thread_limit", 1}, {"min_area_cm", 0}};
        std::unique_ptr<QuadSurface> result(tracer(*volume.array, 1.f, {64, 64, 64}, params, 1.f, source.get()));
        REQUIRE(result);
        // Nearest coarse sampling never saw this hole. Interpolation alone
        // must not imply that an Inpaint solve repaired the exclusion.
        CHECK_FALSE(vc::geometry::validSurfacePoint(result->rawPointsPtr()->at<cv::Vec3f>(hole)));
        CHECK(result->vertexMask()(hole) == 0);
        CHECK(result->channel("growth_exclusions").at<uint8_t>(hole) == 255);
        CHECK(result->pointsForStorage()(hole) == original);
    }
    std::error_code ignored;
    fs::remove_all(dir, ignored);
}
