// vc_render_tifxyz's Zarr output: writing a pyramid, resuming, precreated parts and rotation.
// The tool's binary comes from VC_RENDER_TIFXYZ_BIN.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "support/LoopbackHttpServer.hpp"

#include <vc_reader/array.hpp>
#include <vc_reader/image_write.hpp>
#include <vc_reader/zarr.hpp>
#include <vc_reader/zarr_write.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Raw {
    vcr::Index shape;
    std::vector<std::uint8_t> bytes;
    bool operator==(const Raw&) const = default;
};

Raw readLevel(const fs::path& path)
{
    const auto opened = vcr::zarr::openArray(path.string(), {});
    Raw raw{opened.meta.shape, {}};
    std::int64_t count = opened.meta.dtype == "<u2" ? 2 : 1;
    for (const auto extent : raw.shape) count *= extent;
    raw.bytes.resize(std::size_t(count));
    opened.array->read_many({vcr::Index(raw.shape.size(), 0)}, {raw.shape}, {raw.bytes.data()});
    return raw;
}

void writeSurfaceAxis(const fs::path& path, int rows, int cols, auto&& value)
{
    std::vector<float> values(std::size_t(rows) * std::size_t(cols));
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x) values[std::size_t(y) * cols + x] = value(y, x);
    vcr::sources::writeTiff(path, values.data(), "<f4", rows, cols);
}

void removeChunks(const fs::path& level)
{
    std::vector<fs::path> chunks;
    for (const auto& entry : fs::recursive_directory_iterator(level))
        if (entry.is_regular_file() && !entry.path().filename().string().starts_with(".")) chunks.push_back(entry.path());
    for (const auto& chunk : chunks) fs::remove(chunk);
}

}  // namespace

TEST_CASE("vc_render_tifxyz writes, resumes and partitions a Zarr pyramid")
{
    const char* renderer = std::getenv("VC_RENDER_TIFXYZ_BIN");
    REQUIRE(renderer);
    for (const std::string dtype : {"|u1", "<u2"}) {
        for (const std::string separator : {"/", "."}) {
            CAPTURE(dtype);
            CAPTURE(separator);
            const auto root = fs::temp_directory_path() /
                              ("vc-render-zarr-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            const bool wide = dtype == "<u2";
            const auto volume = root / "volume";
            vcr::zarr::createGroup(volume);
            auto writer = vcr::zarr::Writer::create(volume / "0", {.shape = {16, 272, 272}, .chunks = {16, 272, 272}, .dtype = dtype, .compressor = ""});
            // Independent of z so the normal's direction does not change the reference.
            std::vector<std::uint8_t> voxels(std::size_t(16 * 272 * 272) * (wide ? 2 : 1));
            for (int z = 0; z < 16; ++z)
                for (int y = 0; y < 272; ++y)
                    for (int x = 0; x < 272; ++x) {
                        const auto index = std::size_t(((z * 272 + y) * 272) + x);
                        const int value = (wide ? 1000 : 10) + (y + x) % 101;
                        if (wide) {
                            const auto narrow = std::uint16_t(value);
                            std::copy_n(reinterpret_cast<const std::uint8_t*>(&narrow), 2, voxels.data() + index * 2);
                        } else {
                            voxels[index] = std::uint8_t(value);
                        }
                    }
            writer.writeChunk({0, 0, 0}, voxels.data(), voxels.size());

            const int width = 257, height = 259;
            const auto surface = root / "surface";
            fs::create_directories(surface);
            writeSurfaceAxis(surface / "x.tif", height, width, [](int, int x) { return float(x + 4); });
            writeSurfaceAxis(surface / "y.tif", height, width, [](int y, int) { return float(y + 4); });
            writeSurfaceAxis(surface / "z.tif", height, width, [](int, int) { return 8.0f; });
            std::ofstream(surface / "meta.json") << nlohmann::json{
                {"format", "tifxyz"}, {"type", "seg"}, {"uuid", "surface"}, {"scale", {1, 1}},
                {"bbox", {{4, 4, 8}, {width + 3, height + 3, 8}}}}.dump();

            const auto render = [&](const fs::path& out, const std::string& extra = "", bool pyramid = true) {
                const std::string command = std::string("OMP_NUM_THREADS=2 \"") + renderer + "\" -v \"" + volume.string() +
                    "\" -s \"" + surface.string() + "\" -g 0 --scale 1 --num-slices 3 --zarr-compressor none --zarr-separator " +
                    separator + " --pyramid " + (pyramid ? "true" : "false") + " --zarr-output \"" + out.string() + "\" " + extra +
                    " > \"" + (root / "log.txt").string() + "\" 2>&1";
                const int status = std::system(command.c_str());
                if (status != 0) {
                    std::ifstream log(root / "log.txt");
                    MESSAGE(command << "\n" << std::string(std::istreambuf_iterator<char>(log), {}));
                }
                REQUIRE(status == 0);
            };

            const auto out = root / "render";
            render(out);
            std::vector<Raw> reference;
            for (int level = 0; level < 6; ++level) reference.push_back(readLevel(out / std::to_string(level)));
            CHECK(vcr::zarr::openArray((out / "0").string(), {}).meta.dtype == dtype);
            CHECK(reference[0].shape == vcr::Index{3, height, width});
            CHECK(std::ranges::any_of(reference[0].bytes, [](std::uint8_t b) { return b != 0; }));

            // Resume reads stored level 0 chunks to reconstruct the inline pyramid.
            for (int level = 1; level < 6; ++level) removeChunks(out / std::to_string(level));
            render(out, "--resume");
            for (int level = 0; level < 6; ++level) CHECK(readLevel(out / std::to_string(level)) == reference[level]);

            // --pre creates arrays; separate workers reopen them for writes.
            const auto parts = root / "parts";
            render(parts, "--pre");
            for (int part = 0; part < 2; ++part)
                render(parts, "--num-parts 2 --part-id " + std::to_string(part), false);
            CHECK(readLevel(parts / "0") == reference[0]);

            // Rotation forces the on-disk pyramid builder instead of inline accumulation.
            const std::string crop = "--crop-width 256 --crop-height 256";
            render(root / "aligned", crop);
            render(root / "rotated", crop + " --rotate 180");
            const auto aligned = readLevel(root / "aligned" / "0");
            const auto rotated = readLevel(root / "rotated" / "0");
            REQUIRE(rotated.shape == aligned.shape);
            const std::size_t element = wide ? 2 : 1, plane = std::size_t(256 * 256);
            bool reversed = true;
            for (std::size_t z = 0; z < 3; ++z)
                for (std::size_t i = 0; i < plane; ++i)
                    reversed = reversed && std::memcmp(rotated.bytes.data() + (z * plane + i) * element,
                                                       aligned.bytes.data() + (z * plane + plane - 1 - i) * element, element) == 0;
            CHECK(reversed);
            const auto coarse = readLevel(root / "rotated" / "5");
            CHECK(std::ranges::any_of(coarse.bytes, [](std::uint8_t b) { return b != 0; }));
            std::error_code ignored;
            fs::remove_all(root, ignored);
        }
    }
}

TEST_CASE("vc_render_tifxyz exits non-zero when a remote volume cannot be read")
{
    const char* renderer = std::getenv("VC_RENDER_TIFXYZ_BIN");
    REQUIRE(renderer);
    const auto root = fs::temp_directory_path() /
                      ("vc-render-remote-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto store = root / "store" / "ct.zarr";
    vcr::zarr::createGroup(store);
    auto writer = vcr::zarr::Writer::create(store / "0", {.shape = {16, 64, 64}, .chunks = {16, 64, 64}, .compressor = ""});
    const std::vector<std::uint8_t> voxels(16 * 64 * 64, 100);
    writer.writeChunk({0, 0, 0}, voxels.data(), voxels.size());
    const int width = 32, height = 32;
    const auto surface = root / "surface";
    fs::create_directories(surface);
    writeSurfaceAxis(surface / "x.tif", height, width, [](int, int x) { return float(x + 4); });
    writeSurfaceAxis(surface / "y.tif", height, width, [](int y, int) { return float(y + 4); });
    writeSurfaceAxis(surface / "z.tif", height, width, [](int, int) { return 8.0f; });
    std::ofstream(surface / "meta.json") << nlohmann::json{
        {"format", "tifxyz"}, {"type", "seg"}, {"uuid", "surface"}, {"scale", {1, 1}},
        {"bbox", {{4, 4, 8}, {width + 3, height + 3, 8}}}}.dump();

    vc::test::LoopbackHttpServer server;
    server.serveDirectory(root / "store");
    server.setHandler([](const vc::test::LoopbackHttpServer::Request& request)
                          -> std::optional<vc::test::LoopbackHttpServer::Response> {
        // Metadata is served; the chunk fails.
        if (request.path.ends_with("/0/0.0.0")) return vc::test::LoopbackHttpServer::Response{.status = 500};
        return std::nullopt;
    });
    const auto config = root / "config";
    fs::create_directories(config);
    std::ofstream(config / "cache.json") << nlohmann::json{{"version", 1}, {"root", (root / "cache").string()}}.dump();
    const std::string command = "VC3D_CONFIG_DIR=\"" + config.string() + "\" \"" + renderer + "\" -v " + server.url() + "/ct.zarr -s \"" +
                                surface.string() + "\" -g 0 --scale 1 --zarr-compressor none --pyramid false --zarr-output \"" +
                                (root / "render").string() + "\" > \"" + (root / "log.txt").string() + "\" 2>&1";
    CHECK(std::system(command.c_str()) != 0);
    CHECK(server.requestCount("/ct.zarr/0/0.0.0") > 0);
    std::error_code ignored;
    fs::remove_all(root, ignored);
}
