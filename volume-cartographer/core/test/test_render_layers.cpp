// vc::sampleRenderLayers against vc_render_tifxyz: a window of layers sampled through the
// shared helper holds the bytes the tool writes for the same pixels of a whole render.
// The tool's binary comes from VC_RENDER_TIFXYZ_BIN.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/RenderLayers.hpp"

#include <vc_reader/affine.hpp>
#include <vc_reader/array.hpp>
#include <vc_reader/image_write.hpp>
#include <vc_reader/zarr.hpp>
#include <vc_reader/zarr_write.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// A curved surface of 31 x 22 vertices stored at scale 0.1: a 310 x 220 canvas at
// level 0, several renderer tiles. It has a hole inside, a cut corner, a ragged right
// edge and a notch in the bottom edge.
constexpr int kCols = 31, kRows = 22;
const cv::Rect kHole(12, 8, 5, 4);  // vertices

bool invalidVertex(int r, int c)
{
    return kHole.contains({c, r}) || r + c < 3 || (c == kCols - 1 && r >= 15) || (r == kRows - 1 && c >= 4 && c < 9);
}

void writeSurface(const fs::path& dir)
{
    fs::create_directories(dir);
    const auto axis = [&](const char* name, auto&& value) {
        std::vector<float> values(std::size_t(kRows) * kCols);
        for (int r = 0; r < kRows; ++r)
            for (int c = 0; c < kCols; ++c)
                values[std::size_t(r) * kCols + c] = invalidVertex(r, c) ? -1.0f : float(value(double(r), double(c)));
        vcr::sources::writeTiff(dir / name, values.data(), "<f4", kRows, kCols);
    };
    axis("x.tif", [](double r, double c) { return 8.0 + 10.0 * c + 3.0 * std::sin(r * 0.4); });
    axis("y.tif", [](double r, double c) { return 6.0 + 10.0 * r + 2.0 * std::cos(c * 0.3); });
    axis("z.tif", [](double r, double c) { return 26.0 + 8.0 * std::sin(c * 0.25) * std::cos(r * 0.2) + 0.012 * c * r; });
    std::ofstream(dir / "meta.json") << nlohmann::json{
        {"format", "tifxyz"}, {"type", "seg"}, {"uuid", "surface"}, {"scale", {0.1, 0.1}},
        {"bbox", {{5, 4, 14}, {311, 218, 44}}}}.dump();
}

// A volume of two pyramid levels in several chunks. Its values change along every axis,
// so a shifted pixel or a swapped layer shows. The surface runs past its +x face.
void writeVolume(const fs::path& root, bool wide)
{
    const int depth = 56, height = 236, width = 300;
    vcr::zarr::createGroup(root);
    auto writer = vcr::zarr::Writer::create(
        root / "0", {.shape = {depth, height, width}, .chunks = {32, 64, 64}, .dtype = wide ? "<u2" : "|u1", .compressor = ""});
    std::vector<std::uint8_t> voxels(std::size_t(depth) * height * width * (wide ? 2 : 1));
    for (int z = 0; z < depth; ++z)
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                const auto index = (std::size_t(z) * height + y) * width + x;
                const int value = (x * 7 + y * 13 + z * 31 + ((x * y) % 17) * 5 + ((z * x) % 11) * 3) % 251;
                if (wide) {
                    const auto word = std::uint16_t(value * 257 + (x + y + z) % 256);
                    std::memcpy(voxels.data() + index * 2, &word, 2);
                } else {
                    voxels[index] = std::uint8_t(value);
                }
            }
    writer.writeRegion({0, 0, 0}, {depth, height, width}, voxels.data());
    vcr::zarr::writePyramid(root, {.levels = 1});
}

// Level 0 of a render: layers, rows, columns.
struct Render {
    vcr::Index shape;
    std::size_t element = 1;
    std::vector<std::uint8_t> bytes;
};

Render readRender(const fs::path& zarr)
{
    const auto opened = vcr::zarr::openArray((zarr / "0").string(), {});
    Render render{opened.meta.shape, opened.meta.type.itemsize, {}};
    std::size_t count = render.element;
    for (const auto extent : render.shape) count *= std::size_t(extent);
    render.bytes.resize(count);
    opened.array->read_many({vcr::Index(render.shape.size(), 0)}, {render.shape}, {render.bytes.data()});
    return render;
}

// The bytes of `pixels` in every layer of a render, layers first.
std::vector<std::uint8_t> windowOf(const Render& render, const cv::Rect& pixels)
{
    const auto height = std::size_t(render.shape[1]), width = std::size_t(render.shape[2]);
    std::vector<std::uint8_t> out;
    for (std::size_t k = 0; k < std::size_t(render.shape[0]); ++k)
        for (int y = pixels.y; y < pixels.y + pixels.height; ++y) {
            const auto* row = render.bytes.data() + ((k * height + std::size_t(y)) * width + std::size_t(pixels.x)) * render.element;
            out.insert(out.end(), row, row + std::size_t(pixels.width) * render.element);
        }
    return out;
}

std::vector<std::uint8_t> bytesOf(const cv::Mat& layers)
{
    return {layers.data, layers.data + layers.total() * layers.elemSize()};
}

// Pixels that read 0 in every layer: the surface has no point there.
std::size_t emptyPixels(const std::vector<std::uint8_t>& window, int layers)
{
    const std::size_t plane = window.size() / std::size_t(layers);
    std::size_t empty = 0;
    for (std::size_t p = 0; p < plane; ++p) {
        bool any = false;
        for (int k = 0; k < layers && !any; ++k) any = window[std::size_t(k) * plane + p] != 0;
        empty += any ? 0 : 1;
    }
    return empty;
}

}  // namespace

TEST_CASE("renderLayerOffsets centers the layers on the surface")
{
    CHECK(vc::renderLayerOffsets(1, 1.0) == std::vector<float>{0.0f});
    CHECK(vc::renderLayerOffsets(5, 1.0) == std::vector<float>{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
    CHECK(vc::renderLayerOffsets(4, 2.0) == std::vector<float>{-3.0f, -1.0f, 1.0f, 3.0f});
    CHECK(vc::renderLayerOffsets(2, 1.0, {0.0f, 0.5f}) == std::vector<float>{-0.5f, 0.0f, 0.5f, 1.0f});
}

TEST_CASE("windows sampled by sampleRenderLayers equal the same pixels of a vc_render_tifxyz render")
{
    const char* renderer = std::getenv("VC_RENDER_TIFXYZ_BIN");
    REQUIRE(renderer);
    const auto root = fs::temp_directory_path() /
                      ("vc-render-layers-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto surfaceDir = root / "surface";
    writeSurface(surfaceDir);
    // The surface as VC3D holds it: invalid vertices keep tifxyz's -1 marker.
    const auto surf = load_quad_from_tifxyz(surfaceDir);
    REQUIRE(surf);
    const int layers = 9;
    const auto offsets = vc::renderLayerOffsets(layers, 1.0);

    struct Case {
        bool wide;
        int group;
        bool flip;
        const char* scale = "1";  // --scale
    };
    // At --scale 0.7 the rounding of a tile's raster depends on where the tile starts, so
    // a window sampled from its own origin, not in the renderer's tiles, differs from the render.
    for (const Case c : {Case{false, 0, false}, Case{false, 0, true}, Case{true, 0, false}, Case{false, 1, false},
                         Case{false, 0, false, "0.7"}}) {
        const std::string name = std::string(c.wide ? "u16" : "u8") + "_g" + std::to_string(c.group) + "_s" + c.scale +
                                 (c.flip ? "_flip" : "");
        const auto volume = root / (c.wide ? "volume_u16" : "volume_u8");
        if (!fs::exists(volume)) writeVolume(volume, c.wide);

        // The whole canvas, rendered by the tool.
        const auto out = root / ("render_" + name);
        const std::string command = std::string("OMP_NUM_THREADS=2 \"") + renderer + "\" -v \"" + volume.string() + "\" -s \"" +
            surfaceDir.string() + "\" -g " + std::to_string(c.group) + " --scale " + c.scale + " --num-slices " + std::to_string(layers) +
            (c.flip ? " --flip-normals" : "") + " --zarr-compressor none --pyramid false --zarr-output \"" + out.string() +
            "\" > \"" + (root / "log.txt").string() + "\" 2>&1";
        const int status = std::system(command.c_str());
        if (status != 0) {
            std::ifstream log(root / "log.txt");
            MESSAGE(command << "\n" << std::string(std::istreambuf_iterator<char>(log), {}));
        }
        REQUIRE(status == 0);
        const Render whole = readRender(out);

        // The same surface, level and canvas through the helper.
        vcr::zarr::OpenOptions options;
        options.memoryCache = std::make_shared<vcr::MemoryCache>(std::size_t(64) << 20);
        const auto working = vcr::zarr::openPyramidLevel(volume.string(), 0, options);
        const auto rendered = vcr::zarr::openPyramidLevel(volume.string(), c.group, options);
        const auto map = vc::renderLevelMap(working.placement, rendered.placement);
        const auto canvas = vc::RenderCanvas::make(surf->rawPointsPtr()->size(), surf->scale(),
                                                   double(std::stof(c.scale)) * double(map.voxelScale()));
        const int W = canvas.size.width, H = canvas.size.height;
        REQUIRE(whole.shape == vcr::Index{layers, H, W});
        REQUIRE(whole.element == (c.wide ? 2u : 1u));
        const auto surface = vc::renderSurface(*surf, vcr::grid::Interpolation::Bicubic, 1.0f, false, {}, map);
        const vcr::grid::Level level{rendered.array.get(), rendered.meta.type};
        const auto sample = [&](const cv::Rect& pixels, bool flip) {
            return bytesOf(vc::sampleRenderLayers(*surf, surface, level, canvas, pixels, offsets, flip));
        };

        // The hole on the canvas: the cells next to an invalid vertex have no point either.
        const cv::Vec2d a = canvas.gridToCanvas({double(kHole.x - 1), double(kHole.y - 1)});
        const cv::Vec2d b = canvas.gridToCanvas({double(kHole.x + kHole.width), double(kHole.y + kHole.height)});
        const cv::Rect hole(int(a[0]), int(a[1]), int(b[0]) - int(a[0]), int(b[1]) - int(a[1]));

        struct Window {
            const char* what;
            cv::Rect pixels;
            bool empty;  // holds pixels without a surface point
        };
        const Window windows[] = {
            {"the whole canvas", {0, 0, W, H}, true},
            {"several tiles from an off-tile origin", {37, 21, W - 63, H - 34}, true},
            {"around the hole", {hole.x - 7, hole.y - 7, hole.width + 18, hole.height + 18}, true},
            {"the cut corner at the canvas origin", {0, 0, 45, 33}, true},
            {"the ragged edge at the far canvas corner", {W - 53, H - 47, 53, 47}, true},
            {"the notch in the bottom canvas edge", {19, H - 30, 90, 30}, true},
            {"two columns across a tile border", {vc::kRenderTile - 1, 0, 2, H}, false},
        };
        for (const auto& w : windows) {
            const auto window = sample(w.pixels, c.flip);
            REQUIRE(window.size() == std::size_t(layers) * std::size_t(w.pixels.area()) * whole.element);
            CHECK_MESSAGE(window == windowOf(whole, w.pixels), name << ": " << w.what << " differs from the render");
            const std::size_t empty = emptyPixels(window, layers), plane = window.size() / std::size_t(layers);
            CHECK_MESSAGE(empty < plane, name << ": " << w.what << " holds no data");
            if (w.empty) {
                CHECK_MESSAGE(empty > 0, name << ": " << w.what << " holds no empty pixel");
            }
            // Control: the window one pixel to the side is a different window.
            if (w.pixels.width == W) continue;
            const cv::Rect shifted = w.pixels + cv::Point(w.pixels.x + w.pixels.width < W ? 1 : -1, 0);
            CHECK_MESSAGE(window != windowOf(whole, shifted), name << ": " << w.what << " equals the render shifted by a pixel");
        }

        // Control: negated normals reverse the layers, so the same window is a different one.
        const cv::Rect pixels = windows[1].pixels;
        const auto reversed = sample(pixels, !c.flip);
        const auto forward = windowOf(whole, pixels);
        CHECK(reversed != forward);
        const std::size_t plane = forward.size() / std::size_t(layers);
        bool mirrored = true;
        for (int k = 0; k < layers; ++k)
            mirrored = mirrored && std::equal(reversed.begin() + std::ptrdiff_t(std::size_t(k) * plane),
                                              reversed.begin() + std::ptrdiff_t(std::size_t(k + 1) * plane),
                                              forward.begin() + std::ptrdiff_t(std::size_t(layers - 1 - k) * plane));
        CHECK(mirrored);

        if (c.group == 0 && !c.wide && !c.flip) {
            CHECK_THROWS_AS((void)sample({W - 10, 0, 11, 5}, false), std::invalid_argument);
            CHECK_THROWS_AS((void)sample({0, 0, 0, 5}, false), std::invalid_argument);
            // A canvas of another grid, a surface whose origin was moved off the canvas, and a
            // level that carries the placement the surface already holds.
            const auto other = vc::RenderCanvas::make({kCols + 1, kRows}, surf->scale(), canvas.renderScale);
            CHECK_THROWS_AS((void)vc::sampleRenderLayers(*surf, surface, level, other, pixels, offsets), std::invalid_argument);
            const auto moved = load_quad_from_tifxyz(surfaceDir);
            moved->shiftSurfaceOrigin({3.0, 0.0});
            CHECK_THROWS_AS((void)vc::sampleRenderLayers(*moved, surface, level, canvas, pixels, offsets), std::invalid_argument);
            auto placed = level;
            placed.scale = {0.5, 0.5, 0.5};
            CHECK_THROWS_AS((void)vc::sampleRenderLayers(*surf, surface, placed, canvas, pixels, offsets), std::invalid_argument);
            const std::atomic<bool> cancel{true};
            CHECK_THROWS_AS((void)vc::sampleRenderLayers(*surf, surface, level, canvas, pixels, offsets, false, &cancel),
                            vcr::Cancelled);
        }
    }
    std::error_code ignored;
    fs::remove_all(root, ignored);
}
