// vcr::tifxyz: a surface written and read back, listing, retargeting, points between vertices.
#include "vc_reader/image_write.hpp"
#include "vc_reader/source_reader.hpp"
#include "vc_reader/tifxyz.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace vcr;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

static tifxyz::Surface plane(int rows, int cols, double scale, const std::string& uuid)
{
    std::vector<float> xyz(std::size_t(rows) * std::size_t(cols) * 3);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            float* p = &xyz[(std::size_t(r) * std::size_t(cols) + std::size_t(c)) * 3];
            p[0] = float(c) * 2 + 10, p[1] = float(r) * 3 + 20, p[2] = 30;
        }
    tifxyz::Metadata metadata;
    metadata.uuid = uuid;
    metadata.scale = {scale, scale};
    metadata.extra = {{"name", "a plane"}};
    return tifxyz::Surface(rows, cols, std::move(xyz), {}, std::move(metadata));
}

int main()
{
    const auto root = fs::temp_directory_path() / ("vc_reader_tifxyz_" + std::to_string(::getpid()));
    fs::create_directories(root);
    for (const auto& compression : {std::string("none"), std::string("lzw"), std::string("deflate")})
        for (const auto tile : {std::optional<int>(32), std::optional<int>()}) {
            auto surface = plane(40, 50, 0.05, "");
            tifxyz::WriteOptions options;
            options.compression = compression;
            options.tile = tile;
            const auto path = root / ("s_" + compression + (tile ? "_tiled" : "_strip"));
            tifxyz::write(path, surface, options);
            CHECK(!surface.metadata().uuid.empty());
            const auto back = tifxyz::read(path.string());
            CHECK(back.xyz() == surface.xyz());
            CHECK(back.valid() == surface.valid());
            CHECK(back.metadata().uuid == surface.metadata().uuid);
            CHECK(back.metadata().extra["name"] == "a plane");
            CHECK(back.metadata().extra["format"] == "tifxyz");
            CHECK(back.metadata().bbox);
            CHECK((*back.metadata().bbox)[3] == 108.0);
            CHECK((back.fullShape() == std::array<std::int64_t, 2>{800, 1000}));
        }
    // Existing directories need overwrite, which replaces them.
    {
        auto surface = plane(4, 5, 1, "u");
        const auto path = root / "exists";
        fs::create_directories(path);
        std::ofstream(path / "old.txt") << "old";
        bool raised = false;
        try {
            tifxyz::write(path, surface);
        } catch (const fs::filesystem_error& e) {
            raised = e.code() == std::errc::file_exists;
        }
        CHECK(raised);
        tifxyz::WriteOptions options;
        options.overwrite = true;
        tifxyz::write(path, surface, options);
        CHECK(!fs::exists(path / "old.txt"));
        CHECK(fs::exists(path / "x.tif"));
    }
    // Validity: mask, negative and non-finite coordinates; all three coordinates of a missing vertex are -1.
    {
        std::vector<float> xyz(4 * 3, 5.f);
        xyz[3] = -2.f;
        xyz[7] = NAN;
        std::vector<std::uint8_t> mask{1, 1, 1, 0};
        tifxyz::Surface surface(2, 2, std::move(xyz), std::move(mask), {});
        CHECK((surface.valid() == std::vector<std::uint8_t>{1, 0, 0, 0}));
        for (int a = 0; a < 3; ++a) CHECK(surface.xyz()[std::size_t(3 + a)] == -1.f);
    }
    // A -1 in a stored box is replaced by the real extent.
    {
        auto surface = plane(4, 5, 1, "marker");
        const auto path = root / "tree" / "marker";
        tifxyz::write(path, surface);
        auto document = nlohmann::json::parse(std::ifstream(path / "meta.json"));
        document["bbox"] = {{-1, -1, -1}, {-1, -1, -1}};
        std::ofstream(path / "meta.json") << document.dump();
        CHECK(tifxyz::hasMissingMarker(*tifxyz::readMetadata(path.string()).bbox));
        const auto box = *tifxyz::read(path.string()).metadata().bbox;
        CHECK(box[2] == 30.0 && box[5] == 30.0);
        // Listing: marker resolved only to filter on it; nested; a box outside the range is left out.
        auto far = plane(4, 5, 1, "nested");
        tifxyz::write(root / "tree" / "deeper" / "nested", far);
        CHECK(tifxyz::list((root / "tree").string()).size() == 2);
        CHECK(tifxyz::list((root / "tree").string(), false).size() == 1);
        CHECK(tifxyz::list((root / "tree").string(), true, std::array<double, 2>{0, 29}).empty());
        CHECK(tifxyz::list((root / "tree").string(), true, std::array<double, 2>{29, 31}).size() == 2);
    }
    // Retarget.
    {
        const auto half = plane(4, 5, 0.5, "r").retargeted(2);
        CHECK(half.xyz()[0] == 5.f && half.xyz()[1] == 10.f && half.xyz()[2] == 15.f);
        CHECK(half.metadata().scale[0] == 1.0);
        CHECK(half.metadata().bbox);
        CHECK(!half.metadata().area);
    }
    // Points between vertices on a plane, in both modes; none outside the grid; normals are unit, column x row.
    {
        const auto surface = plane(6, 8, 1, "p");
        const std::vector<float> xy{2.5f, 3.25f, 7.f, 5.f, 8.5f, 1.f};
        for (const auto mode : {grid::Interpolation::Linear, grid::Interpolation::Smooth}) {
            std::vector<float> p(9), n(9);
            surface.pointsAt(xy.data(), 3, mode, p.data(), n.data());
            CHECK(std::fabs(p[0] - 15.f) < 1e-5f && std::fabs(p[1] - 29.75f) < 1e-5f && std::fabs(p[2] - 30.f) < 1e-5f);
            CHECK(std::fabs(p[3] - 24.f) < 1e-5f && std::fabs(p[4] - 35.f) < 1e-5f);
            CHECK(std::isnan(p[6]));
            CHECK(std::fabs(n[2] - 1.f) < 1e-5f);
            std::vector<float> r(3);
            surface.resample(grid::Raster{2.5, 3.25, 1, 1, 1, 1}, mode, r.data());
            CHECK(r[0] == p[0] && r[1] == p[1]);
        }
    }
    // Plain images: strips, tiles in any order, page stacks, zero pages, first page, mappable or not.
    {
        namespace src = vcr::sources;
        const auto path = (root / "image.tif").string();
        std::vector<std::uint16_t> image(3 * 40 * 50);
        for (std::size_t i = 0; i < image.size(); ++i) image[i] = std::uint16_t(i);
        src::TiffOptions pages;
        pages.pages = 3;
        pages.compression = "lzw";
        src::writeTiff(path, image.data(), "<u2", 40, 50, 1, pages);
        auto whole = src::readImage(path);
        CHECK(whole.shape == (std::vector<std::int64_t>{3, 40, 50}) && whole.dtype == "<u2");
        CHECK(std::memcmp(whole.data.get(), image.data(), whole.nbytes) == 0);
        auto second = src::readImagePage(path, 1);
        CHECK(second.shape == (std::vector<std::int64_t>{40, 50}));
        CHECK(reinterpret_cast<const std::uint16_t*>(second.data.get())[0] == image[2000]);
        CHECK(src::imageSource(path).axes == (std::vector<std::string>{"z", "y", "x"}));
        bool refused = false;
        try {
            (void)src::tiffMapping(path);
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        CHECK(refused);

        const auto tiled = (root / "tiled.tif").string();
        src::TiffOptions tiles;
        tiles.tile = std::array<int, 2>{32, 32};
        tiles.compression = "zstd";
        {
            src::TiffWriter writer(tiled, "<u2", 40, 50, 1, tiles);
            std::vector<std::uint16_t> tile(8 * 18, 7);
            writer.writeTile(32, 32, tile.data(), 8, 18);
            writer.finish();
        }
        auto back = src::readImage(tiled);
        const auto* values = reinterpret_cast<const std::uint16_t*>(back.data.get());
        CHECK(values[0] == 0 && values[39 * 50 + 49] == 7);
        CHECK(!fs::exists(tiled + "." + std::to_string(::getpid()) + ".tmp"));

        const auto blank = (root / "blank.tif").string();
        {
            src::TiffWriter writer(blank, "|u1", 3000, 200);
            writer.writeBlankPage();
            writer.finish();
        }
        const auto mapping = src::tiffMapping(blank);
        CHECK(mapping.shape == (std::vector<std::int64_t>{3000, 200}) && mapping.dtype == "|u1" && mapping.offset > 0);
    }
    fs::remove_all(root);
    return 0;
}
