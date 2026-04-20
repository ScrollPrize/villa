#include "test.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "utils/Json.hpp"

#include "vc/core/types/VcDataset.hpp"
#include "vc/core/util/GridStore.hpp"

namespace {

namespace fs = std::filesystem;

class ScopedTempDir {
public:
    ScopedTempDir()
    {
        static std::atomic<unsigned long long> counter{0};
        const auto suffix = std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count())
            + "_" + std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
        path_ = fs::temp_directory_path() / ("vc_test_vc_gen_normalgrids_" + suffix);
        fs::create_directories(path_);
    }

    ~ScopedTempDir()
    {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }

    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

void writeJson(const fs::path& path, const utils::Json& json)
{
    std::ofstream out(path);
    out << json.dump(2) << '\n';
}

void createInputVolume(const fs::path& root)
{
    fs::create_directories(root);
    writeJson(root / "meta.json", {
        {"uuid", "vol-test"},
        {"name", "vol-test"},
        {"type", "vol"},
        {"format", "zarr"},
        {"width", 32},
        {"height", 32},
        {"slices", 32},
        {"voxelsize", 1.0},
        {"min", 0.0},
        {"max", 255.0}
    });

    auto ds = vc::createZarrDataset(root, "0", {32, 32, 32}, {16, 16, 16}, vc::VcDtype::uint8, "none", "/");
    std::vector<uint8_t> volume(32 * 32 * 32, 0);
    auto idx = [](int z, int y, int x) {
        return static_cast<size_t>((z * 32 + y) * 32 + x);
    };
    for (int z = 8; z < 24; ++z) {
        for (int y = 8; y < 24; ++y) {
            for (int x = 8; x < 24; ++x) {
                volume[idx(z, y, x)] = 255;
            }
        }
    }
    ds->writeRegion({0, 0, 0}, {32, 32, 32}, volume.data());
}

std::string quote(const fs::path& path)
{
    return "\"" + path.string() + "\"";
}

} // namespace

TEST(GenNormalGridsSmoke, GeneratesGridOutputsFromTinyZarr)
{
    ScopedTempDir tempDir;
    const auto inputRoot = tempDir.path() / "input.zarr";
    const auto outputRoot = tempDir.path() / "normal_grids";
    const auto metricsPath = tempDir.path() / "metrics.json";
    createInputVolume(inputRoot);

    std::string cmd = std::string(VC_GEN_NORMALGRIDS_BIN)
        + " generate -i " + quote(inputRoot)
        + " -o " + quote(outputRoot)
        + " --sparse-volume 4"
        + " --preview-every 0"
        + " --metrics-json " + quote(metricsPath)
        + " --level 0";

    ASSERT_EQ(std::system(cmd.c_str()), 0);

    ASSERT_TRUE(fs::is_directory(outputRoot / "xy"));
    ASSERT_TRUE(fs::is_directory(outputRoot / "xz"));
    ASSERT_TRUE(fs::is_directory(outputRoot / "yz"));
    ASSERT_TRUE(fs::exists(metricsPath));

    bool foundNonEmpty = false;
    for (const auto& plane : {"xy", "xz", "yz"}) {
        for (const auto& entry : fs::directory_iterator(outputRoot / plane)) {
            if (entry.is_regular_file() && entry.path().extension() == ".grid" && fs::file_size(entry.path()) > 0) {
                vc::core::util::GridStore grid(entry.path().string());
                EXPECT_GT(grid.numSegments(), 0u);
                foundNonEmpty = true;
                break;
            }
        }
    }
    EXPECT_TRUE(foundNonEmpty);
}
