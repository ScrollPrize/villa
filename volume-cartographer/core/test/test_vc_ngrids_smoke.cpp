#include "test.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vc/core/types/VcDataset.hpp"

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
        path_ = fs::temp_directory_path() / ("vc_test_vc_ngrids_" + suffix);
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

void writeJson(const fs::path& path, const nlohmann::json& json)
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
        {"width", 64},
        {"height", 64},
        {"slices", 64},
        {"voxelsize", 1.0},
        {"min", 0.0},
        {"max", 255.0}
    });

    auto ds = vc::createZarrDataset(root, "0", {64, 64, 64}, {16, 16, 16}, vc::VcDtype::uint8, "none", "/");
    std::vector<uint8_t> volume(64 * 64 * 64, 0);
    auto idx = [](int z, int y, int x) {
        return static_cast<size_t>((z * 64 + y) * 64 + x);
    };
    for (int z = 16; z < 48; ++z) {
        for (int y = 16; y < 48; ++y) {
            for (int x = 16; x < 48; ++x) {
                volume[idx(z, y, x)] = 255;
            }
        }
    }
    ds->writeRegion({0, 0, 0}, {64, 64, 64}, volume.data());
}

std::string quote(const fs::path& path)
{
    return "\"" + path.string() + "\"";
}

std::vector<uint8_t> readDataset(const fs::path& path)
{
    vc::VcDataset ds(path);
    const auto& shape = ds.shape();
    std::vector<uint8_t> data(shape[0] * shape[1] * shape[2], 0);
    ds.readRegion({0, 0, 0}, {shape[0], shape[1], shape[2]}, data.data());
    return data;
}

void expectDatasetsEqual(const fs::path& aRoot, const fs::path& bRoot, const std::vector<std::string>& groups)
{
    for (const auto& group : groups) {
        EXPECT_EQ(readDataset(aRoot / group / "0"), readDataset(bRoot / group / "0"));
    }
}

size_t countValidNormals(const fs::path& zarrRoot)
{
    const auto x = readDataset(zarrRoot / "x" / "0");
    const auto y = readDataset(zarrRoot / "y" / "0");
    const auto z = readDataset(zarrRoot / "z" / "0");

    size_t valid = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        if (!(x[i] == 128 && y[i] == 128 && z[i] == 128)) {
            ++valid;
        }
    }
    return valid;
}

} // namespace

TEST(NGridsSmoke, FitAndAlignNormalsAreDeterministicSingleThreaded)
{
    ScopedTempDir tempDir;
    const auto inputRoot = tempDir.path() / "input.zarr";
    const auto gridsRoot = tempDir.path() / "normal_grids";
    const auto fit1Root = tempDir.path() / "fit1.zarr";
    const auto fit2Root = tempDir.path() / "fit2.zarr";
    const auto align1Root = tempDir.path() / "align1.zarr";
    const auto align2Root = tempDir.path() / "align2.zarr";

    createInputVolume(inputRoot);

    std::string gen_cmd = std::string("OMP_NUM_THREADS=1 ")
        + VC_GEN_NORMALGRIDS_BIN
        + " generate -i " + quote(inputRoot)
        + " -o " + quote(gridsRoot)
        + " --sparse-volume 4"
        + " --preview-every 0"
        + " --level 0";
    ASSERT_EQ(std::system(gen_cmd.c_str()), 0);

    std::string fit1_cmd = std::string("OMP_NUM_THREADS=1 ")
        + VC_NGRIDS_BIN
        + " -i " + quote(gridsRoot)
        + " --fit-normals --output-zarr " + quote(fit1Root);
    std::string fit2_cmd = std::string("OMP_NUM_THREADS=1 ")
        + VC_NGRIDS_BIN
        + " -i " + quote(gridsRoot)
        + " --fit-normals --output-zarr " + quote(fit2Root);
    ASSERT_EQ(std::system(fit1_cmd.c_str()), 0);
    ASSERT_EQ(std::system(fit2_cmd.c_str()), 0);

    EXPECT_TRUE(fs::is_directory(fit1Root / "x" / "0"));
    EXPECT_GT(countValidNormals(fit1Root), 0u);
    expectDatasetsEqual(
        fit1Root,
        fit2Root,
        {"x", "y", "z", "fit_rms", "fit_frac_short_paths", "fit_used_radius", "fit_segment_count"});

    std::string align1_cmd = std::string("OMP_NUM_THREADS=1 ")
        + VC_NGRIDS_BIN
        + " -i " + quote(fit1Root)
        + " --align-normals --output-zarr " + quote(align1Root);
    std::string align2_cmd = std::string("OMP_NUM_THREADS=1 ")
        + VC_NGRIDS_BIN
        + " -i " + quote(fit1Root)
        + " --align-normals --output-zarr " + quote(align2Root);
    ASSERT_EQ(std::system(align1_cmd.c_str()), 0);
    ASSERT_EQ(std::system(align2_cmd.c_str()), 0);

    EXPECT_TRUE(fs::is_directory(align1Root / "x" / "0"));
    EXPECT_GT(countValidNormals(align1Root), 0u);
    expectDatasetsEqual(align1Root, align2Root, {"x", "y", "z"});
}
