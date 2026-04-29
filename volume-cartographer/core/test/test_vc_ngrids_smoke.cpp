#include "test.hpp"

#include <atomic>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "utils/Json.hpp"
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

struct NormalVoxel {
    int z = 0;
    int y = 0;
    int x = 0;
    uint8_t nx = 128;
    uint8_t ny = 128;
    uint8_t nz = 128;
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

void createNormalsInput(
    const fs::path& root,
    const std::array<size_t, 3>& shape,
    const std::array<size_t, 3>& chunks,
    const std::vector<NormalVoxel>& voxels)
{
    const std::vector<size_t> shape_vec = {shape[0], shape[1], shape[2]};
    const std::vector<size_t> chunk_vec = {chunks[0], chunks[1], chunks[2]};
    auto dsx = vc::createZarrDataset(root / "x", "0", shape_vec, chunk_vec, vc::VcDtype::uint8, "blosc", "/", 128);
    auto dsy = vc::createZarrDataset(root / "y", "0", shape_vec, chunk_vec, vc::VcDtype::uint8, "blosc", "/", 128);
    auto dsz = vc::createZarrDataset(root / "z", "0", shape_vec, chunk_vec, vc::VcDtype::uint8, "blosc", "/", 128);

    std::vector<uint8_t> x(shape[0] * shape[1] * shape[2], 128);
    std::vector<uint8_t> y(shape[0] * shape[1] * shape[2], 128);
    std::vector<uint8_t> z(shape[0] * shape[1] * shape[2], 128);
    auto idx = [&](int zz, int yy, int xx) {
        return static_cast<size_t>((zz * static_cast<int>(shape[1]) + yy) * static_cast<int>(shape[2]) + xx);
    };

    for (const auto& voxel : voxels) {
        x[idx(voxel.z, voxel.y, voxel.x)] = voxel.nx;
        y[idx(voxel.z, voxel.y, voxel.x)] = voxel.ny;
        z[idx(voxel.z, voxel.y, voxel.x)] = voxel.nz;
    }

    dsx->writeRegion({0, 0, 0}, shape_vec, x.data());
    dsy->writeRegion({0, 0, 0}, shape_vec, y.data());
    dsz->writeRegion({0, 0, 0}, shape_vec, z.data());
    utils::Json origin = utils::Json::array();
    origin.push_back(static_cast<int64_t>(0));
    origin.push_back(static_cast<int64_t>(0));
    origin.push_back(static_cast<int64_t>(0));
    utils::Json attrs = utils::Json::object();
    attrs["grid_origin_xyz"] = origin;
    attrs["sample_step"] = 1;
    vc::writeZarrAttributes(root, attrs);
}

void createChunkedNormalsInput(const fs::path& root)
{
    std::vector<NormalVoxel> voxels;
    for (int zz = 7; zz < 9; ++zz) {
        for (int yy = 8; yy < 10; ++yy) {
            for (int xx = 9; xx < 11; ++xx) {
                voxels.push_back(NormalVoxel{
                    .z = zz,
                    .y = yy,
                    .x = xx,
                    .nx = 255,
                    .ny = 128,
                    .nz = 128,
                });
            }
        }
    }
    createNormalsInput(root, {80, 80, 80}, {64, 64, 64}, voxels);
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

uint8_t readVoxel(const fs::path& path, size_t z, size_t y, size_t x)
{
    vc::VcDataset ds(path);
    uint8_t value = 0;
    ds.readRegion({z, y, x}, {1, 1, 1}, &value);
    return value;
}

nlohmann::json readJsonFile(const fs::path& path)
{
    return nlohmann::json::parse(std::ifstream(path));
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
        + " --spiral-step 4"
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

TEST(NGridsSmoke, AlignCropPreservesFillOutsidePartialChunks)
{
    ScopedTempDir tempDir;
    const auto inputRoot = tempDir.path() / "normals_input.zarr";
    const auto outputRoot = tempDir.path() / "aligned_output.zarr";

    createChunkedNormalsInput(inputRoot);

    std::string align_cmd = std::string("OMP_NUM_THREADS=1 ")
        + VC_NGRIDS_BIN
        + " -i " + quote(inputRoot)
        + " --align-normals"
        + " --crop 9 8 7 11 10 9"
        + " --output-zarr " + quote(outputRoot);
    ASSERT_EQ(std::system(align_cmd.c_str()), 0);

    EXPECT_EQ(readVoxel(outputRoot / "x" / "0", 0, 0, 0), static_cast<uint8_t>(128));
    EXPECT_EQ(readVoxel(outputRoot / "y" / "0", 0, 0, 0), static_cast<uint8_t>(128));
    EXPECT_EQ(readVoxel(outputRoot / "z" / "0", 0, 0, 0), static_cast<uint8_t>(128));

    EXPECT_NE(readVoxel(outputRoot / "x" / "0", 7, 8, 9), static_cast<uint8_t>(128));
}

TEST(NGridsSmoke, AlignRerunReplacesPreviousCropData)
{
    ScopedTempDir tempDir;
    const auto inputRoot = tempDir.path() / "normals_input.zarr";
    const auto outputRoot = tempDir.path() / "aligned_output.zarr";

    createNormalsInput(
        inputRoot,
        {80, 80, 80},
        {64, 64, 64},
        {
            NormalVoxel{.z = 7, .y = 8, .x = 9, .nx = 255, .ny = 128, .nz = 128},
            NormalVoxel{.z = 7, .y = 8, .x = 20, .nx = 255, .ny = 128, .nz = 128},
        });

    std::string align_first_cmd = std::string("OMP_NUM_THREADS=1 ")
        + VC_NGRIDS_BIN
        + " -i " + quote(inputRoot)
        + " --align-normals"
        + " --crop 9 8 7 10 9 8"
        + " --output-zarr " + quote(outputRoot);
    std::string align_second_cmd = std::string("OMP_NUM_THREADS=1 ")
        + VC_NGRIDS_BIN
        + " -i " + quote(inputRoot)
        + " --align-normals"
        + " --crop 20 8 7 21 9 8"
        + " --output-zarr " + quote(outputRoot);

    ASSERT_EQ(std::system(align_first_cmd.c_str()), 0);
    EXPECT_NE(readVoxel(outputRoot / "x" / "0", 7, 8, 9), static_cast<uint8_t>(128));

    ASSERT_EQ(std::system(align_second_cmd.c_str()), 0);
    EXPECT_EQ(readVoxel(outputRoot / "x" / "0", 7, 8, 9), static_cast<uint8_t>(128));
    EXPECT_NE(readVoxel(outputRoot / "x" / "0", 7, 8, 20), static_cast<uint8_t>(128));
}

TEST(NGridsSmoke, AlignOutputChunksAreCapped)
{
    ScopedTempDir tempDir;
    const auto inputRoot = tempDir.path() / "normals_input.zarr";
    const auto outputRoot = tempDir.path() / "aligned_output.zarr";

    createNormalsInput(
        inputRoot,
        {80, 80, 80},
        {80, 80, 80},
        {
            NormalVoxel{.z = 7, .y = 8, .x = 9, .nx = 255, .ny = 128, .nz = 128},
        });

    std::string align_cmd = std::string("OMP_NUM_THREADS=1 ")
        + VC_NGRIDS_BIN
        + " -i " + quote(inputRoot)
        + " --align-normals"
        + " --crop 9 8 7 10 9 8"
        + " --output-zarr " + quote(outputRoot);
    ASSERT_EQ(std::system(align_cmd.c_str()), 0);

    const auto zarray = readJsonFile(outputRoot / "x" / "0" / ".zarray");
    ASSERT_TRUE(zarray.contains("chunks"));
    ASSERT_EQ(zarray["chunks"][0].get<int>(), 64);
    ASSERT_EQ(zarray["chunks"][1].get<int>(), 64);
    ASSERT_EQ(zarray["chunks"][2].get<int>(), 64);
}
