#include "test.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vc/core/types/VcDataset.hpp"
#include "vc/core/types/VolumePkg.hpp"

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
        path_ = fs::temp_directory_path() / ("vc_test_volumepkg_" + suffix);
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

void createVolpkgConfig(const fs::path& root)
{
    fs::create_directories(root);
    writeJson(root / "config.json", {
        {"name", "test"},
        {"version", 1}
    });
}

void createVolumeDir(const fs::path& root,
                     const std::string& dirName,
                     const std::string& volumeId,
                     const std::vector<size_t>& shape)
{
    const fs::path volumeDir = root / "volumes" / dirName;
    fs::create_directories(volumeDir);
    writeJson(volumeDir / "meta.json", {
        {"uuid", volumeId},
        {"name", dirName},
        {"type", "vol"},
        {"format", "zarr"},
        {"width", static_cast<int>(shape[2])},
        {"height", static_cast<int>(shape[1])},
        {"slices", static_cast<int>(shape[0])},
        {"voxelsize", 1.0},
        {"min", 0.0},
        {"max", 255.0}
    });
    vc::createZarrDataset(volumeDir, "0", shape, shape, vc::VcDtype::uint8, "none");
}

}  // namespace

TEST(VolumePkg, ConstructorLoadsInitialVolumes)
{
    ScopedTempDir tempDir;
    createVolpkgConfig(tempDir.path());
    createVolumeDir(tempDir.path(), "scan_00", "vol-001", {2, 3, 4});

    auto pkg = VolumePkg::New(tempDir.path());

    ASSERT_TRUE(pkg != nullptr);
    EXPECT_TRUE(pkg->hasVolumes());
    EXPECT_TRUE(pkg->hasVolume("vol-001"));
    EXPECT_EQ(pkg->numberOfVolumes(), 1u);
}

TEST(VolumePkg, AddSingleVolumeSucceedsAndRejectsDuplicate)
{
    ScopedTempDir tempDir;
    createVolpkgConfig(tempDir.path());

    auto pkg = VolumePkg::New(tempDir.path());
    ASSERT_TRUE(pkg != nullptr);
    EXPECT_FALSE(pkg->hasVolumes());

    createVolumeDir(tempDir.path(), "scan_01", "vol-002", {2, 2, 2});

    EXPECT_TRUE(pkg->addSingleVolume("scan_01"));
    EXPECT_TRUE(pkg->hasVolume("vol-002"));
    EXPECT_EQ(pkg->numberOfVolumes(), 1u);

    EXPECT_FALSE(pkg->addSingleVolume("scan_01"));
    EXPECT_EQ(pkg->numberOfVolumes(), 1u);
}
