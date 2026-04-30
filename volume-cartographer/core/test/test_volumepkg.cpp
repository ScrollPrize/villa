#include "test.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "utils/Json.hpp"

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

void writeJson(const fs::path& path, const utils::Json& json)
{
    std::ofstream out(path);
    out << json.dump(2) << '\n';
}

fs::path makeZarrVolume(const fs::path& parent,
                        const std::string& dirName,
                        const std::string& volumeId,
                        const std::vector<size_t>& shape)
{
    const fs::path volumeDir = parent / dirName;
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
    return volumeDir;
}

fs::path makeProjectFile(const fs::path& dir, const utils::Json& body)
{
    const auto file = dir / "project.volpkg.json";
    writeJson(file, body);
    return file;
}

}

TEST(VolumePkg, LoadJsonFileWithSingleZarr)
{
    ScopedTempDir tmp;
    const auto volPath = makeZarrVolume(tmp.path(), "scan_00", "vol-001", {2, 3, 4});

    utils::Json body;
    body["name"] = "demo";
    body["version"] = 1;
    auto vols = utils::Json::array();
    vols.push_back(utils::Json(volPath.string()));
    body["volumes"] = vols;

    auto file = makeProjectFile(tmp.path(), body);
    auto pkg = VolumePkg::load(file);

    ASSERT_TRUE(pkg != nullptr);
    EXPECT_TRUE(pkg->hasVolumes());
    EXPECT_EQ(pkg->numberOfVolumes(), 1u);
    EXPECT_TRUE(pkg->hasVolume("vol-001"));
    EXPECT_EQ(pkg->name(), "demo");
}

TEST(VolumePkg, LoadJsonFileAutoDetectsFolderOfZarrs)
{
    ScopedTempDir tmp;
    fs::create_directories(tmp.path() / "vols");
    makeZarrVolume(tmp.path() / "vols", "scan_00", "vol-001", {2, 3, 4});
    makeZarrVolume(tmp.path() / "vols", "scan_01", "vol-002", {2, 3, 4});

    utils::Json body;
    body["name"] = "demo";
    body["version"] = 1;
    auto vols = utils::Json::array();
    vols.push_back(utils::Json((tmp.path() / "vols").string()));
    body["volumes"] = vols;

    auto file = makeProjectFile(tmp.path(), body);
    auto pkg = VolumePkg::load(file);

    ASSERT_TRUE(pkg != nullptr);
    EXPECT_EQ(pkg->numberOfVolumes(), 2u);
    EXPECT_TRUE(pkg->hasVolume("vol-001"));
    EXPECT_TRUE(pkg->hasVolume("vol-002"));
}

TEST(VolumePkg, IsLocationRemoteRecognizesUriSchemes)
{
    EXPECT_TRUE(vc::project::isLocationRemote("s3://bucket/key"));
    EXPECT_TRUE(vc::project::isLocationRemote("s3+us-west-2://bucket/key"));
    EXPECT_TRUE(vc::project::isLocationRemote("http://example.com/v.zarr"));
    EXPECT_TRUE(vc::project::isLocationRemote("https://example.com/v.zarr"));
    EXPECT_FALSE(vc::project::isLocationRemote("/abs/local/path"));
    EXPECT_FALSE(vc::project::isLocationRemote("file:///abs/local/path"));
    EXPECT_FALSE(vc::project::isLocationRemote("relative/path"));
}

TEST(VolumePkg, ResolveLocalPathStripsFileScheme)
{
    EXPECT_EQ(vc::project::resolveLocalPath("/abs/path"), fs::path("/abs/path"));
    EXPECT_EQ(vc::project::resolveLocalPath("file:///abs/path"), fs::path("/abs/path"));
}

TEST(VolumePkg, EntriesWithTagsRoundTripJson)
{
    ScopedTempDir tmp;
    auto pkg = VolumePkg::newEmpty();
    const auto v1 = makeZarrVolume(tmp.path(), "v1", "vol-aaa", {2, 2, 2});
    pkg->addVolumeEntry(v1.string(), {"normal3d", "experimental"});

    const auto file = tmp.path() / "out.volpkg.json";
    pkg->save(file);

    auto reloaded = VolumePkg::load(file);
    ASSERT_TRUE(reloaded != nullptr);
    ASSERT_EQ(reloaded->volumeEntries().size(), 1u);
    EXPECT_EQ(reloaded->volumeEntries()[0].location, v1.string());
    ASSERT_EQ(reloaded->volumeEntries()[0].tags.size(), 2u);
    EXPECT_EQ(reloaded->volumeEntries()[0].tags[0], "normal3d");
    EXPECT_EQ(reloaded->volumeTags("vol-aaa").size(), 2u);
    auto n3d = reloaded->normal3dZarrPaths();
    EXPECT_EQ(n3d.size(), 1u);
}

TEST(VolumePkg, OutputSegmentsRoundTrip)
{
    ScopedTempDir tmp;
    auto pkg = VolumePkg::newEmpty();
    const auto segDir = tmp.path() / "paths";
    fs::create_directories(segDir);
    pkg->addSegmentsEntry(segDir.string());
    pkg->setOutputSegments(segDir.string());

    const auto file = tmp.path() / "out.volpkg.json";
    pkg->save(file);

    auto reloaded = VolumePkg::load(file);
    ASSERT_TRUE(reloaded != nullptr);
    EXPECT_TRUE(reloaded->hasOutputSegments());
    EXPECT_EQ(reloaded->outputSegmentsPath(), segDir);
}

TEST(VolumePkg, MissingLocationsAreSkippedNotFatal)
{
    ScopedTempDir tmp;
    utils::Json body;
    body["name"] = "demo";
    body["version"] = 1;
    auto vols = utils::Json::array();
    vols.push_back(utils::Json("/nonexistent/path/to/volume"));
    body["volumes"] = vols;

    auto file = makeProjectFile(tmp.path(), body);
    auto pkg = VolumePkg::load(file);

    ASSERT_TRUE(pkg != nullptr);
    EXPECT_FALSE(pkg->hasVolumes());
}

TEST(VolumePkg, JsonOutputIsHumanReadable)
{
    ScopedTempDir tmp;
    auto pkg = VolumePkg::newEmpty();
    pkg->setName("scroll-1");
    const auto file = tmp.path() / "out.volpkg.json";
    pkg->save(file);

    std::ifstream in(file);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_NE(text.find('\n'), std::string::npos);
    EXPECT_NE(text.find("  "), std::string::npos);
    EXPECT_NE(text.find("\"name\""), std::string::npos);
}
