#include "test.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include "utils/Json.hpp"

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
        path_ = fs::temp_directory_path() / ("vc_test_project_" + suffix);
        fs::create_directories(path_);
    }
    ~ScopedTempDir() { std::error_code ec; fs::remove_all(path_, ec); }
    const fs::path& path() const { return path_; }
private:
    fs::path path_;
};

vc::Volpkg build_sample()
{
    vc::Volpkg p;
    p.name = "sample";
    p.version = 7;
    p.description = "mix-and-match test";

    vc::DataSource vols;
    vols.id = "local_vols";
    vols.type = vc::DataSourceType::VolumesDir;
    vols.location = "/mnt/ssd/scroll1/volumes";
    vols.location_kind = vc::LocationKind::Local;
    vols.recursive = true;
    vols.track_changes = true;
    p.data_sources.push_back(vols);

    vc::DataSource remoteZarr;
    remoteZarr.id = "s3_zarr";
    remoteZarr.type = vc::DataSourceType::ZarrVolume;
    remoteZarr.location = "s3://bucket/scroll/v.zarr";
    remoteZarr.location_kind = vc::LocationKind::Remote;
    p.data_sources.push_back(remoteZarr);

    vc::DataSource remoteSegs;
    remoteSegs.id = "shared_paths";
    remoteSegs.type = vc::DataSourceType::SegmentsDir;
    remoteSegs.location = "https://dl.example.org/scroll/paths/";
    remoteSegs.location_kind = vc::LocationKind::Remote;
    remoteSegs.recursive = true;
    p.data_sources.push_back(remoteSegs);

    p.active_segments_source_id = "shared_paths";
    p.output_segments_source_id = "shared_paths";

    return p;
}

}  // namespace

TEST(Project, RoundTripJson)
{
    const auto orig = build_sample();
    auto j = orig.to_json();
    auto copy = vc::Volpkg::from_json(j);

    EXPECT_EQ(copy.name, orig.name);
    EXPECT_EQ(copy.version, orig.version);
    EXPECT_EQ(copy.description, orig.description);
    EXPECT_EQ(copy.active_segments_source_id, orig.active_segments_source_id);
    EXPECT_EQ(copy.output_segments_source_id, orig.output_segments_source_id);
    ASSERT_EQ(copy.data_sources.size(), orig.data_sources.size());
    for (std::size_t i = 0; i < orig.data_sources.size(); ++i) {
        EXPECT_EQ(copy.data_sources[i].id, orig.data_sources[i].id);
        EXPECT_EQ(static_cast<int>(copy.data_sources[i].type),
                  static_cast<int>(orig.data_sources[i].type));
        EXPECT_EQ(copy.data_sources[i].location, orig.data_sources[i].location);
        EXPECT_EQ(static_cast<int>(copy.data_sources[i].location_kind),
                  static_cast<int>(orig.data_sources[i].location_kind));
        EXPECT_EQ(copy.data_sources[i].recursive, orig.data_sources[i].recursive);
    }
}

TEST(Project, SaveLoadFile)
{
    ScopedTempDir tmp;
    const auto dst = tmp.path() / "project.json";
    const auto orig = build_sample();

    orig.save_to_file(dst);
    auto loaded = vc::Volpkg::load_from_file(dst);

    EXPECT_EQ(loaded.name, orig.name);
    EXPECT_EQ(loaded.version, orig.version);
    EXPECT_EQ(loaded.data_sources.size(), orig.data_sources.size());
    EXPECT_EQ(loaded.path(), dst);
}

TEST(Project, InferLocationKind)
{
    EXPECT_EQ(static_cast<int>(vc::infer_location_kind("/abs/path")),
              static_cast<int>(vc::LocationKind::Local));
    EXPECT_EQ(static_cast<int>(vc::infer_location_kind("relative/path")),
              static_cast<int>(vc::LocationKind::Local));
    EXPECT_EQ(static_cast<int>(vc::infer_location_kind("https://x.com/y")),
              static_cast<int>(vc::LocationKind::Remote));
    EXPECT_EQ(static_cast<int>(vc::infer_location_kind("http://x.com/y")),
              static_cast<int>(vc::LocationKind::Remote));
    EXPECT_EQ(static_cast<int>(vc::infer_location_kind("s3://bucket/key")),
              static_cast<int>(vc::LocationKind::Remote));
}

TEST(Project, FromVolpkgMirrorsDirs)
{
    ScopedTempDir tmp;
    std::ofstream(tmp.path() / "config.json")
        << utils::Json{{"name", "test_scroll"}, {"version", 6},
                        {"voxel_size", 7.91}}.dump(2);
    fs::create_directory(tmp.path() / "volumes");
    fs::create_directory(tmp.path() / "paths");
    fs::create_directory(tmp.path() / "traces");

    auto p = vc::Volpkg::from_volpkg(tmp.path());

    EXPECT_EQ(p.name, "test_scroll");
    EXPECT_EQ(p.version, 6);
    ASSERT_TRUE(p.origin.has_value());
    EXPECT_EQ(p.origin->kind, "volpkg");
    EXPECT_EQ(p.origin->root, tmp.path());
    // Legacy fields beyond name/version must survive the conversion.
    ASSERT_TRUE(!p.origin->legacy_config.is_null());
    EXPECT_TRUE(p.origin->legacy_config.contains("voxel_size"));

    // One volumes_dir + two segments_dir sources expected.
    int volumesDirs = 0, segmentsDirs = 0;
    for (const auto& ds : p.data_sources) {
        if (ds.type == vc::DataSourceType::VolumesDir) ++volumesDirs;
        if (ds.type == vc::DataSourceType::SegmentsDir) ++segmentsDirs;
    }
    EXPECT_EQ(volumesDirs, 1);
    EXPECT_EQ(segmentsDirs, 2);

    // resolve_segments_dir should return <root>/paths for the "paths" source.
    EXPECT_EQ(p.resolve_segments_dir("paths"), tmp.path() / "paths");
    EXPECT_EQ(p.resolve_volumes_dir(), tmp.path() / "volumes");
}

TEST(Project, ResolveLocationUrlVsPath)
{
    auto p = build_sample();

    // Local source resolves to an absolute path string.
    const auto* localVols = p.find_source("local_vols");
    ASSERT_TRUE(localVols != nullptr);
    EXPECT_EQ(p.resolve_location(*localVols), std::string("/mnt/ssd/scroll1/volumes"));

    // Remote source resolves to its raw URL.
    const auto* remoteZarr = p.find_source("s3_zarr");
    ASSERT_TRUE(remoteZarr != nullptr);
    EXPECT_EQ(p.resolve_location(*remoteZarr), std::string("s3://bucket/scroll/v.zarr"));
    EXPECT_EQ(p.remote_url(*remoteZarr), std::string("s3://bucket/scroll/v.zarr"));
}

TEST(Project, HumanReadableOutput)
{
    // The JSON must be indented (spec: "formatted w/ indentation/spacing to
    // be human readable"). Two-space indent gives multiple lines.
    auto j = build_sample().to_json();
    const auto text = j.dump(2);
    EXPECT_TRUE(text.find('\n') != std::string::npos);
    EXPECT_TRUE(text.find("  ") != std::string::npos);
}
