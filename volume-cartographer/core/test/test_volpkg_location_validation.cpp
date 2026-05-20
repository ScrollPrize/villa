#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

class TempDir {
public:
    TempDir()
    {
        const auto base = std::filesystem::temp_directory_path();
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = base / ("vc-volpkg-validation-" + std::to_string(nonce));
        std::filesystem::create_directories(path_);
    }

    ~TempDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

void writeFile(const std::filesystem::path& path, const char* contents)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out << contents;
}

} // namespace

TEST_CASE("volume location validation accepts zarr v3 pyramids")
{
    TempDir temp;
    const auto zarr = temp.path() / "volume.zarr";

    writeFile(zarr / "zarr.json", R"({"zarr_format":3,"node_type":"group"})");
    writeFile(zarr / "0" / "zarr.json",
              R"({"zarr_format":3,"node_type":"array","shape":[1,1,1],"data_type":"uint8","chunk_grid":{"name":"regular","configuration":{"chunk_shape":[1,1,1]}},"chunk_key_encoding":{"name":"default","configuration":{"separator":"/"}},"fill_value":0,"codecs":[{"name":"bytes","configuration":{"endian":"little"}}]})");

    CHECK(vc::project::validateLocation(vc::project::Category::Volumes,
                                        zarr.string()).empty());
}

TEST_CASE("volume collection validation accepts child zarr v3 pyramids")
{
    TempDir temp;
    const auto zarr = temp.path() / "volumes" / "volume.zarr";

    writeFile(zarr / "zarr.json", R"({"zarr_format":3,"node_type":"group"})");
    writeFile(zarr / "0" / "zarr.json",
              R"({"zarr_format":3,"node_type":"array","shape":[1,1,1],"data_type":"uint8","chunk_grid":{"name":"regular","configuration":{"chunk_shape":[1,1,1]}},"chunk_key_encoding":{"name":"default","configuration":{"separator":"/"}},"fill_value":0,"codecs":[{"name":"bytes","configuration":{"endian":"little"}}]})");

    CHECK(vc::project::validateLocation(vc::project::Category::Volumes,
                                        zarr.parent_path().string()).empty());
}

TEST_CASE("volume open accepts zarr v3 levels padded to storage chunk boundaries")
{
    TempDir temp;
    const auto zarr = temp.path() / "volume.zarr";

    writeFile(zarr / "zarr.json", R"({"zarr_format":3,"node_type":"group"})");
    writeFile(zarr / "0" / "zarr.json",
              R"({"zarr_format":3,"node_type":"array","shape":[1000,1000,1000],"data_type":"uint8","chunk_grid":{"name":"regular","configuration":{"chunk_shape":[512,512,512]}},"chunk_key_encoding":{"name":"default"},"fill_value":0,"codecs":[{"name":"sharding_indexed","configuration":{"chunk_shape":[64,64,64],"codecs":[{"name":"c3d","configuration":{"target_ratio":25.0}}],"index_codecs":[{"name":"bytes","configuration":{"endian":"little"}}],"index_location":"start"}}]})");
    writeFile(zarr / "1" / "zarr.json",
              R"({"zarr_format":3,"node_type":"array","shape":[768,512,512],"data_type":"uint8","chunk_grid":{"name":"regular","configuration":{"chunk_shape":[512,512,512]}},"chunk_key_encoding":{"name":"default"},"fill_value":0,"codecs":[{"name":"sharding_indexed","configuration":{"chunk_shape":[64,64,64],"codecs":[{"name":"c3d","configuration":{"target_ratio":25.0}}],"index_codecs":[{"name":"bytes","configuration":{"endian":"little"}}],"index_location":"start"}}]})");

    const auto volume = Volume::New(zarr);

    CHECK((volume->shape() == std::array<int, 3>{1000, 1000, 1000}));
    CHECK((volume->shape(1) == std::array<int, 3>{768, 512, 512}));
    CHECK((volume->chunkShape(1) == std::array<int, 3>{64, 64, 64}));
}
