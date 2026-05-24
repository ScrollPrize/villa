#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LineModel.hpp"
#include "vc/lasagna/Manifest.hpp"

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace {

fs::path makeTmpDir(const std::string& tag)
{
    auto dir = fs::temp_directory_path() / ("vc_lasagna_manifest_" + tag);
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

class ConstantNormalSampler final : public vc::lasagna::NormalSampler {
public:
    explicit ConstantNormalSampler(cv::Vec3d normal)
        : normal_(normal)
    {
    }

    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d& /*volumePoint*/) const override
    {
        return {normal_, true, {}};
    }

private:
    cv::Vec3d normal_;
};

} // namespace

TEST_CASE("LasagnaDatasetManifest parses tolerant minimal normal-grid manifest")
{
    const auto dir = makeTmpDir("normal_grid");
    const auto manifestPath = dir / "dataset.lasagna.json";
    {
        std::ofstream out(manifestPath);
        out << R"({
            "name": "demo",
            "volume_path": "volume.zarr",
            "normal_grid_path": "normal_grids",
            "unknown_future_field": {"kept": true}
        })";
    }

    auto manifest = vc::lasagna::LasagnaDatasetManifest::parseFile(manifestPath);

    REQUIRE(manifest.volumePath.has_value());
    REQUIRE(manifest.normalPath.has_value());
    CHECK(*manifest.volumePath == fs::absolute(dir / "volume.zarr").lexically_normal());
    CHECK(*manifest.normalPath == fs::absolute(dir / "normal_grids").lexically_normal());
    CHECK(manifest.normalSourceKind == vc::lasagna::NormalSourceKind::NormalGrid);
    CHECK(manifest.normalSourceKey == "normal_grid_path");
    CHECK(manifest.hasNormalSource());
    CHECK(manifest.raw.contains("unknown_future_field"));

    fs::remove_all(dir);
}

TEST_CASE("LasagnaDatasetManifest parses nested dense-normal aliases")
{
    const auto dir = makeTmpDir("dense_zarr");
    const auto manifestPath = dir / "dataset.lasagna.json";

    const auto manifest = vc::lasagna::LasagnaDatasetManifest::parseText(
        R"({
            "paths": {
                "volume": "volumes/main.zarr",
                "normals": {"path": "../normals.zarr", "type": "zarr"}
            }
        })",
        manifestPath);

    REQUIRE(manifest.volumePath.has_value());
    REQUIRE(manifest.normalPath.has_value());
    CHECK(*manifest.volumePath == fs::absolute(dir / "volumes/main.zarr").lexically_normal());
    CHECK(*manifest.normalPath == fs::absolute(dir / "../normals.zarr").lexically_normal());
    CHECK(manifest.normalSourceKind == vc::lasagna::NormalSourceKind::DenseZarr);
    CHECK(manifest.normalSourceKey == "paths.normals");

    fs::remove_all(dir);
}

TEST_CASE("LasagnaDataset wraps manifest and reports missing normal source")
{
    auto manifest = vc::lasagna::LasagnaDatasetManifest::parseText(R"({"name":"no normals"})");
    vc::lasagna::LasagnaDataset dataset(std::move(manifest));

    CHECK_FALSE(dataset.hasNormalSource());
    CHECK_THROWS_AS(dataset.normalSourcePath(), std::runtime_error);
}

TEST_CASE("NormalSampler interface supports framework tests without Qt")
{
    ConstantNormalSampler sampler({0.0, 0.0, 1.0});

    const auto sample = sampler.sampleNormal({10.0, 20.0, 30.0});

    CHECK(sample.valid);
    CHECK(sample.normal[0] == doctest::Approx(0.0));
    CHECK(sample.normal[1] == doctest::Approx(0.0));
    CHECK(sample.normal[2] == doctest::Approx(1.0));
}
