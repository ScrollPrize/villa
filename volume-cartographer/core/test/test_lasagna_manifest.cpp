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

TEST_CASE("LasagnaDatasetManifest parses channel groups from canonical Lasagna JSON")
{
    const auto dir = makeTmpDir("groups");
    const auto manifestPath = dir / "dataset.lasagna.json";
    {
        std::ofstream out(manifestPath);
        out << R"({
            "version": 2,
            "source_to_base": 2.0,
            "grad_mag_encode_scale": 1000.0,
            "groups": {
                "pred": {
                    "zarr": "pred.zarr",
                    "scaledown": 4,
                    "channels": ["cos", "grad_mag", "nx", "ny"]
                },
                "extra": {
                    "zarr": "../extra.zarr",
                    "scaledown": 5,
                    "channels": ["pred_dt"]
                }
            }
        })";
    }

    auto manifest = vc::lasagna::LasagnaDatasetManifest::parseFile(manifestPath);

    CHECK(manifest.version == 2);
    CHECK(manifest.sourceToBase == doctest::Approx(2.0));
    REQUIRE(manifest.groups.size() == 2);
    const auto* pred = manifest.groupForChannel("nx");
    REQUIRE(pred != nullptr);
    CHECK(pred->name == "pred");
    CHECK(pred->zarrPath == fs::absolute(dir / "pred.zarr").lexically_normal());
    CHECK(pred->scaleFactor() == 16);
    CHECK(pred->hasChannel("nx"));
    REQUIRE(pred->channelIndex("ny").has_value());
    CHECK(*pred->channelIndex("ny") == 3);
    REQUIRE(manifest.groupForChannel("grad_mag") != nullptr);
    CHECK(manifest.hasNormalSource());
    CHECK(manifest.normalSourceKind == vc::lasagna::NormalSourceKind::DenseZarr);
    CHECK(manifest.normalSourceKey == "groups.grad_mag_nx_ny");
    CHECK(manifest.raw.contains("grad_mag_encode_scale"));

    fs::remove_all(dir);
}

TEST_CASE("LasagnaDataset wraps manifest and reports missing normal source")
{
    auto manifest = vc::lasagna::LasagnaDatasetManifest::parseText(R"({
        "version": 2,
        "groups": {
            "pred": {"zarr": "pred.zarr", "scaledown": 4, "channels": ["cos", "grad_mag"]}
        }
    })");
    vc::lasagna::LasagnaDataset dataset(std::move(manifest));

    CHECK_FALSE(dataset.hasNormalSource());
    CHECK_THROWS_AS(dataset.normalSourcePath(), std::runtime_error);
}

TEST_CASE("LasagnaDatasetManifest requires grad_mag for normal source")
{
    auto manifest = vc::lasagna::LasagnaDatasetManifest::parseText(R"({
        "version": 2,
        "groups": {
            "pred": {"zarr": "pred.zarr", "scaledown": 4, "channels": ["nx", "ny"]}
        }
    })");

    CHECK_FALSE(manifest.hasNormalSource());
    CHECK(manifest.normalSourceKind == vc::lasagna::NormalSourceKind::None);
    CHECK(manifest.normalSourceKey.empty());
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
