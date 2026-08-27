#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LineModel.hpp"
#include "vc/lasagna/Manifest.hpp"

#include <utils/zarr.hpp>

#include <cstdlib>
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

bool requireNetwork()
{
    const char* value = std::getenv("VC_TEST_REQUIRE_NETWORK");
    return value && value[0] && value[0] != '0';
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

TEST_CASE("LasagnaDatasetManifest finds complete fiber prediction channel sets")
{
    const auto manifest = vc::lasagna::LasagnaDatasetManifest::parseText(R"({
        "version": 2,
        "groups": {
            "base": {
                "zarr": "base.zarr",
                "channels": ["presence", "nx", "ny"]
            },
            "predictions": {
                "zarr": "predictions.zarr",
                "channels": [
                    "ink_presence", "ink_nx", "ink_ny",
                    "ink_presence", "incomplete_presence", "incomplete_nx"
                ]
            }
        }
    })");

    const auto prefixes = manifest.fiberPredictionPrefixes();
    REQUIRE(prefixes.size() == 2);
    CHECK(prefixes[0].empty());
    CHECK(prefixes[1] == "ink");
}

TEST_CASE("LasagnaDatasetManifest rejects incomplete fiber prediction channel sets")
{
    const auto manifest = vc::lasagna::LasagnaDatasetManifest::parseText(R"({
        "version": 2,
        "groups": {
            "predictions": {
                "zarr": "predictions.zarr",
                "channels": ["fiber_presence", "fiber_nx", "ny"]
            }
        }
    })");

    CHECK(manifest.fiberPredictionPrefixes().empty());
}

TEST_CASE("LasagnaDataset applies runtime coordinate scale without mutating the manifest file")
{
    const auto dir = makeTmpDir("working-scale");
    const auto manifestPath = dir / "dataset.lasagna.json";
    {
        std::ofstream out(manifestPath);
        out << R"({
            "version": 2,
            "source_to_base": 1.0,
            "base_shape_zyx": [100, 200, 300],
            "groups": {}
        })";
    }

    const auto dataset = vc::lasagna::LasagnaDataset::open(manifestPath, {8.0});
    CHECK(dataset.manifest().workingToBaseScale == doctest::Approx(8.0));
    REQUIRE(dataset.manifest().baseShapeZYX.has_value());
    CHECK(*dataset.manifest().baseShapeZYX ==
          std::array<std::size_t, 3>{100, 200, 300});
    CHECK_THROWS(vc::lasagna::LasagnaDataset::open(manifestPath, {0.0}));

    fs::remove_all(dir);
}

TEST_CASE("LasagnaDataset recognizes a validated remote cache marker")
{
    const auto dir = makeTmpDir("remote-marker");
    const auto manifestPath = dir / "dataset.lasagna.json";
    {
        std::ofstream out(manifestPath);
        out << R"({"version":2,"groups":{}})";
    }
    {
        std::ofstream out(dir / vc::lasagna::kLasagnaRemoteMarker);
        out << R"({
          "artifact_url":"https://example.test/lasagna",
          "manifest_file":"dataset.lasagna.json"
        })";
    }
    const auto dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    CHECK(dataset.manifest().remoteBaseUrl == "https://example.test/lasagna");
    CHECK(dataset.manifest().remoteCacheRoot == dir);

    {
        std::ofstream out(dir / vc::lasagna::kLasagnaRemoteMarker);
        out << R"({
          "artifact_url":"https://example.test/lasagna",
          "manifest_file":"different.lasagna.json"
        })";
    }
    CHECK_THROWS(vc::lasagna::LasagnaDataset::open(manifestPath));
    fs::remove_all(dir);
}

TEST_CASE("LasagnaDataset resolves marker-backed remote relative groups")
{
    const auto dir = makeTmpDir("remote-marker-groups");
    const auto manifestPath = dir / "dataset.lasagna.json";
    {
        std::ofstream out(manifestPath);
        out << R"({
            "version": 2,
            "groups": {
                "pred": {
                    "zarr": "pred.zarr",
                    "scaledown": 2,
                    "channels": ["presence"]
                }
            }
        })";
    }
    {
        std::ofstream out(dir / vc::lasagna::kLasagnaRemoteMarker);
        out << R"({
          "artifact_url":"s3://bucket/path/artifact/",
          "anonymous":true,
          "manifest_file":"dataset.lasagna.json"
        })";
    }

    const auto dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    REQUIRE(dataset.manifest().groups.size() == 1);
    const auto& group = dataset.manifest().groups.front();
    CHECK(group.isRemote());
    CHECK(group.remoteZarrBaseUrl ==
          "https://bucket.s3.us-east-1.amazonaws.com/path/artifact");
    CHECK(group.remoteZarrKey == "pred.zarr");
    CHECK(group.remoteCacheRoot == dir);
    CHECK(group.sourceLocation ==
          "s3://bucket/path/artifact/pred.zarr");
    CHECK_FALSE(dataset.manifest().discoverAwsCredentials);
    CHECK_FALSE(group.discoverAwsCredentials);

    fs::remove_all(dir);
}

TEST_CASE("LasagnaDataset rejects escaping relative remote groups")
{
    const auto dir = makeTmpDir("remote-marker-escape");
    const auto manifestPath = dir / "dataset.lasagna.json";
    {
        std::ofstream out(manifestPath);
        out << R"({
            "version": 2,
            "groups": {
                "pred": {
                    "zarr": "../pred.zarr",
                    "channels": ["presence"]
                }
            }
        })";
    }
    {
        std::ofstream out(dir / vc::lasagna::kLasagnaRemoteMarker);
        out << R"({
          "artifact_url":"https://example.test/artifact",
          "manifest_file":"dataset.lasagna.json"
        })";
    }

    CHECK_THROWS_AS(vc::lasagna::LasagnaDataset::open(manifestPath),
                    std::runtime_error);

    fs::remove_all(dir);
}

TEST_CASE("LasagnaDataset preserves absolute local group paths")
{
    const auto dir = makeTmpDir("absolute-local-group");
    const auto zarrPath = fs::absolute(dir / "absolute.zarr").lexically_normal();
    const auto manifestPath = dir / "dataset.lasagna.json";
    {
        std::ofstream out(manifestPath);
        out << R"({
            "version": 2,
            "groups": {
                "pred": {
                    "zarr": ")" << zarrPath.generic_string() << R"(",
                    "channels": ["presence"]
                }
            }
        })";
    }

    const auto dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    REQUIRE(dataset.manifest().groups.size() == 1);
    const auto& group = dataset.manifest().groups.front();
    CHECK_FALSE(group.isRemote());
    CHECK(group.zarrPath == zarrPath);

    fs::remove_all(dir);
}

TEST_CASE("LasagnaDataset supports absolute remote group paths with explicit cache")
{
    const auto dir = makeTmpDir("absolute-remote-group");
    const auto manifestPath = dir / "dataset.lasagna.json";
    {
        std::ofstream out(manifestPath);
        out << R"({
            "version": 2,
            "groups": {
                "pred": {
                    "zarr": "s3+eu-west-2://bucket/other/pred.zarr",
                    "channels": ["presence"]
                }
            }
        })";
    }

    CHECK_THROWS_AS(vc::lasagna::LasagnaDataset::open(manifestPath),
                    std::runtime_error);

    const auto cacheRoot = dir / "cache";
    const auto dataset = vc::lasagna::LasagnaDataset::open(
        manifestPath,
        vc::lasagna::LasagnaDatasetOpenOptions{1.0, cacheRoot});
    REQUIRE(dataset.manifest().groups.size() == 1);
    const auto& group = dataset.manifest().groups.front();
    CHECK(group.isRemote());
    CHECK(group.remoteZarrBaseUrl ==
          "https://bucket.s3.eu-west-2.amazonaws.com/other/pred.zarr");
    CHECK(group.remoteZarrKey.empty());
    CHECK(group.remoteCacheRoot ==
          cacheRoot / "remote_sources" / "s3+eu-west-2" / "bucket" /
              "other" / "pred.zarr");
    CHECK(group.sourceLocation ==
          "s3+eu-west-2://bucket/other/pred.zarr");

    fs::remove_all(dir);
}

TEST_CASE("LasagnaDataset remote manifest requires explicit cache before fetch")
{
    CHECK(vc::lasagna::isRemoteLasagnaLocation(
        "s3://bucket/path/fiber.lasagna.json"));
    CHECK(vc::lasagna::isRemoteLasagnaLocation(
        "https://example.test/fiber.lasagna.json"));
    CHECK_FALSE(vc::lasagna::isRemoteLasagnaLocation(
        "/tmp/fiber.lasagna.json"));
    CHECK_THROWS_AS(
        vc::lasagna::LasagnaDataset::openLocation(
            "s3://bucket/path/fiber.lasagna.json"),
        std::runtime_error);
}

TEST_CASE("LasagnaDataset rejects a malformed remote manifest source")
{
    const auto dir = makeTmpDir("remote-fetch-diagnostics");
    try {
        (void)vc::lasagna::LasagnaDataset::openLocation(
            "http://",
            vc::lasagna::LasagnaDatasetOpenOptions{1.0, dir});
        FAIL("expected invalid remote fetch to throw");
    } catch (const std::exception& exc) {
        const std::string what = exc.what();
        CHECK(what.find("remote file cache source") != std::string::npos);
    }
    fs::remove_all(dir);
}

TEST_CASE("LasagnaDataset persistently caches direct remote manifests")
{
    const auto dir = makeTmpDir("remote-manifest-cache");
    int fetches = 0;
    vc::lasagna::LasagnaDatasetOpenOptions options;
    options.remoteCacheRoot = dir;
    options.remoteAuth.access_key = "test-access";
    options.remoteAuth.secret_key = "test-secret";
    options.remoteFileFetcher = [&](const std::string&, const fs::path& tmp) {
        ++fetches;
        std::ofstream(tmp) << R"({
          "version": 2,
          "groups": {
            "pred": {
              "zarr": "pred.zarr",
              "scaledown": 3,
              "channels": ["presence"]
            }
          }
        })";
    };

    const std::string location =
        "s3+eu-west-2://bucket/artifact/fiber.lasagna.json";
    const auto first = vc::lasagna::LasagnaDataset::openLocation(location, options);
    const auto second = vc::lasagna::LasagnaDataset::openLocation(location, options);

    CHECK(fetches == 1);
    CHECK(first.manifest().manifestIsRemote);
    CHECK(first.manifest().manifestLocation == location);
    REQUIRE(first.manifest().groups.size() == 1);
    CHECK(first.manifest().groups.front().remoteZarrBaseUrl ==
          "https://bucket.s3.eu-west-2.amazonaws.com/artifact");
    CHECK(first.manifest().groups.front().remoteZarrKey == "pred.zarr");
    CHECK(first.manifest().groups.front().sourceLocation ==
          "s3+eu-west-2://bucket/artifact/pred.zarr");
    CHECK(first.manifest().groups.front().remoteAuth.access_key == "test-access");
    CHECK(first.manifest().groups.front().remoteCacheRoot ==
          first.manifest().manifestPath.parent_path());
    CHECK(first.manifest().manifestPath ==
          dir / "remote_sources" / "s3+eu-west-2" / "bucket" /
              "artifact" / "fiber.lasagna.json");
    CHECK(second.manifest().manifestPath == first.manifest().manifestPath);
    options.cachePolicy = vc::core::util::RemoteFileCachePolicy::Refresh;
    (void)vc::lasagna::LasagnaDataset::openLocation(location, options);
    CHECK(fetches == 2);
    CHECK(fs::is_regular_file(first.manifest().manifestPath));
    fs::remove_all(dir);
}

TEST_CASE("public S3 Lasagna ignores rejected ambient-style credentials")
{
    if (!requireNetwork()) {
        MESSAGE("Set VC_TEST_REQUIRE_NETWORK=1 to run the public S3 smoke test");
        return;
    }

    const auto dir = makeTmpDir("public-s3-auth-fallback");
    vc::lasagna::LasagnaDatasetOpenOptions options;
    options.remoteCacheRoot = dir;
    options.remoteAuth.access_key = "AKIAIOSFODNN7EXAMPLE";
    options.remoteAuth.secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
    options.remoteAuth.region = "us-east-1";

    const std::string location =
        "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/representations/"
        "predictions/fibers/20260102150214-fibers-20260801084232-L1/"
        "PHerc0139-20260102150214-las-sd1-92481a4c.lasagna.json";
    try {
        const auto dataset = vc::lasagna::LasagnaDataset::openLocation(
            location, options);
        const auto* presence = dataset.manifest().groupForChannel("presence");
        REQUIRE(presence != nullptr);
        const auto array = vc::lasagna::openLasagnaChannelArray(
            dataset.manifest(), *presence, 1);
        CHECK(array.metadata().ndim() == 3);
        CHECK(array.metadata().shape ==
              std::vector<std::size_t>{9620, 3314, 3314});
    } catch (const std::exception& error) {
        fs::remove_all(dir);
        FAIL("public S3 Lasagna fallback failed: " << error.what());
    }
    fs::remove_all(dir);
}

TEST_CASE("LasagnaDataset refreshes a corrupt size-valid cached manifest once")
{
    const auto dir = makeTmpDir("remote-manifest-corrupt");
    int fetches = 0;
    vc::lasagna::LasagnaDatasetOpenOptions options;
    options.remoteCacheRoot = dir;
    options.remoteFileFetcher = [&](const std::string&, const fs::path& tmp) {
        ++fetches;
        std::ofstream(tmp) << R"({"version":2,"groups":{}})";
    };
    const std::string location = "https://example.test/data/test.lasagna.json";
    const auto initial = vc::lasagna::LasagnaDataset::openLocation(location, options);
    {
        std::ofstream(initial.manifest().manifestPath, std::ios::trunc) <<
            std::string(fs::file_size(initial.manifest().manifestPath), 'x');
    }
    const auto recovered = vc::lasagna::LasagnaDataset::openLocation(location, options);
    CHECK(fetches == 2);
    CHECK(recovered.manifest().version == 2);
    fs::remove_all(dir);
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
