#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "support/LoopbackHttpServer.hpp"
#include "vc/core/util/MemMap.hpp"
#include "vc/core/util/RemoteCacheSettings.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LineModel.hpp"
#include "vc/lasagna/Manifest.hpp"


#include <cstdlib>
#include <filesystem>
#include <fstream>
#include "vc_reader/zarr.hpp"

namespace fs = std::filesystem;

namespace {

fs::path makeTmpDir(const std::string& tag)
{
    // Per process: the suite runs these executables in parallel.
    auto dir = fs::temp_directory_path() /
        ("vc_lasagna_manifest_" + std::to_string(vc::memmap::pid()) + "_" + tag);
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

TEST_CASE("Lasagna shape pairing identifies a dyadic coordinate scale")
{
    using vc::lasagna::dyadicCoordinateScaleBetweenShapes;
    const std::array<std::size_t, 3> l0{75784, 32694, 32694};
    const std::array<std::size_t, 3> l2{18946, 8174, 8174};
    CHECK(dyadicCoordinateScaleBetweenShapes(l2, l0, 5) ==
          doctest::Approx(4.0));
    CHECK(dyadicCoordinateScaleBetweenShapes(l0, l2, 5) ==
          doctest::Approx(0.25));

    // Same-level manifests may mix voxel-count and inclusive-coordinate
    // conventions independently on each axis.
    const std::array<std::size_t, 3> l0Inclusive{75784, 32693, 32693};
    CHECK(dyadicCoordinateScaleBetweenShapes(l0Inclusive, l0, 5) ==
          doctest::Approx(1.0));
    CHECK(dyadicCoordinateScaleBetweenShapes(l0, l0Inclusive, 5) ==
          doctest::Approx(1.0));
    CHECK_THROWS(dyadicCoordinateScaleBetweenShapes(
        std::array<std::size_t, 3>{75784, 32692, 32694}, l0, 5));

    // Both common odd-extent pyramid conventions are accepted.
    CHECK(dyadicCoordinateScaleBetweenShapes(
              std::array<std::size_t, 3>{13, 25, 38},
              std::array<std::size_t, 3>{100, 200, 300}, 5) ==
          doctest::Approx(8.0));
    CHECK_THROWS(dyadicCoordinateScaleBetweenShapes(
        std::array<std::size_t, 3>{18946, 8175, 8174}, l0, 5));
    CHECK_THROWS(dyadicCoordinateScaleBetweenShapes(l2, l2, -1));
}

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
    CHECK(dataset.manifest().remoteSourceBaseLocation == "https://example.test/lasagna");
    CHECK(dataset.manifest().remoteCacheRoot == vc::settings::remoteCachePath());

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
    CHECK(group.remoteCacheRoot == vc::settings::remoteCachePath());
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
    // vc_reader lays out the cache by URL below the caller's cache root.
    CHECK(group.remoteCacheRoot == fs::absolute(cacheRoot).lexically_normal());
    CHECK(group.sourceLocation ==
          "s3+eu-west-2://bucket/other/pred.zarr");

    fs::remove_all(dir);
}

TEST_CASE("LasagnaDataset remote manifest requires explicit cache before fetch")
{
    CHECK_THROWS_AS(
        vc::lasagna::LasagnaDataset::openLocation(
            "s3://bucket/path/fiber.lasagna.json"),
        std::runtime_error);
}

TEST_CASE("LasagnaDataset rejects a malformed remote manifest source")
{
    const auto dir = makeTmpDir("remote-fetch-diagnostics");
    CHECK_THROWS(vc::lasagna::LasagnaDataset::openLocation(
        "http://", vc::lasagna::LasagnaDatasetOpenOptions{1.0, dir}));
    fs::remove_all(dir);
}

// A signed loopback bucket holding one manifest, reached through AWS_ENDPOINT_URL.
struct SignedManifestBucket {
    fs::path dir;
    vc::test::LoopbackHttpServer server;
    explicit SignedManifestBucket(const std::string& tag) : dir(makeTmpDir(tag))
    {
        fs::create_directories(dir / "bucket" / "artifact");
        std::ofstream(dir / "bucket" / "artifact" / "fiber.lasagna.json") << R"({
          "version": 2,
          "groups": {
            "pred": {
              "zarr": "pred.zarr",
              "scaledown": 3,
              "channels": ["presence"]
            }
          }
        })";
        server.serveBucket("bucket", dir / "bucket", true);
        // The environment's keys sign the bucket's requests (vcr::zarr::resolveAccess).
        setenv("AWS_ACCESS_KEY_ID", "test-access", 1);
        setenv("AWS_SECRET_ACCESS_KEY", "test-secret", 1);
        setenv("AWS_REGION", "us-east-1", 1);
        setenv("AWS_ENDPOINT_URL", server.url().c_str(), 1);
    }
    ~SignedManifestBucket()
    {
        for (const char* name : {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "AWS_ENDPOINT_URL"})
            unsetenv(name);
        fs::remove_all(dir);
    }
    std::size_t fetches() const { return server.requestCount("/bucket/artifact/fiber.lasagna.json"); }
    vc::lasagna::LasagnaDatasetOpenOptions options() const
    {
        vc::lasagna::LasagnaDatasetOpenOptions options;
        options.remoteCacheRoot = dir / "cache";
        return options;
    }
};

TEST_CASE("LasagnaDataset persistently caches direct remote manifests")
{
    SignedManifestBucket bucket("remote-manifest-cache");
    auto options = bucket.options();
    const std::string location = "s3://bucket/artifact/fiber.lasagna.json";
    const auto first = vc::lasagna::LasagnaDataset::openLocation(location, options);
    const auto second = vc::lasagna::LasagnaDataset::openLocation(location, options);

    CHECK(bucket.fetches() == 1);
    for (const auto& request : bucket.server.requests()) CHECK(request.signedV4());
    CHECK(first.manifest().manifestIsRemote);
    CHECK(first.manifest().manifestLocation == location);
    REQUIRE(first.manifest().groups.size() == 1);
    CHECK(first.manifest().groups.front().sourceLocation == "s3://bucket/artifact/pred.zarr");
    const auto root = fs::absolute(bucket.dir / "cache").lexically_normal();
    CHECK(first.manifest().groups.front().remoteCacheRoot == root);
    CHECK(first.manifest().manifestPath == vcr::zarr::decodedCachePath(root, location));
    CHECK(second.manifest().manifestPath == first.manifest().manifestPath);
    options.refreshManifest = true;
    (void)vc::lasagna::LasagnaDataset::openLocation(location, options);
    CHECK(bucket.fetches() == 2);
    CHECK(fs::is_regular_file(first.manifest().manifestPath));
}

TEST_CASE("LasagnaDataset reads remote manifests under VC's cache settings")
{
    // Settings from this process's VC3D_CONFIG_DIR reach the first Lasagna read, before any
    // CT volume: a free-space floor nothing can satisfy stops caching, reports it and deletes
    // nothing cached; lifting it lets the read through.
    SignedManifestBucket bucket("remote-manifest-settings");
    const auto config = bucket.dir / "config";
    fs::create_directories(config);
    setenv("VC3D_CONFIG_DIR", config.c_str(), 1);
    std::ofstream(config / "VC3D.ini") << "[perf]\nremote_cache_min_free_gib=1000000000\n";
    const auto options = bucket.options();
    const std::string location = "s3://bucket/artifact/fiber.lasagna.json";
    CHECK_THROWS_WITH_AS(vc::lasagna::LasagnaDataset::openLocation(location, options),
                         doctest::Contains("nothing cached was removed"), vcr::zarr::LowDiskSpace);
    const auto root = fs::absolute(bucket.dir / "cache").lexically_normal();
    CHECK_FALSE(fs::exists(vcr::zarr::decodedCachePath(root, location)));
    std::ofstream(config / "VC3D.ini", std::ios::trunc) << "[perf]\nremote_cache_min_free_gib=0\n";
    const auto dataset = vc::lasagna::LasagnaDataset::openLocation(location, options);
    CHECK(dataset.manifest().groups.size() == 1);
    CHECK(fs::is_regular_file(dataset.manifest().manifestPath));
    unsetenv("VC3D_CONFIG_DIR");
}

TEST_CASE("public S3 Lasagna is read anonymously whatever keys the environment holds")
{
    if (!requireNetwork()) {
        MESSAGE("Set VC_TEST_REQUIRE_NETWORK=1 to run the public S3 smoke test");
        return;
    }

    const auto dir = makeTmpDir("public-s3-auth-fallback");
    vc::lasagna::LasagnaDatasetOpenOptions options;
    options.remoteCacheRoot = dir;
    // Rejected keys in the environment: Open Data is never signed, so they are not used.
    setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE", 1);
    setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", 1);
    setenv("AWS_SESSION_TOKEN", "malformed-session-token", 1);
    struct Unset { ~Unset() { for (const char* n : {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"}) unsetenv(n); } } unset;

    const std::string location =
        "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/representations/"
        "predictions/fibers/20260102150214-fibers-20260801084232-L1/"
        "PHerc0139-20260102150214-las-sd1-92481a4c.lasagna.json";
    try {
        const auto dataset = vc::lasagna::LasagnaDataset::openLocation(
            location, options);
        const auto* presence = dataset.manifest().groupForChannel("presence");
        REQUIRE(presence != nullptr);
        const auto array = vc::lasagna::openLasagnaGroupArray(*presence, nullptr);
        CHECK(array.meta.shape == vcr::Index{9620, 3314, 3314});
    } catch (const std::exception& error) {
        fs::remove_all(dir);
        FAIL("public S3 Lasagna fallback failed: " << error.what());
    }
    fs::remove_all(dir);
}

TEST_CASE("LasagnaDataset trusts a cached manifest until an explicit refresh")
{
    SignedManifestBucket bucket("remote-manifest-corrupt");
    auto options = bucket.options();
    const std::string location = "s3://bucket/artifact/fiber.lasagna.json";
    const auto initial = vc::lasagna::LasagnaDataset::openLocation(location, options);
    {
        std::ofstream(initial.manifest().manifestPath, std::ios::trunc) <<
            std::string(fs::file_size(initial.manifest().manifestPath), 'x');
    }
    // A corrupt cached copy is an error, not a reason to download again on its own.
    CHECK_THROWS(vc::lasagna::LasagnaDataset::openLocation(location, options));
    CHECK(bucket.fetches() == 1);
    options.refreshManifest = true;
    const auto recovered = vc::lasagna::LasagnaDataset::openLocation(location, options);
    CHECK(bucket.fetches() == 2);
    CHECK(recovered.manifest().version == 2);
}

TEST_CASE("LasagnaDataset keeps a signed location's query for requests, not for names")
{
    const auto dir = makeTmpDir("signed-query");
    vc::test::LoopbackHttpServer server;
    const std::string base = server.url();
    server.setHandler([base](const vc::test::LoopbackHttpServer::Request& request)
                          -> std::optional<vc::test::LoopbackHttpServer::Response> {
        const auto sig = request.query.find("sig");
        if (sig == request.query.end() || sig->second != "secret")
            return vc::test::LoopbackHttpServer::Response{403, "unsigned", {}, false};
        if (request.path != "/art/data.lasagna.json")
            return vc::test::LoopbackHttpServer::Response{404, "", {}, false};
        return vc::test::LoopbackHttpServer::Response{200,
            R"({"version":2,"groups":{"nx":{"zarr":"nx.ome.zarr/4","scaledown":4,"channels":["nx"]},)"
            R"("ny":{"zarr":")" + base + R"(/other/ny.ome.zarr/4?sig=secret","scaledown":4,"channels":["ny"]}}})",
            {}, false};
    });
    vc::lasagna::LasagnaDatasetOpenOptions options;
    options.remoteCacheRoot = dir;
    const std::string location = base + "/art/data.lasagna.json?sig=secret";
    const auto dataset = vc::lasagna::LasagnaDataset::openLocation(location, options);
    REQUIRE(server.requests().size() == 1);
    CHECK(server.requests().front().query.at("sig") == "secret");
    // The identity and the cache name carry no credential; requests and group locations do.
    CHECK(dataset.manifest().manifestLocation == base + "/art/data.lasagna.json");
    CHECK(dataset.manifest().manifestPath == vcr::zarr::decodedCachePath(fs::absolute(dir).lexically_normal(), location));
    CHECK(dataset.manifest().manifestPath.string().find("sig=") == std::string::npos);
    CHECK(dataset.manifest().remoteSourceBaseLocation == base + "/art?sig=secret");
    REQUIRE(dataset.manifest().groups.size() == 2);
    CHECK(dataset.manifest().groups[0].sourceLocation == base + "/art/nx.ome.zarr/4?sig=secret");
    CHECK(dataset.manifest().groups[1].sourceLocation == base + "/other/ny.ome.zarr/4?sig=secret");
    // Cached: no request. Refreshed: one more, signed.
    (void)vc::lasagna::LasagnaDataset::openLocation(location, options);
    CHECK(server.requests().size() == 1);
    options.refreshManifest = true;
    (void)vc::lasagna::LasagnaDataset::openLocation(location, options);
    CHECK(server.requests().size() == 2);
    CHECK(server.requests().back().query.at("sig") == "secret");
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
