// Targeted coverage for the two small JSON helpers in ChunkedTensor.cpp.
// The header pulls in heavy deps but we only exercise the two free functions.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/ChunkedTensor.hpp"

#include <filesystem>
#include <fstream>
#include <random>
#include <string>

namespace fs = std::filesystem;

namespace {

fs::path makeDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_chunked_meta_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

} // namespace

TEST_CASE("write then read meta.json round-trips a real dataset path")
{
    auto dir = makeDir("roundtrip");
    auto datasetDir = makeDir("dataset");
    write_cache_meta_json(dir, datasetDir);
    auto got = read_cache_meta_dataset_path(dir / "meta.json");
    CHECK(got == fs::canonical(datasetDir));
    fs::remove_all(dir);
    fs::remove_all(datasetDir);
}

TEST_CASE("read_cache_meta_dataset_path: missing file returns empty")
{
    fs::path bogus = "/__nonexistent__/meta.json";
    auto got = read_cache_meta_dataset_path(bogus);
    CHECK(got.empty());
}

TEST_CASE("read_cache_meta_dataset_path: malformed JSON returns empty")
{
    auto dir = makeDir("malformed");
    {
        std::ofstream f(dir / "meta.json");
        f << "{ this is not json";
    }
    auto got = read_cache_meta_dataset_path(dir / "meta.json");
    CHECK(got.empty());
    fs::remove_all(dir);
}

TEST_CASE("read_cache_meta_dataset_path: missing key returns empty")
{
    auto dir = makeDir("nokey");
    {
        std::ofstream f(dir / "meta.json");
        f << R"({"other_key":"x"})";
    }
    auto got = read_cache_meta_dataset_path(dir / "meta.json");
    CHECK(got.empty());
    fs::remove_all(dir);
}

TEST_CASE("read_cache_meta_dataset_path: non-string value returns empty")
{
    auto dir = makeDir("nonstr");
    {
        std::ofstream f(dir / "meta.json");
        f << R"({"dataset_source_path":12345})";
    }
    auto got = read_cache_meta_dataset_path(dir / "meta.json");
    CHECK(got.empty());
    fs::remove_all(dir);
}

TEST_CASE("read_cache_meta_dataset_path: missing dataset dir returns empty")
{
    auto dir = makeDir("badds");
    {
        std::ofstream f(dir / "meta.json");
        f << R"({"dataset_source_path":"/__truly__/__not__/__here__"})";
    }
    auto got = read_cache_meta_dataset_path(dir / "meta.json");
    CHECK(got.empty());
    fs::remove_all(dir);
}
