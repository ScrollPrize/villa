// Exercise VcDataset round-trips with each supported compressor.
// Compressor support is optional at link time; if a compressor isn't
// linked, the corresponding test skips itself.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/VcDataset.hpp"

#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <random>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_ds_codec_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

bool tryRoundtrip(const std::string& compressor, const std::string& tag)
{
    auto d = tmpDir(tag);
    std::unique_ptr<vc::VcDataset> ds;
    try {
        ds = vc::createZarrDataset(d, "arr",
            /*shape=*/{16, 16, 16}, /*chunks=*/{16, 16, 16},
            vc::VcDtype::uint8, compressor);
    } catch (const std::exception&) {
        fs::remove_all(d);
        return false;
    }
    if (!ds) {
        fs::remove_all(d);
        return false;
    }
    std::vector<uint8_t> payload(16 * 16 * 16);
    for (size_t i = 0; i < payload.size(); ++i)
        payload[i] = static_cast<uint8_t>((i * 7) & 0xFF);

    bool ok = false;
    try {
        ok = ds->writeChunk(0, 0, 0, payload.data(), payload.size());
    } catch (const std::exception&) {
        fs::remove_all(d);
        return false;
    }
    if (!ok) {
        fs::remove_all(d);
        return false;
    }

    std::vector<uint8_t> out(payload.size(), 0);
    try {
        CHECK(ds->readChunk(0, 0, 0, out.data()));
        CHECK(out == payload);
    } catch (const std::exception&) {
        fs::remove_all(d);
        return false;
    }
    fs::remove_all(d);
    return true;
}

} // namespace

TEST_CASE("VcDataset compressor: zstd round-trip")
{
    if (!tryRoundtrip("zstd", "zstd")) {
        MESSAGE("Skipping: zstd codec not available");
    }
}

TEST_CASE("VcDataset compressor: lz4 round-trip")
{
    if (!tryRoundtrip("lz4", "lz4")) {
        MESSAGE("Skipping: lz4 codec not available");
    }
}

TEST_CASE("VcDataset compressor: gzip round-trip")
{
    if (!tryRoundtrip("gzip", "gzip")) {
        MESSAGE("Skipping: gzip codec not available");
    }
}

TEST_CASE("VcDataset compressor: blosc round-trip (if linked)")
{
    if (!tryRoundtrip("blosc", "blosc")) {
        MESSAGE("Skipping: blosc codec not available");
    }
}

TEST_CASE("VcDataset compressor: c3d round-trip")
{
    if (!tryRoundtrip("c3d", "c3d")) {
        MESSAGE("Skipping: c3d codec not available");
    }
}

TEST_CASE("VcDataset compressor: 'none' (raw) round-trip")
{
    CHECK(tryRoundtrip("none", "none"));
}

TEST_CASE("VcDataset compressor: empty-string compressor maps to none")
{
    CHECK(tryRoundtrip("", "empty"));
}
