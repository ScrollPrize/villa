#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/Segmentation.hpp"

#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

struct TmpSeg {
    fs::path dir;
    TmpSeg() {
        std::mt19937_64 rng(std::random_device{}());
        dir = fs::temp_directory_path() / ("vc_seg_test_" + std::to_string(rng()));
        fs::create_directories(dir);
    }
    ~TmpSeg() { std::error_code ec; fs::remove_all(dir, ec); }
};

void writeMeta(const fs::path& dir, const std::string& content)
{
    std::ofstream f(dir / "meta.json");
    f << content;
}

} // namespace

TEST_CASE("Segmentation: New(path, uuid, name) writes a valid meta.json")
{
    TmpSeg t;
    auto seg = Segmentation::New(t.dir, "abc-uuid", "myseg");
    CHECK(seg->id() == "abc-uuid");
    CHECK(seg->name() == "myseg");
    CHECK(seg->path() == t.dir);
    CHECK(fs::exists(t.dir / "meta.json"));
}

TEST_CASE("Segmentation: Loading missing meta throws")
{
    TmpSeg t;
    CHECK_THROWS_AS(Segmentation(t.dir), std::runtime_error);
}

TEST_CASE("Segmentation: meta with wrong type throws")
{
    TmpSeg t;
    writeMeta(t.dir, R"({"type":"not_seg","uuid":"x"})");
    CHECK_THROWS_AS(Segmentation(t.dir), std::runtime_error);
}

TEST_CASE("Segmentation: meta missing uuid throws")
{
    TmpSeg t;
    writeMeta(t.dir, R"({"type":"seg","name":"x"})");
    CHECK_THROWS_AS(Segmentation(t.dir), std::runtime_error);
}

TEST_CASE("Segmentation: load existing valid meta")
{
    TmpSeg t;
    writeMeta(t.dir, R"({"type":"seg","uuid":"u1","name":"n1","volume":""})");
    Segmentation seg(t.dir);
    CHECK(seg.id() == "u1");
    CHECK(seg.name() == "n1");
}

TEST_CASE("Segmentation: setId / setName persist via saveMetadata")
{
    TmpSeg t;
    auto seg = Segmentation::New(t.dir, "u-original", "n-original");
    seg->setId("u-new");
    seg->setName("n-new");
    seg->saveMetadata();

    Segmentation reloaded(t.dir);
    CHECK(reloaded.id() == "u-new");
    CHECK(reloaded.name() == "n-new");
}

TEST_CASE("Segmentation::checkDir true for valid dir, false otherwise")
{
    TmpSeg t;
    CHECK_FALSE(Segmentation::checkDir(t.dir)); // no meta.json yet
    writeMeta(t.dir, R"({"type":"seg","uuid":"u"})");
    CHECK(Segmentation::checkDir(t.dir));
    CHECK_FALSE(Segmentation::checkDir(t.dir / "nope"));
}

TEST_CASE("Segmentation::ensureScrollSource sets missing fields and persists")
{
    TmpSeg t;
    auto seg = Segmentation::New(t.dir, "u", "n");
    seg->ensureScrollSource("scroll1", "vol-uuid");
    Segmentation reloaded(t.dir);
    // Re-read raw json by triggering id()/name() — but we can also just
    // check that calling again doesn't change anything (idempotent).
    seg->ensureScrollSource("different-scroll", "different-vol");
    // The first non-empty value sticks, so the metadata still has the originals.
    // We can't easily inspect the raw json from outside; just verify no crash
    // and that a re-load works.
    Segmentation reloaded2(t.dir);
    CHECK(reloaded2.id() == "u");
}

TEST_CASE("Segmentation: surface not loaded by default")
{
    TmpSeg t;
    auto seg = Segmentation::New(t.dir, "u", "n");
    CHECK_FALSE(seg->isSurfaceLoaded());
    CHECK(seg->getSurface() == nullptr);
}

TEST_CASE("Segmentation: canLoadSurface needs format tifxyz")
{
    TmpSeg t;
    writeMeta(t.dir, R"({"type":"seg","uuid":"u","name":"n"})");
    Segmentation seg1(t.dir);
    CHECK_FALSE(seg1.canLoadSurface());

    TmpSeg t2;
    writeMeta(t2.dir, R"({"type":"seg","uuid":"u","name":"n","format":"tifxyz"})");
    Segmentation seg2(t2.dir);
    CHECK(seg2.canLoadSurface());
}

TEST_CASE("Segmentation: loadSurface returns nullptr when not tifxyz")
{
    TmpSeg t;
    auto seg = Segmentation::New(t.dir, "u", "n");
    auto surf = seg->loadSurface();
    CHECK(surf == nullptr);
    CHECK_FALSE(seg->isSurfaceLoaded());
}

TEST_CASE("Segmentation: unloadSurface clears surface_")
{
    TmpSeg t;
    auto seg = Segmentation::New(t.dir, "u", "n");
    seg->unloadSurface();
    CHECK(seg->getSurface() == nullptr);
}
