// Coverage for core/src/render/Colormaps.cpp.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/Colormaps.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace vc;

TEST_CASE("specs() returns a non-empty fixed list")
{
    const auto& s = specs();
    CHECK_FALSE(s.empty());
    // Known IDs from the impl.
    bool hasFire = false, hasViridis = false, hasGlasbey = false;
    for (const auto& sp : s) {
        if (sp.id == "fire") hasFire = true;
        if (sp.id == "viridis") hasViridis = true;
        if (sp.id == "glasbey_black0") hasGlasbey = true;
    }
    CHECK(hasFire);
    CHECK(hasViridis);
    CHECK(hasGlasbey);
}

TEST_CASE("resolve(): valid id returns matching spec; unknown id is handled")
{
    const auto& sp = resolve("fire");
    CHECK(sp.id == "fire");
    CHECK(sp.kind == OverlayColormapKind::OpenCv);
    // resolve() of an unknown id — impl behaviour: returns some default
    // (commonly first entry) without throwing. We just exercise the path
    // and accept whatever non-empty id it returns.
    const auto& sp2 = resolve("__nope__");
    CHECK_FALSE(sp2.id.empty());
}

TEST_CASE("entries(SharedOnly) excludes overlay-only entries")
{
    const auto& shared = entries(EntryScope::SharedOnly);
    for (const auto& e : shared) {
        // Glasbey is OverlayOnly per the impl; must not appear here.
        CHECK(e.id != "glasbey_black0");
    }
    CHECK_FALSE(shared.empty());
}

TEST_CASE("entries(OverlayCompatible) includes overlay-only entries")
{
    const auto& full = entries(EntryScope::OverlayCompatible);
    bool hasGlasbey = false;
    for (const auto& e : full) if (e.id == "glasbey_black0") hasGlasbey = true;
    CHECK(hasGlasbey);
}

TEST_CASE("makeColors: OpenCv-kind colormap fills outBuf")
{
    cv::Mat_<uint8_t> values(4, 4);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            values(r, c) = static_cast<uint8_t>(r * 16 + c * 16);
    std::vector<uint32_t> out(4 * 4, 0);
    const auto& spec = resolve("fire");
    makeColors(values, spec, out.data(), /*outStride=*/4);
    // At least one non-zero output (zero input maps to alpha=FF anyway under most LUTs)
    bool anyNon00000000 = false;
    for (auto v : out) if (v != 0) anyNon00000000 = true;
    CHECK(anyNon00000000);
}

TEST_CASE("makeColors: Tint-kind colormap")
{
    cv::Mat_<uint8_t> values(2, 2, uint8_t(200));
    std::vector<uint32_t> out(2 * 2, 0);
    const auto& spec = resolve("red");
    makeColors(values, spec, out.data(), 2);
    // Red tint: high R channel
    uint8_t r = (out[0] >> 16) & 0xFF;
    CHECK(int(r) > 100);
}

TEST_CASE("makeColors: DiscreteLut (glasbey) maps label 0 to black, alpha=0xFF")
{
    cv::Mat_<uint8_t> values(2, 2, uint8_t(0));
    std::vector<uint32_t> out(2 * 2, 0);
    const auto& spec = resolve("glasbey_black0");
    makeColors(values, spec, out.data(), 2);
    // Label 0 → black (0xFF000000)
    CHECK(out[0] == 0xFF000000u);
}

TEST_CASE("applyPackedLut: identity-ish LUT round-trips")
{
    // Build a LUT mapping i -> ARGB with R=G=B=i.
    std::array<uint32_t, 256> lut{};
    for (int i = 0; i < 256; ++i) {
        lut[i] = 0xFF000000u | (uint32_t(i) << 16) | (uint32_t(i) << 8) | uint32_t(i);
    }
    cv::Mat_<uint8_t> values(2, 4);
    for (int c = 0; c < 4; ++c) {
        values(0, c) = static_cast<uint8_t>(c * 40);
        values(1, c) = static_cast<uint8_t>(c * 40 + 10);
    }
    std::vector<uint32_t> out(2 * 4, 0);
    applyPackedLut(values, lut.data(), out.data(), 4);
    CHECK(out[0] == lut[0]);
    CHECK(out[1] == lut[40]);
    CHECK(out[4] == lut[10]);
}

TEST_CASE("applyPackedLut handles non-contiguous stride")
{
    std::array<uint32_t, 256> lut{};
    for (int i = 0; i < 256; ++i) lut[i] = 0xFF000000u | uint32_t(i);
    cv::Mat_<uint8_t> values(2, 2, uint8_t(42));
    // outStride=4 leaves padding columns 2 and 3 untouched.
    std::vector<uint32_t> out(2 * 4, 0xDEADBEEFu);
    applyPackedLut(values, lut.data(), out.data(), 4);
    CHECK(out[0] == lut[42]);
    CHECK(out[1] == lut[42]);
    CHECK(out[2] == 0xDEADBEEFu); // padding preserved
    CHECK(out[4] == lut[42]);
}
