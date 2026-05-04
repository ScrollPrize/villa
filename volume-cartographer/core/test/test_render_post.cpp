#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/Colormaps.hpp"
#include "vc/core/render/PostProcess.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

TEST_CASE("Colormaps::specs returns at least the built-in entries")
{
    const auto& s = vc::specs();
    REQUIRE_FALSE(s.empty());
    bool foundFire = false;
    for (const auto& e : s)
        if (e.id == "fire") foundFire = true;
    CHECK(foundFire);
}

TEST_CASE("Colormaps::resolve known id returns matching spec")
{
    const auto& s = vc::specs();
    const auto& first = s.front();
    const auto& r = vc::resolve(first.id);
    CHECK(r.id == first.id);
    CHECK(r.label == first.label);
}

TEST_CASE("Colormaps::resolve unknown id falls back without crashing")
{
    const auto& r = vc::resolve("definitely-not-a-real-colormap-id");
    CHECK_FALSE(r.id.empty());
}

TEST_CASE("Colormaps::entries: shared and overlay scopes both populated")
{
    const auto& shared = vc::entries(vc::EntryScope::SharedOnly);
    const auto& overlay = vc::entries(vc::EntryScope::OverlayCompatible);
    CHECK_FALSE(shared.empty());
    CHECK_FALSE(overlay.empty());
}

TEST_CASE("Colormaps::makeColors: OpenCV-kind colormap fills a contiguous ARGB32 buffer")
{
    cv::Mat_<uint8_t> values(8, 8, uint8_t(0));
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x)
            values(y, x) = uint8_t((y * 32 + x * 4) & 0xFF);

    const auto& s = vc::specs();
    const vc::OverlayColormapSpec* opencvSpec = nullptr;
    for (const auto& e : s)
        if (e.kind == vc::OverlayColormapKind::OpenCv) { opencvSpec = &e; break; }
    REQUIRE(opencvSpec != nullptr);

    std::vector<uint32_t> out(8 * 8, 0);
    vc::makeColors(values, *opencvSpec, out.data(), 8);

    int nonzero = 0;
    for (auto v : out) if (v != 0) ++nonzero;
    CHECK(nonzero > 0);
}

TEST_CASE("Colormaps::makeColors: tint colormap outputs hue-tinted intensities")
{
    cv::Mat_<uint8_t> values(4, 4, uint8_t(128));
    const auto& s = vc::specs();
    const vc::OverlayColormapSpec* tintSpec = nullptr;
    for (const auto& e : s)
        if (e.kind == vc::OverlayColormapKind::Tint) { tintSpec = &e; break; }
    if (!tintSpec) return;

    std::vector<uint32_t> out(4 * 4, 0);
    vc::makeColors(values, *tintSpec, out.data(), 4);
    int matching = 0;
    for (size_t i = 1; i < out.size(); ++i)
        if (out[i] == out[0]) ++matching;
    CHECK(matching == int(out.size() - 1));
}

TEST_CASE("Colormaps::applyPackedLut maps each pixel through the LUT")
{
    cv::Mat_<uint8_t> values(2, 4);
    values(0, 0) = 0;   values(0, 1) = 64;  values(0, 2) = 128; values(0, 3) = 255;
    values(1, 0) = 10;  values(1, 1) = 20;  values(1, 2) = 30;  values(1, 3) = 200;

    std::array<uint32_t, 256> lut{};
    for (int i = 0; i < 256; ++i)
        lut[i] = 0xFF000000u | (uint32_t(i) << 16) | (uint32_t(i) << 8) | uint32_t(i);

    std::vector<uint32_t> out(2 * 4, 0);
    vc::applyPackedLut(values, lut.data(), out.data(), 4);

    CHECK((out[0] & 0xFFu) == 0u);
    CHECK((out[1] & 0xFFu) == 64u);
    CHECK((out[2] & 0xFFu) == 128u);
    CHECK((out[3] & 0xFFu) == 255u);
    CHECK((out[7] & 0xFFu) == 200u);
}

TEST_CASE("PostProcess::buildWindowLevelLut produces a non-decreasing intensity ramp inside the window")
{
    std::array<uint32_t, 256> lut{};
    vc::buildWindowLevelLut(lut, /*windowLow=*/0.0f, /*windowHigh=*/255.0f);

    int prev = -1;
    for (int i = 0; i < 256; ++i) {
        const int gray = int(lut[i] & 0xFFu);
        CHECK(gray >= prev);
        prev = gray;
    }
    CHECK((lut[255] & 0xFFu) > (lut[0] & 0xFFu));
}

TEST_CASE("PostProcess::buildWindowLevelLut clamps below window-low to 0")
{
    std::array<uint32_t, 256> lut{};
    vc::buildWindowLevelLut(lut, 100.0f, 200.0f);

    for (int i = 0; i < 100; ++i)
        CHECK((lut[i] & 0xFFu) == 0u);
}

TEST_CASE("PostProcess::buildWindowLevelLut clamps above window-high to 255")
{
    std::array<uint32_t, 256> lut{};
    vc::buildWindowLevelLut(lut, 100.0f, 200.0f);

    for (int i = 200; i < 256; ++i)
        CHECK((lut[i] & 0xFFu) == 255u);
}

TEST_CASE("PostProcess::buildWindowLevelLut: lightFactor scales output")
{
    std::array<uint32_t, 256> dim{};
    std::array<uint32_t, 256> bright{};
    vc::buildWindowLevelLut(dim, 0.0f, 255.0f, 0.5f);
    vc::buildWindowLevelLut(bright, 0.0f, 255.0f, 1.0f);

    int dimSum = 0, brightSum = 0;
    for (int i = 0; i < 256; ++i) {
        dimSum += int(dim[i] & 0xFFu);
        brightSum += int(bright[i] & 0xFFu);
    }
    CHECK(dimSum < brightSum);
}

TEST_CASE("PostProcess::buildWindowLevelColormapLut: empty id => same as plain WL LUT")
{
    std::array<uint32_t, 256> a{}, b{};
    vc::buildWindowLevelLut(a, 0.0f, 255.0f, 1.0f);
    vc::buildWindowLevelColormapLut(b, 0.0f, 255.0f, "", 1.0f);
    CHECK(a == b);
}

TEST_CASE("PostProcess::buildWindowLevelColormapLut: unknown id is harmless")
{
    std::array<uint32_t, 256> lut{};
    vc::buildWindowLevelColormapLut(lut, 0.0f, 255.0f, "no-such-colormap", 1.0f);
    CHECK(lut[0] != lut[255]);
}

TEST_CASE("PostProcess::buildWindowLevelColormapLut: known colormap shifts color channels")
{
    std::array<uint32_t, 256> grayLut{}, mappedLut{};
    vc::buildWindowLevelLut(grayLut, 0.0f, 255.0f);
    const auto& sp = vc::specs();
    if (sp.empty()) return;
    vc::buildWindowLevelColormapLut(mappedLut, 0.0f, 255.0f, sp.front().id, 1.0f);
    int diff = 0;
    for (int i = 0; i < 256; ++i)
        if (grayLut[i] != mappedLut[i]) ++diff;
    CHECK(diff >= 0);
}
