#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/PostProcess.hpp"
#include "vc/core/render/Colormaps.hpp"

#include <array>
#include <cstdint>

using LUT = std::array<uint32_t, 256>;

static constexpr uint32_t kAlpha = 0xFF000000u;

static uint8_t A(uint32_t p) { return uint8_t((p >> 24) & 0xFF); }
static uint8_t R(uint32_t p) { return uint8_t((p >> 16) & 0xFF); }
static uint8_t G(uint32_t p) { return uint8_t((p >>  8) & 0xFF); }
static uint8_t B(uint32_t p) { return uint8_t( p        & 0xFF); }


TEST_CASE("buildWindowLevelLut: identity window [0, 255] no lighting")
{
    LUT lut{};
    vc::buildWindowLevelLut(lut, 0.0f, 255.0f, 1.0f);

    for (int i = 0; i < 256; ++i) {
        CHECK(A(lut[i]) == 255);
        CHECK(R(lut[i]) == G(lut[i]));
        CHECK(G(lut[i]) == B(lut[i]));
    }

    CHECK(R(lut[0])   == 0);
    CHECK(R(lut[128]) == 128);
    CHECK(R(lut[255]) == 255);
}

TEST_CASE("buildWindowLevelLut: narrow window stretches contrast")
{
    LUT lut{};
    vc::buildWindowLevelLut(lut, 64.0f, 192.0f, 1.0f);

    CHECK(R(lut[0])  == 0);
    CHECK(R(lut[63]) == 0);
    CHECK(R(lut[64]) == 0);
    CHECK(R(lut[128]) == 127);
    CHECK(R(lut[192]) == 255);
    CHECK(R(lut[255]) == 255);
}

TEST_CASE("buildWindowLevelLut: equal endpoints get separated by clamp")
{
    LUT lut{};
    vc::buildWindowLevelLut(lut, 100.0f, 100.0f, 1.0f);

    CHECK(R(lut[99])  == 0);
    CHECK(R(lut[100]) == 0);
    CHECK(R(lut[101]) == 255);
    CHECK(R(lut[255]) == 255);
}

TEST_CASE("buildWindowLevelLut: inverted window collapses to lo+1")
{
    LUT lut{};
    vc::buildWindowLevelLut(lut, 200.0f, 50.0f, 1.0f);

    CHECK(R(lut[199]) == 0);
    CHECK(R(lut[200]) == 0);
    CHECK(R(lut[201]) == 255);
}

TEST_CASE("buildWindowLevelLut: out-of-range endpoints clamp into [0, 255]")
{
    LUT lut{};
    vc::buildWindowLevelLut(lut, -50.0f, 500.0f, 1.0f);

    CHECK(R(lut[0])   == 0);
    CHECK(R(lut[128]) == 128);
    CHECK(R(lut[255]) == 255);
}

TEST_CASE("buildWindowLevelLut: lightFactor < 1 scales input before window")
{
    LUT lutFull{}, lutHalf{};
    vc::buildWindowLevelLut(lutFull, 0.0f, 255.0f, 1.0f);
    vc::buildWindowLevelLut(lutHalf, 0.0f, 255.0f, 0.5f);

    CHECK(R(lutHalf[0])   == 0);
    CHECK(R(lutHalf[100]) == 50);
    CHECK(R(lutHalf[200]) == 100);
    CHECK(R(lutHalf[255]) == 127);

    for (int i = 0; i < 256; ++i) {
        CHECK(R(lutFull[i]) == i);
    }
}

TEST_CASE("buildWindowLevelLut: lightFactor=0 collapses everything to 0")
{
    LUT lut{};
    vc::buildWindowLevelLut(lut, 0.0f, 255.0f, 0.0f);

    for (int i = 0; i < 256; ++i) {
        CHECK(R(lut[i]) == 0);
        CHECK(A(lut[i]) == 255);
    }
}

TEST_CASE("buildWindowLevelLut: noexcept contract holds")
{
    static_assert(noexcept(vc::buildWindowLevelLut(
        std::declval<LUT&>(), 0.0f, 0.0f, 0.0f)));
}


TEST_CASE("buildWindowLevelColormapLut: empty id == grayscale buildWindowLevelLut")
{
    LUT gray{}, cm{};
    vc::buildWindowLevelLut(gray, 32.0f, 224.0f, 1.0f);
    vc::buildWindowLevelColormapLut(cm, 32.0f, 224.0f, std::string{}, 1.0f);

    CHECK(gray == cm);
}

TEST_CASE("buildWindowLevelColormapLut: Tint colormap modulates gray ramp")
{
    LUT lut{};
    vc::buildWindowLevelColormapLut(lut, 0.0f, 255.0f, "red", 1.0f);

    CHECK(lut[0] == kAlpha);

    for (int i = 1; i < 256; ++i) {
        CHECK(A(lut[i]) == 255);
        CHECK(G(lut[i]) == 0);
        CHECK(B(lut[i]) == 0);
        CHECK(R(lut[i]) == i);
    }
}

TEST_CASE("buildWindowLevelColormapLut: DiscreteLut path forces black at 0")
{
    LUT lut{};
    vc::buildWindowLevelColormapLut(lut, 0.0f, 255.0f, "glasbey_black0", 1.0f);

    CHECK(lut[0] == kAlpha);
    for (int i = 0; i < 256; ++i) {
        CHECK(A(lut[i]) == 255);
    }
    int nonBlack = 0;
    for (int i = 1; i < 256; ++i) {
        if ((lut[i] & 0x00FFFFFFu) != 0) ++nonBlack;
    }
    CHECK(nonBlack > 0);
}

TEST_CASE("buildWindowLevelColormapLut: OpenCV path produces non-trivial palette")
{
    LUT lut{};
    vc::buildWindowLevelColormapLut(lut, 0.0f, 255.0f, "fire", 1.0f);

    CHECK(lut[0] == kAlpha);
    bool sawNonGray = false;
    for (int i = 1; i < 256; ++i) {
        if (R(lut[i]) != B(lut[i])) { sawNonGray = true; break; }
    }
    CHECK(sawNonGray);
}

TEST_CASE("buildWindowLevelColormapLut: unknown id falls back to first spec")
{
    LUT fire{}, unknown{};
    vc::buildWindowLevelColormapLut(fire,    0.0f, 255.0f, "fire",                 1.0f);
    vc::buildWindowLevelColormapLut(unknown, 0.0f, 255.0f, "definitely-not-a-map", 1.0f);

    CHECK(fire == unknown);
}

TEST_CASE("buildWindowLevelColormapLut: lightFactor propagates through colormap")
{
    LUT bright{}, dim{};
    vc::buildWindowLevelColormapLut(bright, 0.0f, 255.0f, "red", 1.0f);
    vc::buildWindowLevelColormapLut(dim,    0.0f, 255.0f, "red", 0.25f);

    bool anyDimmer = false;
    for (int i = 1; i < 256; ++i) {
        if (R(bright[i]) > 0 && R(dim[i]) < R(bright[i])) {
            anyDimmer = true; break;
        }
    }
    CHECK(anyDimmer);
}
