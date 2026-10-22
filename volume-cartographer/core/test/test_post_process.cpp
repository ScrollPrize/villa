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


TEST_CASE("buildWindowLevelLut: identity window [0, 255]")
{
    LUT lut{};
    vc::buildWindowLevelLut(lut, 0.0f, 255.0f);

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
    vc::buildWindowLevelLut(lut, 64.0f, 192.0f);

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
    vc::buildWindowLevelLut(lut, 100.0f, 100.0f);

    CHECK(R(lut[99])  == 0);
    CHECK(R(lut[100]) == 0);
    CHECK(R(lut[101]) == 255);
    CHECK(R(lut[255]) == 255);
}

TEST_CASE("buildWindowLevelLut: inverted window collapses to lo+1")
{
    LUT lut{};
    vc::buildWindowLevelLut(lut, 200.0f, 50.0f);

    CHECK(R(lut[199]) == 0);
    CHECK(R(lut[200]) == 0);
    CHECK(R(lut[201]) == 255);
}

TEST_CASE("buildWindowLevelLut: out-of-range endpoints clamp into [0, 255]")
{
    LUT lut{};
    vc::buildWindowLevelLut(lut, -50.0f, 500.0f);

    CHECK(R(lut[0])   == 0);
    CHECK(R(lut[128]) == 128);
    CHECK(R(lut[255]) == 255);
}

TEST_CASE("buildWindowLevelLut: noexcept contract holds")
{
    static_assert(noexcept(vc::buildWindowLevelLut(
        std::declval<LUT&>(), 0.0f, 0.0f)));
}


TEST_CASE("buildWindowLevelColormapLut: empty id == grayscale buildWindowLevelLut")
{
    LUT gray{}, cm{};
    vc::buildWindowLevelLut(gray, 32.0f, 224.0f);
    vc::buildWindowLevelColormapLut(cm, 32.0f, 224.0f, std::string{});

    CHECK(gray == cm);
}

TEST_CASE("buildWindowLevelColormapLut: Tint colormap modulates gray ramp")
{
    LUT lut{};
    vc::buildWindowLevelColormapLut(lut, 0.0f, 255.0f, "red");

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
    vc::buildWindowLevelColormapLut(lut, 0.0f, 255.0f, "glasbey_black0");

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
    vc::buildWindowLevelColormapLut(lut, 0.0f, 255.0f, "fire");

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
    vc::buildWindowLevelColormapLut(fire,    0.0f, 255.0f, "fire");
    vc::buildWindowLevelColormapLut(unknown, 0.0f, 255.0f, "definitely-not-a-map");

    CHECK(fire == unknown);
}


TEST_CASE("buildOverlayLut: default mode is the colormap LUT band-passed to the window")
{
    LUT color{}, lut{};
    vc::buildWindowLevelColormapLut(color, 40.0f, 200.0f, "viridis");
    vc::buildOverlayLut(lut, 40.0f, 200.0f, "viridis", false);
    for (int i = 0; i < 256; ++i)
        CHECK(lut[i] == ((i >= 40 && i <= 200) ? color[i] : 0u));
}

TEST_CASE("buildOverlayLut: intensity-as-alpha keeps a tint bright and ramps alpha")
{
    LUT lut{};
    vc::buildOverlayLut(lut, 0.0f, 255.0f, "red", true);
    CHECK(lut[0] == 0u);
    // Premultiplied red at alpha 100: over black it is 100, over white 255.
    CHECK(lut[100] == ((100u << 24) | (100u << 16)));
    CHECK(vc::blendOverlayArgb(kAlpha, lut[100], 1.0f) == (kAlpha | (100u << 16)));
    CHECK(vc::blendOverlayArgb(0xFFFFFFFFu, lut[100], 1.0f) == 0xFFFFFFFFu - 100u * 0x101u);
}

TEST_CASE("buildOverlayLut: intensity-as-alpha scales straight maps, labels are opaque")
{
    LUT color{}, lut{};
    vc::buildWindowLevelColormapLut(color, 0.0f, 255.0f, "viridis");
    vc::buildOverlayLut(lut, 0.0f, 255.0f, "viridis", true);
    CHECK(A(lut[255]) == 255);
    CHECK((lut[255] & 0x00FFFFFFu) == (color[255] & 0x00FFFFFFu));
    CHECK(G(lut[51]) == (G(color[51]) * 51 + 127) / 255);

    vc::buildOverlayLut(lut, 0.0f, 255.0f, "glasbey_black0", true);
    CHECK(lut[0] == 0u);
    CHECK(A(lut[7]) == 255);
}

TEST_CASE("blendOverlayArgb: opaque entry matches the plain opacity mix")
{
    const uint32_t base = 0xFF204060u, over = 0xFFC08010u;
    // (1 - 0.3) * b + 0.3 * o per channel, rounded.
    CHECK(vc::blendOverlayArgb(base, over, 0.3f) == 0xFF505348u);
    CHECK(vc::blendOverlayArgb(base, 0u, 0.3f) == base);
}
