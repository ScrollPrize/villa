#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "DisplayPalette.hpp"
#include "Colormaps.hpp"

#include <algorithm>
#include <array>
#include <cstdint>

using LUT = std::array<uint32_t, 256>;

static constexpr uint32_t kAlpha = 0xFF000000u;

static uint8_t A(uint32_t p) { return uint8_t((p >> 24) & 0xFF); }
static uint8_t R(uint32_t p) { return uint8_t((p >> 16) & 0xFF); }
static uint8_t G(uint32_t p) { return uint8_t((p >>  8) & 0xFF); }
static uint8_t B(uint32_t p) { return uint8_t( p        & 0xFF); }

TEST_CASE("buildWindowLevelColormapLut: empty id == grayscale buildWindowLevelLut")
{
    LUT gray{}, cm{};
    vcr::display::windowLevelLut(gray, 32.0f, 224.0f);
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
    CHECK(vcr::display::blendArgb(kAlpha, lut[100], 1.0f) == (kAlpha | (100u << 16)));
    CHECK(vcr::display::blendArgb(0xFFFFFFFFu, lut[100], 1.0f) == 0xFFFFFFFFu - 100u * 0x101u);
}

TEST_CASE("buildOverlayLut: intensity-as-alpha scales straight maps")
{
    LUT color{}, lut{};
    vc::buildWindowLevelColormapLut(color, 0.0f, 255.0f, "viridis");
    vc::buildOverlayLut(lut, 0.0f, 255.0f, "viridis", true);
    CHECK(A(lut[255]) == 255);
    CHECK((lut[255] & 0x00FFFFFFu) == (color[255] & 0x00FFFFFFu));
    CHECK(G(lut[51]) == (G(color[51]) * 51 + 127) / 255);

}

TEST_CASE("buildOverlayLut: intensity-as-alpha alpha is the brightest channel for black-based maps")
{
    for (const char* id : {"fire", "magma", "cyan", ""}) {
        CAPTURE(id);
        LUT color{}, lut{};
        vc::buildWindowLevelColormapLut(color, 20.0f, 200.0f, id);
        vc::buildOverlayLut(lut, 20.0f, 200.0f, id, true);
        for (int i = 0; i < 256; ++i) {
            CAPTURE(i);
            if (i <= 20) {
                CHECK(lut[i] == 0u);
                continue;
            }
            CHECK(A(lut[i]) == std::max({R(color[i]), G(color[i]), B(color[i])}));
            // Over black, the overlay is exactly the opaque colormap.
            CHECK(vcr::display::blendArgb(kAlpha, lut[i], 1.0f) == color[i]);
        }
    }
    LUT magma{};
    vc::buildOverlayLut(magma, 20.0f, 200.0f, "magma", true);
    CHECK(A(magma[255]) == 253);  // magma's top colour is not white
}

TEST_CASE("buildOverlayLut: label maps ignore intensity-as-alpha")
{
    LUT plain{}, alpha{};
    vc::buildOverlayLut(plain, 5.0f, 40.0f, "glasbey_black0", false);
    vc::buildOverlayLut(alpha, 5.0f, 40.0f, "glasbey_black0", true);
    CHECK(plain == alpha);
}
