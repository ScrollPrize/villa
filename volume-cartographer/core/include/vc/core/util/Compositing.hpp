#pragma once

#include <string>
#include <vector>
#include <cstdint>

// Parameters for multi-layer compositing
struct CompositeParams {
    // Compositing method: "mean", "max", "min", "alpha", "beerLambert",
    // "volumetric" (GUI-only perspective volumetric mode; ignored by the
    // shared compositeLayerStack dispatch)
    std::string method = "mean";

    // Alpha compositing parameters
    float alphaMin = 0.0f;
    float alphaMax = 1.0f;
    float alphaOpacity = 1.0f;
    float alphaCutoff = 1.0f;

    // Beer-Lambert parameters (volume rendering with emission + absorption)
    float blExtinction = 1.5f;        // Absorption coefficient (higher = more opaque)
    float blEmission = 1.5f;          // Emission scale (higher = brighter)
    float blAmbient = 0.1f;           // Ambient light (background illumination)

    // Pre-processing
    uint8_t isoCutoff = 0;           // Highpass filter: values below this are set to 0

    // Volumetric mode camera + transfer function (GUI viewer only; unused by
    // the CLI/scalar methods). Turntable model: camAzimuthDeg spins the patch
    // about the surface normal (a pure in-plane rotation of the image);
    // camTiltDeg then tips the view about the screen-horizontal axis, from
    // straight down (0) toward flat — on screen always a vertical
    // foreshortening. camPerspective 0 = orthographic, (0..1] = pinhole
    // perspective with half-FOV = camPerspective * 45deg (coverage matched to
    // orthographic at the view-center depth). tfGamma shapes the opacity
    // transfer function.
    float camAzimuthDeg = 0.0f;
    float camTiltDeg = 0.0f;
    float camPerspective = 0.0f;
    float tfGamma = 1.5f;
    // Relief exaggeration: stretches the slab along the normal (w) axis
    // before the rotated render. Segments are typically much wider than the
    // slab is thick, so 1:1 relief is barely visible under tilt.
    float wScale = 2.5f;

    bool operator==(const CompositeParams&) const = default;
};

// Consolidated rendering settings for composite mode (Qt-free)
struct CompositeRenderSettings {
    bool enabled = false;
    int layersFront = 8;
    int layersBehind = 0;
    bool reverseDirection = false;

    bool planeEnabled = false;
    int planeLayersFront = 4;
    int planeLayersBehind = 4;
    bool planeReverseDirection = false;

    CompositeParams params;  // method, alpha, BL, isoCutoff

    bool operator==(const CompositeRenderSettings&) const = default;
};

// Composite settings for the overlay volume, independent of the primary
// volume's CompositeRenderSettings. Only max/mean/min are supported; applies
// to generated-surface views only (plane views stay single-slice).
struct OverlayCompositeSettings {
    bool enabled = false;
    std::string method = "max";  // "max" | "mean" | "min"
    int layersFront = 8;
    int layersBehind = 0;

    bool operator==(const OverlayCompositeSettings&) const = default;
};

// Layer values for a single pixel across all layers
// Used by compositing methods to process per-pixel data
struct LayerStack {
    std::vector<float> values;  // Values at each layer (after cutoff/equalization)
    int validCount = 0;         // Number of valid (sampled) layers
};

// Compositing method interface
// Each method takes a stack of layer values and returns a single output value
namespace CompositeMethod {

float mean(const LayerStack& stack) noexcept;
float max(const LayerStack& stack) noexcept;
float min(const LayerStack& stack) noexcept;
float alpha(const LayerStack& stack, const CompositeParams& params) noexcept;
float beerLambert(const LayerStack& stack, const CompositeParams& params) noexcept;

} // namespace CompositeMethod

// Apply compositing to a single pixel's layer stack
// Returns the final composited value (0-255)
float compositeLayerStack(
    const LayerStack& stack,
    const CompositeParams& params
) noexcept;
