#pragma once

#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <numbers>

// Parameters for multi-layer compositing
struct CompositeParams {
    // Compositing method: "mean", "max", "min", "alpha", "beerLambert"
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

    // Volumetric-method shadow integration: at each view-ray sample a
    // secondary ray walks this many voxels toward the light, integrating
    // density to attenuate the voxel's emission. Higher = softer/more
    // accurate shadows, linear cost.
    int shadowSteps = 8;

    // Directional lighting parameters (applied per-sample in the compositor)
    bool lightingEnabled = false;     // Enable surface lighting
    // Source of the surface normal used for Lambertian shading:
    //   0 = mesh normal (describes the unrolled sheet orientation; flat for
    //       papyrus detail — can't reveal crackle or fiber).
    //   1 = volume gradient ∇V at the base sample position (reveals local
    //       density structure — this is where ink, fiber, and crackle live).
    int lightNormalSource = 0;
    float lightAzimuth = 45.0f;       // Light direction azimuth (degrees, 0=right, 90=up)
    float lightElevation = 45.0f;     // Light direction elevation (degrees above horizon)
    float lightDiffuse = 0.7f;        // Diffuse lighting strength (0-1)
    float lightAmbient = 0.3f;        // Ambient lighting (0-1, ensures shadows aren't pure black)

    // Pre-computed light direction (call updateLightDir() after changing azimuth/elevation)
    float lightDirX = 0.5f;
    float lightDirY = 0.5f;
    float lightDirZ = 0.70710678f;    // default: azimuth=45, elevation=45

    // Pre-processing
    uint8_t isoCutoff = 0;           // Highpass filter: values below this are set to 0

    // Ambient term for the `dvr` composite method. Added to the final
    // transmittance-weighted color so voids don't stay pitch-black in the
    // rendered output. 0 disables; 1-255 adds a flat background.
    float dvrAmbient = 0.0f;

    // PBR Cook-Torrance parameters for the `pbrIso` composite method.
    // roughness: 0 = mirror-smooth, 1 = fully diffuse (Lambertian).
    // metallic:  0 = dielectric (Fresnel F0=0.04), 1 = metal (F0≈0.7 —
    //            carbon-like, the only material in a carbonized scroll).
    // Uses lightDir + lightDiffuse + lightAmbient from the existing
    // lighting block for the light direction + intensity knobs.
    float pbrRoughness = 0.5f;
    float pbrMetallic = 0.0f;

    // Recompute lightDir from lightAzimuth/lightElevation (degrees)
    void updateLightDir() noexcept {
        float azRad = lightAzimuth * (std::numbers::pi_v<float> / 180.0f);
        float elRad = lightElevation * (std::numbers::pi_v<float> / 180.0f);
        float ce = std::cos(elRad);
        lightDirX = ce * std::cos(azRad);
        lightDirY = ce * std::sin(azRad);
        lightDirZ = std::sin(elRad);
    }

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

    bool useVolumeGradients = false;

    CompositeParams params;  // method, alpha, BL, lighting, isoCutoff

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

// Utility: check if method requires all layer values to be stored
// (as opposed to running accumulator like max/min)
bool methodRequiresLayerStorage(const std::string& method) noexcept;

// Compute directional lighting factor for a surface normal
// Returns a multiplier (0-1) based on Lambertian diffuse lighting
// normal: surface normal (should be normalized)
// params: contains light direction and strength settings
float computeLightingFactor(const cv::Vec3f& normal, const CompositeParams& params) noexcept;
