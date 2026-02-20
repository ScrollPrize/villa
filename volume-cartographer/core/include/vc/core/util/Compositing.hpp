#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <cstdint>

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

    // Directional lighting parameters
    bool lightingEnabled = false;     // Enable surface lighting
    float lightAzimuth = 45.0f;       // Light direction azimuth (degrees, 0=right, 90=up)
    float lightElevation = 45.0f;     // Light direction elevation (degrees above horizon)
    float lightDiffuse = 0.7f;        // Diffuse lighting strength (0-1)
    float lightAmbient = 0.3f;        // Ambient lighting (0-1, ensures shadows aren't pure black)

    // Pre-processing
    uint8_t isoCutoff = 0;           // Highpass filter: values below this are set to 0

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

    // Postprocessing (applied after composite render)
    bool postStretchValues = false;
    bool postRemoveSmallComponents = false;
    int postMinComponentSize = 50;

    bool operator==(const CompositeRenderSettings&) const = default;
};

// Apply postprocessing to a composited image (stretch values, remove small components)
void postprocessComposite(cv::Mat_<uint8_t>& img, const CompositeRenderSettings& settings);

// Layer values for a single pixel across all layers
// Used by compositing methods to process per-pixel data
struct LayerStack {
    std::vector<float> values;  // Values at each layer (after cutoff/equalization)
    int validCount = 0;         // Number of valid (sampled) layers
};

// Compositing method interface
// Each method takes a stack of layer values and returns a single output value
namespace CompositeMethod {

float mean(const LayerStack& stack);
float max(const LayerStack& stack);
float min(const LayerStack& stack);
float alpha(const LayerStack& stack, const CompositeParams& params);
float beerLambert(const LayerStack& stack, const CompositeParams& params);

} // namespace CompositeMethod

// Apply compositing to a single pixel's layer stack
// Returns the final composited value (0-255)
float compositeLayerStack(
    const LayerStack& stack,
    const CompositeParams& params
);

// Utility: check if method requires all layer values to be stored
// (as opposed to running accumulator like max/min)
bool methodRequiresLayerStorage(const std::string& method);

// Utility: get list of available compositing methods
std::vector<std::string> availableCompositeMethods();

// Compute directional lighting factor for a surface normal
// Returns a multiplier (0-1) based on Lambertian diffuse lighting
// normal: surface normal (should be normalized)
// params: contains light direction and strength settings
float computeLightingFactor(const cv::Vec3f& normal, const CompositeParams& params);
