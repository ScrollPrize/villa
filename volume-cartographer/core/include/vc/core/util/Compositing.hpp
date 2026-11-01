#pragma once

#include <vc_reader/composite.hpp>

#include <string>
#include <vector>
#include <cstdint>

// Parameters for multi-layer compositing: the reduction of the sampled layers
// (vcr::composite), unless `volumetric`, the GUI's perspective volumetric mode, which
// composites its own layer stack with the camera below.
struct CompositeParams {
    vcr::composite::Options options;
    bool volumetric = false;

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

    // Lambertian shading strength for the flattened volumetric view, in
    // [0,1]. Normals are computed from finite differences in the extracted
    // voxel layer stack. Zero preserves the existing unlit output.
    float lightingStrength = 0.0f;

    bool operator==(const CompositeParams&) const = default;
};

// The persisted and bridge names of a composite method: "mean", "max", "min", "median",
// "alpha", "beerLambert" (vcr::composite reductions) and "volumetric".
inline std::string compositeMethodName(const CompositeParams& params)
{
    using Method = vcr::composite::Method;
    if (params.volumetric) return "volumetric";
    switch (params.options.method) {
    case Method::Mean: return "mean";
    case Method::Max: return "max";
    case Method::Min: return "min";
    case Method::Median: return "median";
    case Method::MinAbs: return "minAbs";
    case Method::Alpha: return "alpha";
    case Method::BeerLambert: return "beerLambert";
    }
    return "mean";
}

// Sets the method a name stands for; false, leaving `params` as they were, for an unknown name.
inline bool setCompositeMethod(CompositeParams& params, const std::string& name)
{
    using Method = vcr::composite::Method;
    if (name == "volumetric") {
        params.volumetric = true;
        return true;
    }
    Method method;
    if (name == "mean") method = Method::Mean;
    else if (name == "max") method = Method::Max;
    else if (name == "min") method = Method::Min;
    else if (name == "median") method = Method::Median;
    else if (name == "minAbs") method = Method::MinAbs;
    else if (name == "alpha") method = Method::Alpha;
    else if (name == "beerLambert") method = Method::BeerLambert;
    else return false;
    params.volumetric = false;
    params.options.method = method;
    return true;
}

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
