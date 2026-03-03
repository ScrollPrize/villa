#pragma once
#include <span>
#include <vector>
#include <array>
#include <cstddef>
#include <cmath>
#include <concepts>
#include <algorithm>
#include <numeric>
#include <numbers>
#include <string_view>
#include <cstdint>

namespace utils {

// ---------------------------------------------------------------------------
// Compositing method selection
// ---------------------------------------------------------------------------

enum class CompositingMethod : std::uint8_t {
    mean,
    max,
    min,
    alpha,
    beer_lambert
};

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

struct CompositeParams {
    CompositingMethod method = CompositingMethod::mean;

    // Alpha blending params
    float alpha_min = 0.0f;       // values below this are fully transparent
    float alpha_max = 1.0f;       // values above this are fully opaque
    float alpha_opacity = 1.0f;   // per-layer opacity multiplier
    float alpha_cutoff = 1.0f;    // accumulated alpha threshold for early termination

    // Beer-Lambert params
    float extinction = 1.5f;      // absorption coefficient (higher = more opaque)
    float emission   = 1.5f;      // emission scale (higher = brighter)
    float ambient    = 0.1f;      // ambient light (background illumination)

    // Lighting params (for normal-aware compositing)
    float light_azimuth   = 0.0f;    // radians
    float light_elevation = 0.78f;   // radians (~45 degrees)
    float light_diffuse   = 0.7f;    // diffuse lighting strength (0-1)
    float light_ambient   = 0.3f;    // ambient lighting (0-1)

    // Pre-processing
    std::uint8_t iso_cutoff = 0;     // values below this are zeroed before compositing
};

/// Parse a compositing method name (e.g., "mean", "max", "alpha", "beerLambert").
/// Returns CompositingMethod::mean for unrecognized strings.
[[nodiscard]] constexpr CompositingMethod parse_compositing_method(
    std::string_view name) noexcept
{
    if (name == "mean")         return CompositingMethod::mean;
    if (name == "max")          return CompositingMethod::max;
    if (name == "min")          return CompositingMethod::min;
    if (name == "alpha")        return CompositingMethod::alpha;
    if (name == "beerLambert")  return CompositingMethod::beer_lambert;
    return CompositingMethod::mean;
}

/// Check if a method requires all layer values stored (vs running accumulator).
[[nodiscard]] constexpr bool method_requires_storage(CompositingMethod m) noexcept {
    return m != CompositingMethod::max &&
           m != CompositingMethod::min &&
           m != CompositingMethod::mean;
}

// ---------------------------------------------------------------------------
// Post-processing utilities
// ---------------------------------------------------------------------------

/// Clamp to [0, 1].
[[nodiscard]] constexpr float saturate(float v) noexcept {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

/// Window / level linear remap.
[[nodiscard]] constexpr float window_level(
    float value, float window, float level) noexcept
{
    if (window == 0.0f) return 0.0f;
    float lo = level - window * 0.5f;
    return saturate((value - lo) / window);
}

/// Normalize span in-place to [0, 1].
inline void value_stretch(std::span<float> data) noexcept {
    if (data.empty()) return;

    float lo = data[0];
    float hi = data[0];
    for (float v : data) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    float range = hi - lo;
    if (range == 0.0f) {
        std::fill(data.begin(), data.end(), 0.0f);
        return;
    }
    float inv = 1.0f / range;
    for (float& v : data) {
        v = (v - lo) * inv;
    }
}

// ---------------------------------------------------------------------------
// Lighting helpers
// ---------------------------------------------------------------------------

/// Convert azimuth / elevation (radians) to a unit direction vector.
[[nodiscard]] constexpr std::array<float, 3> spherical_to_direction(
    float azimuth, float elevation) noexcept
{
    float ce = std::cos(elevation);
    return {
        ce * std::cos(azimuth),
        ce * std::sin(azimuth),
        std::sin(elevation)
    };
}

/// Lambertian diffuse factor: max(dot(normal, light_dir), 0).
[[nodiscard]] constexpr float lambertian_factor(
    const std::array<float, 3>& normal,
    float azimuth, float elevation) noexcept
{
    auto light = spherical_to_direction(azimuth, elevation);
    float dot = normal[0] * light[0]
              + normal[1] * light[1]
              + normal[2] * light[2];
    return dot > 0.0f ? dot : 0.0f;
}

// ---------------------------------------------------------------------------
// Individual compositing functions
// ---------------------------------------------------------------------------

[[nodiscard]] constexpr float composite_mean(
    std::span<const float> layers) noexcept
{
    if (layers.empty()) return 0.0f;
    float sum = 0.0f;
    for (float v : layers) sum += v;
    return sum / static_cast<float>(layers.size());
}

[[nodiscard]] constexpr float composite_max(
    std::span<const float> layers) noexcept
{
    if (layers.empty()) return 0.0f;
    float m = layers[0];
    for (std::size_t i = 1; i < layers.size(); ++i) {
        if (layers[i] > m) m = layers[i];
    }
    return m;
}

[[nodiscard]] constexpr float composite_min(
    std::span<const float> layers) noexcept
{
    if (layers.empty()) return 0.0f;
    float m = layers[0];
    for (std::size_t i = 1; i < layers.size(); ++i) {
        if (layers[i] < m) m = layers[i];
    }
    return m;
}

/// Alpha blending (front-to-back), ported from VC3D's alpha composite.
/// Each layer value is mapped to a normalized alpha via alpha_min/max,
/// then blended front-to-back with opacity and cutoff controls.
[[nodiscard]] constexpr float composite_alpha(
    std::span<const float> layers,
    float alpha_min, float alpha_max,
    float alpha_opacity = 1.0f,
    float alpha_cutoff = 1.0f) noexcept
{
    if (layers.empty()) return 0.0f;

    float range = alpha_max - alpha_min;
    if (range == 0.0f) return 0.0f;
    float inv_range = 1.0f / range;
    float offset = alpha_min / range;

    float alpha = 0.0f;
    float value_acc = 0.0f;

    for (float density : layers) {
        float normalized = density * inv_range - offset;
        if (normalized <= 0.0f) continue;
        if (normalized > 1.0f) normalized = 1.0f;
        if (alpha >= alpha_cutoff) break;

        float opacity = normalized * alpha_opacity;
        if (opacity > 1.0f) opacity = 1.0f;
        float weight = (1.0f - alpha) * opacity;
        value_acc += weight * normalized;
        alpha += weight;
    }

    return value_acc;
}

/// Beer-Lambert volume rendering (front-to-back).
///   T(d) = exp(-extinction * density)
///   accumulated += emission * density * transmittance
///   transmittance *= T
///   result = accumulated + ambient * transmittance
[[nodiscard]] constexpr float composite_beer_lambert(
    std::span<const float> layers,
    float extinction, float emission_coeff, float ambient) noexcept
{
    if (layers.empty()) return ambient;

    float transmittance = 1.0f;
    float accumulated   = 0.0f;

    for (float density : layers) {
        float T = std::exp(-extinction * density);
        accumulated   += emission_coeff * density * transmittance;
        transmittance *= T;

        if (transmittance < 1e-6f) break;
    }
    return accumulated + ambient * transmittance;
}

// ---------------------------------------------------------------------------
// Stack compositing (dispatch by method)
// ---------------------------------------------------------------------------

[[nodiscard]] constexpr float composite_stack(
    std::span<const float> layers,
    CompositingMethod method,
    const CompositeParams& params = {}) noexcept
{
    switch (method) {
        case CompositingMethod::mean:
            return composite_mean(layers);
        case CompositingMethod::max:
            return composite_max(layers);
        case CompositingMethod::min:
            return composite_min(layers);
        case CompositingMethod::alpha:
            return composite_alpha(layers, params.alpha_min, params.alpha_max,
                                   params.alpha_opacity, params.alpha_cutoff);
        case CompositingMethod::beer_lambert:
            return composite_beer_lambert(
                layers, params.extinction, params.emission, params.ambient);
    }
    return 0.0f;
}

/// Composite with per-layer surface normals (Beer-Lambert + Lambertian
/// lighting).  Each layer's emission contribution is modulated by the
/// diffuse lighting factor computed from its normal.
[[nodiscard]] constexpr float composite_stack_lit(
    std::span<const float> layers,
    std::span<const std::array<float, 3>> normals,
    const CompositeParams& params) noexcept
{
    if (layers.empty()) return params.ambient;

    std::size_t n = layers.size();
    if (normals.size() < n) n = normals.size();

    float transmittance = 1.0f;
    float accumulated   = 0.0f;

    for (std::size_t i = 0; i < n; ++i) {
        float density = layers[i];
        float T = std::exp(-params.extinction * density);

        float lf = lambertian_factor(
            normals[i], params.light_azimuth, params.light_elevation);

        accumulated   += params.emission * density * lf * transmittance;
        transmittance *= T;

        if (transmittance < 1e-6f) break;
    }
    return accumulated + params.ambient * transmittance;
}

// ---------------------------------------------------------------------------
// Batch image compositing
// ---------------------------------------------------------------------------

/// Composite a multi-layer image into a single-layer output.
/// \p layers is a flat buffer laid out as [num_pixels][num_layers].
/// \p output must have space for \p num_pixels elements.
inline void composite_image(
    std::span<const float> layers,
    std::size_t num_pixels,
    std::size_t num_layers,
    std::span<float> output,
    const CompositeParams& params)
{
    for (std::size_t px = 0; px < num_pixels; ++px) {
        std::span<const float> stack =
            layers.subspan(px * num_layers, num_layers);
        output[px] = composite_stack(stack, params.method, params);
    }
}

} // namespace utils
