#pragma once
#include <span>
#include <vector>
#include <cstddef>
#include <cmath>
#include <algorithm>
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
    beer_lambert,
    dvr,
    first_hit_iso,
    dev_from_mean,
    emission_dvr,
    max_above_iso,
    gamma_weighted,
    gradient_mag,
    pbr_iso,
    shaded_dvr
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
    if (name == "dvr")          return CompositingMethod::dvr;
    if (name == "firstHitIso")  return CompositingMethod::first_hit_iso;
    if (name == "devFromMean")  return CompositingMethod::dev_from_mean;
    if (name == "emissionDvr")  return CompositingMethod::emission_dvr;
    if (name == "maxAboveIso")  return CompositingMethod::max_above_iso;
    if (name == "gammaWeighted") return CompositingMethod::gamma_weighted;
    if (name == "gradientMag")  return CompositingMethod::gradient_mag;
    if (name == "pbrIso")       return CompositingMethod::pbr_iso;
    if (name == "shadedDvr")    return CompositingMethod::shaded_dvr;
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

/// Front-to-back emissive DVR. Opacity = density/255; emission = density.
/// Mirrors the per-layer integration used in VC3D's interactive viewer.
[[nodiscard]] inline float composite_dvr(
    std::span<const float> layers, float ambient = 0.0f) noexcept
{
    float color = 0.0f;
    float trans = 1.0f;
    for (float I : layers) {
        const float op = I * (1.0f / 255.0f);
        color += I * trans * op;
        trans *= (1.0f - op);
        if (trans < 0.001f) break;
    }
    return color + ambient * trans;
}

/// First-hit iso: return the first layer value strictly greater than
/// iso_cutoff. Returns 0 when no layer exceeds the threshold. Does not
/// apply any shading — the caller is expected to multiply by an external
/// lighting factor.
[[nodiscard]] inline float composite_first_hit_iso(
    std::span<const float> layers, float iso_cutoff) noexcept
{
    for (float I : layers) {
        if (I > iso_cutoff) return I;
    }
    return 0.0f;
}

/// Mean absolute deviation from the ray mean, computed over layers strictly
/// greater than iso_cutoff. Measures how much the ray deviates from its own
/// local average — useful for surfacing ink / void outliers against a papyrus
/// baseline.
[[nodiscard]] inline float composite_dev_from_mean(
    std::span<const float> layers, float iso_cutoff) noexcept
{
    double sum = 0.0;
    int n = 0;
    for (float I : layers) {
        if (I > iso_cutoff) { sum += double(I); ++n; }
    }
    if (n == 0) return 0.0f;
    const float m = float(sum / double(n));
    double dev = 0.0;
    for (float I : layers) {
        if (I > iso_cutoff) dev += std::fabs(I - m);
    }
    return float(dev / double(n));
}

/// Emission-only DVR: sum(I * I / 255) — no absorption, so layers behind
/// others still contribute. Complement to `dvr`.
[[nodiscard]] inline float composite_emission_dvr(
    std::span<const float> layers) noexcept
{
    float sum = 0.0f;
    for (float I : layers) sum += I * I * (1.0f / 255.0f);
    return sum;
}

/// Max of samples strictly greater than iso_cutoff. Like `max` but ignores
/// air/substrate — useful when iso separates the density band of interest
/// from background noise.
[[nodiscard]] inline float composite_max_above_iso(
    std::span<const float> layers, float iso_cutoff) noexcept
{
    float m = 0.0f;
    for (float I : layers) {
        if (I > iso_cutoff && I > m) m = I;
    }
    return m;
}

/// sum(w*I) / sum(w) with w = max(0, I - iso)². Quadratic weight amplifies
/// the density offset of ink relative to papyrus while staying a mean
/// (robust to single-voxel outliers that max/first-hit pick up).
[[nodiscard]] inline float composite_gamma_weighted(
    std::span<const float> layers, float iso_cutoff) noexcept
{
    float sumWI = 0.0f, sumW = 0.0f;
    for (float I : layers) {
        const float d = I - iso_cutoff;
        if (d <= 0.0f) continue;
        const float w = d * d;
        sumWI += w * I;
        sumW  += w;
    }
    return sumW > 0.0f ? sumWI / sumW : 0.0f;
}

/// Peak |∂I/∂z| via central difference, ×8 gain. Lights up sharp intensity
/// steps along the ray (cracks, ink/papyrus boundaries).
[[nodiscard]] inline float composite_gradient_mag(
    std::span<const float> layers) noexcept
{
    if (layers.size() < 3) return 0.0f;
    float best = 0.0f;
    for (std::size_t i = 1; i + 1 < layers.size(); ++i) {
        const float g = std::fabs(layers[i + 1] - layers[i - 1]) * 0.5f;
        if (g > best) best = g;
    }
    return best * 8.0f;
}

// ---------------------------------------------------------------------------
// Stack compositing (dispatch by method)
// ---------------------------------------------------------------------------

[[nodiscard]] inline float composite_stack(
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
        case CompositingMethod::dvr:
            return composite_dvr(layers, params.ambient);
        case CompositingMethod::first_hit_iso:
            return composite_first_hit_iso(layers, float(params.iso_cutoff));
        case CompositingMethod::dev_from_mean:
            return composite_dev_from_mean(layers, float(params.iso_cutoff));
        case CompositingMethod::emission_dvr:
            return composite_emission_dvr(layers);
        case CompositingMethod::max_above_iso:
            return composite_max_above_iso(layers, float(params.iso_cutoff));
        case CompositingMethod::gamma_weighted:
            return composite_gamma_weighted(layers, float(params.iso_cutoff));
        case CompositingMethod::gradient_mag:
            return composite_gradient_mag(layers);
        case CompositingMethod::pbr_iso:
            // The batch renderer lacks view/light/gradient context needed
            // for Cook-Torrance; fall back to first-hit intensity so the
            // CLI still produces a sensible output for this method.
            return composite_first_hit_iso(layers, float(params.iso_cutoff));
        case CompositingMethod::shaded_dvr:
            // Same: no per-sample gradient context here — fall back to plain
            // absorptive DVR so the CLI produces a usable image.
            return composite_dvr(layers, params.ambient);
    }
    return 0.0f;
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
