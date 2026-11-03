#pragma once
#include <span>
#include <cstddef>
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

} // namespace utils
