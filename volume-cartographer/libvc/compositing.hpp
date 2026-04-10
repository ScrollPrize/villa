#pragma once
#include <cmath>
#include <cstdint>
#include <span>

namespace vc {

enum class Composite : uint8_t { mean, max, min, alpha, beer_lambert };

struct CompositeParams {
    float alpha_min = 0.0f, alpha_max = 1.0f;
    float alpha_opacity = 1.0f, alpha_cutoff = 1.0f;
    float extinction = 1.5f, emission = 1.5f, ambient = 0.1f;
    uint8_t iso_cutoff = 0;
};

constexpr float composite_mean(std::span<const float> layers) noexcept {
    float sum = 0;
    for (float v : layers) sum += v;
    return sum / float(layers.size());
}

constexpr float composite_max(std::span<const float> layers) noexcept {
    float m = layers[0];
    for (size_t i = 1; i < layers.size(); ++i)
        if (layers[i] > m) m = layers[i];
    return m;
}

constexpr float composite_min(std::span<const float> layers) noexcept {
    float m = layers[0];
    for (size_t i = 1; i < layers.size(); ++i)
        if (layers[i] < m) m = layers[i];
    return m;
}

constexpr float composite_alpha(std::span<const float> layers,
                                const CompositeParams& p) noexcept {
    float range = p.alpha_max - p.alpha_min;
    if (range == 0) return 0;
    float inv = 1.0f / range, off = p.alpha_min * inv;
    float alpha = 0, acc = 0;
    for (float d : layers) {
        float n = d * inv - off;
        if (n <= 0) continue;
        if (n > 1) n = 1;
        if (alpha >= p.alpha_cutoff) break;
        float o = n * p.alpha_opacity;
        if (o > 1) o = 1;
        float w = (1 - alpha) * o;
        acc += w * n;
        alpha += w;
    }
    return acc;
}

constexpr float composite_beer_lambert(std::span<const float> layers,
                                       const CompositeParams& p) noexcept {
    float T = 1, acc = 0;
    for (float d : layers) {
        float t = std::exp(-p.extinction * d);
        acc += p.emission * d * T;
        T *= t;
        if (T < 1e-6f) break;
    }
    return acc + p.ambient * T;
}

constexpr float composite(std::span<const float> layers, Composite method,
                          const CompositeParams& p = {}) noexcept {
    switch (method) {
    case Composite::mean:         return composite_mean(layers);
    case Composite::max:          return composite_max(layers);
    case Composite::min:          return composite_min(layers);
    case Composite::alpha:        return composite_alpha(layers, p);
    case Composite::beer_lambert: return composite_beer_lambert(layers, p);
    }
    std::unreachable();
}

} // namespace vc
