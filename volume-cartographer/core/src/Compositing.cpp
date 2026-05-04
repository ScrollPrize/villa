#include "vc/core/util/Compositing.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace {

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

constexpr CompositingMethod parse_compositing_method(std::string_view name) noexcept
{
    if (name == "mean")          return CompositingMethod::mean;
    if (name == "max")           return CompositingMethod::max;
    if (name == "min")           return CompositingMethod::min;
    if (name == "alpha")         return CompositingMethod::alpha;
    if (name == "beerLambert")   return CompositingMethod::beer_lambert;
    if (name == "dvr")           return CompositingMethod::dvr;
    if (name == "firstHitIso")   return CompositingMethod::first_hit_iso;
    if (name == "devFromMean")   return CompositingMethod::dev_from_mean;
    if (name == "emissionDvr")   return CompositingMethod::emission_dvr;
    if (name == "maxAboveIso")   return CompositingMethod::max_above_iso;
    if (name == "gammaWeighted") return CompositingMethod::gamma_weighted;
    if (name == "gradientMag")   return CompositingMethod::gradient_mag;
    if (name == "pbrIso")        return CompositingMethod::pbr_iso;
    if (name == "shadedDvr")     return CompositingMethod::shaded_dvr;
    return CompositingMethod::mean;
}

constexpr bool method_requires_storage(CompositingMethod m) noexcept
{
    return m != CompositingMethod::max && m != CompositingMethod::min && m != CompositingMethod::mean;
}

float composite_mean(std::span<const float> layers) noexcept
{
    if (layers.empty()) return 0.0f;
    float sum = 0.0f;
    for (float v : layers) sum += v;
    return sum / static_cast<float>(layers.size());
}

float composite_max(std::span<const float> layers) noexcept
{
    if (layers.empty()) return 0.0f;
    float m = layers[0];
    for (std::size_t i = 1; i < layers.size(); ++i) if (layers[i] > m) m = layers[i];
    return m;
}

float composite_min(std::span<const float> layers) noexcept
{
    if (layers.empty()) return 0.0f;
    float m = layers[0];
    for (std::size_t i = 1; i < layers.size(); ++i) if (layers[i] < m) m = layers[i];
    return m;
}

float composite_alpha(std::span<const float> layers,
                     float alpha_min, float alpha_max,
                     float alpha_opacity, float alpha_cutoff) noexcept
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

float composite_dvr(std::span<const float> layers, float ambient) noexcept
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

float composite_first_hit_iso(std::span<const float> layers, float iso_cutoff) noexcept
{
    for (float I : layers) if (I > iso_cutoff) return I;
    return 0.0f;
}

float composite_dev_from_mean(std::span<const float> layers, float iso_cutoff) noexcept
{
    double sum = 0.0;
    int n = 0;
    for (float I : layers) if (I > iso_cutoff) { sum += double(I); ++n; }
    if (n == 0) return 0.0f;
    const float m = float(sum / double(n));
    double dev = 0.0;
    for (float I : layers) if (I > iso_cutoff) dev += std::fabs(I - m);
    return float(dev / double(n));
}

float composite_emission_dvr(std::span<const float> layers) noexcept
{
    float sum = 0.0f;
    for (float I : layers) sum += I * I * (1.0f / 255.0f);
    return sum;
}

float composite_max_above_iso(std::span<const float> layers, float iso_cutoff) noexcept
{
    float m = 0.0f;
    for (float I : layers) if (I > iso_cutoff && I > m) m = I;
    return m;
}

float composite_gamma_weighted(std::span<const float> layers, float iso_cutoff) noexcept
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

float composite_gradient_mag(std::span<const float> layers) noexcept
{
    if (layers.size() < 3) return 0.0f;
    float best = 0.0f;
    for (std::size_t i = 1; i + 1 < layers.size(); ++i) {
        const float g = std::fabs(layers[i + 1] - layers[i - 1]) * 0.5f;
        if (g > best) best = g;
    }
    return best * 8.0f;
}

}

namespace CompositeMethod {

float mean(const LayerStack& stack) noexcept
{
    if (stack.validCount == 0) return 0.0f;
    return composite_mean(std::span<const float>(stack.values.data(), stack.validCount));
}

float max(const LayerStack& stack) noexcept
{
    if (stack.validCount == 0) return 0.0f;
    return composite_max(std::span<const float>(stack.values.data(), stack.validCount));
}

float min(const LayerStack& stack) noexcept
{
    if (stack.validCount == 0) return 255.0f;
    return composite_min(std::span<const float>(stack.values.data(), stack.validCount));
}

float alpha(const LayerStack& stack, const CompositeParams& params) noexcept
{
    if (stack.validCount == 0) return 0.0f;
    float result = composite_alpha(
        std::span<const float>(stack.values.data(), stack.validCount),
        params.alphaMin * 255.0f, params.alphaMax * 255.0f,
        params.alphaOpacity, params.alphaCutoff);
    return result * 255.0f;
}

float beerLambert(const LayerStack& stack, const CompositeParams& params) noexcept
{
    if (stack.validCount == 0) return 0.0f;

    const float extinctionScaled = params.blExtinction / 255.0f;
    const float emissionScaled = params.blEmission / 255.0f;

    float transmittance = 1.0f;
    float accumulatedColor = 0.0f;

    for (int i = 0; i < stack.validCount; i++) {
        const float value = stack.values[i];
        if (value < 0.255f) continue;

        const float emission = value * emissionScaled;
        const float layerTransmittance = std::exp(-extinctionScaled * value);

        accumulatedColor += emission * transmittance * (1.0f - layerTransmittance);
        transmittance *= layerTransmittance;

        if (transmittance < 0.001f) break;
    }

    accumulatedColor += params.blAmbient * transmittance;
    return std::min(255.0f, accumulatedColor * 255.0f);
}

}

float compositeLayerStack(
    const LayerStack& stack,
    const CompositeParams& params) noexcept
{
    if (stack.validCount == 0) return 0.0f;

    auto method = parse_compositing_method(params.method);
    std::span<const float> layers(stack.values.data(), stack.validCount);

    switch (method) {
        case CompositingMethod::mean:           return CompositeMethod::mean(stack);
        case CompositingMethod::max:            return CompositeMethod::max(stack);
        case CompositingMethod::min:            return CompositeMethod::min(stack);
        case CompositingMethod::alpha:          return CompositeMethod::alpha(stack, params);
        case CompositingMethod::beer_lambert:   return CompositeMethod::beerLambert(stack, params);
        case CompositingMethod::dvr:            return composite_dvr(layers, params.dvrAmbient);
        case CompositingMethod::first_hit_iso:  return composite_first_hit_iso(layers, float(params.isoCutoff));
        case CompositingMethod::dev_from_mean:  return composite_dev_from_mean(layers, float(params.isoCutoff));
        case CompositingMethod::emission_dvr:   return composite_emission_dvr(layers);
        case CompositingMethod::max_above_iso:  return composite_max_above_iso(layers, float(params.isoCutoff));
        case CompositingMethod::gamma_weighted: return composite_gamma_weighted(layers, float(params.isoCutoff));
        case CompositingMethod::gradient_mag:   return composite_gradient_mag(layers);
        case CompositingMethod::pbr_iso:        return composite_first_hit_iso(layers, float(params.isoCutoff));
        case CompositingMethod::shaded_dvr:     return composite_dvr(layers, params.dvrAmbient);
    }

    return CompositeMethod::mean(stack);
}

bool methodRequiresLayerStorage(const std::string& method) noexcept
{
    return method_requires_storage(parse_compositing_method(method));
}

void buildTfLut256(bool enabled,
                   uint8_t x1, uint8_t y1,
                   uint8_t x2, uint8_t y2,
                   uint8_t lut[256]) noexcept
{
    if (!enabled) {
        for (int i = 0; i < 256; ++i) lut[i] = uint8_t(i);
        return;
    }
    if (x1 > x2) { std::swap(x1, x2); std::swap(y1, y2); }
    auto lerp = [](float x, float x0, float x1, float y0, float y1) {
        const float d = x1 - x0;
        if (d <= 0.f) return y0;
        const float t = (x - x0) / d;
        return y0 + t * (y1 - y0);
    };
    for (int i = 0; i < 256; ++i) {
        float y;
        if (i <= int(x1))      y = lerp(float(i), 0.f,      float(x1), 0.f,      float(y1));
        else if (i <= int(x2)) y = lerp(float(i), float(x1), float(x2), float(y1), float(y2));
        else                   y = lerp(float(i), float(x2), 255.f,     float(y2), 255.f);
        if (y < 0.f) y = 0.f;
        if (y > 255.f) y = 255.f;
        lut[i] = uint8_t(y + 0.5f);
    }
}

float computeLightingFactor(const cv::Vec3f& normal, const CompositeParams& params) noexcept
{
    if (!params.lightingEnabled) {
        return 1.0f;
    }

    float normalLen = std::sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
    if (normalLen < 0.0001f) {
        return params.lightAmbient;
    }

    float invLen = 1.0f / normalLen;
    float nDotL = (normal[0] * invLen) * params.lightDirX
                + (normal[1] * invLen) * params.lightDirY
                + (normal[2] * invLen) * params.lightDirZ;
    if (nDotL < 0.0f) nDotL = 0.0f;

    float lighting = params.lightAmbient + params.lightDiffuse * nDotL;
    return std::min(1.0f, std::max(0.0f, lighting));
}
