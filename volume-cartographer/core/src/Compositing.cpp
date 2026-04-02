#include "vc/core/util/Compositing.hpp"

#include <algorithm>
#include <cmath>
#include <span>

#include <utils/compositing.hpp>

namespace CompositeMethod {

float mean(const LayerStack& stack)
{
    if (stack.validCount == 0) return 0.0f;
    return utils::composite_mean(
        std::span<const float>(stack.values.data(), stack.validCount));
}

float max(const LayerStack& stack)
{
    if (stack.validCount == 0) return 0.0f;
    return utils::composite_max(
        std::span<const float>(stack.values.data(), stack.validCount));
}

float min(const LayerStack& stack)
{
    if (stack.validCount == 0) return 255.0f;
    return utils::composite_min(
        std::span<const float>(stack.values.data(), stack.validCount));
}

float alpha(const LayerStack& stack, const CompositeParams& params)
{
    if (stack.validCount == 0) return 0.0f;

    // Scale thresholds to [0,255] range to avoid per-layer normalization.
    // composite_alpha computes (density - alpha_min) / (alpha_max - alpha_min),
    // so scaling both min/max by 255 cancels the layer's [0,255] range.
    // alpha_cutoff is compared against accumulated alpha (already [0,1]), not
    // layer values, so it must NOT be scaled.
    float result = utils::composite_alpha(
        std::span<const float>(stack.values.data(), stack.validCount),
        params.alphaMin * 255.0f, params.alphaMax * 255.0f,
        params.alphaOpacity, params.alphaCutoff);
    return result * 255.0f;
}

float beerLambert(const LayerStack& stack, const CompositeParams& params)
{
    if (stack.validCount == 0) return 0.0f;

    // Pre-scale extinction into [0,255] domain so we avoid per-layer /255.
    const float extinctionScaled = params.blExtinction / 255.0f;
    const float emissionScaled = params.blEmission / 255.0f;

    float transmittance = 1.0f;
    float accumulatedColor = 0.0f;

    for (int i = 0; i < stack.validCount; i++) {
        const float value = stack.values[i];

        if (value < 0.255f) continue;  // ~0.001 * 255

        const float emission = value * emissionScaled;
        const float layerTransmittance = std::exp(-extinctionScaled * value);

        accumulatedColor += emission * transmittance * (1.0f - layerTransmittance);
        transmittance *= layerTransmittance;

        if (transmittance < 0.001f) break;
    }

    accumulatedColor += params.blAmbient * transmittance;
    return std::min(255.0f, accumulatedColor * 255.0f);
}

} // namespace CompositeMethod

float compositeLayerStack(
    const LayerStack& stack,
    const CompositeParams& params)
{
    if (stack.validCount == 0) return 0.0f;

    // Use utils enum-based dispatch for simple methods
    auto method = utils::parse_compositing_method(params.method);

    switch (method) {
        case utils::CompositingMethod::mean:
            return CompositeMethod::mean(stack);
        case utils::CompositingMethod::max:
            return CompositeMethod::max(stack);
        case utils::CompositingMethod::min:
            return CompositeMethod::min(stack);
        case utils::CompositingMethod::alpha:
            return CompositeMethod::alpha(stack, params);
        case utils::CompositingMethod::beer_lambert:
            return CompositeMethod::beerLambert(stack, params);
    }

    return CompositeMethod::mean(stack);
}

bool methodRequiresLayerStorage(const std::string& method)
{
    return utils::method_requires_storage(utils::parse_compositing_method(method));
}

float computeLightingFactor(const cv::Vec3f& normal, const CompositeParams& params)
{
    if (!params.lightingEnabled) {
        return 1.0f;
    }

    // Normalize the surface normal
    float normalLen = std::sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
    if (normalLen < 0.0001f) {
        return params.lightAmbient;
    }

    float invLen = 1.0f / normalLen;
    float nDotL = (normal[0] * invLen) * params.lightDirX
                + (normal[1] * invLen) * params.lightDirY
                + (normal[2] * invLen) * params.lightDirZ;
    if (nDotL < 0.0f) nDotL = 0.0f;

    // Combine: ambient + diffuse
    float lighting = params.lightAmbient + params.lightDiffuse * nDotL;
    return std::min(1.0f, std::max(0.0f, lighting));
}

