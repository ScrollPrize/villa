#include "vc/core/util/Compositing.hpp"

#include <algorithm>
#include <cmath>
#include <span>

#include <opencv2/imgproc.hpp>
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

    // Layer values are 0-255 but alphaMin/alphaMax are normalized 0-1.
    // Normalize layers to [0,1] before passing to utils, then scale back.
    std::array<float, 256> normalized;
    for (int i = 0; i < stack.validCount; i++)
        normalized[i] = stack.values[i] / 255.0f;

    float result = utils::composite_alpha(
        std::span<const float>(normalized.data(), stack.validCount),
        params.alphaMin, params.alphaMax, params.alphaOpacity, params.alphaCutoff);
    return result * 255.0f;
}

float beerLambert(const LayerStack& stack, const CompositeParams& params)
{
    if (stack.validCount == 0) return 0.0f;
    auto layers = std::span<const float>(stack.values.data(), stack.validCount);

    // Normalize values to [0,1] for utils::composite_beer_lambert
    // then scale result back to [0,255]
    float transmittance = 1.0f;
    float accumulatedColor = 0.0f;

    for (int i = 0; i < stack.validCount; i++) {
        const float value = stack.values[i];
        const float density = value / 255.0f;

        if (density < 0.001f) continue;

        const float emission = density * params.blEmission;
        const float layerTransmittance = std::exp(-params.blExtinction * density);

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

std::vector<std::string> availableCompositeMethods()
{
    return {
        "mean",
        "max",
        "min",
        "alpha",
        "beerLambert"
    };
}

float computeLightingFactor(const cv::Vec3f& normal, const CompositeParams& params)
{
    if (!params.lightingEnabled) {
        return 1.0f;
    }

    // Convert degrees to radians for utils
    const float azimuthRad = params.lightAzimuth * static_cast<float>(M_PI) / 180.0f;
    const float elevationRad = params.lightElevation * static_cast<float>(M_PI) / 180.0f;

    // Normalize the surface normal
    float normalLen = std::sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
    if (normalLen < 0.0001f) {
        return params.lightAmbient;
    }

    std::array<float, 3> n = {normal[0] / normalLen, normal[1] / normalLen, normal[2] / normalLen};

    // Use utils lambertian_factor
    float nDotL = utils::lambertian_factor(n, azimuthRad, elevationRad);

    // Combine: ambient + diffuse
    float lighting = params.lightAmbient + params.lightDiffuse * nDotL;
    return std::min(1.0f, std::max(0.0f, lighting));
}

void postprocessComposite(cv::Mat_<uint8_t>& img, const CompositeRenderSettings& settings)
{
    if (img.empty()) return;

    // Stretch values to full range
    if (settings.postStretchValues) {
        double minVal, maxVal;
        cv::minMaxLoc(img, &minVal, &maxVal);
        if (maxVal > minVal) {
            img.convertTo(img, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
        }
    }

    // Remove small connected components
    if (settings.postRemoveSmallComponents && settings.postMinComponentSize > 1) {
        cv::Mat_<uint8_t> binary;
        cv::threshold(img, binary, 0, 255, cv::THRESH_BINARY);

        cv::Mat labels, stats, centroids;
        int numComponents = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);

        cv::Mat_<uint8_t> keepMask = cv::Mat_<uint8_t>::zeros(img.size());
        for (int i = 1; i < numComponents; i++) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area >= settings.postMinComponentSize) {
                keepMask.setTo(255, labels == i);
            }
        }

        cv::Mat_<uint8_t> filtered;
        img.copyTo(filtered, keepMask);
        img = filtered;
    }
}
