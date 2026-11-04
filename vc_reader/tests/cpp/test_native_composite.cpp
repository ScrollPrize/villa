// Composite reductions: known stacks reduce as documented (cutoff and NaN layers
// dropped, NaN without remaining layers, alpha and Beer-Lambert front to back),
// medians refuse more than 256 layers, and with CUDA the device reduction equals
// the CPU one (exactly, except for Beer-Lambert's expf in its last bits).
#include "vc_reader/composite.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef VC_READER_CUDA
#include <cuda_runtime.h>
#endif

namespace composite = vcr::composite;
using M = composite::Method;

namespace {
void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

float reduceOne(const std::vector<float>& stack, composite::Options options)
{
    float out = 0;
    composite::reduce(stack.data(), stack.size(), 1, options, &out);
    return out;
}

void knownStacks()
{
    composite::Options o;
    o.isoCutoff = 10;
    const std::vector<float> stack{5, 50, NAN, 20, 200, 90};
    const auto with = [&](M m) {
        o.method = m;
        return reduceOne(stack, o);
    };
    require(with(M::Mean) == 90 && with(M::Max) == 200 && with(M::Min) == 20 && with(M::Median) == 90 &&
                with(M::MinAbs) == 90,
            "known stack reductions (the layer below the cutoff and the NaN layer are dropped)");
    o.isoCutoff = 1000;
    require(std::isnan(with(M::Mean)) && std::isnan(with(M::Alpha)), "no remaining layer is NaN");

    // Alpha, front to back: each layer adds (1 - alpha) * opacity * normalized value.
    composite::Options a;
    a.method = M::Alpha;
    a.alphaMin = 0.2f;
    a.alphaMax = 0.8f;
    a.alphaOpacity = 0.5f;
    double acc = 0, alpha = 0;
    const std::vector<float> layers{40, 100, 255, 180};
    for (const float v : layers) {
        const double t = std::min(1.0, (v - 0.2 * 255) / (0.6 * 255));
        if (t <= 0)
            continue;
        const double w = (1 - alpha) * t * 0.5;
        acc += w * t;
        alpha += w;
    }
    require(std::abs(reduceOne(layers, a) - acc * 255) < 1e-3, "alpha compositing");
    a.alphaCutoff = 0.3f;  // stops once the accumulated alpha reaches the cutoff
    require(reduceOne(layers, a) < acc * 255 - 1, "alpha cutoff stops the integration");

    // Beer-Lambert: emission attenuated by the transmittance of the layers in front.
    composite::Options b;
    b.method = M::BeerLambert;
    double light = 0, transmittance = 1;
    for (const float v : layers) {
        const double layer = std::exp(-(1.5 / 255) * v);
        light += v * (1.5 / 255) * transmittance * (1 - layer);
        transmittance *= layer;
    }
    require(std::abs(reduceOne(layers, b) - std::min(255.0, (light + 0.1 * transmittance) * 255)) < 1e-2,
            "Beer-Lambert compositing");

    // Pixels are independent: values[k * n + p].
    const std::vector<float> two{1, 10, 3, 30, 5, 50};  // pixel 0: 1 3 5, pixel 1: 10 30 50
    composite::Options mean;
    float out[2];
    composite::reduce(two.data(), 3, 2, mean, out);
    require(out[0] == 3 && out[1] == 30, "pixel-major layout");

    composite::Options median;
    median.method = M::Median;
    const std::vector<float> many(257, 1.0f);
    bool threw = false;
    try {
        reduceOne(many, median);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "median of more than 256 layers is refused");
}

#ifdef VC_READER_CUDA
void cudaMatches()
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "no CUDA device: CUDA comparison skipped\n";
        return;
    }
    const std::size_t layers = 33, n = 5000;
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> d(0.0f, 255.0f);
    std::vector<float> values(layers * n);
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = i % 17 == 0 ? NAN : d(rng);
    float* deviceValues = nullptr;
    float* deviceOut = nullptr;
    require(cudaMalloc(&deviceValues, values.size() * sizeof(float)) == cudaSuccess &&
                cudaMalloc(&deviceOut, n * sizeof(float)) == cudaSuccess,
            "CUDA allocation");
    cudaMemcpy(deviceValues, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice);
    for (const auto method : {M::Mean, M::Max, M::Min, M::Median, M::MinAbs, M::Alpha, M::BeerLambert}) {
        composite::Options o;
        o.method = method;
        o.isoCutoff = 30;
        o.alphaMin = 0.1f;
        o.alphaMax = 0.9f;
        o.alphaOpacity = 0.4f;
        std::vector<float> cpu(n), gpu(n);
        composite::reduce(values.data(), layers, n, o, cpu.data());
        composite::cuda::reduce(deviceValues, layers, n, o, deviceOut);
        cudaMemcpy(gpu.data(), deviceOut, n * sizeof(float), cudaMemcpyDeviceToHost);
        // Beer-Lambert's expf may differ in its last bits between the host and CUDA.
        const float tolerance = method == M::BeerLambert ? 1e-5f : 0.0f;
        for (std::size_t p = 0; p < n; ++p)
            require(std::abs(cpu[p] - gpu[p]) <= tolerance * std::abs(cpu[p]) ||
                        (std::isnan(cpu[p]) && std::isnan(gpu[p])),
                    "CUDA composite " + std::to_string(int(method)) + " pixel " + std::to_string(p));
    }
    cudaFree(deviceValues);
    cudaFree(deviceOut);
    std::cout << "CUDA matches CPU\n";
}
#endif
}  // namespace

int main()
{
    try {
        knownStacks();
#ifdef VC_READER_CUDA
        cudaMatches();
#endif
        std::cout << "composite reductions passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
