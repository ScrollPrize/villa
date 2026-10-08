#pragma once

// The volume sampling of vc_render_tifxyz on a CUDA device (--gpu).
//
// The renderer's CPU samplers (core/src/Slicing.cpp: readMultiSlice and sampleTileSlices for layer
// stacks, readCompositeFast for collapsed bands) are transcribed to CUDA kernels operation by
// operation: the same float expressions in the same order, the same trilinear fma chain, the same
// nearest-neighbour rounding, clamping and truncation. A --gpu render therefore writes the same
// bytes as the CPU render of the same command line. Surface generation, accumulation, rotation and
// the output writers stay on the CPU, unchanged; only base + dirs * offset -> voxel moves.
//
// The CUDA driver API and the NVRTC runtime compiler are loaded at run time, so the build needs no
// CUDA toolkit. A machine without an NVIDIA driver, without NVRTC or without a device reports why
// and the caller renders on the CPU. Decoded chunks come from the renderer's own chunk cache (any
// compressor, local or remote) and are held on the device in a pool with least-recently-used
// replacement.

#include "vc/core/render/IChunkedArray.hpp"

#include <opencv2/core/mat.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct CompositeParams;

namespace vc::render::cuda {

struct Stats {
    std::uint64_t bands = 0;            // sampleSlices / composite calls
    std::uint64_t passes = 0;           // device passes (a band splits when its chunks exceed the pool)
    std::uint64_t chunksUploaded = 0;
    std::uint64_t bytesUploaded = 0;
    std::uint64_t evictions = 0;
    double fetchSeconds = 0;            // waiting on the chunk cache
    double uploadSeconds = 0;
    double deviceSeconds = 0;           // kernels, and moving geometry and samples across
};

class GpuSampler {
public:
    using Logger = std::function<void(const std::string&)>;

    // Device 0 of the devices the driver shows this process (CUDA_VISIBLE_DEVICES), the kernels
    // compiled for it, and a chunk pool of poolBytes (0: half of the free device memory) for
    // `array` at `level`. nullptr, with the reason in `why`, when there is no usable device,
    // driver, NVRTC or memory. `array` must outlive the sampler.
    static std::unique_ptr<GpuSampler> open(IChunkedArray& array, int level, std::size_t poolBytes,
                                            Logger log, std::string& why);
    ~GpuSampler();
    GpuSampler(const GpuSampler&) = delete;
    GpuSampler& operator=(const GpuSampler&) = delete;

    const std::string& device() const;
    std::size_t poolChunks() const;

    // readMultiSlice / sampleTileSlices: out[i](r, c) is the volume at base(r, c) + dirs(r, c) *
    // offsets[i], trilinear; 0 outside the volume, in a missing chunk or for a non-finite position;
    // uint8 truncated, uint16 rounded. `out` becomes offsets.size() images of base's size. Throws
    // std::runtime_error on a device error or a failed chunk fetch.
    void sampleSlices(const cv::Mat_<cv::Vec3f>& base, const cv::Mat_<cv::Vec3f>& dirs,
                      const std::vector<float>& offsets, std::vector<cv::Mat_<uint8_t>>& out);
    void sampleSlices(const cv::Mat_<cv::Vec3f>& base, const cv::Mat_<cv::Vec3f>& dirs,
                      const std::vector<float>& offsets, std::vector<cv::Mat_<uint16_t>>& out);

    // readCompositeFast with nearest-neighbour sampling for the scalar reducers (max, min, mean
    // and median, honouring isoCutoff). alpha, beerLambert and minabs stay on the CPU, which
    // compositeSupported reports. `out` must be base's size: a pixel without a finite base or
    // direction, or with no layer above the cutoff, keeps its incoming value, as on the CPU.
    static bool compositeSupported(const CompositeParams& params, int numLayers, std::string* why = nullptr);
    void composite(const cv::Mat_<cv::Vec3f>& base, const cv::Mat_<cv::Vec3f>& dirs, float zStep,
                   int zStart, int zEnd, const CompositeParams& params, cv::Mat_<uint8_t>& out);

    const Stats& stats() const;
    std::string summary() const;  // one line for the log

private:
    struct Impl;
    explicit GpuSampler(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace vc::render::cuda
