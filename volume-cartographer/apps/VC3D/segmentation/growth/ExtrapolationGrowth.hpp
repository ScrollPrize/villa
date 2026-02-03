#pragma once

#include "SegmentationGrowth.hpp"

#include <opencv2/core.hpp>
#include <vc/core/util/ChunkCache.hpp>
#include <z5/dataset.hxx>
#include <cstdint>
#include <memory>
#include <unordered_map>

class QuadSurface;

// Dijkstra connectivity for 3D pathfinding
enum class DijkstraConnectivity {
    Conn6 = 6,    // Face neighbors only
    Conn18 = 18,  // Face + edge neighbors
    Conn26 = 26,  // Face + edge + corner neighbors
};

// Slice orientation for up/down growth (which plane to extract)
enum class SkeletonSliceOrientation {
    X = 0,  // YZ plane (slice perpendicular to X axis)
    Y = 1,  // XZ plane (slice perpendicular to Y axis)
};

// Local SDT chunk cache entry
struct SDTChunk {
    std::unique_ptr<float[]> data;
    cv::Vec3i origin;  // World coordinate origin of this chunk
    cv::Vec3i size;    // Dimensions of this chunk
};

// Parameters for Newton refinement
struct NewtonRefinementParams {
    int maxSteps = 5;              // Max iterations (1-10)
    float stepSize = 0.8f;         // Step size α (0.1-2.0)
    float convergenceThreshold = 0.5f;  // Stop if |d| < this (voxels)
    float minImprovement = 0.1f;   // Stop if improvement < this (voxels)
    int chunkSize = 128;           // Size of local SDT chunks to compute
};

// Context for SDT-guided refinement
struct SDTContext {
    z5::Dataset* binaryDataset = nullptr;
    ChunkCache<uint8_t>* cache = nullptr;
    NewtonRefinementParams params;

    // Local SDT chunk cache (keyed by chunk origin)
    std::unordered_map<uint64_t, SDTChunk> sdtChunks;

    uint64_t chunkKey(const cv::Vec3i& origin, uint32_t componentLabel) const {
        uint64_t key = 0xcbf29ce484222325ULL;
        auto hashCombine = [&](uint64_t v) {
            key ^= v + 0x9e3779b97f4a7c15ULL + (key << 6) + (key >> 2);
        };
        hashCombine(static_cast<uint64_t>(origin[0]));
        hashCombine(static_cast<uint64_t>(origin[1]));
        hashCombine(static_cast<uint64_t>(origin[2]));
        hashCombine(static_cast<uint64_t>(componentLabel));
        return key;
    }

    // Clear cached SDT chunks to force recomputation
    void clearChunkCache() { sdtChunks.clear(); }
};

// Parameters for skeleton path extrapolation
struct SkeletonPathParams {
    DijkstraConnectivity connectivity{DijkstraConnectivity::Conn26};
    SkeletonSliceOrientation sliceOrientation{SkeletonSliceOrientation::X};
    int chunkSize{128};  // Size of local chunks to load
    int searchRadius{5}; // Radius to search for nearest component when on background (pixels)
};

// Context for skeleton path extrapolation
struct SkeletonPathContext {
    z5::Dataset* binaryDataset = nullptr;
    ChunkCache<uint8_t>* cache = nullptr;
    SkeletonPathParams params;

    // Cache for loaded binary chunks (keyed by chunk origin)
    std::unordered_map<int64_t, std::unique_ptr<uint8_t[]>> binaryChunks;

    int64_t chunkKey(const cv::Vec3i& origin) const {
        return (int64_t(origin[0]) << 40) | (int64_t(origin[1]) << 20) | int64_t(origin[2]);
    }

    void clearChunkCache() { binaryChunks.clear(); }
};

TracerGrowthResult runExtrapolationGrowth(
    QuadSurface* surface,
    SegmentationGrowthDirection direction,
    int steps,
    int pointCount,
    ExtrapolationType extrapolationType,
    SDTContext* sdtContext = nullptr,  // Optional SDT refinement
    SkeletonPathContext* skeletonContext = nullptr);  // Optional skeleton path
