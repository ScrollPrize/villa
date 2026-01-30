#include "ExtrapolationGrowth.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <QLoggingCategory>
#include <QStringList>

#include "edt.hpp"
#include "cc3d_binary.hpp"
#include "dijkstra3d.hpp"
#include <opencv2/ximgproc.hpp>
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xtensor.hpp)

Q_DECLARE_LOGGING_CATEGORY(lcSegGrowth)

namespace
{

// Compute aligned chunk origin for a world point
cv::Vec3i getChunkOrigin(const cv::Vec3f& worldPt, int chunkSize) {
    return cv::Vec3i(
        static_cast<int>(std::floor(worldPt[0] / chunkSize)) * chunkSize,
        static_cast<int>(std::floor(worldPt[1] / chunkSize)) * chunkSize,
        static_cast<int>(std::floor(worldPt[2] / chunkSize)) * chunkSize
    );
}

// Load binary chunk and compute SDT
SDTChunk* getOrComputeSDTChunk(SDTContext& ctx, const cv::Vec3f& worldPt) {
    const int cs = ctx.params.chunkSize;
    cv::Vec3i origin = getChunkOrigin(worldPt, cs);

    // Load binary data from zarr
    cv::Vec3i size(cs, cs, cs);
    xt::xtensor<uint8_t, 3, xt::layout_type::column_major> binaryData(
        std::array<size_t, 3>{(size_t)cs, (size_t)cs, (size_t)cs});
    binaryData.fill(0);

    // Clamp to dataset bounds
    auto shape = ctx.binaryDataset->shape();
    cv::Vec3i clampedOrigin(
        std::max(0, origin[0]),
        std::max(0, origin[1]),
        std::max(0, origin[2])
    );
    cv::Vec3i clampedEnd(
        std::min((int)shape[2], origin[0] + cs),
        std::min((int)shape[1], origin[1] + cs),
        std::min((int)shape[0], origin[2] + cs)
    );
    cv::Vec3i readSize = clampedEnd - clampedOrigin;

    if (readSize[0] > 0 && readSize[1] > 0 && readSize[2] > 0) {
        // Zarr volumes are [z,y,x]; translate from world XYZ to ZYX for reading.
        cv::Vec3i clampedOriginZYX(clampedOrigin[2], clampedOrigin[1], clampedOrigin[0]);
        cv::Vec3i readSizeZYX(readSize[2], readSize[1], readSize[0]);
        xt::xtensor<uint8_t, 3, xt::layout_type::column_major> readBuf(
            std::array<size_t, 3>{(size_t)readSizeZYX[0], (size_t)readSizeZYX[1], (size_t)readSizeZYX[2]});
        readArea3D(readBuf, clampedOriginZYX, ctx.binaryDataset, ctx.cache);

        // Copy into binaryData at correct offset
        cv::Vec3i offset = clampedOrigin - origin;
        for (int z = 0; z < readSize[2]; z++) {
            for (int y = 0; y < readSize[1]; y++) {
                for (int x = 0; x < readSize[0]; x++) {
                    binaryData(x + offset[0], y + offset[1], z + offset[2]) = readBuf(z, y, x);
                }
            }
        }
    }

    const size_t voxels = static_cast<size_t>(cs) * cs * cs;
    uint8_t* sdtSource = binaryData.data();
    std::vector<uint8_t> componentMask;

    // Restrict SDT to the 26-connected component containing (or nearest to) the current point.
    size_t fgCount = 0;
    int lx = std::clamp(static_cast<int>(std::round(worldPt[0])) - origin[0], 0, cs - 1);
    int ly = std::clamp(static_cast<int>(std::round(worldPt[1])) - origin[1], 0, cs - 1);
    int lz = std::clamp(static_cast<int>(std::round(worldPt[2])) - origin[2], 0, cs - 1);
    size_t nearestIdx = static_cast<size_t>(-1);
    int bestDistSq = std::numeric_limits<int>::max();

    for (int z = 0; z < cs; z++) {
        for (int y = 0; y < cs; y++) {
            for (int x = 0; x < cs; x++) {
                const size_t idx = static_cast<size_t>(x + y * cs + z * cs * cs);
                if (binaryData.data()[idx] == 0) {
                    continue;
                }
                fgCount++;
                if (nearestIdx == static_cast<size_t>(-1)) {
                    nearestIdx = idx;
                }
                const int dx = x - lx;
                const int dy = y - ly;
                const int dz = z - lz;
                const int distSq = dx * dx + dy * dy + dz * dz;
                if (distSq < bestDistSq) {
                    bestDistSq = distSq;
                    nearestIdx = idx;
                }
            }
        }
    }

    uint32_t currentLabel = 0;
    uint32_t* labels = nullptr;
    if (fgCount > 0) {
        size_t numLabels = 0;
        labels = cc3d::connected_components3d_26_binary<uint8_t, uint32_t>(
            binaryData.data(), cs, cs, cs, fgCount, nullptr, numLabels);

        if (nearestIdx != static_cast<size_t>(-1)) {
            currentLabel = labels[nearestIdx];
        }
    }

    const uint64_t key = ctx.chunkKey(origin, currentLabel);
    auto it = ctx.sdtChunks.find(key);
    if (it != ctx.sdtChunks.end()) {
        if (labels) {
            delete[] labels;
        }
        return &it->second;
    }

    if (currentLabel > 0 && labels) {
        componentMask.assign(voxels, 0);
        for (size_t i = 0; i < voxels; i++) {
            if (labels[i] == currentLabel) {
                componentMask[i] = 1;
            }
        }
        sdtSource = componentMask.data();
    }
    if (labels) {
        delete[] labels;
    }

    // Create inverted mask for EDT (distance from outside to surface)
    std::vector<uint8_t> inverted(voxels);
    for (size_t i = 0; i < voxels; i++) {
        inverted[i] = sdtSource[i] ? 0 : 1;
    }

    // Compute EDT outside (distance from outside points to surface)
    float* edtOutside = edt::binary_edt<uint8_t>(
        inverted.data(), cs, cs, cs,
        1.0f, 1.0f, 1.0f, false, 1);

    // Compute EDT inside (distance from inside points to boundary)
    float* edtInside = edt::binary_edt<uint8_t>(
        sdtSource, cs, cs, cs,
        1.0f, 1.0f, 1.0f, false, 1);

    // Combine into SDT: positive outside, negative inside
    SDTChunk chunk;
    chunk.origin = origin;
    chunk.size = size;
    chunk.data = std::make_unique<float[]>(cs * cs * cs);
    for (size_t i = 0; i < voxels; i++) {
        if (sdtSource[i]) {
            chunk.data[i] = -edtInside[i];  // Inside: negative
        } else {
            chunk.data[i] = edtOutside[i];   // Outside: positive
        }
    }

    delete[] edtOutside;
    delete[] edtInside;

    auto [insertIt, inserted] = ctx.sdtChunks.emplace(key, std::move(chunk));
    return &insertIt->second;
}

// Sample SDT at a world point (nearest neighbor)
float sampleSDT(SDTContext& ctx, const cv::Vec3f& worldPt) {
    SDTChunk* chunk = getOrComputeSDTChunk(ctx, worldPt);
    if (!chunk) return 0.0f;

    cv::Vec3f local = worldPt - cv::Vec3f(chunk->origin);
    int x = std::clamp(static_cast<int>(local[0]), 0, chunk->size[0] - 1);
    int y = std::clamp(static_cast<int>(local[1]), 0, chunk->size[1] - 1);
    int z = std::clamp(static_cast<int>(local[2]), 0, chunk->size[2] - 1);
    return chunk->data[z * chunk->size[1] * chunk->size[0] + y * chunk->size[0] + x];
}

// Newton refinement towards SDT=0 (the surface boundary)
cv::Vec3f refineToSurface(SDTContext& ctx, const cv::Vec3f& point) {
    const float h = 1.0f;  // Finite difference step
    cv::Vec3f p = point;
    float prevDist = std::abs(sampleSDT(ctx, p));

    for (int iter = 0; iter < ctx.params.maxSteps; iter++) {
        float d = sampleSDT(ctx, p);
        float absDist = std::abs(d);

        if (absDist < ctx.params.convergenceThreshold) break;

        // Gradient via central differences
        cv::Vec3f grad;
        grad[0] = (sampleSDT(ctx, p + cv::Vec3f(h,0,0)) -
                   sampleSDT(ctx, p - cv::Vec3f(h,0,0))) / (2*h);
        grad[1] = (sampleSDT(ctx, p + cv::Vec3f(0,h,0)) -
                   sampleSDT(ctx, p - cv::Vec3f(0,h,0))) / (2*h);
        grad[2] = (sampleSDT(ctx, p + cv::Vec3f(0,0,h)) -
                   sampleSDT(ctx, p - cv::Vec3f(0,0,h))) / (2*h);

        float gradMagSq = grad.dot(grad);
        if (gradMagSq < 1e-10f) break;

        cv::Vec3f pNew = p - ctx.params.stepSize * (d / gradMagSq) * grad;

        float newDist = std::abs(sampleSDT(ctx, pNew));
        if (newDist >= prevDist - ctx.params.minImprovement) break;

        p = pNew;
        prevDist = newDist;
    }

    return p;
}

// Check if a point is valid (not the invalid marker -1,-1,-1)
bool isValidPoint(const cv::Vec3f& p)
{
    return p[0] != -1.f && p[1] != -1.f && p[2] != -1.f;
}

// Linear extrapolation: fit line y = a + b*t to N points, predict at t=N
// Uses least squares: minimize sum((y_i - a - b*t_i)^2)
cv::Vec3f extrapolateLinear(const std::vector<cv::Vec3f>& points)
{
    const int n = static_cast<int>(points.size());
    if (n < 2) {
        return {-1.f, -1.f, -1.f};
    }

    // Linear regression for each component (x, y, z)
    // Using parameter t = 0, 1, 2, ... n-1, predict at t = n
    cv::Vec3d sum{0.0, 0.0, 0.0};
    double sumT = 0.0;
    double sumT2 = 0.0;
    cv::Vec3d sumTP{0.0, 0.0, 0.0};

    for (int i = 0; i < n; ++i) {
        const double t = static_cast<double>(i);
        sum += cv::Vec3d(points[i]);
        sumT += t;
        sumT2 += t * t;
        sumTP += t * cv::Vec3d(points[i]);
    }

    const double det = n * sumT2 - sumT * sumT;
    if (std::abs(det) < 1e-10) {
        // Degenerate case - return last point
        return points.back();
    }

    // y = a + b*t, solve for a, b
    cv::Vec3d b = (n * sumTP - sumT * sum) / det;
    cv::Vec3d a = (sum - b * sumT) / n;

    // Extrapolate to t = n
    cv::Vec3d result = a + b * static_cast<double>(n);
    return cv::Vec3f(result);
}

// Quadratic extrapolation: fit y = a + b*t + c*t^2 to N points, predict at t=N
// Uses least squares normal equations
cv::Vec3f extrapolateQuadratic(const std::vector<cv::Vec3f>& points)
{
    const int n = static_cast<int>(points.size());
    if (n < 3) {
        // Fall back to linear if not enough points for quadratic
        return extrapolateLinear(points);
    }

    cv::Vec3f result;

    // Fit each dimension independently
    for (int dim = 0; dim < 3; ++dim) {
        // Build normal equations for quadratic fit
        // [s0 s1 s2][a]   [y0]
        // [s1 s2 s3][b] = [y1]
        // [s2 s3 s4][c]   [y2]
        double s0 = n, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
        double y0 = 0, y1 = 0, y2 = 0;

        for (int i = 0; i < n; ++i) {
            const double t = static_cast<double>(i);
            const double t2 = t * t;
            s1 += t;
            s2 += t2;
            s3 += t2 * t;
            s4 += t2 * t2;

            const double y = static_cast<double>(points[i][dim]);
            y0 += y;
            y1 += y * t;
            y2 += y * t2;
        }

        // Solve 3x3 system using Cramer's rule
        const double det = s0 * (s2 * s4 - s3 * s3)
                         - s1 * (s1 * s4 - s2 * s3)
                         + s2 * (s1 * s3 - s2 * s2);

        if (std::abs(det) < 1e-10) {
            // Fall back to linear for this dimension
            result[dim] = extrapolateLinear(points)[dim];
            continue;
        }

        const double a = (y0 * (s2 * s4 - s3 * s3)
                        - s1 * (y1 * s4 - s3 * y2)
                        + s2 * (y1 * s3 - s2 * y2)) / det;
        const double b = (s0 * (y1 * s4 - s3 * y2)
                        - y0 * (s1 * s4 - s2 * s3)
                        + s2 * (s1 * y2 - y1 * s2)) / det;
        const double c = (s0 * (s2 * y2 - s3 * y1)
                        - s1 * (s1 * y2 - s2 * y1)
                        + y0 * (s1 * s3 - s2 * s2)) / det;

        // Extrapolate to t = n
        const double t = static_cast<double>(n);
        result[dim] = static_cast<float>(a + b * t + c * t * t);
    }

    return result;
}

// Get aligned chunk origin for a world point (used for skeleton path too)
cv::Vec3i getSkeletonChunkOrigin(const cv::Vec3f& worldPt, int chunkSize) {
    return cv::Vec3i(
        static_cast<int>(std::floor(worldPt[0] / chunkSize)) * chunkSize,
        static_cast<int>(std::floor(worldPt[1] / chunkSize)) * chunkSize,
        static_cast<int>(std::floor(worldPt[2] / chunkSize)) * chunkSize
    );
}

// Load or get cached binary chunk
uint8_t* getOrLoadBinaryChunk(SkeletonPathContext& ctx, const cv::Vec3i& origin, const cv::Vec3i& size) {
    int64_t key = ctx.chunkKey(origin);
    auto it = ctx.binaryChunks.find(key);
    if (it != ctx.binaryChunks.end()) {
        return it->second.get();
    }

    // Allocate and load chunk
    const size_t voxels = static_cast<size_t>(size[0]) * size[1] * size[2];
    auto chunk = std::make_unique<uint8_t[]>(voxels);
    std::memset(chunk.get(), 0, voxels);

    // Clamp to dataset bounds
    auto shape = ctx.binaryDataset->shape();
    cv::Vec3i clampedOrigin(
        std::max(0, origin[0]),
        std::max(0, origin[1]),
        std::max(0, origin[2])
    );
    cv::Vec3i clampedEnd(
        std::min(static_cast<int>(shape[2]), origin[0] + size[0]),
        std::min(static_cast<int>(shape[1]), origin[1] + size[1]),
        std::min(static_cast<int>(shape[0]), origin[2] + size[2])
    );
    cv::Vec3i readSize = clampedEnd - clampedOrigin;

    if (readSize[0] > 0 && readSize[1] > 0 && readSize[2] > 0) {
        // Zarr volumes are [z,y,x]; translate from world XYZ to ZYX for reading.
        cv::Vec3i clampedOriginZYX(clampedOrigin[2], clampedOrigin[1], clampedOrigin[0]);
        cv::Vec3i readSizeZYX(readSize[2], readSize[1], readSize[0]);
        xt::xtensor<uint8_t, 3, xt::layout_type::column_major> readBuf(
            std::array<size_t, 3>{static_cast<size_t>(readSizeZYX[0]),
                                  static_cast<size_t>(readSizeZYX[1]),
                                  static_cast<size_t>(readSizeZYX[2])});
        readArea3D(readBuf, clampedOriginZYX, ctx.binaryDataset, ctx.cache);

        // Copy into chunk at correct offset
        cv::Vec3i offset = clampedOrigin - origin;
        for (int z = 0; z < readSize[2]; z++) {
            for (int y = 0; y < readSize[1]; y++) {
                for (int x = 0; x < readSize[0]; x++) {
                    size_t dstIdx = static_cast<size_t>(x + offset[0]) +
                                   static_cast<size_t>(y + offset[1]) * size[0] +
                                   static_cast<size_t>(z + offset[2]) * size[0] * size[1];
                    chunk[dstIdx] = readBuf(z, y, x);
                }
            }
        }
    }

    uint8_t* ptr = chunk.get();
    ctx.binaryChunks.emplace(key, std::move(chunk));
    return ptr;
}

// Extract 2D slice from binary volume
// For Z slice (Left/Right growth): extract XY plane at given Z
// For X slice (Up/Down growth with X orientation): extract YZ plane at given X
// For Y slice (Up/Down growth with Y orientation): extract XZ plane at given Y
cv::Mat extract2DSlice(
    SkeletonPathContext& ctx,
    const cv::Vec3f& centerPt,
    SegmentationGrowthDirection direction)
{
    const int cs = ctx.params.chunkSize;

    // Determine slice axis and index based on growth direction
    int sliceAxis;  // 0=X, 1=Y, 2=Z
    int sliceIndex;

    if (direction == SegmentationGrowthDirection::Left ||
        direction == SegmentationGrowthDirection::Right) {
        // Z slice (XY plane)
        sliceAxis = 2;
        sliceIndex = static_cast<int>(std::round(centerPt[2]));
    } else {
        // Up/Down growth - use configured orientation
        if (ctx.params.sliceOrientation == SkeletonSliceOrientation::X) {
            // X slice (YZ plane)
            sliceAxis = 0;
            sliceIndex = static_cast<int>(std::round(centerPt[0]));
        } else {
            // Y slice (XZ plane)
            sliceAxis = 1;
            sliceIndex = static_cast<int>(std::round(centerPt[1]));
        }
    }

    // Compute chunk origin that contains this slice
    cv::Vec3i chunkOrigin;
    cv::Vec3i chunkSize(cs, cs, cs);

    if (sliceAxis == 2) {
        // Z slice - center chunk on point for maximum context
        chunkOrigin = cv::Vec3i(
            static_cast<int>(std::round(centerPt[0])) - cs/2,
            static_cast<int>(std::round(centerPt[1])) - cs/2,
            sliceIndex
        );
        chunkSize[2] = 1;
    } else if (sliceAxis == 0) {
        // X slice - center chunk on point
        chunkOrigin = cv::Vec3i(
            sliceIndex,
            static_cast<int>(std::round(centerPt[1])) - cs/2,
            static_cast<int>(std::round(centerPt[2])) - cs/2
        );
        chunkSize[0] = 1;
    } else {
        // Y slice - center chunk on point
        chunkOrigin = cv::Vec3i(
            static_cast<int>(std::round(centerPt[0])) - cs/2,
            sliceIndex,
            static_cast<int>(std::round(centerPt[2])) - cs/2
        );
        chunkSize[1] = 1;
    }

    // Load the binary data
    uint8_t* binaryData = getOrLoadBinaryChunk(ctx, chunkOrigin, chunkSize);

    // Debug: count foreground pixels
    int fgCount = 0;
    size_t totalVoxels = static_cast<size_t>(chunkSize[0]) * chunkSize[1] * chunkSize[2];
    for (size_t i = 0; i < totalVoxels; i++) {
        if (binaryData[i]) fgCount++;
    }
    qCDebug(lcSegGrowth) << "extract2DSlice: origin" << chunkOrigin[0] << chunkOrigin[1] << chunkOrigin[2]
                         << "size" << chunkSize[0] << chunkSize[1] << chunkSize[2]
                         << "fgPixels" << fgCount << "/" << totalVoxels;

    // Create 2D cv::Mat from the slice
    cv::Mat slice;
    if (sliceAxis == 2) {
        // Z slice -> XY plane
        slice = cv::Mat(chunkSize[1], chunkSize[0], CV_8UC1);
        for (int y = 0; y < chunkSize[1]; y++) {
            for (int x = 0; x < chunkSize[0]; x++) {
                slice.at<uint8_t>(y, x) = binaryData[x + y * chunkSize[0]];
            }
        }
    } else if (sliceAxis == 0) {
        // X slice -> YZ plane (rows=Z, cols=Y)
        slice = cv::Mat(chunkSize[2], chunkSize[1], CV_8UC1);
        for (int z = 0; z < chunkSize[2]; z++) {
            for (int y = 0; y < chunkSize[1]; y++) {
                slice.at<uint8_t>(z, y) = binaryData[y * chunkSize[0] + z * chunkSize[0] * chunkSize[1]];
            }
        }
    } else {
        // Y slice -> XZ plane (rows=Z, cols=X)
        slice = cv::Mat(chunkSize[2], chunkSize[0], CV_8UC1);
        for (int z = 0; z < chunkSize[2]; z++) {
            for (int x = 0; x < chunkSize[0]; x++) {
                slice.at<uint8_t>(z, x) = binaryData[x + z * chunkSize[0] * chunkSize[1]];
            }
        }
    }

    return slice;
}

// Find skeleton endpoints (pixels with exactly 1 neighbor)
std::vector<cv::Point> findSkeletonEndpoints(const cv::Mat& skeleton)
{
    std::vector<cv::Point> endpoints;

    for (int y = 0; y < skeleton.rows; y++) {
        for (int x = 0; x < skeleton.cols; x++) {
            if (skeleton.at<uint8_t>(y, x) == 0) continue;

            // Count 8-connected neighbors
            int neighbors = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx >= 0 && nx < skeleton.cols && ny >= 0 && ny < skeleton.rows) {
                        if (skeleton.at<uint8_t>(ny, nx) > 0) {
                            neighbors++;
                        }
                    }
                }
            }

            if (neighbors == 1) {
                endpoints.push_back(cv::Point(x, y));
            }
        }
    }

    return endpoints;
}

// Skeleton path extrapolation for a single point
cv::Vec3f extrapolateSkeletonPath(
    SkeletonPathContext& ctx,
    const cv::Vec3f& currentPt,
    const std::vector<cv::Vec3f>& historyPts,
    SegmentationGrowthDirection direction,
    float stepSize)
{
    // Extract 2D slice at the current point
    cv::Mat slice = extract2DSlice(ctx, currentPt, direction);
    if (slice.empty()) {
        qCDebug(lcSegGrowth) << "Skeleton path: empty slice at" << currentPt[0] << currentPt[1] << currentPt[2];
        return {-1.f, -1.f, -1.f};
    }

    // Binarize if needed (ensure it's 0/255)
    cv::Mat binarySlice;
    if (slice.type() == CV_8UC1) {
        cv::threshold(slice, binarySlice, 0, 255, cv::THRESH_BINARY);
    } else {
        slice.convertTo(binarySlice, CV_8UC1);
        cv::threshold(binarySlice, binarySlice, 0, 255, cv::THRESH_BINARY);
    }

    // Run 2D connected components (8-connected)
    size_t numLabels = 0;
    uint32_t* labels = cc3d::connected_components2d_8_binary<uint8_t, uint32_t>(
        binarySlice.data, binarySlice.cols, binarySlice.rows,
        static_cast<size_t>(binarySlice.cols) * binarySlice.rows, nullptr, numLabels);

    if (numLabels == 0) {
        qCDebug(lcSegGrowth) << "Skeleton path: no connected components at" << currentPt[0] << currentPt[1] << currentPt[2]
                             << "slice size" << binarySlice.cols << "x" << binarySlice.rows;
        delete[] labels;
        return {-1.f, -1.f, -1.f};
    }

    // Convert current point to slice coordinates
    const int cs = ctx.params.chunkSize;
    int sliceX, sliceY;

    if (direction == SegmentationGrowthDirection::Left ||
        direction == SegmentationGrowthDirection::Right) {
        // Z slice (XY plane) - chunk is centered on point
        int chunkOriginX = static_cast<int>(std::round(currentPt[0])) - cs/2;
        int chunkOriginY = static_cast<int>(std::round(currentPt[1])) - cs/2;
        sliceX = static_cast<int>(std::round(currentPt[0])) - chunkOriginX;
        sliceY = static_cast<int>(std::round(currentPt[1])) - chunkOriginY;
    } else if (ctx.params.sliceOrientation == SkeletonSliceOrientation::X) {
        // X slice (YZ plane) - cols=Y, rows=Z - chunk is centered on point
        int chunkOriginY = static_cast<int>(std::round(currentPt[1])) - cs/2;
        int chunkOriginZ = static_cast<int>(std::round(currentPt[2])) - cs/2;
        sliceX = static_cast<int>(std::round(currentPt[1])) - chunkOriginY;
        sliceY = static_cast<int>(std::round(currentPt[2])) - chunkOriginZ;
    } else {
        // Y slice (XZ plane) - cols=X, rows=Z - chunk is centered on point
        int chunkOriginX = static_cast<int>(std::round(currentPt[0])) - cs/2;
        int chunkOriginZ = static_cast<int>(std::round(currentPt[2])) - cs/2;
        sliceX = static_cast<int>(std::round(currentPt[0])) - chunkOriginX;
        sliceY = static_cast<int>(std::round(currentPt[2])) - chunkOriginZ;
    }

    // Clamp to slice bounds
    sliceX = std::clamp(sliceX, 0, binarySlice.cols - 1);
    sliceY = std::clamp(sliceY, 0, binarySlice.rows - 1);

    // Get the label at current point
    uint32_t currentLabel = labels[sliceY * binarySlice.cols + sliceX];
    if (currentLabel == 0) {
        // Point is on background - try to find nearest foreground
        int searchRadius = ctx.params.searchRadius;
        bool found = false;
        for (int r = 1; r <= searchRadius && !found; r++) {
            for (int dy = -r; dy <= r && !found; dy++) {
                for (int dx = -r; dx <= r && !found; dx++) {
                    int nx = sliceX + dx;
                    int ny = sliceY + dy;
                    if (nx >= 0 && nx < binarySlice.cols && ny >= 0 && ny < binarySlice.rows) {
                        uint32_t label = labels[ny * binarySlice.cols + nx];
                        if (label > 0) {
                            currentLabel = label;
                            sliceX = nx;
                            sliceY = ny;
                            found = true;
                        }
                    }
                }
            }
        }
        if (!found) {
            qCDebug(lcSegGrowth) << "Skeleton path: point on background, no foreground within radius" << searchRadius
                                 << "at" << currentPt[0] << currentPt[1] << currentPt[2]
                                 << "sliceXY" << sliceX << sliceY;
            delete[] labels;
            return {-1.f, -1.f, -1.f};
        }
    }

    // Extract only the component containing current point
    cv::Mat component = cv::Mat::zeros(binarySlice.size(), CV_8UC1);
    for (int y = 0; y < binarySlice.rows; y++) {
        for (int x = 0; x < binarySlice.cols; x++) {
            if (labels[y * binarySlice.cols + x] == currentLabel) {
                component.at<uint8_t>(y, x) = 255;
            }
        }
    }
    delete[] labels;

    // Run morphological thinning (skeletonization)
    cv::Mat skeleton;
    cv::ximgproc::thinning(component, skeleton, cv::ximgproc::THINNING_ZHANGSUEN);

    // Find skeleton endpoints
    std::vector<cv::Point> endpoints = findSkeletonEndpoints(skeleton);

    if (endpoints.empty()) {
        // No endpoints - skeleton might be a loop or too small
        qCDebug(lcSegGrowth) << "Skeleton path: no skeleton endpoints at" << currentPt[0] << currentPt[1] << currentPt[2]
                             << "- skeleton may be a loop or too small";
        return {-1.f, -1.f, -1.f};
    }

    // Select endpoint furthest in growth direction
    cv::Point targetEndpoint = endpoints[0];

    auto growthScoreFor = [&](const cv::Point& ep) -> float {
        // Score based on growth direction
        // Left/Right: score based on X in slice (which maps to X in world for Z slice)
        // Up/Down: score based on appropriate axis
        if (direction == SegmentationGrowthDirection::Left) {
            return -static_cast<float>(ep.x - sliceX);  // More negative X is better
        }
        if (direction == SegmentationGrowthDirection::Right) {
            return static_cast<float>(ep.x - sliceX);   // More positive X is better
        }
        if (direction == SegmentationGrowthDirection::Up) {
            if (ctx.params.sliceOrientation == SkeletonSliceOrientation::X) {
                // X slice: cols=Y, rows=Z, Up means -Y
                return -static_cast<float>(ep.x - sliceX);
            }
            // Y slice: cols=X, rows=Z, Up typically means -Y but we're in XZ plane
            // For Up, we want points with lower row index (lower Z typically, but depends on orientation)
            return -static_cast<float>(ep.y - sliceY);
        }
        if (direction == SegmentationGrowthDirection::Down) {
            if (ctx.params.sliceOrientation == SkeletonSliceOrientation::X) {
                return static_cast<float>(ep.x - sliceX);
            }
            return static_cast<float>(ep.y - sliceY);
        }
        return 0.0f;
    };

    // Prefer endpoint aligned with linear-fit extrapolation to avoid backtracking.
    bool useTravelDir = false;
    cv::Vec2f travelDir2d(0.0f, 0.0f);
    if (historyPts.size() >= 2) {
        const cv::Vec3f predictedPt = extrapolateLinear(historyPts);
        if (std::isfinite(predictedPt[0]) && std::isfinite(predictedPt[1]) && std::isfinite(predictedPt[2])) {
            const cv::Vec3f travel3d = predictedPt - currentPt;
            if (direction == SegmentationGrowthDirection::Left ||
                direction == SegmentationGrowthDirection::Right) {
                travelDir2d = cv::Vec2f(travel3d[0], travel3d[1]);
            } else if (ctx.params.sliceOrientation == SkeletonSliceOrientation::X) {
                travelDir2d = cv::Vec2f(travel3d[1], travel3d[2]);
            } else {
                travelDir2d = cv::Vec2f(travel3d[0], travel3d[2]);
            }
            const float travelLen = std::sqrt(travelDir2d[0] * travelDir2d[0] +
                                              travelDir2d[1] * travelDir2d[1]);
            if (travelLen > 1e-3f) {
                travelDir2d[0] /= travelLen;
                travelDir2d[1] /= travelLen;
                useTravelDir = true;
            }
        }
    }

    if (useTravelDir) {
        float bestTravelScore = -std::numeric_limits<float>::max();
        for (const auto& ep : endpoints) {
            const cv::Vec2f delta(static_cast<float>(ep.x - sliceX),
                                  static_cast<float>(ep.y - sliceY));
            const float travelScore = delta[0] * travelDir2d[0] + delta[1] * travelDir2d[1];
            if (travelScore > bestTravelScore) {
                bestTravelScore = travelScore;
                targetEndpoint = ep;
            }
        }

        if (bestTravelScore <= 0.0f) {
            float bestScore = -std::numeric_limits<float>::max();
            for (const auto& ep : endpoints) {
                const float score = growthScoreFor(ep);
                if (score > bestScore) {
                    bestScore = score;
                    targetEndpoint = ep;
                }
            }
        }
    } else {
        float bestScore = -std::numeric_limits<float>::max();
        for (const auto& ep : endpoints) {
            const float score = growthScoreFor(ep);
            if (score > bestScore) {
                bestScore = score;
                targetEndpoint = ep;
            }
        }
    }

    // Convert target endpoint back to 3D world coordinates
    cv::Vec3f target3D;
    if (direction == SegmentationGrowthDirection::Left ||
        direction == SegmentationGrowthDirection::Right) {
        int chunkOriginX = static_cast<int>(std::round(currentPt[0])) - cs/2;
        int chunkOriginY = static_cast<int>(std::round(currentPt[1])) - cs/2;
        target3D[0] = static_cast<float>(targetEndpoint.x + chunkOriginX);
        target3D[1] = static_cast<float>(targetEndpoint.y + chunkOriginY);
        target3D[2] = currentPt[2];  // Same Z slice
    } else if (ctx.params.sliceOrientation == SkeletonSliceOrientation::X) {
        int chunkOriginY = static_cast<int>(std::round(currentPt[1])) - cs/2;
        int chunkOriginZ = static_cast<int>(std::round(currentPt[2])) - cs/2;
        target3D[0] = currentPt[0];  // Same X slice
        target3D[1] = static_cast<float>(targetEndpoint.x + chunkOriginY);
        target3D[2] = static_cast<float>(targetEndpoint.y + chunkOriginZ);
    } else {
        int chunkOriginX = static_cast<int>(std::round(currentPt[0])) - cs/2;
        int chunkOriginZ = static_cast<int>(std::round(currentPt[2])) - cs/2;
        target3D[0] = static_cast<float>(targetEndpoint.x + chunkOriginX);
        target3D[1] = currentPt[1];  // Same Y slice
        target3D[2] = static_cast<float>(targetEndpoint.y + chunkOriginZ);
    }

    // If target is very close to current, just return it directly
    float dist = cv::norm(target3D - currentPt);
    if (dist < stepSize * 0.5f) {
        return target3D;
    }

    // Load 3D binary chunk for Dijkstra pathfinding
    // Compute bounding box containing both points with margin
    int margin = 10;
    cv::Vec3i minPt(
        static_cast<int>(std::min(currentPt[0], target3D[0])) - margin,
        static_cast<int>(std::min(currentPt[1], target3D[1])) - margin,
        static_cast<int>(std::min(currentPt[2], target3D[2])) - margin
    );
    cv::Vec3i maxPt(
        static_cast<int>(std::max(currentPt[0], target3D[0])) + margin,
        static_cast<int>(std::max(currentPt[1], target3D[1])) + margin,
        static_cast<int>(std::max(currentPt[2], target3D[2])) + margin
    );

    cv::Vec3i boxSize = maxPt - minPt;
    boxSize[0] = std::max(boxSize[0], 1);
    boxSize[1] = std::max(boxSize[1], 1);
    boxSize[2] = std::max(boxSize[2], 1);

    // Load 3D binary chunk
    uint8_t* binaryChunk = getOrLoadBinaryChunk(ctx, minPt, boxSize);

    // Convert world coords to local chunk indices
    cv::Vec3i localSource(
        static_cast<int>(std::round(currentPt[0])) - minPt[0],
        static_cast<int>(std::round(currentPt[1])) - minPt[1],
        static_cast<int>(std::round(currentPt[2])) - minPt[2]
    );
    cv::Vec3i localTarget(
        static_cast<int>(std::round(target3D[0])) - minPt[0],
        static_cast<int>(std::round(target3D[1])) - minPt[1],
        static_cast<int>(std::round(target3D[2])) - minPt[2]
    );

    // Clamp to valid range
    localSource[0] = std::clamp(localSource[0], 0, boxSize[0] - 1);
    localSource[1] = std::clamp(localSource[1], 0, boxSize[1] - 1);
    localSource[2] = std::clamp(localSource[2], 0, boxSize[2] - 1);
    localTarget[0] = std::clamp(localTarget[0], 0, boxSize[0] - 1);
    localTarget[1] = std::clamp(localTarget[1], 0, boxSize[1] - 1);
    localTarget[2] = std::clamp(localTarget[2], 0, boxSize[2] - 1);

    size_t sx = static_cast<size_t>(boxSize[0]);
    size_t sy = static_cast<size_t>(boxSize[1]);
    size_t sz = static_cast<size_t>(boxSize[2]);

    size_t sourceIdx = localSource[0] + localSource[1] * sx + localSource[2] * sx * sy;
    size_t targetIdx = localTarget[0] + localTarget[1] * sx + localTarget[2] * sx * sy;

    // Run 3D Dijkstra on binary volume
    int connectivity = static_cast<int>(ctx.params.connectivity);
    qCDebug(lcSegGrowth) << "Skeleton path: running Dijkstra from" << currentPt[0] << currentPt[1] << currentPt[2]
                         << "to target" << target3D[0] << target3D[1] << target3D[2]
                         << "boxSize" << boxSize[0] << boxSize[1] << boxSize[2];
    std::vector<uint32_t> path = dijkstra::binary_dijkstra3d<uint32_t>(
        binaryChunk, sx, sy, sz,
        sourceIdx, targetIdx,
        connectivity,
        1.0f, 1.0f, 1.0f, true);

    if (path.empty()) {
        // Dijkstra failed - fall back to simple interpolation toward target
        qCDebug(lcSegGrowth) << "Skeleton path: Dijkstra failed from" << currentPt[0] << currentPt[1] << currentPt[2]
                             << "to" << target3D[0] << target3D[1] << target3D[2];
        cv::Vec3f dir = target3D - currentPt;
        float len = cv::norm(dir);
        if (len > 0.001f) {
            dir /= len;
            return currentPt + dir * std::min(stepSize, len);
        }
        return {-1.f, -1.f, -1.f};
    }

    // Find point along path at approximately stepSize distance from source
    float accumDist = 0.0f;
    cv::Vec3f prevWorldPt = currentPt;

    // Path is returned target->source, so we iterate from the end (source side)
    for (int i = static_cast<int>(path.size()) - 2; i >= 0; i--) {
        uint32_t idx = path[i];
        int lx = idx % boxSize[0];
        int ly = (idx / boxSize[0]) % boxSize[1];
        int lz = idx / (boxSize[0] * boxSize[1]);

        cv::Vec3f worldPt(
            static_cast<float>(lx + minPt[0]),
            static_cast<float>(ly + minPt[1]),
            static_cast<float>(lz + minPt[2])
        );

        float segDist = cv::norm(worldPt - prevWorldPt);
        accumDist += segDist;

        if (accumDist >= stepSize) {
            // Interpolate to get exactly stepSize distance
            float overshoot = accumDist - stepSize;
            float t = (segDist > 0.001f) ? (segDist - overshoot) / segDist : 1.0f;
            return prevWorldPt + t * (worldPt - prevWorldPt);
        }

        prevWorldPt = worldPt;
    }

    // Path was shorter than stepSize - return the target
    return target3D;
}

// Collect N valid points from boundary going inward
// Direction specifies which edge we're growing from, so we collect points
// going the opposite direction (into the existing surface)
std::vector<cv::Vec3f> collectBoundaryPoints(
    const cv::Mat_<cv::Vec3f>& points,
    int row,
    int col,
    SegmentationGrowthDirection direction,
    int count)
{
    std::vector<cv::Vec3f> result;
    result.reserve(count);

    // Determine step direction (opposite of growth direction)
    int dr = 0, dc = 0;
    switch (direction) {
    case SegmentationGrowthDirection::Up:
        dr = 1;  // Collect from below (going down into existing surface)
        break;
    case SegmentationGrowthDirection::Down:
        dr = -1; // Collect from above (going up into existing surface)
        break;
    case SegmentationGrowthDirection::Left:
        dc = 1;  // Collect from right (going right into existing surface)
        break;
    case SegmentationGrowthDirection::Right:
        dc = -1; // Collect from left (going left into existing surface)
        break;
    default:
        return result;
    }

    int r = row;
    int c = col;
    for (int i = 0; i < count; ++i) {
        if (r < 0 || r >= points.rows || c < 0 || c >= points.cols) {
            break;
        }
        if (isValidPoint(points(r, c))) {
            result.push_back(points(r, c));
        }
        r += dr;
        c += dc;
    }

    // Reverse so points are in order from oldest to newest (for extrapolation)
    std::reverse(result.begin(), result.end());
    return result;
}

// Stats for fallback tracking
struct GrowthStats {
    int totalFallbacks = 0;
    std::vector<std::pair<int, int>> perStepFallbacks; // (step, count)
};

// Process growth in a single direction
GrowthStats growInDirection(
    cv::Mat_<cv::Vec3f>& newPoints,
    const cv::Mat_<cv::Vec3f>& oldPoints,
    cv::Mat_<uint16_t>& newGenerations,
    const cv::Mat_<uint16_t>& oldGenerations,
    SegmentationGrowthDirection direction,
    int steps,
    int pointCount,
    ExtrapolationType extrapolationType,
    int rowOffset,
    int colOffset,
    uint16_t baseGeneration,
    SDTContext* sdtContext,
    SkeletonPathContext* skeletonContext,
    float avgSpacing)
{
    const int oldRows = oldPoints.rows;
    const int oldCols = oldPoints.cols;

    // Track fallback statistics per step
    struct StepStats {
        int skeletonSuccess = 0;
        int linearFallback = 0;
    };
    std::vector<StepStats> stepStats(steps);
    int currentStep = 0;

    auto extrapolate = [extrapolationType, sdtContext, skeletonContext, direction, avgSpacing, &stepStats, &currentStep](const std::vector<cv::Vec3f>& pts) {
        cv::Vec3f pt;
        if (extrapolationType == ExtrapolationType::SkeletonPath && skeletonContext && skeletonContext->binaryDataset) {
            // Use skeleton path extrapolation
            pt = extrapolateSkeletonPath(*skeletonContext, pts.back(), pts, direction, avgSpacing);
            // Fall back to linear if skeleton path fails
            if (!isValidPoint(pt)) {
                const auto& failPt = pts.back();
                qCDebug(lcSegGrowth) << "Skeleton path failed at" << failPt[0] << failPt[1] << failPt[2] << "- falling back to linear";
                pt = extrapolateLinear(pts);
                stepStats[currentStep].linearFallback++;
            } else {
                stepStats[currentStep].skeletonSuccess++;
            }
        } else if (extrapolationType == ExtrapolationType::Quadratic) {
            pt = extrapolateQuadratic(pts);
        } else {
            pt = extrapolateLinear(pts);
        }
        // Apply Newton refinement towards SDT=0 if context is available
        if (sdtContext && sdtContext->binaryDataset) {
            pt = refineToSurface(*sdtContext, pt);
        }
        return pt;
    };

    for (int step = 0; step < steps; ++step) {
        currentStep = step;
        // Clear caches at each step to ensure fresh values
        if (sdtContext) {
            sdtContext->clearChunkCache();
        }
        if (skeletonContext) {
            skeletonContext->clearChunkCache();
        }

        const uint16_t newGen = baseGeneration + static_cast<uint16_t>(step + 1);

        switch (direction) {
        case SegmentationGrowthDirection::Right: {
            // Growing right: for each row, extrapolate from existing columns
            const int targetCol = oldCols + colOffset + step;
            for (int r = 0; r < oldRows; ++r) {
                const int targetRow = r + rowOffset;
                auto pts = collectBoundaryPoints(newPoints, targetRow, targetCol - 1,
                                                 direction, pointCount);
                if (pts.size() >= 2) {
                    newPoints(targetRow, targetCol) = extrapolate(pts);
                    newGenerations(targetRow, targetCol) = newGen;
                }
            }
            break;
        }
        case SegmentationGrowthDirection::Left: {
            // Growing left: extrapolate from right side of each row
            const int targetCol = colOffset - step - 1;
            for (int r = 0; r < oldRows; ++r) {
                const int targetRow = r + rowOffset;
                auto pts = collectBoundaryPoints(newPoints, targetRow, targetCol + 1,
                                                 direction, pointCount);
                if (pts.size() >= 2) {
                    newPoints(targetRow, targetCol) = extrapolate(pts);
                    newGenerations(targetRow, targetCol) = newGen;
                }
            }
            break;
        }
        case SegmentationGrowthDirection::Down: {
            // Growing down: for each column, extrapolate from existing rows
            const int targetRow = oldRows + rowOffset + step;
            for (int c = 0; c < oldCols; ++c) {
                const int targetCol = c + colOffset;
                auto pts = collectBoundaryPoints(newPoints, targetRow - 1, targetCol,
                                                 direction, pointCount);
                if (pts.size() >= 2) {
                    newPoints(targetRow, targetCol) = extrapolate(pts);
                    newGenerations(targetRow, targetCol) = newGen;
                }
            }
            break;
        }
        case SegmentationGrowthDirection::Up: {
            // Growing up: extrapolate from bottom of each column
            const int targetRow = rowOffset - step - 1;
            qCInfo(lcSegGrowth) << "Up step" << step << "targetRow=" << targetRow
                                << "rowOffset=" << rowOffset << "oldCols=" << oldCols;
            int validCount = 0;
            for (int c = 0; c < oldCols; ++c) {
                const int targetCol = c + colOffset;
                auto pts = collectBoundaryPoints(newPoints, targetRow + 1, targetCol,
                                                 direction, pointCount);
                if (pts.size() >= 2) {
                    cv::Vec3f newPt = extrapolate(pts);
                    newPoints(targetRow, targetCol) = newPt;
                    newGenerations(targetRow, targetCol) = newGen;
                    validCount++;
                    // Log first column's extrapolation for debugging
                    if (c == 0) {
                        cv::Vec3f lastPt = pts.back();
                        float dist = cv::norm(newPt - lastPt);
                        qCInfo(lcSegGrowth) << "Up extrapolation col0: lastPt="
                                            << lastPt[0] << "," << lastPt[1] << "," << lastPt[2]
                                            << "newPt=" << newPt[0] << "," << newPt[1] << "," << newPt[2]
                                            << "dist=" << dist << "numPts=" << pts.size();
                    }
                } else if (c == 0) {
                    qCInfo(lcSegGrowth) << "Up col0: insufficient points, got" << pts.size();
                }
            }
            qCInfo(lcSegGrowth) << "Up step" << step << "valid extrapolations:" << validCount << "/" << oldCols;
            break;
        }
        default:
            break;
        }
    }

    // Build return stats
    GrowthStats stats;
    for (int s = 0; s < steps; ++s) {
        if (stepStats[s].linearFallback > 0) {
            stats.totalFallbacks += stepStats[s].linearFallback;
            stats.perStepFallbacks.emplace_back(s + 1, stepStats[s].linearFallback);
        }
    }
    return stats;
}

} // namespace

TracerGrowthResult runExtrapolationGrowth(
    QuadSurface* surface,
    SegmentationGrowthDirection direction,
    int steps,
    int pointCount,
    ExtrapolationType extrapolationType,
    SDTContext* sdtContext,
    SkeletonPathContext* skeletonContext)
{
    TracerGrowthResult result;

    if (!surface) {
        result.error = QStringLiteral("No surface provided for extrapolation");
        return result;
    }

    cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        result.error = QStringLiteral("Surface has no valid points");
        return result;
    }

    const int oldRows = points->rows;
    const int oldCols = points->cols;

    // Clamp point count to reasonable range
    pointCount = std::clamp(pointCount, 2, 20);

    // Get existing generations channel
    cv::Mat genMat = surface->channel("generations");
    cv::Mat_<uint16_t> oldGenerations;
    if (!genMat.empty() && genMat.type() == CV_16UC1) {
        oldGenerations = genMat;
    } else {
        // Create default generations (all 1)
        oldGenerations = cv::Mat_<uint16_t>(oldRows, oldCols, static_cast<uint16_t>(1));
    }

    // Find max generation
    double minGen = 0.0, maxGen = 0.0;
    cv::minMaxLoc(oldGenerations, &minGen, &maxGen);
    const auto baseGeneration = static_cast<uint16_t>(std::round(maxGen));

    // Determine new dimensions based on direction
    int newRows = oldRows;
    int newCols = oldCols;
    int rowOffset = 0;
    int colOffset = 0;

    switch (direction) {
    case SegmentationGrowthDirection::Up:
        newRows += steps;
        rowOffset = steps;  // Old data shifts down
        break;
    case SegmentationGrowthDirection::Down:
        newRows += steps;
        break;
    case SegmentationGrowthDirection::Left:
        newCols += steps;
        colOffset = steps;  // Old data shifts right
        break;
    case SegmentationGrowthDirection::Right:
        newCols += steps;
        break;
    case SegmentationGrowthDirection::All:
        // Grow in all four directions
        newRows += steps * 2;
        newCols += steps * 2;
        rowOffset = steps;
        colOffset = steps;
        break;
    }

    // Create new expanded points matrix (initialized to invalid)
    cv::Mat_<cv::Vec3f> newPoints(newRows, newCols, cv::Vec3f(-1.f, -1.f, -1.f));

    // Create new generations matrix
    cv::Mat_<uint16_t> newGenerations(newRows, newCols, static_cast<uint16_t>(0));

    // Copy existing points and generations to new location
    for (int r = 0; r < oldRows; ++r) {
        for (int c = 0; c < oldCols; ++c) {
            newPoints(r + rowOffset, c + colOffset) = (*points)(r, c);
            newGenerations(r + rowOffset, c + colOffset) = oldGenerations(r, c);
        }
    }

    cv::Vec2f surfScale = surface->scale();

    // Calculate average spacing between adjacent points to understand grid resolution
    double avgSpacing = 0.0;
    int spacingCount = 0;
    for (int r = 0; r < std::min(10, oldRows); ++r) {
        for (int c = 0; c < std::min(10, oldCols - 1); ++c) {
            const cv::Vec3f& p0 = (*points)(r, c);
            const cv::Vec3f& p1 = (*points)(r, c + 1);
            if (isValidPoint(p0) && isValidPoint(p1)) {
                avgSpacing += cv::norm(p1 - p0);
                spacingCount++;
            }
        }
    }
    if (spacingCount > 0) {
        avgSpacing /= spacingCount;
    }

    qCInfo(lcSegGrowth) << "Extrapolation growth:"
                        << "direction=" << segmentationGrowthDirectionToString(direction)
                        << "steps=" << steps
                        << "pointCount=" << pointCount
                        << "type=" << extrapolationTypeToString(extrapolationType)
                        << "oldSize=" << oldCols << "x" << oldRows
                        << "newSize=" << newCols << "x" << newRows
                        << "scale=" << surfScale[0] << "," << surfScale[1]
                        << "avgPointSpacing=" << avgSpacing;

    // Perform extrapolation growth
    const float stepSize = static_cast<float>(avgSpacing > 0.0 ? avgSpacing : 1.0);
    int totalFallbacks = 0;
    std::vector<std::pair<int, int>> allPerStepFallbacks;

    auto aggregateStats = [&totalFallbacks, &allPerStepFallbacks](const GrowthStats& stats) {
        totalFallbacks += stats.totalFallbacks;
        for (const auto& [step, count] : stats.perStepFallbacks) {
            // Merge with existing step if present
            bool found = false;
            for (auto& [existingStep, existingCount] : allPerStepFallbacks) {
                if (existingStep == step) {
                    existingCount += count;
                    found = true;
                    break;
                }
            }
            if (!found) {
                allPerStepFallbacks.emplace_back(step, count);
            }
        }
    };

    if (direction == SegmentationGrowthDirection::All) {
        // For "All", process each direction sequentially
        // Order: down, right, up, left (so newly generated points can be used)
        aggregateStats(growInDirection(newPoints, *points, newGenerations, oldGenerations,
                        SegmentationGrowthDirection::Down, steps, pointCount,
                        extrapolationType, rowOffset, colOffset, baseGeneration,
                        sdtContext, skeletonContext, stepSize));
        aggregateStats(growInDirection(newPoints, *points, newGenerations, oldGenerations,
                        SegmentationGrowthDirection::Right, steps, pointCount,
                        extrapolationType, rowOffset, colOffset, baseGeneration,
                        sdtContext, skeletonContext, stepSize));
        aggregateStats(growInDirection(newPoints, *points, newGenerations, oldGenerations,
                        SegmentationGrowthDirection::Up, steps, pointCount,
                        extrapolationType, rowOffset, colOffset, baseGeneration,
                        sdtContext, skeletonContext, stepSize));
        aggregateStats(growInDirection(newPoints, *points, newGenerations, oldGenerations,
                        SegmentationGrowthDirection::Left, steps, pointCount,
                        extrapolationType, rowOffset, colOffset, baseGeneration,
                        sdtContext, skeletonContext, stepSize));
    } else {
        aggregateStats(growInDirection(newPoints, *points, newGenerations, oldGenerations,
                        direction, steps, pointCount, extrapolationType,
                        rowOffset, colOffset, baseGeneration, sdtContext,
                        skeletonContext, stepSize));
    }

    // Create new surface
    auto* newSurface = new QuadSurface(newPoints, surface->scale());

    // Set generations channel
    newSurface->setChannel("generations", newGenerations);

    // Preserve approval mask if it exists
    cv::Mat oldApproval = surface->channel("approval", SURF_CHANNEL_NORESIZE);
    if (!oldApproval.empty()) {
        cv::Mat newApproval;
        if (oldApproval.channels() == 3) {
            newApproval = cv::Mat(newRows, newCols, CV_8UC3, cv::Scalar(0, 0, 0));
        } else {
            newApproval = cv::Mat(newRows, newCols, CV_8UC1, cv::Scalar(0));
        }

        // Copy old approval values to same grid positions (with offset)
        int copyRows = std::min(oldApproval.rows, oldRows);
        int copyCols = std::min(oldApproval.cols, oldCols);
        if (copyRows > 0 && copyCols > 0) {
            cv::Rect srcRoi(0, 0, copyCols, copyRows);
            cv::Rect dstRoi(colOffset, rowOffset, copyCols, copyRows);
            oldApproval(srcRoi).copyTo(newApproval(dstRoi));
        }
        newSurface->setChannel("approval", newApproval);
    }

    // Copy metadata
    if (surface->meta && surface->meta->is_object()) {
        newSurface->meta = std::make_unique<nlohmann::json>(*surface->meta);
        // Update max_gen
        (*newSurface->meta)["max_gen"] = static_cast<int>(baseGeneration + steps);
    }

    result.surface = newSurface;

    // Build status message with fallback info
    if (totalFallbacks > 0 && extrapolationType == ExtrapolationType::SkeletonPath) {
        // Sort per-step fallbacks by step number
        std::sort(allPerStepFallbacks.begin(), allPerStepFallbacks.end());

        QStringList stepWarnings;
        for (const auto& [step, count] : allPerStepFallbacks) {
            stepWarnings.append(QStringLiteral("step %1: %2 pts").arg(step).arg(count));
        }
        result.statusMessage = QStringLiteral(
            "Extrapolation growth completed (%1 steps). Warning: %2 points fell back to linear (%3)")
            .arg(steps)
            .arg(totalFallbacks)
            .arg(stepWarnings.join(QStringLiteral(", ")));
    } else {
        result.statusMessage = QStringLiteral("Extrapolation growth completed (%1 steps)").arg(steps);
    }

    return result;
}
