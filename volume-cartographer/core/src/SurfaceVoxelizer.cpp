#include "vc/core/util/SurfaceVoxelizer.hpp"
#include "vc/core/util/Surface.hpp"
#include <z5/factory.hxx>
#include <z5/filesystem/handle.hxx>
#include <z5/multiarray/xtensor_access.hxx>
#include <z5/attributes.hxx>
#include <nlohmann/json.hpp>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <omp.h>



void SurfaceVoxelizer::voxelizeSurfaces(
    const std::string& outputPath,
    const std::map<std::string, QuadSurface*>& surfaces,
    const VolumeInfo& volumeInfo,
    const VoxelizationParams& params,
    std::function<void(int)> progressCallback)
{
    if (surfaces.empty()) {
        throw std::runtime_error("No surfaces provided for voxelization");
    }

    // Use the provided volume dimensions
    size_t nx = volumeInfo.width;
    size_t ny = volumeInfo.height;
    size_t nz = volumeInfo.depth;

    std::cout << "Creating voxel grid with volume dimensions: "
              << nx << " x " << ny << " x " << nz << std::endl;
    std::cout << "Voxel size: " << volumeInfo.voxelSize << " mm" << std::endl;
    std::cout << "Using chunk-based processing with chunk size: " << params.chunkSize << std::endl;

    // Create zarr file structure
    z5::filesystem::handle::File zarrFile(outputPath);
    z5::createFile(zarrFile, true); // true = zarr format

    // Create base resolution dataset
    std::vector<size_t> shape = {nz, ny, nx};
    std::vector<size_t> chunks = {
        static_cast<size_t>(params.chunkSize),
        static_cast<size_t>(params.chunkSize),
        static_cast<size_t>(params.chunkSize)
    };

    // Adjust chunk size if smaller than grid dimensions
    for (size_t i = 0; i < 3; ++i) {
        chunks[i] = std::min(chunks[i], shape[i]);
    }

    auto ds0 = z5::createDataset(zarrFile, "0", "uint8", shape, chunks);

    // Calculate total number of chunks for progress tracking
    size_t totalChunks = 0;
    for (size_t cz = 0; cz < nz; cz += params.chunkSize) {
        for (size_t cy = 0; cy < ny; cy += params.chunkSize) {
            for (size_t cx = 0; cx < nx; cx += params.chunkSize) {
                totalChunks++;
            }
        }
    }

    std::atomic<int> completedChunks(0);

    // Create a list of all chunk coordinates to process
    std::vector<std::array<size_t, 3>> chunkCoords;
    for (size_t cz = 0; cz < nz; cz += params.chunkSize) {
        for (size_t cy = 0; cy < ny; cy += params.chunkSize) {
            for (size_t cx = 0; cx < nx; cx += params.chunkSize) {
                chunkCoords.push_back({cx, cy, cz});
            }
        }
    }

    // Process chunks in parallel
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < chunkCoords.size(); ++i) {
        size_t cx = chunkCoords[i][0];
        size_t cy = chunkCoords[i][1];
        size_t cz = chunkCoords[i][2];

        // Calculate actual chunk dimensions
        size_t chunk_nx = std::min(static_cast<size_t>(params.chunkSize), nx - cx);
        size_t chunk_ny = std::min(static_cast<size_t>(params.chunkSize), ny - cy);
        size_t chunk_nz = std::min(static_cast<size_t>(params.chunkSize), nz - cz);

        // Create chunk bounds for intersection testing
        Rect3D chunkBounds = {
            cv::Vec3f(static_cast<float>(cx), static_cast<float>(cy), static_cast<float>(cz)),
            cv::Vec3f(static_cast<float>(cx + chunk_nx), static_cast<float>(cy + chunk_ny), static_cast<float>(cz + chunk_nz))
        };

        // Find surfaces that intersect this chunk
        std::vector<QuadSurface*> relevantSurfaces;
        for (const auto& [name, surface] : surfaces) {
            if (!surface) continue;

            Rect3D surfaceBBox = surface->bbox();
            if (intersect(surfaceBBox, chunkBounds)) {
                relevantSurfaces.push_back(surface);
            }
        }

        // Skip empty chunks
        if (relevantSurfaces.empty()) {
            int completed = completedChunks.fetch_add(1) + 1;
            if (progressCallback) {
                int progress = (completed * 100) / totalChunks;
                progressCallback(progress);
            }
            continue;
        }

        // Allocate chunk memory
        xt::xarray<uint8_t> chunk = xt::zeros<uint8_t>({chunk_nz, chunk_ny, chunk_nx});

        // Process each relevant surface directly
        cv::Vec3i chunkOffset(cx, cy, cz);
        cv::Vec3i chunkSize(chunk_nx, chunk_ny, chunk_nz);

        for (QuadSurface* surface : relevantSurfaces) {
            voxelizeSurfaceChunk(surface, chunk, chunkOffset, chunkSize, params);
        }

        // Write chunk to zarr
        z5::types::ShapeType offset = {cz, cy, cx};
        z5::multiarray::writeSubarray<uint8_t>(ds0, chunk, offset.begin());

        // Update progress
        int completed = completedChunks.fetch_add(1) + 1;
        if (progressCallback) {
            int progress = (completed * 100) / totalChunks;
            progressCallback(progress);
        }
    }

    // Create multi-resolution pyramid
    createPyramid(zarrFile, shape);

    // Write metadata
    nlohmann::json attrs;
    attrs["surfaces"] = nlohmann::json::array();
    for (const auto& [name, _] : surfaces) {
        attrs["surfaces"].push_back(name);
    }
    attrs["voxel_size"] = volumeInfo.voxelSize;
    attrs["volume_dimensions"] = {nx, ny, nz};

    // Get current time as ISO string
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
    attrs["created"] = ss.str();

    z5::writeAttributes(zarrFile, attrs);
}

void SurfaceVoxelizer::voxelizeSurfaceChunk(
    QuadSurface* surface,
    xt::xarray<uint8_t>& chunk,
    const cv::Vec3i& chunkOffset,
    const cv::Vec3i& chunkSize,
    const VoxelizationParams& params)
{
    cv::Mat_<cv::Vec3f> points = surface->rawPoints();

    // Calculate sampling step based on density parameter
    float step = params.samplingDensity;

    // Pre-calculate chunk bounds for faster checking
    const float chunkMinX = static_cast<float>(chunkOffset[0]);
    const float chunkMaxX = static_cast<float>(chunkOffset[0] + chunkSize[0]);
    const float chunkMinY = static_cast<float>(chunkOffset[1]);
    const float chunkMaxY = static_cast<float>(chunkOffset[1] + chunkSize[1]);
    const float chunkMinZ = static_cast<float>(chunkOffset[2]);
    const float chunkMaxZ = static_cast<float>(chunkOffset[2] + chunkSize[2]);

    // Iterate through surface grid
    for (int j = 0; j < points.rows - 1; j++) {
        for (int i = 0; i < points.cols - 1; i++) {
            // Get quad corners
            cv::Vec3f p00 = points(j, i);
            cv::Vec3f p01 = points(j, i+1);
            cv::Vec3f p10 = points(j+1, i);
            cv::Vec3f p11 = points(j+1, i+1);

            // Skip invalid quads
            if (p00[0] == -1 || p01[0] == -1 || p10[0] == -1 || p11[0] == -1)
                continue;

            // Quick bounds check - skip if quad is entirely outside chunk
            float minX = std::min({p00[0], p01[0], p10[0], p11[0]});
            float maxX = std::max({p00[0], p01[0], p10[0], p11[0]});
            if (maxX < chunkMinX || minX >= chunkMaxX)
                continue;

            float minY = std::min({p00[1], p01[1], p10[1], p11[1]});
            float maxY = std::max({p00[1], p01[1], p10[1], p11[1]});
            if (maxY < chunkMinY || minY >= chunkMaxY)
                continue;

            float minZ = std::min({p00[2], p01[2], p10[2], p11[2]});
            float maxZ = std::max({p00[2], p01[2], p10[2], p11[2]});
            if (maxZ < chunkMinZ || minZ >= chunkMaxZ)
                continue;

            // Adaptive sampling based on quad size
            float quadDiag = std::max(cv::norm(p11 - p00), cv::norm(p10 - p01));
            float adaptiveStep = std::min(step, 1.0f / std::max(2.0f, quadDiag));

            // Sample within the quad more efficiently
            int numU = static_cast<int>(1.0f / adaptiveStep) + 1;
            int numV = static_cast<int>(1.0f / adaptiveStep) + 1;

            std::vector<cv::Vec3f> quadPoints;
            quadPoints.reserve(numU * numV);

            // Generate all sample points first
            for (int vi = 0; vi <= numV; vi++) {
                float v = static_cast<float>(vi) / numV;
                cv::Vec3f p0 = (1-v) * p00 + v * p10;
                cv::Vec3f p1 = (1-v) * p01 + v * p11;

                for (int ui = 0; ui <= numU; ui++) {
                    float u = static_cast<float>(ui) / numU;
                    cv::Vec3f p = (1-u) * p0 + u * p1;

                    // Only store points that might be in chunk
                    if (p[0] >= chunkMinX - 1 && p[0] < chunkMaxX + 1 &&
                        p[1] >= chunkMinY - 1 && p[1] < chunkMaxY + 1 &&
                        p[2] >= chunkMinZ - 1 && p[2] < chunkMaxZ + 1) {
                        quadPoints.push_back(p);
                    }
                }
            }

            // Voxelize all points
            for (const auto& p : quadPoints) {
                // Convert to voxel coordinates relative to chunk
                int vx = static_cast<int>(std::floor(p[0])) - chunkOffset[0];
                int vy = static_cast<int>(std::floor(p[1])) - chunkOffset[1];
                int vz = static_cast<int>(std::floor(p[2])) - chunkOffset[2];

                // Bounds check within chunk
                if (vx >= 0 && vx < chunkSize[0] &&
                    vy >= 0 && vy < chunkSize[1] &&
                    vz >= 0 && vz < chunkSize[2]) {
                    // Mark voxel (note: chunk is in ZYX order)
                    chunk(vz, vy, vx) = 255;
                }
            }

            // Fill gaps if requested
            if (params.fillGaps && quadPoints.size() > 1) {
                // Connect consecutive points in sampling order
                for (size_t pi = 1; pi < quadPoints.size(); pi++) {
                    cv::Vec3f p0 = quadPoints[pi-1] - cv::Vec3f(chunkOffset[0], chunkOffset[1], chunkOffset[2]);
                    cv::Vec3f p1 = quadPoints[pi] - cv::Vec3f(chunkOffset[0], chunkOffset[1], chunkOffset[2]);

                    // Only draw lines between nearby points
                    if (cv::norm(p1 - p0) < 2.0f) {
                        drawLine3D(chunk, p0, p1);
                    }
                }

                // Connect quad edges directly without temporary array
                cv::Vec3f c00 = p00 - cv::Vec3f(chunkOffset[0], chunkOffset[1], chunkOffset[2]);
                cv::Vec3f c01 = p01 - cv::Vec3f(chunkOffset[0], chunkOffset[1], chunkOffset[2]);
                cv::Vec3f c10 = p10 - cv::Vec3f(chunkOffset[0], chunkOffset[1], chunkOffset[2]);
                cv::Vec3f c11 = p11 - cv::Vec3f(chunkOffset[0], chunkOffset[1], chunkOffset[2]);

                connectEdge(chunk, c00, c01, params);
                connectEdge(chunk, c01, c11, params);
                connectEdge(chunk, c11, c10, params);
                connectEdge(chunk, c10, c00, params);
            }
        }
    }
}

void SurfaceVoxelizer::drawLine3D(
    xt::xarray<uint8_t>& grid,
    const cv::Vec3f& start,
    const cv::Vec3f& end)
{
    // Convert to voxel coordinates
    int x0 = std::round(start[0]);
    int y0 = std::round(start[1]);
    int z0 = std::round(start[2]);
    int x1 = std::round(end[0]);
    int y1 = std::round(end[1]);
    int z1 = std::round(end[2]);

    // 3D Bresenham implementation
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int dz = abs(z1 - z0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int sz = z0 < z1 ? 1 : -1;

    int dm = std::max({dx, dy, dz});
    if (dm == 0) return;

    int i = dm;
    int ex = dm/2;
    int ey = dm/2;
    int ez = dm/2;

    while (i-- > 0) {
        // Set voxel if in bounds (note: grid is in ZYX order)
        if (x0 >= 0 && x0 < grid.shape(2) &&
            y0 >= 0 && y0 < grid.shape(1) &&
            z0 >= 0 && z0 < grid.shape(0)) {
            grid(z0, y0, x0) = 255;
        }

        ex += dx;
        if (ex >= dm) {
            ex -= dm;
            x0 += sx;
        }
        ey += dy;
        if (ey >= dm) {
            ey -= dm;
            y0 += sy;
        }
        ez += dz;
        if (ez >= dm) {
            ez -= dm;
            z0 += sz;
        }
    }
}

void SurfaceVoxelizer::connectEdge(
    xt::xarray<uint8_t>& grid,
    const cv::Vec3f& p0,
    const cv::Vec3f& p1,
    const VoxelizationParams& params)
{
    // Sample along the edge
    float length = cv::norm(p1 - p0);
    int numSamples = std::ceil(length / params.samplingDensity);
    if (numSamples < 2) numSamples = 2;

    cv::Vec3f lastPoint = p0;
    for (int i = 0; i <= numSamples; ++i) {
        float t = static_cast<float>(i) / numSamples;
        cv::Vec3f point = (1 - t) * p0 + t * p1;

        if (i > 0) {
            drawLine3D(grid, lastPoint, point);
        }
        lastPoint = point;
    }
}

void SurfaceVoxelizer::createPyramid(
    z5::filesystem::handle::File& zarrFile,
    const std::vector<size_t>& baseShape)
{
    // Create datasets for levels 1-4
    for (int level = 1; level < 5; level++) {
        int scale = 1 << level; // 2, 4, 8, 16

        std::vector<size_t> shape = {
            (baseShape[0] + scale - 1) / scale,
            (baseShape[1] + scale - 1) / scale,
            (baseShape[2] + scale - 1) / scale
        };

        std::vector<size_t> chunks = {64, 64, 64};
        // Adjust chunk size if smaller than shape
        for (size_t i = 0; i < 3; ++i) {
            chunks[i] = std::min(chunks[i], shape[i]);
        }

        auto ds = z5::createDataset(zarrFile, std::to_string(level),
                                   "uint8", shape, chunks);

        // Downsample from previous level
        downsampleLevel(zarrFile, level);
    }
}

void SurfaceVoxelizer::downsampleLevel(
    z5::filesystem::handle::File& zarrFile,
    int targetLevel)
{
    auto srcDs = z5::openDataset(zarrFile, std::to_string(targetLevel - 1));
    auto dstDs = z5::openDataset(zarrFile, std::to_string(targetLevel));

    const auto& srcShape = srcDs->shape();
    const auto& dstShape = dstDs->shape();

    // Process in chunks
    const size_t chunkSize = 64;

    // Create list of chunks to process
    std::vector<std::array<size_t, 3>> pyramidChunkCoords;
    for (size_t dz = 0; dz < dstShape[0]; dz += chunkSize) {
        for (size_t dy = 0; dy < dstShape[1]; dy += chunkSize) {
            for (size_t dx = 0; dx < dstShape[2]; dx += chunkSize) {
                pyramidChunkCoords.push_back({dx, dy, dz});
            }
        }
    }

    // Process pyramid chunks in parallel
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < pyramidChunkCoords.size(); ++i) {
        size_t dx = pyramidChunkCoords[i][0];
        size_t dy = pyramidChunkCoords[i][1];
        size_t dz = pyramidChunkCoords[i][2];

        // Calculate chunk dimensions
        size_t chunk_nz = std::min(chunkSize, dstShape[0] - dz);
        size_t chunk_ny = std::min(chunkSize, dstShape[1] - dy);
        size_t chunk_nx = std::min(chunkSize, dstShape[2] - dx);

        // Create output chunk
        xt::xarray<uint8_t> dstChunk = xt::zeros<uint8_t>({chunk_nz, chunk_ny, chunk_nx});

        // Read corresponding source region (2x size)
        size_t src_z = dz * 2;
        size_t src_y = dy * 2;
        size_t src_x = dx * 2;
        size_t src_nz = std::min(chunk_nz * 2, srcShape[0] - src_z);
        size_t src_ny = std::min(chunk_ny * 2, srcShape[1] - src_y);
        size_t src_nx = std::min(chunk_nx * 2, srcShape[2] - src_x);

        // Create properly sized source chunk
        xt::xarray<uint8_t> srcChunk = xt::zeros<uint8_t>({src_nz, src_ny, src_nx});

        // Thread-safe read
        #pragma omp critical(zarr_read_pyramid)
        {
            z5::types::ShapeType srcOffset = {src_z, src_y, src_x};
            z5::multiarray::readSubarray<uint8_t>(srcDs, srcChunk, srcOffset.begin());
        }

        // Max-pooling: if any voxel in 2x2x2 block is set, output is set
        for (size_t z = 0; z < chunk_nz; ++z) {
            for (size_t y = 0; y < chunk_ny; ++y) {
                for (size_t x = 0; x < chunk_nx; ++x) {
                    uint8_t maxVal = 0;
                    for (int ddz = 0; ddz < 2 && z*2+ddz < src_nz; ++ddz) {
                        for (int ddy = 0; ddy < 2 && y*2+ddy < src_ny; ++ddy) {
                            for (int ddx = 0; ddx < 2 && x*2+ddx < src_nx; ++ddx) {
                                maxVal = std::max(maxVal, srcChunk(z*2+ddz, y*2+ddy, x*2+ddx));
                            }
                        }
                    }
                    dstChunk(z, y, x) = maxVal;
                }
            }
        }

        // Thread-safe write
        #pragma omp critical(zarr_write_pyramid)
        {
            z5::types::ShapeType dstOffset = {dz, dy, dx};
            z5::multiarray::writeSubarray<uint8_t>(dstDs, dstChunk, dstOffset.begin());
        }
    }
}