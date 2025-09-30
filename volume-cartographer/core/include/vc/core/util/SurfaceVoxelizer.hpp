#pragma once

#include <string>
#include <map>
#include <opencv2/core.hpp>
#include <z5/filesystem/handle.hxx>

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xarray.hpp)
#include "vc/core/util/Surface.hpp"



/**
 * @brief Voxelizes quadmesh surfaces into a multi-resolution zarr array
 *
 * This class converts surfaces (stored as tifxyz format) into voxel grids,
 * mimicking how surfaces are rendered as intersections in CVolumeViewer.
 */
class SurfaceVoxelizer {
public:
    /**
     * @brief Parameters for voxelization process
     */
    struct VoxelizationParams {
        float voxelSize;           ///< Size of each voxel in world units
        float samplingDensity;    ///< How densely to sample surface (0.25 = 4x4 samples per cell)
        bool fillGaps;             ///< Whether to connect samples with lines
        int chunkSize;               ///< Zarr chunk size

        VoxelizationParams() :
            voxelSize(1.0f),
            samplingDensity(0.5f),
            fillGaps(true),
            chunkSize(64) {}
    };

    /**
     * @brief Volume dimensions and metadata
     */
    struct VolumeInfo {
        size_t width;     ///< Volume width (X dimension)
        size_t height;    ///< Volume height (Y dimension)
        size_t depth;     ///< Volume depth (Z dimension)
        float voxelSize;  ///< Size of each voxel in mm
    };

    /**
     * @brief Main entry point for voxelizing multiple surfaces
     * @param outputPath Path to output zarr file
     * @param surfaces Map of surface names to QuadSurface pointers
     * @param volumeInfo Information about the target volume dimensions
     * @param params Voxelization parameters
     * @param progressCallback Optional callback for progress updates (0-100)
     */
    static void voxelizeSurfaces(
        const std::string& outputPath,
        const std::map<std::string, QuadSurface*>& surfaces,
        const VolumeInfo& volumeInfo,
        const VoxelizationParams& params = VoxelizationParams(),
        std::function<void(int)> progressCallback = nullptr
    );

private:
    /**
     * @brief Voxelize a surface into a specific chunk
     */
    static void voxelizeSurfaceChunk(
        QuadSurface* surface,
        xt::xarray<uint8_t>& chunk,
        const cv::Vec3i& chunkOffset,  // Offset in voxel coordinates
        const cv::Vec3i& chunkSize,    // Size of chunk in voxels
        const VoxelizationParams& params
    );

    /**
     * @brief 3D line drawing using Bresenham's algorithm
     */
    static void drawLine3D(
        xt::xarray<uint8_t>& grid,
        const cv::Vec3f& start,
        const cv::Vec3f& end
    );

    /**
     * @brief Connect edge between two points
     */
    static void connectEdge(
        xt::xarray<uint8_t>& grid,
        const cv::Vec3f& p0,
        const cv::Vec3f& p1,
        const VoxelizationParams& params
    );

    /**
     * @brief Create multi-resolution pyramid
     */
    static void createPyramid(
        z5::filesystem::handle::File& zarrFile,
        const std::vector<size_t>& baseShape
    );

    /**
     * @brief Downsample a level for pyramid generation
     */
    static void downsampleLevel(
        z5::filesystem::handle::File& zarrFile,
        int targetLevel
    );
};

