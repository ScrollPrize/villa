#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <z5/dataset.hxx>
#include <z5/filesystem/handle.hxx>

// Map a tile index through rotation + flip (pure integer tile coordinate transform).
// Used by both zarr and tif writers.
inline void mapTileIndex(int tx, int ty, int tilesX, int tilesY,
                         int quadRot, int flipType,
                         int& outTx, int& outTy, int& outTilesX, int& outTilesY)
{
    bool swap = (quadRot % 2) == 1;
    int rTX = swap ? tilesY : tilesX, rTY = swap ? tilesX : tilesY;
    int rx = tx, ry = ty;
    switch (quadRot) {
        case 1: rx = ty;              ry = tilesX - 1 - tx; break;
        case 2: rx = tilesX - 1 - tx; ry = tilesY - 1 - ty; break;
        case 3: rx = tilesY - 1 - ty; ry = tx;              break;
        default: break;
    }
    int fx = rx, fy = ry;
    if (flipType == 0)      fy = rTY - 1 - ry;
    else if (flipType == 1) fx = rTX - 1 - rx;
    else if (flipType == 2) { fx = rTX - 1 - rx; fy = rTY - 1 - ry; }
    outTx = fx; outTy = fy; outTilesX = rTX; outTilesY = rTY;
}

// Write one band's slices as zarr chunks using writeChunk (fast, avoids subarray overhead).
// chunks0 = {chunkZ, chunkY, chunkX} from the L0 dataset's chunk shape.
template <typename T>
void writeZarrBand(z5::Dataset* dsOut, const std::vector<cv::Mat>& slices,
                   uint32_t bandIdx, const std::vector<size_t>& chunks0,
                   size_t tilesXSrc, size_t tilesYSrc,
                   int rotQuad, int flipAxis);

// 2×2×2 mean downsample: dst voxel = mean of up to 8 src voxels.
// src layout: srcZ × srcY × srcX (row-major).  dst layout: dstZ × dstY × dstX.
// dstZ/Y/X should be (srcZ+1)/2 etc., but edge chunks may be smaller.
// srcActualZ/Y/X = actual valid extent in src (may be < chunk shape for edge chunks).
template <typename T>
void downsampleChunk(const T* src, size_t srcZ, size_t srcY, size_t srcX,
                     T* dst, size_t dstZ, size_t dstY, size_t dstX,
                     size_t srcActualZ, size_t srcActualY, size_t srcActualX);

// 2×2×2 mean downsample from src tile into a sub-region of dst at (dstOffY, dstOffX).
// dst layout: dstZ × dstY × dstX (row-major). Writes at offset within the Y×X plane.
// srcActualZ/Y/X = actual valid extent in src (may be < srcZ/Y/X for edge tiles).
template <typename T>
void downsampleTileInto(const T* src, size_t srcZ, size_t srcY, size_t srcX,
                        T* dst, size_t dstZ, size_t dstY, size_t dstX,
                        size_t srcActualZ, size_t srcActualY, size_t srcActualX,
                        size_t dstOffY, size_t dstOffX);

// Build one pyramid level (2x mean downsample) via readChunk/writeChunk + OMP.
// numParts/partId partition the output tile-rows across VMs (1/0 = no partitioning).
template <typename T>
void buildPyramidLevel(z5::filesystem::handle::File& outFile, int level,
                       size_t CH, size_t CW,
                       int numParts = 1, int partId = 0);

// Create pyramid level datasets L1-L5 (metadata only, no data).
// Called by --pre so that multi-part workers can open existing datasets.
void createPyramidDatasets(z5::filesystem::handle::File& outFile,
                           const std::vector<size_t>& shape0,
                           size_t CH, size_t CW, bool isU16);

// Write OME-Zarr .zattrs multiscales JSON.
void writeZarrAttrs(z5::filesystem::handle::File& outFile,
                    const std::filesystem::path& volPath, int groupIdx,
                    size_t baseZ, double sliceStep, double accumStep,
                    const std::string& accumTypeStr, size_t accumSamples,
                    const cv::Size& canvasSize, size_t CZ, size_t CH, size_t CW);

