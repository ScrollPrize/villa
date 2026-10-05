#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace vc { class VcDataset; }
// Declared, not included: this header is deliberately free of the JSON wrapper
// so that including it does not pull a JSON implementation into every user.
// Callers of buildMultiscales() include utils/Json.hpp themselves.
namespace utils { class Json; }

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
void writeZarrBand(vc::VcDataset* dsOut, const std::vector<cv::Mat>& slices,
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

// 2×2×1 mean downsample from src tile into a sub-region of dst at (dstOffY, dstOffX).
// Z is preserved (no decimation in Z), while Y/X are downsampled by factor 2.
// dst layout: dstZ × dstY × dstX (row-major). Writes at offset within the Y×X plane.
template <typename T>
void downsampleTileIntoPreserveZ(const T* src, size_t srcZ, size_t srcY, size_t srcX,
                                T* dst, size_t dstZ, size_t dstY, size_t dstX,
                                size_t srcActualZ, size_t srcActualY, size_t srcActualX,
                                size_t dstOffY, size_t dstOffX);

// Build one pyramid level (2x mean downsample) via readChunk/writeChunk + OMP.
// numParts/partId partition the output tile-rows across VMs (1/0 = no partitioning).
template <typename T>
void buildPyramidLevel(const std::filesystem::path& outFile, int level,
                       size_t CH, size_t CW,
                       int numParts = 1, int partId = 0);

// Create pyramid level datasets L1-L5 (metadata only, no data).
// Called by --pre so that multi-part workers can open existing datasets.
// compressor/compressionLevel/dimensionSeparator are forwarded to
// createZarrDataset so the pyramid matches L0.
void createPyramidDatasets(const std::filesystem::path& outFile,
                           const std::vector<size_t>& shape0,
                           size_t CH, size_t CW, bool isU16,
                           const std::string& compressor = "blosc",
                           int compressionLevel = -1,
                           const std::string& dimensionSeparator = ".");

// The value of the OME-NGFF "multiscales" attribute, as one descriptor object.
//
// Split out of writeZarrAttrs so the representation can be tested without a
// filesystem or a Volume: the rules about units and relative scaling described
// below are the part that went wrong, and they are pure JSON construction.
[[nodiscard]] utils::Json buildMultiscales(double baseVoxelSize,
                                           const std::string& voxelUnit,
                                           double sliceStep,
                                           double pixelsPerVoxel);

// Write OME-Zarr .zattrs multiscales JSON.
//
// The multiscales block is always written: it is the image's discovery metadata
// (axes, dataset paths, coordinateTransformations), and a reader that cannot find
// it does not recognise the output as a multiscale image at all.
//
// The two things a `scale` can mean are kept separate:
//
//   * Physical scale, when `baseVoxelSize > 0`. The declared scale is physical
//     units per axis -- Z = baseVoxelSize * sliceStep, Y/X = baseVoxelSize /
//     pixelsPerVoxel (baseVoxelSize describes one source voxel at the rendered
//     level) -- and `axes[*].unit` carries `voxelUnit`.
//
//   * Relative pyramid scale, when `baseVoxelSize <= 0`, meaning the physical
//     size is unknown. No unit is declared anywhere, and each dataset's scale
//     states the factor between that level and level 0, as OME-NGFF 0.4 requires
//     when a physical scale is unavailable: level 0 is [1, 1, 1] and Y/X double
//     per level because the pyramid halves only Y/X (see createPyramidDatasets).
//     Nothing in the document is a physical length, so a reader cannot mistake
//     the numbers for one -- which is what a bare scale of 1.0 with no unit
//     allowed.
void writeZarrAttrs(const std::filesystem::path& outFile,
                    const std::filesystem::path& volPath, int groupIdx,
                    size_t baseZ, double sliceStep, double accumStep,
                    const std::string& accumTypeStr, size_t accumSamples,
                    const cv::Size& canvasSize, size_t CZ, size_t CH, size_t CW,
                    double baseVoxelSize = 1.0,
                    const std::string& voxelUnit = "",
                    double pixelsPerVoxel = 1.0);

// Write a dense uint8 ZYX subregion into a freshly created dataset via
// writeChunk(). Chunks overlapping the region are materialized; untouched
// chunks are left absent and are expected to read back as the dataset's
// configured fill value.
//
// Precondition: dsOut must refer to a new output dataset. This helper does not
// preserve any prior chunk contents outside the written subregion.
void writeZarrRegionU8ByChunk(vc::VcDataset* dsOut,
                              const std::vector<size_t>& offset,
                              const std::vector<size_t>& regionShape,
                              const uint8_t* data,
                              uint8_t fillValue = 0);
