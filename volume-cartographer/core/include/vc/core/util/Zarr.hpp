#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

// Declared, not included: this header is deliberately free of the JSON wrapper
// so that including it does not pull a JSON implementation into every user.
// Callers of buildMultiscales() include utils/Json.hpp themselves.
namespace utils { class Json; }

// 2×2×1 mean downsample from src tile into a sub-region of dst at (dstOffY, dstOffX).
// Z is preserved (no decimation in Z), while Y/X are downsampled by factor 2.
// dst layout: dstZ × dstY × dstX (row-major). Writes at offset within the Y×X plane.
template <typename T>
void downsampleTileIntoPreserveZ(const T* src, size_t srcZ, size_t srcY, size_t srcX,
                                T* dst, size_t dstZ, size_t dstY, size_t dstX,
                                size_t srcActualZ, size_t srcActualY, size_t srcActualX,
                                size_t dstOffY, size_t dstOffX);

// Build chunk rows [rowStart, rowEnd) of one pyramid level (2x mean downsample of
// the level below on disk) in parallel; rows past the level's end are ignored.
template <typename T>
void buildPyramidLevel(const std::filesystem::path& outFile, int level,
                       size_t rowStart = 0, size_t rowEnd = SIZE_MAX);

// Create pyramid level datasets L1-L5 (metadata only, no data).
// Called by --pre so that multi-part workers can open existing datasets.
// compressor/compressionLevel/dimensionSeparator are forwarded to
// vcr::zarr::Writer so the pyramid matches L0.
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
