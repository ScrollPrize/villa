#pragma once
// Surfaces stored as tifxyz directories: x.tif, y.tif, z.tif, meta.json and an optional
// mask.tif, local or remote. The one module that knows the file layout: Reader decodes
// windows of the stored grid with their validity, Surface holds a whole grid in memory,
// and positions between vertices come from vcr::grid.
#include "vc_reader/grid.hpp"
#include "vc_reader/source_reader.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vcr::tifxyz {

using Bounds = std::array<double, 6>;  // x min, y min, z min, x max, y max, z max

struct Metadata {
    std::string uuid;
    std::array<double, 2> scale{1, 1};  // stored cells per full-resolution pixel: rows, columns
    std::optional<Bounds> bbox;
    std::optional<double> area;
    nlohmann::json extra = nlohmann::json::object();  // every other meta.json field
};

// The extent of a grid at full resolution: as vc_render_tifxyz sizes its output, the stored
// scale taken as float32, rounded half up, at least 1. A scale that is not a positive finite
// float32 leaves the stored extent.
[[nodiscard]] std::int64_t fullExtent(std::int64_t stored, double scale);

// Whether a stored bounding box holds tifxyz's -1 missing-point marker, so is not a real extent.
[[nodiscard]] bool hasMissingMarker(const Bounds& bbox);

// Whether stored XYZ coordinates are a valid vertex: all finite and nonnegative. A negative
// coordinate marks the vertex invalid, whatever its value.
[[nodiscard]] inline bool vertexValid(const float* xyz)
{
    return xyz[0] >= 0 && xyz[1] >= 0 && xyz[2] >= 0 && std::isfinite(xyz[0]) && std::isfinite(xyz[1]) && std::isfinite(xyz[2]);
}

// The raster of a full-resolution window over a grid stored at `scale` (rows, columns):
// `start` and `size` in rows and columns of full-resolution pixels, one pixel each.
[[nodiscard]] grid::Raster fullRaster(const std::array<double, 2>& scale, const std::array<std::int64_t, 2>& start, const std::array<std::int64_t, 2>& size);

// A window of a stored grid: rows x cols x 3 XYZ floats as stored, and which vertices are
// valid (vertexValid, and the mask when one was read).
struct Coordinates {
    std::int64_t rows = 0, cols = 0;
    std::array<std::int64_t, 2> whole{};  // rows, cols of the grid
    std::vector<float> xyz;
    std::vector<std::uint8_t> valid;
    std::size_t storedBytes = 0;  // coordinate bytes decoded for this window
};

// meta.json of a tifxyz directory. Scale must be two positive finite numbers.
[[nodiscard]] Metadata parseMetadata(const nlohmann::json& document, const std::string& location);
[[nodiscard]] Metadata readMetadata(const std::string& location, const sources::OptionsFor& options = {});

// mask.tif at `location` for a grid of rows x cols: nonzero where allowed, empty when there is none.
// Its first page counts; at an integer multiple of the grid a vertex needs its whole block, any
// other shape raises.
[[nodiscard]] std::vector<std::uint8_t> readMask(const std::string& location, std::int64_t rows, std::int64_t cols, const sources::OptionsFor& options = {});

// One tifxyz directory open for windowed reads: x.tif, y.tif, z.tif and, with `useMask`,
// mask.tif (none when there is no file). Reads go through source readers with `settings`,
// their caches included; a window decodes only the strips and tiles it touches.
class Reader {
public:
    explicit Reader(std::string location, bool useMask = true, sources::SourceReader::Settings settings = {});
    ~Reader();
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;

    [[nodiscard]] const std::string& location() const;
    [[nodiscard]] const Metadata& metadata() const;
    [[nodiscard]] std::array<std::int64_t, 2> shape() const;  // rows, cols
    // The window (row, column, rows, columns), inside the grid.
    [[nodiscard]] Coordinates window(const std::array<std::int64_t, 4>& window);
    [[nodiscard]] Coordinates whole();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class Surface {
public:
    // `xyz` holds rows x cols x 3 XYZ floats. A vertex is valid when vertexValid says so and
    // `mask` (nonzero; may be empty) allows it; all three coordinates of any other are set to -1.
    Surface(std::int64_t rows, std::int64_t cols, std::vector<float> xyz, std::vector<std::uint8_t> mask, Metadata metadata, std::string location = {});

    [[nodiscard]] std::int64_t rows() const { return rows_; }
    [[nodiscard]] std::int64_t cols() const { return cols_; }
    [[nodiscard]] const std::vector<float>& xyz() const { return xyz_; }
    [[nodiscard]] const std::vector<std::uint8_t>& valid() const { return valid_; }
    [[nodiscard]] const Metadata& metadata() const { return metadata_; }
    [[nodiscard]] Metadata& metadata() { return metadata_; }
    [[nodiscard]] const std::string& location() const { return location_; }
    [[nodiscard]] std::array<std::int64_t, 2> fullShape() const;
    // The extent of the valid vertices; none without any.
    [[nodiscard]] std::optional<Bounds> bounds() const;
    // The surface in a volume `factor` times coarser: coordinates divided by it, scale multiplied.
    [[nodiscard]] Surface retargeted(double factor) const;

    // Positions and normals at `n` real-valued grid positions (`xy`: column, row pairs), NaN
    // where no patch covers one. Normals are as grid::Surface::resample gives them: with the
    // unit spacing here, unit normals (column direction x row direction).
    void pointsAt(const float* xy, std::size_t n, grid::Interpolation interpolation, float* xyz, float* normals = nullptr) const;
    // The same at every pixel of a raster.
    void resample(const grid::Raster& raster, grid::Interpolation interpolation, float* xyz, float* normals = nullptr) const;
    // tifxyz::fullRaster at this surface's scale.
    [[nodiscard]] grid::Raster fullRaster(const std::array<std::int64_t, 2>& start, const std::array<std::int64_t, 2>& size) const;

private:
    struct Interpolators;
    const grid::Surface& interpolator(grid::Interpolation interpolation) const;
    std::int64_t rows_ = 0, cols_ = 0;
    std::vector<float> xyz_;
    std::vector<std::uint8_t> valid_;
    Metadata metadata_;
    std::string location_;
    std::shared_ptr<Interpolators> interpolators_;
};

// A whole surface. Without `useMask`, mask.tif is ignored. A stored bounding box with a missing-point marker becomes the real extent.
[[nodiscard]] Surface read(const std::string& location, bool useMask = true, const sources::OptionsFor& options = {});

struct Entry {
    std::string location;
    Metadata metadata;
    std::string error;  // why its metadata could not be read; empty when it could
};
// Directories below `folder` holding x.tif, y.tif, z.tif and meta.json, by location. Only their
// metadata is read, except that with `zRange` a box with a missing-point marker is resolved from
// the coordinates to filter on it. A surface with no box is not filtered out.
[[nodiscard]] std::vector<Entry> list(const std::string& folder, bool recursive = true, const std::optional<std::array<double, 2>>& zRange = std::nullopt, const sources::OptionsFor& options = {});

struct WriteOptions {
    std::string compression = "lzw";  // as TiffOptions
    std::optional<int> tile = 1024;  // none: each image is one strip
    bool mask = true;  // write mask.tif (255 valid, 0 invalid)
    bool overwrite = false;
};
// Writes the directory through a temporary beside it. An existing `path` raises unless
// `overwrite`, which replaces the whole directory. Metadata carries the extent of the valid
// vertices (zeros without any). A surface without a UUID gets one, kept on `surface`.
void write(const std::filesystem::path& path, Surface& surface, const WriteOptions& options = {});

}  // namespace vcr::tifxyz
