#include "vc/core/util/AffineTransform.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfaceIdentity.hpp"

#include <tiffio.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace vc::core::util {
namespace {
using Tiff = std::unique_ptr<TIFF, decltype(&TIFFClose)>;

Tiff openTiff(const std::filesystem::path& path, const char* mode)
{
    Tiff file(TIFFOpen(path.string().c_str(), mode), TIFFClose);
    if (!file) throw std::runtime_error("Cannot open TIFF: " + path.string());
    return file;
}

// Sequential rows, with one decoded tile row retained for tiled inputs.
class FloatRows {
public:
    explicit FloatRows(const std::filesystem::path& path) : file(openTiff(path, "r"))
    {
        uint16_t bits = 0, format = 0, samples = 0, orientation = 0;
        TIFFGetField(file.get(), TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(file.get(), TIFFTAG_IMAGELENGTH, &height);
        TIFFGetFieldDefaulted(file.get(), TIFFTAG_BITSPERSAMPLE, &bits);
        TIFFGetFieldDefaulted(file.get(), TIFFTAG_SAMPLEFORMAT, &format);
        TIFFGetFieldDefaulted(file.get(), TIFFTAG_SAMPLESPERPIXEL, &samples);
        TIFFGetFieldDefaulted(file.get(), TIFFTAG_ORIENTATION, &orientation);
        if (!width || !height || bits != 32 || format != SAMPLEFORMAT_IEEEFP ||
            samples != 1 || orientation != ORIENTATION_TOPLEFT)
            throw std::runtime_error("Expected top-left single-channel float32 TIFF: " + path.string());
        if (TIFFIsTiled(file.get())) {
            TIFFGetField(file.get(), TIFFTAG_TILEWIDTH, &tileWidth);
            TIFFGetField(file.get(), TIFFTAG_TILELENGTH, &tileHeight);
            if (!tileWidth || !tileHeight) throw std::runtime_error("Invalid TIFF tile shape");
            tile.resize((TIFFTileSize(file.get()) + sizeof(float) - 1) / sizeof(float));
            rows.resize(size_t(width) * tileHeight);
        } else {
            rows.resize(width);
            uint16_t compression = 0;
            TIFFGetFieldDefaulted(file.get(), TIFFTAG_COMPRESSION, &compression);
            if (compression == COMPRESSION_NONE) {
                // TIFFReadScanline can buffer the entire source strip, which may
                // be the whole image. Read plain float rows at the recorded offsets.
                TIFFGetFieldDefaulted(file.get(), TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);
                if (!rowsPerStrip || !TIFFGetField(file.get(), TIFFTAG_STRIPOFFSETS, &stripOffsets) ||
                    !TIFFGetField(file.get(), TIFFTAG_STRIPBYTECOUNTS, &stripBytes))
                    throw std::runtime_error("Missing TIFF strip layout");
                raw.open(path, std::ios::binary);
                raw.exceptions(std::ios::badbit | std::ios::failbit);
            }
        }
    }

    const float* read(uint32_t y)
    {
        if (!tileWidth) {
            if (raw.is_open()) {
                const uint32_t strip = y / rowsPerStrip;
                const uint64_t offset = uint64_t(y % rowsPerStrip) * width * sizeof(float);
                const uint64_t bytes = uint64_t(width) * sizeof(float);
                if (strip >= TIFFNumberOfStrips(file.get()) || offset > stripBytes[strip] ||
                    bytes > stripBytes[strip] - offset ||
                    stripOffsets[strip] > uint64_t(std::numeric_limits<std::streamoff>::max()) - offset)
                    throw std::runtime_error("Truncated coordinate TIFF strip");
                raw.seekg(std::streamoff(stripOffsets[strip] + offset));
                raw.read(reinterpret_cast<char*>(rows.data()), std::streamsize(bytes));
                if (TIFFIsByteSwapped(file.get())) TIFFSwabArrayOfFloat(rows.data(), width);
                return rows.data();
            }
            if (TIFFReadScanline(file.get(), rows.data(), y, 0) != 1)
                throw std::runtime_error("Failed reading coordinate TIFF row");
            return rows.data();
        }
        if (y % tileHeight == 0) {
            for (uint32_t x = 0; x < width; x += tileWidth) {
                if (TIFFReadTile(file.get(), tile.data(), x, y, 0, 0) != TIFFTileSize(file.get()))
                    throw std::runtime_error("Failed reading coordinate TIFF tile");
                for (uint32_t dy = 0; dy < std::min(tileHeight, height - y); ++dy)
                    std::copy_n(tile.data() + size_t(dy) * tileWidth,
                        std::min(tileWidth, width - x), rows.data() + size_t(dy) * width + x);
            }
        }
        return rows.data() + size_t(y % tileHeight) * width;
    }
    uint32_t width = 0, height = 0;
private:
    Tiff file;
    uint32_t tileWidth = 0, tileHeight = 0;
    std::vector<float> rows, tile;
    std::ifstream raw;
    uint32_t rowsPerStrip = 0;
    uint64_t* stripOffsets = nullptr;
    uint64_t* stripBytes = nullptr;
};

void scaleMetadata(utils::Json& meta, double factor,
                   const std::optional<std::array<int, 3>>& targetShape)
{
    if (!std::isfinite(factor) || factor <= 0)
        throw std::invalid_argument("Scale must be finite and positive");
    if (!meta.contains("scale") || meta["scale"].size() != 2)
        throw std::runtime_error("TIFXYZ metadata must contain two grid scale values");
    for (int i = 0; i < 2; ++i) {
        const double scale = meta["scale"][i].get_double() / factor;
        if (!std::isfinite(scale) || scale <= 0 || !std::isfinite(float(scale)) || float(scale) == 0)
            throw std::runtime_error("Scaled grid spacing is not representable");
        meta["scale"][i] = scale;
    }
    if (meta.contains("bbox"))
        for (int side = 0; side < 2; ++side)
            for (int axis = 0; axis < 3; ++axis)
                meta["bbox"][side][axis] = meta["bbox"][side][axis].get_double() * factor;
    if (meta.contains("area_vx2")) meta["area_vx2"] = meta["area_vx2"].get_double() * factor * factor;
    if (meta.contains("output_step_vx")) meta["output_step_vx"] = meta["output_step_vx"].get_double() * factor;
    if (meta.contains("voxel_size_um")) meta["voxel_size_um"] = meta["voxel_size_um"].get_double() / factor;
    else if (meta.contains("fit_config") && meta["fit_config"].contains("voxel_size_um"))
        meta["voxel_size_um"] = meta["fit_config"]["voxel_size_um"].get_double() / factor;
    // fit_config is historical provenance, not the transformed coordinate system.
    if (targetShape) {
        for (int size : *targetShape)
            if (size <= 0) throw std::invalid_argument("Target shape must be positive");
        meta["base_shape_zyx"] = utils::Json::array();
        for (int size : *targetShape) meta["base_shape_zyx"].push_back(size);
    } else if (factor != 1 && meta.contains("base_shape_zyx")) {
        throw std::invalid_argument("Input records base_shape_zyx; supply the actual target shape (pyramid rounding cannot be inferred)");
    }
}

cv::Vec3f scaledPoint(const cv::Vec3f& point, double factor)
{
    if (point[0] == -1 || !std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2]))
        return point;
    cv::Vec3f result;
    for (int axis = 0; axis < 3; ++axis) {
        const double value = double(point[axis]) * factor;
        if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
            throw std::runtime_error("Scaled coordinate exceeds float32 range");
        result[axis] = float(value);
    }
    return result;
}
} // namespace

void scaleSurfaceWithoutResampling(QuadSurface* surface, double factor,
                                  std::optional<std::array<int, 3>> targetShape)
{
    if (!surface) throw std::invalid_argument("Missing surface");
    auto meta = surface->meta;
    meta["scale"] = utils::Json::array();
    for (int axis = 0; axis < 2; ++axis) meta["scale"].push_back(double(surface->scale()[axis]));
    scaleMetadata(meta, factor, targetShape);
    auto* points = surface->rawPointsPtr();
    for (auto& point : *points) point = scaledPoint(point, factor);
    surface->_scale = cv::Vec2f(meta["scale"][0].get_double(), meta["scale"][1].get_double());
    surface->meta = std::move(meta);
    refreshTransformedSurfaceState(surface);
}

void scaleTifxyzWithoutResampling(const std::filesystem::path& input,
                                 const std::filesystem::path& requestedOutput, double factor,
                                 std::optional<std::array<int, 3>> targetShape)
{
    namespace fs = std::filesystem;
    auto output = fs::absolute(requestedOutput).lexically_normal();
    if (output.filename().empty()) output = output.parent_path();
    if (fs::exists(output)) throw std::runtime_error("Output already exists: " + output.string());
    if (fs::exists(input / "corrections.json"))
        throw std::runtime_error("Streaming scaling does not yet transform corrections.json; refusing to copy stale coordinates");
    auto meta = utils::Json::parse_file(input / "meta.json");
    scaleMetadata(meta, factor, targetShape);
    std::array<FloatRows, 3> readers{FloatRows(input / "x.tif"), FloatRows(input / "y.tif"), FloatRows(input / "z.tif")};
    const auto width = readers[0].width, height = readers[0].height;
    for (const auto& reader : readers)
        if (reader.width != width || reader.height != height)
            throw std::runtime_error("Coordinate TIFF dimensions differ");
    auto parent = output.parent_path();
    if (parent.empty()) parent = ".";
    fs::create_directories(parent);
    static std::atomic<unsigned> serial{0};
    fs::path staging;
    do {
        staging = parent / (output.filename().string() + ".scale-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
            std::to_string(serial++) + ".tmp");
    } while (!fs::create_directory(staging));
    try {
        const std::array<const char*, 3> names{"x.tif", "y.tif", "z.tif"};
        std::array<Tiff, 3> writers{openTiff(staging / names[0], "w8"),
            openTiff(staging / names[1], "w8"), openTiff(staging / names[2], "w8")};
        for (auto& writer : writers) {
            auto* t = writer.get();
            if (!TIFFSetField(t, TIFFTAG_IMAGEWIDTH, width) || !TIFFSetField(t, TIFFTAG_IMAGELENGTH, height) ||
                !TIFFSetField(t, TIFFTAG_BITSPERSAMPLE, 32) || !TIFFSetField(t, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP) ||
                !TIFFSetField(t, TIFFTAG_SAMPLESPERPIXEL, 1) || !TIFFSetField(t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG) ||
                !TIFFSetField(t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK) ||
                !TIFFSetField(t, TIFFTAG_COMPRESSION, COMPRESSION_NONE) || !TIFFSetField(t, TIFFTAG_ROWSPERSTRIP, 1))
                throw std::runtime_error("Cannot configure coordinate TIFF output");
        }
        std::array<std::vector<float>, 3> rows;
        for (auto& row : rows) row.resize(width);
        for (uint32_t y = 0; y < height; ++y) {
            std::array<const float*, 3> in{readers[0].read(y), readers[1].read(y), readers[2].read(y)};
            for (uint32_t x = 0; x < width; ++x) {
                const auto point = scaledPoint({in[0][x], in[1][x], in[2][x]}, factor);
                for (int axis = 0; axis < 3; ++axis) rows[axis][x] = point[axis];
            }
            for (int axis = 0; axis < 3; ++axis)
                if (TIFFWriteScanline(writers[axis].get(), rows[axis].data(), y, 0) != 1)
                    throw std::runtime_error("Failed writing coordinate TIFF row");
        }
        for (auto& writer : writers) {
            if (!TIFFWriteDirectory(writer.get())) throw std::runtime_error("Failed finalizing coordinate TIFF");
            writer.reset();
        }
        for (const auto& entry : fs::directory_iterator(input)) {
            const auto name = entry.path().filename().string();
            if (name == "meta.json" || name == "x.tif" || name == "y.tif" || name == "z.tif") continue;
            // Presence of this marker makes VC3D treat a surface as immutable.
            if (name == "catalog-origin.json") continue;
            if (entry.is_regular_file()) fs::copy_file(entry.path(), staging / entry.path().filename());
        }
        meta["uuid"] = newDerivedSurfaceIdentity(meta);
        meta["name"] = output.filename().string();
        std::ofstream out(staging / "meta.json");
        out.exceptions(std::ios::badbit | std::ios::failbit);
        out << meta.dump(4) << '\n';
        out.close();
        if (fs::exists(output)) throw std::runtime_error("Output appeared during conversion");
        fs::rename(staging, output);
    } catch (...) {
        std::error_code ignored;
        fs::remove_all(staging, ignored);
        throw;
    }
}
} // namespace vc::core::util
