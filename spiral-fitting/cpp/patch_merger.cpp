#include "patch_merger.hpp"

#include <spiral_graph/surface_index.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <tiffio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vc_spiral::patch_merger {
namespace {

using Clock = std::chrono::steady_clock;
using json = nlohmann::json;
using surfcore::QueryScratch;
using surfcore::SurfaceData;
using surfcore::SurfaceHit;
using surfcore::SurfacePatchIndex;
using surfcore::SurfaceTileCandidate;
using surfcore::Vec3;

constexpr double kClusterThresholdSquared = 4.0;

double seconds_since(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

struct Vec2 {
    double x = 0.0;
    double y = 0.0;
};

Vec2 operator+(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }
Vec2 operator-(Vec2 a, Vec2 b) { return {a.x - b.x, a.y - b.y}; }
Vec2 operator*(Vec2 a, double value) { return {a.x * value, a.y * value}; }
Vec2 operator/(Vec2 a, double value) { return {a.x / value, a.y / value}; }
double dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
double norm_squared(Vec2 value) { return dot(value, value); }

struct Pose {
    double r00 = 1.0;
    double r01 = 0.0;
    double r10 = 0.0;
    double r11 = 1.0;
    double tx = 0.0;
    double ty = 0.0;

    Vec2 apply(Vec2 value) const
    {
        return {
            r00 * value.x + r01 * value.y + tx,
            r10 * value.x + r11 * value.y + ty,
        };
    }

    Pose inverse() const
    {
        Pose result;
        result.r00 = r00;
        result.r01 = r10;
        result.r10 = r01;
        result.r11 = r11;
        result.tx = -(result.r00 * tx + result.r01 * ty);
        result.ty = -(result.r10 * tx + result.r11 * ty);
        return result;
    }

    bool reflected() const { return r00 * r11 - r01 * r10 < 0.0; }
};

struct Patch {
    std::filesystem::path path;
    std::string id;
    json metadata;
    std::shared_ptr<SurfaceData> surface;
    double scale_row = 1.0;
    double scale_col = 1.0;
    int erode_cells = 0;
    double area = 0.0;
    std::vector<float> boundary_distance_squared;
    std::optional<double> area_cm2_per_vx2;
};

struct TiffInfo {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint16_t bits = 0;
    std::uint16_t samples = 1;
    std::uint16_t format = SAMPLEFORMAT_UINT;
    std::uint16_t planar = PLANARCONFIG_CONTIG;
    bool tiled = false;
};

class TiffHandle {
public:
    TiffHandle(const std::filesystem::path& path, const char* mode)
        : path_(path), handle_(TIFFOpen(path.string().c_str(), mode))
    {
        if (!handle_) {
            throw std::runtime_error("cannot open TIFF " + path.string());
        }
    }
    ~TiffHandle() { TIFFClose(handle_); }
    TiffHandle(const TiffHandle&) = delete;
    TiffHandle& operator=(const TiffHandle&) = delete;
    TIFF* get() const { return handle_; }
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
    TIFF* handle_ = nullptr;
};

TiffInfo tiff_info(const std::filesystem::path& path)
{
    TiffHandle file(path, "r");
    TiffInfo result;
    TIFFGetField(file.get(), TIFFTAG_IMAGEWIDTH, &result.width);
    TIFFGetField(file.get(), TIFFTAG_IMAGELENGTH, &result.height);
    TIFFGetFieldDefaulted(file.get(), TIFFTAG_BITSPERSAMPLE, &result.bits);
    TIFFGetFieldDefaulted(file.get(), TIFFTAG_SAMPLESPERPIXEL, &result.samples);
    TIFFGetFieldDefaulted(file.get(), TIFFTAG_SAMPLEFORMAT, &result.format);
    TIFFGetFieldDefaulted(file.get(), TIFFTAG_PLANARCONFIG, &result.planar);
    result.tiled = TIFFIsTiled(file.get()) != 0;
    if (result.width == 0 || result.height == 0) {
        throw std::runtime_error("empty TIFF " + path.string());
    }
    return result;
}

std::vector<float> read_float_tiff(
    const std::filesystem::path& path, const TiffInfo& expected)
{
    TiffHandle file(path, "r");
    const TiffInfo actual = tiff_info(path);
    if (actual.width != expected.width || actual.height != expected.height
        || actual.samples != 1 || actual.bits != 32
        || actual.format != SAMPLEFORMAT_IEEEFP) {
        throw std::runtime_error(
            "expected matching one-plane float32 TIFF: " + path.string());
    }
    std::vector<float> values(
        static_cast<std::size_t>(actual.width) * actual.height);
    if (!actual.tiled) {
        for (std::uint32_t row = 0; row < actual.height; ++row) {
            if (TIFFReadScanline(
                    file.get(), values.data() + static_cast<std::size_t>(row)
                        * actual.width, row) < 0) {
                throw std::runtime_error("failed reading TIFF " + path.string());
            }
        }
        return values;
    }

    std::uint32_t tile_width = 0, tile_height = 0;
    TIFFGetField(file.get(), TIFFTAG_TILEWIDTH, &tile_width);
    TIFFGetField(file.get(), TIFFTAG_TILELENGTH, &tile_height);
    std::vector<float> tile(static_cast<std::size_t>(tile_width) * tile_height);
    for (std::uint32_t row = 0; row < actual.height; row += tile_height) {
        for (std::uint32_t col = 0; col < actual.width; col += tile_width) {
            const ttile_t index = TIFFComputeTile(file.get(), col, row, 0, 0);
            if (TIFFReadEncodedTile(
                    file.get(), index, tile.data(),
                    static_cast<tmsize_t>(tile.size() * sizeof(float))) < 0) {
                throw std::runtime_error("failed reading tiled TIFF " + path.string());
            }
            const std::uint32_t copy_rows = std::min(tile_height, actual.height - row);
            const std::uint32_t copy_cols = std::min(tile_width, actual.width - col);
            for (std::uint32_t local_row = 0; local_row < copy_rows; ++local_row) {
                std::copy_n(
                    tile.data() + static_cast<std::size_t>(local_row) * tile_width,
                    copy_cols,
                    values.data() + static_cast<std::size_t>(row + local_row)
                        * actual.width + col);
            }
        }
    }
    return values;
}

bool sentinel(const Vec3& value)
{
    return value.x == -1.0f && value.y == -1.0f && value.z == -1.0f;
}

std::vector<std::uint8_t> erode_four_connected(
    std::vector<std::uint8_t> valid, std::size_t rows, std::size_t cols,
    int iterations)
{
    std::vector<std::uint8_t> next(valid.size(), 0);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::fill(next.begin(), next.end(), 0);
        for (std::size_t row = 1; row + 1 < rows; ++row) {
            for (std::size_t col = 1; col + 1 < cols; ++col) {
                const std::size_t index = row * cols + col;
                next[index] = valid[index] && valid[index - 1] && valid[index + 1]
                    && valid[index - cols] && valid[index + cols];
            }
        }
        valid.swap(next);
    }
    return valid;
}

void edt_1d(const std::vector<double>& input, std::vector<double>& output,
            std::size_t count, double spacing)
{
    std::vector<std::size_t> envelope(count);
    std::vector<double> intersections(count + 1);
    std::size_t k = 0;
    envelope[0] = 0;
    intersections[0] = -std::numeric_limits<double>::infinity();
    intersections[1] = std::numeric_limits<double>::infinity();
    const double spacing_squared = spacing * spacing;
    for (std::size_t q = 1; q < count; ++q) {
        double crossing = 0.0;
        while (true) {
            const std::size_t p = envelope[k];
            crossing = ((input[q] + spacing_squared * q * q)
                        - (input[p] + spacing_squared * p * p))
                / (2.0 * spacing_squared * static_cast<double>(q - p));
            if (crossing > intersections[k] || k == 0) break;
            --k;
        }
        ++k;
        envelope[k] = q;
        intersections[k] = crossing;
        intersections[k + 1] = std::numeric_limits<double>::infinity();
    }
    k = 0;
    for (std::size_t q = 0; q < count; ++q) {
        while (intersections[k + 1] < static_cast<double>(q)) ++k;
        const double delta = static_cast<double>(q) - envelope[k];
        output[q] = spacing_squared * delta * delta + input[envelope[k]];
    }
}

std::vector<float> boundary_distances_squared(
    const std::vector<std::uint8_t>& valid, std::size_t rows, std::size_t cols,
    double row_spacing, double col_spacing)
{
    const std::size_t padded_rows = rows + 2;
    const std::size_t padded_cols = cols + 2;
    constexpr double far = 1e24;
    std::vector<double> first(padded_rows * padded_cols, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            first[(row + 1) * padded_cols + col + 1]
                = valid[row * cols + col] ? far : 0.0;
        }
    }
    std::vector<double> second(first.size(), 0.0);
    std::vector<double> input(std::max(padded_rows, padded_cols));
    std::vector<double> output(input.size());
    for (std::size_t row = 0; row < padded_rows; ++row) {
        std::copy_n(first.data() + row * padded_cols, padded_cols, input.data());
        edt_1d(input, output, padded_cols, col_spacing);
        std::copy_n(output.data(), padded_cols, second.data() + row * padded_cols);
    }
    for (std::size_t col = 0; col < padded_cols; ++col) {
        for (std::size_t row = 0; row < padded_rows; ++row) {
            input[row] = second[row * padded_cols + col];
        }
        edt_1d(input, output, padded_rows, row_spacing);
        for (std::size_t row = 0; row < padded_rows; ++row) {
            first[row * padded_cols + col] = output[row];
        }
    }
    std::vector<float> result(rows * cols, 0.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            if (valid[row * cols + col]) {
                result[row * cols + col] = static_cast<float>(
                    first[(row + 1) * padded_cols + col + 1]);
            }
        }
    }
    return result;
}

Patch load_patch(
    const std::filesystem::path& path, const MergeOptions& options, bool& dropped)
{
    dropped = false;
    std::ifstream metadata_stream(path / "meta.json");
    if (!metadata_stream) {
        throw std::runtime_error("missing meta.json in " + path.string());
    }
    json metadata;
    try {
        metadata_stream >> metadata;
    } catch (const std::exception& error) {
        throw std::runtime_error(
            "invalid metadata in " + path.string() + ": " + error.what());
    }
    if (metadata.value("format", std::string("tifxyz")) != "tifxyz") {
        throw std::runtime_error("input is not tifxyz: " + path.string());
    }
    if (!metadata.contains("scale") || !metadata["scale"].is_array()
        || metadata["scale"].size() != 2) {
        throw std::runtime_error("metadata scale must have two values: " + path.string());
    }
    const double scale_row = metadata["scale"][0].get<double>();
    const double scale_col = metadata["scale"][1].get<double>();
    if (!(scale_row > 0.0) || !(scale_col > 0.0)
        || !std::isfinite(scale_row) || !std::isfinite(scale_col)) {
        throw std::runtime_error("metadata scale must be positive and finite");
    }
    surfcore::Aabb metadata_bounds;
    bool has_metadata_bounds = false;
    if (metadata.contains("bbox")) {
        const json& bbox = metadata["bbox"];
        if (!bbox.is_array() || bbox.size() != 2
            || !bbox[0].is_array() || bbox[0].size() != 3
            || !bbox[1].is_array() || bbox[1].size() != 3) {
            throw std::runtime_error("metadata bbox must be [[x,y,z],[x,y,z]]");
        }
        metadata_bounds.low = {
            bbox[0][0].get<float>(), bbox[0][1].get<float>(),
            bbox[0][2].get<float>(),
        };
        metadata_bounds.high = {
            bbox[1][0].get<float>(), bbox[1][1].get<float>(),
            bbox[1][2].get<float>(),
        };
        if (!metadata_bounds.valid()) {
            throw std::runtime_error("metadata bbox must be finite and ordered");
        }
        has_metadata_bounds = true;
    }
    int erosion = options.erode_cells;
    if (metadata.contains("spiral_patch_erode_cells")) {
        if (!metadata["spiral_patch_erode_cells"].is_number_integer()) {
            throw std::runtime_error("spiral_patch_erode_cells must be an integer");
        }
        erosion = metadata["spiral_patch_erode_cells"].get<int>();
    }
    if (erosion < 0) throw std::runtime_error("patch erosion must be non-negative");

    const TiffInfo x_info = tiff_info(path / "x.tif");
    if (x_info.samples != 1 || x_info.bits != 32
        || x_info.format != SAMPLEFORMAT_IEEEFP) {
        throw std::runtime_error("x.tif must be one-plane float32");
    }
    const TiffInfo y_info = tiff_info(path / "y.tif");
    const TiffInfo z_info = tiff_info(path / "z.tif");
    if (y_info.width != x_info.width || y_info.height != x_info.height
        || z_info.width != x_info.width || z_info.height != x_info.height
        || y_info.samples != 1 || z_info.samples != 1 || y_info.bits != 32
        || z_info.bits != 32 || y_info.format != SAMPLEFORMAT_IEEEFP
        || z_info.format != SAMPLEFORMAT_IEEEFP) {
        throw std::runtime_error("tifxyz coordinate plane dimensions or types differ");
    }
    if (x_info.width < 2 || x_info.height < 2) {
        dropped = true;
        return {};
    }

    auto surface = std::make_shared<SurfaceData>();
    surface->rows = x_info.height;
    surface->cols = x_info.width;
    surface->scale_i = static_cast<float>(scale_col);
    surface->scale_j = static_cast<float>(scale_row);
    surface->metadata_bounds = metadata_bounds;
    surface->has_metadata_bounds = has_metadata_bounds;
    const bool has_mask = std::filesystem::is_regular_file(path / "mask.tif");
    if (!has_mask) {
        try {
            surface->point_source = surfcore::open_mapped_tifxyz_point_source(
                path, surface->rows, surface->cols);
        } catch (const std::exception&) {
            surface->point_source.reset();
        }
    }
    if (!surface->point_source) {
        const auto x = read_float_tiff(path / "x.tif", x_info);
        const auto y = read_float_tiff(path / "y.tif", x_info);
        const auto z = read_float_tiff(path / "z.tif", x_info);
        surface->xyz.resize(x.size());
        for (std::size_t index = 0; index < x.size(); ++index) {
            surface->xyz[index] = {x[index], y[index], z[index]};
        }
    }

    std::vector<std::uint8_t> mask;
    if (has_mask) {
        mask = surfcore::read_tifxyz_mask(
            path / "mask.tif", x_info.height, x_info.width);
    }
    std::vector<std::uint8_t> valid(surface->rows * surface->cols, 0);
    for (std::size_t row = 0; row < surface->rows; ++row) {
        for (std::size_t col = 0; col < surface->cols; ++col) {
            const std::size_t index = row * surface->cols + col;
            const Vec3 point = surface->at(row, col);
            if (!surfcore::finite(point)) {
                throw std::runtime_error("non-finite coordinate in " + path.string());
            }
            const bool is_valid = !sentinel(point) && (mask.empty() || mask[index]);
            valid[index] = is_valid ? 1 : 0;
            if (!is_valid && !surface->xyz.empty()) {
                surface->xyz[index] = {-1.0f, -1.0f, -1.0f};
            }
        }
    }
    valid = erode_four_connected(
        std::move(valid), surface->rows, surface->cols, erosion);
    if (!surface->xyz.empty()) {
        for (std::size_t index = 0; index < valid.size(); ++index) {
            if (!valid[index]) surface->xyz[index] = {-1.0f, -1.0f, -1.0f};
        }
    }
    surface->valid_quads.assign(
        (surface->rows - 1) * (surface->cols - 1), 0);
    std::size_t valid_quad_count = 0;
    for (std::size_t row = 0; row + 1 < surface->rows; ++row) {
        for (std::size_t col = 0; col + 1 < surface->cols; ++col) {
            const bool quad = valid[row * surface->cols + col]
                && valid[row * surface->cols + col + 1]
                && valid[(row + 1) * surface->cols + col]
                && valid[(row + 1) * surface->cols + col + 1];
            surface->valid_quads[row * (surface->cols - 1) + col] = quad ? 1 : 0;
            valid_quad_count += quad ? 1 : 0;
        }
    }
    if (valid_quad_count == 0) {
        dropped = true;
        return {};
    }

    Patch result;
    result.path = std::filesystem::absolute(path);
    result.metadata = metadata;
    result.id = metadata.value("uuid", path.filename().string());
    if (result.id.empty()) result.id = path.filename().string();
    if (result.id == "." || result.id == ".."
        || result.id.find('/') != std::string::npos
        || result.id.find('\\') != std::string::npos) {
        throw std::runtime_error("patch ID is not a safe directory name: " + result.id);
    }
    surface->id = result.id;
    result.surface = std::move(surface);
    result.scale_row = scale_row;
    result.scale_col = scale_col;
    result.erode_cells = erosion;
    result.area = static_cast<double>(valid_quad_count) / (scale_row * scale_col);
    result.boundary_distance_squared = boundary_distances_squared(
        valid, result.surface->rows, result.surface->cols,
        1.0 / scale_row, 1.0 / scale_col);
    if (metadata.contains("area_vx2") && metadata.contains("area_cm2")) {
        const double vx2 = metadata["area_vx2"].get<double>();
        const double cm2 = metadata["area_cm2"].get<double>();
        if (vx2 > 0.0 && cm2 >= 0.0 && std::isfinite(vx2) && std::isfinite(cm2)) {
            result.area_cm2_per_vx2 = cm2 / vx2;
        }
    }
    return result;
}

struct Sample {
    Vec3 xyz;
    Vec3 tangent_u;
    Vec3 tangent_v;
    float boundary_weight = 0.0f;
};

std::optional<Sample> sample_patch(const Patch& patch, Vec2 metric)
{
    double col_coordinate = metric.x * patch.scale_col;
    double row_coordinate = metric.y * patch.scale_row;
    const double max_col = static_cast<double>(patch.surface->cols - 1);
    const double max_row = static_cast<double>(patch.surface->rows - 1);
    const double epsilon = 1e-8;
    if (col_coordinate < -epsilon || row_coordinate < -epsilon
        || col_coordinate > max_col + epsilon || row_coordinate > max_row + epsilon) {
        return std::nullopt;
    }
    col_coordinate = std::clamp(col_coordinate, 0.0, max_col);
    row_coordinate = std::clamp(row_coordinate, 0.0, max_row);
    std::size_t col = static_cast<std::size_t>(std::floor(col_coordinate));
    std::size_t row = static_cast<std::size_t>(std::floor(row_coordinate));
    double col_fraction = col_coordinate - col;
    double row_fraction = row_coordinate - row;
    if (col + 1 >= patch.surface->cols) {
        if (col == 0) return std::nullopt;
        --col;
        col_fraction = 1.0;
    }
    if (row + 1 >= patch.surface->rows) {
        if (row == 0) return std::nullopt;
        --row;
        row_fraction = 1.0;
    }
    if (!patch.surface->valid_quad(row, col)) return std::nullopt;
    const Vec3 p00 = patch.surface->at(row, col);
    const Vec3 p10 = patch.surface->at(row, col + 1);
    const Vec3 p01 = patch.surface->at(row + 1, col);
    const Vec3 p11 = patch.surface->at(row + 1, col + 1);
    const auto interpolate = [&](double a, double b, double c, double d) {
        return (1.0 - row_fraction)
                * ((1.0 - col_fraction) * a + col_fraction * b)
            + row_fraction * ((1.0 - col_fraction) * c + col_fraction * d);
    };
    const std::size_t cols = patch.surface->cols;
    const auto derivative = [](const Vec3& a, const Vec3& b, double scale) {
        return Vec3{
            static_cast<float>((b.x - a.x) * scale),
            static_cast<float>((b.y - a.y) * scale),
            static_cast<float>((b.z - a.z) * scale),
        };
    };
    const Vec3 col_low = derivative(p00, p10, patch.scale_col);
    const Vec3 col_high = derivative(p01, p11, patch.scale_col);
    const Vec3 row_low = derivative(p00, p01, patch.scale_row);
    const Vec3 row_high = derivative(p10, p11, patch.scale_row);
    return Sample{
        {
            static_cast<float>(interpolate(p00.x, p10.x, p01.x, p11.x)),
            static_cast<float>(interpolate(p00.y, p10.y, p01.y, p11.y)),
            static_cast<float>(interpolate(p00.z, p10.z, p01.z, p11.z)),
        },
        {
            static_cast<float>((1.0 - row_fraction) * col_low.x
                               + row_fraction * col_high.x),
            static_cast<float>((1.0 - row_fraction) * col_low.y
                               + row_fraction * col_high.y),
            static_cast<float>((1.0 - row_fraction) * col_low.z
                               + row_fraction * col_high.z),
        },
        {
            static_cast<float>((1.0 - col_fraction) * row_low.x
                               + col_fraction * row_high.x),
            static_cast<float>((1.0 - col_fraction) * row_low.y
                               + col_fraction * row_high.y),
            static_cast<float>((1.0 - col_fraction) * row_low.z
                               + col_fraction * row_high.z),
        },
        static_cast<float>(interpolate(
            patch.boundary_distance_squared[row * cols + col],
            patch.boundary_distance_squared[row * cols + col + 1],
            patch.boundary_distance_squared[(row + 1) * cols + col],
            patch.boundary_distance_squared[(row + 1) * cols + col + 1])),
    };
}

struct Correspondence {
    Vec2 target;
    Vec2 seed;
    Vec3 target_tangent_u;
    Vec3 target_tangent_v;
    Vec3 seed_tangent_u;
    Vec3 seed_tangent_v;
    float surface_distance = 0.0f;
    std::uint64_t cell = 0;
};

double tangent_mismatch_squared(const Pose& pose, const Correspondence& value)
{
    const auto normalized_difference_squared = [](const Vec3& a, const Vec3& b) {
        const double a_norm = std::sqrt(
            static_cast<double>(a.x) * a.x + static_cast<double>(a.y) * a.y
            + static_cast<double>(a.z) * a.z);
        const double b_norm = std::sqrt(
            static_cast<double>(b.x) * b.x + static_cast<double>(b.y) * b.y
            + static_cast<double>(b.z) * b.z);
        if (!(a_norm > 1e-8) || !(b_norm > 1e-8)) return 0.0;
        const double dx = a.x / a_norm - b.x / b_norm;
        const double dy = a.y / a_norm - b.y / b_norm;
        const double dz = a.z / a_norm - b.z / b_norm;
        return dx * dx + dy * dy + dz * dz;
    };
    // target(t) == seed(R t + translation), hence J_target == J_seed R.
    const Vec3 predicted_u{
        static_cast<float>(pose.r00 * value.seed_tangent_u.x
                           + pose.r10 * value.seed_tangent_v.x),
        static_cast<float>(pose.r00 * value.seed_tangent_u.y
                           + pose.r10 * value.seed_tangent_v.y),
        static_cast<float>(pose.r00 * value.seed_tangent_u.z
                           + pose.r10 * value.seed_tangent_v.z),
    };
    const Vec3 predicted_v{
        static_cast<float>(pose.r01 * value.seed_tangent_u.x
                           + pose.r11 * value.seed_tangent_v.x),
        static_cast<float>(pose.r01 * value.seed_tangent_u.y
                           + pose.r11 * value.seed_tangent_v.y),
        static_cast<float>(pose.r01 * value.seed_tangent_u.z
                           + pose.r11 * value.seed_tangent_v.z),
    };
    return normalized_difference_squared(value.target_tangent_u, predicted_u)
        + normalized_difference_squared(value.target_tangent_v, predicted_v);
}

std::uint64_t splitmix64(std::uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

class CorrespondenceReservoir {
public:
    explicit CorrespondenceReservoir(std::size_t capacity) : capacity_(capacity) {}

    bool add(Correspondence value)
    {
        auto found = values_.find(value.cell);
        if (found != values_.end()) {
            if (value.surface_distance < found->second.surface_distance) {
                found->second = value;
                return true;
            }
            return false;
        }
        const std::uint64_t priority = splitmix64(value.cell);
        if (values_.size() >= capacity_) {
            while (!priorities_.empty()
                   && values_.find(priorities_.top().second) == values_.end()) {
                priorities_.pop();
            }
            if (!priorities_.empty()
                && std::pair{priority, value.cell} >= priorities_.top()) {
                return false;
            }
            values_.erase(priorities_.top().second);
            priorities_.pop();
        }
        priorities_.push({priority, value.cell});
        values_.emplace(value.cell, value);
        return true;
    }

    std::vector<Correspondence> sorted_values() const
    {
        std::vector<Correspondence> result;
        result.reserve(values_.size());
        for (const auto& [cell, value] : values_) result.push_back(value);
        std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
            return a.cell < b.cell;
        });
        return result;
    }

    std::size_t size() const noexcept { return values_.size(); }

private:
    std::size_t capacity_;
    std::unordered_map<std::uint64_t, Correspondence> values_;
    std::priority_queue<std::pair<std::uint64_t, std::uint64_t>> priorities_;
};

std::uint64_t thinning_cell(Vec2 uv, double spacing)
{
    const auto col = static_cast<std::uint32_t>(
        static_cast<std::int32_t>(std::floor(uv.x / spacing)));
    const auto row = static_cast<std::uint32_t>(
        static_cast<std::int32_t>(std::floor(uv.y / spacing)));
    return (static_cast<std::uint64_t>(row) << 32) | col;
}

std::uint64_t stable_hash(const std::string& value)
{
    std::uint64_t result = 1469598103934665603ULL;
    for (unsigned char character : value) {
        result ^= character;
        result *= 1099511628211ULL;
    }
    return result;
}

Pose two_point_pose(
    const Correspondence& a, const Correspondence& b, bool reflection,
    bool& valid)
{
    const Vec2 source_delta = b.target - a.target;
    const Vec2 target_delta = b.seed - a.seed;
    if (norm_squared(source_delta) < 1e-12 || norm_squared(target_delta) < 1e-12) {
        valid = false;
        return {};
    }
    const double source_angle = std::atan2(source_delta.y, source_delta.x);
    const double target_angle = std::atan2(target_delta.y, target_delta.x);
    const double angle = reflection ? target_angle + source_angle
                                    : target_angle - source_angle;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    Pose result;
    if (reflection) {
        result.r00 = cosine;
        result.r01 = sine;
        result.r10 = sine;
        result.r11 = -cosine;
    } else {
        result.r00 = cosine;
        result.r01 = -sine;
        result.r10 = sine;
        result.r11 = cosine;
    }
    const Vec2 source_middle = (a.target + b.target) * 0.5;
    const Vec2 target_middle = (a.seed + b.seed) * 0.5;
    const Vec2 rotated = result.apply(source_middle);
    result.tx += target_middle.x - rotated.x;
    result.ty += target_middle.y - rotated.y;
    valid = true;
    return result;
}

Pose weighted_pose(
    const std::vector<Correspondence>& values,
    const std::vector<std::uint8_t>& included,
    const std::vector<double>& weights,
    bool reflection)
{
    double total = 0.0;
    Vec2 source_mean, target_mean;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!included[index]) continue;
        const double weight = weights.empty() ? 1.0 : weights[index];
        total += weight;
        source_mean = source_mean + values[index].target * weight;
        target_mean = target_mean + values[index].seed * weight;
    }
    if (!(total > 0.0)) throw std::runtime_error("empty pose refit");
    source_mean = source_mean / total;
    target_mean = target_mean / total;
    double cosine_term = 0.0;
    double sine_term = 0.0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!included[index]) continue;
        const double weight = weights.empty() ? 1.0 : weights[index];
        const Vec2 source = values[index].target - source_mean;
        const Vec2 target = values[index].seed - target_mean;
        if (reflection) {
            cosine_term += weight * (target.x * source.x - target.y * source.y);
            sine_term += weight * (target.x * source.y + target.y * source.x);
        } else {
            cosine_term += weight * (target.x * source.x + target.y * source.y);
            sine_term += weight * (target.y * source.x - target.x * source.y);
        }
    }
    const double angle = std::atan2(sine_term, cosine_term);
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    Pose result;
    if (reflection) {
        result.r00 = cosine;
        result.r01 = sine;
        result.r10 = sine;
        result.r11 = -cosine;
    } else {
        result.r00 = cosine;
        result.r01 = -sine;
        result.r10 = sine;
        result.r11 = cosine;
    }
    const Vec2 rotated = result.apply(source_mean);
    result.tx += target_mean.x - rotated.x;
    result.ty += target_mean.y - rotated.y;
    return result;
}

struct FitResult {
    bool accepted = false;
    Pose pose;
    std::size_t inliers = 0;
    double rms = std::numeric_limits<double>::infinity();
    double major_spread = 0.0;
    double minor_spread = 0.0;
    std::string rejection;
};

struct AxisSpreads {
    double major = 0.0;
    double minor = 0.0;
};

AxisSpreads spreads(
    const std::vector<Correspondence>& values,
    const std::vector<std::uint8_t>& included)
{
    Vec2 mean;
    std::size_t count = 0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (included[index]) {
            mean = mean + values[index].seed;
            ++count;
        }
    }
    mean = mean / static_cast<double>(count);
    double xx = 0.0, xy = 0.0, yy = 0.0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!included[index]) continue;
        const Vec2 delta = values[index].seed - mean;
        xx += delta.x * delta.x;
        xy += delta.x * delta.y;
        yy += delta.y * delta.y;
    }
    const double angle = 0.5 * std::atan2(2.0 * xy, xx - yy);
    const Vec2 major{std::cos(angle), std::sin(angle)};
    const Vec2 minor{-major.y, major.x};
    double major_low = std::numeric_limits<double>::infinity();
    double major_high = -major_low;
    double minor_low = major_low;
    double minor_high = -major_low;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!included[index]) continue;
        const double a = dot(values[index].seed, major);
        const double b = dot(values[index].seed, minor);
        major_low = std::min(major_low, a);
        major_high = std::max(major_high, a);
        minor_low = std::min(minor_low, b);
        minor_high = std::max(minor_high, b);
    }
    double first = major_high - major_low;
    double second = minor_high - minor_low;
    if (second > first) std::swap(first, second);
    return {first, second};
}

FitResult fit_pair(
    const std::vector<Correspondence>& values, const MergeOptions& options,
    std::uint64_t random_seed)
{
    FitResult result;
    if (values.size() < options.min_inliers) {
        result.rejection = "insufficient_correspondences";
        return result;
    }
    const double threshold_squared
        = options.uv_inlier_tolerance * options.uv_inlier_tolerance;
    // A wrong reflection can fit a thin overlap in UV while sending the rest
    // of the patch back across the seed. Local 3D tangents disambiguate it.
    constexpr double tangent_threshold_squared = 1.0;
    std::size_t best_count = 0;
    double best_error = std::numeric_limits<double>::infinity();
    Pose best_pose;
    std::size_t hypothesis_limit = options.ransac_max_hypotheses;
    std::size_t hypothesis_count = 0;
    std::uint64_t state = random_seed;
    while (hypothesis_count < hypothesis_limit) {
        state = splitmix64(state);
        std::size_t first = state % values.size();
        state = splitmix64(state);
        std::size_t second = state % (values.size() - 1);
        if (second >= first) ++second;
        for (int reflected = 0;
             reflected <= (options.allow_reflection ? 1 : 0)
             && hypothesis_count < hypothesis_limit;
             ++reflected, ++hypothesis_count) {
            bool valid = false;
            const Pose pose = two_point_pose(
                values[first], values[second], reflected != 0, valid);
            if (!valid) continue;
            std::size_t count = 0;
            double error = 0.0;
            for (const auto& value : values) {
                const double squared = norm_squared(pose.apply(value.target) - value.seed);
                if (squared <= threshold_squared
                    && tangent_mismatch_squared(pose, value)
                        <= tangent_threshold_squared) {
                    ++count;
                    error += squared;
                }
            }
            if (count > best_count || (count == best_count && error < best_error)) {
                best_count = count;
                best_error = error;
                best_pose = pose;
                const double ratio = static_cast<double>(count) / values.size();
                const double success = ratio * ratio;
                if (success >= 1.0) {
                    hypothesis_limit = std::min(hypothesis_limit, hypothesis_count + 1);
                } else if (success > 0.0) {
                    const double needed = std::ceil(
                        std::log(1.0 - options.ransac_confidence)
                        / std::log(1.0 - success));
                    hypothesis_limit = std::min(
                        hypothesis_limit,
                        std::max(hypothesis_count + 1,
                                 static_cast<std::size_t>(needed)));
                }
            }
        }
    }
    if (best_count < options.min_inliers) {
        result.rejection = "insufficient_inliers";
        result.inliers = best_count;
        return result;
    }

    std::vector<std::uint8_t> included(values.size(), 0);
    for (std::size_t index = 0; index < values.size(); ++index) {
        included[index] = norm_squared(best_pose.apply(values[index].target)
                                       - values[index].seed) <= threshold_squared
            && tangent_mismatch_squared(best_pose, values[index])
                <= tangent_threshold_squared;
    }
    std::vector<double> weights(values.size(), 0.0);
    Pose refitted = best_pose;
    const bool reflection = best_pose.reflected();
    const double huber_delta = std::min(1.0, options.uv_inlier_tolerance);
    for (int iteration = 0; iteration < 6; ++iteration) {
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (!included[index]) continue;
            const double residual = std::sqrt(norm_squared(
                refitted.apply(values[index].target) - values[index].seed));
            weights[index] = residual <= huber_delta || residual == 0.0
                ? 1.0 : huber_delta / residual;
        }
        refitted = weighted_pose(values, included, weights, reflection);
    }
    std::size_t final_count = 0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        included[index] = norm_squared(refitted.apply(values[index].target)
                                       - values[index].seed) <= threshold_squared
            && tangent_mismatch_squared(refitted, values[index])
                <= tangent_threshold_squared;
        final_count += included[index] ? 1 : 0;
    }
    if (final_count < options.min_inliers) {
        result.rejection = "insufficient_inliers_after_refit";
        result.inliers = final_count;
        return result;
    }
    for (int iteration = 0; iteration < 3; ++iteration) {
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (!included[index]) {
                weights[index] = 0.0;
                continue;
            }
            const double residual = std::sqrt(norm_squared(
                refitted.apply(values[index].target) - values[index].seed));
            weights[index] = residual <= huber_delta || residual == 0.0
                ? 1.0 : huber_delta / residual;
        }
        refitted = weighted_pose(values, included, weights, reflection);
    }
    double squared_error = 0.0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (included[index]) {
            squared_error += norm_squared(
                refitted.apply(values[index].target) - values[index].seed);
        }
    }
    result.rms = std::sqrt(squared_error / final_count);
    result.inliers = final_count;
    result.pose = refitted;
    const AxisSpreads inlier_spreads = spreads(values, included);
    result.major_spread = inlier_spreads.major;
    result.minor_spread = inlier_spreads.minor;
    if (result.major_spread + 1e-9 < options.min_major_spread) {
        result.rejection = "insufficient_major_axis_spread";
        return result;
    }
    if (result.minor_spread + 1e-9 < options.min_minor_spread) {
        result.rejection = "insufficient_minor_axis_spread";
        return result;
    }
    if (result.rms > options.max_refit_rms) {
        result.rejection = "refit_rms";
        return result;
    }
    result.accepted = true;
    return result;
}

struct OverlapConsistency {
    std::size_t samples = 0;
    std::size_t agreements = 0;
};

Pose refine_pose_in_3d(
    const Patch& seed, const Patch& target, Pose pose)
{
    const bool reflection = pose.reflected();
    const std::size_t vertex_count = target.surface->rows * target.surface->cols;
    const std::size_t stride = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::sqrt(
               static_cast<double>(vertex_count) / 1024.0)));
    for (int iteration = 0; iteration < 10; ++iteration) {
        double normal[3][3]{};
        double gradient[3]{};
        std::size_t included = 0;
        for (std::size_t row = 0; row + 1 < target.surface->rows; row += stride) {
            for (std::size_t col = 0; col + 1 < target.surface->cols; col += stride) {
                if (!target.surface->valid_quad(row, col)) continue;
                const Vec2 target_uv{
                    static_cast<double>(col) / target.scale_col,
                    static_cast<double>(row) / target.scale_row,
                };
                const auto target_sample = sample_patch(target, target_uv);
                const auto seed_sample = sample_patch(seed, pose.apply(target_uv));
                if (!target_sample || !seed_sample) continue;
                const Vec3 residual = seed_sample->xyz - target_sample->xyz;
                const double residual_squared
                    = static_cast<double>(residual.x) * residual.x
                    + static_cast<double>(residual.y) * residual.y
                    + static_cast<double>(residual.z) * residual.z;
                // Grossly conflicting sheets are negative evidence for the
                // pose, not observations that should pull the optimizer.
                if (residual_squared > 100.0) continue;
                const double residual_norm = std::sqrt(residual_squared);
                const double weight = residual_norm <= 2.0 || residual_norm == 0.0
                    ? 1.0 : 2.0 / residual_norm;
                const double dsx = reflection
                    ? -pose.r10 * target_uv.x + pose.r00 * target_uv.y
                    : -pose.r10 * target_uv.x - pose.r00 * target_uv.y;
                const double dsy = reflection
                    ? pose.r00 * target_uv.x + pose.r10 * target_uv.y
                    : pose.r00 * target_uv.x - pose.r10 * target_uv.y;
                const Vec3 jacobian[3]{
                    {
                        static_cast<float>(dsx * seed_sample->tangent_u.x
                                           + dsy * seed_sample->tangent_v.x),
                        static_cast<float>(dsx * seed_sample->tangent_u.y
                                           + dsy * seed_sample->tangent_v.y),
                        static_cast<float>(dsx * seed_sample->tangent_u.z
                                           + dsy * seed_sample->tangent_v.z),
                    },
                    seed_sample->tangent_u,
                    seed_sample->tangent_v,
                };
                for (int a = 0; a < 3; ++a) {
                    const double ja_dot_residual
                        = static_cast<double>(jacobian[a].x) * residual.x
                        + static_cast<double>(jacobian[a].y) * residual.y
                        + static_cast<double>(jacobian[a].z) * residual.z;
                    gradient[a] += weight * ja_dot_residual;
                    for (int b = 0; b < 3; ++b) {
                        normal[a][b] += weight * (
                            static_cast<double>(jacobian[a].x) * jacobian[b].x
                            + static_cast<double>(jacobian[a].y) * jacobian[b].y
                            + static_cast<double>(jacobian[a].z) * jacobian[b].z);
                    }
                }
                ++included;
            }
        }
        if (included < 16) break;
        const double determinant
            = normal[1][1] * normal[2][2] - normal[1][2] * normal[2][1];
        if (std::abs(determinant) < 1e-10) break;
        const double delta_x = std::clamp(
            (-gradient[1] * normal[2][2]
             + gradient[2] * normal[1][2]) / determinant,
            -5.0, 5.0);
        const double delta_y = std::clamp(
            (-gradient[2] * normal[1][1]
             + gradient[1] * normal[2][1]) / determinant,
            -5.0, 5.0);
        pose.tx += delta_x;
        pose.ty += delta_y;
        if (std::abs(delta_x) < 1e-5 && std::abs(delta_y) < 1e-5) {
            break;
        }
    }
    return pose;
}

OverlapConsistency overlap_consistency(
    const Patch& seed, const Patch& target, const Pose& target_to_seed,
    const MergeOptions& options)
{
    OverlapConsistency result;
    const std::size_t vertex_count = seed.surface->rows * seed.surface->cols;
    const std::size_t stride = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::sqrt(
               static_cast<double>(vertex_count) / 512.0)));
    const Pose seed_to_target = target_to_seed.inverse();
    const double agreement_distance = options.tolerance
        + options.uv_inlier_tolerance;
    const double agreement_squared = agreement_distance * agreement_distance;
    for (std::size_t row = 0; row + 1 < seed.surface->rows; row += stride) {
        for (std::size_t col = 0; col + 1 < seed.surface->cols; col += stride) {
            if (!seed.surface->valid_quad(row, col)) continue;
            const Vec2 seed_uv{
                static_cast<double>(col) / seed.scale_col,
                static_cast<double>(row) / seed.scale_row,
            };
            const auto seed_sample = sample_patch(seed, seed_uv);
            const auto target_sample = sample_patch(
                target, seed_to_target.apply(seed_uv));
            if (!seed_sample || !target_sample) continue;
            ++result.samples;
            if (surfcore::distance_squared(
                    seed_sample->xyz, target_sample->xyz)
                <= agreement_squared) {
                ++result.agreements;
            }
        }
    }
    return result;
}

FitResult fit_consistent_pair(
    const std::vector<Correspondence>& values, const Patch& seed,
    const Patch& target, const MergeOptions& options, std::uint64_t random_seed)
{
    std::vector<Correspondence> remaining = values;
    FitResult conflicting;
    constexpr int maximum_modes = 4;
    for (int mode = 0; mode < maximum_modes; ++mode) {
        FitResult fit = fit_pair(
            remaining, options, random_seed + static_cast<std::uint64_t>(mode));
        if (!fit.accepted) {
            return conflicting.rejection.empty() ? fit : conflicting;
        }
        fit.pose = refine_pose_in_3d(seed, target, fit.pose);
        const OverlapConsistency consistency = overlap_consistency(
            seed, target, fit.pose, options);
        if (consistency.samples < options.min_inliers
            || consistency.agreements * 5 >= consistency.samples * 3) {
            return fit;
        }

        conflicting = fit;
        conflicting.accepted = false;
        conflicting.rejection = "overlap_conflict";
        const double threshold_squared
            = options.uv_inlier_tolerance * options.uv_inlier_tolerance;
        remaining.erase(std::remove_if(
            remaining.begin(), remaining.end(), [&](const Correspondence& value) {
                return norm_squared(fit.pose.apply(value.target) - value.seed)
                    <= threshold_squared;
            }), remaining.end());
        if (remaining.size() < options.min_inliers) break;
    }
    return conflicting;
}

struct PairResult {
    std::size_t high = 0;
    std::size_t low = 0;
    std::size_t correspondences = 0;
    FitResult fit;
};

struct PairWork {
    explicit PairWork(std::size_t capacity) : reservoir(capacity) {}

    CorrespondenceReservoir reservoir;
    std::vector<SurfaceTileCandidate> tile_pairs;
    std::vector<SurfaceTileCandidate> hit_tile_pairs;
    FitResult fit;
    std::size_t exact_correspondences = 0;
    bool exact_attempted = false;
};

struct HarvestCounters {
    std::atomic<std::size_t> completed_seeds{0};
    std::atomic<std::size_t> bbox_pairs{0};
    std::atomic<std::size_t> tile_pairs{0};
    std::atomic<std::size_t> vertex_samples{0};
    std::atomic<std::size_t> triangle_hits{0};
    std::atomic<std::size_t> dense_pairs{0};
    std::atomic<std::size_t> dense_samples{0};
    std::atomic<std::size_t> exact_samples{0};
};

struct LocalHarvestCounters {
    std::size_t bbox_pairs = 0;
    std::size_t tile_pairs = 0;
    std::size_t vertex_samples = 0;
    std::size_t triangle_hits = 0;
    std::size_t dense_pairs = 0;
    std::size_t dense_samples = 0;
    std::size_t exact_samples = 0;
};

struct HarvestCounterCommit {
    HarvestCounters* totals = nullptr;
    LocalHarvestCounters& local;

    ~HarvestCounterCommit()
    {
        if (!totals) return;
        totals->bbox_pairs.fetch_add(local.bbox_pairs, std::memory_order_relaxed);
        totals->tile_pairs.fetch_add(local.tile_pairs, std::memory_order_relaxed);
        totals->vertex_samples.fetch_add(local.vertex_samples, std::memory_order_relaxed);
        totals->triangle_hits.fetch_add(local.triangle_hits, std::memory_order_relaxed);
        totals->dense_pairs.fetch_add(local.dense_pairs, std::memory_order_relaxed);
        totals->dense_samples.fetch_add(local.dense_samples, std::memory_order_relaxed);
        totals->exact_samples.fetch_add(local.exact_samples, std::memory_order_relaxed);
    }
};

bool vertex_touches_valid_quad(const SurfaceData& surface, size_t row, size_t col)
{
    const size_t row_begin = row == 0 ? 0 : row - 1;
    const size_t col_begin = col == 0 ? 0 : col - 1;
    const size_t row_end = std::min(row, surface.rows - 2);
    const size_t col_end = std::min(col, surface.cols - 2);
    for (size_t quad_row = row_begin; quad_row <= row_end; ++quad_row) {
        for (size_t quad_col = col_begin; quad_col <= col_end; ++quad_col) {
            if (surface.valid_quad(quad_row, quad_col)) return true;
        }
    }
    return false;
}

std::vector<PairResult> harvest_and_fit(
    std::size_t high, const std::vector<Patch>& patches,
    const SurfacePatchIndex& index, const MergeOptions& options,
    HarvestCounters* counters)
{
    LocalHarvestCounters local;
    HarvestCounterCommit commit{counters, local};
    QueryScratch scratch;
    std::vector<int32_t> broad_surfaces;
    index.query_surface_candidates(
        static_cast<int32_t>(high), static_cast<float>(options.tolerance),
        static_cast<int32_t>(high + 1), broad_surfaces);
    local.bbox_pairs += broad_surfaces.size();
    if (broad_surfaces.empty()) return {};

    std::vector<uint8_t> included(patches.size(), 0);
    for (const int32_t target : broad_surfaces) {
        included[static_cast<size_t>(target)] = 1;
    }

    const std::size_t landmark_capacity = std::min(
        options.max_correspondences,
        std::max<std::size_t>(64, options.min_inliers * 4));
    std::map<std::size_t, PairWork> pairs;
    std::vector<SurfaceTileCandidate> tile_candidates;
    index.query_surface_tile_candidates(
        static_cast<int32_t>(high), static_cast<float>(options.tolerance),
        static_cast<int32_t>(high + 1), tile_candidates, scratch, -1, &included);
    local.tile_pairs += tile_candidates.size();
    for (const SurfaceTileCandidate& candidate : tile_candidates) {
        auto [found, inserted] = pairs.try_emplace(
            static_cast<std::size_t>(candidate.surface), landmark_capacity);
        found->second.tile_pairs.push_back(candidate);
    }
    if (pairs.empty()) return {};

    std::vector<SurfaceHit> hits;
    const Patch& seed = patches[high];

    const auto record_hit = [&](CorrespondenceReservoir& reservoir,
                                std::size_t low, const Vec2& seed_uv,
                                const SurfaceHit& hit) {
        const Vec2 target_uv{
            static_cast<double>(hit.i) / patches[low].scale_col,
            static_cast<double>(hit.j) / patches[low].scale_row,
        };
        const auto seed_sample = sample_patch(seed, seed_uv);
        const auto target_sample = sample_patch(patches[low], target_uv);
        if (!seed_sample || !target_sample) return false;
        return reservoir.add({
            target_uv,
            seed_uv,
            target_sample->tangent_u,
            target_sample->tangent_v,
            seed_sample->tangent_u,
            seed_sample->tangent_v,
            hit.distance,
            thinning_cell(seed_uv, options.thinning_spacing),
        });
    };

    // Each bbox-approved pair gets a small, spatially distributed landmark
    // set. The nine passes visit the center, corners, then quarter points of
    // every intersecting seed tile. A pair stops as soon as its bounded
    // reservoir is full; RANSAC never sees the dense patch point cloud.
    constexpr std::array<std::array<double, 2>, 9> landmark_pattern{{
        {0.5, 0.5},
        {0.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {1.0, 0.0},
        {0.25, 0.25}, {0.75, 0.75}, {0.25, 0.75}, {0.75, 0.25},
    }};
    for (auto& [low, work] : pairs) {
        std::sort(work.tile_pairs.begin(), work.tile_pairs.end(), [&](const auto& a,
                                                                      const auto& b) {
            const auto priority = [&](const SurfaceTileCandidate& value) {
                return splitmix64(
                    (static_cast<std::uint64_t>(value.seed_tile) << 32)
                    ^ static_cast<std::uint64_t>(value.target_tile)
                    ^ stable_hash(seed.id) ^ (stable_hash(patches[low].id) << 1));
            };
            const std::uint64_t pa = priority(a);
            const std::uint64_t pb = priority(b);
            if (pa != pb) return pa < pb;
            if (a.seed_tile != b.seed_tile) return a.seed_tile < b.seed_tile;
            return a.target_tile < b.target_tile;
        });
        for (const auto& fraction : landmark_pattern) {
            for (const SurfaceTileCandidate& candidate : work.tile_pairs) {
                if (work.reservoir.size() >= landmark_capacity) break;
                const int row_end = std::min(
                    candidate.seed_row + index.tile_stride(),
                    static_cast<int>(seed.surface->rows) - 1);
                const int col_end = std::min(
                    candidate.seed_col + index.tile_stride(),
                    static_cast<int>(seed.surface->cols) - 1);
                const double grid_col = static_cast<double>(candidate.seed_col)
                    + fraction[0] * static_cast<double>(col_end - candidate.seed_col);
                const double grid_row = static_cast<double>(candidate.seed_row)
                    + fraction[1] * static_cast<double>(row_end - candidate.seed_row);
                const Vec2 seed_uv{
                    grid_col / seed.scale_col,
                    grid_row / seed.scale_row,
                };
                const auto sample = sample_patch(seed, seed_uv);
                if (!sample) continue;
                ++local.vertex_samples;
                SurfaceHit hit;
                if (!index.query_tile_point(
                        candidate, sample->xyz,
                        static_cast<float>(options.tolerance), false, hit)) {
                    continue;
                }
                ++local.triangle_hits;
                if (record_hit(work.reservoir, low, seed_uv, hit)) {
                    work.hit_tile_pairs.push_back(candidate);
                }
            }
            // The center pass is the cheap geometry gate. A candidate pair
            // with no center landmark does not earn eight more probes per
            // tile (or a dense rescue); broad overlaps produce many center
            // hits, while bbox-only neighbors disappear here.
            if (work.reservoir.size() == 0) break;
            if (work.reservoir.size() >= landmark_capacity) break;
        }
    }

    const auto preliminary_fit = [&](std::size_t low, PairWork& work) {
        const auto values = work.reservoir.sorted_values();
        return fit_consistent_pair(
            values, seed, patches[low], options,
            stable_hash(seed.id) ^ (stable_hash(patches[low].id) << 1));
    };

    const auto exact_fit = [&](std::size_t low, PairWork& work) {
        const std::size_t validation_capacity = std::min(
            landmark_capacity,
            std::max<std::size_t>(options.min_inliers + 16,
                                  options.min_inliers));
        CorrespondenceReservoir preliminary_inliers(validation_capacity);
        const double threshold_squared
            = options.uv_inlier_tolerance * options.uv_inlier_tolerance;
        for (const Correspondence& approximate : work.reservoir.sorted_values()) {
            if (norm_squared(work.fit.pose.apply(approximate.target)
                             - approximate.seed) <= threshold_squared) {
                preliminary_inliers.add(approximate);
            }
        }
        CorrespondenceReservoir exact(validation_capacity);
        for (const Correspondence& approximate
             : preliminary_inliers.sorted_values()) {
            const auto sample = sample_patch(seed, approximate.seed);
            if (!sample) continue;
            ++local.exact_samples;
            hits.clear();
            index.query_point(
                sample->xyz, static_cast<float>(options.tolerance), hits, scratch,
                nullptr, 0, static_cast<int32_t>(low), true);
            if (!hits.empty()) {
                record_hit(exact, low, approximate.seed, hits.front());
            }
        }
        const auto values = exact.sorted_values();
        work.exact_attempted = true;
        work.exact_correspondences = values.size();
        work.fit = fit_consistent_pair(
            values, seed, patches[low], options,
            stable_hash(seed.id) ^ (stable_hash(patches[low].id) << 1));
    };

    for (auto& [low, work] : pairs) {
        work.fit = preliminary_fit(low, work);
        if (work.fit.accepted) exact_fit(low, work);
    }

    // Only unresolved broad-phase pairs are densified, and only inside seed
    // tiles whose padded AABBs overlap that target. Progressive spacing lets
    // ordinary overlaps finish at native vertices or 10-vx refinement while
    // retaining the requested 1-vx coverage for thin, sub-cell overlaps.
    const double native_spacing = std::max(
        1.0 / seed.scale_col, 1.0 / seed.scale_row);
    for (auto& [low, work] : pairs) {
        const std::size_t rescue_landmarks = std::max<std::size_t>(
            4, options.min_inliers / 4);
        if (work.fit.accepted || work.reservoir.size() < rescue_landmarks) continue;
        ++local.dense_pairs;
        std::vector<SurfaceTileCandidate> dense_tiles = work.hit_tile_pairs;
        std::sort(dense_tiles.begin(), dense_tiles.end(), [&](const auto& a, const auto& b) {
            const auto priority = [&](const SurfaceTileCandidate& value) {
                return splitmix64(
                    (static_cast<std::uint64_t>(value.seed_tile) << 32)
                    ^ static_cast<std::uint64_t>(value.target_tile)
                    ^ stable_hash(seed.id) ^ (stable_hash(patches[low].id) << 1));
            };
            const std::uint64_t pa = priority(a);
            const std::uint64_t pb = priority(b);
            if (pa != pb) return pa < pb;
            if (a.seed_tile != b.seed_tile) return a.seed_tile < b.seed_tile;
            return a.target_tile < b.target_tile;
        });
        dense_tiles.erase(std::unique(
            dense_tiles.begin(), dense_tiles.end(), [](const auto& a, const auto& b) {
                return a.seed_tile == b.seed_tile && a.target_tile == b.target_tile;
            }), dense_tiles.end());
        const std::size_t level_sample_budget = std::max<std::size_t>(
            2048, options.min_inliers * 16);
        double spacing = std::max(options.dense_spacing, native_spacing * 0.5);
        while (true) {
            bool accepted_during_batch = false;
            bool level_budget_reached = false;
            std::size_t processed_tiles = 0;
            std::size_t level_samples = 0;
            for (const SurfaceTileCandidate& tile : dense_tiles) {
                const int row_end = std::min(
                    tile.seed_row + index.tile_stride(),
                    static_cast<int>(seed.surface->rows) - 1);
                const int col_end = std::min(
                    tile.seed_col + index.tile_stride(),
                    static_cast<int>(seed.surface->cols) - 1);
                const double minimum_u = static_cast<double>(tile.seed_col) / seed.scale_col;
                const double maximum_u = static_cast<double>(col_end) / seed.scale_col;
                const double minimum_v = static_cast<double>(tile.seed_row) / seed.scale_row;
                const double maximum_v = static_cast<double>(row_end) / seed.scale_row;
                const auto first_u = static_cast<std::int64_t>(
                    std::ceil(minimum_u / spacing - 1e-9));
                const auto last_u = static_cast<std::int64_t>(
                    std::floor(maximum_u / spacing + 1e-9));
                const auto first_v = static_cast<std::int64_t>(
                    std::ceil(minimum_v / spacing - 1e-9));
                const auto last_v = static_cast<std::int64_t>(
                    std::floor(maximum_v / spacing + 1e-9));
                for (std::int64_t v_index = first_v; v_index <= last_v; ++v_index) {
                    for (std::int64_t u_index = first_u; u_index <= last_u; ++u_index) {
                        if (level_samples >= level_sample_budget
                            || work.reservoir.size() >= landmark_capacity) {
                            level_budget_reached = true;
                            break;
                        }
                        const Vec2 seed_uv{
                            static_cast<double>(u_index) * spacing,
                            static_cast<double>(v_index) * spacing,
                        };
                        const auto sample = sample_patch(seed, seed_uv);
                        if (!sample) continue;
                        ++level_samples;
                        ++local.dense_samples;
                        SurfaceHit hit;
                        if (index.query_tile_point(
                                tile, sample->xyz,
                                static_cast<float>(options.tolerance), false, hit)) {
                            record_hit(work.reservoir, low, seed_uv, hit);
                        }
                    }
                    if (level_budget_reached) break;
                }
                if (level_budget_reached) break;
                ++processed_tiles;
                if (processed_tiles % 16 == 0
                    && work.reservoir.size() >= options.min_inliers) {
                    work.fit = preliminary_fit(low, work);
                    if (work.fit.accepted) exact_fit(low, work);
                    if (work.fit.accepted) {
                        accepted_during_batch = true;
                        break;
                    }
                }
            }
            if (!accepted_during_batch) {
                work.fit = preliminary_fit(low, work);
                if (work.fit.accepted) exact_fit(low, work);
            }
            if (work.fit.accepted
                || spacing <= options.dense_spacing * (1.0 + 1e-12)) {
                break;
            }
            spacing = std::max(options.dense_spacing, spacing * 0.5);
        }
    }

    std::vector<PairResult> results;
    results.reserve(pairs.size());
    for (const auto& [low, work] : pairs) {
        if (work.reservoir.size() == 0) continue;
        PairResult result;
        result.high = high;
        result.low = low;
        result.correspondences = work.exact_attempted
            ? work.exact_correspondences : work.reservoir.size();
        result.fit = work.fit;
        results.push_back(std::move(result));
    }
    return results;
}

json settings_json(const MergeOptions& options)
{
    return {
        {"tolerance", options.tolerance},
        {"dense_spacing", options.dense_spacing},
        {"erode_cells", options.erode_cells},
        {"output_step", options.output_step},
        {"thinning_spacing", options.thinning_spacing},
        {"max_correspondences", options.max_correspondences},
        {"uv_inlier_tolerance", options.uv_inlier_tolerance},
        {"min_inliers", options.min_inliers},
        {"min_major_spread", options.min_major_spread},
        {"min_minor_spread", options.min_minor_spread},
        {"max_refit_rms", options.max_refit_rms},
        {"containment_threshold", options.containment_threshold},
        {"ransac_confidence", options.ransac_confidence},
        {"ransac_max_hypotheses", options.ransac_max_hypotheses},
        {"allow_reflection", options.allow_reflection},
    };
}

json pose_json(
    const std::string& member, const Pose& pose, std::size_t inliers, double rms)
{
    return {
        {"member_id", member},
        {"matrix", {
            {pose.r00, pose.r01, pose.tx},
            {pose.r10, pose.r11, pose.ty},
            {0.0, 0.0, 1.0},
        }},
        {"reflected", pose.reflected()},
        {"inliers", inliers},
        {"rms", rms},
    };
}

struct Member {
    std::size_t patch = 0;
    Pose to_seed;
    std::size_t inliers = 0;
    double rms = 0.0;
};

Pose compose_pose(const Pose& outer, const Pose& inner)
{
    Pose result;
    result.r00 = outer.r00 * inner.r00 + outer.r01 * inner.r10;
    result.r01 = outer.r00 * inner.r01 + outer.r01 * inner.r11;
    result.r10 = outer.r10 * inner.r00 + outer.r11 * inner.r10;
    result.r11 = outer.r10 * inner.r01 + outer.r11 * inner.r11;
    result.tx = outer.r00 * inner.tx + outer.r01 * inner.ty + outer.tx;
    result.ty = outer.r10 * inner.tx + outer.r11 * inner.ty + outer.ty;
    return result;
}

OverlapConsistency transformed_overlap_consistency(
    const Patch& candidate, const Pose& candidate_to_seed,
    const Patch& placed, const Pose& placed_to_seed,
    const MergeOptions& options)
{
    OverlapConsistency result;
    const std::size_t vertex_count
        = candidate.surface->rows * candidate.surface->cols;
    const std::size_t stride = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::sqrt(
               static_cast<double>(vertex_count) / 512.0)));
    const Pose seed_to_placed = placed_to_seed.inverse();
    const double agreement_squared = options.tolerance * options.tolerance;
    for (std::size_t row = 0; row + 1 < candidate.surface->rows; row += stride) {
        for (std::size_t col = 0; col + 1 < candidate.surface->cols; col += stride) {
            if (!candidate.surface->valid_quad(row, col)) continue;
            const Vec2 candidate_uv{
                static_cast<double>(col) / candidate.scale_col,
                static_cast<double>(row) / candidate.scale_row,
            };
            const auto candidate_sample = sample_patch(candidate, candidate_uv);
            const auto placed_sample = sample_patch(
                placed, seed_to_placed.apply(
                    candidate_to_seed.apply(candidate_uv)));
            if (!candidate_sample || !placed_sample) continue;
            ++result.samples;
            if (surfcore::distance_squared(
                    candidate_sample->xyz, placed_sample->xyz)
                <= agreement_squared) {
                ++result.agreements;
            }
        }
    }
    return result;
}

bool pose_preserves_output_metric(
    const Patch& patch, const Pose& patch_to_seed)
{
    const std::size_t vertex_count = patch.surface->rows * patch.surface->cols;
    const std::size_t stride = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::sqrt(
               static_cast<double>(vertex_count) / 512.0)));
    std::size_t samples = 0;
    std::size_t consistent = 0;
    for (std::size_t row = 0; row + 1 < patch.surface->rows; row += stride) {
        for (std::size_t col = 0; col + 1 < patch.surface->cols; col += stride) {
            if (!patch.surface->valid_quad(row, col)) continue;
            const auto sample = sample_patch(patch, {
                static_cast<double>(col) / patch.scale_col,
                static_cast<double>(row) / patch.scale_row,
            });
            if (!sample) continue;
            // t = R^-1 s. These are the XYZ derivatives produced by one
            // metric unit along each output axis after placing the patch.
            const Vec3 output_u{
                static_cast<float>(patch_to_seed.r00 * sample->tangent_u.x
                                   + patch_to_seed.r01 * sample->tangent_v.x),
                static_cast<float>(patch_to_seed.r00 * sample->tangent_u.y
                                   + patch_to_seed.r01 * sample->tangent_v.y),
                static_cast<float>(patch_to_seed.r00 * sample->tangent_u.z
                                   + patch_to_seed.r01 * sample->tangent_v.z),
            };
            const Vec3 output_v{
                static_cast<float>(patch_to_seed.r10 * sample->tangent_u.x
                                   + patch_to_seed.r11 * sample->tangent_v.x),
                static_cast<float>(patch_to_seed.r10 * sample->tangent_u.y
                                   + patch_to_seed.r11 * sample->tangent_v.y),
                static_cast<float>(patch_to_seed.r10 * sample->tangent_u.z
                                   + patch_to_seed.r11 * sample->tangent_v.z),
            };
            const auto length = [](const Vec3& value) {
                return std::sqrt(
                    static_cast<double>(value.x) * value.x
                    + static_cast<double>(value.y) * value.y
                    + static_cast<double>(value.z) * value.z);
            };
            const double u_length = length(output_u);
            const double v_length = length(output_v);
            ++samples;
            if (u_length >= 0.75 && u_length <= 1.25
                && v_length >= 0.75 && v_length <= 1.25) {
                ++consistent;
            }
        }
    }
    return samples == 0 || consistent == samples;
}

std::array<double, 4> transformed_patch_bounds(
    const Patch& patch, const Pose& to_seed)
{
    const double maximum_u
        = static_cast<double>(patch.surface->cols - 1) / patch.scale_col;
    const double maximum_v
        = static_cast<double>(patch.surface->rows - 1) / patch.scale_row;
    std::array<double, 4> bounds{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    for (const Vec2 corner : std::array<Vec2, 4>{
             Vec2{0.0, 0.0}, Vec2{maximum_u, 0.0},
             Vec2{0.0, maximum_v}, Vec2{maximum_u, maximum_v}}) {
        const Vec2 value = to_seed.apply(corner);
        bounds[0] = std::min(bounds[0], value.x);
        bounds[1] = std::min(bounds[1], value.y);
        bounds[2] = std::max(bounds[2], value.x);
        bounds[3] = std::max(bounds[3], value.y);
    }
    return bounds;
}

bool transformed_seams_are_consistent(
    const Patch& candidate, const Pose& candidate_to_seed,
    const Patch& placed, const Pose& placed_to_seed,
    const MergeOptions& options)
{
    const auto candidate_bounds = transformed_patch_bounds(
        candidate, candidate_to_seed);
    const auto placed_bounds = transformed_patch_bounds(placed, placed_to_seed);
    const double low_u = std::max(
        candidate_bounds[0], placed_bounds[0] - options.output_step);
    const double low_v = std::max(
        candidate_bounds[1], placed_bounds[1] - options.output_step);
    const double high_u = std::min(
        candidate_bounds[2], placed_bounds[2] + options.output_step);
    const double high_v = std::min(
        candidate_bounds[3], placed_bounds[3] + options.output_step);
    if (low_u > high_u || low_v > high_v) return true;
    const auto first_u = static_cast<std::int64_t>(
        std::ceil(low_u / options.output_step - 1e-9));
    const auto last_u = static_cast<std::int64_t>(
        std::floor(high_u / options.output_step + 1e-9));
    const auto first_v = static_cast<std::int64_t>(
        std::ceil(low_v / options.output_step - 1e-9));
    const auto last_v = static_cast<std::int64_t>(
        std::floor(high_v / options.output_step + 1e-9));
    const Pose seed_to_candidate = candidate_to_seed.inverse();
    const Pose seed_to_placed = placed_to_seed.inverse();
    constexpr std::array<int, 4> delta_u{-1, 0, 0, 1};
    constexpr std::array<int, 4> delta_v{0, -1, 1, 0};
    const double minimum_squared
        = 0.25 * options.output_step * options.output_step;
    const double maximum_squared
        = 2.25 * options.output_step * options.output_step;
    for (std::int64_t v = first_v; v <= last_v; ++v) {
        for (std::int64_t u = first_u; u <= last_u; ++u) {
            const Vec2 seed_uv{
                static_cast<double>(u) * options.output_step,
                static_cast<double>(v) * options.output_step,
            };
            const auto candidate_sample = sample_patch(
                candidate, seed_to_candidate.apply(seed_uv));
            if (!candidate_sample) continue;
            for (std::size_t direction = 0; direction < 4; ++direction) {
                const Vec2 neighbor_uv{
                    seed_uv.x + delta_u[direction] * options.output_step,
                    seed_uv.y + delta_v[direction] * options.output_step,
                };
                const auto placed_sample = sample_patch(
                    placed, seed_to_placed.apply(neighbor_uv));
                if (!placed_sample) continue;
                const double squared = surfcore::distance_squared(
                    candidate_sample->xyz, placed_sample->xyz);
                if (squared < minimum_squared || squared > maximum_squared) {
                    return false;
                }
            }
        }
    }
    return true;
}

struct Raster {
    std::size_t rows = 0;
    std::size_t cols = 0;
    double origin_u = 0.0;
    double origin_v = 0.0;
    std::vector<Vec3> xyz;
};

Raster rasterize(
    const std::vector<Patch>& patches, const std::vector<Member>& members,
    const MergeOptions& options)
{
    double minimum_u = std::numeric_limits<double>::infinity();
    double minimum_v = minimum_u;
    double maximum_u = -minimum_u;
    double maximum_v = -minimum_u;
    for (const Member& member : members) {
        const Patch& patch = patches[member.patch];
        for (std::size_t row = 0; row + 1 < patch.surface->rows; ++row) {
            for (std::size_t col = 0; col + 1 < patch.surface->cols; ++col) {
                if (!patch.surface->valid_quad(row, col)) continue;
                for (int dr = 0; dr <= 1; ++dr) {
                    for (int dc = 0; dc <= 1; ++dc) {
                        const Vec2 transformed = member.to_seed.apply({
                            static_cast<double>(col + dc) / patch.scale_col,
                            static_cast<double>(row + dr) / patch.scale_row,
                        });
                        minimum_u = std::min(minimum_u, transformed.x);
                        maximum_u = std::max(maximum_u, transformed.x);
                        minimum_v = std::min(minimum_v, transformed.y);
                        maximum_v = std::max(maximum_v, transformed.y);
                    }
                }
            }
        }
    }
    Raster result;
    result.origin_u = std::floor(minimum_u / options.output_step) * options.output_step;
    result.origin_v = std::floor(minimum_v / options.output_step) * options.output_step;
    const double end_u = std::ceil(maximum_u / options.output_step) * options.output_step;
    const double end_v = std::ceil(maximum_v / options.output_step) * options.output_step;
    const double column_steps = (end_u - result.origin_u) / options.output_step;
    const double row_steps = (end_v - result.origin_v) / options.output_step;
    if (!std::isfinite(column_steps) || !std::isfinite(row_steps)
        || column_steps > std::numeric_limits<std::uint32_t>::max() - 1.0
        || row_steps > std::numeric_limits<std::uint32_t>::max() - 1.0) {
        throw std::runtime_error("merged raster dimensions overflow");
    }
    result.cols = static_cast<std::size_t>(std::llround(column_steps)) + 1;
    result.rows = static_cast<std::size_t>(std::llround(row_steps)) + 1;
    if (result.rows == 0 || result.cols == 0
        || result.rows > std::numeric_limits<std::uint32_t>::max()
        || result.cols > std::numeric_limits<std::uint32_t>::max()
        || result.rows > std::numeric_limits<std::size_t>::max() / result.cols) {
        throw std::runtime_error("merged raster dimensions overflow");
    }
    result.xyz.assign(result.rows * result.cols, {-1.0f, -1.0f, -1.0f});
    struct Contributor {
        Vec3 xyz;
        double weight;
    };
    struct ClusterValue {
        Vec3 xyz;
        double weight = 0.0;
    };
    std::vector<Contributor> contributors;
    contributors.reserve(members.size());
    for (std::size_t row = 0; row < result.rows; ++row) {
        for (std::size_t col = 0; col < result.cols; ++col) {
            const Vec2 seed_uv{
                result.origin_u + col * options.output_step,
                result.origin_v + row * options.output_step,
            };
            contributors.clear();
            for (std::size_t member_index = 0; member_index < members.size(); ++member_index) {
                const Member& member = members[member_index];
                const auto sample = sample_patch(
                    patches[member.patch], member.to_seed.inverse().apply(seed_uv));
                if (sample) {
                    contributors.push_back({
                        sample->xyz,
                        std::max(0.0, static_cast<double>(sample->boundary_weight)),
                    });
                }
            }
            if (contributors.empty()) continue;
            // Deterministic complete-link clustering: every pair within a
            // cluster must agree, so a chain of individually close samples
            // cannot bridge two contributors that differ by more than 2 vx.
            std::vector<std::vector<std::size_t>> clusters;
            for (std::size_t index = 0; index < contributors.size(); ++index) {
                bool inserted = false;
                for (auto& cluster : clusters) {
                    bool consistent = true;
                    for (const std::size_t other : cluster) {
                        if (surfcore::distance_squared(
                                contributors[index].xyz, contributors[other].xyz)
                            > kClusterThresholdSquared) {
                            consistent = false;
                            break;
                        }
                    }
                    if (consistent) {
                        cluster.push_back(index);
                        inserted = true;
                        break;
                    }
                }
                if (!inserted) clusters.push_back({index});
            }
            std::vector<ClusterValue> cluster_values(clusters.size());
            for (std::size_t cluster = 0; cluster < clusters.size(); ++cluster) {
                std::array<double, 3> blended{};
                for (const std::size_t index : clusters[cluster]) {
                    const double weight = std::max(contributors[index].weight, 1e-12);
                    cluster_values[cluster].weight += weight;
                    blended[0] += weight * contributors[index].xyz.x;
                    blended[1] += weight * contributors[index].xyz.y;
                    blended[2] += weight * contributors[index].xyz.z;
                }
                cluster_values[cluster].xyz = {
                    static_cast<float>(blended[0] / cluster_values[cluster].weight),
                    static_cast<float>(blended[1] / cluster_values[cluster].weight),
                    static_cast<float>(blended[2] / cluster_values[cluster].weight),
                };
            }
            std::size_t best_cluster = 0;
            for (std::size_t cluster = 1; cluster < clusters.size(); ++cluster) {
                if (cluster_values[cluster].weight
                    > cluster_values[best_cluster].weight) {
                    best_cluster = cluster;
                }
            }
            const std::size_t output_index = row * result.cols + col;
            result.xyz[output_index] = cluster_values[best_cluster].xyz;
        }
    }
    return result;
}

std::size_t retain_largest_quad_component(Raster& raster)
{
    if (raster.rows < 2 || raster.cols < 2) {
        return 0;
    }

    const std::size_t quad_rows = raster.rows - 1;
    const std::size_t quad_cols = raster.cols - 1;
    std::vector<std::uint8_t> valid_quads(quad_rows * quad_cols, 0);
    for (std::size_t row = 0; row < quad_rows; ++row) {
        for (std::size_t col = 0; col < quad_cols; ++col) {
            const std::size_t upper_left = row * raster.cols + col;
            valid_quads[row * quad_cols + col]
                = !sentinel(raster.xyz[upper_left])
                && !sentinel(raster.xyz[upper_left + 1])
                && !sentinel(raster.xyz[upper_left + raster.cols])
                && !sentinel(raster.xyz[upper_left + raster.cols + 1]);
        }
    }

    std::vector<std::uint8_t> visited(valid_quads.size(), 0);
    std::vector<std::size_t> best_component;
    std::queue<std::size_t> pending;
    for (std::size_t seed = 0; seed < valid_quads.size(); ++seed) {
        if (!valid_quads[seed] || visited[seed]) continue;
        std::vector<std::size_t> component;
        visited[seed] = 1;
        pending.push(seed);
        while (!pending.empty()) {
            const std::size_t current = pending.front();
            pending.pop();
            component.push_back(current);
            const std::size_t row = current / quad_cols;
            const std::size_t col = current % quad_cols;
            const std::size_t row_begin = row == 0 ? 0 : row - 1;
            const std::size_t col_begin = col == 0 ? 0 : col - 1;
            const std::size_t row_end = std::min(row + 1, quad_rows - 1);
            const std::size_t col_end = std::min(col + 1, quad_cols - 1);
            for (std::size_t next_row = row_begin; next_row <= row_end; ++next_row) {
                for (std::size_t next_col = col_begin; next_col <= col_end; ++next_col) {
                    const std::size_t next = next_row * quad_cols + next_col;
                    if (valid_quads[next] && !visited[next]) {
                        visited[next] = 1;
                        pending.push(next);
                    }
                }
            }
        }
        // The first row-major component wins an exact-size tie.
        if (component.size() > best_component.size()) {
            best_component = std::move(component);
        }
    }
    if (best_component.empty()) {
        return 0;
    }

    std::vector<std::uint8_t> retained_vertices(raster.xyz.size(), 0);
    for (const std::size_t quad : best_component) {
        const std::size_t row = quad / quad_cols;
        const std::size_t col = quad % quad_cols;
        const std::size_t upper_left = row * raster.cols + col;
        retained_vertices[upper_left] = 1;
        retained_vertices[upper_left + 1] = 1;
        retained_vertices[upper_left + raster.cols] = 1;
        retained_vertices[upper_left + raster.cols + 1] = 1;
    }
    for (std::size_t index = 0; index < raster.xyz.size(); ++index) {
        if (!retained_vertices[index]) {
            raster.xyz[index] = {-1.0f, -1.0f, -1.0f};
        }
    }
    return best_component.size();
}

double raster_member_coverage(
    const Raster& raster, const Patch& patch, const Pose& patch_to_seed,
    const MergeOptions& options)
{
    const Pose seed_to_patch = patch_to_seed.inverse();
    std::size_t sampled = 0;
    std::size_t represented = 0;
    for (std::size_t row = 0; row < raster.rows; ++row) {
        for (std::size_t col = 0; col < raster.cols; ++col) {
            const Vec2 seed_uv{
                raster.origin_u + col * options.output_step,
                raster.origin_v + row * options.output_step,
            };
            const auto member_sample = sample_patch(
                patch, seed_to_patch.apply(seed_uv));
            if (!member_sample) continue;
            ++sampled;
            const Vec3 output = raster.xyz[row * raster.cols + col];
            if (!sentinel(output)
                && surfcore::distance_squared(output, member_sample->xyz)
                    <= kClusterThresholdSquared + 1e-9) {
                ++represented;
            }
        }
    }
    return sampled == 0 ? 0.0
                        : static_cast<double>(represented) / sampled;
}

double patch_coverage(
    const Patch& reference, const Patch& candidate,
    const Pose& candidate_to_reference, std::size_t maximum_samples)
{
    const std::size_t valid_quad_count = static_cast<std::size_t>(std::count(
        candidate.surface->valid_quads.begin(),
        candidate.surface->valid_quads.end(), std::uint8_t{1}));
    if (valid_quad_count == 0) return 0.0;
    constexpr std::array<double, 2> fractions{0.25, 0.75};
    constexpr std::size_t samples_per_quad
        = fractions.size() * fractions.size();
    const std::size_t quad_sample_budget = std::max<std::size_t>(
        1, maximum_samples / samples_per_quad);
    const std::size_t stride = std::max<std::size_t>(
        1, (valid_quad_count + quad_sample_budget - 1) / quad_sample_budget);
    std::size_t ordinal = 0;
    std::size_t sampled = 0;
    std::size_t represented = 0;
    for (std::size_t row = 0; row + 1 < candidate.surface->rows; ++row) {
        for (std::size_t col = 0; col + 1 < candidate.surface->cols; ++col) {
            if (!candidate.surface->valid_quad(row, col)) continue;
            if (ordinal++ % stride != 0) continue;
            for (const double row_fraction : fractions) {
                for (const double col_fraction : fractions) {
                    const Vec2 candidate_uv{
                        (static_cast<double>(col) + col_fraction)
                            / candidate.scale_col,
                        (static_cast<double>(row) + row_fraction)
                            / candidate.scale_row,
                    };
                    const auto candidate_sample = sample_patch(
                        candidate, candidate_uv);
                    const auto reference_sample = sample_patch(
                        reference, candidate_to_reference.apply(candidate_uv));
                    if (!candidate_sample) continue;
                    ++sampled;
                    if (reference_sample
                        && surfcore::distance_squared(
                               reference_sample->xyz, candidate_sample->xyz)
                            <= kClusterThresholdSquared + 1e-9) {
                        ++represented;
                    }
                }
            }
        }
    }
    return sampled == 0 ? 0.0
                        : static_cast<double>(represented) / sampled;
}

void write_float_tiff(
    const std::filesystem::path& path, const Raster& raster, int component)
{
    TiffHandle file(path, "w");
    TIFFSetField(file.get(), TIFFTAG_IMAGEWIDTH, static_cast<std::uint32_t>(raster.cols));
    TIFFSetField(file.get(), TIFFTAG_IMAGELENGTH, static_cast<std::uint32_t>(raster.rows));
    TIFFSetField(file.get(), TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(file.get(), TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(file.get(), TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(file.get(), TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(file.get(), TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(file.get(), TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField(file.get(), TIFFTAG_ROWSPERSTRIP,
                 TIFFDefaultStripSize(file.get(), raster.cols * sizeof(float)));
    std::vector<float> line(raster.cols);
    for (std::size_t row = 0; row < raster.rows; ++row) {
        for (std::size_t col = 0; col < raster.cols; ++col) {
            const Vec3 value = raster.xyz[row * raster.cols + col];
            line[col] = component == 0 ? value.x : (component == 1 ? value.y : value.z);
        }
        if (TIFFWriteScanline(file.get(), line.data(), static_cast<std::uint32_t>(row)) < 0) {
            throw std::runtime_error("failed writing TIFF " + path.string());
        }
    }
}

std::optional<std::vector<std::size_t>> write_output_candidate(
    const std::filesystem::path& output_root, const Patch& seed,
    const std::vector<Patch>& patches, const std::vector<Member>& members,
    const std::vector<Member>& containment_candidates,
    Raster& raster, const MergeOptions& options, std::size_t output_index)
{
    const std::size_t valid_quad_count = retain_largest_quad_component(raster);
    if (valid_quad_count == 0) return std::nullopt;
    std::vector<std::size_t> contained_members;
    json contained_member_ids = json::array();
    for (const Member& member : containment_candidates) {
        if (patches[member.patch].id == seed.id) continue;
        const double coverage = raster_member_coverage(
            raster, patches[member.patch], member.to_seed, options);
        if (coverage + 1e-12 >= options.containment_threshold) {
            contained_members.push_back(member.patch);
            contained_member_ids.push_back(patches[member.patch].id);
        }
    }
    const std::filesystem::path temporary
        = output_root / ("." + seed.id + ".tmp." + std::to_string(output_index));
    std::filesystem::create_directory(temporary);
    write_float_tiff(temporary / "x.tif", raster, 0);
    write_float_tiff(temporary / "y.tif", raster, 1);
    write_float_tiff(temporary / "z.tif", raster, 2);

    std::array<double, 3> low{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    std::array<double, 3> high{-low[0], -low[1], -low[2]};
    for (std::size_t index = 0; index < raster.xyz.size(); ++index) {
        if (sentinel(raster.xyz[index])) continue;
        low[0] = std::min(low[0], static_cast<double>(raster.xyz[index].x));
        low[1] = std::min(low[1], static_cast<double>(raster.xyz[index].y));
        low[2] = std::min(low[2], static_cast<double>(raster.xyz[index].z));
        high[0] = std::max(high[0], static_cast<double>(raster.xyz[index].x));
        high[1] = std::max(high[1], static_cast<double>(raster.xyz[index].y));
        high[2] = std::max(high[2], static_cast<double>(raster.xyz[index].z));
    }
    const double area_vx2 = valid_quad_count * options.output_step * options.output_step;
    json member_ids = json::array();
    json direct_ids = json::array();
    json poses = json::array();
    for (const Member& member : members) {
        member_ids.push_back(patches[member.patch].id);
        if (patches[member.patch].id != seed.id) {
            direct_ids.push_back(patches[member.patch].id);
        }
        poses.push_back(pose_json(
            patches[member.patch].id, member.to_seed, member.inliers, member.rms));
    }
    json metadata{
        {"format", "tifxyz"},
        {"type", "seg"},
        {"uuid", seed.id},
        {"source", "merge_overlapping_patches"},
        {"scale", {1.0 / options.output_step, 1.0 / options.output_step}},
        {"bbox", {low, high}},
        {"area_vx2", area_vx2},
        {"seed_id", seed.id},
        {"member_ids", member_ids},
        {"direct_member_ids", direct_ids},
        {"contained_member_ids", contained_member_ids},
        {"fitted_poses", poses},
        {"seed_metric_origin", {raster.origin_u, raster.origin_v}},
        {"merge_settings", settings_json(options)},
    };
    if (seed.area_cm2_per_vx2) {
        metadata["area_cm2"] = area_vx2 * *seed.area_cm2_per_vx2;
    }
    std::ofstream stream(temporary / "meta.json");
    if (!stream) throw std::runtime_error("cannot write output metadata");
    stream << metadata.dump(2) << '\n';
    stream.close();
    if (!stream) throw std::runtime_error("failed writing output metadata");
    return contained_members;
}

void validate_options(const MergeOptions& options)
{
    const auto positive_finite = [](double value, const char* name) {
        if (!(value > 0.0) || !std::isfinite(value)) {
            throw std::invalid_argument(std::string(name) + " must be positive and finite");
        }
    };
    positive_finite(options.tolerance, "tolerance");
    positive_finite(options.dense_spacing, "dense_spacing");
    positive_finite(options.output_step, "output_step");
    positive_finite(options.thinning_spacing, "thinning_spacing");
    positive_finite(options.uv_inlier_tolerance, "uv_inlier_tolerance");
    positive_finite(options.min_major_spread, "min_major_spread");
    positive_finite(options.min_minor_spread, "min_minor_spread");
    positive_finite(options.max_refit_rms, "max_refit_rms");
    if (!(options.containment_threshold > 0.0
          && options.containment_threshold <= 1.0)
        || !std::isfinite(options.containment_threshold)) {
        throw std::invalid_argument(
            "containment_threshold must be finite and in (0, 1]");
    }
    if (options.erode_cells < 0) throw std::invalid_argument("erode_cells must be non-negative");
    if (options.max_correspondences < 2) {
        throw std::invalid_argument("max_correspondences must be at least 2");
    }
    if (options.min_inliers < 2 || options.min_inliers > options.max_correspondences) {
        throw std::invalid_argument("min_inliers is outside the correspondence limit");
    }
    if (!(options.ransac_confidence > 0.0 && options.ransac_confidence < 1.0)) {
        throw std::invalid_argument("ransac_confidence must be between zero and one");
    }
    if (options.ransac_max_hypotheses == 0) {
        throw std::invalid_argument("ransac_max_hypotheses must be positive");
    }
    if (options.threads < 0) throw std::invalid_argument("threads must be non-negative");
}

json report_document(const MergeReport& report)
{
    return {
        {"input_count", report.input_count},
        {"output_count", report.output_count},
        {"accepted_pair_count", report.accepted_pair_count},
        {"rejected_pair_count", report.rejected_pair_count},
        {"dropped_invalid_count", report.dropped_invalid_count},
        {"dropped_output_count", report.dropped_output_count},
        {"contained_patch_count", report.contained_patch_count},
        {"output_nms_suppressed_count", report.output_nms_suppressed_count},
        {"output_nms_accepted_pair_count", report.output_nms_accepted_pair_count},
        {"output_nms_rejected_pair_count", report.output_nms_rejected_pair_count},
        {"total_duration", report.total_duration},
        {"stage_timings", report.stage_timings},
        {"rejections", report.rejections},
        {"output_nms_rejections", report.output_nms_rejections},
    };
}

} // namespace

MergeReport merge_patch_directory(
    const std::filesystem::path& patches_dir,
    const std::filesystem::path& output_dir,
    const MergeOptions& options)
{
    validate_options(options);
    const Clock::time_point total_start = Clock::now();
    if (!std::filesystem::is_directory(patches_dir)) {
        throw std::invalid_argument("patch directory does not exist: " + patches_dir.string());
    }
    if (std::filesystem::exists(output_dir)) {
        if (!std::filesystem::is_directory(output_dir)
            || !std::filesystem::is_empty(output_dir)) {
            throw std::invalid_argument("output directory must be new or empty");
        }
    }
#ifdef _OPENMP
    const int worker_count = options.threads > 0 ? options.threads
                                                 : omp_get_num_procs();
    omp_set_num_threads(worker_count);
#else
    const int worker_count = 1;
#endif

    MergeReport report;
    const auto log_stage = [&](const char* name) {
        if (options.progress) {
            std::cerr << "[patch-merger] " << name << "="
                      << report.stage_timings[name] << "s elapsed="
                      << seconds_since(total_start) << "s\n";
        }
    };
    Clock::time_point stage = Clock::now();
    std::vector<std::filesystem::path> inputs;
    for (const auto& entry : std::filesystem::directory_iterator(patches_dir)) {
        if (entry.is_directory()
            && std::filesystem::is_regular_file(entry.path() / "meta.json")) {
            inputs.push_back(entry.path());
        }
    }
    std::sort(inputs.begin(), inputs.end());
    report.input_count = inputs.size();
    report.stage_timings["discovery"] = seconds_since(stage);
    log_stage("discovery");
    if (inputs.empty()) throw std::runtime_error("no tifxyz patches found");

    stage = Clock::now();
    std::vector<std::optional<Patch>> loaded(inputs.size());
    std::vector<std::string> errors(inputs.size());
    std::vector<std::uint8_t> dropped(inputs.size(), 0);
#pragma omp parallel for schedule(dynamic, 1)
    for (std::int64_t index = 0;
         index < static_cast<std::int64_t>(inputs.size()); ++index) {
        try {
            bool was_dropped = false;
            Patch patch = load_patch(inputs[static_cast<std::size_t>(index)], options,
                                     was_dropped);
            dropped[static_cast<std::size_t>(index)] = was_dropped ? 1 : 0;
            if (!was_dropped) loaded[static_cast<std::size_t>(index)] = std::move(patch);
        } catch (const std::exception& error) {
            errors[static_cast<std::size_t>(index)] = error.what();
        }
    }
    for (std::size_t index = 0; index < errors.size(); ++index) {
        if (!errors[index].empty()) {
            throw std::runtime_error(inputs[index].string() + ": " + errors[index]);
        }
        report.dropped_invalid_count += dropped[index] ? 1 : 0;
    }
    std::vector<Patch> patches;
    patches.reserve(inputs.size() - report.dropped_invalid_count);
    for (auto& patch : loaded) {
        if (patch) patches.push_back(std::move(*patch));
    }
    std::stable_sort(patches.begin(), patches.end(), [](const Patch& a, const Patch& b) {
        return a.area > b.area || (a.area == b.area && a.id < b.id);
    });
    for (std::size_t index = 1; index < patches.size(); ++index) {
        if (patches[index - 1].id == patches[index].id) {
            throw std::runtime_error("duplicate patch ID: " + patches[index].id);
        }
    }
    report.stage_timings["loading"] = seconds_since(stage);
    log_stage("loading");

    stage = Clock::now();
    std::vector<std::shared_ptr<SurfaceData>> surfaces;
    surfaces.reserve(patches.size());
    for (const Patch& patch : patches) surfaces.push_back(patch.surface);
    SurfacePatchIndex index;
    index.rebuild(std::move(surfaces), 0.0f, 1);
    report.stage_timings["indexing"] = seconds_since(stage);
    log_stage("indexing");
    if (options.progress) {
        std::cerr << "[patch-merger] indexed_surfaces=" << patches.size()
                  << " indexed_tiles=" << index.tile_count() << '\n';
    }

    stage = Clock::now();
    std::vector<std::vector<PairResult>> pairs_by_high(patches.size());
    HarvestCounters harvest_counters;
#pragma omp parallel for schedule(dynamic, 1)
    for (std::int64_t high = 0;
         high < static_cast<std::int64_t>(patches.size()); ++high) {
        if (static_cast<std::size_t>(high) + 1 < patches.size()) {
            pairs_by_high[static_cast<std::size_t>(high)] = harvest_and_fit(
                static_cast<std::size_t>(high), patches, index, options,
                options.progress ? &harvest_counters : nullptr);
        }
        const std::size_t completed = harvest_counters.completed_seeds.fetch_add(
            1, std::memory_order_relaxed) + 1;
        if (options.progress
            && (completed <= 20 || completed == patches.size()
                || completed % 250 == 0)) {
#pragma omp critical(patch_merger_progress)
            {
                std::cerr << "[patch-merger] fitting_progress=" << completed
                          << '/' << patches.size()
                          << " bbox_pairs=" << harvest_counters.bbox_pairs.load(
                                 std::memory_order_relaxed)
                          << " tile_pairs=" << harvest_counters.tile_pairs.load(
                                 std::memory_order_relaxed)
                          << " vertex_samples=" << harvest_counters.vertex_samples.load(
                                 std::memory_order_relaxed)
                          << " triangle_hits=" << harvest_counters.triangle_hits.load(
                                 std::memory_order_relaxed)
                          << " dense_pairs=" << harvest_counters.dense_pairs.load(
                                 std::memory_order_relaxed)
                          << " dense_samples=" << harvest_counters.dense_samples.load(
                                 std::memory_order_relaxed)
                          << " exact_samples=" << harvest_counters.exact_samples.load(
                                 std::memory_order_relaxed)
                          << " elapsed=" << seconds_since(total_start) << "s\n";
            }
        }
    }
    std::vector<PairResult> pair_results;
    for (auto& values : pairs_by_high) {
        pair_results.insert(pair_results.end(),
                            std::make_move_iterator(values.begin()),
                            std::make_move_iterator(values.end()));
    }
    for (const PairResult& pair : pair_results) {
        if (pair.fit.accepted) {
            ++report.accepted_pair_count;
        } else {
            ++report.rejected_pair_count;
            ++report.rejections[pair.fit.rejection];
        }
    }
    const std::size_t total_pair_count = patches.size() < 2 ? 0
        : patches.size() * (patches.size() - 1) / 2;
    const std::size_t no_correspondence_count
        = total_pair_count - pair_results.size();
    report.rejected_pair_count += no_correspondence_count;
    if (no_correspondence_count != 0) {
        report.rejections["no_correspondences"] += no_correspondence_count;
    }
    report.stage_timings["fitting"] = seconds_since(stage);
    log_stage("fitting");

    stage = Clock::now();
    std::vector<std::vector<Member>> direct_members(patches.size());
    for (std::size_t seed = 0; seed < patches.size(); ++seed) {
        direct_members[seed].push_back({seed, {}, 0, 0.0});
    }
    std::map<std::pair<std::size_t, std::size_t>, const PairResult*> pair_lookup;
    for (const PairResult& pair : pair_results) {
        if (!pair.fit.accepted) continue;
        pair_lookup[{pair.high, pair.low}] = &pair;
        direct_members[pair.high].push_back({
            pair.low, pair.fit.pose, pair.fit.inliers, pair.fit.rms});
        direct_members[pair.low].push_back({
            pair.high, pair.fit.pose.inverse(), pair.fit.inliers, pair.fit.rms});
    }
    std::vector<std::vector<Member>> outputs(patches.size());
    for (std::size_t seed = 0; seed < direct_members.size(); ++seed) {
        outputs[seed].push_back({seed, {}, 0, 0.0});
        std::sort(direct_members[seed].begin() + 1, direct_members[seed].end(),
                  [&](const Member& a, const Member& b) {
                      if (patches[a.patch].area != patches[b.patch].area) {
                          return patches[a.patch].area > patches[b.patch].area;
                      }
                      if (a.inliers != b.inliers) return a.inliers > b.inliers;
                      if (a.rms != b.rms) return a.rms < b.rms;
                      return patches[a.patch].id < patches[b.patch].id;
                  });
        for (std::size_t candidate_index = 1;
             candidate_index < direct_members[seed].size(); ++candidate_index) {
            const Member& candidate = direct_members[seed][candidate_index];
            std::vector<Pose> proposals{candidate.to_seed};
            for (std::size_t placed_index = 1;
                 placed_index < outputs[seed].size(); ++placed_index) {
                const Member& placed = outputs[seed][placed_index];
                const std::size_t high = std::max(candidate.patch, placed.patch);
                const std::size_t low = std::min(candidate.patch, placed.patch);
                const auto found = pair_lookup.find({high, low});
                if (found == pair_lookup.end()) continue;
                const PairResult& relation = *found->second;
                const Pose candidate_to_placed = placed.patch == high
                    ? relation.fit.pose : relation.fit.pose.inverse();
                proposals.push_back(compose_pose(
                    placed.to_seed, candidate_to_placed));
            }

            std::optional<Pose> best_pose;
            std::size_t best_agreements = 0;
            for (const Pose& proposal : proposals) {
                if (!pose_preserves_output_metric(
                        patches[candidate.patch], proposal)) {
                    continue;
                }
                bool consistent = true;
                std::size_t agreements = 0;
                for (const Member& placed : outputs[seed]) {
                    const OverlapConsistency overlap
                        = transformed_overlap_consistency(
                            patches[candidate.patch], proposal,
                            patches[placed.patch], placed.to_seed, options);
                    if (overlap.samples >= 8
                        && overlap.agreements != overlap.samples) {
                        consistent = false;
                        break;
                    }
                    if (!transformed_seams_are_consistent(
                            patches[candidate.patch], proposal,
                            patches[placed.patch], placed.to_seed, options)) {
                        consistent = false;
                        break;
                    }
                    agreements += overlap.agreements;
                }
                if (consistent && (!best_pose || agreements > best_agreements)) {
                    best_pose = proposal;
                    best_agreements = agreements;
                }
            }
            if (best_pose) {
                Member accepted = candidate;
                accepted.to_seed = *best_pose;
                outputs[seed].push_back(accepted);
            }
        }
        std::sort(outputs[seed].begin() + 1, outputs[seed].end(),
                  [&](const Member& a, const Member& b) {
                      return patches[a.patch].id < patches[b.patch].id;
                  });
    }
    report.stage_timings["atlas_construction"] = seconds_since(stage);
    log_stage("atlas_construction");

    std::filesystem::create_directories(output_dir);
    std::vector<std::string> output_errors(patches.size());
    std::vector<std::vector<std::size_t>> contained_by_output(patches.size());
    std::vector<std::uint8_t> output_candidate_available(patches.size(), 0);
    std::atomic<double> raster_seconds{0.0};
    std::atomic<double> io_seconds{0.0};
#pragma omp parallel for schedule(dynamic, 1)
    for (std::int64_t seed = 0;
         seed < static_cast<std::int64_t>(patches.size()); ++seed) {
        try {
            Clock::time_point task = Clock::now();
            Raster raster = rasterize(
                patches, outputs[static_cast<std::size_t>(seed)], options);
            raster_seconds.fetch_add(seconds_since(task), std::memory_order_relaxed);
            task = Clock::now();
            auto candidate = write_output_candidate(
                output_dir, patches[static_cast<std::size_t>(seed)], patches,
                outputs[static_cast<std::size_t>(seed)],
                direct_members[static_cast<std::size_t>(seed)], raster, options,
                static_cast<std::size_t>(seed));
            if (candidate) {
                contained_by_output[static_cast<std::size_t>(seed)]
                    = std::move(*candidate);
                output_candidate_available[static_cast<std::size_t>(seed)] = 1;
            }
            io_seconds.fetch_add(seconds_since(task), std::memory_order_relaxed);
        } catch (const std::exception& error) {
            output_errors[static_cast<std::size_t>(seed)] = error.what();
        }
    }
    for (std::size_t seed = 0; seed < output_errors.size(); ++seed) {
        if (!output_errors[seed].empty()) {
            throw std::runtime_error(
                "failed writing seed " + patches[seed].id + ": " + output_errors[seed]);
        }
    }
    stage = Clock::now();
    std::vector<std::uint8_t> source_suppressed(patches.size(), 0);
    for (std::size_t seed = 0; seed < patches.size(); ++seed) {
        if (source_suppressed[seed] || !output_candidate_available[seed]) continue;
        for (const std::size_t member : contained_by_output[seed]) {
            // Input patches are ordered largest-first. Earlier seeds cannot
            // be retroactively suppressed by a later output.
            if (member > seed) source_suppressed[member] = 1;
        }
    }
    report.stage_timings["source_suppression"] = seconds_since(stage);
    log_stage("source_suppression");

    stage = Clock::now();
    MergeOptions nms_load_options = options;
    nms_load_options.erode_cells = 0;
    nms_load_options.progress = false;
    std::unordered_map<std::string, std::size_t> original_seed_by_id;
    original_seed_by_id.reserve(patches.size());
    for (std::size_t seed = 0; seed < patches.size(); ++seed) {
        original_seed_by_id.emplace(patches[seed].id, seed);
    }
    std::vector<Patch> nms_patches;
    nms_patches.reserve(patches.size());
    for (std::size_t seed = 0; seed < patches.size(); ++seed) {
        if (source_suppressed[seed] || !output_candidate_available[seed]) continue;
        const std::filesystem::path temporary
            = output_dir / ("." + patches[seed].id + ".tmp." + std::to_string(seed));
        bool dropped = false;
        Patch candidate = load_patch(temporary, nms_load_options, dropped);
        if (dropped) {
            throw std::runtime_error(
                "temporary output unexpectedly contains no valid quads: "
                + patches[seed].id);
        }
        nms_patches.push_back(std::move(candidate));
    }
    std::stable_sort(
        nms_patches.begin(), nms_patches.end(), [](const Patch& a, const Patch& b) {
            return a.area > b.area || (a.area == b.area && a.id < b.id);
        });
    report.stage_timings["output_nms_loading"] = seconds_since(stage);
    log_stage("output_nms_loading");

    stage = Clock::now();
    std::vector<std::vector<PairResult>> nms_pairs_by_high(nms_patches.size());
    if (nms_patches.size() > 1) {
        std::vector<std::shared_ptr<SurfaceData>> nms_surfaces;
        nms_surfaces.reserve(nms_patches.size());
        for (const Patch& patch : nms_patches) nms_surfaces.push_back(patch.surface);
        SurfacePatchIndex nms_index;
        nms_index.rebuild(std::move(nms_surfaces), 0.0f, 1);
#pragma omp parallel for schedule(dynamic, 1)
        for (std::int64_t high = 0;
             high < static_cast<std::int64_t>(nms_patches.size()); ++high) {
            if (static_cast<std::size_t>(high) + 1 < nms_patches.size()) {
                nms_pairs_by_high[static_cast<std::size_t>(high)] = harvest_and_fit(
                    static_cast<std::size_t>(high), nms_patches, nms_index,
                    options, nullptr);
            }
        }
    }
    for (const auto& pairs : nms_pairs_by_high) {
        for (const PairResult& pair : pairs) {
            if (pair.fit.accepted) {
                ++report.output_nms_accepted_pair_count;
            } else {
                ++report.output_nms_rejected_pair_count;
                ++report.output_nms_rejections[pair.fit.rejection];
            }
        }
    }
    report.stage_timings["output_nms_fitting"] = seconds_since(stage);
    log_stage("output_nms_fitting");

    stage = Clock::now();
    std::vector<std::vector<double>> nms_coverages(nms_patches.size());
#pragma omp parallel for schedule(dynamic, 1)
    for (std::int64_t high = 0;
         high < static_cast<std::int64_t>(nms_patches.size()); ++high) {
        const auto& pairs = nms_pairs_by_high[static_cast<std::size_t>(high)];
        auto& coverages = nms_coverages[static_cast<std::size_t>(high)];
        coverages.assign(pairs.size(), 0.0);
        for (std::size_t index = 0; index < pairs.size(); ++index) {
            const PairResult& pair = pairs[index];
            if (!pair.fit.accepted) continue;
            coverages[index] = patch_coverage(
                nms_patches[pair.high], nms_patches[pair.low], pair.fit.pose,
                options.max_correspondences);
        }
    }
    std::vector<std::uint8_t> nms_suppressed(nms_patches.size(), 0);
    std::vector<std::uint8_t> output_nms_suppressed(patches.size(), 0);
    for (std::size_t high = 0; high < nms_patches.size(); ++high) {
        if (nms_suppressed[high]) continue;
        const auto& pairs = nms_pairs_by_high[high];
        for (std::size_t index = 0; index < pairs.size(); ++index) {
            const PairResult& pair = pairs[index];
            if (!pair.fit.accepted || nms_suppressed[pair.low]) continue;
            if (nms_coverages[high][index] + 1e-12
                < options.containment_threshold) {
                continue;
            }
            nms_suppressed[pair.low] = 1;
            output_nms_suppressed[
                original_seed_by_id.at(nms_patches[pair.low].id)] = 1;
            ++report.output_nms_suppressed_count;
        }
    }
    report.stage_timings["output_nms_coverage"] = seconds_since(stage);
    log_stage("output_nms_coverage");

    const Clock::time_point finalize_start = Clock::now();
    for (std::size_t seed = 0; seed < patches.size(); ++seed) {
        const std::filesystem::path temporary
            = output_dir / ("." + patches[seed].id + ".tmp." + std::to_string(seed));
        if (source_suppressed[seed] || output_nms_suppressed[seed]) {
            if (output_candidate_available[seed]) std::filesystem::remove_all(temporary);
            ++report.contained_patch_count;
            continue;
        }
        if (!output_candidate_available[seed]) {
            ++report.dropped_output_count;
            continue;
        }
        std::filesystem::rename(temporary, output_dir / patches[seed].id);
        ++report.output_count;
    }
    const double timing_divisor = worker_count;
    report.stage_timings["rasterization"]
        = raster_seconds.load(std::memory_order_relaxed) / timing_divisor;
    report.stage_timings["filesystem_io"]
        = io_seconds.load(std::memory_order_relaxed) / timing_divisor
        + seconds_since(finalize_start);
    report.total_duration = seconds_since(total_start);

    json root = report_document(report);
    root["settings"] = settings_json(options);
    root["settings"]["threads"] = options.threads;
    std::ofstream report_stream(output_dir / "report.json");
    if (!report_stream) throw std::runtime_error("cannot write root report.json");
    report_stream << root.dump(2) << '\n';
    report_stream.close();
    if (!report_stream) throw std::runtime_error("failed writing root report.json");
    return report;
}

std::string report_json(const MergeReport& report)
{
    return report_document(report).dump(2);
}

} // namespace vc_spiral::patch_merger
