#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(_WIN32)
#define VC_EXPORT __declspec(dllexport)
#else
#define VC_EXPORT __attribute__((visibility("default")))
#endif

#ifndef VC_SOURCE_FINGERPRINT
#define VC_SOURCE_FINGERPRINT "unversioned"
#endif

namespace {

constexpr std::uint32_t kAbiVersion = 1;

struct RasterRequest {
    const double* target_x;
    const double* target_y;
    const double* target_z;
    const std::uint8_t* target_valid;
    const double* uv_rows;
    const double* uv_cols;
    const std::uint8_t* uv_valid;
    const double* source_x;
    const double* source_y;
    const double* source_z;
    const std::uint8_t* source_valid;
    const double* filled_uv_rows;
    const double* filled_uv_cols;
    const std::uint8_t* source_validity;
    std::int64_t* output_source_indices;
    std::uint8_t* output_validity;
    double* output_distances;

    std::int64_t target_height;
    std::int64_t target_width;
    std::int64_t source_height;
    std::int64_t source_width;
    std::int64_t label_height;
    std::int64_t label_width;
    std::int64_t output_height;
    std::int64_t output_width;
    std::int64_t row_start;
    std::int64_t row_end;
    std::int64_t col_start;
    std::int64_t col_end;

    double label_offset_y;
    double label_offset_x;
    double max_distance;
    std::uint32_t fill_seams;
    std::uint32_t has_source_validity;
    std::uint32_t abi_version;
};

struct RasterResult {
    std::int64_t target_surface_valid;
    std::int64_t measured_pixels;
    std::int64_t seam_filled_pixels;
    std::int64_t inherited_filled_pixels;
};

struct BilinearPosition {
    std::int64_t y0;
    std::int64_t y1;
    std::int64_t x0;
    std::int64_t x1;
    double wy;
    double wx;
};

inline std::int64_t flat_index(
    const std::int64_t row,
    const std::int64_t col,
    const std::int64_t width
) {
    return row * width + col;
}

inline BilinearPosition output_position(
    const std::int64_t row,
    const std::int64_t col,
    const RasterRequest& request
) {
    const double grid_y =
        (static_cast<double>(row) + 0.5) *
        static_cast<double>(request.target_height) /
        static_cast<double>(request.output_height);
    const double grid_x =
        (static_cast<double>(col) + 0.5) *
        static_cast<double>(request.target_width) /
        static_cast<double>(request.output_width);
    auto y0 = static_cast<std::int64_t>(std::floor(grid_y));
    auto x0 = static_cast<std::int64_t>(std::floor(grid_x));
    if (y0 < 0) {
        y0 = 0;
    } else if (y0 >= request.target_height) {
        y0 = request.target_height - 1;
    }
    if (x0 < 0) {
        x0 = 0;
    } else if (x0 >= request.target_width) {
        x0 = request.target_width - 1;
    }
    const auto y1 =
        y0 + 1 < request.target_height ? y0 + 1 : y0;
    const auto x1 =
        x0 + 1 < request.target_width ? x0 + 1 : x0;
    return {y0, y1, x0, x1, grid_y - y0, grid_x - x0};
}

inline bool corners_valid(
    const std::uint8_t* valid,
    const BilinearPosition& position,
    const std::int64_t width
) {
    return
        valid[flat_index(position.y0, position.x0, width)] != 0 &&
        valid[flat_index(position.y0, position.x1, width)] != 0 &&
        valid[flat_index(position.y1, position.x0, width)] != 0 &&
        valid[flat_index(position.y1, position.x1, width)] != 0;
}

inline double bilinear(
    const double* field,
    const BilinearPosition& position,
    const std::int64_t width
) {
    const double f00 = field[flat_index(position.y0, position.x0, width)];
    const double f01 = field[flat_index(position.y0, position.x1, width)];
    const double f10 = field[flat_index(position.y1, position.x0, width)];
    const double f11 = field[flat_index(position.y1, position.x1, width)];
    const double one_minus_x = 1.0 - position.wx;
    const double one_minus_y = 1.0 - position.wy;
    const double top = f00 * one_minus_x + f01 * position.wx;
    const double bottom = f10 * one_minus_x + f11 * position.wx;
    return top * one_minus_y + bottom * position.wy;
}

inline bool label_index(
    const double mapped_row,
    const double mapped_col,
    const RasterRequest& request,
    std::int64_t& index
) {
    if (!std::isfinite(mapped_row) || !std::isfinite(mapped_col)) {
        return false;
    }
    const double scaled_row =
        mapped_row * static_cast<double>(request.label_height) /
        static_cast<double>(request.source_height) - request.label_offset_y;
    const double scaled_col =
        mapped_col * static_cast<double>(request.label_width) /
        static_cast<double>(request.source_width) - request.label_offset_x;
    if (
        !std::isfinite(scaled_row) || !std::isfinite(scaled_col) ||
        scaled_row < 0.0 || scaled_row >= request.label_height ||
        scaled_col < 0.0 || scaled_col >= request.label_width
    ) {
        return false;
    }
    const auto row = static_cast<std::int64_t>(std::floor(scaled_row));
    const auto col = static_cast<std::int64_t>(std::floor(scaled_col));
    index = flat_index(row, col, request.label_width);
    return true;
}

inline bool sample_source_xyz(
    const double mapped_row,
    const double mapped_col,
    const RasterRequest& request,
    double& x,
    double& y,
    double& z
) {
    if (
        !std::isfinite(mapped_row) || !std::isfinite(mapped_col) ||
        mapped_row < 0.0 || mapped_row > request.source_height - 1 ||
        mapped_col < 0.0 || mapped_col > request.source_width - 1
    ) {
        return false;
    }
    auto row0 = static_cast<std::int64_t>(std::floor(mapped_row));
    auto col0 = static_cast<std::int64_t>(std::floor(mapped_col));
    if (row0 >= request.source_height - 1) {
        row0 = request.source_height - 2;
    }
    if (col0 >= request.source_width - 1) {
        col0 = request.source_width - 2;
    }
    const double row_fraction = mapped_row - row0;
    const double col_fraction = mapped_col - col0;
    const auto p00 = flat_index(row0, col0, request.source_width);
    const auto p10 = flat_index(row0, col0 + 1, request.source_width);
    const auto p01 = flat_index(row0 + 1, col0, request.source_width);
    const auto p11 = flat_index(row0 + 1, col0 + 1, request.source_width);
    if (
        request.source_valid[p00] == 0 ||
        request.source_valid[p10] == 0 ||
        request.source_valid[p01] == 0 ||
        request.source_valid[p11] == 0
    ) {
        return false;
    }

    const bool first_triangle = row_fraction + col_fraction <= 1.0;
    const auto interpolate_coordinate = [&](const double* field) {
        if (first_triangle) {
            const double col_term =
                col_fraction * (field[p10] - field[p00]);
            const double row_term =
                row_fraction * (field[p01] - field[p00]);
            return field[p00] + col_term + row_term;
        }
        const double first = (1.0 - row_fraction) * field[p10];
        const double second =
            (row_fraction + col_fraction - 1.0) * field[p11];
        const double third = (1.0 - col_fraction) * field[p01];
        return first + second + third;
    };
    x = interpolate_coordinate(request.source_x);
    y = interpolate_coordinate(request.source_y);
    z = interpolate_coordinate(request.source_z);
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

}  // namespace

extern "C" {

VC_EXPORT std::uint32_t vc_tifxyz_rasterizer_abi_version() {
    return kAbiVersion;
}

VC_EXPORT const char* vc_tifxyz_rasterizer_source_fingerprint() {
    return VC_SOURCE_FINGERPRINT;
}

VC_EXPORT std::size_t vc_tifxyz_rasterizer_request_size() {
    return sizeof(RasterRequest);
}

VC_EXPORT std::size_t vc_tifxyz_rasterizer_result_size() {
    return sizeof(RasterResult);
}

VC_EXPORT int vc_tifxyz_rasterize(
    const RasterRequest* request_pointer,
    RasterResult* result
) {
    if (request_pointer == nullptr || result == nullptr) {
        return 1;
    }
    const RasterRequest& request = *request_pointer;
    if (request.abi_version != kAbiVersion) {
        return 2;
    }
    if (
        request.target_x == nullptr || request.target_y == nullptr ||
        request.target_z == nullptr || request.target_valid == nullptr ||
        request.uv_rows == nullptr || request.uv_cols == nullptr ||
        request.uv_valid == nullptr || request.source_x == nullptr ||
        request.source_y == nullptr || request.source_z == nullptr ||
        request.source_valid == nullptr ||
        request.output_source_indices == nullptr ||
        request.output_validity == nullptr ||
        request.output_distances == nullptr
    ) {
        return 3;
    }
    if (
        request.target_height < 2 || request.target_width < 2 ||
        request.source_height < 2 || request.source_width < 2 ||
        request.label_height < 1 || request.label_width < 1 ||
        request.output_height < 1 || request.output_width < 1 ||
        request.row_start < 0 || request.row_end > request.output_height ||
        request.col_start < 0 || request.col_end > request.output_width ||
        request.row_start >= request.row_end ||
        request.col_start >= request.col_end ||
        !std::isfinite(request.max_distance) || request.max_distance <= 0.0
    ) {
        return 4;
    }
    if (
        request.fill_seams != 0 &&
        (request.filled_uv_rows == nullptr || request.filled_uv_cols == nullptr)
    ) {
        return 5;
    }
    if (request.has_source_validity != 0 && request.source_validity == nullptr) {
        return 6;
    }

    result->target_surface_valid = 0;
    result->measured_pixels = 0;
    result->seam_filled_pixels = 0;
    result->inherited_filled_pixels = 0;
    const auto tile_width = request.col_end - request.col_start;
    const double invalid_distance = std::numeric_limits<double>::quiet_NaN();

    for (auto row = request.row_start; row < request.row_end; ++row) {
        for (auto col = request.col_start; col < request.col_end; ++col) {
            const auto local =
                (row - request.row_start) * tile_width +
                (col - request.col_start);
            request.output_source_indices[local] = -1;
            request.output_validity[local] = 0;
            request.output_distances[local] = invalid_distance;

            const BilinearPosition position = output_position(row, col, request);
            const bool target_corners_valid = corners_valid(
                request.target_valid, position, request.target_width
            );
            const double target_x = bilinear(
                request.target_x, position, request.target_width
            );
            const double target_y = bilinear(
                request.target_y, position, request.target_width
            );
            const double target_z = bilinear(
                request.target_z, position, request.target_width
            );
            const bool target_pixel_valid =
                target_corners_valid && std::isfinite(target_x) &&
                std::isfinite(target_y) && std::isfinite(target_z);
            if (target_pixel_valid) {
                ++result->target_surface_valid;
            }

            const bool uv_corners_valid = corners_valid(
                request.uv_valid, position, request.target_width
            );
            const double mapped_row = bilinear(
                request.uv_rows, position, request.target_width
            );
            const double mapped_col = bilinear(
                request.uv_cols, position, request.target_width
            );
            double source_x = 0.0;
            double source_y = 0.0;
            double source_z = 0.0;
            const bool source_xyz_valid =
                uv_corners_valid && sample_source_xyz(
                    mapped_row,
                    mapped_col,
                    request,
                    source_x,
                    source_y,
                    source_z
                );
            const double dx = source_x - target_x;
            const double dy = source_y - target_y;
            const double dz = source_z - target_z;
            const double xy_squared = dx * dx + dy * dy;
            const double distance = std::sqrt(xy_squared + dz * dz);
            const bool distance_valid =
                source_xyz_valid && std::isfinite(distance) &&
                distance <= request.max_distance;

            std::int64_t source_index = -1;
            if (
                target_pixel_valid && distance_valid &&
                label_index(mapped_row, mapped_col, request, source_index)
            ) {
                const std::uint8_t source_validity =
                    request.has_source_validity != 0
                        ? request.source_validity[source_index]
                        : 255;
                if (source_validity != 0) {
                    request.output_source_indices[local] = source_index;
                    request.output_validity[local] =
                        source_validity == 128 ? 128 : 255;
                    request.output_distances[local] = distance;
                    ++result->measured_pixels;
                    if (source_validity == 128) {
                        ++result->inherited_filled_pixels;
                    }
                }
            }

            if (
                request.fill_seams != 0 && target_pixel_valid &&
                request.output_validity[local] == 0
            ) {
                const double seam_row = bilinear(
                    request.filled_uv_rows, position, request.target_width
                );
                const double seam_col = bilinear(
                    request.filled_uv_cols, position, request.target_width
                );
                std::int64_t seam_index = -1;
                if (label_index(seam_row, seam_col, request, seam_index)) {
                    const bool seam_source_valid =
                        request.has_source_validity == 0 ||
                        request.source_validity[seam_index] != 0;
                    if (seam_source_valid) {
                        request.output_source_indices[local] = seam_index;
                        request.output_validity[local] = 128;
                        ++result->seam_filled_pixels;
                    }
                }
            }
        }
    }
    return 0;
}

}  // extern "C"
