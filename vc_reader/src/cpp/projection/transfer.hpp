#pragma once
#include "../grid/points.hpp"
#ifdef __CUDACC__
#define VCR_TRANSFER_HD __host__ __device__
#else
#define VCR_TRANSFER_HD
#endif
namespace vcr::projection {
struct RasterRequest {
    const float* uv_rows;
    const float* uv_cols;
    const std::uint8_t* uv_valid;
    const float* filled_uv_rows;
    const float* filled_uv_cols;
    const float* filled_uv_anchor_distance;
    const std::uint8_t* source_validity;
    std::int64_t* output_source_indices;
    std::uint8_t* output_validity;
    float* output_distances;

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

    float label_offset_y;
    float label_offset_x;
    float max_distance;
    float max_seam_distance;
    std::uint32_t fill_seams;
    std::uint32_t has_seam_limit;
    std::uint32_t has_source_validity;
    grid::GridView target, source;
};

struct RasterResult {
    std::int64_t target_surface_valid;
    std::int64_t measured_pixels;
    std::int64_t seam_filled_pixels;
    std::int64_t seam_blocked_pixels;
    std::int64_t inherited_filled_pixels;
};

VCR_TRANSFER_HD inline std::int64_t flat_index(std::int64_t row,std::int64_t col,std::int64_t width) {
    return row*width+col;
}

VCR_TRANSFER_HD inline bool label_index(
    const float mapped_row,
    const float mapped_col,
    const RasterRequest& request,
    std::int64_t& index
) {
    if (!std::isfinite(mapped_row) || !std::isfinite(mapped_col)) {
        return false;
    }
    const float scaled_row =
        mapped_row * static_cast<float>(request.label_height) /
        static_cast<float>(request.source_height) - request.label_offset_y;
    const float scaled_col =
        mapped_col * static_cast<float>(request.label_width) /
        static_cast<float>(request.source_width) - request.label_offset_x;
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

VCR_TRANSFER_HD inline void transferPixel(const RasterRequest& request,RasterResult* result,std::int64_t local) {
    const auto tile_width=request.col_end-request.col_start;
    const auto row=request.row_start+local/tile_width,col=request.col_start+local%tile_width;
    const float invalid_distance=NAN;
    request.output_source_indices[local] = -1;
    request.output_validity[local] = 0;
    request.output_distances[local] = invalid_distance;

    const float gy=sampling::pixelCenter(row,request.target_height,request.output_height);
    const float gx=sampling::pixelCenter(col,request.target_width,request.output_width);
    const float rc[2]{gy,gx};const std::int64_t shape[2]{request.target_height,request.target_width};
    const auto field=[&](const float* values){return sampling::clamped<2>(values,shape,rc);};
    float target[3]{},source[3]{};
    grid::PointMap targetMap;targetMap.height=int(request.target_height);targetMap.width=int(request.target_width);
    const bool target_pixel_valid=grid::point(targetMap,request.target,
        fminf(gx,request.target_width-1.f),fminf(gy,request.target_height-1.f),target);
    float target_x=target[0],target_y=target[1],target_z=target[2];
    if(target_pixel_valid)++result->target_surface_valid;
    bool uv_corners_valid=true;
    sampling::clampedCorners<2>(shape,rc,[&](std::size_t at,float){uv_corners_valid&=request.uv_valid[at]!=0;});
    float mapped_row=field(request.uv_rows);
    float mapped_col=field(request.uv_cols);
    grid::PointMap sourceMap;sourceMap.height=int(request.source_height);sourceMap.width=int(request.source_width);
    const bool source_xyz_valid=uv_corners_valid&&grid::point(sourceMap,request.source,mapped_col,mapped_row,source);
    float source_x=source[0],source_y=source[1],source_z=source[2];
    const float dx = source_x - target_x;
    const float dy = source_y - target_y;
    const float dz = source_z - target_z;
    const float xy_squared = dx * dx + dy * dy;
    const float distance = std::sqrt(xy_squared + dz * dz);
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
        // Reject seam fill beyond the selected anchor distance limit.
        if (request.has_seam_limit != 0) {
            const float anchor_distance = field(request.filled_uv_anchor_distance);
            if (
                !std::isfinite(anchor_distance) ||
                anchor_distance > request.max_seam_distance
            ) {
                ++result->seam_blocked_pixels;
                return;
            }
        }
        const float seam_row = field(request.filled_uv_rows);
        const float seam_col = field(request.filled_uv_cols);
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
void transfer(const RasterRequest&,RasterResult&,int device=-1);
#ifdef VC_READER_CUDA
void transferCuda(const RasterRequest&,RasterResult&,int);
#endif
}
#undef VCR_TRANSFER_HD
