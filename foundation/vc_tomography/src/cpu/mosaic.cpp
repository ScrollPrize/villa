// vc_tomography/mosaic.hpp: the kernels that join a mosaic's tiles in one sinogram, on the CPU.
#include "vc_tomography/mosaic.hpp"
#include "../common/mosaic_rebin.hpp"
#include "../common/elementwise_core.hpp"
#include "each.hpp"

namespace vc_tomography::cpu {

void tile_rows(const float* s, const float* w, const float* gain, int n_rows, int n, int stride, float* out_s, float* out_w) {
    each(static_cast<std::size_t>(n_rows) * n, elementwise::TileRows{s, w, gain, n, stride, out_s, out_w});
}
void rebin_tile_rows(CanvasRebin grid, const float* s, const float* w, const float* gain, int n_rows, int n, int stride, float* out_s, float* out_w) {
    each(static_cast<std::size_t>(n_rows) * n, elementwise::RebinTileRows{grid, s, w, gain, n_rows, n, stride, out_s, out_w});
}
void complete_views(const CanvasOpposite* plan, const float* sums, const float* weights, int rows, int width, float* out, float* support) {
    each(static_cast<std::size_t>(rows) * width, elementwise::CompleteViews{plan, sums, weights, width, out, support});
}
void pad_measured(const CanvasOpposite* plan, const float* sino, int rows, int width, int padded, int split, float* out) {
    each(static_cast<std::size_t>(rows) * padded, elementwise::PadMeasured{plan, sino, width, padded, split, out});
}
void gather_rows(CanvasGather grid, const float* s, const float* w, int rows, int n, int stride, float* out_s, float* out_w) {
    each(static_cast<std::size_t>(rows) * n, elementwise::GatherRows{grid, s, w, rows, n, stride, out_s, out_w});
}
void rebin_placed_rows(const CanvasBracket* brackets, const float* s, const float* w, int n_rows, int n, int stride, float* out_s, float* out_w) {
    each(static_cast<std::size_t>(n_rows) * n, elementwise::RebinPlacedRows{brackets, s, w, n, stride, out_s, out_w});
}
void place_rows(const float* rows, const std::int32_t* first, float scale, bool nonneg, int n_rows, int n, int stride, float* canvas, int width) {
    each(static_cast<std::size_t>(n_rows) * n, elementwise::PlaceRows{rows, first, scale, nonneg, n, stride, width, canvas});
}
void place_exposure(const float* signal, const float* weight, const std::int32_t* first, const float* quality, float level,
                    int rows, int n, int stride, float* sums, float* support, int width) {
    each(static_cast<std::size_t>(rows) * n, elementwise::PlaceExposure{signal, weight, first, quality, level, n, stride, sums, support, width});
}
void blend_region(const CanvasDisc* discs, int count, int focus, SliceRoi region, SliceRoi output,
                  const float* image, float* result) {
    each(static_cast<std::size_t>(region.y1 - region.y0) * (region.x1 - region.x0),
         elementwise::BlendRegion{discs, count, focus, region, output, image, result});
}
void fold_opposite(float* S, int n_rows, int width) {
    each(static_cast<std::size_t>(n_rows / 2) * width, elementwise::FoldOpposite{S, n_rows / 2, width});
}

}  // namespace vc_tomography::cpu
