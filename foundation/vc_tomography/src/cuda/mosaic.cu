// vc_tomography/mosaic.hpp: the kernels that join a mosaic's tiles in one sinogram, on CUDA.
#include "cuda_utils.cuh"
#include "vc_tomography/mosaic.hpp"
#include "../common/elementwise_core.hpp"

namespace vc_tomography::cuda {

void tile_rows(const float* s, const float* w, const float* gain, int n_rows, int n, int stride, float* out_s, float* out_w, void* stream) {
    each(static_cast<std::size_t>(n_rows) * n, elementwise::TileRows{s, w, gain, n, stride, out_s, out_w}, as_stream(stream));
}
void place_rows(const float* rows, const std::int32_t* first, float scale, bool nonneg, int n_rows, int n, int stride, float* canvas, int width,
                void* stream) {
    each(static_cast<std::size_t>(n_rows) * n, elementwise::PlaceRows{rows, first, scale, nonneg, n, stride, width, canvas}, as_stream(stream));
}
void fold_opposite(float* S, int n_rows, int width, void* stream) {
    each(static_cast<std::size_t>(n_rows / 2) * width, elementwise::FoldOpposite{S, n_rows / 2, width}, as_stream(stream));
}

}  // namespace vc_tomography::cuda
