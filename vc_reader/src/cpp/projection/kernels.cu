#include "plan.hpp"
#include "../cuda_buffer.hpp"
namespace vcr::projection {
template <class T>
__global__ void project(Raster g, const float* uv, const float* distance, const std::int64_t* ids, const std::int64_t* offsets, const T* label, T* out, bool* coverage, bool* conflict,
                        float* best, std::int64_t* winner, float* uvOut, bool* geometry)
{
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < g.count; i += std::size_t(blockDim.x) * gridDim.x)
        raster(g, uv, distance, ids, offsets, label, out, coverage, conflict, best, winner, uvOut, geometry, i);
}
void launch(const Plan& p, Raster g, const void* label, ElementType type, void* out, bool* coverage, bool* conflict, float* best, std::int64_t* winner, float* uvOut, bool* geometry, int id)
{
    auto upload = [&](const auto& values) {
        auto a = device::allocate(values.size() * sizeof(values[0]), id);
        if (a->bytes) device::check(cudaMemcpyAsync(a->ptr, values.data(), a->bytes, cudaMemcpyHostToDevice, device::stream()));
        return a;
    };
    auto uv = upload(p.uv), distance = upload(p.distance), ids = upload(p.ids), offsets = upload(p.offsets);
    type.visit_load([&](auto load) {
        using T = decltype(load(static_cast<const std::uint8_t*>(nullptr)));
        project<<<unsigned((g.count + 127) / 128), 128, 0, device::stream()>>>(g, static_cast<const float*>(uv->ptr), static_cast<const float*>(distance->ptr), static_cast<const std::int64_t*>(ids->ptr),
                                                                               static_cast<const std::int64_t*>(offsets->ptr), static_cast<const T*>(label), static_cast<T*>(out), coverage, conflict, best, winner,
                                                                               uvOut, geometry);
    });
    device::check(cudaGetLastError());
    device::complete();
}
}  // namespace vcr::projection
