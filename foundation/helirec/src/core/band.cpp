#include "helirec/band.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef HELIREC_WITH_CUDA
#include "../cuda/cuda_utils.cuh"
#endif

namespace helirec {

struct BandSampler::Impl {
    Device dev = Device::CPU;
    std::size_t n = 0;
    int K = 1;
    std::vector<int32_t> zlo;            // host copy (per-slice ranges)
    BandBox box;
    int h_first = 0, h_last = -1;
    std::vector<int> last_h;             // per set: last slice received
    std::vector<char> have_last;
    int prepared_h = 0;                  // slice whose pixel list is current
    bool prepared = false;
    std::size_t list_count = 0;
    std::size_t n_bp = 0, n_sampled = 0;
    // CPU storage
    std::vector<float> fz, y, x, vals;   // vals: (K, n)
    std::vector<std::vector<float>> img; // 2 per set
    std::vector<uint8_t> mask;
    std::vector<int32_t> list;
#ifdef HELIREC_WITH_CUDA
    cuda::DeviceBuffer<float> d_fz, d_y, d_x, d_vals;
    std::vector<cuda::DeviceBuffer<float>> d_img;
    cuda::DeviceBuffer<uint8_t> d_mask;
    cuda::DeviceBuffer<int32_t> d_list;
    cuda::DeviceBuffer<float2> d_cs;
    int cs_angles = 0;
#endif

    std::pair<std::size_t, std::size_t> range(int h) const {
        auto lo = std::lower_bound(zlo.begin(), zlo.end(), h), hi = std::upper_bound(zlo.begin(), zlo.end(), h);
        return {static_cast<std::size_t>(lo - zlo.begin()), static_cast<std::size_t>(hi - zlo.begin())};
    }
    std::size_t npx() const { return static_cast<std::size_t>(box.w) * box.h; }
};

BandSampler::BandSampler(Device device, const int32_t* zlo, const float* fz, const float* y, const float* x,
                         std::size_t n, int grid_size, int n_sets)
    : impl_(std::make_unique<Impl>()) {
    auto& I = *impl_;
    I.dev = device;
    I.n = n;
    I.K = n_sets;
    if (!n) throw Error("BandSampler: no points");
    if (n_sets < 1) throw Error("BandSampler: n_sets must be >= 1");
    I.zlo.assign(zlo, zlo + n);
    if (!std::is_sorted(I.zlo.begin(), I.zlo.end())) throw Error("BandSampler: points must be sorted by zlo");
    I.h_first = I.zlo.front();
    I.h_last = I.zlo.back() + 1;
    I.last_h.assign(n_sets, 0);
    I.have_last.assign(n_sets, 0);
    int x0 = grid_size, y0 = grid_size, x1 = -1, y1 = -1;
    for (std::size_t k = 0; k < n; ++k) {
        if (!(fz[k] >= 0.f && fz[k] <= 1.f)) throw Error("BandSampler: fz outside [0, 1]");
        if (!(x[k] >= 0.f && y[k] >= 0.f && x[k] <= grid_size - 1 && y[k] <= grid_size - 1))
            throw Error("BandSampler: point outside the slice grid");
        const int ix = std::min(static_cast<int>(std::floor(x[k])), grid_size - 2);
        const int iy = std::min(static_cast<int>(std::floor(y[k])), grid_size - 2);
        x0 = std::min(x0, ix); x1 = std::max(x1, ix + 1);
        y0 = std::min(y0, iy); y1 = std::max(y1, iy + 1);
    }
    I.box = {x0, y0, x1 - x0 + 1, y1 - y0 + 1, grid_size};
    const std::size_t npx = I.npx();
    if (device == Device::CPU) {
        I.fz.assign(fz, fz + n); I.y.assign(y, y + n); I.x.assign(x, x + n);
        I.vals.assign(n * n_sets, std::nanf(""));
        I.img.assign(2 * n_sets, std::vector<float>(npx, 0.f));
        I.mask.assign(npx, 0);
        I.list.resize(npx);
    } else {
#ifdef HELIREC_WITH_CUDA
        I.d_fz.upload(fz, n); I.d_y.upload(y, n); I.d_x.upload(x, n);
        std::vector<float> nan(n * n_sets, std::nanf(""));
        I.d_vals.upload(nan.data(), nan.size());
        I.d_img.resize(2 * n_sets);
        for (auto& b : I.d_img) b.resize(npx);
        I.d_mask.resize(npx);
        I.d_list.resize(npx);
        HELIREC_CUDA_CHECK(cudaDeviceSynchronize());
#else
        throw Error("built without CUDA");
#endif
    }
}

BandSampler::~BandSampler() = default;

std::size_t BandSampler::size() const { return impl_->n; }
int BandSampler::n_sets() const { return impl_->K; }
int BandSampler::h_first() const { return impl_->h_first; }
int BandSampler::h_last() const { return impl_->h_last; }
const BandBox& BandSampler::box() const { return impl_->box; }
Device BandSampler::device() const { return impl_->dev; }
std::size_t BandSampler::pixels_backprojected() const { return impl_->n_bp; }
std::size_t BandSampler::points_sampled() const { return impl_->n_sampled; }

void BandSampler::reset() {
    auto& I = *impl_;
    std::fill(I.have_last.begin(), I.have_last.end(), 0);
    I.prepared = false;
    I.n_bp = I.n_sampled = 0;
    if (I.dev == Device::CPU) std::fill(I.vals.begin(), I.vals.end(), std::nanf(""));
#ifdef HELIREC_WITH_CUDA
    else {
        std::vector<float> nan(I.n * I.K, std::nanf(""));
        I.d_vals.upload(nan.data(), nan.size());
        HELIREC_CUDA_CHECK(cudaDeviceSynchronize());
    }
#endif
}

void BandSampler::slice(int h, const float* filtered, int n_angles, int n_det, const float* angles_host,
                        const BackprojectParams& p, int set) {
    auto& I = *impl_;
    if (set < 0 || set >= I.K) throw Error("BandSampler::slice: set out of range");
    const bool chained = I.have_last[set] && I.last_h[set] == h - 1;  // slice h - 1 is in the other image
    I.last_h[set] = h;
    I.have_last[set] = 1;
    if (h < I.h_first || h > I.h_last) return;
    const auto rp = I.range(h - 1), rc = I.range(h);  // points reading slice h as upper / lower slice
    if (rp.first == rp.second && rc.first == rc.second) return;
    const std::size_t npx = I.npx();
    const int cur = 2 * set + (h & 1), prev = 2 * set + ((h - 1) & 1);
    float* vals_set = nullptr;
    if (I.dev == Device::CPU) {
        if (!I.prepared || I.prepared_h != h) {  // pixel band of slice h (shared by every set)
            std::fill(I.mask.begin(), I.mask.end(), 0);
            cpu::band_mark(I.y.data(), I.x.data(), rp.first, rp.second, I.box, I.mask.data());
            cpu::band_mark(I.y.data(), I.x.data(), rc.first, rc.second, I.box, I.mask.data());
            I.list_count = cpu::band_compact(I.mask.data(), npx, I.list.data());
            I.prepared_h = h;
            I.prepared = true;
        }
        cpu::backproject_list(filtered, angles_host, n_angles, n_det, I.list.data(), I.list_count, I.box, p,
                              I.img[cur].data());
        I.n_bp += I.list_count;
        vals_set = I.vals.data() + static_cast<std::size_t>(set) * I.n;
        if (chained && rp.second > rp.first) {
            cpu::band_sample(I.img[prev].data(), I.img[cur].data(), I.fz.data(), I.y.data(), I.x.data(), rp.first,
                             rp.second, I.box, vals_set);
            I.n_sampled += rp.second - rp.first;
        }
        return;
    }
#ifdef HELIREC_WITH_CUDA
    if (I.cs_angles != n_angles) {
        std::vector<float2> cs(n_angles);
        for (int a = 0; a < n_angles; ++a) cs[a] = make_float2(std::cos(angles_host[a]), std::sin(angles_host[a]));
        I.d_cs.upload(cs.data(), cs.size());
        I.cs_angles = n_angles;
    }
    if (!I.prepared || I.prepared_h != h) {
        HELIREC_CUDA_CHECK(cudaMemsetAsync(I.d_mask.data(), 0, npx));
        cuda::band_mark(I.d_y.data(), I.d_x.data(), rp.first, rp.second, I.box, I.d_mask.data());
        cuda::band_mark(I.d_y.data(), I.d_x.data(), rc.first, rc.second, I.box, I.d_mask.data());
        I.list_count = cuda::band_compact(I.d_mask.data(), npx, I.d_list.data());
        I.prepared_h = h;
        I.prepared = true;
    }
    cuda::backproject_list(filtered, I.d_cs.data(), n_angles, n_det, I.d_list.data(), I.list_count, I.box, p,
                           I.d_img[cur].data());
    I.n_bp += I.list_count;
    if (chained && rp.second > rp.first) {
        cuda::band_sample(I.d_img[prev].data(), I.d_img[cur].data(), I.d_fz.data(), I.d_y.data(), I.d_x.data(),
                          rp.first, rp.second, I.box, I.d_vals.data() + static_cast<std::size_t>(set) * I.n);
        I.n_sampled += rp.second - rp.first;
    }
#endif
}

void BandSampler::values(float* out, int set) const {
    const auto& I = *impl_;
    if (set < 0 || set >= I.K) throw Error("BandSampler::values: set out of range");
    if (I.dev == Device::CPU) {
        std::memcpy(out, I.vals.data() + static_cast<std::size_t>(set) * I.n, I.n * sizeof(float));
        return;
    }
#ifdef HELIREC_WITH_CUDA
    HELIREC_CUDA_CHECK(cudaDeviceSynchronize());
    HELIREC_CUDA_CHECK(cudaMemcpy(out, I.d_vals.data() + static_cast<std::size_t>(set) * I.n, I.n * sizeof(float),
                                  cudaMemcpyDeviceToHost));
#endif
}

}  // namespace helirec
