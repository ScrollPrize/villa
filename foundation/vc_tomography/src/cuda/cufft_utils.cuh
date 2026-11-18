#pragma once
#include <cufft.h>

#include <string>
#include <utility>

#include "cuda_utils.cuh"
#include "vc_tomography/common.hpp"

namespace vc_tomography::cuda {

inline void cufft_check(cufftResult r, const char* what) {
    if (r != CUFFT_SUCCESS) throw Error(std::string("cuFFT error ") + std::to_string(static_cast<int>(r)) + " in " + what);
}

// `b` with room for n elements. Growing frees the old block, which work still queued on the device may read or
// write (the transforms return before they have run): the device is brought to rest first.
template <class T>
T* grown(DeviceBuffer<T>& b, std::size_t n) {
    if (n > b.size()) {
        VC_TOMOGRAPHY_CUDA_CHECK(cudaDeviceSynchronize());
        b.resize(n);
    }
    return b.data();
}

// Device memory the calling thread's transforms share, kept between calls (and given back by ThreadScratch::release):
// `work`, the work area of its cuFFT plans (CufftPlan), and two pairs of buffers for what a transform is made of and
// made into (padded rows or frames, their spectra). A reconstructor runs a dozen plans and half a dozen filters one
// after another on one stream: each holding its own work area and buffers cost it several GB of the device.
struct FftScratch {
    DeviceBuffer<char> work;
    DeviceBuffer<float> real[2];
    DeviceBuffer<cufftComplex> cplx[2];
};
inline FftScratch& fft_scratch() {
    thread_local FftScratch s;
    thread_local ThreadScratch scratch(s);
    return s;
}

// A cuFFT plan that works in the calling thread's shared work area (fft_scratch().work) instead of one of its own.
// The plans of a thread must run one after another (one stream, or streams that wait for each other): bind() gives
// the handle ready for the next execution, on `stream` and with the work area as it now is.
class CufftPlan {
  public:
    CufftPlan() = default;
    ~CufftPlan() { destroy(); }
    CufftPlan(const CufftPlan&) = delete;
    CufftPlan& operator=(const CufftPlan&) = delete;
    CufftPlan(CufftPlan&& o) noexcept : h_(o.h_), work_(o.work_), made_(o.made_) { o.made_ = false; }
    CufftPlan& operator=(CufftPlan&& o) noexcept {
        if (this != &o) {
            destroy();
            h_ = o.h_; work_ = o.work_; made_ = o.made_;
            o.made_ = false;
        }
        return *this;
    }
    explicit operator bool() const { return made_; }
    // `batch` transforms of rank dimensions n; the layout as cufftPlanMany's (null: contiguous)
    void make(int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch,
              const char* what) {
        destroy();
        cufft_check(cufftCreate(&h_), what);
        made_ = true;
        cufft_check(cufftSetAutoAllocation(h_, 0), what);
        cufft_check(cufftMakePlanMany(h_, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, &work_), what);
    }
    void make(int rank, int* n, cufftType type, int batch, const char* what) { make(rank, n, nullptr, 1, 0, nullptr, 1, 0, type, batch, what); }
    void destroy() {
        if (made_) cufftDestroy(h_);
        made_ = false;
    }
    cufftHandle bind(cudaStream_t stream) {
        cufft_check(cufftSetWorkArea(h_, grown(fft_scratch().work, work_)), "cufftSetWorkArea");
        cufft_check(cufftSetStream(h_, stream), "cufftSetStream");
        return h_;
    }

  private:
    cufftHandle h_ = 0;
    std::size_t work_ = 0;
    bool made_ = false;
};

}  // namespace vc_tomography::cuda
