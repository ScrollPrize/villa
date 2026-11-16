#pragma once
// FFTW plans owned by a handle: made and destroyed under fftw_planner_mutex() (FFTW's planner is not thread safe;
// executing a plan is, and needs no lock). make_fftw_plan(make) runs make() under the lock, so it may also set
// fftwf_plan_with_nthreads around its planner call.
#include <fftw3.h>

#include <mutex>
#include <utility>

#include "vc_tomography/fourier.hpp"

namespace vc_tomography {

namespace fftw_detail {
inline void destroy(fftwf_plan p) { fftwf_destroy_plan(p); }
inline void destroy(fftw_plan p) { fftw_destroy_plan(p); }
}  // namespace fftw_detail

template <class Plan>
class FftwPlanOwner {
  public:
    FftwPlanOwner() = default;
    explicit FftwPlanOwner(Plan p) : p_(p) {}
    ~FftwPlanOwner() { reset(); }
    FftwPlanOwner(const FftwPlanOwner&) = delete;
    FftwPlanOwner& operator=(const FftwPlanOwner&) = delete;
    FftwPlanOwner(FftwPlanOwner&& o) noexcept : p_(std::exchange(o.p_, nullptr)) {}
    FftwPlanOwner& operator=(FftwPlanOwner&& o) noexcept {
        if (this != &o) reset(), p_ = std::exchange(o.p_, nullptr);
        return *this;
    }
    void reset() {
        if (!p_) return;
        std::lock_guard<std::mutex> lock(fftw_planner_mutex());
        fftw_detail::destroy(p_);
        p_ = nullptr;
    }
    Plan get() const { return p_; }
    operator Plan() const { return p_; }

  private:
    Plan p_ = nullptr;
};
using FftwfPlan = FftwPlanOwner<fftwf_plan>;
using FftwPlan = FftwPlanOwner<fftw_plan>;

// the plan make() returns, made under the planner lock; an Error when FFTW cannot plan it
template <class Make>
auto make_fftw_plan(Make&& make) -> FftwPlanOwner<decltype(make())> {
    std::lock_guard<std::mutex> lock(fftw_planner_mutex());
    const auto p = make();
    if (!p) throw Error("FFTW could not plan a transform");
    return FftwPlanOwner<decltype(make())>(p);
}

}  // namespace vc_tomography
