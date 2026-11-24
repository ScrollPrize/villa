#include "vc_tomography/tomo.hpp"

#include "vc_tomography/sino.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

#ifdef VC_TOMOGRAPHY_WITH_CUDA
#include "../cuda/cuda_utils.cuh"
#endif

namespace vc_tomography {

std::vector<std::string> tomo_algorithms() { return {"gridrec", "mlem", "sirt"}; }

bool is_tomo_method(const std::string& backprojector) {
    const auto names = tomo_algorithms();
    return std::find(names.begin(), names.end(), backprojector) != names.end();
}

TomoArgs tomo_args(const std::string& algorithm, const FilterParams& p) {
    TomoArgs a;
    a.algorithm = algorithm;
    if (!is_tomo_method(algorithm)) throw Error("unknown method '" + algorithm + "' (sirt, mlem, gridrec)");
    const bool gridrec = algorithm == "gridrec", sirt = algorithm == "sirt";
    const std::string known = gridrec ? "filter_name, filter_par"
                              : sirt  ? "num_iter, relax, norm, weighting, projector, init, init_value, init_divergence"
                                      : "num_iter, projector, init, init_value, init_divergence";
    for (const auto& [k, v] : p.values) {
        const std::string err = algorithm + ": unknown parameter '" + k + "' (" + known + ")";
        if (gridrec) {
            if (k == "filter_name") a.filter_name = v;
            else if (k == "filter_par") a.filter_par = ramp_filter_par(v);
            else throw Error(err);
        } else if (k == "num_iter") {
            const double n = p.num(k);
            if (!std::isfinite(n) || n < 0 || n > 2147483647.0 || n != std::floor(n))
                throw Error(algorithm + ": num_iter must be a nonnegative integer");
            a.num_iter = static_cast<int>(n);
        }
        else if (k == "init") a.init = v;
        else if (k == "init_value") a.init_value = static_cast<float>(p.num(k));
        else if (k == "init_divergence") {
            if (v != "on" && v != "off") throw Error(algorithm + ": init_divergence must be on or off");
            a.init_divergence = v == "on";
        }
        else if (k == "projector") {
            if (v == "chord") a.projector = 0;
            else if (v == "sf") a.projector = 1;
            else throw Error(algorithm + ": projector must be chord or sf");
        } else if (sirt && k == "relax") a.relax = static_cast<float>(p.num(k));
        else if (sirt && k == "weighting") {
            if (v != "mask" && v != "confidence") throw Error("sirt: weighting must be mask or confidence");
            a.confidence_weights = v == "confidence";
        }
        else if (sirt && k == "norm") {
            if (v != "rowsum") throw Error("sirt: norm must be rowsum; len2 has no supported stable relaxation bound");
        } else
            throw Error(err);
    }
    if (a.init != "constant" && a.init != "gridrec" && a.init != "fbp") throw Error(algorithm + ": init must be constant, gridrec or fbp");
    if (a.init_divergence && a.init != "fbp") throw Error(algorithm + ": init_divergence is of init=fbp");
    if (a.num_iter < 0) throw Error(algorithm + ": num_iter >= 0");
    if (!std::isfinite(a.init_value)) throw Error(algorithm + ": init_value must be finite");
    if (sirt && (!std::isfinite(a.relax) || a.relax <= 0.f || a.relax >= 2.f))
        throw Error("sirt: relax must be finite and strictly between 0 and 2");
    return a;
}

namespace {
// the pipeline backprojectors "sirt", "mlem", "gridrec": the whole slice, then the ROI crop x scale, outer circle
// cleared
class TomoBackprojector final : public ExternalBackprojector {
  public:
    explicit TomoBackprojector(TomoArgs a) : a_(std::move(a)) {}
    bool filtered() const override { return false; }
    bool measured_only() const override { return a_.algorithm != "gridrec"; }
    int fbp_start() const override { return a_.algorithm != "gridrec" && a_.init == "fbp" ? (a_.init_divergence ? 2 : 1) : 0; }
    void set_start(const float* image) override { start_ = image; }
    void slice(int, const float* rows, const float* weights, int n_angles, int n_det, const float* angles,
               const BackprojectParams& p, int N, const SliceRoi& roi, Device device, float* out) override {
        // Whole-slice methods receive one row per bin of the full turn, even when
        // nearest-bin gridding shifts the representative angle within a bin.
        a_.gridrec_angle_step = static_cast<float>(2 * M_PI / n_angles);
        const float c = p.center >= 0 ? p.center : 0.5f * (N - 1), rad = p.clip_radius >= 0 ? p.clip_radius : c;
        TomoGeom t;
        t.n_angles = n_angles; t.n_det = n_det; t.N = N; t.rot = p.rot_center;
        t.R = p.clip_outer_circle ? rad : 1e30f;
        const int rx = roi.x1 - roi.x0, ry = roi.y1 - roi.y0;
        const float* crop = nullptr;       // the ROI, rows of `pitch`
        std::size_t pitch = rx;
        if (device == Device::CPU) {
            img_host_.resize(static_cast<std::size_t>(N) * N);
            cpu::tomo_reconstruct(a_, t, rows, weights, angles, img_host_.data(), &ws_cpu_, start_);
            crop = img_host_.data() + static_cast<std::size_t>(roi.y0) * N + roi.x0;
            pitch = N;
        } else {
#ifdef VC_TOMOGRAPHY_WITH_CUDA
            img_.ensure(static_cast<std::size_t>(N) * N);
            cuda::tomo_reconstruct(a_, t, rows, weights, angles, img_.data(), &ws_cuda_, start_);
            crop_.ensure(static_cast<std::size_t>(rx) * ry);
            VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpy2D(crop_.data(), rx * sizeof(float),
                                            img_.data() + static_cast<std::size_t>(roi.y0) * N + roi.x0,
                                            N * sizeof(float), rx * sizeof(float), ry, cudaMemcpyDeviceToHost));
            crop = crop_.data();
#else
            throw Error("built without CUDA");
#endif
        }
        // pipeline units: the methods give attenuation per pixel -> x scale (1 / voxel cm) x unit_gain (1; legacy
        // reference mode: 1 / angular span, as the reference's FBP normalisation)
        const float units = p.scale * p.unit_gain;
#pragma omp parallel for schedule(static)
        for (int r = 0; r < ry; ++r)
            for (int q = 0; q < rx; ++q) {
                const float X = roi.x0 + q - c, Y = roi.y0 + r - c;
                const bool inside = !p.clip_outer_circle || X * X + Y * Y <= rad * rad;
                out[static_cast<std::size_t>(r) * rx + q] = inside ? crop[static_cast<std::size_t>(r) * pitch + q] * units : 0.f;
            }
    }

  private:
    TomoArgs a_;
    const float* start_ = nullptr;   // set_start
    std::vector<float> img_host_;
    std::unique_ptr<TomoWorkspace> ws_cpu_, ws_cuda_;
#ifdef VC_TOMOGRAPHY_WITH_CUDA
    cuda::DeviceBuffer<float> img_;
    cuda::PinnedBuffer<float> crop_;   // staging of the ROI's download
#endif
};
}  // namespace

std::shared_ptr<ExternalBackprojector> make_tomo_backprojector(const std::string& algorithm, const FilterParams& p) {
    return std::make_shared<TomoBackprojector>(tomo_args(algorithm, p));
}

}  // namespace vc_tomography
