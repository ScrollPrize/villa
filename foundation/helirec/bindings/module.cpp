// nanobind module `helirec._core`.
// Every kernel takes host NumPy arrays and a `device` ("cpu" | "cuda"); the CUDA
// path uploads, runs and downloads, so the two backends can be compared 1:1.
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>

#include "helirec/fourier.hpp"
#include "helirec/ghbp.hpp"
#include "helirec/filters.hpp"
#include "helirec/tomo.hpp"
#include <nanobind/stl/function.h>
#include "helirec/sino.hpp"
#include "helirec/raw_reader.hpp"
#include "helirec/delta3d.hpp"
#include "helirec/pipeline.hpp"
#include "helirec/accumulate.hpp"
#include "helirec/band.hpp"
#include <nanobind/stl/optional.h>
#include <nanobind/stl/array.h>
#include "helirec/helical.hpp"
#include "helirec/kernels.hpp"
#include "ndarray_utils.hpp"

#ifdef HELIREC_WITH_CUDA
#include "../src/cuda/cuda_utils.cuh"
#endif

#include <cstring>

namespace nb = nanobind;
using namespace nb::literals;
using namespace helirec;
using namespace helirec::py;

namespace {

HostOut flatfield_py(HostIn raw, HostIn dark, HostIn flat, float threshold, bool take_log,
                     const std::string& device) {
    require(raw.ndim() >= 2, "raw must be (n_proj, ...) with frame dims");
    std::size_t n_pix = dark.size();
    require(flat.size() == n_pix, "dark and flat must have the same size");
    require(raw.size() % n_pix == 0, "raw frame size does not match dark/flat size");
    const std::size_t n_proj = raw.size() / n_pix;
    HostOut out = new_array(shape_of(raw));
    FlatFieldParams p{threshold, take_log};

    if (parse_device(device) == Device::CPU) {
        nb::gil_scoped_release nogil;
        cpu::flatfield(raw.data(), dark.data(), flat.data(), data(out), n_proj, n_pix, p);
        return out;
    }
#ifdef HELIREC_WITH_CUDA
    {
        nb::gil_scoped_release nogil;
        cuda::DeviceBuffer<float> d_raw, d_dark, d_flat, d_out(raw.size());
        d_raw.upload(raw.data(), raw.size());
        d_dark.upload(dark.data(), n_pix);
        d_flat.upload(flat.data(), n_pix);
        cuda::flatfield(d_raw.data(), d_dark.data(), d_flat.data(), d_out.data(), n_proj, n_pix, p);
        d_out.download(data(out));
    }
#endif
    return out;
}

int projector_id(const std::string& p) {
    if (p == "chord") return 0;
    if (p == "sf") return 1;
    throw Error("projector must be 'chord' or 'sf'");
}

HostOut backproject_py(HostIn sino, HostIn angles, int ny, int nx, float rot_center, float scale,
                       float center, int x0, int y0, bool clip_outer_circle, float clip_radius,
                       const std::string& device, int interp, float u_scale) {
    require(sino.ndim() == 2, "sino must be (n_angles, n_det)");
    const int n_angles = static_cast<int>(sino.shape(0)), n_det = static_cast<int>(sino.shape(1));
    require(angles.size() == static_cast<std::size_t>(n_angles), "angles length != sino rows");
    HostOut out = new_array({static_cast<std::size_t>(ny), static_cast<std::size_t>(nx)});
    BackprojectParams p;
    p.rot_center = rot_center; p.scale = scale; p.center = center; p.x0 = x0; p.y0 = y0;
    p.clip_outer_circle = clip_outer_circle; p.clip_radius = clip_radius;
    p.interp = interp; p.u_scale = u_scale;
    if (parse_device(device) == Device::CPU) {
        nb::gil_scoped_release nogil;
        cpu::backproject(sino.data(), angles.data(), data(out), n_angles, n_det, ny, nx, p);
        return out;
    }
#ifdef HELIREC_WITH_CUDA
    {
        nb::gil_scoped_release nogil;
        cuda::DeviceBuffer<float> d_sino, d_out(static_cast<std::size_t>(ny) * nx);
        d_sino.upload(sino.data(), sino.size());
        cuda::backproject(d_sino.data(), angles.data(), d_out.data(), n_angles, n_det, ny, nx, p);
        d_out.download(data(out));
    }
#endif
    return out;
}

// Python-facing FourierFilter: owns a CPU or CUDA implementation.
class PyFourierFilter {
  public:
    PyFourierFilter(int ndim, int h, int w, int ph, int pw, int pad_top, int pad_left,
                    const std::string& mode, HostIn filter, const std::string& device)
        : device_(parse_device(device)) {
        FourierFilterSpec s;
        s.ndim = ndim; s.h = h; s.w = w; s.ph = ph; s.pw = pw;
        s.pad_top = pad_top; s.pad_left = pad_left; s.mode = pad_mode_from_string(mode);
        s.filter.assign(filter.data(), filter.data() + filter.size());
        if (device_ == Device::CPU) impl_ = cpu::make_fourier_filter(s);
#ifdef HELIREC_WITH_CUDA
        else impl_ = cuda::make_fourier_filter(s);
#endif
    }
    HostOut apply(HostIn stack) {
        const auto& s = impl_->spec();
        const std::size_t frame = static_cast<std::size_t>(s.h) * s.w;
        require(stack.size() % frame == 0 && stack.ndim() >= 2 &&
                    stack.shape(stack.ndim() - 1) == static_cast<std::size_t>(s.w) &&
                    stack.shape(stack.ndim() - 2) == static_cast<std::size_t>(s.h),
                "stack must be (..., h, w) matching the filter spec");
        const int n = static_cast<int>(stack.size() / frame);
        HostOut out = new_array(shape_of(stack));
        nb::gil_scoped_release nogil;
        if (device_ == Device::CPU) {
            impl_->apply(stack.data(), data(out), n);
        }
#ifdef HELIREC_WITH_CUDA
        else {
            cuda::DeviceBuffer<float> d(stack.size());
            d.upload(stack.data(), stack.size());
            impl_->apply(d.data(), d.data(), n);
            d.download(data(out));
        }
#endif
        return out;
    }

    PyFourierFilter(const FourierFilterSpec& s, const std::string& device) : device_(parse_device(device)) {
        if (device_ == Device::CPU) impl_ = cpu::make_fourier_filter(s);
#ifdef HELIREC_WITH_CUDA
        else impl_ = cuda::make_fourier_filter(s);
#endif
    }
    nb::dict info() const {
        const auto& s = impl_->spec();
        nb::dict d;
        d["ndim"] = s.ndim; d["h"] = s.h; d["w"] = s.w; d["ph"] = s.ph; d["pw"] = s.pw;
        d["pad_top"] = s.pad_top; d["pad_left"] = s.pad_left; d["complex"] = !s.filter_imag.empty();
        return d;
    }

  private:
    Device device_;
    std::unique_ptr<FourierFilter> impl_;
};

// ------------------------------------------------------------------ filter chains from Python
std::string to_text(nb::handle v) {
    if (nb::isinstance<nb::bool_>(v)) return nb::cast<bool>(v) ? "1" : "0";
    if (nb::isinstance<nb::str>(v)) return nb::cast<std::string>(v);
    return nb::cast<std::string>(nb::str(v));
}

// [{"type": "paganin", "delta_beta": 1000}, ...]
FilterChain chain_from(nb::handle obj) {
    FilterChain chain;
    if (obj.is_none()) return chain;
    for (nb::handle h : nb::iter(obj)) {
        nb::dict d = nb::cast<nb::dict>(h);
        FilterSpec f;
        for (auto [k, v] : d) {
            const std::string key = nb::cast<std::string>(k);
            if (key == "type") f.type = nb::cast<std::string>(v);
            else f.params.values[key] = to_text(v);
        }
        if (f.type.empty()) throw Error("filter without 'type'");
        chain.push_back(std::move(f));
    }
    return chain;
}

// "python" filters: a function registered in helirec.filters (frames (n, H, W) float32, modified in place or
// returned); on CUDA with host=1 (default) it gets a NumPy copy, with host=0 the device array (DLPack: cupy / torch)
class PythonFilter final : public ProjectionFilter {
  public:
    PythonFilter(const FilterParams& p) : margin_(static_cast<int>(p.num("margin", 0))), host_(p.num("host", 1) != 0) {
        nb::gil_scoped_acquire gil;
        const std::string name = p.str("fn", "");
        nb::dict reg = nb::cast<nb::dict>(nb::module_::import_("helirec.filters").attr("_python_filters"));
        if (!reg.contains(name.c_str())) throw Error("python filter: '" + name + "' is not registered (helirec.filters.register)");
        fn_ = reg[name.c_str()];
        nb::dict kw;
        kwargs_ = kw;
        for (const auto& [k, v] : p.values)
            if (k != "fn" && k != "margin" && k != "host" && k != "stage") {
                try { kw[k.c_str()] = std::stod(v); } catch (...) { kw[k.c_str()] = v; }
            }
    }
    ~PythonFilter() override {
        nb::gil_scoped_acquire gil;
        fn_.reset();
        kwargs_.reset();
    }
    int margin_rows() const override { return margin_; }
    void apply(float* frames, int n, int H, int W, Device device) override {
        nb::gil_scoped_acquire gil;
        const std::size_t shape[3]{static_cast<std::size_t>(n), static_cast<std::size_t>(H), static_cast<std::size_t>(W)};
        const std::size_t count = shape[0] * shape[1] * shape[2];
        auto call = [&](nb::object arr) {
            nb::object r = fn_(arr, **nb::borrow<nb::dict>(kwargs_));
            return r;
        };
        if (device == Device::CPU) {
            nb::object arr = nb::cast(nb::ndarray<nb::numpy, float, nb::ndim<3>>(frames, 3, shape, nb::handle()),
                                      nb::rv_policy::reference);
            nb::object r = call(arr);
            if (!r.is_none() && !r.is(arr)) {
                auto out = nb::cast<nb::ndarray<const float, nb::c_contig, nb::device::cpu>>(r);
                if (out.size() != count) throw Error("python filter returned a different shape");
                std::memcpy(frames, out.data(), count * sizeof(float));
            }
            return;
        }
#ifdef HELIREC_WITH_CUDA
        HELIREC_CUDA_CHECK(cudaDeviceSynchronize());
        if (host_) {
            std::vector<float> buf(count);
            HELIREC_CUDA_CHECK(cudaMemcpy(buf.data(), frames, count * sizeof(float), cudaMemcpyDeviceToHost));
            nb::object arr = nb::cast(nb::ndarray<nb::numpy, float, nb::ndim<3>>(buf.data(), 3, shape, nb::handle()),
                                      nb::rv_policy::reference);
            nb::object r = call(arr);
            const float* src = buf.data();
            nb::ndarray<const float, nb::c_contig, nb::device::cpu> out;
            if (!r.is_none() && !r.is(arr)) {
                out = nb::cast<nb::ndarray<const float, nb::c_contig, nb::device::cpu>>(r);
                if (out.size() != count) throw Error("python filter returned a different shape");
                src = out.data();
            }
            HELIREC_CUDA_CHECK(cudaMemcpy(frames, src, count * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            int dev = 0;
            HELIREC_CUDA_CHECK(cudaGetDevice(&dev));
            nb::object arr = nb::cast(nb::ndarray<float, nb::ndim<3>, nb::device::cuda>(frames, 3, shape, nb::handle(),
                                                                                        nullptr, nb::dtype<float>(),
                                                                                        nb::device::cuda::value, dev),
                                      nb::rv_policy::reference);
            nb::object r = call(arr);
            if (!r.is_none() && !r.is(arr)) throw Error("python filter (host=0): modify the device array in place");
            HELIREC_CUDA_CHECK(cudaDeviceSynchronize());
        }
#endif
    }

  private:
    int margin_;
    bool host_;
    nb::object fn_, kwargs_;  // null until the constructor holds the GIL (it may run with the GIL released)
};

// external backprojectors implemented in Python (helirec.backprojectors.register)
class PyExternalBackprojector final : public ExternalBackprojector {
  public:
    explicit PyExternalBackprojector(nb::dict spec) {
        nb::gil_scoped_acquire gil;
        const std::string name = nb::cast<std::string>(spec["fn"]);
        nb::dict reg = nb::cast<nb::dict>(nb::module_::import_("helirec.backprojectors").attr("_registry"));
        if (!reg.contains(name.c_str())) throw Error("python backprojector '" + name + "' is not registered");
        nb::object entry = reg[name.c_str()];
        fn_ = entry.attr("fn");
        filtered_ = nb::cast<bool>(entry.attr("filtered"));
        host_ = spec.contains("host") ? nb::cast<bool>(spec["host"]) : nb::cast<bool>(entry.attr("host"));
        nb::dict kw;
        for (auto [k, v] : spec) {
            const std::string key = nb::cast<std::string>(k);
            if (key != "fn" && key != "host" && key != "type") kw[k] = v;
        }
        kwargs_ = kw;
    }
    ~PyExternalBackprojector() override {
        nb::gil_scoped_acquire gil;
        fn_.reset();
        kwargs_.reset();
    }
    bool filtered() const override { return filtered_; }
    void slice(int h, const float* rows, const float* weights, int n_angles, int n_det, const float* angles,
               const BackprojectParams& p, int grid_size, const SliceRoi& roi, Device device, float* out) override {
        nb::gil_scoped_acquire gil;
        const std::size_t shape[2]{static_cast<std::size_t>(n_angles), static_cast<std::size_t>(n_det)};
        const std::size_t count = shape[0] * shape[1];
        const std::size_t ry = roi.y1 - roi.y0, rx = roi.x1 - roi.x0;
        std::vector<float> hr, hw;
        nb::object r_arr, w_arr;
        auto host_view = [&](const float* ptr) {
            return nb::cast(nb::ndarray<nb::numpy, const float, nb::ndim<2>>(ptr, 2, shape, nb::handle()), nb::rv_policy::reference);
        };
        if (device == Device::CPU) {
            r_arr = host_view(rows);
            w_arr = host_view(weights);
        }
#ifdef HELIREC_WITH_CUDA
        else if (host_) {
            HELIREC_CUDA_CHECK(cudaDeviceSynchronize());
            hr.resize(count); hw.resize(count);
            HELIREC_CUDA_CHECK(cudaMemcpy(hr.data(), rows, count * sizeof(float), cudaMemcpyDeviceToHost));
            HELIREC_CUDA_CHECK(cudaMemcpy(hw.data(), weights, count * sizeof(float), cudaMemcpyDeviceToHost));
            r_arr = host_view(hr.data());
            w_arr = host_view(hw.data());
        } else {
            HELIREC_CUDA_CHECK(cudaDeviceSynchronize());
            int dev = 0;
            HELIREC_CUDA_CHECK(cudaGetDevice(&dev));
            auto dview = [&](const float* ptr) {
                return nb::cast(nb::ndarray<const float, nb::ndim<2>, nb::device::cuda>(ptr, 2, shape, nb::handle(), nullptr,
                                                                                         nb::dtype<float>(), nb::device::cuda::value, dev),
                                nb::rv_policy::reference);
            };
            r_arr = dview(rows);
            w_arr = dview(weights);
        }
#endif
        auto ang = new_array({static_cast<std::size_t>(n_angles)});
        std::memcpy(data(ang), angles, n_angles * sizeof(float));
        nb::dict ctx;
        ctx["angles"] = ang;
        ctx["rot_center"] = p.rot_center;
        ctx["scale"] = p.scale;
        ctx["unit_gain"] = p.unit_gain;
        ctx["center"] = p.center;
        ctx["grid_size"] = grid_size;
        ctx["roi"] = nb::make_tuple(roi.x0, roi.x1, roi.y0, roi.y1);
        ctx["clip_outer_circle"] = p.clip_outer_circle;
        ctx["clip_radius"] = p.clip_radius;
        ctx["view_height"] = h;
        ctx["device"] = device == Device::CUDA ? "cuda" : "cpu";
        nb::object res = fn_(r_arr, w_arr, ctx, **nb::borrow<nb::dict>(kwargs_));
        // result: (ry, rx) float32, NumPy or a CUDA array (DLPack)
        try {
            auto a = nb::cast<nb::ndarray<const float, nb::c_contig, nb::device::cpu>>(res);
            if (a.size() != ry * rx) throw Error("python backprojector returned a wrong shape");
            std::memcpy(out, a.data(), ry * rx * sizeof(float));
            return;
        } catch (const nb::cast_error&) {
        }
#ifdef HELIREC_WITH_CUDA
        auto a = nb::cast<nb::ndarray<const float, nb::c_contig, nb::device::cuda>>(res);
        if (a.size() != ry * rx) throw Error("python backprojector returned a wrong shape");
        HELIREC_CUDA_CHECK(cudaMemcpy(out, a.data(), ry * rx * sizeof(float), cudaMemcpyDeviceToHost));
#else
        throw Error("python backprojector must return a float32 NumPy array");
#endif
    }

  private:
    bool filtered_ = true, host_ = true;
    nb::object fn_, kwargs_;
};

HostOut copy_of(HostIn a) {
    HostOut out = new_array(shape_of(a));
    std::memcpy(data(out), a.data(), a.size() * sizeof(float));
    return out;
}

HostOut minus_log_py(HostIn x, float clip_min, float clip_max, const std::string& device) {
    HostOut out = copy_of(x);
    const Device dev = parse_device(device);
    nb::gil_scoped_release nogil;
    if (dev == Device::CPU) {
        cpu::minus_log(data(out), x.size(), clip_min, clip_max);
        return out;
    }
#ifdef HELIREC_WITH_CUDA
    cuda::DeviceBuffer<float> d;
    d.upload(data(out), x.size());
    cuda::minus_log(d.data(), x.size(), clip_min, clip_max);
    d.download(data(out));
#endif
    return out;
}

std::pair<HostOut, HostOut> rebalance_py(HostIn R, HostIn W, float rot_center, bool legacy, const std::string& device) {
    require(R.ndim() == 3 && W.ndim() == 3 && R.size() == W.size(), "R, W must be (n_proj, n_rows, n_cols)");
    HostOut r = copy_of(R), w = copy_of(W);
    const int np = static_cast<int>(R.shape(0)), nr = static_cast<int>(R.shape(1)), nc = static_cast<int>(R.shape(2));
    if (parse_device(device) == Device::CPU) {
        nb::gil_scoped_release nogil;
        cpu::rebalance(data(r), data(w), np, nr, nc, rot_center, legacy);
        return {r, w};
    }
#ifdef HELIREC_WITH_CUDA
    {
        nb::gil_scoped_release nogil;
        cuda::DeviceBuffer<float> dr, dw;
        dr.upload(data(r), R.size());
        dw.upload(data(w), W.size());
        cuda::rebalance(dr.data(), dw.data(), np, nr, nc, rot_center, legacy);
        dr.download(data(r));
        dw.download(data(w));
    }
#endif
    return {r, w};
}

HostOut ramp_filter_py(HostIn sino, float rot_center, int pad_factor, bool normalize, bool legacy, const std::string& device) {
    require(sino.ndim() == 2, "sino must be (n_proj, n_bins)");
    const int n_proj = static_cast<int>(sino.shape(0)), n_bins = static_cast<int>(sino.shape(1));
    const Device dev = parse_device(device);
    HostOut out = new_array(shape_of(sino));
    nb::gil_scoped_release nogil;
    const RampGeometry g = ramp_geometry(n_bins, n_proj, rot_center, pad_factor, legacy);
    const FourierFilterSpec spec = ramp_filter_spec(g, normalize ? ramp_scale(n_proj, legacy) : 1.f);
    if (dev == Device::CPU) {
        std::vector<float> padded(static_cast<std::size_t>(n_proj) * g.npad);
        cpu::ramp_pad(sino.data(), padded.data(), g);
        cpu::make_fourier_filter(spec)->apply(padded.data(), padded.data(), n_proj);
        cpu::ramp_crop(padded.data(), data(out), g);
        return out;
    }
#ifdef HELIREC_WITH_CUDA
    cuda::DeviceBuffer<float> d_sino, d_pad(static_cast<std::size_t>(n_proj) * g.npad), d_out(sino.size());
    d_sino.upload(sino.data(), sino.size());
    cuda::ramp_pad(d_sino.data(), d_pad.data(), g);
    cuda::make_fourier_filter(spec)->apply(d_pad.data(), d_pad.data(), n_proj);
    cuda::ramp_crop(d_pad.data(), d_out.data(), g);
    d_out.download(data(out));
#endif
    return out;
}

HostOut ghbp_py(HostIn sino, HostIn angles, int grid_size, float rot_center, float scale_factor, int legs,
                int reduction_steps, bool clip_outer_circle, double axis_source_meters,
                float voxel_size_microns, float whf_x, float whf_y, float fac, const std::string& device) {
    require(sino.ndim() == 2, "sino must be (n_angles, n_bins)");
    const int n_angles = static_cast<int>(sino.shape(0)), n_bins = static_cast<int>(sino.shape(1));
    require(angles.size() == static_cast<std::size_t>(n_angles), "angles length != sino rows");
    GhbpParams p;
    p.grid_size = grid_size; p.rot_center = rot_center; p.scale_factor = scale_factor; p.legs = legs;
    p.reduction_steps = reduction_steps; p.clip_outer_circle = clip_outer_circle;
    p.axis_source_meters = axis_source_meters; p.voxel_size_microns = voxel_size_microns;
    p.whf_x = whf_x; p.whf_y = whf_y; p.fac = fac;
    const Device dev = parse_device(device);
    HostOut out = new_array({static_cast<std::size_t>(grid_size), static_cast<std::size_t>(grid_size)});
    nb::gil_scoped_release nogil;
    GhbpPlan plan = make_ghbp_plan(angles.data(), n_angles, n_bins, p);
    if (dev == Device::CPU) {
        cpu::ghbp(plan, sino.data(), data(out));
        return out;
    }
#ifdef HELIREC_WITH_CUDA
    cuda::Ghbp g(plan);
    cuda::DeviceBuffer<float> d_sino, d_out(static_cast<std::size_t>(grid_size) * grid_size);
    d_sino.upload(sino.data(), sino.size());
    g.run(d_sino.data(), d_out.data());
    d_out.download(data(out));
#endif
    return out;
}

nb::dict ghbp_plan_info_py(HostIn angles, int n_bins, int grid_size, int legs, int reduction_steps) {
    GhbpParams p;
    p.grid_size = grid_size; p.legs = legs; p.reduction_steps = reduction_steps;
    GhbpPlan plan = make_ghbp_plan(angles.data(), static_cast<int>(angles.size()), n_bins, p);
    nb::list levels;
    for (const auto& L : plan.levels) {
        nb::dict d;
        d["h"] = L.h; d["w"] = L.w; d["grids_per_leg"] = L.g; d["reduction"] = L.red;
        levels.append(d);
    }
    nb::dict info;
    info["levels"] = levels;
    return info;
}

using DIn = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T, class A>
std::vector<T> vec(const A& a) { return std::vector<T>(a.data(), a.data() + a.size()); }

ScanData make_scan_data(DIn angles_deg, DIn z_translation_m, double pixel_size_m, int det_rows, int det_cols,
                        int shift_x, float x_readjustment, float z_readjustment, DIn control_A, HostIn flat_angles_deg,
                        HostIn dark, HostIn flats, HostIn weight, std::shared_ptr<RawReader> reader, IntIn file_index,
                        IntIn frame_index) {
    ScanData d;
    d.traj.angles_deg = vec<double>(angles_deg);
    d.traj.z_translation_m = vec<double>(z_translation_m);
    d.traj.pixel_size_m = pixel_size_m;
    d.traj.det_rows = det_rows;
    d.traj.det_cols = det_cols;
    d.traj.shift_x = shift_x;
    d.traj.x_readjustment = x_readjustment;
    d.traj.z_readjustment = z_readjustment;
    d.control_A = vec<double>(control_A);
    d.flat_info.angles_deg = vec<float>(flat_angles_deg);
    d.dark = vec<float>(dark);
    d.flats = vec<float>(flats);
    d.weight = vec<float>(weight);
    d.reader = std::move(reader);
    d.file_index = vec<int32_t>(file_index);
    d.frame_index = vec<int32_t>(frame_index);
    return d;
}

class PySliceReconstructor {
  public:
    PySliceReconstructor(std::vector<ScanData> scans, const nb::dict& c) {
        ReconConfig cfg;
        auto get = [&](const char* k) { return c.contains(k) ? nb::object(c[k]) : nb::object(); };
        auto f = [&](const char* k, auto& dst) {
            nb::object o = get(k);
            if (o.is_valid() && !o.is_none()) dst = nb::cast<std::decay_t<decltype(dst)>>(o);
        };
        std::string device = "cpu";
        f("device", device);
        cfg.device = parse_device(device);
        f("energy_kev", cfg.energy_kev);
        f("distance_m", cfg.distance_m);
        f("pixel_size_m", cfg.pixel_size_m);
        if (c.contains("filters")) cfg.filters = chain_from(c["filters"]);
        f("margin", cfg.margin);
        f("sld_fraction", cfg.sld.fraction);
        f("sld_length", cfg.sld.length);
        f("sld_bin", cfg.sld.bin);
        f("sld_stride", cfg.sld.stride);
        f("n_grid", cfg.n_grid);
        f("angular_tolerance_steps", cfg.angular_tolerance_steps);
        f("luminosity_repartition", cfg.luminosity_repartition);
        f("x_interp", cfg.x_interp);
        f("mend_threshold", cfg.mend_threshold);
        f("legacy_reference", cfg.legacy_reference);
        f("log_clip_min", cfg.log_clip_min);
        f("log_clip_max", cfg.log_clip_max);
        f("block_angles", cfg.block_angles);
        f("rot_center", cfg.rot_center);
        f("backprojector", cfg.backprojector);
        if (c.contains("tomo_params") && !nb::object(c["tomo_params"]).is_none())
            for (auto [k, v] : nb::cast<nb::dict>(c["tomo_params"])) cfg.tomo_params.values[nb::cast<std::string>(k)] = to_text(v);
        f("bp_interp", cfg.bp_interp);
        f("bp_oversample", cfg.bp_oversample);
        if (c.contains("external_bp") && !nb::object(c["external_bp"]).is_none())
            cfg.external = std::make_shared<PyExternalBackprojector>(nb::cast<nb::dict>(c["external_bp"]));
        f("grid_size", cfg.grid_size);
        f("ghbp_legs", cfg.ghbp_legs);
        f("ghbp_reduction_steps", cfg.ghbp_reduction_steps);
        f("clip_outer_circle", cfg.clip_outer_circle);
        f("voxel_size_m", cfg.pixel_size_m);
        f("axis_source_meters", cfg.axis_source_meters);
        nb::gil_scoped_release nogil;
        impl_ = std::make_unique<SliceReconstructor>(std::move(scans), cfg);
    }
    nb::dict info() const {
        nb::dict d;
        auto [lo, hi] = impl_->doable();
        d["doable"] = nb::make_tuple(lo, hi);
        d["margin"] = impl_->margin();
        d["n_grid"] = impl_->n_grid();
        d["grid_size"] = impl_->grid_size();
        d["merged_width"] = impl_->merged_width();
        const auto [mx0, mx1] = impl_->mend_domain();
        d["mend_domain"] = nb::make_tuple(mx0, mx1);
        d["legacy_reference"] = impl_->config().legacy_reference;
        return d;
    }
    nb::object reconstruct(int h_lo, int h_hi, bool debug, bool timing, std::optional<std::array<int, 4>> roi,
                           std::optional<std::array<int, 2>> next) {
        const int N = impl_->grid_size();
        SliceRoi r{0, N, 0, N};
        if (roi) r = SliceRoi{(*roi)[0], (*roi)[1], (*roi)[2], (*roi)[3]};
        require(0 <= r.x0 && r.x0 < r.x1 && r.x1 <= N && 0 <= r.y0 && r.y0 < r.y1 && r.y1 <= N,
                "roi (x0, x1, y0, y1) must lie within the slice");
        require(h_hi > h_lo, "empty height range");
        HostOut out = new_array({static_cast<std::size_t>(h_hi - h_lo), static_cast<std::size_t>(r.y1 - r.y0),
                                 static_cast<std::size_t>(r.x1 - r.x0)});
        ChunkDebug dbg;
        dbg.keep = debug;
        dbg.timing = timing;
        {
            nb::gil_scoped_release nogil;
            impl_->reconstruct(h_lo, h_hi, data(out), &dbg, &r, next ? (*next)[0] : 0, next ? (*next)[1] : 0);
        }
        nb::dict sec;
        for (const auto& [k, v] : dbg.seconds) sec[k.c_str()] = v;
        for (const auto& [k, v] : dbg.stats) sec[("stat_" + k).c_str()] = v;
        if (!debug && !timing) return nb::cast(out);
        if (!debug) {
            nb::dict d;
            d["slices"] = out;
            d["seconds"] = sec;
            return d;
        }
        const std::size_t DY = h_hi - h_lo, NG = impl_->n_grid(), NX = impl_->merged_width();
        auto mk = [&](const std::vector<float>& v, std::vector<std::size_t> shape) {
            HostOut a = new_array(shape);
            std::memcpy(data(a), v.data(), v.size() * sizeof(float));
            return a;
        };
        nb::dict d;
        d["slices"] = out;
        d["sino"] = mk(dbg.sino, {NG, DY, NX});
        d["weights"] = mk(dbg.weights, {NG, DY, NX});
        d["wbal"] = mk(dbg.wbal, {NG, DY, NX});
        d["filtered"] = mk(dbg.filtered, {DY, NG, NX});
        if (timing) d["seconds"] = sec;
        return d;
    }
    static std::vector<FilterChain> chains_from(const nb::object& sweep) {
        std::vector<FilterChain> out;
        if (sweep.is_none()) return out;
        for (nb::handle h : nb::iter(sweep)) out.push_back(chain_from(h));
        return out;
    }
    int sweep_margin(const nb::object& sweep) const { return impl_->sweep_margin(chains_from(sweep)); }
    std::string describe(const nb::object& chain) const { return impl_->describe(chain_from(chain)); }
    nb::object reconstruct_band(int h_lo, int h_hi, BandSampler& band, bool timing, const nb::object& sweep,
                                std::optional<std::array<int, 2>> next) {
        ChunkDebug dbg;
        dbg.timing = timing;
        const auto sets = chains_from(sweep);
        {
            nb::gil_scoped_release nogil;
            impl_->reconstruct_band(h_lo, h_hi, band, &dbg, sets.empty() ? nullptr : &sets, next ? (*next)[0] : 0,
                                    next ? (*next)[1] : 0);
        }
        nb::dict sec;
        for (const auto& [k, v] : dbg.seconds) sec[k.c_str()] = v;
        for (const auto& [k, v] : dbg.stats) sec[("stat_" + k).c_str()] = v;
        return sec;
    }

  private:
    std::unique_ptr<SliceReconstructor> impl_;
};

}  // namespace

void bind_calib(nb::module_& m);  // bindings/calib_bindings.cpp

NB_MODULE(_core, m) {
    register_filter("python", [](const FilterParams& p, const FilterContext&) -> std::unique_ptr<ProjectionFilter> {
        return std::make_unique<PythonFilter>(p);
    });

    bind_calib(m);
    m.doc() = "helirec C++/CUDA kernels";
    nb::register_exception_translator([](const std::exception_ptr& p, void*) {
        try {
            std::rethrow_exception(p);
        } catch (const helirec::Error& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

    m.def("cuda_available", &cuda_available, "True if built with CUDA and a GPU is usable");
    m.def("release_pinned_memory", &release_pinned_memory,
          "free the page-locked host buffers kept for reuse by later reconstructors (CUDA; up to 24 GB)");
#ifdef HELIREC_WITH_CUDA
    m.attr("built_with_cuda") = true;
#else
    m.attr("built_with_cuda") = false;
#endif

    m.def("flatfield", &flatfield_py, "raw"_a, "dark"_a, "flat"_a, "threshold"_a = 1e-4f,
          "take_log"_a = true, "device"_a = "cpu",
          "-log(max((raw - dark) / (flat - dark), threshold)) per frame");
    m.def("backproject", &backproject_py, "sino"_a, "angles"_a, "ny"_a, "nx"_a, "rot_center"_a,
          "scale"_a = 1.f, "center"_a = -1.f, "x0"_a = 0, "y0"_a = 0, "clip_outer_circle"_a = false,
          "clip_radius"_a = -1.f, "device"_a = "cpu", "interp"_a = 0, "u_scale"_a = 1.f,
          "Exact pixel-driven parallel-beam backprojection; interp 0 linear, 1 cubic (Keys), 2 lanczos3, 3 bspline3, 4 bspline5 "
          "(B-spline: pass bspline_prefilter'd rows); rows "
          "sampled u_scale times per bin");
    m.def("upsample_rows",
          [](HostIn rows, int os, const std::string& device) {
              require(rows.ndim() == 2, "rows must be (n_rows, n)");
              const int nr = static_cast<int>(rows.shape(0)), n = static_cast<int>(rows.shape(1));
              HostOut out = new_array({static_cast<std::size_t>(nr), static_cast<std::size_t>(n) * os});
              nb::gil_scoped_release nogil;
              if (parse_device(device) == Device::CPU) {
                  cpu::upsample_rows(rows.data(), data(out), nr, n, os);
                  return out;
              }
#ifdef HELIREC_WITH_CUDA
              cuda::DeviceBuffer<float> di, dout(static_cast<std::size_t>(nr) * n * os);
              di.upload(rows.data(), rows.size());
              cuda::upsample_rows(di.data(), dout.data(), nr, n, os);
              dout.download(data(out));
#endif
              return out;
          },
          "rows"_a, "os"_a, "device"_a = "cpu", "band-limited (FFT) upsampling of rows by os");
    m.def("bspline_prefilter",
          [](HostIn rows, int degree, const std::string& device) {
              require(rows.ndim() == 2, "rows must be (n_rows, n)");
              require(degree == 3 || degree == 5, "degree must be 3 or 5");
              const int nr = static_cast<int>(rows.shape(0)), n = static_cast<int>(rows.shape(1));
              HostOut out = new_array({static_cast<std::size_t>(nr), static_cast<std::size_t>(n)});
              nb::gil_scoped_release nogil;
              if (parse_device(device) == Device::CPU) {
                  cpu::bspline_prefilter(rows.data(), data(out), nr, n, degree);
                  return out;
              }
#ifdef HELIREC_WITH_CUDA
              cuda::DeviceBuffer<float> d;
              d.upload(rows.data(), rows.size());
              cuda::bspline_prefilter(d.data(), d.data(), nr, n, degree);
              d.download(data(out));
#endif
              return out;
          },
          "rows"_a, "degree"_a, "device"_a = "cpu", "rows -> B-spline coefficients (degree 3 or 5, zero-extended rows)");

    nb::class_<PyFourierFilter>(m, "FourierFilter",
        "Padded real FFT filter: pad -> rfft (2D, or 1D per row) -> * filter -> irfft -> crop")
        .def(nb::init<int, int, int, int, int, int, int, const std::string&, HostIn, const std::string&>(),
             "ndim"_a, "h"_a, "w"_a, "ph"_a, "pw"_a, "pad_top"_a, "pad_left"_a, "mode"_a, "filter"_a,
             "device"_a = "cpu")
        .def("apply", &PyFourierFilter::apply, "stack"_a)
        .def_prop_ro("info", &PyFourierFilter::info);

    m.def("tomo_algorithms", &tomo_algorithms, "whole-slice methods (backprojectors sirt, mlem, gridrec)");
    m.def("chord_project",
          [](HostIn img, HostIn angles, int n_det, float rot, const std::string& device, const std::string& projector) {
              require(img.ndim() == 2 && img.shape(0) == img.shape(1), "img must be (N, N)");
              const int pj = projector_id(projector);
              const int N = static_cast<int>(img.shape(0)), na = static_cast<int>(angles.size());
              HostOut out = new_array({static_cast<std::size_t>(na), static_cast<std::size_t>(n_det)});
              nb::gil_scoped_release nogil;
              if (parse_device(device) == Device::CPU) {
                  cpu::chord_project(img.data(), data(out), angles.data(), na, n_det, N, rot, pj);
                  return out;
              }
#ifdef HELIREC_WITH_CUDA
              cuda::DeviceBuffer<float> di, ds(static_cast<std::size_t>(na) * n_det);
              di.upload(img.data(), img.size());
              cuda::chord_project(di.data(), ds.data(), angles.data(), na, n_det, N, rot, pj);
              ds.download(data(out));
#endif
              return out;
          },
          "img"_a, "angles"_a, "n_det"_a, "rot"_a, "device"_a = "cpu", "projector"_a = "chord",
          "forward projection, rays u = x cos a + y sin a + rot (pixel centres x = ix - (N-1)/2); projector 'chord' "
          "(exact chord lengths at the cell centre) or 'sf' (separable footprint: integrated over the cell)");
    m.def("chord_backproject",
          [](HostIn sino, HostIn angles, int N, float rot, const std::string& device, const std::string& projector) {
              require(sino.ndim() == 2 && sino.shape(0) == angles.size(), "sino must be (n_angles, n_det)");
              const int pj = projector_id(projector);
              const int na = static_cast<int>(sino.shape(0)), nd = static_cast<int>(sino.shape(1));
              HostOut out = new_array({static_cast<std::size_t>(N), static_cast<std::size_t>(N)});
              nb::gil_scoped_release nogil;
              if (parse_device(device) == Device::CPU) {
                  cpu::chord_backproject(sino.data(), data(out), angles.data(), na, nd, N, rot, pj);
                  return out;
              }
#ifdef HELIREC_WITH_CUDA
              cuda::DeviceBuffer<float> ds, di(static_cast<std::size_t>(N) * N);
              ds.upload(sino.data(), sino.size());
              cuda::chord_backproject(ds.data(), di.data(), angles.data(), na, nd, N, rot, pj);
              di.download(data(out));
#endif
              return out;
          },
          "sino"_a, "angles"_a, "N"_a, "rot"_a, "device"_a = "cpu", "projector"_a = "chord", "adjoint of chord_project");
    m.def("tomo_reconstruct",
          [](HostIn rows, nb::object weights_obj, HostIn angles, int N, float rot, const std::string& algorithm,
             nb::dict params, const std::string& device, float R) {
              require(rows.ndim() == 2 && rows.shape(0) == angles.size(), "rows must be (n_angles, n_det)");
              TomoGeom t;
              t.n_angles = static_cast<int>(rows.shape(0)); t.n_det = static_cast<int>(rows.shape(1));
              t.N = N; t.rot = rot; t.R = R > 0 ? R : 1e30f;
              const std::size_t W = N, H = N;
              std::vector<float> w(rows.size(), 1.f);
              if (!weights_obj.is_none()) {
                  auto wa = nb::cast<HostIn>(weights_obj);
                  require(wa.size() == rows.size(), "weights must match rows");
                  std::memcpy(w.data(), wa.data(), w.size() * sizeof(float));
              }
              FilterParams fp;
              for (auto [k, v] : params) fp.values[nb::cast<std::string>(k)] = to_text(v);
              const TomoArgs a = tomo_args(algorithm, fp);
              HostOut out = new_array({H, W});
              nb::gil_scoped_release nogil;
              if (parse_device(device) == Device::CPU) {
                  cpu::tomo_reconstruct(a, t, rows.data(), w.data(), angles.data(), data(out));
                  return out;
              }
#ifdef HELIREC_WITH_CUDA
              cuda::DeviceBuffer<float> dr, dw, dx(W * H);
              dr.upload(rows.data(), rows.size());
              dw.upload(w.data(), w.size());
              cuda::tomo_reconstruct(a, t, dr.data(), dw.data(), angles.data(), dx.data());
              dx.download(data(out));
#endif
              return out;
          },
          "rows"_a, "weights"_a.none(), "angles"_a, "N"_a, "rot"_a, "algorithm"_a, "params"_a = nb::dict(), "device"_a = "cpu",
          "R"_a = 0.f,
          "one whole slice (per-pixel units); weights: measured support for sirt/mlem, balanced weights for gridrec, "
          "or None for ones; R: disk radius");
    m.def("mend_null_weights",
          [](HostIn R, HostIn RW, float threshold, int x0, int x1, const std::string& device) {
              require(R.ndim() == 2 && RW.ndim() == 2 && R.shape(0) == RW.shape(0) && R.shape(1) == RW.shape(1),
                      "R, RW must be (rows, NX)");
              HostOut out = copy_of(R);
              const int H = static_cast<int>(R.shape(0)), NX = static_cast<int>(R.shape(1));
              nb::gil_scoped_release nogil;
              if (parse_device(device) == Device::CPU) {
                  cpu::mend_null_weights(data(out), RW.data(), 1, H, NX, threshold, x0, x1);
                  return out;
              }
#ifdef HELIREC_WITH_CUDA
              cuda::DeviceBuffer<float> dr, dw;
              dr.upload(data(out), R.size());
              dw.upload(RW.data(), RW.size());
              cuda::mend_null_weights(dr.data(), dw.data(), 1, H, NX, threshold, x0, x1);
              HELIREC_CUDA_CHECK(cudaDeviceSynchronize());
              dr.download(data(out));
#endif
              return out;
          },
          "R"_a, "RW"_a, "threshold"_a, "x0"_a, "x1"_a, "device"_a = "cpu",
          "nabuxx mending on the usable columns [x0, x1) of each row: columns whose weight is below threshold x mean "
          "weight take the nearest valid column's value");
    m.def("filter_types", &filter_types, "registered projection filter types");
    m.def("filter_margin",
          [](nb::object chain, int det_rows, int det_cols, double energy_kev, double distance_m, double pixel_size_m,
             bool legacy) {
              const FilterPipeline P(chain_from(chain), FilterContext{energy_kev, distance_m, pixel_size_m, det_rows, det_cols, legacy});
              return P.margin_rows();
          },
          "chain"_a, "det_rows"_a, "det_cols"_a, "energy_kev"_a = 0.0, "distance_m"_a = 0.0, "pixel_size_m"_a = 0.0,
          "legacy_reference"_a = false,
          "margin rows a filter chain needs (list of {'type': ..., params})");
    m.def("filter_describe",
          [](nb::object chain, int det_rows, int det_cols, double energy_kev, double distance_m, double pixel_size_m,
             bool legacy) {
              return FilterPipeline(chain_from(chain), FilterContext{energy_kev, distance_m, pixel_size_m, det_rows, det_cols, legacy})
                  .describe();
          },
          "chain"_a, "det_rows"_a, "det_cols"_a, "energy_kev"_a = 0.0, "distance_m"_a = 0.0, "pixel_size_m"_a = 0.0,
          "legacy_reference"_a = false);
    m.def("apply_filters",
          [](HostIn stack, nb::object chain, int margin, const std::string& stage, int det_rows, int det_cols,
             double energy_kev, double distance_m, double pixel_size_m, const std::string& device, bool legacy) {
              require(stack.ndim() == 3, "stack must be (n_frames, H, W)");
              const int n = static_cast<int>(stack.shape(0)), H = static_cast<int>(stack.shape(1)), W = static_cast<int>(stack.shape(2));
              const FilterPipeline P(chain_from(chain), FilterContext{energy_kev, distance_m, pixel_size_m, det_rows, det_cols, legacy});
              const FilterStage st = stage == "log" ? FilterStage::Log : FilterStage::Intensity;
              const Device dev = parse_device(device);
              HostOut out = copy_of(stack);
              nb::gil_scoped_release nogil;
              const auto& steps = P.steps();
              float* host = data(out);
              float* buf = host;
#ifdef HELIREC_WITH_CUDA
              cuda::DeviceBuffer<float> d;
              if (dev == Device::CUDA) { d.upload(host, stack.size()); buf = d.data(); }
#endif
              for (std::size_t s = 0; s < steps.size(); ++s) {
                  if (steps[s].stage != st) continue;
                  if (steps[s].fourier) {
                      const auto& spec = P.step_spec(s, H, W, steps.size() == 1 ? margin : -1);
#ifdef HELIREC_WITH_CUDA
                      auto f = dev == Device::CUDA ? cuda::make_fourier_filter(spec) : cpu::make_fourier_filter(spec);
#else
                      auto f = cpu::make_fourier_filter(spec);
#endif
                      f->apply(buf, buf, n);
                  } else {
                      for (auto* fl : steps[s].members) fl->apply(buf, n, H, W, dev);
                  }
              }
#ifdef HELIREC_WITH_CUDA
              if (dev == Device::CUDA) d.download(host);
#endif
              return out;
          },
          "stack"_a, "chain"_a, "margin"_a, "stage"_a = "intensity", "det_rows"_a = 0, "det_cols"_a = 0,
          "energy_kev"_a = 0.0, "distance_m"_a = 0.0, "pixel_size_m"_a = 0.0, "device"_a = "cpu", "legacy_reference"_a = false,
          "Apply a filter chain's steps of `stage` to a stack (n_frames, H, W) as the pipeline does (single-step "
          "chains use `margin`); keep rows [margin, H - margin) of the result");
    m.def("sld_correct",
          [](HostIn frames, double fraction, double length, int bin) {
              require(frames.ndim() == 3, "frames must be (n, rows, cols)");
              SldParams p;
              p.fraction = fraction;
              p.length = length;
              p.bin = bin;
              HostOut out = copy_of(frames);
              nb::gil_scoped_release nogil;
              ScintillatorGlare g(static_cast<int>(frames.shape(1)), static_cast<int>(frames.shape(2)), p);
              g.correct(data(out), static_cast<int>(frames.shape(0)));
              return out;
          },
          "frames"_a, "fraction"_a, "length"_a = 100.0, "bin"_a = 8,
          "Scintillator light diffusion correction of detector frames (n, rows, cols): the frames minus their glare, "
          "i.e. the deconvolved frames / (1 + fraction) (the factor cancels between projections and flats)");
    m.def("roundfft", &roundfft, "n"_a);

    m.def("delta3d_available", &delta3d_available, "True if built with the vc-delta3d codec");
    m.def(
        "delta3d_compress",
        [](nb::ndarray<const uint8_t, nb::ndim<3>, nb::c_contig, nb::device::cpu> a, const std::string& preset) {
            std::vector<uint8_t> enc;
            {
                nb::gil_scoped_release nogil;
                enc = delta3d_compress_u8(a.data(), static_cast<int>(a.shape(0)), static_cast<int>(a.shape(1)),
                                          static_cast<int>(a.shape(2)), preset.c_str());
            }
            return nb::bytes(reinterpret_cast<const char*>(enc.data()), enc.size());
        },
        "chunk"_a, "preset"_a = "fast", "lossless vc-delta3d payload of a (z, y, x) uint8 chunk");
    m.def(
        "delta3d_decompress",
        [](nb::bytes payload, std::vector<std::size_t> shape) {
            auto out = new_array_t<uint8_t>(shape);
            const auto* src = reinterpret_cast<const uint8_t*>(payload.c_str());
            const std::size_t n = payload.size(), size = out.size();
            {
                nb::gil_scoped_release nogil;
                delta3d_decompress(src, n, static_cast<uint8_t*>(out.data()), size);
            }
            return out;
        },
        "payload"_a, "shape"_a, "decode a vc-delta3d payload into a uint8 array of `shape`");

    m.def("span_info",
          [](DIn angles_deg, DIn z_translation_m, double pixel_size_m, int det_rows, float z_readjustment, double tol) {
              ScanTrajectory t;
              t.angles_deg = vec<double>(angles_deg);
              t.z_translation_m = vec<double>(z_translation_m);
              t.pixel_size_m = pixel_size_m;
              t.det_rows = det_rows;
              t.z_readjustment = z_readjustment;
              SpanInfo s = make_span_info(t, tol);
              auto iarr = [](const std::vector<int32_t>& v) {
                  auto a = new_array_t<int32_t>({v.size()});
                  std::copy(v.begin(), v.end(), static_cast<int32_t*>(a.data()));
                  return a;
              };
              auto farr = [](const std::vector<float>& v) {
                  HostOut a = new_array({v.size()});
                  std::copy(v.begin(), v.end(), data(a));
                  return a;
              };
              nb::dict d;
              d["integer_shift_v"] = iarr(s.integer_shift_v);
              d["fract_complement"] = farr(s.fract_complement);
              d["z_pix_per_proj"] = farr(s.z_pix_per_proj);
              d["h_first"] = s.h_first;
              d["sunshine_start"] = iarr(s.sunshine_start);
              d["sunshine_end"] = iarr(s.sunshine_end);
              d["doable"] = nb::make_tuple(s.doable_lo, s.doable_hi);
              return d;
          },
          "angles_deg"_a, "z_translation_m"_a, "pixel_size_m"_a, "det_rows"_a, "z_readjustment"_a = 0.f,
          "angular_tolerance_steps"_a = 5.0);
    nb::class_<ScanData>(m, "ScanData", "one scan (raw access + calibration), see helirec.scan_data()");
    m.def("make_scan_data", &make_scan_data, "angles_deg"_a, "z_translation_m"_a, "pixel_size_m"_a, "det_rows"_a,
          "det_cols"_a, "shift_x"_a, "x_readjustment"_a, "z_readjustment"_a, "control_A"_a, "flat_angles_deg"_a,
          "dark"_a, "flats"_a, "weight"_a, "reader"_a, "file_index"_a, "frame_index"_a);
    nb::class_<PySliceReconstructor>(m, "SliceReconstructor")
        .def(nb::init<std::vector<ScanData>, const nb::dict&>(), "scans"_a, "config"_a)
        .def_prop_ro("info", &PySliceReconstructor::info)
        .def("reconstruct", &PySliceReconstructor::reconstruct, "h_lo"_a, "h_hi"_a, "debug"_a = false,
             "timing"_a = false, "roi"_a = nb::none(), "next"_a = nb::none(),
             "slices for view heights [h_lo, h_hi) -> (n, N, N), or (n, y1 - y0, x1 - x0) with roi=(x0, x1, y0, y1); "
             "debug=True returns a dict with intermediates; timing=True returns a dict with 'slices' and per-stage "
             "'seconds' (GPU synchronised per stage); next=(h_lo, h_hi): the chunk that will follow, read in the "
             "background (CUDA)")
        .def("reconstruct_band", &PySliceReconstructor::reconstruct_band, "h_lo"_a, "h_hi"_a, "band"_a,
             "timing"_a = false, "sweep"_a = nb::none(), "next"_a = nb::none(),
             "reconstruct the chunk [h_lo, h_hi) only at the band pixels `band` needs (exact backprojector) and "
             "sample its points; returns per-stage seconds (with timing=True). Call with contiguous chunks. "
             "sweep: list of dicts of phase parameters (delta_beta, unsharp_*), one per value set of `band` (CUDA): "
             "everything before the phase filter is computed once per chunk. next=(lo, hi): the following chunk, whose "
             "raw windows are then read in the background (CUDA).")
        .def("describe", &PySliceReconstructor::describe, "chain"_a, "compiled steps of a filter chain")
        .def("sweep_margin", &PySliceReconstructor::sweep_margin, "sweep"_a,
             "phase margin (rows) a sweep uses: the largest any of its sets needs");

    nb::class_<BandSampler>(m, "BandSampler",
                            "Sample points for band reconstruction: zlo (int32, sorted), fz, y, x (float32), in view "
                            "heights / slice pixels; values() are the trilinear samples of the reconstructed volume")
        .def("__init__", [](BandSampler* self, IntIn zlo, HostIn fz, HostIn y, HostIn x, int grid_size,
                            const std::string& device, int n_sets) {
                 const std::size_t n = zlo.shape(0);
                 require(fz.size() == n && y.size() == n && x.size() == n, "zlo, fz, y, x must have one length");
                 new (self) BandSampler(parse_device(device), zlo.data(), fz.data(), y.data(), x.data(), n, grid_size,
                                        n_sets);
             }, "zlo"_a, "fz"_a, "y"_a, "x"_a, "grid_size"_a, "device"_a = "cuda", "n_sets"_a = 1)
        .def_prop_ro("size", &BandSampler::size)
        .def_prop_ro("n_sets", &BandSampler::n_sets)
        .def_prop_ro("h_first", &BandSampler::h_first)
        .def_prop_ro("h_last", &BandSampler::h_last)
        .def_prop_ro("box", [](const BandSampler& b) {
            const auto& x = b.box();
            return nb::make_tuple(x.x0, x.x0 + x.w, x.y0, x.y0 + x.h);
        }, "(x0, x1, y0, y1) of the needed pixels")
        .def_prop_ro("pixels_backprojected", &BandSampler::pixels_backprojected)
        .def_prop_ro("points_sampled", &BandSampler::points_sampled)
        .def("values", [](const BandSampler& b, int set) {
            HostOut out = new_array({b.size()});
            b.values(data(out), set);
            return out;
        }, "set"_a = 0)
        .def("reset", &BandSampler::reset)
        .def("feed", [](BandSampler& b, int h, HostIn filtered, HostIn angles, float rot_center, float scale,
                        float center, bool clip_outer_circle, float clip_radius, int set) {
                 require(filtered.ndim() == 2 && filtered.shape(0) == angles.size(), "filtered must be (n_angles, n_det)");
                 BackprojectParams p;
                 p.rot_center = rot_center; p.scale = scale; p.center = center;
                 p.clip_outer_circle = clip_outer_circle; p.clip_radius = clip_radius;
                 const int na = static_cast<int>(filtered.shape(0)), nd = static_cast<int>(filtered.shape(1));
                 nb::gil_scoped_release nogil;
                 if (b.device() == Device::CPU) { b.slice(h, filtered.data(), na, nd, angles.data(), p, set); return; }
#ifdef HELIREC_WITH_CUDA
                 cuda::DeviceBuffer<float> d;
                 d.upload(filtered.data(), filtered.size());
                 b.slice(h, d.data(), na, nd, angles.data(), p, set);
                 HELIREC_CUDA_CHECK(cudaDeviceSynchronize());
#endif
             }, "h"_a, "filtered"_a, "angles"_a, "rot_center"_a, "scale"_a = 1.f, "center"_a = -1.f,
             "clip_outer_circle"_a = false, "clip_radius"_a = -1.f, "set"_a = 0,
             "feed one slice's filtered sinogram (host array) -- for tests; the pipeline feeds device data");

    nb::class_<RawReader>(m, "RawReader", "pread-based reader of one-chunk-per-frame uint16 HDF5 frame stacks")
        .def(nb::init<std::vector<std::string>, std::string, int, int>(), "files"_a, "dataset"_a, "rows"_a, "cols"_a)
        .def_prop_ro("rows", &RawReader::rows)
        .def_prop_ro("cols", &RawReader::cols)
        .def("read_rows",
             [](RawReader& r, IntIn file_index, IntIn frame_index, IntIn row0, int nrows, int n_threads) {
                 const std::size_t n = file_index.shape(0);
                 require(frame_index.shape(0) == n && row0.shape(0) == n, "index arrays must have equal length");
                 auto out = new_array_t<uint16_t>({n, static_cast<std::size_t>(nrows), static_cast<std::size_t>(r.cols())});
                 nb::gil_scoped_release nogil;
                 r.read_rows(file_index.data(), frame_index.data(), row0.data(), n, nrows,
                             static_cast<uint16_t*>(out.data()), n_threads);
                 return out;
             },
             "file_index"_a, "frame_index"_a, "row0"_a, "nrows"_a, "n_threads"_a = 0,
             "rows [row0, row0+nrows) of each frame -> (n, nrows, cols) uint16; outside rows are 0")
        .def("read_frames",
             [](RawReader& r, IntIn file_index, IntIn frame_index, int n_threads) {
                 const std::size_t n = file_index.shape(0);
                 require(frame_index.shape(0) == n, "index arrays must have equal length");
                 auto out = new_array_t<uint16_t>({n, static_cast<std::size_t>(r.rows()), static_cast<std::size_t>(r.cols())});
                 nb::gil_scoped_release nogil;
                 r.read_frames(file_index.data(), frame_index.data(), n, static_cast<uint16_t*>(out.data()), n_threads);
                 return out;
             },
             "file_index"_a, "frame_index"_a, "n_threads"_a = 0);
    m.def("minus_log", &minus_log_py, "x"_a, "clip_min"_a = 1e-6f, "clip_max"_a = 10.f, "device"_a = "cpu");
    m.def("rebalance", &rebalance_py, "R"_a, "W"_a, "rot_center"_a, "legacy_reference"_a = false, "device"_a = "cpu",
          "Hole filling + half-acquisition balanced weights (opposite view, periodic; legacy_reference: the nabuxx "
          "endpoint handling); returns (R, W_bal) copies");
    m.def("ramp_filter", &ramp_filter_py, "sino"_a, "rot_center"_a, "pad_factor"_a = 1, "normalize"_a = true,
          "legacy_reference"_a = false, "device"_a = "cpu",
          "Ram-Lak filter with opposite-ray padding, (n_proj, n_bins) -> same shape. normalize: x 2 pi / n_proj "
          "(attenuation for pair-balanced full-turn data; legacy_reference: pi / n_proj and integer padding reflection)");
    m.def("ramp_scale", &ramp_scale, "n_proj"_a, "legacy_reference"_a = false);
    m.def("ramp_geometry", [](int n_bins, int n_proj, float rot_center, int pad_factor, bool legacy) {
              const RampGeometry g = ramp_geometry(n_bins, n_proj, rot_center, pad_factor, legacy);
              nb::dict d;
              d["npad"] = g.npad; d["pad_l"] = g.pad_l; d["two_rots"] = g.two_rots; d["my_rot"] = g.my_rot;
              d["R"] = g.R; d["switch_pos"] = g.switch_pos; d["legacy"] = g.legacy;
              return d;
          }, "n_bins"_a, "n_proj"_a, "rot_center"_a, "pad_factor"_a = 1, "legacy_reference"_a = false);

    m.def("ghbp", &ghbp_py, "sino"_a, "angles"_a, "grid_size"_a, "rot_center"_a, "scale_factor"_a = 1.f,
          "legs"_a = 4, "reduction_steps"_a = 2, "clip_outer_circle"_a = false,
          "axis_source_meters"_a = 1e9, "voxel_size_microns"_a = 1.f, "whf_x"_a = 1.f, "whf_y"_a = 1.f,
          "fac"_a = 1.f, "device"_a = "cpu",
          "Hierarchical backprojection (port of nabuxx GHBP). angles: nabu angles A (radians).");
    m.def("ghbp_plan_info", &ghbp_plan_info_py, "angles"_a, "n_bins"_a, "grid_size"_a, "legs"_a = 4,
          "reduction_steps"_a = 2, "Hierarchy levels GHBP would use");
}
