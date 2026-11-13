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
#include <nanobind/stl/tuple.h>
#include "helirec/helical.hpp"
#include "helirec/kernels.hpp"
#include "helirec/progress.hpp"
#include "helirec/reference.hpp"
#include "helirec/zarr.hpp"
#include "helirec/reconstruction.hpp"
#include "../src/io/image_files.hpp"
#include "helirec/calibration/scan.hpp"
#include "helirec/calibration/intensity.hpp"
#include "ndarray_utils.hpp"

#ifdef HELIREC_WITH_CUDA
#include "../src/cuda/cuda_utils.cuh"
#endif

#include <chrono>
#include <cmath>
#include <cstring>

namespace nb = nanobind;
using namespace nb::literals;
using namespace helirec;
using namespace helirec::py;

namespace {

// (..., ny, nx) float32 -> uint8 (a 1-D array is one row); disk = (n, x0, y0): the frames are the ROI at (x0, y0)
// of (n, n) slices
using MaskIn = nb::ndarray<const std::uint8_t, nb::c_contig, nb::device::cpu>;

// A mask for SliceReconstructor.reconstruct(mask=...) with the object owning its memory: a NumPy array, or the
// ReferenceMask whose buffer it is.
struct PyOutputMask {
    OutputMask m;
    nb::object keep;
};

nb::ndarray<nb::numpy, std::uint8_t> to_uint8_py(HostIn img, double lo, double hi,
                                                 std::optional<std::array<int, 3>> disk, const std::string& device,
                                                 nb::object mask) {
    require(img.ndim() >= (disk ? 2 : 1), "img must be (..., ny, nx)");
    require(hi > lo, "the window needs hi > lo");
    const std::size_t ny = img.ndim() >= 2 ? img.shape(img.ndim() - 2) : 1, nx = img.shape(img.ndim() - 1), plane = ny * nx;
    const std::size_t frames = plane ? img.size() / plane : 0;
    Uint8Window w;
    w.lo = lo;
    w.hi = hi;
    if (disk) { w.disk_n = (*disk)[0]; w.x0 = (*disk)[1]; w.y0 = (*disk)[2]; }
    // the mask: an array of img's shape, or an OutputMask over the frames (possibly coarser)
    OutputMask k;
    std::optional<MaskIn> mask_array;
    if (nb::isinstance<PyOutputMask>(mask)) {
        k = nb::cast<PyOutputMask&>(mask).m;
    } else if (!mask.is_none()) {
        mask_array = nb::cast<MaskIn>(mask);
        require(mask_array->size() == img.size(), "a mask array must have the shape of img (an OutputMask may be coarser)");
        k.data = mask_array->data();
        k.nz = static_cast<int>(frames); k.ny = static_cast<int>(ny); k.nx = static_cast<int>(nx);
    }
    const std::size_t mask_plane = static_cast<std::size_t>(k.ny) * k.nx;
    auto mask_slice = [&](std::size_t f) { return std::min(std::max(static_cast<int>(f) + k.oz, 0) / k.fz, k.nz - 1); };
    w.mask_ny = k.ny; w.mask_nx = k.nx;
    w.mask_fy = k.fy; w.mask_fx = k.fx;
    w.mask_oy = k.oy; w.mask_ox = k.ox;
    auto out = new_array_t<std::uint8_t>(shape_of(img));
    auto* o = static_cast<std::uint8_t*>(out.data());
    const Device dev = parse_device(device);
    nb::gil_scoped_release nogil;
    if (dev == Device::CPU) {
        for (std::size_t f = 0; f < frames; ++f) {
            w.mask = k.data ? k.data + mask_slice(f) * mask_plane : nullptr;
            cpu::to_uint8(img.data() + f * plane, nx, o + f * plane, static_cast<int>(ny), static_cast<int>(nx), w);
        }
        return out;
    }
#ifdef HELIREC_WITH_CUDA
    cuda::DeviceBuffer<float> d_in(plane);
    cuda::DeviceBuffer<std::uint8_t> d_out(plane), d_mask(k.data ? mask_plane : 0);
    for (std::size_t f = 0; f < frames; ++f) {
        HELIREC_CUDA_CHECK(cudaMemcpy(d_in.data(), img.data() + f * plane, plane * sizeof(float), cudaMemcpyHostToDevice));
        if (k.data) {
            HELIREC_CUDA_CHECK(cudaMemcpy(d_mask.data(), k.data + mask_slice(f) * mask_plane, mask_plane, cudaMemcpyHostToDevice));
            w.mask = d_mask.data();
        }
        cuda::to_uint8(d_in.data(), nx, d_out.data(), static_cast<int>(ny), static_cast<int>(nx), w);
        HELIREC_CUDA_CHECK(cudaMemcpy(o + f * plane, d_out.data(), plane, cudaMemcpyDeviceToHost));
    }
#endif
    return out;
}

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
                       const std::string& device, int interp, float u_scale, std::optional<std::vector<std::int32_t>> tiles) {
    require(sino.ndim() == 2, "sino must be (n_angles, n_det)");
    const int n_angles = static_cast<int>(sino.shape(0)), n_det = static_cast<int>(sino.shape(1));
    require(angles.size() == static_cast<std::size_t>(n_angles), "angles length != sino rows");
    HostOut out = new_array({static_cast<std::size_t>(ny), static_cast<std::size_t>(nx)});
    BackprojectParams p;
    p.rot_center = rot_center; p.scale = scale; p.center = center; p.x0 = x0; p.y0 = y0;
    p.clip_outer_circle = clip_outer_circle; p.clip_radius = clip_radius;
    p.interp = interp; p.u_scale = u_scale;
    if (tiles) {
        const int T = kBackprojectTile, n_tiles = ((nx + T - 1) / T) * ((ny + T - 1) / T);
        std::vector<char> seen(static_cast<std::size_t>(std::max(n_tiles, 0)), 0);
        for (std::int32_t t : *tiles) {
            require(t >= 0 && t < n_tiles && !seen[t], "tiles: indices of the slice's tiles, each once");
            seen[t] = 1;
        }
        static const std::int32_t none = 0;   // (an empty list is still a list: nothing is computed)
        p.tiles = tiles->empty() ? &none : tiles->data();
        p.n_tiles = static_cast<int>(tiles->size());
    }
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

HostOut angular_fir_py(HostIn sino, const std::vector<float>& taps, std::optional<HostIn> support, const std::string& device) {
    require(sino.ndim() == 2, "sino must be (n_proj, n_cols)");
    require(!taps.empty(), "taps: g[0..radius]");
    require(!support || (support->ndim() == 2 && support->size() == sino.size()), "support must have sino's shape");
    const int np = static_cast<int>(sino.shape(0)), nc = static_cast<int>(sino.shape(1)), radius = static_cast<int>(taps.size()) - 1;
    HostOut out = copy_of(sino);
    const float* in = static_cast<const float*>(sino.data());
    const float* sup = support ? static_cast<const float*>(support->data()) : nullptr;
    const Device dev = parse_device(device);
    nb::gil_scoped_release nogil;
    if (dev == Device::CPU) {
        cpu::angular_fir(in, data(out), np, nc, taps.data(), radius, sup);
        return out;
    }
#ifdef HELIREC_WITH_CUDA
    cuda::DeviceBuffer<float> d_in, d_out(sino.size()), d_sup;
    d_in.upload(in, sino.size());
    if (sup) d_sup.upload(sup, sino.size());
    cuda::angular_fir(d_in.data(), d_out.data(), np, nc, taps.data(), radius, sup ? d_sup.data() : nullptr);
    d_out.download(data(out));
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

std::pair<HostOut, HostOut> rebalance_turns_py(HostIn R, HostIn W, int turn, float rot_center, const std::string& device) {
    require(R.ndim() == 3 && W.ndim() == 3 && R.size() == W.size(), "R, W must be (n_proj, n_rows, n_cols)");
    HostOut r = copy_of(R), w = copy_of(W);
    const int np = static_cast<int>(R.shape(0)), nr = static_cast<int>(R.shape(1)), nc = static_cast<int>(R.shape(2));
    if (parse_device(device) == Device::CPU) {
        nb::gil_scoped_release nogil;
        cpu::rebalance_turns(data(r), data(w), np, nr, nc, turn, rot_center);
        return {r, w};
    }
#ifdef HELIREC_WITH_CUDA
    {
        nb::gil_scoped_release nogil;
        cuda::DeviceBuffer<float> dr, dw;
        dr.upload(data(r), R.size());
        dw.upload(data(w), W.size());
        cuda::rebalance_turns(dr.data(), dw.data(), np, nr, nc, turn, rot_center);
        dr.download(data(r));
        dw.download(data(w));
    }
#endif
    return {r, w};
}

HostOut ramp_filter_py(HostIn sino, float rot_center, int pad_factor, bool normalize, bool legacy, const std::string& device,
                       const std::string& filter_name, const std::string& filter_par, const std::vector<int32_t>& opposite, int turn) {
    require(sino.ndim() == 2, "sino must be (n_proj, n_bins)");
    const int n_proj = static_cast<int>(sino.shape(0)), n_bins = static_cast<int>(sino.shape(1));
    // opposite: the row holding the opposite view of each row, for rows that are not one full turn (`turn` rows are)
    require(opposite.empty() || static_cast<int>(opposite.size()) == n_proj, "opposite: one row index per row");
    for (int32_t o : opposite) require(o >= 0 && o < n_proj, "opposite: a row index out of range");
    require(opposite.empty() ? turn == 0 : turn > 0, "opposite and turn are given together");
    const int32_t* opp = opposite.empty() ? nullptr : opposite.data();
    const Device dev = parse_device(device);
    HostOut out = new_array(shape_of(sino));
    nb::gil_scoped_release nogil;
    RampGeometry g = ramp_geometry(n_bins, opp ? turn : n_proj, rot_center, pad_factor, legacy);
    g.n_proj = n_proj;
    const FourierFilterSpec spec = ramp_filter_spec(g, normalize ? ramp_scale(opp ? turn : n_proj, legacy) : 1.f, filter_name,
                                                    ramp_filter_par(filter_par));
    if (dev == Device::CPU) {
        std::vector<float> padded(static_cast<std::size_t>(n_proj) * g.npad);
        cpu::ramp_pad(sino.data(), padded.data(), g, opp);
        cpu::make_fourier_filter(spec)->apply(padded.data(), padded.data(), n_proj);
        cpu::ramp_crop(padded.data(), data(out), g);
        return out;
    }
#ifdef HELIREC_WITH_CUDA
    cuda::DeviceBuffer<float> d_sino, d_pad(static_cast<std::size_t>(n_proj) * g.npad), d_out(sino.size());
    d_sino.upload(sino.data(), sino.size());
    cuda::DeviceBuffer<int32_t> d_opp;
    if (opp) d_opp.upload(opp, opposite.size());
    cuda::ramp_pad(d_sino.data(), d_pad.data(), g, nullptr, opp ? d_opp.data() : nullptr);
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
                        IntIn frame_index, double z_translation_scale, std::vector<float> row_offset,
                        std::vector<float> row_slope, float slope_col, std::vector<float> col_offset,
                        std::vector<float> col_slope, float slope_row, std::vector<int32_t> map_i, std::vector<int32_t> map_j,
                        std::vector<double> map_eu, std::vector<double> map_ev, double map_c0, double map_cs, double map_v0,
                        double map_vs, std::vector<float> col_scale, float scale_col, std::vector<float> row_scale,
                        float scale_row) {
    ScanData d;
    d.traj.map_i = std::move(map_i);
    d.traj.map_j = std::move(map_j);
    d.traj.map_eu = std::move(map_eu);
    d.traj.map_ev = std::move(map_ev);
    d.traj.map_c0 = map_c0;
    d.traj.map_cs = map_cs;
    d.traj.map_v0 = map_v0;
    d.traj.map_vs = map_vs;
    d.traj.col_scale = std::move(col_scale);
    d.traj.scale_col = scale_col;
    d.traj.row_scale = std::move(row_scale);
    d.traj.scale_row = scale_row;
    d.traj.z_translation_scale = z_translation_scale;
    d.traj.row_offset = std::move(row_offset);
    d.traj.row_slope = std::move(row_slope);
    d.traj.slope_col = slope_col;
    d.traj.col_offset = std::move(col_offset);
    d.traj.col_slope = std::move(col_slope);
    d.traj.slope_row = slope_row;
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

// a detector map handed over from Python, planned for one detector (detector_map.hpp); `view` points into it
struct PyDetectorMap {
    std::vector<int32_t> i, j;
    std::vector<double> eu, ev;
    std::vector<float> ev_f;
    DetectorMapView view;
    DetectorMapPlan plan;
    PyDetectorMap(std::vector<int32_t> i_, std::vector<int32_t> j_, std::vector<double> eu_, std::vector<double> ev_, double c0,
                  double cs, double v0, double vs, int det_rows, int det_cols, double row_slope, double col_scale, double row_scale)
        : i(std::move(i_)), j(std::move(j_)), eu(std::move(eu_)), ev(std::move(ev_)) {
        require(j.size() == i.size() && eu.size() == i.size() && ev.size() == i.size(), "map_i, map_j, map_eu, map_ev: one value per term");
        ev_f.assign(ev.begin(), ev.end());
        view.n = static_cast<int>(i.size());
        view.i = i.data(); view.j = j.data(); view.eu = eu.data(); view.ev = ev.data(); view.ev_f = ev_f.data();
        view.c0 = c0; view.cs = cs; view.v0 = v0; view.vs = vs;
        view.det_rows = det_rows; view.det_cols = det_cols;
        plan = detector_map_plan(view, row_slope, col_scale, row_scale);
        view.passes = plan.passes;
        view.settle = plan.settle;
    }
    PyDetectorMap(const PyDetectorMap&) = delete;
};
using DoubleIn = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;

template <class T>
double max_abs(const std::vector<T>& v) {
    double m = 0;
    for (T x : v) m = std::max(m, std::fabs(static_cast<double>(x)));
    return m;
}

HostOut resample_rows_py(HostIn rows, std::optional<DoubleIn> positions, float shift, const std::vector<float>& shifts,
                         const std::vector<double>& row_y, const std::vector<float>& row_slope, float slope_col,
                         const std::vector<float>& col_scale, float scale_col, const std::vector<float>& row_scale, float scale_row,
                         std::vector<int32_t> map_i, std::vector<int32_t> map_j, std::vector<double> map_eu, std::vector<double> map_ev,
                         double map_c0, double map_cs, double map_v0, double map_vs, int det_rows, const std::string& device) {
    require(rows.ndim() == 2, "rows must be (n_rows, n)");
    const int nr = static_cast<int>(rows.shape(0)), n = static_cast<int>(rows.shape(1));
    const std::size_t count = static_cast<std::size_t>(nr) * n;
    require(!positions || (positions->ndim() == 2 && positions->size() == count && static_cast<int>(positions->shape(0)) == nr),
            "positions must be (n_rows, n)");
    if (positions)
        for (std::size_t q = 0; q < count; ++q) require(std::isfinite(positions->data()[q]), "positions must be finite");
    auto per_row = [&](std::size_t size) { return size == 0 || size == static_cast<std::size_t>(nr); };
    require(per_row(shifts.size()) && per_row(row_y.size()) && per_row(row_slope.size()) && per_row(col_scale.size()) &&
            per_row(row_scale.size()), "shifts, row_y, row_slope, col_scale, row_scale: one value per row (or none)");
    const Device dev = parse_device(device);
    PyDetectorMap map(std::move(map_i), std::move(map_j), std::move(map_eu), std::move(map_ev), map_c0, map_cs, map_v0, map_vs,
                      det_rows, n, max_abs(row_slope), max_abs(col_scale), max_abs(row_scale));
    std::vector<ResampleRow> row(nr);
    for (int r = 0; r < nr; ++r) {
        row[r].y = row_y.empty() ? 0.0 : row_y[r];
        row[r].shift = shifts.empty() ? shift : shift + shifts[r];
        row[r].row_slope = row_slope.empty() ? 0.f : row_slope[r];
        row[r].col_scale = col_scale.empty() ? 0.f : col_scale[r];
        row[r].row_scale = row_scale.empty() ? 0.f : row_scale[r];
    }
    ResampleGeometry g;
    g.map = map.view;
    g.slope_col = slope_col;
    g.scale_col = scale_col;
    g.scale_row = scale_row;
    const int S = shift_row_stride(n);
    HostOut out = new_array({static_cast<std::size_t>(nr), static_cast<std::size_t>(n)});
    nb::gil_scoped_release nogil;
    std::vector<float> work(static_cast<std::size_t>(nr) * S, 0.f);
    for (int r = 0; r < nr; ++r)
        std::copy(rows.data() + static_cast<std::size_t>(r) * n, rows.data() + static_cast<std::size_t>(r + 1) * n,
                  work.begin() + static_cast<std::size_t>(r) * S);
    if (dev == Device::CPU) {
        if (positions) cpu::resample_rows(work.data(), nr, n, positions->data());
        else cpu::resample_rows(work.data(), nr, n, row.data(), g);
    } else {
#ifdef HELIREC_WITH_CUDA
        cuda::DeviceBuffer<float> d;
        d.upload(work.data(), work.size());
        if (positions) {
            cuda::DeviceBuffer<double> dp;
            dp.upload(positions->data(), count);
            cuda::resample_rows(d.data(), nr, n, dp.data());
            d.download(work.data());
        } else {
            cuda::DeviceBuffer<ResampleRow> d_row;
            cuda::DeviceBuffer<int32_t> d_i, d_j;
            cuda::DeviceBuffer<double> d_eu, d_ev;
            d_row.upload(row.data(), row.size());
            if (map.view.n > 0) {
                d_i.upload(map.i.data(), map.i.size());
                d_j.upload(map.j.data(), map.j.size());
                d_eu.upload(map.eu.data(), map.eu.size());
                d_ev.upload(map.ev.data(), map.ev.size());
                g.map.i = d_i.data(); g.map.j = d_j.data(); g.map.eu = d_eu.data(); g.map.ev = d_ev.data();
                g.map.ev_f = nullptr;   // (the rows' positions are solved in double precision)
            }
            cuda::resample_rows(d.data(), nr, n, d_row.data(), g);
            d.download(work.data());
        }
#endif
    }
    for (int r = 0; r < nr; ++r)
        std::copy(work.begin() + static_cast<std::size_t>(r) * S, work.begin() + static_cast<std::size_t>(r) * S + n,
                  data(out) + static_cast<std::size_t>(r) * n);
    return out;
}

// Seconds of each of `repeats` calls of a row routine on (n_rows, n) rows of noise, without the transfers:
// "shift" = shift_rows by 0.37, "positions" = resample_rows at the positions j - 0.37 (no map to solve),
// "map" = resample_rows through the given detector map and scales (rows spread over the detector's rows, shift 0.37).
std::vector<double> time_rows_py(const std::string& what, int n_rows, int n, int repeats, const std::string& device,
                                 std::vector<int32_t> map_i, std::vector<int32_t> map_j, std::vector<double> map_eu,
                                 std::vector<double> map_ev, double map_c0, double map_cs, double map_v0, double map_vs, int det_rows,
                                 float row_slope, float slope_col, float col_scale, float scale_col, float row_scale, float scale_row) {
    require(what == "shift" || what == "positions" || what == "map", "what: shift, positions or map");
    require(n_rows > 0 && n > 1 && repeats > 0, "n_rows, n, repeats must be positive");
    const Device dev = parse_device(device);
    PyDetectorMap map(std::move(map_i), std::move(map_j), std::move(map_eu), std::move(map_ev), map_c0, map_cs, map_v0, map_vs,
                      det_rows, n, row_slope, col_scale, row_scale);
    const int S = shift_row_stride(n);
    const float shift = 0.37f;
    std::vector<float> work(static_cast<std::size_t>(n_rows) * S);
    std::uint32_t state = 12345u;
    for (auto& v : work) {
        state = state * 1664525u + 1013904223u;
        v = static_cast<float>(state >> 8) * (1.0f / 16777216.0f);
    }
    std::vector<double> positions;
    if (what == "positions") {
        positions.resize(static_cast<std::size_t>(n_rows) * n);
        for (std::size_t q = 0; q < positions.size(); ++q) positions[q] = static_cast<double>(q % n) - static_cast<double>(shift);
    }
    std::vector<ResampleRow> row(n_rows);
    for (int r = 0; r < n_rows; ++r) {
        row[r].y = (r + 0.5) * det_rows / n_rows - 0.5;
        row[r].shift = shift;
        row[r].row_slope = row_slope;
        row[r].col_scale = col_scale;
        row[r].row_scale = row_scale;
    }
    ResampleGeometry g;
    g.map = map.view;
    g.slope_col = slope_col;
    g.scale_col = scale_col;
    g.scale_row = scale_row;
    std::vector<double> seconds;
    nb::gil_scoped_release nogil;
    auto timed = [&](auto&& call, auto&& sync) {
        for (int i = 0; i < repeats; ++i) {
            sync();
            const auto t0 = std::chrono::steady_clock::now();
            call();
            sync();
            seconds.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
    };
    if (dev == Device::CPU) {
        timed([&] {
            if (what == "shift") cpu::shift_rows(work.data(), n_rows, n, shift);
            else if (what == "positions") cpu::resample_rows(work.data(), n_rows, n, positions.data());
            else cpu::resample_rows(work.data(), n_rows, n, row.data(), g);
        }, [] {});
    } else {
#ifdef HELIREC_WITH_CUDA
        cuda::DeviceBuffer<float> d;
        cuda::DeviceBuffer<double> dp, d_eu, d_ev;
        cuda::DeviceBuffer<ResampleRow> d_row;
        cuda::DeviceBuffer<int32_t> d_i, d_j;
        d.upload(work.data(), work.size());
        if (!positions.empty()) dp.upload(positions.data(), positions.size());
        d_row.upload(row.data(), row.size());
        if (map.view.n > 0) {
            d_i.upload(map.i.data(), map.i.size());
            d_j.upload(map.j.data(), map.j.size());
            d_eu.upload(map.eu.data(), map.eu.size());
            d_ev.upload(map.ev.data(), map.ev.size());
            g.map.i = d_i.data(); g.map.j = d_j.data(); g.map.eu = d_eu.data(); g.map.ev = d_ev.data();
            g.map.ev_f = nullptr;
        }
        timed([&] {
            if (what == "shift") cuda::shift_rows(d.data(), n_rows, n, shift);
            else if (what == "positions") cuda::resample_rows(d.data(), n_rows, n, dp.data());
            else cuda::resample_rows(d.data(), n_rows, n, d_row.data(), g);
        }, [] { HELIREC_CUDA_CHECK(cudaDeviceSynchronize()); });
#endif
    }
    return seconds;
}

SldParams sld_from(const nb::dict& c) {
    SldParams p;
    auto get = [&](const char* key, auto& dst) {
        if (c.contains(key) && !c[key].is_none()) dst = nb::cast<std::decay_t<decltype(dst)>>(c[key]);
    };
    get("sld_fraction", p.fraction);
    get("sld_length", p.length);
    get("sld_bin", p.bin);
    get("sld_stride", p.stride);
    get("sld_lit_outside", p.lit_outside);
    get("sld_bm18", p.bm18);
    get("sld_fractions", p.fractions);
    get("sld_lengths", p.lengths);
    get("sld_border_x", p.border_x);
    get("sld_border_y", p.border_y);
    return p;
}

// the backprojector of type "python" a Reconstruction asks for (its parameters as text, read back as --set reads them)
std::shared_ptr<ExternalBackprojector> python_backprojector(const FilterParams& p) {
    nb::gil_scoped_acquire gil;
    nb::object loads = nb::module_::import_("json").attr("loads");
    nb::dict d;
    for (const auto& [k, v] : p.values) d[k.c_str()] = loads(parse_value(v));
    return std::make_shared<PyExternalBackprojector>(d);
}

nb::object json_loads(const std::string& text) { return nb::module_::import_("json").attr("loads")(text); }

class PySliceReconstructor {
  public:
    // the reconstructor a Reconstruction made (shared with it)
    explicit PySliceReconstructor(std::shared_ptr<SliceReconstructor> rec) : impl_(std::move(rec)) {}
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
        cfg.sld = sld_from(c);
        f("n_grid", cfg.n_grid);
        f("angular_tolerance_steps", cfg.angular_tolerance_steps);
        f("luminosity_repartition", cfg.luminosity_repartition);
        f("x_interp", cfg.x_interp);
        f("sharpness_weighting", cfg.sharpness_weighting);
        f("sharp_f", cfg.sharp_f);
        f("sharp_band", cfg.sharp_band);
        f("z_interp", cfg.z_interp);
        f("angle_interp", cfg.angle_interp);
        f("exposure_arc", cfg.exposure_arc);
        f("rolling_shutter", cfg.rolling_shutter);
        f("angle_subsets", cfg.angle_subsets);
        f("angle_subset", cfg.angle_subset);
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
        f("ramp_filter_name", cfg.ramp_filter_name);
        if (nb::object o = get("ramp_filter_par"); o.is_valid() && !o.is_none()) cfg.ramp_filter_par = ramp_filter_par(to_text(o));
        if (c.contains("external_bp") && !nb::object(c["external_bp"]).is_none())
            cfg.external = std::make_shared<PyExternalBackprojector>(nb::cast<nb::dict>(c["external_bp"]));
        f("grid_size", cfg.grid_size);
        f("ghbp_legs", cfg.ghbp_legs);
        f("ghbp_reduction_steps", cfg.ghbp_reduction_steps);
        f("clip_outer_circle", cfg.clip_outer_circle);
        f("voxel_size_m", cfg.pixel_size_m);
        f("axis_source_meters", cfg.axis_source_meters);
        nb::gil_scoped_release nogil;
        impl_ = std::make_shared<SliceReconstructor>(std::move(scans), cfg);
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
        d["exposure_arc_bins"] = impl_->exposure_arc_bins();
        d["rolling_shutter"] = impl_->rolling_shutter();
        d["angle_interp"] = impl_->angle_interp();
        d["z_interp"] = impl_->z_interp();
        d["angle_subsets"] = impl_->config().angle_subsets;
        d["angle_subset"] = impl_->config().angle_subset;
        d["projections"] = impl_->projection_counts();
        return d;
    }
    nb::object reconstruct(int h_lo, int h_hi, bool debug, bool timing, std::optional<std::array<int, 4>> roi,
                           std::optional<std::array<int, 2>> next, std::optional<std::tuple<double, double, bool>> uint8,
                           bool float32, ProgressBar* progress, nb::object mask,
                           std::optional<std::array<int, 4>> region, OmeZarrWriter* float32_to, OmeZarrWriter* uint8_to,
                           std::int64_t z_to) {
        const int N = impl_->grid_size();
        SliceRoi r{0, N, 0, N};
        if (roi) r = SliceRoi{(*roi)[0], (*roi)[1], (*roi)[2], (*roi)[3]};
        require(0 <= r.x0 && r.x0 < r.x1 && r.x1 <= N && 0 <= r.y0 && r.y0 < r.y1 && r.y1 <= N,
                "roi (x0, x1, y0, y1) must lie within the slice");
        require(h_hi > h_lo, "empty height range");
        require(float32 || uint8, "no output: float32=False needs uint8=(lo, hi, zero_outside)");
        require(float32 || !debug, "debug=True returns the float32 slices");
        const std::vector<std::size_t> shape{static_cast<std::size_t>(h_hi - h_lo), static_cast<std::size_t>(r.y1 - r.y0),
                                             static_cast<std::size_t>(r.x1 - r.x0)};
        SliceOutput o;
        nb::object f32 = nb::none(), u8 = nb::none();
        // a writer takes an output as the blocks computed, in place of an array
        auto sink = [&](OmeZarrWriter* w, const char* dtype, const char* what) -> BlockSink* {
            require(w->dtype() == dtype, std::string(what) + ": the writer's dtype is not the output's");
            require(static_cast<std::size_t>(w->shape()[1]) == shape[1] && static_cast<std::size_t>(w->shape()[2]) == shape[2] &&
                    z_to >= 0 && z_to + static_cast<std::int64_t>(shape[0]) <= w->shape()[0],
                    std::string(what) + ": the writer's volume does not hold these slices (z_to, y1 - y0, x1 - x0)");
            return w;
        };
        o.sink_z = z_to;
        require(!debug || !(float32_to || uint8_to), "debug=True returns arrays");
        if (float32 && float32_to) {
            o.f32_sink = sink(float32_to, "<f4", "float32_to");
        } else if (float32) {
            HostOut a = new_array(shape);
            o.f32 = data(a);
            f32 = nb::cast(a);
        }
        if (uint8) {
            require(std::get<1>(*uint8) > std::get<0>(*uint8), "uint8=(lo, hi, zero_outside) needs hi > lo");
            if (uint8_to) {
                o.u8_sink = sink(uint8_to, "|u1", "uint8_to");
            } else {
                auto a = new_array_t<std::uint8_t>(shape);
                o.u8 = static_cast<std::uint8_t*>(a.data());
                u8 = nb::cast(a);
            }
            o.u8_lo = std::get<0>(*uint8);
            o.u8_hi = std::get<1>(*uint8);
            o.u8_zero_outside = std::get<2>(*uint8);
        }
        if (region) {
            require((*region)[0] < (*region)[1] && (*region)[2] < (*region)[3], "region (x0, x1, y0, y1) is empty");
            o.region = SliceRoi{(*region)[0], (*region)[1], (*region)[2], (*region)[3]};
        }
        require(mask.is_none() || uint8, "mask applies to the uint8 output: pass uint8=(lo, hi, zero_outside)");
        std::optional<MaskIn> mask_array;   // (kept alive through the call)
        if (nb::isinstance<PyOutputMask>(mask)) {
            o.u8_mask = nb::cast<PyOutputMask&>(mask).m;
        } else if (!mask.is_none()) {
            mask_array = nb::cast<MaskIn>(mask);
            require(mask_array->ndim() == 3 && mask_array->shape(0) == shape[0] && mask_array->shape(1) == shape[1] &&
                    mask_array->shape(2) == shape[2],
                    "a mask array must be (n, y1 - y0, x1 - x0), the shape of the output (an OutputMask may be coarser)");
            o.u8_mask.data = mask_array->data();
            o.u8_mask.nz = static_cast<int>(shape[0]);
            o.u8_mask.ny = static_cast<int>(shape[1]);
            o.u8_mask.nx = static_cast<int>(shape[2]);
        }
        ChunkDebug dbg;
        dbg.keep = debug;
        dbg.timing = timing;
        {
            nb::gil_scoped_release nogil;
            impl_->reconstruct(h_lo, h_hi, o, &dbg, &r, next ? (*next)[0] : 0, next ? (*next)[1] : 0, progress);
        }
        if (!debug && !timing) {
            if (float32 && uint8) return nb::make_tuple(f32, u8);
            return float32 ? f32 : u8;
        }
        nb::dict sec;
        for (const auto& [k, v] : dbg.seconds) sec[k.c_str()] = v;
        for (const auto& [k, v] : dbg.stats) sec[("stat_" + k).c_str()] = v;
        nb::dict d;
        if (o.f32) d["slices"] = f32;
        if (o.u8) d["uint8"] = u8;
        if (timing) d["seconds"] = sec;
        if (!debug) return d;
        // (the sinogram's rows: the grid angles, or with angle_interp views the bins the chunk's projections span,
        // row r being bin first_bin + r)
        const std::size_t DY = h_hi - h_lo, NG = dbg.n_rows > 0 ? dbg.n_rows : impl_->n_grid(), NX = impl_->merged_width();
        d["first_bin"] = dbg.first_bin;
        auto mk = [&](const std::vector<float>& v, std::vector<std::size_t> shape) {
            HostOut a = new_array(shape);
            std::memcpy(data(a), v.data(), v.size() * sizeof(float));
            return a;
        };
        d["sino"] = mk(dbg.sino, {NG, DY, NX});
        d["weights"] = mk(dbg.weights, {NG, DY, NX});
        d["wbal"] = mk(dbg.wbal, {NG, DY, NX});
        d["filtered"] = mk(dbg.filtered, {DY, NG, NX});
        // the angles each slice is backprojected at (radians, GHBP / nabu sign: -the exact backprojector's)
        d["angles"] = mk(dbg.angles, {DY, NG});
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
    std::shared_ptr<SliceReconstructor> impl_;
};

}  // namespace

void bind_intensity(nb::module_& m);  // bindings/intensity_bindings.cpp
void bind_spectra(nb::module_& m);  // bindings/spectra_bindings.cpp
void bind_calibration(nb::module_& m);  // bindings/calibration_bindings.cpp
void bind_projection(nb::module_& m);  // bindings/projection_bindings.cpp
void bind_geometry(nb::module_& m);  // bindings/geometry_bindings.cpp
void bind_geometry_calibration(nb::module_& m);  // bindings/geometry_calibration_bindings.cpp
void bind_scan(nb::module_& m);  // bindings/scan_bindings.cpp

NB_MODULE(_core, m) {
    register_filter("python", [](const FilterParams& p, const FilterContext&) -> std::unique_ptr<ProjectionFilter> {
        return std::make_unique<PythonFilter>(p);
    });

    bind_intensity(m);
    bind_spectra(m);
    bind_calibration(m);
    bind_projection(m);
    bind_geometry(m);
    bind_geometry_calibration(m);
    bind_scan(m);
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
          "clip_radius"_a = -1.f, "device"_a = "cpu", "interp"_a = 0, "u_scale"_a = 1.f, "tiles"_a = nb::none(),
          "Exact pixel-driven parallel-beam backprojection; interp 0 linear, 1 cubic (Keys), 2 lanczos3, 3 bspline3, 4 bspline5 "
          "(B-spline: pass bspline_prefilter'd rows); rows "
          "sampled u_scale times per bin. tiles: only these 32 x 32 pixel tiles of the slice (index ty * ceil(nx / 32) + "
          "tx) are computed, the others are 0");
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
    m.def("shift_rows",
          [](HostIn rows, float shift, const std::string& device) {
              require(rows.ndim() == 2, "rows must be (n_rows, n)");
              const int nr = static_cast<int>(rows.shape(0)), n = static_cast<int>(rows.shape(1));
              const int S = shift_row_stride(n);
              HostOut out = new_array({static_cast<std::size_t>(nr), static_cast<std::size_t>(n)});
              nb::gil_scoped_release nogil;
              std::vector<float> work(static_cast<std::size_t>(nr) * S, 0.f);
              for (int r = 0; r < nr; ++r)
                  std::copy(rows.data() + static_cast<std::size_t>(r) * n, rows.data() + static_cast<std::size_t>(r + 1) * n,
                            work.begin() + static_cast<std::size_t>(r) * S);
              if (parse_device(device) == Device::CPU) {
                  cpu::shift_rows(work.data(), nr, n, shift);
              } else {
#ifdef HELIREC_WITH_CUDA
                  cuda::DeviceBuffer<float> d;
                  d.upload(work.data(), work.size());
                  cuda::shift_rows(d.data(), nr, n, shift);
                  d.download(work.data());
#endif
              }
              for (int r = 0; r < nr; ++r)
                  std::copy(work.begin() + static_cast<std::size_t>(r) * S, work.begin() + static_cast<std::size_t>(r) * S + n,
                            data(out) + static_cast<std::size_t>(r) * n);
              return out;
          },
          "rows"_a, "shift"_a, "device"_a = "cpu",
          "rows delayed by `shift` samples without loss (a phase ramp on their transform): out[r, i] = row r at i - shift");
    m.def("resample_rows", &resample_rows_py, "rows"_a, nb::arg("positions") = nb::none(), "shift"_a = 0.f,
          "shifts"_a = std::vector<float>{}, "row_y"_a = std::vector<double>{}, "row_slope"_a = std::vector<float>{},
          "slope_col"_a = 0.f, "col_scale"_a = std::vector<float>{}, "scale_col"_a = 0.f, "row_scale"_a = std::vector<float>{},
          "scale_row"_a = 0.f, "map_i"_a = std::vector<int32_t>{}, "map_j"_a = std::vector<int32_t>{},
          "map_eu"_a = std::vector<double>{}, "map_ev"_a = std::vector<double>{}, "map_c0"_a = 0.0, "map_cs"_a = 1.0,
          "map_v0"_a = 0.0, "map_vs"_a = 1.0, "det_rows"_a = 1, "device"_a = "cpu",
          "rows read at positions of their own without loss (sino.hpp): out[r, j] = the band-limited interpolant of row "
          "r, the one shift_rows delays, at positions[r, j] (float64, (n_rows, n)); or, without positions, at the "
          "detector column p that a detector map puts at the ideal column X = j - shift - shifts[r]: p + Eu(p, v) + "
          "col_scale[r] (p - scale_col) = X, v + Ev(p, v) + row_scale[r] (v - scale_row) = row_y[r] + row_slope[r] (p - "
          "slope_col) (detector_map.hpp; the rows span the detector's columns, det_rows its rows)");
    m.def("time_rows", &time_rows_py, "what"_a, "n_rows"_a, "n"_a, "repeats"_a = 5, "device"_a = "cpu",
          "map_i"_a = std::vector<int32_t>{}, "map_j"_a = std::vector<int32_t>{}, "map_eu"_a = std::vector<double>{},
          "map_ev"_a = std::vector<double>{}, "map_c0"_a = 0.0, "map_cs"_a = 1.0, "map_v0"_a = 0.0, "map_vs"_a = 1.0,
          "det_rows"_a = 1, "row_slope"_a = 0.f, "slope_col"_a = 0.f, "col_scale"_a = 0.f, "scale_col"_a = 0.f,
          "row_scale"_a = 0.f, "scale_row"_a = 0.f,
          "benchmark: seconds of each of `repeats` calls of a row routine on (n_rows, n) rows, transfers excluded: what = "
          "'shift' (shift_rows), 'positions' (resample_rows at given positions) or 'map' (resample_rows through the "
          "detector map and scales given, as for resample_rows); the first call includes the transform plans");
    m.def("detector_map_plan",
          [](std::vector<int32_t> map_i, std::vector<int32_t> map_j, std::vector<double> map_eu, std::vector<double> map_ev,
             double map_c0, double map_cs, double map_v0, double map_vs, int det_rows, int det_cols, double row_slope,
             double col_scale, double row_scale) {
              PyDetectorMap map(std::move(map_i), std::move(map_j), std::move(map_eu), std::move(map_ev), map_c0, map_cs, map_v0,
                                map_vs, det_rows, det_cols, row_slope, col_scale, row_scale);
              nb::dict d;
              d["passes"] = map.plan.passes;
              d["max_eu"] = map.plan.max_eu;
              d["max_ev"] = map.plan.max_ev;
              d["gradient"] = map.plan.gradient;
              d["contraction"] = map.plan.contraction;
              d["settle"] = map.plan.settle;
              d["tolerance"] = detector_map_tolerance;
              return d;
          },
          "map_i"_a, "map_j"_a, "map_eu"_a, "map_ev"_a, "map_c0"_a, "map_cs"_a, "map_v0"_a, "map_vs"_a, "det_rows"_a, "det_cols"_a,
          "row_slope"_a = 0.0, "col_scale"_a = 0.0, "row_scale"_a = 0.0,
          "the inversion of a detector map (detector_map.hpp): fixed-point passes, bounds of |Eu|, |Ev| and of their "
          "gradient over the detector, for projections whose |row_slope|, |col_scale|, |row_scale| are at most those given");
    m.def("detector_map_source",
          [](DoubleIn x, DoubleIn y, std::vector<int32_t> map_i, std::vector<int32_t> map_j, std::vector<double> map_eu,
             std::vector<double> map_ev, double map_c0, double map_cs, double map_v0, double map_vs, int det_rows, int det_cols,
             double row_slope, double slope_col, double col_scale, double scale_col, double row_scale, double scale_row) {
              require(x.size() == y.size(), "x and y must have the same size");
              PyDetectorMap map(std::move(map_i), std::move(map_j), std::move(map_eu), std::move(map_ev), map_c0, map_cs, map_v0,
                                map_vs, det_rows, det_cols, row_slope, col_scale, row_scale);
              auto c = new_array_t<double>({x.size()}), v = new_array_t<double>({x.size()});
              detector_map_source(map.view, x.data(), y.data(), x.size(), row_slope, slope_col, col_scale, scale_col, row_scale,
                                  scale_row, c.data(), v.data());
              return nb::make_tuple(c, v);
          },
          "x"_a, "y"_a, "map_i"_a, "map_j"_a, "map_eu"_a, "map_ev"_a, "map_c0"_a, "map_cs"_a, "map_v0"_a, "map_vs"_a, "det_rows"_a,
          "det_cols"_a, "row_slope"_a = 0.0, "slope_col"_a = 0.0, "col_scale"_a = 0.0, "scale_col"_a = 0.0, "row_scale"_a = 0.0,
          "scale_row"_a = 0.0,
          "the detector positions (c, v) read for the ideal columns x of rows whose ideal row is y at column slope_col, "
          "as resample_rows solves them (double precision): c + Eu + col_scale (c - scale_col) = x, v + Ev + row_scale "
          "(v - scale_row) = y + row_slope (c - slope_col)");
    m.def("detector_map_source_row",
          [](IntIn c, IntIn base, HostIn pos, std::vector<int32_t> map_i, std::vector<int32_t> map_j, std::vector<double> map_eu,
             std::vector<double> map_ev, double map_c0, double map_cs, double map_v0, double map_vs, int det_rows, int det_cols,
             float row_scale, float scale_row) {
              require(c.size() == pos.size() && base.size() == pos.size(), "c, base and pos must have the same size");
              PyDetectorMap map(std::move(map_i), std::move(map_j), std::move(map_eu), std::move(map_ev), map_c0, map_cs, map_v0,
                                map_vs, det_rows, det_cols, 0.0, 0.0, row_scale);
              HostOut out = new_array({pos.size()});
              detector_map_source_row(map.view, c.data(), base.data(), pos.data(), pos.size(), row_scale, scale_row, data(out));
              return out;
          },
          "c"_a, "base"_a, "pos"_a, "map_i"_a, "map_j"_a, "map_eu"_a, "map_ev"_a, "map_c0"_a, "map_cs"_a, "map_v0"_a, "map_vs"_a,
          "det_rows"_a, "det_cols"_a, "row_scale"_a = 0.f, "scale_row"_a = 0.f,
          "the detector row read at the detector column c for the ideal row base + pos, minus base, as the gridding "
          "solves it (single precision): v + Ev(c, v) + row_scale (v - scale_row) = base + pos");
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

    m.def("sld_correct_config", [](HostIn frames, const nb::dict& config) {
        require(frames.ndim() == 3, "frames must be (n, rows, cols)");
        auto p = sld_from(config);
        HostOut out = copy_of(frames);
        if (p.enabled()) {
            nb::gil_scoped_release nogil;
            ScintillatorGlare g(static_cast<int>(frames.shape(1)), static_cast<int>(frames.shape(2)), p);
            g.correct(data(out), static_cast<int>(frames.shape(0)));
        }
        return out;
    }, "frames"_a, "config"_a, "Correct frames with a resolved SLD configuration (helirec.sld.sld_config)");

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
          "dark"_a, "flats"_a, "weight"_a, "reader"_a, "file_index"_a, "frame_index"_a, "z_translation_scale"_a = 1.0,
          "row_offset"_a = std::vector<float>{}, "row_slope"_a = std::vector<float>{}, "slope_col"_a = 0.f,
          "col_offset"_a = std::vector<float>{}, "col_slope"_a = std::vector<float>{}, "slope_row"_a = 0.f,
          "map_i"_a = std::vector<int32_t>{}, "map_j"_a = std::vector<int32_t>{}, "map_eu"_a = std::vector<double>{},
          "map_ev"_a = std::vector<double>{}, "map_c0"_a = 0.0, "map_cs"_a = 1.0, "map_v0"_a = 0.0, "map_vs"_a = 1.0,
          "col_scale"_a = std::vector<float>{}, "scale_col"_a = 0.f, "row_scale"_a = std::vector<float>{}, "scale_row"_a = 0.f,
          "row_offset, row_slope, slope_col: registration of the scan's frames per projection (helical.hpp): rows added "
          "to the vertical position, and a row offset of row_slope (column - slope_col); col_offset, col_slope, "
          "slope_row: the lateral counterpart, a column offset of col_offset + col_slope (row - slope_row), applied "
          "where columns are shifted exactly (x_interp=sinc). map_*: the scan's detector map (detector_map.hpp): pixel "
          "(c, v) looks at the ideal position (c + Eu, v + Ev), Eu = sum map_eu x^map_i y^map_j, Ev likewise, x = (c - "
          "map_c0) / map_cs, y = (v - map_v0) / map_vs; every projection is read on the ideal grid. col_scale, "
          "scale_col, row_scale, scale_row: per projection, pixel (c, v) looks at the ideal column c + Eu + col_scale "
          "(c - scale_col) and the ideal row v + Ev + row_scale (v - scale_row). Lateral parts need x_interp=sinc");
    nb::class_<PySliceReconstructor>(m, "SliceReconstructor")
        .def(nb::init<std::vector<ScanData>, const nb::dict&>(), "scans"_a, "config"_a)
        .def_prop_ro("info", &PySliceReconstructor::info)
        .def("reconstruct", &PySliceReconstructor::reconstruct, "h_lo"_a, "h_hi"_a, "debug"_a = false,
             "timing"_a = false, "roi"_a = nb::none(), "next"_a = nb::none(), "uint8"_a = nb::none(), "float32"_a = true,
             nb::arg("progress").none() = nb::none(), "mask"_a = nb::none(), "region"_a = nb::none(),
             nb::arg("float32_to").none() = nb::none(), nb::arg("uint8_to").none() = nb::none(), "z_to"_a = 0,
             "slices for view heights [h_lo, h_hi) -> (n, N, N), or (n, y1 - y0, x1 - x0) with roi=(x0, x1, y0, y1); "
             "uint8=(lo, hi, zero_outside): also the uint8 slices of that window (to_uint8; 0 outside the "
             "reconstruction disk with zero_outside), converted on the device with CUDA -> (float32, uint8), or the "
             "uint8 slices alone with float32=False; "
             "debug=True returns a dict with intermediates; timing=True returns a dict with 'slices' / 'uint8' and "
             "per-stage 'seconds' (GPU synchronised per stage); next=(h_lo, h_hi): the chunk that will follow, read "
             "in the background (CUDA); progress: a ProgressBar advanced by one per slice; mask: the uint8 slices are 0 "
             "where it is 0: a uint8 array of the output's shape, or an OutputMask (possibly coarser, e.g. "
             "ReferenceMask.read); region=(x0, x1, y0, y1): only these pixels of the slice are computed, the rest of "
             "the output is 0. With uint8 slices alone, only the pixels within the bounds of what the mask keeps are "
             "computed (the others are 0 either way); the pixels computed are those of a whole-slice run. "
             "float32_to / uint8_to: an OmeZarrWriter that takes the float32 / uint8 slices as slices [z_to, z_to + n) of "
             "its volume, each as the block of it that was computed, so that no whole slice is built; None is returned "
             "in place of that array")
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

    // ---------------------------------------------------------------- the reconstruction of a scan pair (reconstruction.hpp)
    m.def("reconstruction_default_parameters", &default_parameters, "the default parameters (JSON text)");
    m.def("reconstruction_legacy_parameters", &legacy_reference_parameters, "the parameters mode legacy_reference pins (JSON text)");
    m.def("reconstruction_calibration_keys", &calibration_keys);
    m.def("reconstruction_mode_conventions", &mode_conventions, "mode"_a);
    m.def("reconstruction_display_window", &display_window, "mode"_a);
    m.def("parse_filter_chain", &parse_filter_chain, "text"_a, "a filter chain from its text (JSON text of the list)");
    m.def("parse_value", &parse_value, "text"_a, "--set's reading of a value (JSON text)");
    m.def("override_filter_chain", &override_filter_chain, "chain"_a, "changes"_a, "a chain with TYPE[.k].param changes (JSON texts)");
    m.def("filter_chain_text", &filter_chain_text, "chain"_a);
    m.def("is_filter_key", &is_filter_key, "key"_a);
    m.def("parse_settings", &parse_settings, "items"_a, "--set KEY=VALUE items as a JSON object of changes (text)");
    m.def("update_parameters", &update_parameters, "params"_a, "changes"_a, "the parameters changed (JSON texts)");
    m.def("sld_config", [](const std::string& spec) { std::string r; sld_params(spec, &r); return r; }, "spec"_a,
          "a scintillator light diffusion model resolved to the reconstructor's keys (JSON text)");
    m.def("backprojector_config", &backprojector_config, "spec"_a, "a backprojector spec as the reconstructor's settings (JSON text)");
    m.def("angle_subset_config", &parse_angle_subset, "spec"_a);
    m.def("build_revision", &build_revision);
    m.def("write_png", [](const std::string& path, nb::ndarray<const std::uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> img) {
        write_png(path, img.data(), static_cast<int>(img.shape(0)), static_cast<int>(img.shape(1)));
    }, "path"_a, "img"_a, "(ny, nx) uint8 as an 8-bit grey PNG");
    m.def(
        "geometry_calibration_file",
        [](const std::string& given, const std::string& cache_dir, const std::vector<std::string>& masters, const std::string& row_time_us, const std::string& device,
           std::optional<std::function<void(const std::string&)>> log,
           std::optional<std::function<std::string(std::vector<std::string>, std::optional<double>, std::string)>> make) {
            std::function<void(const std::string&)> say;
            if (log) say = [log](const std::string& line) { (*log)(line); };
            GeometryMaker maker;
            if (make)
                maker = [make](const std::vector<std::string>& m, const calibration::geometry_calibration::Options& o) {
                    return (*make)(m, std::isnan(o.row_time) ? std::nullopt : std::optional<double>(o.row_time), o.device == Device::CUDA ? "cuda" : "cpu");
                };
            const auto r = geometry_calibration_file(given, cache_dir, masters, row_time_us, parse_device(device), say, maker);
            return nb::make_tuple(r.first, r.second);
        },
        "given"_a, "cache_dir"_a, "masters"_a, "row_time_us"_a = "", "device"_a = "cpu", "log"_a = nb::none(), "make"_a = nb::none(),
        "(path, text) of the geometry calibration: given, or geometry.json of cache_dir made by make(masters, row_time_s, device) (default: the "
        "geometry calibration step) when absent or of other scans (helirec::geometry_calibration_file)");

    nb::class_<Reconstruction>(m, "Reconstruction",
                               "The reconstruction of a scan pair from its raw folders: the scans, the geometry and intensity calibrations, "
                               "and a SliceReconstructor of the parameters (helirec::Reconstruction, reconstruction.hpp)")
        .def(
            "__init__",
            [](Reconstruction* self, const std::string& projections, const std::string& params, const std::string& cache_dir, const std::string& device,
               std::optional<std::function<void(const std::string&)>> log, bool progress) {
                Reconstruction::Options o;
                o.cache_dir = cache_dir;
                o.device = parse_device(device);
                o.progress = progress;
                if (log)
                    o.log = [log](const std::string& line) {
                        nb::gil_scoped_acquire gil;
                        (*log)(line);
                    };
                o.external_backprojector = python_backprojector;
                nb::gil_scoped_release nogil;
                new (self) Reconstruction(projections, params, std::move(o));
            },
            "projections"_a, "params"_a = "{}", "cache_dir"_a = "", "device"_a = "cuda", "log"_a = nb::none(), "progress"_a = false)
        .def("set_parameters", [](Reconstruction& r, const std::string& changes) {
            nb::gil_scoped_release nogil;
            r.set_parameters(changes);
        }, "changes"_a)
        .def_prop_ro("parameters", &Reconstruction::parameters)
        .def_prop_ro("main", [](Reconstruction& r) -> const calibration::Scan& { return r.main(); }, nb::rv_policy::reference_internal)
        .def_prop_ro("companion", [](Reconstruction& r) -> const calibration::Scan& { return r.companion(); }, nb::rv_policy::reference_internal)
        .def_prop_ro("intensity", [](Reconstruction& r) -> const calibration::IntensityCalibration& { return r.intensity(); }, nb::rv_policy::reference_internal)
        .def_prop_ro("reconstructor", [](Reconstruction& r) { return PySliceReconstructor(r.reconstructor()); })
        .def_prop_ro("doable", &Reconstruction::doable)
        .def_prop_ro("mode", &Reconstruction::mode)
        .def("rot_center", &Reconstruction::rot_center)
        .def("rolling_shutter", &Reconstruction::rolling_shutter)
        .def("exposure_arc", &Reconstruction::exposure_arc)
        .def("sharpness_weighting", &Reconstruction::sharpness_weighting)
        .def("resolved_sld", &Reconstruction::resolved_sld)
        .def("info", &Reconstruction::info)
        .def("slice_info", [](const Reconstruction& r, double f) {
            const SliceInfo s = r.slice_info(f);
            return nb::make_tuple(s.fraction, s.view_height, s.nabu_index, s.height_mm);
        }, "fraction"_a)
        .def("provenance", &Reconstruction::provenance);

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

    m.def("htj2k_float_frames_supported", &htj2k_float_frames_supported);
    nb::class_<RawReader>(m, "RawReader", "pread-based uint16/float32 HDF5 reader with JPEG 2000 resolution selection")
        .def(nb::init<std::vector<std::string>, std::string, int, int, int, std::string, bool, std::string>(), "files"_a, "dataset"_a, "rows"_a, "cols"_a, "reduce"_a = 0, "dtype"_a = "uint16", "selective"_a = true, "decoder"_a = "auto",
             "dtype: uint16 detector frames or float32 reconstructed volumes. decoder=openjph enables selective uint16 HTJ2K reads too. "
             "rows, cols: of a frame as stored. reduce 1, 2, 3: frames stored as JPEG 2000 are read at a half, a quarter, an eighth "
             "(the codestream's own lower resolutions: rows, cols and every row index are then the reduced ones; value (j, k) is "
             "the wavelet's low band at the stored pixel (j, k) * 2**reduce).")
        .def_prop_ro("rows", &RawReader::rows)
        .def_prop_ro("cols", &RawReader::cols)
        .def_prop_ro("reduce", &RawReader::reduce)
        .def("storage", [](RawReader& r, int file) {
            RawReader::Storage s;
            { nb::gil_scoped_release nogil; s = r.storage(file); }
            nb::dict d;
            d["compressed"] = s.compressed; d["chunk_rows"] = s.chunk_rows;
            d["codec"] = s.codec; d["reversible"] = s.reversible;
            d["frames"] = s.frames; d["bit_depth"] = s.bit_depth;
            return d;
        }, "file"_a = 0)
        .def_prop_ro("dtype", &RawReader::dtype)
        .def_prop_ro("bytes_read", &RawReader::bytes_read,
                     "Payload bytes requested, excluding HDF5 metadata and filesystem read-ahead.")
        .def("set_prefetch_threads", &RawReader::set_prefetch_threads, "threads"_a)
        .def("prefetch_rows", [](RawReader& r, IntIn files, IntIn frames, IntIn row0, int nrows) {
            require(files.shape(0) == frames.shape(0) && files.shape(0) == row0.shape(0), "index arrays must have equal length");
            require(nrows >= 0, "nrows must be nonnegative");
            nb::gil_scoped_release nogil;
            r.prefetch_rows(files.data(), frames.data(), row0.data(), files.shape(0), nrows);
        }, "file_index"_a, "frame_index"_a, "row0"_a, "nrows"_a)
        .def("set_cache_bytes", &RawReader::set_cache_bytes, "bytes"_a,
             "Decoded chunks of compressed frames kept between reads, up to this many bytes (0, the default: none).")
        .def("read_rows",
             [](RawReader& r, IntIn file_index, IntIn frame_index, IntIn row0, int nrows, int n_threads) -> nb::ndarray<nb::numpy> {
                 const std::size_t n = file_index.shape(0);
                 require(frame_index.shape(0) == n && row0.shape(0) == n, "index arrays must have equal length");
                 require(nrows >= 0, "nrows must be nonnegative");
                 if (r.dtype() == "float32") {
                     auto out = new_array_t<float>({n, static_cast<std::size_t>(nrows), static_cast<std::size_t>(r.cols())});
                     nb::gil_scoped_release nogil;
                     r.read_rows(file_index.data(), frame_index.data(), row0.data(), n, nrows, out.data(), n_threads);
                     return nb::ndarray<nb::numpy>(out);
                 }
                 auto out = new_array_t<uint16_t>({n, static_cast<std::size_t>(nrows), static_cast<std::size_t>(r.cols())});
                 nb::gil_scoped_release nogil;
                 r.read_rows(file_index.data(), frame_index.data(), row0.data(), n, nrows,
                             static_cast<uint16_t*>(out.data()), n_threads);
                 return nb::ndarray<nb::numpy>(out);
             },
             "file_index"_a, "frame_index"_a, "row0"_a, "nrows"_a, "n_threads"_a = 0,
             "rows [row0, row0+nrows) of each frame -> (n, nrows, cols) in reader dtype; outside rows are 0")
        .def("read_frames",
             [](RawReader& r, IntIn file_index, IntIn frame_index, int n_threads) -> nb::ndarray<nb::numpy> {
                 const std::size_t n = file_index.shape(0);
                 require(frame_index.shape(0) == n, "index arrays must have equal length");
                 if (r.dtype() == "float32") {
                     auto out = new_array_t<float>({n, static_cast<std::size_t>(r.rows()), static_cast<std::size_t>(r.cols())});
                     nb::gil_scoped_release nogil;
                     r.read_frames(file_index.data(), frame_index.data(), n, out.data(), n_threads);
                     return nb::ndarray<nb::numpy>(out);
                 }
                 auto out = new_array_t<uint16_t>({n, static_cast<std::size_t>(r.rows()), static_cast<std::size_t>(r.cols())});
                 nb::gil_scoped_release nogil;
                 r.read_frames(file_index.data(), frame_index.data(), n, static_cast<uint16_t*>(out.data()), n_threads);
                 return nb::ndarray<nb::numpy>(out);
             },
             "file_index"_a, "frame_index"_a, "n_threads"_a = 0);
    m.def("minus_log", &minus_log_py, "x"_a, "clip_min"_a = 1e-6f, "clip_max"_a = 10.f, "device"_a = "cpu");
    m.def("to_uint8", &to_uint8_py, "img"_a, "lo"_a, "hi"_a, "disk"_a = nb::none(), "device"_a = "cpu", "mask"_a = nb::none(),
          "(..., ny, nx) float32 -> uint8: rint(clip((img - lo) / (hi - lo), 0, 1) * 255) in float32, NaN -> 0. "
          "disk=(n, x0, y0): the frames are the ROI at (x0, y0) of (n, n) slices, 0 outside their reconstruction disk "
          "(x - c)^2 + (y - c)^2 > (n // 2)^2, c = (n - 1) / 2; mask: the result is 0 where it is 0: a uint8 array of "
          "img's shape, or an OutputMask over the frames (possibly coarser)");

    nb::class_<PyOutputMask>(m, "OutputMask",
                             "A mask of a chunk's uint8 slices, possibly coarser than them: output voxel (s, y, x) is 0 "
                             "where array[(s + oz) // fz, (y + oy) // fy, (x + ox) // fx] is 0 (indices clamped)")
        .def("__init__", [](PyOutputMask* self, nb::ndarray<const std::uint8_t, nb::ndim<3>, nb::c_contig, nb::device::cpu> a,
                            std::array<int, 3> scale, std::array<int, 3> offset) {
                 require(a.size() > 0 && scale[0] >= 1 && scale[1] >= 1 && scale[2] >= 1, "empty mask or scale < 1");
                 new (self) PyOutputMask();
                 self->m.data = a.data();
                 self->m.nz = static_cast<int>(a.shape(0)); self->m.ny = static_cast<int>(a.shape(1)); self->m.nx = static_cast<int>(a.shape(2));
                 self->m.fz = scale[0]; self->m.fy = scale[1]; self->m.fx = scale[2];
                 self->m.oz = offset[0]; self->m.oy = offset[1]; self->m.ox = offset[2];
                 self->keep = nb::cast(a);
             }, "array"_a, "scale"_a = std::array<int, 3>{1, 1, 1}, "offset"_a = std::array<int, 3>{0, 0, 0})
        .def_prop_ro("shape", [](const PyOutputMask& p) { return nb::make_tuple(p.m.nz, p.m.ny, p.m.nx); })
        .def_prop_ro("scale", [](const PyOutputMask& p) { return nb::make_tuple(p.m.fz, p.m.fy, p.m.fx); })
        .def_prop_ro("offset", [](const PyOutputMask& p) { return nb::make_tuple(p.m.oz, p.m.oy, p.m.ox); });

    nb::class_<ReferenceMask>(m, "ReferenceMask",
                              "The mask of a masked uint8 reference volume on the reconstruction's grid, read with "
                              "vc_reader from any of its pyramid levels (`scale`; 0: full resolution). shape: the "
                              "reconstruction's whole volume (z, y, x): the reference's level 0 must have its y and "
                              "x, and its number of slices to within 2 % (slice_ratio).")
        .def("__init__", [](ReferenceMask* self, const std::string& location, int scale, std::array<std::int64_t, 3> shape,
                            const std::string& cache_dir) {
                 nb::gil_scoped_release nogil;
                 new (self) ReferenceMask(location, scale, shape, cache_dir);
             }, "location"_a, "scale"_a, "shape"_a, "cache_dir"_a = "")
        .def_prop_ro("location", &ReferenceMask::location)
        .def_prop_ro("scale", &ReferenceMask::scale)
        .def_prop_ro("factors", &ReferenceMask::factors, "volume voxels per mask voxel (z, y, x)")
        .def_prop_ro("level_shape", &ReferenceMask::level_shape)
        .def_prop_ro("slice_ratio", &ReferenceMask::slice_ratio,
                     "reference slices per slice of the volume: 1 on the same grid; else the reference's slices are "
                     "spread evenly over the volume's (a reconstruction with another helical pitch calibration)")
        .def("read", [](nb::object self, int z0, int z1, std::array<int, 4> roi, int slot) {
                 auto& ref = nb::cast<ReferenceMask&>(self);
                 PyOutputMask out;
                 {
                     nb::gil_scoped_release nogil;
                     out.m = ref.read(z0, z1, SliceRoi{roi[0], roi[1], roi[2], roi[3]}, slot);
                 }
                 out.keep = self;
                 return out;
             }, "z0"_a, "z1"_a, "roi"_a, "slot"_a = 0,
             "the OutputMask of volume slices [z0, z1) of roi = (x0, x1, y0, y1); its data is this object's buffer "
             "`slot`, valid until the next read into that slot");

    nb::class_<OmeZarrWriter>(m, "OmeZarrWriter",
                              "Streaming OME-Zarr (NGFF 0.4, Zarr v2) writer of a (z, y, x) uint8 or float32 volume: level "
                              "0 is stored z-chunk by z-chunk as slices arrive (any order, each once), chunks raw, zlib or "
                              "(uint8) lossless vc-delta3d, on worker threads; chunks that are all 0 are not stored; "
                              "close() builds the further pyramid levels (2x mean pooling). A SliceReconstructor writes "
                              "into it directly (reconstruct(float32_to= / uint8_to=))")
        .def("__init__", [](OmeZarrWriter* self, const std::string& path, std::array<std::int64_t, 3> shape, nb::object dtype,
                            std::array<std::int64_t, 3> chunks, double voxel_size_um, std::array<std::int64_t, 3> origin_zyx,
                            int levels, const std::string& compression, int zlib_level, nb::object name, nb::object attrs,
                            bool overwrite, int threads, bool skip_empty, const std::string& delta3d_preset) {
                 OmeZarrOptions o;
                 o.shape = shape;
                 o.dtype = nb::cast<std::string>(nb::module_::import_("numpy").attr("dtype")(dtype).attr("str"));
                 require(o.dtype == "|u1" || o.dtype == "<f4", "dtype must be uint8 or float32");
                 o.chunks = chunks;
                 o.voxel_size_um = voxel_size_um;
                 o.origin = origin_zyx;
                 o.levels = levels;
                 o.compression = compression;
                 o.zlib_level = zlib_level;
                 if (!name.is_none()) o.name = nb::cast<std::string>(name);
                 if (!attrs.is_none()) o.attrs_json = nb::cast<std::string>(nb::module_::import_("json").attr("dumps")(attrs));
                 o.overwrite = overwrite;
                 o.threads = threads;
                 o.skip_empty = skip_empty;
                 o.delta3d_preset = delta3d_preset;
                 new (self) OmeZarrWriter(path, o);
             }, "path"_a, "shape"_a, "dtype"_a, "chunks"_a = std::array<std::int64_t, 3>{32, 256, 256}, "voxel_size_um"_a = 1.0,
             "origin_zyx"_a = std::array<std::int64_t, 3>{0, 0, 0}, "levels"_a = 1, "compression"_a = "none", "zlib_level"_a = 1,
             "name"_a = nb::none(), "attrs"_a = nb::none(), "overwrite"_a = false, "threads"_a = 16, "skip_empty"_a = true,
             "delta3d_preset"_a = "fast")
        .def_prop_ro("path", &OmeZarrWriter::path)
        .def_prop_ro("shape", [](const OmeZarrWriter& w) { return nb::make_tuple(w.shape()[0], w.shape()[1], w.shape()[2]); })
        .def_prop_ro("chunks", [](const OmeZarrWriter& w) { return nb::make_tuple(w.chunks()[0], w.chunks()[1], w.chunks()[2]); })
        .def_prop_ro("dtype", [](const OmeZarrWriter& w) { return nb::module_::import_("numpy").attr("dtype")(w.dtype()); })
        .def_prop_ro("levels", &OmeZarrWriter::levels)
        .def_prop_ro("level_shapes", [](const OmeZarrWriter& w) {
            nb::list out;
            for (const auto& s : w.level_shapes()) out.append(nb::make_tuple(s[0], s[1], s[2]));
            return out;
        })
        .def_prop_ro("n_written", &OmeZarrWriter::n_written, "chunks stored so far")
        .def_prop_ro("n_skipped", &OmeZarrWriter::n_skipped, "chunks not stored, being all 0")
        .def("write", [](OmeZarrWriter& w, std::int64_t z0, nb::object data) {
                 // (converted to the volume's dtype as NumPy does, when it is another)
                 nb::object a = nb::module_::import_("numpy").attr("ascontiguousarray")(data, w.dtype());
                 const void* p = nullptr;
                 std::size_t n = 0, ny = 0, nx = 0;
                 auto take = [&](const auto& v) {
                     require(v.ndim() == 3, "data must be (n, Y, X)");
                     p = v.data(); n = v.shape(0); ny = v.shape(1); nx = v.shape(2);
                 };
                 if (w.dtype() == "|u1") take(nb::cast<nb::ndarray<const std::uint8_t, nb::c_contig, nb::device::cpu>>(a));
                 else take(nb::cast<nb::ndarray<const float, nb::c_contig, nb::device::cpu>>(a));
                 require(static_cast<std::int64_t>(ny) == w.shape()[1] && static_cast<std::int64_t>(nx) == w.shape()[2] && z0 >= 0 &&
                         z0 + static_cast<std::int64_t>(n) <= w.shape()[0], "the slices do not fit the volume");
                 nb::gil_scoped_release nogil;
                 w.write(z0, static_cast<std::int64_t>(n), p);
             }, "z0"_a, "data"_a, "write slices [z0, z0 + len(data)) of level 0 (any order / granularity, each slice once)")
        .def("close", [](OmeZarrWriter& w, nb::object progress) {
                 std::function<std::function<void()>(int, std::int64_t)> cb;
                 if (!progress.is_none())
                     cb = [&progress](int level, std::int64_t slabs) -> std::function<void()> {
                         nb::gil_scoped_acquire gil;
                         nb::object step = progress(level, slabs);
                         if (step.is_none()) return {};
                         auto held = std::shared_ptr<nb::object>(new nb::object(std::move(step)), [](nb::object* o) {
                             nb::gil_scoped_acquire gil;
                             delete o;
                         });
                         return [held] {
                             nb::gil_scoped_acquire gil;
                             (*held)();
                         };
                     };
                 nb::gil_scoped_release nogil;
                 w.close(cb);
             }, "progress"_a = nb::none(),
             "store what is left and build the pyramid levels. progress(level, n_slabs): called before a level is built; "
             "its return value, when not None, is called after each of the level's slabs")
        .def("read", [](const OmeZarrWriter& w, int level, std::int64_t z0, std::int64_t z1) {
                 require(level >= 0 && level < w.levels(), "no such level");
                 const auto& s = w.level_shapes()[level];
                 require(0 <= z0 && z0 <= z1 && z1 <= s[0], "slices outside the level");
                 const std::vector<std::size_t> shape{static_cast<std::size_t>(z1 - z0), static_cast<std::size_t>(s[1]),
                                                      static_cast<std::size_t>(s[2])};
                 nb::object out;
                 void* p = nullptr;
                 if (w.dtype() == "|u1") {
                     auto a = new_array_t<std::uint8_t>(shape);
                     p = a.data();
                     out = nb::cast(a);
                 } else {
                     HostOut a = new_array(shape);
                     p = data(a);
                     out = nb::cast(a);
                 }
                 {
                     nb::gil_scoped_release nogil;
                     w.read(level, z0, z1, p);
                 }
                 return out;
             }, "level"_a, "z0"_a, "z1"_a, "slices [z0, z1) of a level (whole y, x) as stored: after close()");

    nb::class_<ProgressBar>(m, "ProgressBar",
                            "tqdm-style terminal progress bar: redrawn in place on a terminal, one line every "
                            "log_interval seconds otherwise. Thread-safe; usable as a context manager.")
        .def("__init__", [](ProgressBar* self, std::uint64_t total, const std::string& desc, const std::string& unit,
                            int fd, double min_interval, double log_interval, int width, std::optional<bool> terminal,
                            bool leave) {
                 ProgressBar::Options o;
                 o.desc = desc;
                 o.unit = unit;
                 o.fd = fd;
                 o.min_interval = min_interval;
                 o.log_interval = log_interval;
                 o.width = width;
                 o.terminal = terminal ? (*terminal ? 1 : 0) : -1;
                 o.leave = leave;
                 new (self) ProgressBar(total, o);
             }, "total"_a, "desc"_a = "", "unit"_a = "it", "fd"_a = 2, "min_interval"_a = 0.1, "log_interval"_a = 10.0,
             "width"_a = 0, "terminal"_a = nb::none(), "leave"_a = true)
        .def("update", [](ProgressBar& b, std::uint64_t n) { nb::gil_scoped_release nogil; b.update(n); }, "n"_a = 1)
        .def("set_postfix", [](ProgressBar& b, const std::string& t) { nb::gil_scoped_release nogil; b.set_postfix(t); }, "text"_a)
        .def("set_description", [](ProgressBar& b, const std::string& t) { nb::gil_scoped_release nogil; b.set_description(t); }, "desc"_a)
        .def("write", [](ProgressBar& b, const std::string& t) { nb::gil_scoped_release nogil; b.write(t); }, "line"_a,
             "print a line above the bar")
        .def("refresh", [](ProgressBar& b) { nb::gil_scoped_release nogil; b.refresh(); })
        .def("close", [](ProgressBar& b) { nb::gil_scoped_release nogil; b.close(); })
        .def_prop_ro("count", &ProgressBar::count)
        .def_prop_ro("total", &ProgressBar::total)
        .def("__enter__", [](nb::object self) { return self; })
        .def("__exit__", [](ProgressBar& b, nb::handle, nb::handle, nb::handle) { nb::gil_scoped_release nogil; b.close(); },
             "type"_a.none(), "value"_a.none(), "traceback"_a.none())
        .def_static("format", &ProgressBar::format, "n"_a, "total"_a, "elapsed"_a, "rate"_a, "desc"_a = "", "unit"_a = "it",
                    "postfix"_a = "", "width"_a = 80, "unicode"_a = true, "the meter text");
    m.def("arc_deblur_taps", &arc_deblur_taps, "arc"_a, "radius"_a = 4,
          "Taps g[0..radius] of the inverse of an exposure arc of `arc` grid angles (1 / sinc(arc f), unit sum)");
    m.def("angular_fir", &angular_fir_py, "sino"_a, "taps"_a, "support"_a = nb::none(), "device"_a = "cpu",
          "Symmetric periodic FIR along the angle of a sinogram (n_proj, n_cols); with support, only where every tap "
          "has support > 0");
    m.def("rebalance_turns", &rebalance_turns_py, "R"_a, "W"_a, "turn"_a, "rot_center"_a, "device"_a = "cpu",
          "rebalance for a sinogram with one row per view over more than one turn (`turn` rows per turn): empty angles "
          "filled, samples without weight taken from the rows whole turns away, weights balanced over every recording "
          "of a line (rows i + k turn / 2); returns (R, W_bal) copies");
    m.def("rebalance", &rebalance_py, "R"_a, "W"_a, "rot_center"_a, "legacy_reference"_a = false, "device"_a = "cpu",
          "Hole filling + half-acquisition balanced weights (opposite view, periodic; legacy_reference: the nabuxx "
          "endpoint handling); returns (R, W_bal) copies");
    m.def("ramp_filter", &ramp_filter_py, "sino"_a, "rot_center"_a, "pad_factor"_a = 1, "normalize"_a = true,
          "legacy_reference"_a = false, "device"_a = "cpu", "filter_name"_a = "ramlak", "filter_par"_a = "",
          "opposite"_a = std::vector<int32_t>(), "turn"_a = 0,
          "Ramp filter with opposite-ray padding, (n_proj, n_bins) -> same shape. normalize: x 2 pi / n_proj "
          "(attenuation for pair-balanced full-turn data; legacy_reference: pi / n_proj and integer padding reflection). "
          "filter_name: ramlak or a windowed ramp (ramp_filter_names), filter_par 'p0:p1'. opposite, turn: for rows "
          "that are not one full turn (one row per view), the row holding each row's opposite view and the rows of a "
          "turn (normalize: x 2 pi / turn)");
    m.def("ramp_filter_names", &ramp_filter_names, "ramp filters of fbp / ghbp and gridrec");
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
