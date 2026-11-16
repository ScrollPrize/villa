// nanobind bindings of the block cross-spectra and the registered reference volume (block_spectra.hpp, reference.hpp)
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "helirec/block_spectra.hpp"
#include "helirec/reference.hpp"
#include "helirec/zarr.hpp"
#include "ndarray_utils.hpp"

#ifdef HELIREC_WITH_CUDA
#include "../src/cuda/cuda_utils.cuh"
#endif

namespace nb = nanobind;
using namespace nb::literals;
using namespace helirec;
using namespace helirec::py;

namespace {

using Vol = nb::ndarray<const float, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
using Table = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Mat = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

nb::ndarray<nb::numpy, double> table_array(const std::vector<double>& v, const BlockSpectraSpec& s) {
    auto out = new_array_t<double>({static_cast<std::size_t>(s.n_shells()), static_cast<std::size_t>(s.thetas)});
    std::copy(v.begin(), v.end(), static_cast<double*>(out.data()));
    return out;
}

nb::ndarray<nb::numpy, double> curve_array(const std::vector<double>& v) {
    auto out = new_array_t<double>({v.size()});
    std::copy(v.begin(), v.end(), static_cast<double*>(out.data()));
    return out;
}

std::array<double, 12> affine_of(const Mat& m) {
    require(m.shape(0) == 3 && m.shape(1) == 4, "affine must be 3 x 4");
    std::array<double, 12> a;
    std::copy(m.data(), m.data() + 12, a.begin());
    return a;
}

class PyBlockCorrelator {
  public:
    PyBlockCorrelator(int n, int oversample, int shells, int thetas, bool align, float align_band, int max_shift,
                      const std::string& device)
        : device_(parse_device(device)) {
        BlockSpectraSpec s;
        s.n = n; s.oversample = oversample; s.shells = shells; s.thetas = thetas;
        s.align = align; s.align_band = align_band; s.max_shift = max_shift;
        if (device_ == Device::CPU) impl_ = cpu::make_block_correlator(s);
#ifdef HELIREC_WITH_CUDA
        else impl_ = cuda::make_block_correlator(s);
#endif
    }
    const BlockSpectraSpec& spec() const { return impl_->spec(); }
    nb::ndarray<nb::numpy, double> counts() const { return table_array(impl_->counts(), spec()); }

    nb::dict compute(Vol a, Vol b, std::optional<std::array<double, 3>> shift) {
        const auto& s = spec();
        const std::size_t n = static_cast<std::size_t>(s.n), m = n * s.oversample;
        require(a.shape(0) == n && a.shape(1) == n && a.shape(2) == n, "a must be (n, n, n)");
        require(b.shape(0) == m && b.shape(1) == m && b.shape(2) == m, "b must be (n * oversample)^3");
        BlockSpectra out;
        {
            nb::gil_scoped_release nogil;
            const double* sh = shift ? shift->data() : nullptr;
            if (device_ == Device::CPU) {
                impl_->compute(a.data(), b.data(), out, sh);
            }
#ifdef HELIREC_WITH_CUDA
            else {
                da_.upload(a.data(), a.size());
                db_.upload(b.data(), b.size());
                impl_->compute(da_.data(), db_.data(), out, sh);
            }
#endif
        }
        nb::dict d;
        d["aa"] = table_array(out.aa, s);
        d["bb"] = table_array(out.bb, s);
        d["ab"] = table_array(out.ab, s);
        d["shift"] = nb::make_tuple(out.shift[0], out.shift[1], out.shift[2]);
        d["peak"] = out.peak;
        d["mean_a"] = out.mean_a; d["mean_b"] = out.mean_b;
        d["var_a"] = out.var_a; d["var_b"] = out.var_b;
        return d;
    }

    nb::dict curves(Table aa, Table bb, Table ab, double theta_lo, double theta_hi) const {
        const auto& s = spec();
        const std::size_t nb_ = static_cast<std::size_t>(s.n_bins());
        require(aa.size() == nb_ && bb.size() == nb_ && ab.size() == nb_, "tables must be (n_shells, thetas)");
        auto vec = [&](const Table& t) { return std::vector<double>(t.data(), t.data() + nb_); };
        const ShellCurves c = shell_curves(s, impl_->counts(), vec(aa), vec(bb), vec(ab), theta_lo, theta_hi);
        nb::dict d;
        d["frequency"] = curve_array(c.frequency);
        d["count"] = curve_array(c.count);
        d["correlation"] = curve_array(c.correlation);
        d["matched"] = curve_array(c.matched);
        d["snr_split"] = curve_array(c.snr_split);
        d["snr_reference"] = curve_array(c.snr_reference);
        d["filter"] = curve_array(c.filter);
        d["power_a"] = curve_array(c.power_a);
        d["power_b"] = curve_array(c.power_b);
        return d;
    }

  private:
    Device device_;
    std::unique_ptr<BlockCorrelator> impl_;
#ifdef HELIREC_WITH_CUDA
    cuda::DeviceBuffer<float> da_, db_;
#endif
};

HostOut resample_py(Vol src, Mat m, std::array<int, 3> shape, const std::string& interp, const std::string& device) {
    const std::array<double, 12> a = affine_of(m);
    require(shape[0] > 0 && shape[1] > 0 && shape[2] > 0, "shape must be positive");
    const int k = resample_interp_from_name(interp);
    const Device dev = parse_device(device);
    HostOut out = new_array({static_cast<std::size_t>(shape[0]), static_cast<std::size_t>(shape[1]), static_cast<std::size_t>(shape[2])});
    const int sz = static_cast<int>(src.shape(0)), sy = static_cast<int>(src.shape(1)), sx = static_cast<int>(src.shape(2));
    nb::gil_scoped_release nogil;
    if (dev == Device::CPU) {
        cpu::resample_affine(src.data(), sz, sy, sx, a.data(), data(out), shape[0], shape[1], shape[2], k);
        return out;
    }
#ifdef HELIREC_WITH_CUDA
    cuda::DeviceBuffer<float> d_out(static_cast<std::size_t>(shape[0]) * shape[1] * shape[2]);
    cuda::resample_affine_host(src.data(), sz, sy, sx, a.data(), d_out.data(), shape[0], shape[1], shape[2], k);
    d_out.download(data(out));
#endif
    return out;
}

}  // namespace

void bind_spectra(nb::module_& m) {
    nb::class_<PyBlockCorrelator>(m, "BlockCorrelator",
                                  "Cross-spectra of two volumes on cubic blocks in polar frequency bins (Fourier shell "
                                  "correlation and what derives from it; block_spectra.hpp). a: (n, n, n) blocks of the "
                                  "volume under test; b: the comparison, (n * oversample)^3 samples of the same block "
                                  "(only the frequencies of a's grid are used). Bins: n_shells shells of |k| of width "
                                  "0.5 / shells cycles per voxel (out to the cube's corners) x thetas bins of the angle "
                                  "of k from the z axis. align: b is registered on a by a sub-voxel shift found from the "
                                  "frequencies below align_band (within max_shift voxels).")
        .def(nb::init<int, int, int, int, bool, float, int, const std::string&>(), "n"_a, "oversample"_a = 1,
             "shells"_a = 16, "thetas"_a = 6, "align"_a = true, "align_band"_a = 0.25f, "max_shift"_a = 4,
             "device"_a = "cpu")
        .def_prop_ro("n", [](const PyBlockCorrelator& c) { return c.spec().n; })
        .def_prop_ro("oversample", [](const PyBlockCorrelator& c) { return c.spec().oversample; })
        .def_prop_ro("shells", [](const PyBlockCorrelator& c) { return c.spec().shells; })
        .def_prop_ro("n_shells", [](const PyBlockCorrelator& c) { return c.spec().n_shells(); })
        .def_prop_ro("thetas", [](const PyBlockCorrelator& c) { return c.spec().thetas; })
        .def_prop_ro("counts", &PyBlockCorrelator::counts, nb::rv_policy::move, "frequencies in each bin (n_shells, thetas)")
        .def("compute", &PyBlockCorrelator::compute, "a"_a, "b"_a, "shift"_a = nb::none(),
             "One block -> dict: aa, bb, ab (n_shells, thetas) sums of |A|^2, |B|^2, Re(A conj B), normalised so that "
             "aa sums to the window-weighted variance of a; shift (z, y, x): the displacement of b's content from a's "
             "that was removed (voxels of a); peak: the whitened band-limited correlation there; mean_a, mean_b, "
             "var_a, var_b (window-weighted). shift=(z, y, x): use this shift instead of finding one.")
        .def("curves", &PyBlockCorrelator::curves, "aa"_a, "bb"_a, "ab"_a, "theta_lo"_a = 0.0, "theta_hi"_a = 90.0,
             "Tables (one block's, or sums over blocks) -> curves over the shells for the polar angles [theta_lo, "
             "theta_hi] degrees (0, 90: the shell correlation; 60, 90: in-plane; 0, 30: axial): frequency, count, "
             "correlation, matched (the correlation after the best per-bin linear filter: unchanged by filters of a), "
             "snr_split = c / (1 - c), snr_reference = c^2 / (1 - c^2), filter = ab / aa, power_a, "
             "power_b");

    m.def("resample_affine", &resample_py, "src"_a, "m"_a, "shape"_a, "interp"_a = "lanczos3", "device"_a = "cpu",
          "out[k, j, i] (shape) = the (z, y, x) volume src at (z, y, x) = m (k, j, i, 1), m 3 x 4; interp linear, "
          "cubic or lanczos3; 0 outside the volume");

    m.def("zarr_levels", [](const std::string& location, const std::string& cache_dir) {
              std::vector<OmeZarrLevel> levels;
              std::vector<ZarrArrayMeta> metas;
              {
                  nb::gil_scoped_release nogil;
                  levels = ome_zarr_levels(location, cache_dir);
                  for (const auto& l : levels) metas.push_back(ZarrArray::open(l.location, cache_dir).meta());
              }
              nb::list out;
              for (std::size_t i = 0; i < levels.size(); ++i) {
                  nb::dict d;
                  d["level"] = levels[i].level;
                  d["location"] = levels[i].location;
                  d["scale"] = nb::make_tuple(levels[i].scale[0], levels[i].scale[1], levels[i].scale[2]);
                  d["offset"] = nb::make_tuple(levels[i].offset[0], levels[i].offset[1], levels[i].offset[2]);
                  d["shape"] = nb::make_tuple(metas[i].shape[0], metas[i].shape[1], metas[i].shape[2]);
                  d["dtype"] = metas[i].dtype;
                  out.append(d);
              }
              return out;
          }, "location"_a, "cache_dir"_a = "",
          "The levels of an OME-Zarr pyramid (or the one level of a plain Zarr v2 array), local or http(s):// / s3://: "
          "dicts with level, location, scale and offset (voxel i lies at scale * i + offset in level-0 voxels; z, y, "
          "x), shape, dtype");
    m.def("zarr_read", [](const std::string& location, std::array<std::int64_t, 3> start, std::array<std::int64_t, 3> shape,
                          const std::string& cache_dir) -> nb::object {
              require(shape[0] > 0 && shape[1] > 0 && shape[2] > 0, "shape must be positive");
              const ZarrArray a = ZarrArray::open(location, cache_dir);
              for (int i = 0; i < 3; ++i)
                  require(start[i] >= 0 && start[i] + shape[i] <= a.meta().shape[i], "region outside the array");
              const std::vector<std::size_t> dims{static_cast<std::size_t>(shape[0]), static_cast<std::size_t>(shape[1]),
                                                  static_cast<std::size_t>(shape[2])};
              if (a.meta().dtype == "<f4") {
                  HostOut out = new_array(dims);
                  {
                      nb::gil_scoped_release nogil;
                      a.read(start, shape, data(out));
                  }
                  return nb::cast(out);
              }
              auto out = new_array_t<std::uint8_t>(dims);
              {
                  nb::gil_scoped_release nogil;
                  a.read(start, shape, out.data());
              }
              return nb::cast(out);
          }, "location"_a, "start"_a, "shape"_a, "cache_dir"_a = "",
          "The region [start, start + shape) (z, y, x) of a Zarr v2 array (one level: zarr_levels), uint8 or float32; "
          "0 where no chunk is stored");

    nb::class_<RegisteredVolume>(m, "RegisteredVolume",
                                 "A reference volume on another grid (a Zarr v2 array or OME-Zarr pyramid, uint8 or "
                                 "float32), resampled on blocks of the reconstruction's. affine: 3 x 4, rows (z, y, x): "
                                 "the reference's level-0 voxel index of the reconstruction's voxel index (z, y, x, 1). "
                                 "level: the pyramid level read.")
        .def("__init__", [](RegisteredVolume* self, const std::string& location, int level, Mat affine,
                            const std::string& cache_dir) {
                 const std::array<double, 12> a = affine_of(affine);
                 nb::gil_scoped_release nogil;
                 new (self) RegisteredVolume(location, level, a, cache_dir);
             }, "location"_a, "level"_a, "affine"_a, "cache_dir"_a = "")
        .def_prop_ro("location", &RegisteredVolume::location)
        .def_prop_ro("level", &RegisteredVolume::level)
        .def_prop_ro("level_shape", &RegisteredVolume::level_shape, nb::rv_policy::move)
        .def("read", [](const RegisteredVolume& v, std::array<double, 3> origin, double step, std::array<int, 3> shape,
                        const std::string& interp, const std::string& device) {
                 require(shape[0] > 0 && shape[1] > 0 && shape[2] > 0, "shape must be positive");
                 const int k = resample_interp_from_name(interp);
                 const Device dev = parse_device(device);
                 HostOut out = new_array({static_cast<std::size_t>(shape[0]), static_cast<std::size_t>(shape[1]),
                                          static_cast<std::size_t>(shape[2])});
                 double coverage = 0;
                 {
                     nb::gil_scoped_release nogil;
                     if (dev == Device::CPU) {
                         coverage = v.read(origin, step, shape, data(out), k, Device::CPU);
                     }
#ifdef HELIREC_WITH_CUDA
                     else {
                         cuda::DeviceBuffer<float> d(static_cast<std::size_t>(shape[0]) * shape[1] * shape[2]);
                         coverage = v.read(origin, step, shape, d.data(), k, Device::CUDA);
                         d.download(data(out));
                     }
#endif
                 }
                 return nb::make_tuple(out, coverage);
             }, "origin"_a, "step"_a, "shape"_a, "interp"_a = "lanczos3", "device"_a = "cpu",
             "the reference at the reconstruction's positions origin + step (k, j, i) (z, y, x), k < shape[0] ... -> "
             "(float32 array, fraction of the reference voxels read that are not 0)");
}
