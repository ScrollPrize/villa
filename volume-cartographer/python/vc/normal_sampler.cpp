#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <stdexcept>
#include <vector>

#include "vc/lasagna/LasagnaNormalSampler.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {
// Own the dataset for the complete sampler lifetime. All loading, caching,
// decoding and interpolation remain in the same library used by native tracing.
class NormalSampler {
    vc::lasagna::LasagnaDataset dataset_;
    vc::lasagna::LasagnaNormalSampler sampler_;
public:
    NormalSampler(const std::string& manifest, const std::string& cacheRoot)
        : dataset_(vc::lasagna::LasagnaDataset::openLocation(manifest, options(cacheRoot))),
          sampler_(dataset_) {}

    static vc::lasagna::LasagnaDatasetOpenOptions options(const std::string& root) {
        vc::lasagna::LasagnaDatasetOpenOptions result;
        result.remoteCacheRoot = root;
        result.workingToBaseScale = 1.0;
        return result;
    }

    nb::tuple sample(nb::ndarray<const float, nb::numpy, nb::ndim<2>, nb::c_contig> points,
                     int threads) {
        if (points.shape(1) != 3 || threads < 1)
            throw std::invalid_argument("expected [N,3] base XYZ points and threads >= 1");
        const size_t count = points.shape(0);
        auto np = nb::module_::import_("numpy");
        auto normals = np.attr("empty")(nb::make_tuple(count, 3), "dtype"_a="float32");
        auto valid = np.attr("empty")(count, "dtype"_a="bool");
        auto output = nb::cast<nb::ndarray<float, nb::numpy, nb::c_contig>>(normals);
        auto flags = nb::cast<nb::ndarray<bool, nb::numpy, nb::c_contig>>(valid);
        {
            nb::gil_scoped_release release;
            std::vector<cv::Vec3f> xyz(count);
            for (size_t i = 0; i < count; ++i)
                xyz[i] = {points.data()[3*i], points.data()[3*i+1], points.data()[3*i+2]};
            std::vector<vc::lasagna::LasagnaNormalSampler::FloatNormalSample> samples;
            (void)sampler_.sampleNormalBatch(xyz, threads, samples);
            for (size_t i = 0; i < count; ++i) {
                flags.data()[i] = samples[i].valid;
                for (int axis = 0; axis < 3; ++axis)
                    output.data()[3*i+axis] = samples[i].normal[axis];
            }
        }
        return nb::make_tuple(normals, valid);
    }
};
}

void bindNormalSampler(nb::module_& m) {
    nb::class_<NormalSampler>(m, "NormalSampler",
        "Native Lasagna normals in base XYZ coordinates; shared tracing interpolation/cache.")
        .def(nb::init<const std::string&, const std::string&>(),
             "manifest"_a, "cache_root"_a="", nb::call_guard<nb::gil_scoped_release>())
        .def("sample", &NormalSampler::sample, "points_xyz"_a, "threads"_a=1,
             "Return (unit normals XYZ, validity). Missing/zero grad_mag is invalid.");
}
