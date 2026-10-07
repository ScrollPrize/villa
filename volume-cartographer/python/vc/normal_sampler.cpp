#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <stdexcept>
#include <vector>

#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "vc/fiber_tracer/FiberReferenceReplayCore.hpp"
#include "vc/fiber_tracer/FiberReplay.hpp"
#include "vc/fiber_tracer/FiberReferenceReplayBenchmark.hpp"
#include <atomic>
#include <memory>

namespace nb = nanobind;
using namespace nb::literals;

namespace {
// Own the dataset for the complete sampler lifetime. All loading, caching,
// decoding and interpolation remain in the same library used by native tracing.
class NormalSampler {
    vc::lasagna::LasagnaDataset dataset_;
    vc::lasagna::LasagnaNormalSampler sampler_;
public:
    std::string replay(nb::ndarray<const double, nb::numpy, nb::ndim<2>, nb::c_contig> points,
                       nb::callable runner, double nominalStep, double threshold,
                       double grace, double tangentFactor, double refine, double maxStepFactor) {
        if (points.shape(1) != 3 || points.shape(0) < 2)
            throw std::invalid_argument("expected reference [N,3] base XYZ points");
        if (!(nominalStep>0 && maxStepFactor>0 && threshold>0 && tangentFactor>0 && grace>=0 && refine>=0)
            || !std::isfinite(nominalStep+maxStepFactor+threshold+tangentFactor+grace+refine))
            throw std::invalid_argument("invalid neural replay settings");
        std::vector<cv::Vec3d> reference(points.shape(0));
        for (size_t i=0; i<reference.size(); ++i)
            for (int a=0; a<3; ++a) {
                const double v=points.data()[3*i+a];
                if (!std::isfinite(v)) throw std::invalid_argument("nonfinite reference point");
                reference[i][a]=v;
            }
        const auto geometry=vc::fiber_tracer::makePolylineArcGeometry(reference);
        vc::fiber_tracer::ReplaySegmentRunner segment = [&](const auto& start, double remaining,
                int maximumSteps, const auto& observer, const auto&) -> std::string {
            nb::gil_scoped_acquire acquire;
            auto active=std::make_shared<bool>(true);
            auto observe=nb::cpp_function([active, observer](
                    nb::ndarray<const double, nb::numpy, nb::ndim<2>, nb::c_contig> batch,
                    int firstStep) {
                if (!*active) throw std::runtime_error("expired replay observer");
                if (batch.shape(1)!=3 || firstStep<0) throw std::invalid_argument("invalid committed points");
                nb::gil_scoped_release release;
                for(size_t i=0;i<batch.shape(0);++i) {
                    cv::Vec3d p(batch.data()[3*i],batch.data()[3*i+1],batch.data()[3*i+2]);
                    for(int a=0;a<3;++a) if(!std::isfinite(p[a])) throw std::invalid_argument("nonfinite committed point");
                    if(!observer(p,0.,firstStep+int(i))) return false;
                }
                return true;
            });
            try {
                auto xyz=[](const cv::Vec3d& p){return nb::make_tuple(p[0],p[1],p[2]);};
                const auto reason=nb::cast<std::string>(runner(xyz(start.point),xyz(start.tangent),remaining,maximumSteps,observe));
                *active=false;
                return reason;
            } catch (...) { *active=false; throw; }
        };
        nb::gil_scoped_release release;
        const auto result=vc::fiber_tracer::traceReferenceReplayCore(geometry,0.,geometry.length(),
            nominalStep,maxStepFactor,threshold,grace,tangentFactor,refine,sampler_,1.,segment,{},{});
        vc::fiber_tracer::FiberReferenceReplayCase replayCase;
        replayCase.pointsBaseXYZ=reference;
        replayCase.referenceLengthBaseVoxels=geometry.length();
        const std::vector outcomes{vc::fiber_tracer::measureFiberReferenceReplayOutcome(replayCase,result)};
        const auto summary=vc::fiber_tracer::summarizeFiberReferenceReplay(1,outcomes,1.);
        auto report=vc::fiber_tracer::fiberReferenceReplayBenchmarkJson(summary,outcomes,
            "neural_follow",{{"error_threshold_base_voxels",threshold}},nlohmann::json::object(),{0,0,0},{0,0,0});
        return nlohmann::json{{"replay",vc::fiber_tracer::fiberReplayTraceJson(result,threshold)},
            {"case",report.at("cases").at(0)}}.dump();
    }
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
             "Return (unit normals XYZ, validity). Missing/zero grad_mag is invalid.")
        .def("replay", &NormalSampler::replay, "points_xyz"_a, "run_segment"_a,
             "nominal_step_base"_a, "threshold_base"_a=40., "grace_base"_a=160.,
             "tangential_factor"_a=2., "match_refine_steps"_a=1., "max_step_factor"_a=3.,
             "Run the shared native reference replay protocol with a callback tracer. Returns replay JSON.");
}
