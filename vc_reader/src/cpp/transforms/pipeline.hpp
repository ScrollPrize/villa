#pragma once
#include "../cuda_available.hpp"
#include "cpu/tracing.hpp"
#include "cpu/palette.hpp"
#include "cpu/distance_transform.hpp"
#include "cpu/structure_tensor.hpp"
#include "cpu/ridges.hpp"
#include "cpu/coherence_enhancing_diffusion.hpp"
#include "cpu/bias_field.hpp"
#include "cpu/plasma_brightness_contrast.hpp"
#include "cpu/plasma_shadow.hpp"
#include "cpu/illumination.hpp"
#include "cpu/vignetting.hpp"
#include "cpu/random_tone_curve.hpp"
#include "cpu/multiplicative_noise.hpp"
#include "cpu/shot_noise.hpp"
#include "cpu/additive_noise.hpp"
#include "cpu/salt_and_pepper.hpp"
#include "cpu/motion_blur.hpp"
#include "cpu/defocus.hpp"
#include "cpu/zoom_blur.hpp"
#include "cpu/equalize.hpp"
#include "cpu/clahe.hpp"
#include "cpu/histogram_matching.hpp"
#include "cpu/pixel_distribution_adaptation.hpp"
#include "cpu/fourier_domain_adaptation.hpp"
#include "cpu/unsharp_mask.hpp"
#include "cpu/gibbs_ringing.hpp"
#include "cpu/ringing_overshoot.hpp"
#include "cpu/foreground_crop.hpp"
#include "cpu/channel_misalignment.hpp"
#include "cpu/grid_dropout.hpp"
#include "cpu/stripe_masking.hpp"
#include "cpu/foreground_dropout.hpp"
#include "cpu/object_dropout.hpp"
#include "validate.hpp"
#include <bit>

namespace vcr::transforms {
struct OneOf;
using Op = std::variant<
    BrightnessMultiply, BrightnessAdd, Contrast, Gamma, Invert, GaussianNoise,
    RicianNoise, ClipOutliers, GaussianBlur, MedianFilter, Sharpen,
    BrightnessGradient, LocalContrast, LocalGamma, LocalSmoothing, BlankRectangle,
    SliceIllumination, Smear, Flip, Rotate90, Transpose, CropPad, Affine, Elastic,
    SheetCompression, LowResolution, ThickSlice, Morphology, RandomBinaryOperator,
    RemoveComponents, ConnectedComponents, RidgeTrace, ReplaceLabel, Regions, OneHot, MaskImage, DeepSupervision,
    ToPseudo2D, FromPseudo2D, MedialSurface, StripGeometry, SurfacePerturb, std::shared_ptr<OneOf>, Normalize, OrientedBlur, DistanceTransform, StructureTensor, Ridges, CoherenceEnhancingDiffusion, BiasField, PlasmaBrightnessContrast, PlasmaShadow, Illumination, Vignetting, RandomToneCurve, MultiplicativeNoise, ShotNoise, AdditiveNoise, SaltAndPepper, MotionBlur, Defocus, ZoomBlur, Equalize, CLAHE, HistogramMatching, PixelDistributionAdaptation, FDA, UnsharpMask, GibbsRinging, RingingOvershoot, ForegroundCrop, ChannelMisalignment, GridDropout, StripeMasking, ForegroundDropout, ObjectDropout>;
using Executor = void (*)(const Op&, Sample&, const Parameters&, void*);
#ifdef VC_READER_CUDA
namespace cuda {
using vcr::cuda::available;
void execute(const Op&, Sample&, const Parameters&, void*);
// Uploads the sample; uint8 images become float32 on the device (their host fields then describe it).
std::shared_ptr<void> begin(Sample&);
void finish(Sample&, void*);
void finish_device(void*);
Parameters sample_parameters(const Op&, const Sample&, Random&, bool, void*);
void validate_program(const std::vector<Op>&);
}
#else
namespace cuda { using vcr::cuda::available; }
#endif
struct OneOf { std::vector<Op> transforms; std::vector<double> weights; double p = 1; };
inline void validate_transform(const Op& op) {
    std::visit([](const auto& t) {
        if constexpr (std::is_same_v<std::decay_t<decltype(t)>, std::shared_ptr<OneOf>>) { probability(t->p); for (const auto& child : t->transforms) validate_transform(child); }
        else validate_transform(t);
    }, op);
}
struct Program { std::vector<Op> transforms; std::map<std::string, Target> targets; };
#ifdef VC_READER_CUDA
namespace cuda {
void run_batch_steps(const Program&, std::vector<Sample>&, const std::vector<std::uint64_t>&,
                     const std::map<std::string,std::vector<double>>&, const std::vector<std::shared_ptr<void>>&);
}
#endif
// uint8 continuous targets other than images become float32 as a run starts (images: see narrow).
inline void widen_entries(Sample& s) {
    for (auto& [name, f] : s) if (narrow_continuous(f) && f.target.role != Role::Image) widen(f.array);
}
struct InputSpec {
    Shape shape; char kind; std::size_t bytes;
    bool operator==(const InputSpec&) const = default;
};
struct Step { Shape path; Parameters parameters; };
struct Plan {
    std::shared_ptr<const Program> program;
    std::map<std::string, InputSpec> inputs;
    std::vector<Step> steps;
};
struct RunResult { Sample sample; Plan plan; };
inline std::map<std::string, InputSpec> input_specs(const Sample& s) {
    std::map<std::string, InputSpec> result;
    // uint8 continuous inputs run as float32: either may replay the other's plan.
    for (const auto& [name, f] : s) result[name] = narrow_continuous(f) ? InputSpec{f.array.shape, 'f', 4} : InputSpec{f.array.shape, f.array.kind, f.array.bytes};
    return result;
}
inline double chance(const Op& op) {
    return std::visit([](const auto& t) {
        using T = std::decay_t<decltype(t)>;
        if constexpr (std::is_same_v<T, std::shared_ptr<OneOf>>) return t->p;
        else if constexpr (requires { t.options; }) return t.options.p;
        else return t.p;
    }, op);
}
inline const Op& resolve(const Program& program, const Shape& path) {
    require(!path.empty() && path[0] < program.transforms.size(), "invalid plan transform path");
    const Op* op = &program.transforms[path[0]];
    for (std::size_t i = 1; i < path.size(); ++i) {
        const auto* one = std::get_if<std::shared_ptr<OneOf>>(op);
        require(one && path[i] < (*one)->transforms.size(), "invalid plan OneOf path"); op = &(*one)->transforms[path[i]];
    }
    return *op;
}
inline void execute(const Op& op, Sample& s, const Parameters& p) {
    std::visit([&](const auto& t) {
        if constexpr (!std::is_same_v<std::decay_t<decltype(t)>, std::shared_ptr<OneOf>>) apply(t, s, p);
        else throw std::invalid_argument("replay plan contains unresolved OneOf");
    }, op);
}
inline void prepare(const Op& op, Sample& s, Random& r, Shape path, std::vector<Step>* record, CompactState& compact, Executor executor = nullptr, void* context = nullptr) {
    if (!r.chance(chance(op))) return;
    if (const auto* choice = std::get_if<std::shared_ptr<OneOf>>(&op)) {
        const auto& t = **choice; require(!t.transforms.empty(), "OneOf needs transforms");
        auto weights = t.weights; if (weights.empty()) weights.assign(t.transforms.size(), 1);
        require(weights.size() == t.transforms.size(), "OneOf weight count mismatch"); double total = 0;
        for (auto w : weights) { require(std::isfinite(w) && w >= 0, "invalid OneOf weight"); total += w; }
        require(std::isfinite(total) && total > 0, "OneOf needs positive total weight");
        double threshold = r.uniform() * total; std::size_t chosen = 0;
        for (; chosen + 1 < weights.size() && threshold >= weights[chosen]; ++chosen) threshold -= weights[chosen];
        path.push_back(chosen); prepare(t.transforms[chosen], s, r, std::move(path), record, compact, executor, context); return;
    }
    std::visit([&](const auto& t) { compact.before(t, s); }, op);
    Parameters parameters;
#ifdef VC_READER_CUDA
    if (executor) parameters = cuda::sample_parameters(op,s,r,record != nullptr,context);
    else
#endif
    parameters = std::visit([&](const auto& t) -> Parameters {
        if constexpr (!std::is_same_v<std::decay_t<decltype(t)>, std::shared_ptr<OneOf>>) return sample(t, s, r);
        else return {};
    }, op);
    if (executor) executor(op, s, parameters, context);
    else std::visit([&](const auto& t) {
        if constexpr (!std::is_same_v<std::decay_t<decltype(t)>, std::shared_ptr<OneOf>>) compact.apply_step(t, s, parameters);
    }, op);
    if (record) {
        parameters.device_field.reset();
        for (auto& c : parameters.channels) c.device_field.reset();
        record->push_back({std::move(path), std::move(parameters)});
    }
}
class Pipeline {
    std::shared_ptr<const Program> program_;
    bool compact_;
    Executor executor_ = nullptr;
    std::string backend_ = "cpu";
public:
    Pipeline(std::vector<Op> transforms, std::map<std::string, Target> targets, bool compact = true, std::string backend = "cpu")
        : program_(std::make_shared<Program>(Program{std::move(transforms), std::move(targets)})), compact_(compact) {
        require(backend == "cpu" || backend == "cuda" || backend == "auto", "backend must be cpu, cuda or auto");
        if (backend == "cuda" || (backend == "auto" && cuda::available())) {
            require(cuda::available(), "CUDA transform is unavailable: build with VC_READER_ENABLE_CUDA=ON and use a CUDA device");
#ifdef VC_READER_CUDA
            cuda::validate_program(program_->transforms);
            executor_ = cuda::execute; backend_ = "cuda"; compact_ = false;
#endif
        }
        require(!program_->targets.empty(), "target schema must not be empty");
        for (const auto& op : program_->transforms) validate_transform(op);
    }
    const auto& backend() const { return backend_; }
    const auto& targets() const { return program_->targets; }
    const auto& program() const { return program_; }
    RunResult run(Sample s, std::uint64_t seed, bool record, std::shared_ptr<void> context = {}) const {
        const bool resident = bool(context);
        require(!resident || executor_, "device inputs require a CUDA pipeline");
        validate(s, !resident); RunResult result; result.plan.program = program_; result.plan.inputs = input_specs(s);
        if (!resident) widen_entries(s);
        CompactState compact(compact_, program_->transforms.size() == 1);
#ifdef VC_READER_CUDA
        if (executor_ && !resident) context = cuda::begin(s);
#endif
        for (std::size_t i = 0; i < program_->transforms.size(); ++i) {
            for (const auto& [name, f] : s) require(f.pyramid.empty(), "DeepSupervision must be the last transform");
            Random r{split_seed(seed, std::to_string(i))};
            prepare(program_->transforms[i], s, r, {i}, record ? &result.plan.steps : nullptr, compact, executor_, context.get());
        }
#ifdef VC_READER_CUDA
        if (executor_) { if (resident) cuda::finish_device(context.get()); else cuda::finish(s,context.get()); }
#endif
        compact.flush(s); widen_images(s);
        result.sample = std::move(s); return result;
    }
    RunResult replay(Sample s, const Plan& plan, std::shared_ptr<void> context = {}) const {
        require(plan.program == program_, "plan belongs to another pipeline; use plan.apply(sample)");
        const bool resident = bool(context);
        require(!resident || executor_, "device inputs require a CUDA pipeline");
        validate(s, !resident); require(input_specs(s) == plan.inputs, "replay requires identical target names, shapes and dtypes");
        if (!resident) widen_entries(s);
        CompactState compact(compact_, program_->transforms.size() == 1);
#ifdef VC_READER_CUDA
        if (executor_ && !resident) context = cuda::begin(s);
#endif
        for (const auto& step : plan.steps) {
            if (executor_) { executor_(resolve(*program_, step.path), s, step.parameters, context.get()); continue; }
            std::visit([&](const auto& t) {
                if constexpr (!std::is_same_v<std::decay_t<decltype(t)>, std::shared_ptr<OneOf>>) {
                    compact.before(t, s); compact.apply_step(t, s, step.parameters);
                } else throw std::invalid_argument("replay plan contains unresolved OneOf");
            }, resolve(*program_, step.path));
        }
#ifdef VC_READER_CUDA
        if (executor_) { if (resident) cuda::finish_device(context.get()); else cuda::finish(s,context.get()); }
#endif
        compact.flush(s); widen_images(s);
        RunResult result; result.sample = std::move(s); return result;
    }
    static Pipeline direct(const Op& op, std::map<std::string, Target> targets, bool compact = true, std::string backend = "cpu");
};

// Direct calls. A transform is random only inside a Pipeline. Called on its own
// it always applies (p ignored) and draws everything else from direct_seed,
// so the same input gives the same output.
inline constexpr std::uint64_t direct_seed = 0;
inline Op always(Op op) {
    std::visit([](auto& t) {
        using T = std::decay_t<decltype(t)>;
        if constexpr (std::is_same_v<T, std::shared_ptr<OneOf>>) { auto copy = std::make_shared<OneOf>(*t); copy->p = 1; t = std::move(copy); }
        else if constexpr (requires { t.options; }) t.options.p = 1;
        else t.p = 1;
    }, op);
    return op;
}
inline Pipeline Pipeline::direct(const Op& op, std::map<std::string, Target> targets, bool compact, std::string backend) {
    return Pipeline({always(op)}, std::move(targets), compact, std::move(backend));
}
// The field a bare array stands for: the op's own `on` default. A named field
// (StructureTensor "image", OneHot "mask"), the first label of a label list
// (DistanceTransform, Morphology, MedialSurface: "mask"), else "image".
inline std::pair<std::string, Target> direct_input(const Op& op) {
    return std::visit([](const auto& t) -> std::pair<std::string, Target> {
        using T = std::decay_t<decltype(t)>;
        Target target; target.valid = "";
        if constexpr (std::is_same_v<T, std::shared_ptr<OneOf>>) {
            require(!t->transforms.empty(), "OneOf needs transforms");
            return direct_input(t->transforms.front());
        } else if constexpr (requires { { t.on } -> std::convertible_to<std::string>; }) {
            if (t.on == "mask") target.role = Role::Label;
            if (t.on == "surface") target.role = Role::Points;
            return {t.on, target};
        } else if constexpr (requires { t.on.front(); }) {
            target.role = Role::Label;
            return {t.on.empty() ? "mask" : t.on.front(), target};
        } else if constexpr (requires { t.options.on; }) {
            return {t.options.on.empty() ? "image" : t.options.on.front(), target};
        } else return {"image", target};
    }, op);
}
// The result of a bare-array call: the one field the op created, else the
// transformed input. Several created fields need a dict call.
inline std::string direct_output(const std::string& input, const std::vector<std::string>& fields) {
    std::string created;
    for (const auto& name : fields) if (name != input) {
        require(created.empty(), "transform creates several fields (" + created + ", " + name + "); call it with a dict");
        created = name;
    }
    return created.empty() ? input : created;
}
// One transform on a sample whose fields carry their targets.
inline Sample apply(const Op& op, Sample s, bool compact = true, std::string backend = "cpu") {
    std::map<std::string, Target> targets;
    for (const auto& [name, f] : s) targets[name] = f.target;
    return Pipeline::direct(op, std::move(targets), compact, std::move(backend)).run(std::move(s), direct_seed, false).sample;
}
// One transform on a channel-first (C, Y, X) / (C, Z, Y, X) array.
inline Array apply(const Op& op, Array a, bool compact = true, std::string backend = "cpu") {
    auto [key, target] = direct_input(op);
    Sample s; s.emplace(key, Field{std::move(a), target, key});
    auto result = apply(op, std::move(s), compact, std::move(backend));
    std::vector<std::string> fields; for (const auto& [name, f] : result) fields.push_back(name);
    return std::move(result.at(direct_output(key, fields)).array);
}
} // namespace vcr::transforms
