#pragma once
#include "tracing.hpp"
#include "palette.hpp"
#include <bit>
#include <chrono>

namespace vcr::augment {
struct OneOf;
using Op = std::variant<
    BrightnessMultiply, BrightnessAdd, Contrast, Gamma, Invert, GaussianNoise,
    RicianNoise, ClipOutliers, GaussianBlur, MedianFilter, Sharpen,
    BrightnessGradient, LocalContrast, LocalGamma, LocalSmoothing, BlankRectangle,
    SliceIllumination, Smear, Flip, Rotate90, Transpose, CropPad, Affine, Elastic,
    SheetCompression, LowResolution, ThickSlice, Morphology, RandomBinaryOperator,
    RemoveComponents, ReplaceLabel, Regions, OneHot, MaskImage, DeepSupervision,
    ToPseudo2D, FromPseudo2D, MedialSurface, StripGeometry, SurfacePerturb, std::shared_ptr<OneOf>, Normalize, OrientedBlur>;
using Executor = void (*)(const Op&, Sample&, const Parameters&, void*);
#ifdef VC_READER_CUDA
namespace cuda {
bool available();
void execute(const Op&, Sample&, const Parameters&, void*);
std::shared_ptr<void> begin(const Sample&);
void finish(Sample&, void*);
void finish_device(void*);
void synchronize();
Parameters sample_parameters(const Op&, const Sample&, Random&, bool, void*);
void validate_program(const std::vector<Op>&);
}
#else
namespace cuda { inline bool available() { return false; } }
#endif
struct OneOf { std::vector<Op> transforms; std::vector<double> weights; double p = 1; };
struct Program { std::vector<Op> transforms; std::map<std::string, Target> targets; };
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
struct RunResult { Sample sample; Plan plan; std::map<std::string, double> timings; };
inline std::map<std::string, InputSpec> input_specs(const Sample& s) {
    std::map<std::string, InputSpec> result;
    for (const auto& [name, f] : s) result[name] = {f.array.shape, f.array.kind, f.array.bytes};
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
            require(cuda::available(), "CUDA augmentation is unavailable: build with VC_READER_ENABLE_CUDA=ON and use a CUDA device");
#ifdef VC_READER_CUDA
            cuda::validate_program(program_->transforms);
            executor_ = cuda::execute; backend_ = "cuda"; compact_ = false;
#endif
        }
        require(!program_->targets.empty(), "target schema must not be empty");
        for (const auto& op : program_->transforms) probability(chance(op));
    }
    const auto& backend() const { return backend_; }
    const auto& targets() const { return program_->targets; }
    const auto& program() const { return program_; }
    RunResult run(Sample s, std::uint64_t seed, bool record, bool profile, std::shared_ptr<void> context = {}) const {
        const bool resident = bool(context);
        require(!resident || executor_, "device inputs require a CUDA pipeline");
        validate(s, !resident); RunResult result; result.plan.program = program_; result.plan.inputs = input_specs(s);
        CompactState compact(compact_);
        const auto upload_start = std::chrono::steady_clock::now();
#ifdef VC_READER_CUDA
        if (executor_ && !resident) context = cuda::begin(s);
#endif
        const double upload_time = std::chrono::duration<double>(std::chrono::steady_clock::now()-upload_start).count();
        for (std::size_t i = 0; i < program_->transforms.size(); ++i) {
            for (const auto& [name, f] : s) require(f.pyramid.empty(), "DeepSupervision must be the last transform");
            auto start = std::chrono::steady_clock::now(); Random r{split_seed(seed, std::to_string(i))};
            prepare(program_->transforms[i], s, r, {i}, record ? &result.plan.steps : nullptr, compact, executor_, context.get());
#ifdef VC_READER_CUDA
            if (executor_ && profile) cuda::synchronize();
#endif
            if (profile) result.timings[std::to_string(i)] = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                + (executor_ && i == 0 ? upload_time : 0);
        }
        auto start = std::chrono::steady_clock::now();
#ifdef VC_READER_CUDA
        if (executor_) { if (resident) cuda::finish_device(context.get()); else cuda::finish(s,context.get()); }
#endif
        compact.flush(s);
        if (profile && !program_->transforms.empty()) result.timings[std::to_string(program_->transforms.size() - 1)] +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        result.sample = std::move(s); return result;
    }
    RunResult replay(Sample s, const Plan& plan, std::shared_ptr<void> context = {}) const {
        require(plan.program == program_, "plan belongs to another pipeline; use plan.apply(sample)");
        const bool resident = bool(context);
        require(!resident || executor_, "device inputs require a CUDA pipeline");
        validate(s, !resident); require(input_specs(s) == plan.inputs, "replay requires identical target names, shapes and dtypes");
        CompactState compact(compact_);
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
        compact.flush(s);
        RunResult result; result.sample = std::move(s); return result;
    }
};
} // namespace vcr::augment
