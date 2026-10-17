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
    ToPseudo2D, FromPseudo2D, MedialSurface, StripGeometry, SurfacePerturb, std::shared_ptr<OneOf>>;
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
inline void prepare(const Op& op, Sample& s, Random& r, Shape path, std::vector<Step>* record, CompactState& compact) {
    if (!r.chance(chance(op))) return;
    if (const auto* choice = std::get_if<std::shared_ptr<OneOf>>(&op)) {
        const auto& t = **choice; require(!t.transforms.empty(), "OneOf needs transforms");
        auto weights = t.weights; if (weights.empty()) weights.assign(t.transforms.size(), 1);
        require(weights.size() == t.transforms.size(), "OneOf weight count mismatch"); double total = 0;
        for (auto w : weights) { require(std::isfinite(w) && w >= 0, "invalid OneOf weight"); total += w; }
        require(std::isfinite(total) && total > 0, "OneOf needs positive total weight");
        double threshold = r.uniform() * total; std::size_t chosen = 0;
        for (; chosen + 1 < weights.size() && threshold >= weights[chosen]; ++chosen) threshold -= weights[chosen];
        path.push_back(chosen); prepare(t.transforms[chosen], s, r, std::move(path), record, compact); return;
    }
    std::visit([&](const auto& t) { compact.before(t, s); }, op);
    auto parameters = std::visit([&](const auto& t) -> Parameters {
        if constexpr (!std::is_same_v<std::decay_t<decltype(t)>, std::shared_ptr<OneOf>>) return sample(t, s, r);
        else return {};
    }, op);
    std::visit([&](const auto& t) {
        if constexpr (!std::is_same_v<std::decay_t<decltype(t)>, std::shared_ptr<OneOf>>) compact.apply_step(t, s, parameters);
    }, op);
    if (record) record->push_back({std::move(path), std::move(parameters)});
}
class Pipeline {
    std::shared_ptr<const Program> program_;
    bool compact_;
public:
    Pipeline(std::vector<Op> transforms, std::map<std::string, Target> targets, bool compact = true)
        : program_(std::make_shared<Program>(Program{std::move(transforms), std::move(targets)})), compact_(compact) {
        require(!program_->targets.empty(), "target schema must not be empty");
        for (const auto& op : program_->transforms) probability(chance(op));
    }
    const auto& targets() const { return program_->targets; }
    const auto& program() const { return program_; }
    RunResult run(Sample s, std::uint64_t seed, bool record, bool profile) const {
        validate(s); RunResult result; result.plan.program = program_; result.plan.inputs = input_specs(s);
        CompactState compact(compact_);
        for (std::size_t i = 0; i < program_->transforms.size(); ++i) {
            for (const auto& [name, f] : s) require(f.pyramid.empty(), "DeepSupervision must be the last transform");
            auto start = std::chrono::steady_clock::now(); Random r{split_seed(seed, std::to_string(i))};
            prepare(program_->transforms[i], s, r, {i}, record ? &result.plan.steps : nullptr, compact);
            if (profile) result.timings[std::to_string(i)] = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
        auto start = std::chrono::steady_clock::now(); compact.flush(s);
        if (profile && !program_->transforms.empty()) result.timings[std::to_string(program_->transforms.size() - 1)] +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        result.sample = std::move(s); return result;
    }
    RunResult replay(Sample s, const Plan& plan) const {
        require(plan.program == program_, "plan belongs to another pipeline; use plan.apply(sample)");
        validate(s); require(input_specs(s) == plan.inputs, "replay requires identical target names, shapes and dtypes");
        CompactState compact(compact_);
        for (const auto& step : plan.steps) {
            std::visit([&](const auto& t) {
                if constexpr (!std::is_same_v<std::decay_t<decltype(t)>, std::shared_ptr<OneOf>>) {
                    compact.before(t, s); compact.apply_step(t, s, step.parameters);
                } else throw std::invalid_argument("replay plan contains unresolved OneOf");
            }, resolve(*program_, step.path));
        }
        compact.flush(s);
        RunResult result; result.sample = std::move(s); return result;
    }
};
} // namespace vcr::augment
