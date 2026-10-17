#include "replay.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace vcr::augment;
using Input = nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>;

// Constructor parsing only. Native execution never reads Python objects.
struct Arguments {
    nb::args positional;
    nb::dict keywords;
    std::size_t index = 0;
    Arguments(nb::args args, nb::kwargs kwargs) : positional(args), keywords(nb::dict(kwargs)) {}
    nb::object take(const char* name, nb::object fallback) {
        if (index < positional.size()) {
            require(!keywords.contains(name), std::string("duplicate argument: ") + name);
            return nb::borrow<nb::object>(positional[index++]);
        }
        return keyword(name, fallback);
    }
    nb::object keyword(const char* name, nb::object fallback) {
        if (!keywords.contains(name)) return fallback;
        auto result = nb::borrow<nb::object>(keywords[name]); nb::del(keywords[name]); return result;
    }
    template<class T> T get(const char* name, T fallback) { return nb::cast<T>(take(name, nb::cast(fallback))); }
    template<class T> T kw(const char* name, T fallback) { return nb::cast<T>(keyword(name, nb::cast(fallback))); }
    void finish() {
        require(index == positional.size(), "too many positional arguments");
        if (keywords.size()) throw nb::type_error(("unexpected arguments: " + nb::cast<std::string>(nb::repr(keywords))).c_str());
    }
};
static Scalar scalar(nb::handle value) {
    if (nb::isinstance<Uniform>(value)) return nb::cast<Uniform>(value);
    if (nb::isinstance<Normal>(value)) return nb::cast<Normal>(value);
    if (nb::isinstance<Choice>(value)) return nb::cast<Choice>(value);
    if (nb::isinstance<BalancedContrast>(value)) return nb::cast<BalancedContrast>(value);
    if (nb::isinstance<nb::tuple>(value) || nb::isinstance<nb::list>(value)) {
        auto values = nb::cast<std::vector<double>>(value); require(values.size() == 2, "range needs two bounds"); return Uniform{values[0], values[1]};
    }
    return nb::cast<double>(value);
}
static Scalar parameter(Arguments& a, const char* name, const Scalar& fallback) {
    return scalar(a.take(name, std::visit([](const auto& v) { return nb::cast(v); }, fallback)));
}
static std::vector<std::string> names(nb::handle value) {
    if (value.is_none()) return {};
    if (nb::isinstance<nb::str>(value)) return {nb::cast<std::string>(value)};
    return nb::cast<std::vector<std::string>>(value);
}
static ChannelOptions options(Arguments& a) {
    ChannelOptions o; o.p = a.kw("p", 1.); o.p_channel = a.kw("p_channel", 1.); o.shared = a.kw("shared", false);
    o.channels = a.kw("channels", std::vector<std::size_t>{}); o.on = names(a.keyword("on", nb::none()));
    probability(o.p); probability(o.p_channel); return o;
}
static LocalOptions local_options(Arguments& a) {
    LocalOptions o; o.scale = parameter(a, "scale", o.scale); o.loc = scalar(a.keyword("loc", nb::cast(Uniform{-1, 2}))); return o;
}
template<class T, class F> void bind(nb::module_& m, const char* name, F initialize, const char* doc = nullptr) {
    nb::class_<T>(m, name, doc).def("__init__", [initialize](T* self, nb::args args, nb::kwargs kwargs) {
        Arguments a(args, kwargs); T value; initialize(value, a);
        if constexpr (requires { value.options; }) value.options = options(a);
        else if constexpr (requires { value.p; }) { value.p = a.kw("p", 1.); probability(value.p); }
        a.finish(); new (self) T(std::move(value));
    });
}
static Morph morph(nb::handle object) {
    auto value = nb::cast<std::string>(object);
    if (value == "dilate") return Morph::Dilate;
    if (value == "erode") return Morph::Erode;
    if (value == "open") return Morph::Open;
    if (value == "close") return Morph::Close;
    throw nb::value_error("morphology operation must be dilate, erode, open or close");
}
static nb::object export_array(Array array, const std::map<std::string, Input>& inputs, bool inplace) {
    // A safe call returns independent outputs even when no transform wrote a
    // borrowed input. Explicit inplace calls can return views of reader buffers.
    if (array.borrowed && !inplace) array.data();
    nb::object input_owner = array.borrowed ? nb::cast(inputs.at(array.owner)) : nb::none();
    auto owner = std::make_unique<Array>(std::move(array)); auto* ptr = owner.get();
    nb::capsule capsule(owner.release(), [](void* p) noexcept { delete static_cast<Array*>(p); });
    nb::dlpack::dtype dtype;
    dtype.code = std::uint8_t(ptr->kind == 'f' ? nb::dlpack::dtype_code::Float : ptr->kind == 'i' ? nb::dlpack::dtype_code::Int : ptr->kind == 'b' ? nb::dlpack::dtype_code::Bool : nb::dlpack::dtype_code::UInt);
    dtype.bits = std::uint8_t(ptr->bytes * 8); dtype.lanes = 1;
    return nb::cast(nb::ndarray<nb::numpy>(const_cast<void*>(std::as_const(*ptr).data()), ptr->shape.size(), ptr->shape.data(), ptr->borrowed ? input_owner : nb::borrow<nb::object>(capsule), nullptr, dtype));
}
static void writable_inputs(const std::map<std::string, Input>& inputs) {
    for (auto it = inputs.begin(); it != inputs.end(); ++it) {
        nb::ndarray<nb::c_contig, nb::device::cpu> writable;
        require(nb::try_cast(nb::cast(it->second), writable, false), "inplace augmentation requires writable contiguous arrays");
        auto begin = reinterpret_cast<std::uintptr_t>(it->second.data());
        require(begin % std::max<std::size_t>(1, it->second.dtype().bits / 8) == 0, "inplace augmentation requires aligned arrays");
        auto end = begin + it->second.nbytes();
        for (auto other = inputs.begin(); other != it; ++other) {
            auto other_begin = reinterpret_cast<std::uintptr_t>(other->second.data());
            auto other_end = other_begin + other->second.nbytes();
            require(begin == end || other_begin == other_end || end <= other_begin || other_end <= begin,
                    "inplace augmentation requires nonoverlapping target arrays");
        }
    }
}
static Sample import_sample(const Pipeline& pipeline, const std::map<std::string, Input>& inputs, bool inplace = false) {
    Sample result;
    for (const auto& [name, input] : inputs) {
        require(pipeline.targets().count(name), "undeclared array field: " + name);
        Shape shape; for (std::size_t a = 0; a < input.ndim(); ++a) shape.push_back(input.shape(a));
        auto type = input.dtype(); char kind = 0;
        if (type.code == std::uint8_t(nb::dlpack::dtype_code::Float)) kind = 'f';
        if (type.code == std::uint8_t(nb::dlpack::dtype_code::Int)) kind = 'i';
        if (type.code == std::uint8_t(nb::dlpack::dtype_code::UInt)) kind = 'u';
        if (type.code == std::uint8_t(nb::dlpack::dtype_code::Bool)) kind = 'b';
        require(type.lanes == 1 && type.bits % 8 == 0, "unsupported input dtype");
        Array array(shape, kind, type.bits / 8, input.data(), name, inplace);
        if (reinterpret_cast<std::uintptr_t>(input.data()) % array.bytes) array.data();
        result.emplace(name, Field{std::move(array), pipeline.targets().at(name), name});
    }
    return result;
}
static nb::tuple export_result(RunResult result, bool plan, const std::map<std::string, Input>& inputs, bool inplace = false) {
    nb::dict arrays, sources;
    for (auto& [name, f] : result.sample) {
        if (f.pyramid.empty()) arrays[nb::str(name.c_str())] = export_array(std::move(f.array), inputs, inplace);
        else { nb::list levels; for (auto& level : f.pyramid) levels.append(export_array(std::move(level), inputs, inplace)); arrays[nb::str(name.c_str())] = levels; }
        sources[nb::str(name.c_str())] = nb::str(f.source.c_str());
    }
    return nb::make_tuple(arrays, sources, nb::cast(result.timings), plan ? nb::cast(std::move(result.plan)) : nb::none());
}

NB_MODULE(_augment, m) {
    m.doc() = "Native CPU augmentation: configuration, sampling, execution and replay.";
    nb::class_<Uniform>(m, "Uniform").def(nb::init<double, double>(), "low"_a, "high"_a);
    nb::class_<Normal>(m, "Normal").def(nb::init<double, double>(), "mean"_a = 0., "std"_a = .1);
    nb::class_<Choice>(m, "Choice").def(nb::init<std::vector<double>>(), "values"_a);
    nb::class_<BalancedContrast>(m, "BalancedContrast").def(nb::init<double, double>(), "low"_a = .75, "high"_a = 1.25);
    nb::class_<Target>(m, "Target");
    m.def("Image", [](double fill, nb::object valid) { Target t; t.fill = fill; t.valid = valid.is_none() ? "" : nb::cast<std::string>(valid); return t; }, "fill"_a = 0., "valid"_a.none() = nb::str("padding_mask"));
    m.def("Label", [](nb::object ignore_value, double fill) { Target t; t.role = Role::Label; t.fill = ignore_value.is_none() ? fill : nb::cast<double>(ignore_value); return t; }, "ignore_value"_a = nb::none(), "fill"_a = 0.);
    m.def("ScalarField", [](double fill) { Target t; t.role = Role::Scalar; t.fill = fill; return t; }, "fill"_a = 0.);
    m.def("ValidMask", [] { Target t; t.role = Role::Valid; return t; });
    m.def("Points", [](const std::string& order, const std::string& outside) { Target t; t.role = Role::Points; t.components = order; t.outside = outside; return t; }, "order"_a = "zyx", "outside"_a = "keep");
    m.def("Vectors", [](const std::string& components, bool normalize) { Target t; t.role = Role::Vectors; t.components = components; t.normalize = normalize; return t; }, "components"_a = "zyx", "normalize"_a = false);
    m.def("VectorField", [](const std::string& components, bool normalize, double fill) { Target t; t.role = Role::VectorField; t.components = components; t.normalize = normalize; t.fill = fill; return t; }, "components"_a = "zyx", "normalize"_a = false, "fill"_a = 0.);
    m.def("NormalField", [](const std::string& components, double fill) { Target t; t.role = Role::NormalField; t.components = components; t.fill = fill; return t; }, "components"_a = "zyx", "fill"_a = 0.);
    m.def("CoordinateField", [](const std::string& components, const std::string& frame, double fill) { require(frame == "external" || frame == "local", "frame must be external or local"); Target t; t.role = Role::CoordinateField; t.components = components; t.external = frame == "external"; t.fill = fill; return t; }, "components"_a = "zyx", "frame"_a = "external", "fill"_a = 0.);

    bind<BrightnessMultiply>(m, "BrightnessMultiply", [](auto& t, auto& a) { t.factor = parameter(a, "factor", t.factor); });
    bind<BrightnessAdd>(m, "BrightnessAdd", [](auto& t, auto& a) { t.shift = parameter(a, "shift", t.shift); });
    bind<Contrast>(m, "Contrast", [](auto& t, auto& a) { t.factor = parameter(a, "factor", t.factor); t.preserve_range = a.get("preserve_range", true); });
    bind<Gamma>(m, "Gamma", [](auto& t, auto& a) { t.gamma = parameter(a, "gamma", t.gamma); t.p_invert = a.get("p_invert", 0.); t.p_retain_stats = a.get("p_retain_stats", 0.); probability(t.p_invert); probability(t.p_retain_stats); });
    bind<Invert>(m, "Invert", [](auto&, auto&) {});
    bind<Normalize>(m, "Normalize", [](auto& t, auto& a) {
        const auto mode = a.get("mode", std::string("zscore"));
        require(mode == "minmax" || mode == "zscore", "normalization mode must be minmax or zscore");
        t.mode = mode == "minmax" ? Normalization::MinMax : Normalization::ZScore;
    });
    bind<GaussianNoise>(m, "GaussianNoise", [](auto& t, auto& a) { t.std = parameter(a, "std", t.std); t.shared_noise = a.get("shared_noise", false); });
    bind<RicianNoise>(m, "RicianNoise", [](auto& t, auto& a) { t.std = parameter(a, "std", t.std); t.shared_noise = a.get("shared_noise", false); });
    bind<ClipOutliers>(m, "ClipOutliers", [](auto& t, auto& a) { t.lower = parameter(a, "lower", t.lower); t.upper = parameter(a, "upper", t.upper); t.p_retain_std = a.get("p_retain_std", 1.); probability(t.p_retain_std); });
    bind<GaussianBlur>(m, "GaussianBlur", [](auto& t, auto& a) {
        t.sigma = parameter(a, "sigma", t.sigma); t.shared_axes = a.get("shared_axes", true); t.truncate = a.get("truncate", 6.);
        auto boundary = a.get("boundary", std::string("mirror")); require(boundary == "mirror" || boundary == "reflect", "boundary must be mirror or reflect"); t.mirror_boundary = boundary == "mirror";
    });
    bind<MedianFilter>(m, "MedianFilter", [](auto& t, auto& a) { t.size = parameter(a, "size", t.size); t.quantize = a.kw("quantize", false); });
    bind<Sharpen>(m, "Sharpen", [](auto& t, auto& a) { t.strength = parameter(a, "strength", t.strength); t.p_clamp = a.get("p_clamp", .5); probability(t.p_clamp); });
    bind<BrightnessGradient>(m, "BrightnessGradient", [](auto& t, auto& a) { t.local = local_options(a); t.strength = parameter(a, "strength", t.strength); t.mean_centered = a.get("mean_centered", true); t.clip = a.get("clip", false); });
    bind<LocalContrast>(m, "LocalContrast", [](auto& t, auto& a) { t.local = local_options(a); t.factor = parameter(a, "factor", t.factor); });
    bind<LocalGamma>(m, "LocalGamma", [](auto& t, auto& a) { t.local = local_options(a); t.gamma = parameter(a, "gamma", t.gamma); });
    bind<LocalSmoothing>(m, "LocalSmoothing", [](auto& t, auto& a) { t.local = local_options(a); t.strength = parameter(a, "strength", t.strength); t.sigma = parameter(a, "sigma", t.sigma); });
    bind<BlankRectangle>(m, "BlankRectangle", [](auto& t, auto& a) {
        auto size = a.take("size", nb::int_(16)); t.size.clear();
        if (nb::isinstance<nb::tuple>(size) || nb::isinstance<nb::list>(size)) { for (auto item : nb::borrow<nb::iterable>(size)) t.size.push_back(scalar(item)); } else t.size.push_back(scalar(size));
        auto value = a.take("value", nb::str("mean")); if (nb::isinstance<nb::str>(value)) t.statistic = nb::cast<std::string>(value); else { t.statistic.clear(); t.value = scalar(value); }
        t.count = parameter(a, "count", t.count); t.square = a.get("square", false);
    });
    bind<SliceIllumination>(m, "SliceIllumination", [](auto& t, auto& a) {
        auto ax = a.get("axis", std::string("z")); require(ax.size() == 1, "axis must be one character"); t.axis = ax[0];
        t.defects = parameter(a, "defects", t.defects); t.width = parameter(a, "width", t.width); t.reduction = parameter(a, "reduction", t.reduction); t.base_p = parameter(a, "base_p", t.base_p); t.base_reduction = parameter(a, "base_reduction", t.base_reduction);
    });
    bind<Smear>(m, "Smear", [](auto& t, auto& a) { auto ax = a.get("axis", std::string("z")); require(ax.size() == 1, "axis must be one character"); t.axis = ax[0]; t.shift = a.get("shift", t.shift); t.alpha = a.get("alpha", t.alpha); t.previous = a.get("previous", t.previous); });
    bind<Flip>(m, "Flip", [](auto& t, auto& a) { t.axes = a.get("axes", t.axes); t.p_axis = a.get("p_axis", t.p_axis); probability(t.p_axis); });
    bind<Rotate90>(m, "Rotate90", [](auto& t, auto& a) { t.planes = a.get("planes", t.planes); auto turns = a.take("turns", nb::cast(t.turns)); t.turns = nb::isinstance<Choice>(turns) ? nb::cast<Choice>(turns) : Choice{nb::cast<std::vector<double>>(turns)}; t.resize = a.get("resize", false); });
    bind<Transpose>(m, "Transpose", [](auto& t, auto& a) { t.order = a.get("order", t.order); });
    bind<CropPad>(m, "CropPad", [](auto& t, auto& a) { t.shape = a.get("shape", t.shape); t.random = a.get("random", false); });
    bind<Affine>(m, "Affine", [](auto& t, auto& a) { t.degrees = parameter(a, "degrees", t.degrees); t.scale = parameter(a, "scale", t.scale); t.translate = parameter(a, "translate", t.translate); t.shear = parameter(a, "shear", t.shear); t.rotation_axes = a.get("rotation_axes", t.rotation_axes); t.shared_scale = a.get("shared_scale", true); t.border = a.get("border", false); t.shape = a.get("shape", t.shape); });
    bind<Elastic>(m, "Elastic", [](auto& t, auto& a) { t.sigma = parameter(a, "sigma", t.sigma); t.magnitude = parameter(a, "magnitude", t.magnitude); t.border = a.get("border", false); });
    bind<SheetCompression>(m, "SheetCompression", [](auto& t, auto& a) { t.strength = parameter(a, "strength", t.strength); t.intensity_sigma = parameter(a, "intensity_sigma", t.intensity_sigma); t.smoothing = parameter(a, "smoothing", t.smoothing); t.axes = a.get("axes", t.axes); t.image = a.get("image", t.image); t.threshold = a.get("threshold", t.threshold); });
    bind<LowResolution>(m, "LowResolution", [](auto& t, auto& a) { t.scale = parameter(a, "scale", t.scale); t.axes = a.get("axes", t.axes); t.shared_axes = a.get("shared_axes", true); });
    bind<ThickSlice>(m, "ThickSlice", [](auto& t, auto& a) { t.scale = parameter(a, "scale", t.scale); t.axes = a.get("axes", t.axes); t.labels = a.get("labels", true); });
    bind<Morphology>(m, "Morphology", [](auto& t, auto& a) { t.operation = morph(a.take("operation", nb::str("close"))); t.radius = parameter(a, "radius", t.radius); t.on = names(a.take("on", nb::none())); t.neutral_border = a.get("neutral_border", false); auto footprint = a.get("footprint", std::string("ball")); require(footprint == "ball" || footprint == "repeated_cross", "footprint must be ball or repeated_cross"); t.repeated_cross = footprint == "repeated_cross"; });
    bind<RandomBinaryOperator>(m, "RandomBinaryOperator", [](auto& t, auto& a) { t.on = a.get("on", t.on); t.channels = a.get("channels", t.channels); t.radius = parameter(a, "radius", t.radius); auto ops = a.take("operations", nb::none()); if (!ops.is_none()) { t.operations.clear(); for (auto op : nb::borrow<nb::iterable>(ops)) t.operations.push_back(morph(op)); } t.p_channel = a.get("p_channel", 1.); });
    bind<RemoveComponents>(m, "RemoveComponents", [](auto& t, auto& a) { t.on = a.get("on", t.on); t.channels = a.get("channels", t.channels); t.max_fraction = a.get("max_fraction", t.max_fraction); t.p_fill = a.get("p_fill", t.p_fill); t.p_channel = a.get("p_channel", t.p_channel); t.full_connectivity = a.get("full_connectivity", true); });
    bind<ReplaceLabel>(m, "ReplaceLabel", [](auto& t, auto& a) { t.value = a.get("value", t.value); t.replacement = a.get("replacement", t.replacement); t.on = a.get("on", t.on); t.channels = a.get("channels", t.channels); });
    bind<Regions>(m, "Regions", [](auto& t, auto& a) { auto labels = a.take("labels", nb::tuple()); for (auto group : nb::borrow<nb::iterable>(labels)) { if (nb::isinstance<nb::list>(group) || nb::isinstance<nb::tuple>(group)) t.labels.push_back(nb::cast<std::vector<double>>(group)); else t.labels.push_back({nb::cast<double>(group)}); } t.on = a.get("on", t.on); t.output = a.get("output", t.output); t.channel = a.get("channel", t.channel); });
    bind<OneHot>(m, "OneHot", [](auto& t, auto& a) { t.labels = a.get("labels", t.labels); t.on = a.get("on", t.on); t.output = a.get("output", t.output); t.append_to = a.get("append_to", t.append_to); t.channel = a.get("channel", t.channel); t.remove_source = a.get("remove_source", false); });
    bind<MaskImage>(m, "MaskImage", [](auto& t, auto& a) { t.mask = a.get("mask", t.mask); t.on = a.get("on", t.on); t.mask_channel = a.get("mask_channel", t.mask_channel); t.channels = a.get("channels", t.channels); t.fill = a.get("fill", t.fill); });
    bind<DeepSupervision>(m, "DeepSupervision", [](auto& t, auto& a) { auto scales = a.take("scales", nb::tuple()); for (auto v : nb::borrow<nb::iterable>(scales)) { if (nb::isinstance<nb::tuple>(v) || nb::isinstance<nb::list>(v)) t.scales.push_back(nb::cast<std::vector<double>>(v)); else t.scales.push_back({nb::cast<double>(v)}); } t.on = a.get("on", t.on); t.output = a.get("output", t.output); });
    bind<ToPseudo2D>(m, "ToPseudo2D", [](auto&, auto&) {});
    bind<FromPseudo2D>(m, "FromPseudo2D", [](auto&, auto&) {});
    bind<MedialSurface>(m, "MedialSurface", [](auto& t, auto& a) { t.on = names(a.take("on", nb::none())); t.suffix = a.get("suffix", t.suffix); t.tube = a.get("tube", true); t.open = a.get("open", false); t.close = a.get("close", true); t.ignore = a.get("ignore", t.ignore); });
    bind<StripGeometry>(m, "StripGeometry", [](auto& t, auto& a) {
        t.shift_x = parameter(a, "shift_x", t.shift_x); t.shift_y = parameter(a, "shift_y", t.shift_y); t.degrees = parameter(a, "degrees", t.degrees); t.scale = parameter(a, "scale", t.scale);
        t.shear_x = parameter(a, "shear_x", t.shear_x); t.shear_y = parameter(a, "shear_y", t.shear_y); t.smooth_offset = parameter(a, "smooth_offset", t.smooth_offset); t.stride = parameter(a, "stride", t.stride);
        t.p_flip_x = a.get("p_flip_x", 0.); t.p_flip_y = a.get("p_flip_y", 0.); t.shape = a.get("shape", t.shape);
    });
    bind<SurfacePerturb>(m, "SurfacePerturb", [](auto& t, auto& a) {
        t.on = a.get("on", t.on); t.blobs = parameter(a, "blobs", t.blobs); t.sigma_fraction = parameter(a, "sigma_fraction", t.sigma_fraction); t.amplitude = parameter(a, "amplitude", t.amplitude);
        t.points_affected = a.get("points_affected", t.points_affected); t.radius_sigma = a.get("radius_sigma", t.radius_sigma); t.max_displacement = a.get("max_displacement", t.max_displacement);
    });
    bind<OneOf>(m, "OneOf", [](auto& t, auto& a) { t.transforms = a.get("transforms", t.transforms); t.weights = a.get("weights", t.weights); });

    nb::class_<Plan>(m, "Plan")
        .def_prop_ro("steps", [](const Plan& p) { return p.steps.size(); })
        .def("dump", [](const Plan& p) { auto bytes = save_plan(p); return nb::bytes(bytes.data(), bytes.size()); });
    m.def("cuda_available", &cuda::available);
    nb::class_<Pipeline>(m, "Pipeline")
        .def(nb::init<std::vector<Op>, std::map<std::string, Target>, bool, std::string>(), "transforms"_a, "targets"_a, "compact"_a = true, "backend"_a = "cpu")
        .def_prop_ro("backend", &Pipeline::backend)
        .def("load_plan", [](const Pipeline& p, nb::bytes bytes) { return load_plan(p.program(), std::string(bytes.c_str(), bytes.size())); })
        .def("run", [](const Pipeline& p, const std::map<std::string, Input>& inputs, std::uint64_t seed, bool record, bool profile, bool inplace) {
            if (inplace) writable_inputs(inputs);
            RunResult result;
            { nb::gil_scoped_release release; result = p.run(import_sample(p, inputs, inplace), seed, record, profile); }
            return export_result(std::move(result), record, inputs, inplace);
        }, "inputs"_a, "seed"_a, "record"_a = false, "profile"_a = false, "inplace"_a = false)
        .def("replay", [](const Pipeline& p, const std::map<std::string, Input>& inputs, const Plan& plan) {
            RunResult result; { nb::gil_scoped_release release; result = p.replay(import_sample(p, inputs), plan); }
            return export_result(std::move(result), false, inputs);
        });
    m.def("seed", [](nb::args parts) {
        std::uint64_t value = UINT64_C(14695981039346656037);
        for (auto part : parts) {
            require(nb::isinstance<nb::int_>(part) || nb::isinstance<nb::str>(part), "seed identifiers must be integers or strings");
            auto text = nb::cast<std::string>(nb::str(part));
            value = split_seed(value, (nb::isinstance<nb::str>(part) ? "s" : "i") + std::to_string(text.size()) + ":" + text);
        }
        return value;
    });
    m.def("translate_crop_box", &translate_crop_box, "box"_a, "points"_a, "seed"_a,
          "p"_a = .3, "minimum"_a = 10, "maximum"_a = 40, "keep"_a = 1. / 3, "attempts"_a = 24,
          nb::call_guard<nb::gil_scoped_release>());
}
