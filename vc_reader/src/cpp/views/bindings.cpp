#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <vc_reader/views.hpp>
#include <memory>
#ifdef VC_READER_CUDA
#include "../cuda_buffer.hpp"
#endif
namespace nb = nanobind;
using namespace nb::literals;
using namespace vcr::views;

namespace {
using Array = nb::ndarray<nb::c_contig>;
template <class T> using Host = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;
using Code = nb::dlpack::dtype_code;

int deviceOf(const Array& a)
{
    if (a.device_type() == nb::device::cpu::value) return -1;
    if (a.device_type() == nb::device::cuda::value) return a.device_id();
    throw nb::type_error("arrays must be on the CPU or a CUDA device");
}
int sameDevice(std::initializer_list<const Array*> arrays)
{
    const int id = deviceOf(**arrays.begin());
    for (const auto* a : arrays)
        if (deviceOf(*a) != id) throw nb::value_error("arrays must be on the same device");
    return id;
}
bool isCode(const Array& a, Code code, int bits) { return a.dtype().code == std::uint8_t(code) && a.dtype().bits == bits && a.dtype().lanes == 1; }
bool isByteMask(const Array& a) { return isCode(a, Code::UInt, 8) || isCode(a, Code::Bool, 8) || isCode(a, Code::Int, 8); }
std::vector<std::int64_t> shapeOf(const Array& a)
{
    std::vector<std::int64_t> s;
    for (std::size_t i = 0; i < a.ndim(); ++i) s.push_back(std::int64_t(a.shape(i)));
    return s;
}
void expectShape(const Array& a, const std::vector<std::int64_t>& shape, const char* name)
{
    if (shapeOf(a) != shape) throw nb::value_error((std::string(name) + " has the wrong shape").c_str());
}
template <class F, class G> void dispatch(int device, std::uintptr_t stream, F&& cpu, G&& gpu)
{
    if (device < 0) {
        nb::gil_scoped_release release;
        cpu();
        return;
    }
#ifdef VC_READER_CUDA
    nb::gil_scoped_release release;
    vcr::device::Guard guard(device);
    vcr::device::StreamScope scope(vcr::device::stream_handle(stream));
    gpu();
#else
    (void)stream; (void)gpu;
    throw std::runtime_error("CUDA support is not built");
#endif
}
template <class T>
nb::ndarray<nb::numpy, T> numpy(std::vector<T> values, std::vector<std::size_t> dims)
{
    auto* owned = new std::vector<T>(std::move(values));
    nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return nb::ndarray<nb::numpy, T>(owned->data(), dims.size(), dims.data(), owner);
}
nb::ndarray<nb::numpy, bool> numpyBool(std::vector<std::uint8_t> values, std::vector<std::size_t> dims)
{
    auto* owned = new std::vector<std::uint8_t>(std::move(values));
    nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<std::uint8_t>*>(p); });
    return nb::ndarray<nb::numpy, bool>(reinterpret_cast<bool*>(owned->data()), dims.size(), dims.data(), owner);
}
Index3 index3(const std::array<std::int64_t, 3>& a) { return {a[0], a[1], a[2]}; }

void tokenSupportInto(const Array& valid, const std::array<std::int64_t, 3>& patch, const Array& out, std::uintptr_t stream)
{
    if (valid.ndim() != 3 && valid.ndim() != 4) throw nb::value_error("valid must be (D,H,W) or (B,D,H,W)");
    if (!isByteMask(valid)) throw nb::type_error("valid must be bool or uint8");
    if (!isCode(out, Code::UInt, 8)) throw nb::type_error("out must be uint8");
    const int id = sameDevice({&valid, &out});
    const auto lead = valid.ndim() - 3;
    const std::size_t batch = lead ? valid.shape(0) : 1;
    const Index3 shape{std::int64_t(valid.shape(lead)), std::int64_t(valid.shape(lead + 1)), std::int64_t(valid.shape(lead + 2))};
    const Index3 p = index3(patch);
    std::vector<std::int64_t> expected;
    if (lead) expected.push_back(std::int64_t(batch));
    for (int a = 0; a < 3; ++a) {
        if (p[a] <= 0 || shape[a] % p[a]) throw nb::value_error("valid shape must be a multiple of patch on every axis");
        expected.push_back(shape[a] / p[a]);
    }
    expectShape(out, expected, "out");
    const auto* in = static_cast<const std::uint8_t*>(valid.data());
    auto* support = static_cast<std::uint8_t*>(out.data());
    dispatch(id, stream, [&] { tokenSupport(in, batch, shape, p, support); },
             [&] {
#ifdef VC_READER_CUDA
                 cuda::tokenSupport(in, batch, shape, p, support);
#endif
             });
}

void correspondenceInto(const Array& support, Host<std::int64_t> supportIndex, Host<std::int64_t> studentOrigins,
                        const std::array<std::int64_t, 3>& studentShape, Host<std::int64_t> studentOrientations,
                        Host<std::int64_t> teacherOrigins, const std::array<std::int64_t, 3>& teacherShape,
                        Host<std::int64_t> teacherOrientations, const Array& teacherIndex, const Array& studentValid,
                        std::optional<Array> lattice, std::optional<Host<std::int64_t>> studentBoxes,
                        const std::array<std::int64_t, 3>& patch, std::uintptr_t stream)
{
    if (support.ndim() != 4) throw nb::value_error("support must be (grids,T0,T1,T2)");
    if (!isByteMask(support)) throw nb::type_error("support must be bool or uint8");
    if (supportIndex.ndim() != 1) throw nb::value_error("support_index must be (N,)");
    const auto n = std::int64_t(supportIndex.shape(0));
    const auto pairsShape = [&](const Host<std::int64_t>& a, std::vector<std::int64_t> shape, const char* name) {
        std::vector<std::int64_t> s;
        for (std::size_t i = 0; i < a.ndim(); ++i) s.push_back(std::int64_t(a.shape(i)));
        if (s != shape) throw nb::value_error((std::string(name) + " has the wrong shape").c_str());
    };
    pairsShape(studentOrigins, {n, 3}, "student_origins");
    pairsShape(teacherOrigins, {n, 3}, "teacher_origins");
    pairsShape(studentOrientations, {n}, "student_orientations");
    pairsShape(teacherOrientations, {n}, "teacher_orientations");
    if (studentBoxes) pairsShape(*studentBoxes, {n, 6}, "student_boxes");
    const auto tokens = studentShape[0] * studentShape[1] * studentShape[2];
    if (!isCode(teacherIndex, Code::Int, 32)) throw nb::type_error("teacher_index must be int32");
    if (!isCode(studentValid, Code::UInt, 8)) throw nb::type_error("student_valid must be uint8");
    expectShape(teacherIndex, {n, tokens}, "teacher_index");
    expectShape(studentValid, {n, tokens}, "student_valid");
    const int id = sameDevice({&support, &teacherIndex, &studentValid});
    std::int32_t* lat = nullptr;
    if (lattice) {
        if (!isCode(*lattice, Code::Int, 32)) throw nb::type_error("lattice must be int32");
        expectShape(*lattice, {n, tokens, 3}, "lattice");
        if (deviceOf(*lattice) != id) throw nb::value_error("arrays must be on the same device");
        lat = static_cast<std::int32_t*>(lattice->data());
    }
    const Supports grids{static_cast<const std::uint8_t*>(support.data()), support.shape(0),
                         {std::int64_t(support.shape(1)), std::int64_t(support.shape(2)), std::int64_t(support.shape(3))}};
    const Pairs pairs{std::size_t(n), index3(studentShape), index3(teacherShape), supportIndex.data(),
                      studentOrigins.data(), studentOrientations.data(), teacherOrigins.data(), teacherOrientations.data(),
                      studentBoxes ? studentBoxes->data() : nullptr, index3(patch)};
    auto* index = static_cast<std::int32_t*>(teacherIndex.data());
    auto* valid = static_cast<std::uint8_t*>(studentValid.data());
    dispatch(id, stream, [&] { correspondence(grids, pairs, index, valid, lat); },
             [&] {
#ifdef VC_READER_CUDA
                 cuda::correspondence(grids, pairs, index, valid, lat);
#endif
             });
}

void maskPatchCellsInto(const Array& images, const Array& masks, const std::array<std::int64_t, 3>& patch, float fill,
                        const Array& out, std::uintptr_t stream)
{
    if (images.ndim() != 5) throw nb::value_error("images must be (B,C,D,H,W)");
    Real type;
    if (isCode(images, Code::Float, 32)) type = Real::Float32;
    else if (isCode(images, Code::Float, 16)) type = Real::Float16;
    else if (isCode(images, Code::Bfloat, 16)) type = Real::BFloat16;
    else throw nb::type_error("images must be float32, float16 or bfloat16");
    if (out.dtype() != images.dtype()) throw nb::type_error("out must have the images' dtype");
    if (!isByteMask(masks)) throw nb::type_error("masks must be bool or uint8");
    const int id = sameDevice({&images, &masks, &out});
    expectShape(out, shapeOf(images), "out");
    CellImages c{images.data(), out.data(), type, images.shape(0), images.shape(1),
                 {std::int64_t(images.shape(2)), std::int64_t(images.shape(3)), std::int64_t(images.shape(4))}, index3(patch)};
    std::vector<std::int64_t> expected{std::int64_t(c.batch)};
    for (int a = 0; a < 3; ++a) {
        if (c.patch[a] <= 0 || c.shape[a] % c.patch[a]) throw nb::value_error("image shape must be a multiple of patch on every axis");
        expected.push_back(c.shape[a] / c.patch[a]);
    }
    expectShape(masks, expected, "masks");
    const auto* m = static_cast<const std::uint8_t*>(masks.data());
    dispatch(id, stream, [&] { maskPatchCells(c, m, fill); },
             [&] {
#ifdef VC_READER_CUDA
                 cuda::maskPatchCells(c, m, fill);
#endif
             });
}
}  // namespace

NB_MODULE(_views, m)
{
    m.attr("ORIENTATION_COUNT") = ORIENTATION_COUNT;
    m.def("cuda_available", [] {
#ifdef VC_READER_CUDA
        return true;
#else
        return false;
#endif
    });
    m.def("orientation", [](int id) {
        const auto o = orientation(id);
        return std::make_tuple(std::make_tuple(o.perm[0], o.perm[1], o.perm[2]),
                               std::make_tuple(o.flips[0], o.flips[1], o.flips[2]));
    }, "id"_a);
    m.def("orientation_id", [](const std::array<int, 3>& perm, const std::array<bool, 3>& flips) {
        return orientationId({perm, flips});
    }, "perm"_a, "flips"_a);
    m.def("_draw_orientations", [](Host<std::uint64_t> seeds) {
        if (seeds.ndim() != 1) throw nb::value_error("seeds must be (B,)");
        std::vector<std::int32_t> ids(seeds.shape(0));
        drawOrientations(seeds.data(), ids.size(), ids.data());
        return numpy(std::move(ids), {seeds.shape(0)});
    }, "seeds"_a);
    m.def("_token_support", &tokenSupportInto, "valid"_a, "patch"_a, "out"_a.noconvert(), "stream"_a);
    m.def("_plan_view_sets", [](Host<std::uint8_t> support, const std::array<std::int64_t, 3>& teacher,
                                const std::vector<std::array<std::int64_t, 3>>& students, std::int64_t sets,
                                double fraction, std::int64_t count, std::int64_t studentAttempts,
                                std::int64_t teacherAttempts, std::uint64_t seed) {
        if (support.ndim() != 3) throw nb::value_error("support must be (T0,T1,T2)");
        if (sets < 0) throw nb::value_error("n_sets must be nonnegative");
        std::vector<Index3> shapes;
        for (const auto& s : students) shapes.push_back(index3(s));
        const Index3 tokens{std::int64_t(support.shape(0)), std::int64_t(support.shape(1)), std::int64_t(support.shape(2))};
        ViewSets v;
        {
            nb::gil_scoped_release release;
            v = planViewSets(support.data(), tokens, index3(teacher), shapes,
                             {std::size_t(sets), fraction, count, studentAttempts, teacherAttempts, seed});
        }
        const auto n = std::size_t(sets), S = shapes.size();
        nb::dict d;
        d["ok"] = v.ok;
        d["set_ok"] = numpyBool(std::move(v.setOk), {n});
        d["teacher_origins"] = numpy(std::move(v.teacherOrigins), {n, 3});
        d["student_origins"] = numpy(std::move(v.studentOrigins), {n, S, 3});
        d["student_overlap"] = numpy(std::move(v.studentOverlap), {n, S});
        d["student_attempts"] = numpy(std::move(v.studentAttempts), {n, S});
        d["teacher_attempts"] = numpy(std::move(v.teacherAttempts), {n});
        d["rejections"] = numpy(std::move(v.rejections), {n});
        return d;
    }, "support"_a, "teacher_shape"_a, "student_shapes"_a, "n_sets"_a, "min_overlap_fraction"_a,
       "min_overlap_count"_a, "max_student_attempts"_a, "max_teacher_attempts"_a, "seed"_a);
    m.def("_correspondence", &correspondenceInto, "support"_a, "support_index"_a, "student_origins"_a,
          "student_shape"_a, "student_orientations"_a, "teacher_origins"_a, "teacher_shape"_a,
          "teacher_orientations"_a, "teacher_index"_a.noconvert(), "student_valid"_a.noconvert(),
          "lattice"_a.noconvert().none(), "student_boxes"_a.none(), "patch"_a, "stream"_a);
    m.def("_patch_masks", [](const std::array<std::int64_t, 3>& shape, Host<std::uint64_t> seeds, const std::string& mode,
                             double probability, double fraction, double minBlock, std::optional<double> maxBlock,
                             double minAspect, std::optional<double> maxAspect, std::int64_t attempts, bool shift) {
        if (seeds.ndim() != 1) throw nb::value_error("seeds must be (B,)");
        MaskOptions o;
        if (mode == "random") o.mode = MaskMode::Random;
        else if (mode == "multiblock") o.mode = MaskMode::Multiblock;
        else throw nb::value_error("mode must be random or multiblock");
        o.probability = probability; o.fraction = fraction; o.minBlockTokens = minBlock; o.maxBlockTokens = maxBlock;
        o.minAspect = minAspect; o.maxAspect = maxAspect; o.attempts = attempts; o.cyclicShift = shift;
        const auto b = seeds.shape(0);
        for (auto s : shape) if (s <= 0) throw nb::value_error("mask shape must be positive");
        std::vector<std::uint8_t> masks(b * std::size_t(shape[0] * shape[1] * shape[2])), applied(b);
        std::vector<std::int64_t> counts(b);
        {
            nb::gil_scoped_release release;
            std::unique_ptr<bool[]> on(new bool[b ? b : 1]);
            patchMasks(index3(shape), seeds.data(), b, o, masks.data(), on.get(), counts.data());
            for (std::size_t i = 0; i < b; ++i) applied[i] = on[i];
        }
        nb::dict d;
        d["masks"] = numpy(std::move(masks), {b, std::size_t(shape[0]), std::size_t(shape[1]), std::size_t(shape[2])});
        d["applied"] = numpyBool(std::move(applied), {b});
        d["counts"] = numpy(std::move(counts), {b});
        return d;
    }, "shape"_a, "seeds"_a, "mode"_a, "probability"_a, "fraction"_a, "min_block_tokens"_a, "max_block_tokens"_a.none(),
       "min_aspect"_a, "max_aspect"_a.none(), "attempts"_a, "cyclic_shift"_a);
    m.def("_mask_patch_cells", &maskPatchCellsInto, "images"_a, "masks"_a, "patch"_a, "fill"_a, "out"_a.noconvert(), "stream"_a);
}
