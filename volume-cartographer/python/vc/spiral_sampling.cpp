#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nb = nanobind;

namespace {

using BoolMatrix = nb::ndarray<nb::numpy, const bool, nb::ndim<2>, nb::c_contig>;
using Int64Vector = nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig>;
using FloatVector = nb::ndarray<nb::numpy, const float, nb::ndim<1>, nb::c_contig>;
using Int32Pairs = nb::ndarray<nb::numpy, const int32_t, nb::shape<-1, 2>, nb::c_contig>;

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> own_1d(std::vector<T>&& values)
{
    auto* held = new std::vector<T>(std::move(values));
    nb::capsule owner(held, [](void* pointer) noexcept {
        delete static_cast<std::vector<T>*>(pointer);
    });
    return nb::ndarray<nb::numpy, T, nb::ndim<1>>(
        held->data(), {held->size()}, owner);
}

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<2>> own_2d(
    std::vector<T>&& values, size_t rows, size_t columns)
{
    auto* held = new std::vector<T>(std::move(values));
    nb::capsule owner(held, [](void* pointer) noexcept {
        delete static_cast<std::vector<T>*>(pointer);
    });
    return nb::ndarray<nb::numpy, T, nb::ndim<2>>(
        held->data(), {rows, columns}, owner);
}

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<3>> own_3d(
    std::vector<T>&& values, size_t a, size_t b, size_t c)
{
    auto* held = new std::vector<T>(std::move(values));
    nb::capsule owner(held, [](void* pointer) noexcept {
        delete static_cast<std::vector<T>*>(pointer);
    });
    return nb::ndarray<nb::numpy, T, nb::ndim<3>>(
        held->data(), {a, b, c}, owner);
}

uint64_t splitmix64(uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

struct PatchData {
    std::vector<std::pair<int, int>> valid_cells;
};

template <typename Rng>
int uniform_int(Rng& rng, int upper_exclusive)
{
    if (upper_exclusive <= 0)
        throw std::runtime_error("cannot sample an empty range");
    return std::uniform_int_distribution<int>(0, upper_exclusive - 1)(rng);
}

template <typename Rng>
float uniform_float(Rng& rng)
{
    return std::generate_canonical<float, 24>(rng);
}

class PatchSamplingAtlas {
public:
    PatchSamplingAtlas() = default;
    explicit PatchSamplingAtlas(const nb::list& masks) { append(masks); }

    void append(const nb::list& masks)
    {
        for (nb::handle item : masks) {
            const BoolMatrix mask = nb::cast<BoolMatrix>(item);
            PatchData patch;
            {
                nb::gil_scoped_release release;
                const int height = static_cast<int>(mask.shape(0));
                const int width = static_cast<int>(mask.shape(1));
                for (int row = 0; row < height; ++row) {
                    for (int column = 0; column < width; ++column) {
                        const bool is_valid = mask(row, column);
                        if (is_valid)
                            patch.valid_cells.emplace_back(row, column);
                    }
                }
            }
            if (patch.valid_cells.empty())
                throw std::runtime_error("patch sampling mask contains no valid quads");
            patches_.push_back(std::move(patch));
        }
    }

    size_t size() const { return patches_.size(); }

    nb::dict sample_patch_points(
        Int64Vector patch_indices, int point_cap, uint64_t seed) const
    {
        if (point_cap <= 0)
            throw std::runtime_error("point_cap must be positive");
        const size_t count = patch_indices.shape(0);
        for (size_t sample = 0; sample < count; ++sample) {
            const int64_t patch_index = patch_indices(sample);
            if (patch_index < 0 || static_cast<size_t>(patch_index) >= patches_.size())
                throw std::runtime_error("patch index is out of range");
        }
        std::vector<float> output(count * static_cast<size_t>(point_cap) * 2);
        std::vector<int64_t> counts(count);
        {
            nb::gil_scoped_release release;
#pragma omp parallel for schedule(static)
            for (int64_t sample = 0; sample < static_cast<int64_t>(count); ++sample) {
                const int64_t patch_index = patch_indices(static_cast<size_t>(sample));
                const PatchData& patch = patches_[static_cast<size_t>(patch_index)];
                std::mt19937_64 rng(splitmix64(seed + static_cast<uint64_t>(sample)));
                const int sample_count = std::min<int>(
                    point_cap, static_cast<int>(patch.valid_cells.size()));
                counts[static_cast<size_t>(sample)] = sample_count;
                std::vector<int> selected_cells;
                selected_cells.reserve(static_cast<size_t>(sample_count));
                const int valid_count = static_cast<int>(patch.valid_cells.size());
                if (sample_count == valid_count) {
                    selected_cells.resize(static_cast<size_t>(valid_count));
                    std::iota(selected_cells.begin(), selected_cells.end(), 0);
                } else {
                    // Floyd's algorithm draws a uniform subset in O(cap) memory,
                    // avoiding a full-patch permutation for large atlases.
                    std::unordered_set<int> selected_set;
                    selected_set.reserve(static_cast<size_t>(sample_count) * 2);
                    for (int candidate = valid_count - sample_count;
                         candidate < valid_count; ++candidate) {
                        const int draw = uniform_int(rng, candidate + 1);
                        if (selected_set.insert(draw).second)
                            selected_cells.push_back(draw);
                        else {
                            selected_set.insert(candidate);
                            selected_cells.push_back(candidate);
                        }
                    }
                }
                // The loss is order-independent, but randomising order prevents
                // padding and diagnostics from inheriting grid-order bias.
                for (int end = sample_count; end > 1; --end) {
                    const int swap_with = uniform_int(rng, end);
                    std::swap(selected_cells[static_cast<size_t>(end - 1)],
                              selected_cells[static_cast<size_t>(swap_with)]);
                }
                for (int point = 0; point < sample_count; ++point) {
                    const auto [row, column] = patch.valid_cells[
                        static_cast<size_t>(selected_cells[
                            static_cast<size_t>(point)])];
                    const size_t base = (static_cast<size_t>(sample) * point_cap
                        + static_cast<size_t>(point)) * 2;
                    output[base] = static_cast<float>(row) + uniform_float(rng);
                    output[base + 1] = static_cast<float>(column) + uniform_float(rng);
                }
                for (int point = sample_count; point < point_cap; ++point) {
                    const size_t base = (static_cast<size_t>(sample) * point_cap
                        + static_cast<size_t>(point)) * 2;
                    const size_t first = static_cast<size_t>(sample) * point_cap * 2;
                    output[base] = output[first];
                    output[base + 1] = output[first + 1];
                }
            }
        }
        nb::dict result;
        result["ijs"] = own_3d(
            std::move(output), count, static_cast<size_t>(point_cap), 2);
        result["counts"] = own_1d(std::move(counts));
        return result;
    }

private:
    std::vector<PatchData> patches_;
};

nb::dict prepare_dt_samples(
    BoolMatrix mask, Int64Vector row_edges, Int64Vector column_edges)
{
    if (row_edges.shape(0) < 2 || column_edges.shape(0) < 2)
        throw std::runtime_error("DT block edges must contain at least two entries");
    const int rows = static_cast<int>(row_edges.shape(0) - 1);
    const int columns = static_cast<int>(column_edges.shape(0) - 1);
    std::vector<float> ijs;
    std::vector<int32_t> block_coordinates;
    {
        nb::gil_scoped_release release;
        for (int block_row = 0; block_row < rows; ++block_row) {
            const int lo_row = static_cast<int>(row_edges(block_row));
            const int hi_row = std::max(
                static_cast<int>(row_edges(block_row + 1)), lo_row + 1);
            for (int block_column = 0; block_column < columns; ++block_column) {
                const int lo_column = static_cast<int>(column_edges(block_column));
                const int hi_column = std::max(
                    static_cast<int>(column_edges(block_column + 1)),
                    lo_column + 1);
                const double center_row = (hi_row - lo_row - 1) / 2.0;
                const double center_column = (hi_column - lo_column - 1) / 2.0;
                double best_distance = std::numeric_limits<double>::infinity();
                int best_row = -1;
                int best_column = -1;
                for (int row = lo_row; row < hi_row; ++row) {
                    for (int column = lo_column; column < hi_column; ++column) {
                        if (!mask(row, column))
                            continue;
                        const double dy = (row - lo_row) - center_row;
                        const double dx = (column - lo_column) - center_column;
                        const double distance = dy * dy + dx * dx;
                        if (distance < best_distance) {
                            best_distance = distance;
                            best_row = row;
                            best_column = column;
                        }
                    }
                }
                if (best_row < 0)
                    continue;
                ijs.push_back(static_cast<float>(best_row) + 0.5F);
                ijs.push_back(static_cast<float>(best_column) + 0.5F);
                block_coordinates.push_back(block_row);
                block_coordinates.push_back(block_column);
            }
        }
    }
    const size_t samples = ijs.size() / 2;
    nb::dict result;
    result["ijs"] = own_2d(std::move(ijs), samples, 2);
    result["block_rc"] = own_2d(std::move(block_coordinates), samples, 2);
    return result;
}

nb::dict unwrap_block_samples(
    FloatVector theta, Int32Pairs block_coordinates,
    int rows, int columns)
{
    const size_t count = theta.shape(0);
    if (block_coordinates.shape(0) != count)
        throw std::runtime_error("theta and block_rc must have equal length");
    std::vector<int64_t> adjustments(count, 0);
    std::vector<int64_t> component(count, -1);
    std::vector<int64_t> grid(static_cast<size_t>(rows) * columns, -1);
    for (size_t index = 0; index < count; ++index) {
        const int row = block_coordinates(index, 0);
        const int column = block_coordinates(index, 1);
        if (row < 0 || row >= rows || column < 0 || column >= columns)
            throw std::runtime_error("block coordinate is outside block_shape");
        grid[static_cast<size_t>(row) * columns + column] = static_cast<int64_t>(index);
    }
    int64_t components = 0;
    std::vector<int64_t> sizes;
    {
        nb::gil_scoped_release release;
        std::vector<size_t> stack;
        for (size_t seed = 0; seed < count; ++seed) {
            if (component[seed] >= 0)
                continue;
            component[seed] = components;
            stack.push_back(seed);
            int64_t size = 0;
            while (!stack.empty()) {
                const size_t current = stack.back();
                stack.pop_back();
                ++size;
                const int row = block_coordinates(current, 0);
                const int column = block_coordinates(current, 1);
                constexpr int offsets[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
                for (const auto& offset : offsets) {
                    const int next_row = row + offset[0];
                    const int next_column = column + offset[1];
                    if (next_row < 0 || next_row >= rows
                        || next_column < 0 || next_column >= columns)
                        continue;
                    const int64_t next = grid[static_cast<size_t>(next_row) * columns + next_column];
                    if (next < 0 || component[static_cast<size_t>(next)] >= 0)
                        continue;
                    const float difference = theta(static_cast<size_t>(next)) - theta(current);
                    const int step = static_cast<int>(difference > static_cast<float>(M_PI))
                        - static_cast<int>(difference < -static_cast<float>(M_PI));
                    adjustments[static_cast<size_t>(next)] = adjustments[current] + step;
                    component[static_cast<size_t>(next)] = components;
                    stack.push_back(static_cast<size_t>(next));
                }
            }
            sizes.push_back(size);
            ++components;
        }
    }
    const int64_t main_component = sizes.empty() ? 0
        : static_cast<int64_t>(std::distance(
            sizes.begin(), std::max_element(sizes.begin(), sizes.end())));
    std::vector<uint8_t> main(count);
    for (size_t index = 0; index < count; ++index)
        main[index] = component[index] == main_component ? 1 : 0;
    nb::dict result;
    result["adjustments"] = own_1d(std::move(adjustments));
    result["main"] = own_1d(std::move(main));
    return result;
}

} // namespace

NB_MODULE(spiral_sampling, module)
{
    module.doc() = "Native packed patch sampling and DT-cache helpers.";
    nb::class_<PatchSamplingAtlas>(module, "PatchSamplingAtlas")
        .def(nb::init<>())
        .def(nb::init<const nb::list&>(), nb::arg("masks"))
        .def("append", &PatchSamplingAtlas::append, nb::arg("masks"))
        .def("sample_patch_points", &PatchSamplingAtlas::sample_patch_points,
             nb::arg("patch_indices"), nb::arg("point_cap"), nb::arg("seed"))
        .def("__len__", &PatchSamplingAtlas::size);
    module.def("prepare_dt_samples", &prepare_dt_samples,
               nb::arg("mask"), nb::arg("row_edges"), nb::arg("column_edges"));
    module.def("unwrap_block_samples", &unwrap_block_samples,
               nb::arg("theta"), nb::arg("block_rc"),
               nb::arg("rows"), nb::arg("columns"));
}
