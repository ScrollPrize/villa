#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
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
using Int64Matrix = nb::ndarray<nb::numpy, const int64_t, nb::shape<-1, 2>, nb::c_contig>;
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
nb::ndarray<nb::numpy, T, nb::ndim<4>> own_4d(
    std::vector<T>&& values, size_t a, size_t b, size_t c, size_t d)
{
    auto* held = new std::vector<T>(std::move(values));
    nb::capsule owner(held, [](void* pointer) noexcept {
        delete static_cast<std::vector<T>*>(pointer);
    });
    return nb::ndarray<nb::numpy, T, nb::ndim<4>>(
        held->data(), {a, b, c, d}, owner);
}

uint64_t splitmix64(uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

struct Run {
    int lo = 0;
    int hi = 0;
    int cumulative = 0;
};

struct PatchData {
    int height = 0;
    int width = 0;
    std::vector<uint8_t> valid;
    std::vector<int> valid_rows;
    std::vector<int> valid_columns;
    std::vector<std::vector<Run>> horizontal_runs;
    std::vector<std::vector<Run>> vertical_runs;

    bool at(int row, int column) const
    {
        return row >= 0 && row < height && column >= 0 && column < width
            && valid[static_cast<size_t>(row) * width + column] != 0;
    }
};

std::vector<Run> line_runs(const PatchData& patch, bool horizontal, int fixed)
{
    const int length = horizontal ? patch.width : patch.height;
    std::vector<Run> runs;
    int cumulative = 0;
    for (int position = 0; position < length;) {
        const bool is_valid = horizontal
            ? patch.at(fixed, position) : patch.at(position, fixed);
        if (!is_valid) {
            ++position;
            continue;
        }
        const int lo = position;
        do {
            ++position;
        } while (position < length && (horizontal
            ? patch.at(fixed, position) : patch.at(position, fixed)));
        cumulative += position - lo;
        runs.push_back({lo, position, cumulative});
    }
    return runs;
}

std::pair<int, int> containing_run(
    const PatchData& patch, bool horizontal, int fixed, int position)
{
    if (!(horizontal ? patch.at(fixed, position) : patch.at(position, fixed)))
        throw std::runtime_error("sample anchor does not lie on a valid quad");
    int lo = position;
    int hi = position + 1;
    while (lo > 0 && (horizontal
        ? patch.at(fixed, lo - 1) : patch.at(lo - 1, fixed)))
        --lo;
    const int length = horizontal ? patch.width : patch.height;
    while (hi < length && (horizontal
        ? patch.at(fixed, hi) : patch.at(hi, fixed)))
        ++hi;
    return {lo, hi};
}

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

template <typename Rng>
std::vector<int> sample_sorted_positions(Rng& rng, int length, int count)
{
    std::vector<int> result(static_cast<size_t>(count));
    if (count <= length) {
        std::vector<int> pool(static_cast<size_t>(length));
        std::iota(pool.begin(), pool.end(), 0);
        for (int i = 0; i < count; ++i) {
            const int selected = i + uniform_int(rng, length - i);
            std::swap(pool[static_cast<size_t>(i)], pool[static_cast<size_t>(selected)]);
            result[static_cast<size_t>(i)] = pool[static_cast<size_t>(i)];
        }
    } else {
        for (int& value : result)
            value = uniform_int(rng, length);
    }
    std::sort(result.begin(), result.end());
    return result;
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
                patch.height = static_cast<int>(mask.shape(0));
                patch.width = static_cast<int>(mask.shape(1));
                patch.valid.resize(static_cast<size_t>(patch.height) * patch.width);
                for (int row = 0; row < patch.height; ++row) {
                    for (int column = 0; column < patch.width; ++column) {
                        patch.valid[static_cast<size_t>(row) * patch.width + column]
                            = mask(row, column) ? 1 : 0;
                    }
                }
                for (int row = 0; row < patch.height; ++row) {
                    auto runs = line_runs(patch, true, row);
                    if (!runs.empty()) {
                        patch.valid_rows.push_back(row);
                        patch.horizontal_runs.push_back(std::move(runs));
                    }
                }
                for (int column = 0; column < patch.width; ++column) {
                    auto runs = line_runs(patch, false, column);
                    if (!runs.empty()) {
                        patch.valid_columns.push_back(column);
                        patch.vertical_runs.push_back(std::move(runs));
                    }
                }
            }
            if (patch.valid_rows.empty() || patch.valid_columns.empty())
                throw std::runtime_error("patch sampling mask contains no valid quads");
            patches_.push_back(std::move(patch));
        }
    }

    size_t size() const { return patches_.size(); }

    auto sample_patch_strips(
        Int64Vector patch_indices, int points_per_direction, uint64_t seed) const
    {
        if (points_per_direction <= 0)
            throw std::runtime_error("points_per_direction must be positive");
        const size_t count = patch_indices.shape(0);
        for (size_t sample = 0; sample < count; ++sample) {
            const int64_t patch_index = patch_indices(sample);
            if (patch_index < 0 || static_cast<size_t>(patch_index) >= patches_.size())
                throw std::runtime_error("patch index is out of range");
        }
        std::vector<float> output(2 * count * points_per_direction * 2);
        {
            nb::gil_scoped_release release;
#pragma omp parallel for schedule(static)
            for (int64_t sample = 0; sample < static_cast<int64_t>(count); ++sample) {
                const int64_t patch_index = patch_indices(static_cast<size_t>(sample));
                const PatchData& patch = patches_[static_cast<size_t>(patch_index)];
                std::mt19937_64 rng(splitmix64(seed + static_cast<uint64_t>(sample)));
                for (int direction = 0; direction < 2; ++direction) {
                    const bool horizontal = direction == 0;
                    const auto& valid_lines = horizontal
                        ? patch.valid_rows : patch.valid_columns;
                    const auto& runs_by_line = horizontal
                        ? patch.horizontal_runs : patch.vertical_runs;
                    const int line_slot = uniform_int(rng, static_cast<int>(valid_lines.size()));
                    const auto& runs = runs_by_line[static_cast<size_t>(line_slot)];
                    const int selected_position = uniform_int(rng, runs.back().cumulative);
                    const auto found = std::lower_bound(
                        runs.begin(), runs.end(), selected_position + 1,
                        [](const Run& run, int value) { return run.cumulative < value; });
                    const int run_length = found->hi - found->lo;
                    const auto positions = sample_sorted_positions(
                        rng, run_length, points_per_direction);
                    const float fixed = static_cast<float>(valid_lines[static_cast<size_t>(line_slot)])
                        + uniform_float(rng);
                    for (int point = 0; point < points_per_direction; ++point) {
                        const float varying = static_cast<float>(found->lo + positions[static_cast<size_t>(point)])
                            + uniform_float(rng);
                        const size_t base = (((static_cast<size_t>(direction) * count
                            + static_cast<size_t>(sample)) * points_per_direction
                            + static_cast<size_t>(point)) * 2);
                        output[base] = horizontal ? fixed : varying;
                        output[base + 1] = horizontal ? varying : fixed;
                    }
                }
            }
        }
        return own_4d(std::move(output), 2, count,
                      static_cast<size_t>(points_per_direction), 2);
    }

    nb::dict sample_l_shapes(
        Int64Vector patch_indices, Int64Matrix anchors,
        int points_per_shape, uint64_t seed) const
    {
        if (anchors.shape(0) != patch_indices.shape(0))
            throw std::runtime_error("anchors and patch_indices must have equal length");
        if (points_per_shape <= 0)
            throw std::runtime_error("points_per_shape must be positive");
        const size_t count = patch_indices.shape(0);
        for (size_t sample = 0; sample < count; ++sample) {
            const int64_t patch_index = patch_indices(sample);
            if (patch_index < 0 || static_cast<size_t>(patch_index) >= patches_.size())
                throw std::runtime_error("patch index is out of range");
        }
        std::vector<float> output(count * 4 * points_per_shape * 2);
        std::vector<uint8_t> valid(count, 0);
        {
            nb::gil_scoped_release release;
#pragma omp parallel for schedule(static)
            for (int64_t anchor_index = 0;
                 anchor_index < static_cast<int64_t>(count); ++anchor_index) {
                const int64_t patch_index = patch_indices(static_cast<size_t>(anchor_index));
                const PatchData& patch = patches_[static_cast<size_t>(patch_index)];
                const int row = std::clamp<int64_t>(
                    anchors(static_cast<size_t>(anchor_index), 0), 0, patch.height - 1);
                const int column = std::clamp<int64_t>(
                    anchors(static_cast<size_t>(anchor_index), 1), 0, patch.width - 1);
                if (!patch.at(row, column))
                    continue;
                valid[static_cast<size_t>(anchor_index)] = 1;
                std::mt19937_64 rng(splitmix64(seed + static_cast<uint64_t>(anchor_index)));
                for (int shape = 0; shape < 4; ++shape) {
                    const bool first_horizontal = shape < 2;
                    const int first_direction = (shape == 0 || shape == 2) ? 1 : -1;
                    const int second_direction = uniform_int(rng, 2) ? 1 : -1;
                    const auto [first_lo, first_hi] = containing_run(
                        patch, first_horizontal,
                        first_horizontal ? row : column,
                        first_horizontal ? column : row);
                    const int first_start = first_horizontal ? column : row;
                    const int first_far = first_direction > 0 ? first_hi - 1 : first_lo;
                    const int first_max = std::abs(first_far - first_start);
                    const int turn_step = uniform_int(rng, first_max + 1);
                    const int turn = first_start + first_direction * turn_step;
                    const int turn_row = first_horizontal ? row : turn;
                    const int turn_column = first_horizontal ? turn : column;
                    const bool second_horizontal = !first_horizontal;
                    const auto [second_lo, second_hi] = containing_run(
                        patch, second_horizontal,
                        second_horizontal ? turn_row : turn_column,
                        second_horizontal ? turn_column : turn_row);
                    const int second_start = second_horizontal ? turn_column : turn_row;
                    const int second_far = second_direction > 0 ? second_hi - 1 : second_lo;
                    const int second_max = std::abs(second_far - second_start);
                    const int total_steps = turn_step + second_max;
                    const auto steps = sample_sorted_positions(
                        rng, total_steps + 1, points_per_shape);
                    const float first_fixed_jitter = uniform_float(rng);
                    const float second_fixed_jitter = uniform_float(rng);
                    for (int point = 0; point < points_per_shape; ++point) {
                        const int step = steps[static_cast<size_t>(point)];
                        float out_row;
                        float out_column;
                        if (step <= turn_step) {
                            const float varying = static_cast<float>(first_start + first_direction * step)
                                + uniform_float(rng);
                            const float fixed = static_cast<float>(first_horizontal ? row : column)
                                + first_fixed_jitter;
                            out_row = first_horizontal ? fixed : varying;
                            out_column = first_horizontal ? varying : fixed;
                        } else {
                            const int second_step = step - turn_step;
                            const float varying = static_cast<float>(second_start + second_direction * second_step)
                                + uniform_float(rng);
                            const float fixed = static_cast<float>(second_horizontal ? turn_row : turn_column)
                                + second_fixed_jitter;
                            out_row = second_horizontal ? fixed : varying;
                            out_column = second_horizontal ? varying : fixed;
                        }
                        const size_t base = (((static_cast<size_t>(anchor_index) * 4
                            + static_cast<size_t>(shape)) * points_per_shape
                            + static_cast<size_t>(point)) * 2);
                        output[base] = out_row;
                        output[base + 1] = out_column;
                    }
                }
            }
        }
        nb::dict result;
        result["ijs"] = own_4d(std::move(output), count, 4,
                               static_cast<size_t>(points_per_shape), 2);
        result["valid"] = own_1d(std::move(valid));
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
        .def("sample_patch_strips", &PatchSamplingAtlas::sample_patch_strips,
             nb::arg("patch_indices"), nb::arg("points_per_direction"), nb::arg("seed"))
        .def("sample_l_shapes", &PatchSamplingAtlas::sample_l_shapes,
             nb::arg("patch_indices"), nb::arg("anchors"),
             nb::arg("points_per_shape"), nb::arg("seed"))
        .def("__len__", &PatchSamplingAtlas::size);
    module.def("prepare_dt_samples", &prepare_dt_samples,
               nb::arg("mask"), nb::arg("row_edges"), nb::arg("column_edges"));
    module.def("unwrap_block_samples", &unwrap_block_samples,
               nb::arg("theta"), nb::arg("block_rc"),
               nb::arg("rows"), nb::arg("columns"));
}
