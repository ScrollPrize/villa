#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <cstring>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <variant>
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
using Int64Pairs = nb::ndarray<nb::numpy, const int64_t, nb::shape<-1, 2>, nb::c_contig>;
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

// Uninitialised output buffer for parallel fills. std::vector's
// value-initialisation memsets large chunk outputs serially before the
// parallel loop ever runs; malloc defers the page faults to the filling
// threads instead.
template <typename T>
struct RawBuffer {
    T* data = nullptr;
    size_t count = 0;

    explicit RawBuffer(size_t count_)
        : data(static_cast<T*>(std::malloc(count_ * sizeof(T)))),
          count(count_)
    {
        if (count_ && data == nullptr)
            throw std::bad_alloc();
    }
    RawBuffer(const RawBuffer&) = delete;
    RawBuffer& operator=(const RawBuffer&) = delete;
    ~RawBuffer() { std::free(data); }

    T* release()
    {
        T* released = data;
        data = nullptr;
        return released;
    }
};

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> own_1d_raw(RawBuffer<T>&& values)
{
    const size_t count = values.count;
    T* data = values.release();
    nb::capsule owner(data, [](void* pointer) noexcept {
        std::free(pointer);
    });
    return nb::ndarray<nb::numpy, T, nb::ndim<1>>(data, {count}, owner);
}

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<2>> own_2d_raw(
    RawBuffer<T>&& values, size_t rows, size_t columns)
{
    T* data = values.release();
    nb::capsule owner(data, [](void* pointer) noexcept {
        std::free(pointer);
    });
    return nb::ndarray<nb::numpy, T, nb::ndim<2>>(
        data, {rows, columns}, owner);
}

uint64_t splitmix64(uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

using IndexStorage = std::variant<std::vector<uint32_t>, std::vector<uint64_t>>;

size_t index_size(const IndexStorage& values)
{
    return std::visit([](const auto& held) { return held.size(); }, values);
}

uint64_t index_at(const IndexStorage& values, size_t position)
{
    return std::visit(
        [position](const auto& held) { return static_cast<uint64_t>(held[position]); },
        values);
}

size_t index_bytes(const IndexStorage& values)
{
    return std::visit(
        [](const auto& held) { return held.size() * sizeof(typename std::decay_t<decltype(held)>::value_type); },
        values);
}

IndexStorage compact_indices(std::vector<uint64_t>&& values, uint64_t maximum)
{
    if (maximum <= std::numeric_limits<uint32_t>::max()) {
        std::vector<uint32_t> compact;
        compact.reserve(values.size());
        for (uint64_t value : values)
            compact.push_back(static_cast<uint32_t>(value));
        return compact;
    }
    return std::move(values);
}

struct PatchData {
    uint64_t height = 0;
    uint64_t width = 0;
    uint64_t row_lo = 0;
    uint64_t column_lo = 0;
    uint64_t rectangle_width = 0;
    bool rectangular = false;
    IndexStorage valid_cells;
    // Solid rectangles use an implicit boustrophedon chain. Ragged patches
    // retain only compact DFS ordinals/parents/subtree ends, never edges.
    IndexStorage preorder_ordinals;
    IndexStorage parent_positions;
    IndexStorage subtree_ends;
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
    PatchSamplingAtlas() { node_offsets_.push_back(0); }
    explicit PatchSamplingAtlas(const nb::list& masks) { append(masks); }

    void append(const nb::list& masks)
    {
        if (node_offsets_.empty())
            node_offsets_.push_back(0);
        // Casting touches Python objects, so the views are collected under
        // the GIL; the per-patch builds are pure C++ over the borrowed
        // buffers and run in parallel with it released.
        std::vector<BoolMatrix> views;
        views.reserve(nb::len(masks));
        for (nb::handle item : masks)
            views.push_back(nb::cast<BoolMatrix>(item));
        const size_t base = patches_.size();
        patches_.resize(base + views.size());
        bool empty_mask = false;
        {
            nb::gil_scoped_release release;
#pragma omp parallel for schedule(dynamic) reduction(|| : empty_mask)
            for (int64_t index = 0;
                 index < static_cast<int64_t>(views.size()); ++index) {
                PatchData patch;
                if (!build_patch(views[static_cast<size_t>(index)], patch)) {
                    empty_mask = true;
                    continue;
                }
                patches_[base + static_cast<size_t>(index)] = std::move(patch);
            }
        }
        if (empty_mask) {
            patches_.resize(base);
            throw std::runtime_error(
                "patch sampling mask contains no valid quads");
        }
        for (size_t index = base; index < patches_.size(); ++index)
            node_offsets_.push_back(
                node_offsets_.back()
                + index_size(patches_[index].valid_cells));
    }

    size_t size() const { return patches_.size(); }

    uint64_t total_valid_cells() const
    {
        return node_offsets_.empty() ? 0 : node_offsets_.back();
    }

    nb::ndarray<nb::numpy, int64_t, nb::ndim<1>> valid_counts() const
    {
        std::vector<int64_t> counts;
        counts.reserve(patches_.size());
        for (const PatchData& patch : patches_)
            counts.push_back(static_cast<int64_t>(index_size(patch.valid_cells)));
        return own_1d(std::move(counts));
    }

    nb::dict memory_stats() const
    {
        uint64_t cell_bytes = 0;
        uint64_t tree_bytes = 0;
        uint64_t rectangle_count = 0;
        for (const PatchData& patch : patches_) {
            cell_bytes += index_bytes(patch.valid_cells);
            tree_bytes += index_bytes(patch.preorder_ordinals);
            tree_bytes += index_bytes(patch.parent_positions);
            tree_bytes += index_bytes(patch.subtree_ends);
            rectangle_count += patch.rectangular ? 1 : 0;
        }
        nb::dict result;
        result["num_patches"] = patches_.size();
        result["num_valid_cells"] = total_valid_cells();
        result["rectangle_patches"] = rectangle_count;
        result["cell_bytes"] = cell_bytes;
        result["tree_bytes"] = tree_bytes;
        result["persistent_bytes"] = cell_bytes + tree_bytes
            + node_offsets_.size() * sizeof(uint64_t);
        return result;
    }

    nb::dict node_ijs(Int64Vector node_ordinals) const
    {
        const size_t count = node_ordinals.shape(0);
        RawBuffer<int64_t> patch_indices(count);
        RawBuffer<float> ijs(count * 2);
        {
            nb::gil_scoped_release release;
            bool out_of_range = false;
#pragma omp parallel for schedule(static) reduction(|| : out_of_range)
            for (int64_t index = 0; index < static_cast<int64_t>(count);
                 ++index) {
                const int64_t requested = node_ordinals(
                    static_cast<size_t>(index));
                if (requested < 0
                    || static_cast<uint64_t>(requested) >= total_valid_cells()) {
                    out_of_range = true;
                    continue;
                }
                const size_t patch_index = locate_node(
                    static_cast<uint64_t>(requested));
                const PatchData& patch = patches_[patch_index];
                const uint64_t local = static_cast<uint64_t>(requested)
                    - node_offsets_[patch_index];
                const uint64_t linear = index_at(patch.valid_cells, local);
                patch_indices.data[static_cast<size_t>(index)] =
                    static_cast<int64_t>(patch_index);
                ijs.data[index * 2] = static_cast<float>(linear / patch.width);
                ijs.data[index * 2 + 1] =
                    static_cast<float>(linear % patch.width);
            }
            if (out_of_range)
                throw std::runtime_error("node ordinal is out of range");
        }
        nb::dict result;
        result["patch_indices"] = own_1d_raw(std::move(patch_indices));
        result["ijs"] = own_2d_raw(std::move(ijs), count, 2);
        return result;
    }

    nb::ndarray<nb::numpy, int64_t, nb::ndim<1>> cell_node_ordinals(
        Int64Vector patch_indices, Int64Pairs cells) const
    {
        const size_t count = patch_indices.shape(0);
        if (cells.shape(0) != count)
            throw std::runtime_error("patch indices and cells must have equal length");
        std::vector<int64_t> output(count);
        {
            nb::gil_scoped_release release;
            for (size_t index = 0; index < count; ++index) {
                const int64_t patch_index_signed = patch_indices(index);
                if (patch_index_signed < 0
                    || static_cast<size_t>(patch_index_signed) >= patches_.size())
                    throw std::runtime_error("patch index is out of range");
                const size_t patch_index = static_cast<size_t>(patch_index_signed);
                const PatchData& patch = patches_[patch_index];
                const int64_t row = cells(index, 0);
                const int64_t column = cells(index, 1);
                if (row < 0 || column < 0
                    || static_cast<uint64_t>(row) >= patch.height
                    || static_cast<uint64_t>(column) >= patch.width)
                    throw std::runtime_error("patch cell is out of range");
                const uint64_t linear = static_cast<uint64_t>(row) * patch.width
                    + static_cast<uint64_t>(column);
                const uint64_t local = valid_ordinal(patch, linear);
                if (local == missing_index)
                    throw std::runtime_error(
                        "patch sampler selected a cell outside crossing topology");
                output[index] = static_cast<int64_t>(
                    node_offsets_[patch_index] + local);
            }
        }
        return own_1d(std::move(output));
    }

    nb::dict tree_chunk(uint64_t lo, uint64_t hi) const
    {
        if (lo > hi || hi > total_valid_cells())
            throw std::runtime_error("tree chunk is out of range");
        const size_t count = static_cast<size_t>(hi - lo);
        RawBuffer<int64_t> nodes(count);
        RawBuffer<int64_t> parents(count);
        RawBuffer<int64_t> exits(count);
        {
            nb::gil_scoped_release release;
            // Contiguous position sub-ranges write disjoint output slices, so
            // the result is identical to a serial pass for any thread count.
            const uint64_t per_range = 262'144;
            const size_t num_ranges = count
                ? static_cast<size_t>((count + per_range - 1) / per_range) : 0;
#pragma omp parallel for schedule(dynamic)
            for (int64_t range_index = 0;
                 range_index < static_cast<int64_t>(num_ranges); ++range_index) {
            const uint64_t range_lo = lo
                + static_cast<uint64_t>(range_index) * per_range;
            const uint64_t range_hi = std::min(hi, range_lo + per_range);
            size_t patch_index = locate_node(range_lo);
            for (uint64_t position = range_lo; position < range_hi; ++position) {
                while (position >= node_offsets_[patch_index + 1])
                    ++patch_index;
                const PatchData& patch = patches_[patch_index];
                const uint64_t patch_start = node_offsets_[patch_index];
                const uint64_t local_position = position - patch_start;
                uint64_t node_local;
                uint64_t parent_local;
                uint64_t exit_position;
                if (patch.rectangular) {
                    node_local = rectangle_preorder_ordinal(
                        patch, local_position);
                    parent_local = local_position == 0 ? node_local
                        : rectangle_preorder_ordinal(patch, local_position - 1);
                    exit_position = local_position == 0 ? patch_start
                        : node_offsets_[patch_index + 1];
                } else {
                    node_local = index_at(patch.preorder_ordinals, local_position);
                    const uint64_t parent_position = index_at(
                        patch.parent_positions, local_position);
                    const uint64_t parent_sentinel =
                        std::holds_alternative<std::vector<uint32_t>>(
                            patch.parent_positions)
                        ? std::numeric_limits<uint32_t>::max()
                        : std::numeric_limits<uint64_t>::max();
                    if (parent_position == parent_sentinel) {
                        parent_local = node_local;
                        exit_position = patch_start + local_position;
                    } else {
                        parent_local = index_at(
                            patch.preorder_ordinals, parent_position);
                        exit_position = patch_start
                            + index_at(patch.subtree_ends, local_position) + 1;
                    }
                }
                const size_t output = static_cast<size_t>(position - lo);
                nodes.data[output] =
                    static_cast<int64_t>(patch_start + node_local);
                parents.data[output] =
                    static_cast<int64_t>(patch_start + parent_local);
                exits.data[output] = static_cast<int64_t>(exit_position);
            }
            }
        }
        nb::dict result;
        result["node_ordinals"] = own_1d_raw(std::move(nodes));
        result["parent_ordinals"] = own_1d_raw(std::move(parents));
        result["exit_positions"] = own_1d_raw(std::move(exits));
        return result;
    }

    nb::dict neighbor_chunk(uint64_t cursor, uint64_t slot_count) const
    {
        const uint64_t end = total_valid_cells() * 4;
        if (cursor > end || slot_count == 0)
            throw std::runtime_error("invalid neighbor chunk request");
        const uint64_t next = std::min(end, cursor + slot_count);
        std::optional<RawBuffer<int64_t>> pairs;
        {
            nb::gil_scoped_release release;
            // Contiguous slot sub-ranges are filled independently and
            // concatenated in range order, so the emitted pair sequence is
            // identical to a single serial pass for any thread count.
            const uint64_t total = next - cursor;
            const uint64_t per_range = 262'144;
            const size_t num_ranges = total
                ? static_cast<size_t>((total + per_range - 1) / per_range) : 0;
            std::vector<std::vector<int64_t>> partial(num_ranges);
#pragma omp parallel for schedule(dynamic)
            for (int64_t range_index = 0;
                 range_index < static_cast<int64_t>(num_ranges); ++range_index) {
                const uint64_t lo = cursor
                    + static_cast<uint64_t>(range_index) * per_range;
                const uint64_t hi = std::min(next, lo + per_range);
                std::vector<int64_t>& out = partial[
                    static_cast<size_t>(range_index)];
                out.reserve(static_cast<size_t>(hi - lo) * 2);
                size_t patch_index = locate_node(lo / 4);
                constexpr int offsets[4][2] = {{0, 1}, {1, -1}, {1, 0}, {1, 1}};
                for (uint64_t slot = lo; slot < hi; ++slot) {
                    const uint64_t node = slot / 4;
                    while (node >= node_offsets_[patch_index + 1])
                        ++patch_index;
                    const PatchData& patch = patches_[patch_index];
                    const uint64_t patch_start = node_offsets_[patch_index];
                    const uint64_t local = node - patch_start;
                    const uint64_t linear = index_at(patch.valid_cells, local);
                    const int64_t row = static_cast<int64_t>(linear / patch.width);
                    const int64_t column = static_cast<int64_t>(linear % patch.width);
                    const int direction = static_cast<int>(slot % 4);
                    const int64_t next_row = row + offsets[direction][0];
                    const int64_t next_column = column + offsets[direction][1];
                    if (next_row < 0 || next_column < 0
                        || static_cast<uint64_t>(next_row) >= patch.height
                        || static_cast<uint64_t>(next_column) >= patch.width)
                        continue;
                    const uint64_t next_linear = static_cast<uint64_t>(next_row)
                        * patch.width + static_cast<uint64_t>(next_column);
                    const uint64_t next_local = valid_ordinal(patch, next_linear);
                    if (next_local == missing_index)
                        continue;
                    out.push_back(static_cast<int64_t>(node));
                    out.push_back(static_cast<int64_t>(patch_start + next_local));
                }
            }
            std::vector<size_t> range_offsets(num_ranges + 1, 0);
            for (size_t range_index = 0; range_index < num_ranges; ++range_index)
                range_offsets[range_index + 1] = range_offsets[range_index]
                    + partial[range_index].size();
            pairs.emplace(range_offsets[num_ranges]);
#pragma omp parallel for schedule(static)
            for (int64_t range_index = 0;
                 range_index < static_cast<int64_t>(num_ranges); ++range_index) {
                const std::vector<int64_t>& out = partial[
                    static_cast<size_t>(range_index)];
                if (!out.empty())
                    std::memcpy(
                        pairs->data + range_offsets[
                            static_cast<size_t>(range_index)],
                        out.data(), out.size() * sizeof(int64_t));
            }
        }
        nb::dict result;
        result["next_cursor"] = next;
        const size_t pair_count = pairs->count / 2;
        result["node_pairs"] = own_2d_raw(std::move(*pairs), pair_count, 2);
        return result;
    }

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
            if (index_size(patches_[static_cast<size_t>(patch_index)].valid_cells)
                > static_cast<size_t>(std::numeric_limits<int>::max()))
                throw std::runtime_error("one patch contains too many valid cells to sample");
        }
        std::vector<float> output(count * static_cast<size_t>(point_cap) * 2);
        std::vector<int64_t> counts(count);
        std::vector<int64_t> node_ordinals(
            count * static_cast<size_t>(point_cap));
        {
            nb::gil_scoped_release release;
#pragma omp parallel for schedule(static)
            for (int64_t sample = 0; sample < static_cast<int64_t>(count); ++sample) {
                const int64_t patch_index = patch_indices(static_cast<size_t>(sample));
                const PatchData& patch = patches_[static_cast<size_t>(patch_index)];
                std::mt19937_64 rng(splitmix64(seed + static_cast<uint64_t>(sample)));
                const int valid_count = static_cast<int>(index_size(patch.valid_cells));
                const int sample_count = std::min(point_cap, valid_count);
                counts[static_cast<size_t>(sample)] = sample_count;
                std::vector<int> selected_cells;
                selected_cells.reserve(static_cast<size_t>(sample_count));
                if (sample_count == valid_count) {
                    selected_cells.resize(static_cast<size_t>(valid_count));
                    std::iota(selected_cells.begin(), selected_cells.end(), 0);
                } else {
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
                for (int shuffle_end = sample_count; shuffle_end > 1; --shuffle_end) {
                    const int swap_with = uniform_int(rng, shuffle_end);
                    std::swap(selected_cells[static_cast<size_t>(shuffle_end - 1)],
                              selected_cells[static_cast<size_t>(swap_with)]);
                }
                for (int point = 0; point < sample_count; ++point) {
                    const uint64_t local = static_cast<uint64_t>(selected_cells[
                        static_cast<size_t>(point)]);
                    const uint64_t linear = index_at(patch.valid_cells, local);
                    const size_t base = (static_cast<size_t>(sample) * point_cap
                        + static_cast<size_t>(point)) * 2;
                    const size_t ordinal_index = static_cast<size_t>(sample)
                        * point_cap + static_cast<size_t>(point);
                    output[base] = static_cast<float>(linear / patch.width)
                        + uniform_float(rng);
                    output[base + 1] = static_cast<float>(linear % patch.width)
                        + uniform_float(rng);
                    node_ordinals[ordinal_index] = static_cast<int64_t>(
                        node_offsets_[static_cast<size_t>(patch_index)] + local);
                }
                for (int point = sample_count; point < point_cap; ++point) {
                    const size_t base = (static_cast<size_t>(sample) * point_cap
                        + static_cast<size_t>(point)) * 2;
                    const size_t first = static_cast<size_t>(sample) * point_cap * 2;
                    const size_t ordinal_index = static_cast<size_t>(sample)
                        * point_cap + static_cast<size_t>(point);
                    output[base] = output[first];
                    output[base + 1] = output[first + 1];
                    node_ordinals[ordinal_index] = node_ordinals[
                        static_cast<size_t>(sample) * point_cap];
                }
            }
        }
        nb::dict result;
        result["ijs"] = own_3d(
            std::move(output), count, static_cast<size_t>(point_cap), 2);
        result["counts"] = own_1d(std::move(counts));
        result["node_ordinals"] = own_2d(
            std::move(node_ordinals), count, static_cast<size_t>(point_cap));
        return result;
    }

private:
    static constexpr uint64_t missing_index = std::numeric_limits<uint64_t>::max();

    static uint64_t valid_ordinal(const PatchData& patch, uint64_t linear)
    {
        return std::visit([linear](const auto& held) -> uint64_t {
            using Value = typename std::decay_t<decltype(held)>::value_type;
            if (linear > static_cast<uint64_t>(std::numeric_limits<Value>::max()))
                return missing_index;
            const auto value = static_cast<Value>(linear);
            const auto found = std::lower_bound(held.begin(), held.end(), value);
            return found != held.end() && *found == value
                ? static_cast<uint64_t>(found - held.begin()) : missing_index;
        }, patch.valid_cells);
    }

    static uint64_t rectangle_preorder_ordinal(
        const PatchData& patch, uint64_t position)
    {
        const uint64_t row = position / patch.rectangle_width;
        const uint64_t offset = position % patch.rectangle_width;
        const uint64_t column = row % 2 == 0
            ? offset : patch.rectangle_width - 1 - offset;
        return row * patch.rectangle_width + column;
    }

    // Pure C++ (no GIL interaction): runs inside the parallel append loop.
    // Returns false for a mask with no valid quads instead of throwing so the
    // caller can fail after the parallel region.
    static bool build_patch(const BoolMatrix& mask, PatchData& patch)
    {
        patch.height = mask.shape(0);
        patch.width = mask.shape(1);
        std::vector<uint64_t> cells;
        uint64_t row_hi = 0;
        uint64_t column_hi = 0;
        patch.row_lo = patch.height;
        patch.column_lo = patch.width;
        for (uint64_t row = 0; row < patch.height; ++row) {
            for (uint64_t column = 0; column < patch.width; ++column) {
                if (!mask(row, column))
                    continue;
                cells.push_back(row * patch.width + column);
                patch.row_lo = std::min(patch.row_lo, row);
                patch.column_lo = std::min(patch.column_lo, column);
                row_hi = std::max(row_hi, row + 1);
                column_hi = std::max(column_hi, column + 1);
            }
        }
        if (cells.empty())
            return false;
        patch.rectangle_width = column_hi - patch.column_lo;
        patch.rectangular = cells.size()
            == (row_hi - patch.row_lo) * patch.rectangle_width;
        const uint64_t maximum = cells.back();
        patch.valid_cells = compact_indices(std::move(cells), maximum);
        if (!patch.rectangular)
            build_ragged_tree(patch);
        return true;
    }

    static void build_ragged_tree(PatchData& patch)
    {
        const size_t count = index_size(patch.valid_cells);
        // Transient dense linear -> ordinal map: the DFS resolves eight
        // neighbours per cell, so O(1) lookups beat per-neighbour binary
        // searches. Freed on return; falls back to searches for the
        // (unrealistic) case of a grid too large for uint32 ordinals.
        constexpr uint32_t no_ordinal = std::numeric_limits<uint32_t>::max();
        const uint64_t area = patch.height * patch.width;
        std::vector<uint32_t> dense_ordinals;
        if (area < no_ordinal && count < no_ordinal) {
            dense_ordinals.assign(static_cast<size_t>(area), no_ordinal);
            for (size_t ordinal = 0; ordinal < count; ++ordinal)
                dense_ordinals[index_at(patch.valid_cells, ordinal)] =
                    static_cast<uint32_t>(ordinal);
        }
        auto resolve = [&patch, &dense_ordinals](uint64_t linear) -> uint64_t {
            if (dense_ordinals.empty())
                return valid_ordinal(patch, linear);
            const uint32_t ordinal = dense_ordinals[
                static_cast<size_t>(linear)];
            return ordinal == no_ordinal ? missing_index : ordinal;
        };
        struct Frame {
            uint64_t ordinal;
            uint64_t preorder_position;
            std::array<uint64_t, 8> neighbors {};
            uint8_t neighbor_count = 0;
            uint8_t next = 0;
        };
        auto make_frame = [&patch, &resolve](
                              uint64_t ordinal, uint64_t preorder_position) {
            Frame frame;
            frame.ordinal = ordinal;
            frame.preorder_position = preorder_position;
            const uint64_t linear = index_at(patch.valid_cells, ordinal);
            const int64_t row = static_cast<int64_t>(linear / patch.width);
            const int64_t column = static_cast<int64_t>(linear % patch.width);
            // Match scipy's directed=False traversal of the legacy forward
            // CSR graph: visit sorted outgoing entries first, followed by the
            // sorted entries from its transpose.
            constexpr int offsets[8][2] = {
                {0, 1}, {1, -1}, {1, 0}, {1, 1},
                {-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
            };
            for (const auto& offset : offsets) {
                const int64_t next_row = row + offset[0];
                const int64_t next_column = column + offset[1];
                if (next_row < 0 || next_column < 0
                    || static_cast<uint64_t>(next_row) >= patch.height
                    || static_cast<uint64_t>(next_column) >= patch.width)
                    continue;
                const uint64_t next_linear = static_cast<uint64_t>(next_row)
                    * patch.width + static_cast<uint64_t>(next_column);
                const uint64_t next_ordinal = resolve(next_linear);
                if (next_ordinal != missing_index)
                    frame.neighbors[frame.neighbor_count++] = next_ordinal;
            }
            return frame;
        };

        std::vector<uint8_t> visited(count, 0);
        std::vector<uint64_t> preorder;
        std::vector<uint64_t> parents;
        std::vector<Frame> stack;
        preorder.reserve(count);
        parents.reserve(count);
        stack.reserve(std::min<size_t>(count, 1'000'000));
        for (uint64_t seed = 0; seed < count; ++seed) {
            if (visited[seed])
                continue;
            visited[seed] = 1;
            const uint64_t root_position = preorder.size();
            preorder.push_back(seed);
            parents.push_back(missing_index);
            stack.push_back(make_frame(seed, root_position));
            while (!stack.empty()) {
                Frame& frame = stack.back();
                if (frame.next >= frame.neighbor_count) {
                    stack.pop_back();
                    continue;
                }
                const uint64_t child = frame.neighbors[frame.next++];
                if (visited[child])
                    continue;
                visited[child] = 1;
                const uint64_t parent_position = frame.preorder_position;
                const uint64_t child_position = preorder.size();
                preorder.push_back(child);
                parents.push_back(parent_position);
                stack.push_back(make_frame(child, child_position));
            }
        }
        std::vector<uint64_t> subtree_sizes(count, 1);
        for (size_t child = count; child-- > 0;) {
            if (parents[child] != missing_index)
                subtree_sizes[parents[child]] += subtree_sizes[child];
        }
        std::vector<uint64_t> subtree_ends(count);
        for (size_t position = 0; position < count; ++position)
            subtree_ends[position] = position + subtree_sizes[position] - 1;
        const uint64_t maximum = count - 1;
        patch.preorder_ordinals = compact_indices(std::move(preorder), maximum);
        patch.parent_positions = compact_indices(std::move(parents), maximum);
        patch.subtree_ends = compact_indices(std::move(subtree_ends), maximum);
    }

    size_t locate_node(uint64_t ordinal) const
    {
        const auto found = std::upper_bound(
            node_offsets_.begin(), node_offsets_.end(), ordinal);
        return static_cast<size_t>(found - node_offsets_.begin() - 1);
    }

    std::vector<PatchData> patches_;
    std::vector<uint64_t> node_offsets_;
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
        .def("valid_counts", &PatchSamplingAtlas::valid_counts)
        .def("total_valid_cells", &PatchSamplingAtlas::total_valid_cells)
        .def("node_ijs", &PatchSamplingAtlas::node_ijs,
             nb::arg("node_ordinals"))
        .def("cell_node_ordinals", &PatchSamplingAtlas::cell_node_ordinals,
             nb::arg("patch_indices"), nb::arg("cells"))
        .def("tree_chunk", &PatchSamplingAtlas::tree_chunk,
             nb::arg("lo"), nb::arg("hi"))
        .def("neighbor_chunk", &PatchSamplingAtlas::neighbor_chunk,
             nb::arg("cursor"), nb::arg("slot_count"))
        .def("memory_stats", &PatchSamplingAtlas::memory_stats)
        .def("sample_patch_points", &PatchSamplingAtlas::sample_patch_points,
             nb::arg("patch_indices"), nb::arg("point_cap"), nb::arg("seed"))
        .def("__len__", &PatchSamplingAtlas::size);
    module.def("prepare_dt_samples", &prepare_dt_samples,
               nb::arg("mask"), nb::arg("row_edges"), nb::arg("column_edges"));
    module.def("unwrap_block_samples", &unwrap_block_samples,
               nb::arg("theta"), nb::arg("block_rc"),
               nb::arg("rows"), nb::arg("columns"));
}
