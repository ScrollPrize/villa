#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__GLIBCXX__) && defined(_OPENMP)
#include <parallel/algorithm>
#include <parallel/tags.h>
#endif

namespace nb = nanobind;

namespace {

using Coordinates = nb::ndarray<nb::numpy, const int32_t,
                                nb::shape<-1, 3>, nb::c_contig>;
using Int64Vector = nb::ndarray<nb::numpy, const int64_t,
                                nb::ndim<1>, nb::c_contig>;
using UInt64Vector = nb::ndarray<nb::numpy, const uint64_t,
                                 nb::ndim<1>, nb::c_contig>;
using UInt32Vector = nb::ndarray<nb::numpy, const uint32_t,
                                 nb::ndim<1>, nb::c_contig>;
using Int8Vector = nb::ndarray<nb::numpy, const int8_t,
                               nb::ndim<1>, nb::c_contig>;
using Int32Vector = nb::ndarray<nb::numpy, const int32_t,
                                nb::ndim<1>, nb::c_contig>;
using Float64Vector = nb::ndarray<nb::numpy, const double,
                                  nb::ndim<1>, nb::c_contig>;
using FloatCoordinates = nb::ndarray<nb::numpy, const float,
                                     nb::shape<-1, 3>, nb::c_contig>;
using Int32Matrix = nb::ndarray<nb::numpy, const int32_t,
                                nb::ndim<2>, nb::c_contig>;

struct Event {
    int32_t first;
    int32_t second;
    int32_t first_local;
    int32_t second_local;
};

static_assert(sizeof(Event) == 16);

struct EventBuffer {
    std::vector<Event> events;

    size_t size() const { return events.size(); }
    size_t memory_bytes() const { return events.capacity() * sizeof(Event); }
};

struct PairRange {
    size_t begin;
    size_t end;
};

struct PairEdge {
    int32_t first;
    int32_t second;
    int32_t first_local;
    int32_t second_local;
    double first_position;
    double second_position;
    double clearance;
};

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

void report(const nb::object& callback, const char* phase,
            uint64_t completed, uint64_t total)
{
    if (!callback.is_none())
        callback(phase, completed, total);
}

int effective_workers(int requested)
{
    if (requested < 1)
        throw std::runtime_error("workers must be positive");
#ifdef _OPENMP
    return std::min(requested, omp_get_num_procs());
#else
    return 1;
#endif
}

nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>> parallel_argsort(
    UInt64Vector packed, int workers, const nb::object& progress)
{
    workers = effective_workers(workers);
    const size_t count = packed.shape(0);
    if (count > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
        throw std::runtime_error(
            "native packed-key radix sort supports at most UINT32_MAX points");

    constexpr unsigned radix_bits = 11;
    constexpr size_t maximum_buckets = size_t{1} << radix_bits;
    constexpr unsigned key_bits = 60;
    constexpr unsigned passes = (key_bits + radix_bits - 1) / radix_bits;
    std::vector<uint64_t> keys_a(count);
    std::vector<uint64_t> keys_b(count);
    std::vector<uint32_t> order_a(count);
    std::vector<uint32_t> order_b(count);
    const uint64_t* input = packed.data();

    report(progress, "radix sorting packed voxel keys", 0, passes);
    {
        nb::gil_scoped_release release;
#pragma omp parallel for schedule(static) num_threads(workers)
        for (int64_t index = 0; index < static_cast<int64_t>(count); ++index) {
            keys_a[index] = input[index];
            order_a[index] = static_cast<uint32_t>(index);
        }
    }

    for (unsigned pass = 0; pass < passes; ++pass) {
        const unsigned shift = pass * radix_bits;
        const unsigned bits = std::min(radix_bits, key_bits - shift);
        const size_t bucket_count = size_t{1} << bits;
        const uint64_t mask = bucket_count - 1;
        std::vector<uint64_t> positions(
            static_cast<size_t>(workers) * maximum_buckets, 0);

        {
            nb::gil_scoped_release release;
#pragma omp parallel num_threads(workers)
            {
#ifdef _OPENMP
                const int thread = omp_get_thread_num();
#else
                const int thread = 0;
#endif
                const size_t begin = count * static_cast<size_t>(thread) / workers;
                const size_t end = count * static_cast<size_t>(thread + 1) / workers;
                uint64_t* local = positions.data()
                    + static_cast<size_t>(thread) * maximum_buckets;
                for (size_t index = begin; index < end; ++index)
                    ++local[(keys_a[index] >> shift) & mask];
            }

            uint64_t destination = 0;
            for (size_t bucket = 0; bucket < bucket_count; ++bucket) {
                for (int thread = 0; thread < workers; ++thread) {
                    uint64_t& slot = positions[
                        static_cast<size_t>(thread) * maximum_buckets + bucket];
                    const uint64_t bucket_size = slot;
                    slot = destination;
                    destination += bucket_size;
                }
            }

#pragma omp parallel num_threads(workers)
            {
#ifdef _OPENMP
                const int thread = omp_get_thread_num();
#else
                const int thread = 0;
#endif
                const size_t begin = count * static_cast<size_t>(thread) / workers;
                const size_t end = count * static_cast<size_t>(thread + 1) / workers;
                uint64_t* local = positions.data()
                    + static_cast<size_t>(thread) * maximum_buckets;
                for (size_t index = begin; index < end; ++index) {
                    const size_t bucket = (keys_a[index] >> shift) & mask;
                    const size_t output = static_cast<size_t>(local[bucket]++);
                    keys_b[output] = keys_a[index];
                    order_b[output] = order_a[index];
                }
            }
        }
        keys_a.swap(keys_b);
        order_a.swap(order_b);
        report(progress, "radix sorting packed voxel keys", pass + 1, passes);
    }
    return own_1d(std::move(order_a));
}

struct Tangent {
    double z = 0.0;
    double y = 0.0;
    double x = 0.0;
    bool valid = false;
};

Tangent track_tangent(const int32_t* coordinates, const int64_t* offsets,
                      int32_t track, int32_t local, double radius = 12.0)
{
    const int64_t begin = offsets[track];
    const int64_t end = offsets[track + 1];
    const int64_t center = begin + local;
    int64_t left = center;
    int64_t right = center;

    auto distance_from_center = [&](int64_t index) {
        const double dz = static_cast<double>(coordinates[3 * index])
            - coordinates[3 * center];
        const double dy = static_cast<double>(coordinates[3 * index + 1])
            - coordinates[3 * center + 1];
        const double dx = static_cast<double>(coordinates[3 * index + 2])
            - coordinates[3 * center + 2];
        return std::sqrt(dz * dz + dy * dy + dx * dx);
    };

    while (left > begin && distance_from_center(left) < radius)
        --left;
    while (right + 1 < end && distance_from_center(right) < radius)
        ++right;
    if (left == right)
        return {};

    const double dz = static_cast<double>(coordinates[3 * right])
        - coordinates[3 * left];
    const double dy = static_cast<double>(coordinates[3 * right + 1])
        - coordinates[3 * left + 1];
    const double dx = static_cast<double>(coordinates[3 * right + 2])
        - coordinates[3 * left + 2];
    const double norm = std::sqrt(dz * dz + dy * dy + dx * dx);
    if (norm == 0.0)
        return {};
    return {dz / norm, dy / norm, dx / norm, true};
}

EventBuffer scan_crossing_events(
    Coordinates coordinates, Int64Vector offsets, Int8Vector family_codes,
    UInt64Vector packed, UInt32Vector order, int workers,
    const nb::object& progress)
{
    workers = effective_workers(workers);
    const size_t point_count = order.shape(0);
    const size_t track_count = family_codes.shape(0);
    if (coordinates.shape(0) != point_count || packed.shape(0) != point_count)
        throw std::runtime_error("coordinates, packed keys, and order must be parallel");
    if (offsets.shape(0) != track_count + 1
        || static_cast<uint64_t>(offsets(track_count)) != point_count)
        throw std::runtime_error("track offsets do not match coordinates");
    if (track_count > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("native crossing scan supports at most INT32_MAX tracks");

    const int32_t* coordinate_data = coordinates.data();
    const int64_t* offset_data = offsets.data();
    const int8_t* family_data = family_codes.data();
    const uint64_t* packed_data = packed.data();
    const uint32_t* order_data = order.data();

    constexpr size_t chunk_size = 500'000;
    std::vector<std::pair<size_t, size_t>> tasks;
    for (size_t begin = 0; begin < point_count;) {
        size_t end = std::min(begin + chunk_size, point_count);
        while (end < point_count
               && packed_data[order_data[end - 1]] == packed_data[order_data[end]])
            ++end;
        tasks.emplace_back(begin, end);
        begin = end;
    }

    EventBuffer result;
    const size_t tasks_per_batch = std::max<size_t>(workers * 2, 1);
    const double angle_cutoff = std::cos(30.0 * std::acos(-1.0) / 180.0);
    uint64_t completed = 0;
    report(progress, "finding exact crossings", 0, point_count);

    for (size_t batch_begin = 0; batch_begin < tasks.size();
         batch_begin += tasks_per_batch) {
        const size_t batch_end = std::min(
            batch_begin + tasks_per_batch, tasks.size());
        std::vector<std::vector<Event>> batch_events(batch_end - batch_begin);

        {
            nb::gil_scoped_release release;
#pragma omp parallel for schedule(dynamic) num_threads(workers)
            for (int64_t task_index = static_cast<int64_t>(batch_begin);
                 task_index < static_cast<int64_t>(batch_end); ++task_index) {
                const auto [position_begin, position_end] = tasks[task_index];
                auto& local_events = batch_events[task_index - batch_begin];
                std::vector<std::pair<int32_t, int32_t>> unique;
                std::vector<Tangent> tangents;

                for (size_t position = position_begin; position < position_end;) {
                    size_t group_end = position + 1;
                    const uint64_t key = packed_data[order_data[position]];
                    while (group_end < position_end
                           && packed_data[order_data[group_end]] == key)
                        ++group_end;
                    if (group_end - position < 2) {
                        position = group_end;
                        continue;
                    }

                    unique.clear();
                    unique.reserve(group_end - position);
                    for (size_t item = position; item < group_end; ++item) {
                        const int64_t flat = order_data[item];
                        const auto* found = std::upper_bound(
                            offset_data + 1, offset_data + track_count + 1, flat);
                        const int32_t track = static_cast<int32_t>(
                            found - (offset_data + 1));
                        if (family_data[track] < 0)
                            continue;
                        const int32_t local = static_cast<int32_t>(
                            flat - offset_data[track]);
                        unique.emplace_back(track, local);
                    }
                    std::sort(unique.begin(), unique.end());
                    auto output = unique.begin();
                    for (auto input = unique.begin(); input != unique.end();) {
                        const int32_t track = input->first;
                        int32_t local = input->second;
                        do {
                            local = std::min(local, input->second);
                            ++input;
                        } while (input != unique.end() && input->first == track);
                        *output++ = {track, local};
                    }
                    unique.erase(output, unique.end());
                    if (unique.size() < 2) {
                        position = group_end;
                        continue;
                    }

                    tangents.resize(unique.size());
                    for (size_t index = 0; index < unique.size(); ++index) {
                        tangents[index] = track_tangent(
                            coordinate_data, offset_data,
                            unique[index].first, unique[index].second);
                    }
                    for (size_t first = 0; first < unique.size(); ++first) {
                        if (!tangents[first].valid)
                            continue;
                        for (size_t second = first + 1; second < unique.size(); ++second) {
                            if (family_data[unique[first].first]
                                    == family_data[unique[second].first]
                                || !tangents[second].valid)
                                continue;
                            const double dot = tangents[first].z * tangents[second].z
                                + tangents[first].y * tangents[second].y
                                + tangents[first].x * tangents[second].x;
                            if (std::abs(dot) > angle_cutoff)
                                continue;
                            local_events.push_back({
                                unique[first].first, unique[second].first,
                                unique[first].second, unique[second].second});
                        }
                    }
                    position = group_end;
                }
            }
        }

        size_t added = 0;
        for (const auto& events : batch_events)
            added += events.size();
        const size_t required = result.events.size() + added;
        if (required > result.events.capacity()) {
            const size_t grown = result.events.capacity()
                + result.events.capacity() / 2;
            result.events.reserve(std::max(required, grown));
        }
        for (auto& events : batch_events) {
            result.events.insert(
                result.events.end(),
                std::make_move_iterator(events.begin()),
                std::make_move_iterator(events.end()));
        }
        for (size_t task = batch_begin; task < batch_end; ++task)
            completed += tasks[task].second - tasks[task].first;
        report(progress, "finding exact crossings", completed, point_count);
    }
    return result;
}

nb::dict consolidate_crossing_events(
    EventBuffer& buffer, Coordinates coordinates, Int64Vector offsets,
    UInt64Vector source_ids, int workers, const nb::object& progress)
{
    workers = effective_workers(workers);
    const size_t track_count = source_ids.shape(0);
    const size_t point_count = coordinates.shape(0);
    if (offsets.shape(0) != track_count + 1
        || static_cast<uint64_t>(offsets(track_count)) != point_count)
        throw std::runtime_error("track offsets do not match coordinates");

    auto event_less = [](const Event& left, const Event& right) {
        return std::tie(left.first, left.second,
                        left.first_local, left.second_local)
            < std::tie(right.first, right.second,
                       right.first_local, right.second_local);
    };
    report(progress, "sorting crossing events", 0, buffer.events.size());
    {
        nb::gil_scoped_release release;
#if defined(__GLIBCXX__) && defined(_OPENMP)
        const int previous_workers = omp_get_max_threads();
        omp_set_num_threads(workers);
        __gnu_parallel::sort(
            buffer.events.begin(), buffer.events.end(), event_less,
            __gnu_parallel::balanced_quicksort_tag());
        omp_set_num_threads(previous_workers);
#else
        std::sort(buffer.events.begin(), buffer.events.end(), event_less);
#endif
    }
    report(progress, "sorting crossing events",
           buffer.events.size(), buffer.events.size());

    std::vector<PairRange> ranges;
    ranges.reserve(buffer.events.size());
    for (size_t begin = 0; begin < buffer.events.size();) {
        size_t end = begin + 1;
        while (end < buffer.events.size()
               && buffer.events[end].first == buffer.events[begin].first
               && buffer.events[end].second == buffer.events[begin].second)
            ++end;
        ranges.push_back({begin, end});
        begin = end;
    }

    report(progress, "computing track arclengths", 0, track_count);
    auto arclength = std::make_unique_for_overwrite<double[]>(point_count);
    const int32_t* coordinate_data = coordinates.data();
    const int64_t* offset_data = offsets.data();
    constexpr size_t track_batch = 1'000'000;
    for (size_t batch_begin = 0; batch_begin < track_count;
         batch_begin += track_batch) {
        const size_t batch_end = std::min(batch_begin + track_batch, track_count);
        {
            nb::gil_scoped_release release;
#pragma omp parallel for schedule(static) num_threads(workers)
            for (int64_t track = static_cast<int64_t>(batch_begin);
                 track < static_cast<int64_t>(batch_end); ++track) {
                const int64_t begin = offset_data[track];
                const int64_t end = offset_data[track + 1];
                if (begin == end)
                    continue;
                arclength[begin] = 0.0;
                for (int64_t point = begin + 1; point < end; ++point) {
                    const double dz = static_cast<double>(coordinate_data[3 * point])
                        - coordinate_data[3 * (point - 1)];
                    const double dy = static_cast<double>(coordinate_data[3 * point + 1])
                        - coordinate_data[3 * (point - 1) + 1];
                    const double dx = static_cast<double>(coordinate_data[3 * point + 2])
                        - coordinate_data[3 * (point - 1) + 2];
                    arclength[point] = arclength[point - 1]
                        + std::sqrt(dz * dz + dy * dy + dx * dx);
                }
            }
        }
        report(progress, "computing track arclengths", batch_end, track_count);
    }

    std::vector<PairEdge> pair_edges(ranges.size());
    uint64_t accepted_events = 0;
    report(progress, "consolidating track pairs", 0, ranges.size());
    const size_t pair_batch = std::max<size_t>(workers * 100'000, 100'000);
    for (size_t batch_begin = 0; batch_begin < ranges.size();
         batch_begin += pair_batch) {
        const size_t batch_end = std::min(batch_begin + pair_batch, ranges.size());
        uint64_t batch_accepted_events = 0;
        {
            nb::gil_scoped_release release;
#pragma omp parallel for schedule(static) num_threads(workers) reduction(+ : batch_accepted_events)
            for (int64_t range_index = static_cast<int64_t>(batch_begin);
                 range_index < static_cast<int64_t>(batch_end); ++range_index) {
                const PairRange range = ranges[range_index];
                Event best{};
                std::tuple<double, double, double, int32_t, int32_t> best_key;
                bool have_best = false;
                uint64_t representatives = 0;

                size_t cluster_begin = range.begin;
                while (cluster_begin < range.end) {
                    size_t cluster_end = cluster_begin + 1;
                    while (cluster_end < range.end
                           && std::abs(buffer.events[cluster_end].first_local
                                       - buffer.events[cluster_end - 1].first_local) <= 4
                           && std::abs(buffer.events[cluster_end].second_local
                                       - buffer.events[cluster_end - 1].second_local) <= 4)
                        ++cluster_end;
                    const Event& candidate = buffer.events[
                        cluster_begin + (cluster_end - cluster_begin) / 2];
                    const int64_t first_point = offset_data[candidate.first]
                        + candidate.first_local;
                    const int64_t second_point = offset_data[candidate.second]
                        + candidate.second_local;
                    const double first_position = arclength[first_point];
                    const double second_position = arclength[second_point];
                    const double clearance = std::min({
                        first_position,
                        arclength[offset_data[candidate.first + 1] - 1]
                            - first_position,
                        second_position,
                        arclength[offset_data[candidate.second + 1] - 1]
                            - second_position,
                    });
                    const auto key = std::make_tuple(
                        clearance, first_position, second_position,
                        candidate.first_local, candidate.second_local);
                    if (!have_best || key > best_key) {
                        have_best = true;
                        best = candidate;
                        best_key = key;
                    }
                    ++representatives;
                    cluster_begin = cluster_end;
                }
                batch_accepted_events += representatives;
                pair_edges[range_index] = {
                    best.first, best.second, best.first_local, best.second_local,
                    std::get<1>(best_key), std::get<2>(best_key),
                    std::get<0>(best_key)};
            }
        }
        accepted_events += batch_accepted_events;
        report(progress, "consolidating track pairs", batch_end, ranges.size());
    }

    std::vector<Event>().swap(buffer.events);
    std::vector<PairRange>().swap(ranges);
    arclength.reset();

    report(progress, "encoding crossing CSR", 0, pair_edges.size());
    std::vector<int64_t> csr_offsets(track_count + 1, 0);
    for (const PairEdge& edge : pair_edges) {
        ++csr_offsets[static_cast<size_t>(edge.first) + 1];
        ++csr_offsets[static_cast<size_t>(edge.second) + 1];
    }
    for (size_t track = 0; track < track_count; ++track)
        csr_offsets[track + 1] += csr_offsets[track];

    const size_t partner_count = static_cast<size_t>(csr_offsets.back());
    std::vector<int32_t> partners(partner_count);
    std::vector<int32_t> self_local(partner_count);
    std::vector<int32_t> partner_local(partner_count);
    std::vector<double> positions(partner_count);
    std::vector<double> clearances(partner_count);
    std::vector<int64_t> cursor(csr_offsets.begin(), csr_offsets.end() - 1);

    constexpr size_t encode_batch = 2'000'000;
    for (size_t begin = 0; begin < pair_edges.size(); begin += encode_batch) {
        const size_t end = std::min(begin + encode_batch, pair_edges.size());
        for (size_t index = begin; index < end; ++index) {
            const PairEdge& edge = pair_edges[index];
            const size_t first_slot = static_cast<size_t>(cursor[edge.first]++);
            partners[first_slot] = edge.second;
            self_local[first_slot] = edge.first_local;
            partner_local[first_slot] = edge.second_local;
            positions[first_slot] = edge.first_position;
            clearances[first_slot] = edge.clearance;

            const size_t second_slot = static_cast<size_t>(cursor[edge.second]++);
            partners[second_slot] = edge.first;
            self_local[second_slot] = edge.second_local;
            partner_local[second_slot] = edge.first_local;
            positions[second_slot] = edge.second_position;
            clearances[second_slot] = edge.clearance;
        }
        report(progress, "encoding crossing CSR", end, pair_edges.size());
    }

    uint64_t paired_track_count = 0;
    for (size_t track = 0; track < track_count; ++track) {
        if (csr_offsets[track + 1] != csr_offsets[track])
            ++paired_track_count;
    }
    nb::dict result;
    result["source_ids"] = source_ids;
    result["offsets"] = own_1d(std::move(csr_offsets));
    result["partners"] = own_1d(std::move(partners));
    result["self_local"] = own_1d(std::move(self_local));
    result["partner_local"] = own_1d(std::move(partner_local));
    result["positions"] = own_1d(std::move(positions));
    result["clearances"] = own_1d(std::move(clearances));
    result["accepted_events"] = accepted_events;
    result["paired_tracks"] = paired_track_count;
    return result;
}

struct PartnerCandidate {
    int32_t partner;
    int32_t self_local;
    int32_t partner_local;
    double position;
    double clearance;
};

std::vector<size_t> select_spaced_candidates(
    const std::vector<PartnerCandidate>& candidates, size_t maximum)
{
    std::vector<size_t> selected;
    if (candidates.empty() || maximum == 0)
        return selected;
    maximum = std::min(maximum, candidates.size());
    selected.reserve(maximum);
    std::vector<uint8_t> used(candidates.size(), 0);

    if (maximum == 1 || candidates.size() == 1) {
        size_t best = 0;
        for (size_t index = 1; index < candidates.size(); ++index) {
            const auto& candidate = candidates[index];
            const auto& current = candidates[best];
            if (candidate.clearance > current.clearance
                || (candidate.clearance == current.clearance
                    && candidate.partner < current.partner))
                best = index;
        }
        selected.push_back(best);
        return selected;
    }

    size_t first = 0;
    for (size_t index = 1; index < candidates.size(); ++index) {
        const auto& candidate = candidates[index];
        const auto& current = candidates[first];
        if (candidate.position < current.position
            || (candidate.position == current.position
                && (candidate.clearance > current.clearance
                    || (candidate.clearance == current.clearance
                        && candidate.partner < current.partner))))
            first = index;
    }
    selected.push_back(first);
    used[first] = 1;

    size_t second = std::numeric_limits<size_t>::max();
    double second_distance = -1.0;
    for (size_t index = 0; index < candidates.size(); ++index) {
        if (used[index])
            continue;
        const double distance = std::abs(
            candidates[index].position - candidates[first].position);
        if (second == std::numeric_limits<size_t>::max()
            || distance > second_distance
            || (distance == second_distance
                && (candidates[index].clearance > candidates[second].clearance
                    || (candidates[index].clearance
                            == candidates[second].clearance
                        && candidates[index].partner
                            < candidates[second].partner)))) {
            second = index;
            second_distance = distance;
        }
    }
    selected.push_back(second);
    used[second] = 1;

    while (selected.size() < maximum) {
        size_t choice = std::numeric_limits<size_t>::max();
        double choice_distance = -1.0;
        for (size_t index = 0; index < candidates.size(); ++index) {
            if (used[index])
                continue;
            double distance = std::numeric_limits<double>::infinity();
            for (size_t chosen : selected) {
                distance = std::min(distance, std::abs(
                    candidates[index].position - candidates[chosen].position));
            }
            if (choice == std::numeric_limits<size_t>::max()
                || distance > choice_distance
                || (distance == choice_distance
                    && (candidates[index].clearance
                            > candidates[choice].clearance
                        || (candidates[index].clearance
                                == candidates[choice].clearance
                            && candidates[index].partner
                                < candidates[choice].partner)))) {
                choice = index;
                choice_distance = distance;
            }
        }
        if (choice == std::numeric_limits<size_t>::max())
            break;
        selected.push_back(choice);
        used[choice] = 1;
    }
    return selected;
}

nb::dict materialize_partner_table(
    UInt64Vector cached_source_ids, Int64Vector offsets,
    Int32Vector partners, Int32Vector self_local,
    Int32Vector partner_local, Float64Vector positions,
    Float64Vector clearances, UInt64Vector selected_source_ids,
    int maximum, int workers, const nb::object& progress)
{
    workers = effective_workers(workers);
    if (maximum < 0)
        throw std::runtime_error("maximum must be non-negative");
    const size_t cached_tracks = cached_source_ids.shape(0);
    const size_t selected_tracks = selected_source_ids.shape(0);
    if (selected_tracks > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("partner tables support at most INT32_MAX tracks");
    if (offsets.shape(0) != cached_tracks + 1 || offsets(0) != 0)
        throw std::runtime_error("crossing CSR offsets have an invalid shape");
    const int64_t edge_count = offsets(cached_tracks);
    if (edge_count < 0
        || partners.shape(0) != static_cast<size_t>(edge_count)
        || self_local.shape(0) != static_cast<size_t>(edge_count)
        || partner_local.shape(0) != static_cast<size_t>(edge_count)
        || positions.shape(0) != static_cast<size_t>(edge_count)
        || clearances.shape(0) != static_cast<size_t>(edge_count))
        throw std::runtime_error("crossing CSR arrays are not parallel");
    for (size_t index = 1; index < cached_tracks; ++index) {
        if (cached_source_ids(index) <= cached_source_ids(index - 1))
            throw std::runtime_error("cached source ids must be strictly increasing");
    }
    for (size_t index = 1; index < selected_tracks; ++index) {
        if (selected_source_ids(index) <= selected_source_ids(index - 1))
            throw std::runtime_error("selected source ids must be strictly increasing");
    }

    std::vector<int32_t> selected_rows(selected_tracks);
    std::vector<int32_t> global_to_local(cached_tracks, -1);
    for (size_t index = 0; index < selected_tracks; ++index) {
        const uint64_t source = selected_source_ids(index);
        const uint64_t* found = std::lower_bound(
            cached_source_ids.data(), cached_source_ids.data() + cached_tracks,
            source);
        if (found == cached_source_ids.data() + cached_tracks || *found != source)
            throw std::runtime_error(
                "crossing cache does not contain every selected track");
        const size_t row = static_cast<size_t>(found - cached_source_ids.data());
        selected_rows[index] = static_cast<int32_t>(row);
        global_to_local[row] = static_cast<int32_t>(index);
    }

    const size_t width = static_cast<size_t>(maximum);
    if (width != 0 && selected_tracks > std::numeric_limits<size_t>::max() / width)
        throw std::runtime_error("partner table dimensions overflow size_t");
    const size_t output_count = selected_tracks * width;
    std::vector<int32_t> output_partners(output_count, -1);
    std::vector<int32_t> output_self_local(output_count, -1);
    std::vector<int32_t> output_partner_local(output_count, -1);
    std::atomic<uint64_t> selected_slots{0};

    report(progress, "selecting crossing partners", 0, selected_tracks);
    {
        nb::gil_scoped_release release;
#pragma omp parallel num_threads(workers)
        {
            std::vector<PartnerCandidate> candidates;
#pragma omp for schedule(dynamic, 4096)
            for (int64_t local_row = 0;
                 local_row < static_cast<int64_t>(selected_tracks); ++local_row) {
                const int32_t cached_row = selected_rows[local_row];
                const int64_t begin = offsets(cached_row);
                const int64_t end = offsets(cached_row + 1);
                candidates.clear();
                candidates.reserve(static_cast<size_t>(end - begin));
                for (int64_t edge = begin; edge < end; ++edge) {
                    const int32_t global_partner = partners(edge);
                    if (global_partner < 0
                        || static_cast<size_t>(global_partner) >= cached_tracks)
                        continue;
                    const int32_t local_partner = global_to_local[global_partner];
                    if (local_partner < 0)
                        continue;
                    candidates.push_back({
                        local_partner, self_local(edge), partner_local(edge),
                        positions(edge), clearances(edge)});
                }
                const auto chosen = select_spaced_candidates(candidates, width);
                const size_t destination = static_cast<size_t>(local_row) * width;
                for (size_t slot = 0; slot < chosen.size(); ++slot) {
                    const auto& candidate = candidates[chosen[slot]];
                    output_partners[destination + slot] = candidate.partner;
                    output_self_local[destination + slot] = candidate.self_local;
                    output_partner_local[destination + slot]
                        = candidate.partner_local;
                }
                selected_slots.fetch_add(chosen.size(), std::memory_order_relaxed);
            }
        }
    }
    report(progress, "selecting crossing partners", selected_tracks, selected_tracks);

    nb::dict result;
    result["partners"] = own_2d(
        std::move(output_partners), selected_tracks, width);
    result["self_local"] = own_2d(
        std::move(output_self_local), selected_tracks, width);
    result["partner_local"] = own_2d(
        std::move(output_partner_local), selected_tracks, width);
    result["selected_slots"] = selected_slots.load(std::memory_order_relaxed);
    return result;
}

std::vector<double> cumulative_arclengths(
    const float* coordinates, int64_t begin, int64_t end)
{
    std::vector<double> cumulative(static_cast<size_t>(end - begin), 0.0);
    for (int64_t point = begin + 1; point < end; ++point) {
        const double dz = static_cast<double>(coordinates[3 * point])
            - coordinates[3 * (point - 1)];
        const double dy = static_cast<double>(coordinates[3 * point + 1])
            - coordinates[3 * (point - 1) + 1];
        const double dx = static_cast<double>(coordinates[3 * point + 2])
            - coordinates[3 * (point - 1) + 2];
        cumulative[static_cast<size_t>(point - begin)]
            = cumulative[static_cast<size_t>(point - begin - 1)]
            + std::sqrt(dz * dz + dy * dy + dx * dx);
    }
    return cumulative;
}

std::vector<int32_t> track_anchor_indices(
    int64_t length, const int32_t* anchors, int64_t anchor_count)
{
    std::vector<int32_t> result;
    if (length <= 0)
        return result;
    result.reserve(static_cast<size_t>(anchor_count) + 2);
    result.push_back(0);
    for (int64_t index = 0; index < anchor_count; ++index) {
        if (anchors[index] >= 0 && anchors[index] < length)
            result.push_back(anchors[index]);
    }
    result.push_back(static_cast<int32_t>(length - 1));
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::vector<double> anchor_arclengths(
    const std::vector<int32_t>& anchor_indices,
    const std::vector<double>& cumulative)
{
    std::vector<double> result;
    result.reserve(anchor_indices.size());
    for (int32_t local : anchor_indices) {
        const double position = cumulative[static_cast<size_t>(local)];
        if (result.empty() || position != result.back())
            result.push_back(position);
    }
    return result;
}

int64_t resampled_track_length(
    const std::vector<double>& anchors, double maximum_spacing)
{
    if (anchors.empty())
        return 0;
    int64_t count = 1;
    for (size_t index = 1; index < anchors.size(); ++index) {
        const double span = anchors[index] - anchors[index - 1];
        count += std::max<int64_t>(
            1, static_cast<int64_t>(std::ceil(span / maximum_spacing)));
    }
    return count;
}

nb::dict resample_tracks(
    FloatCoordinates coordinates, Int64Vector offsets,
    Int32Matrix crossing_partners, Int32Matrix crossing_self_local,
    Int32Matrix crossing_partner_local, double minimum_spacing,
    double maximum_spacing, int workers, const nb::object& progress)
{
    workers = effective_workers(workers);
    if (!(minimum_spacing > 0.0) || !(maximum_spacing > 0.0)
        || minimum_spacing > maximum_spacing
        || !std::isfinite(minimum_spacing)
        || !std::isfinite(maximum_spacing))
        throw std::runtime_error(
            "sample spacing must be finite, positive, and ordered");
    if (offsets.shape(0) == 0)
        throw std::runtime_error("track offsets must contain a zero sentinel");
    const size_t track_count = offsets.shape(0) - 1;
    const size_t point_count = coordinates.shape(0);
    if (offsets(0) != 0
        || offsets(track_count) != static_cast<int64_t>(point_count))
        throw std::runtime_error("track offsets do not match coordinates");
    for (size_t track = 0; track < track_count; ++track) {
        if (offsets(track + 1) < offsets(track))
            throw std::runtime_error("track offsets must be monotonic");
        if (offsets(track + 1) - offsets(track)
            > std::numeric_limits<int32_t>::max())
            throw std::runtime_error("a track exceeds INT32_MAX points");
    }
    const size_t width = crossing_partners.shape(1);
    if (crossing_partners.shape(0) != track_count
        || crossing_self_local.shape(0) != track_count
        || crossing_partner_local.shape(0) != track_count
        || crossing_self_local.shape(1) != width
        || crossing_partner_local.shape(1) != width)
        throw std::runtime_error("crossing tables must have equal shapes");

    const float* coordinate_data = coordinates.data();
    const int64_t* offset_data = offsets.data();
    const int32_t* partner_data = crossing_partners.data();
    const int32_t* self_local_data = crossing_self_local.data();
    const int32_t* partner_local_data = crossing_partner_local.data();

    // Build a compact CSR of every local index that must survive resampling.
    // Entries may repeat, but sorting each row lets lookups remain allocation-free.
    std::vector<uint32_t> anchor_counts(track_count, 0);
    if (width > 0) {
        nb::gil_scoped_release release;
#pragma omp parallel for schedule(static) num_threads(workers)
        for (int64_t track = 0; track < static_cast<int64_t>(track_count); ++track) {
            for (size_t slot = 0; slot < width; ++slot) {
                const size_t table_index = static_cast<size_t>(track) * width + slot;
                const int32_t partner = partner_data[table_index];
                if (partner < 0)
                    continue;
                if (static_cast<size_t>(partner) >= track_count)
                    continue;
#pragma omp atomic update
                anchor_counts[static_cast<size_t>(track)]++;
#pragma omp atomic update
                anchor_counts[static_cast<size_t>(partner)]++;
            }
        }
    }
    std::vector<int64_t> anchor_offsets(track_count + 1, 0);
    for (size_t track = 0; track < track_count; ++track)
        anchor_offsets[track + 1] = anchor_offsets[track] + anchor_counts[track];
    std::vector<int32_t> anchors(static_cast<size_t>(anchor_offsets.back()));
    std::vector<int64_t> anchor_cursor(
        anchor_offsets.begin(), anchor_offsets.end() - 1);
    if (width > 0) {
        nb::gil_scoped_release release;
#pragma omp parallel for schedule(static) num_threads(workers)
        for (int64_t track = 0; track < static_cast<int64_t>(track_count); ++track) {
            for (size_t slot = 0; slot < width; ++slot) {
                const size_t table_index = static_cast<size_t>(track) * width + slot;
                const int32_t partner = partner_data[table_index];
                if (partner < 0 || static_cast<size_t>(partner) >= track_count)
                    continue;
                int64_t self_destination;
                int64_t partner_destination;
#pragma omp atomic capture
                self_destination = anchor_cursor[static_cast<size_t>(track)]++;
#pragma omp atomic capture
                partner_destination = anchor_cursor[static_cast<size_t>(partner)]++;
                anchors[static_cast<size_t>(self_destination)]
                    = self_local_data[table_index];
                anchors[static_cast<size_t>(partner_destination)]
                    = partner_local_data[table_index];
            }
        }
#pragma omp parallel for schedule(dynamic, 4096) num_threads(workers)
        for (int64_t track = 0; track < static_cast<int64_t>(track_count); ++track) {
            std::sort(
                anchors.begin() + anchor_offsets[static_cast<size_t>(track)],
                anchors.begin() + anchor_offsets[static_cast<size_t>(track) + 1]);
        }
    }
    std::vector<int64_t>().swap(anchor_cursor);
    std::vector<uint32_t>().swap(anchor_counts);

    std::vector<int64_t> sampled_lengths(track_count, 0);
    report(progress, "counting resampled track points", 0, track_count);
    {
        nb::gil_scoped_release release;
#pragma omp parallel for schedule(dynamic, 4096) num_threads(workers)
        for (int64_t track = 0; track < static_cast<int64_t>(track_count); ++track) {
            const int64_t begin = offset_data[track];
            const int64_t end = offset_data[track + 1];
            if (begin == end)
                continue;
            const auto cumulative = cumulative_arclengths(
                coordinate_data, begin, end);
            if (cumulative.back() <= 0.0) {
                sampled_lengths[static_cast<size_t>(track)] = 1;
                continue;
            }
            const int64_t anchor_begin = anchor_offsets[static_cast<size_t>(track)];
            const int64_t anchor_end = anchor_offsets[static_cast<size_t>(track) + 1];
            const auto local_anchors = track_anchor_indices(
                end - begin, anchors.data() + anchor_begin,
                anchor_end - anchor_begin);
            const auto positions = anchor_arclengths(local_anchors, cumulative);
            sampled_lengths[static_cast<size_t>(track)]
                = resampled_track_length(positions, maximum_spacing);
        }
    }
    report(progress, "counting resampled track points", track_count, track_count);

    std::vector<int64_t> sampled_offsets(track_count + 1, 0);
    for (size_t track = 0; track < track_count; ++track)
        sampled_offsets[track + 1]
            = sampled_offsets[track] + sampled_lengths[track];
    const size_t sampled_count = static_cast<size_t>(sampled_offsets.back());
    if (sampled_count > std::numeric_limits<size_t>::max() / 3)
        throw std::runtime_error("resampled coordinate count overflows size_t");
    std::vector<float> sampled_coordinates(sampled_count * 3);
    std::vector<int64_t> sampled_source_local(sampled_count);
    std::vector<int32_t> anchor_samples(anchors.size(), -1);
    double minimum_observed = std::numeric_limits<double>::infinity();
    double maximum_observed = 0.0;
    uint64_t undersized_gaps = 0;

    report(progress, "resampling tracks", 0, track_count);
    {
        nb::gil_scoped_release release;
#pragma omp parallel for schedule(dynamic, 1024) num_threads(workers) \
    reduction(min : minimum_observed) reduction(max : maximum_observed) \
    reduction(+ : undersized_gaps)
        for (int64_t track = 0; track < static_cast<int64_t>(track_count); ++track) {
            const int64_t begin = offset_data[track];
            const int64_t end = offset_data[track + 1];
            const int64_t output_begin = sampled_offsets[static_cast<size_t>(track)];
            if (begin == end)
                continue;
            const auto cumulative = cumulative_arclengths(
                coordinate_data, begin, end);
            if (cumulative.back() <= 0.0) {
                for (size_t axis = 0; axis < 3; ++axis)
                    sampled_coordinates[3 * static_cast<size_t>(output_begin) + axis]
                        = coordinate_data[3 * static_cast<size_t>(begin) + axis];
                sampled_source_local[static_cast<size_t>(output_begin)] = 0;
                continue;
            }
            const int64_t anchor_begin = anchor_offsets[static_cast<size_t>(track)];
            const int64_t anchor_end = anchor_offsets[static_cast<size_t>(track) + 1];
            const auto local_anchors = track_anchor_indices(
                end - begin, anchors.data() + anchor_begin,
                anchor_end - anchor_begin);
            const auto anchor_positions = anchor_arclengths(
                local_anchors, cumulative);
            std::vector<double> positions;
            positions.reserve(static_cast<size_t>(
                sampled_lengths[static_cast<size_t>(track)]));
            for (size_t segment = 1; segment < anchor_positions.size(); ++segment) {
                const double left = anchor_positions[segment - 1];
                const double right = anchor_positions[segment];
                const double span = right - left;
                const int64_t intervals = std::max<int64_t>(
                    1, static_cast<int64_t>(std::ceil(span / maximum_spacing)));
                const int64_t allowed_by_minimum = static_cast<int64_t>(
                    std::floor(span / minimum_spacing));
                const bool feasible = intervals <= allowed_by_minimum;
                const double step = feasible
                    ? span / static_cast<double>(intervals)
                    : maximum_spacing;
                for (int64_t interval = 0; interval < intervals; ++interval)
                    positions.push_back(left + static_cast<double>(interval) * step);
            }
            positions.push_back(anchor_positions.back());

            for (size_t output_local = 0; output_local < positions.size(); ++output_local) {
                const double position = positions[output_local];
                auto found = std::upper_bound(
                    cumulative.begin(), cumulative.end(), position);
                size_t right = static_cast<size_t>(found - cumulative.begin());
                right = std::clamp<size_t>(right, 1, cumulative.size() - 1);
                const size_t left = right - 1;
                const double denominator = cumulative[right] - cumulative[left];
                const double alpha = denominator > 0.0
                    ? (position - cumulative[left]) / denominator : 0.0;
                const size_t output = static_cast<size_t>(output_begin) + output_local;
                for (size_t axis = 0; axis < 3; ++axis) {
                    sampled_coordinates[3 * output + axis] = static_cast<float>(
                        coordinate_data[3 * (static_cast<size_t>(begin) + left) + axis]
                            * (1.0 - alpha)
                        + coordinate_data[3 * (static_cast<size_t>(begin) + right) + axis]
                            * alpha);
                }
                sampled_source_local[output] =
                    std::abs(cumulative[right] - position)
                        < std::abs(position - cumulative[left])
                    ? static_cast<int64_t>(right) : static_cast<int64_t>(left);
                if (output_local > 0) {
                    const double observed = position - positions[output_local - 1];
                    minimum_observed = std::min(minimum_observed, observed);
                    maximum_observed = std::max(maximum_observed, observed);
                    undersized_gaps += observed < minimum_spacing - 1.e-9;
                }
            }

            for (int64_t anchor = anchor_begin; anchor < anchor_end; ++anchor) {
                const int32_t local = anchors[static_cast<size_t>(anchor)];
                if (local < 0 || static_cast<size_t>(local) >= cumulative.size())
                    continue;
                const double position = cumulative[static_cast<size_t>(local)];
                auto found = std::lower_bound(
                    positions.begin(), positions.end(), position);
                size_t sample = static_cast<size_t>(found - positions.begin());
                if (sample == positions.size())
                    sample = positions.size() - 1;
                else if (sample > 0
                    && std::abs(positions[sample - 1] - position)
                        <= std::abs(positions[sample] - position))
                    --sample;
                anchor_samples[static_cast<size_t>(anchor)]
                    = static_cast<int32_t>(sample);
            }
        }
    }
    report(progress, "resampling tracks", track_count, track_count);

    std::vector<int32_t> crossing_self_sample(track_count * width, -1);
    std::vector<int32_t> crossing_partner_sample(track_count * width, -1);
    if (width > 0) {
        nb::gil_scoped_release release;
#pragma omp parallel for schedule(static) num_threads(workers)
        for (int64_t track = 0; track < static_cast<int64_t>(track_count); ++track) {
            for (size_t slot = 0; slot < width; ++slot) {
                const size_t table_index = static_cast<size_t>(track) * width + slot;
                const int32_t partner = partner_data[table_index];
                if (partner < 0 || static_cast<size_t>(partner) >= track_count)
                    continue;
                auto lookup = [&](int32_t row, int32_t local) {
                    const int64_t begin = anchor_offsets[static_cast<size_t>(row)];
                    const int64_t end = anchor_offsets[static_cast<size_t>(row) + 1];
                    const int32_t* found = std::lower_bound(
                        anchors.data() + begin, anchors.data() + end, local);
                    if (found == anchors.data() + end || *found != local)
                        return int32_t{-1};
                    return anchor_samples[static_cast<size_t>(found - anchors.data())];
                };
                crossing_self_sample[table_index] = lookup(
                    static_cast<int32_t>(track), self_local_data[table_index]);
                crossing_partner_sample[table_index] = lookup(
                    partner, partner_local_data[table_index]);
            }
        }
    }

    nb::dict result;
    result["coordinates"] = own_2d(
        std::move(sampled_coordinates), sampled_count, 3);
    result["source_local"] = own_1d(std::move(sampled_source_local));
    result["offsets"] = own_1d(std::move(sampled_offsets));
    result["lengths"] = own_1d(std::move(sampled_lengths));
    result["crossing_self_sample"] = own_2d(
        std::move(crossing_self_sample), track_count, width);
    result["crossing_partner_sample"] = own_2d(
        std::move(crossing_partner_sample), track_count, width);
    result["minimum_observed_spacing"] = minimum_observed;
    result["maximum_observed_spacing"] = maximum_observed;
    result["undersized_anchor_gaps"] = undersized_gaps;
    return result;
}

} // namespace

NB_MODULE(track_crossings, module)
{
    module.doc() = "Memory-efficient native exact track-crossing construction.";
    nb::class_<EventBuffer>(module, "EventBuffer")
        .def_prop_ro("event_count", &EventBuffer::size)
        .def_prop_ro("memory_bytes", &EventBuffer::memory_bytes);
    module.def(
        "parallel_argsort", &parallel_argsort,
        nb::arg("packed"), nb::arg("workers") = 1,
        nb::arg("progress") = nb::none());
    module.def(
        "scan_crossing_events", &scan_crossing_events,
        nb::arg("coordinates"), nb::arg("offsets"), nb::arg("family_codes"),
        nb::arg("packed"), nb::arg("order"), nb::arg("workers") = 1,
        nb::arg("progress") = nb::none());
    module.def(
        "consolidate_crossing_events", &consolidate_crossing_events,
        nb::arg("events"), nb::arg("coordinates"), nb::arg("offsets"),
        nb::arg("source_ids"), nb::arg("workers") = 1,
        nb::arg("progress") = nb::none());
    module.def(
        "materialize_partner_table", &materialize_partner_table,
        nb::arg("cached_source_ids"), nb::arg("offsets"),
        nb::arg("partners"), nb::arg("self_local"),
        nb::arg("partner_local"), nb::arg("positions"),
        nb::arg("clearances"), nb::arg("selected_source_ids"),
        nb::arg("maximum"), nb::arg("workers") = 1,
        nb::arg("progress") = nb::none());
    module.def(
        "resample_tracks", &resample_tracks,
        nb::arg("coordinates"), nb::arg("offsets"),
        nb::arg("crossing_partners"), nb::arg("crossing_self_local"),
        nb::arg("crossing_partner_local"),
        nb::arg("minimum_spacing"), nb::arg("maximum_spacing"),
        nb::arg("workers") = 1, nb::arg("progress") = nb::none());
}
