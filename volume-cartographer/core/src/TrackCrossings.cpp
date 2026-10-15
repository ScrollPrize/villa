#include "vc/core/util/TrackCrossings.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__GLIBCXX__) && defined(_OPENMP)
#include <parallel/algorithm>
#include <parallel/tags.h>
#endif
namespace vc::track_crossings {
namespace {
struct PairRange { size_t begin, end; };
struct PairEdge {
    int32_t first, second, first_local, second_local;
    double first_position, second_position, clearance;
};
void report(const Progress& callback, const char* phase, uint64_t done, uint64_t total)
{
    if (callback) callback(phase, done, total);
}
} // namespace
void validateParameters(const Parameters& p)
{
    if (!std::isfinite(p.minAngleDegrees) || p.minAngleDegrees < 0 || p.minAngleDegrees > 90
        || !std::isfinite(p.tangentRadius) || p.tangentRadius <= 0 || p.clusterIndexRadius < 0)
        throw std::invalid_argument("invalid track crossing angle, tangent radius, or cluster radius");
}
void validateTracks(std::span<const int32_t> coordinates, std::span<const int64_t> offsets)
{
    if (coordinates.size() % 3 || offsets.empty() || offsets.front() != 0
        || offsets.back() < 0 || uint64_t(offsets.back()) != coordinates.size() / 3)
        throw std::invalid_argument("track offsets do not match ZYX coordinates");
    if (offsets.size() - 1 > size_t(std::numeric_limits<int32_t>::max()))
        throw std::invalid_argument("too many tracks for crossing indices");
    for (size_t i = 1; i < offsets.size(); ++i)
        if (offsets[i] < offsets[i-1] || offsets[i] - offsets[i-1] > std::numeric_limits<int32_t>::max())
            throw std::invalid_argument("invalid track offset or track too long for point indices");
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

std::vector<uint32_t> parallel_argsort(
    std::span<const uint64_t> packed, int workers, const Progress& progress)
{
    workers = effective_workers(workers);
    const size_t count = packed.size();
    if (count > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
        throw std::runtime_error(
            "native packed-key radix sort supports at most UINT32_MAX points");

    for (uint64_t key : packed)
        if (key >= (uint64_t{1} << 60))
            throw std::invalid_argument("packed voxel keys must fit in 60 bits");

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

#pragma omp parallel for schedule(static) num_threads(workers)
            for (int thread = 0; thread < workers; ++thread) {
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

#pragma omp parallel for schedule(static) num_threads(workers)
            for (int thread = 0; thread < workers; ++thread) {
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
    return order_a;
}

Tangent track_tangent(const int32_t* coordinates, const int64_t* offsets,
                      int32_t track, int32_t local, double radius)
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
    std::span<const int32_t> coordinates, std::span<const int64_t> offsets,
    std::span<const int8_t> family_codes, std::span<const uint64_t> packed,
    std::span<const uint32_t> order, int workers, const Progress& progress,
    const Parameters& params)
{
    workers = effective_workers(workers);
    validateParameters(params);
    validateTracks(coordinates, offsets);
    const size_t point_count = order.size();
    const size_t track_count = family_codes.size();
    if ((coordinates.size() / 3) != point_count || packed.size() != point_count)
        throw std::runtime_error("coordinates, packed keys, and order must be parallel");
    if (offsets.size() != track_count + 1
        || static_cast<uint64_t>(offsets[track_count]) != point_count)
        throw std::runtime_error("track offsets do not match coordinates");
    if (track_count > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("native crossing scan supports at most INT32_MAX tracks");

    for (uint32_t index : order)
        if (index >= point_count)
            throw std::invalid_argument("crossing sort index is outside coordinates");
    for (int8_t family : family_codes)
        if (family < -1 || family > 1)
            throw std::invalid_argument("crossing family code must be -1, 0 or 1");

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
    const double angle_cutoff = std::cos(params.minAngleDegrees * std::acos(-1.0) / 180.0);
    uint64_t completed = 0;
    report(progress, "finding exact crossings", 0, point_count);

    for (size_t batch_begin = 0; batch_begin < tasks.size();
         batch_begin += tasks_per_batch) {
        const size_t batch_end = std::min(
            batch_begin + tasks_per_batch, tasks.size());
        std::vector<std::vector<Event>> batch_events(batch_end - batch_begin);

        {

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
                            unique[index].first, unique[index].second, params.tangentRadius);
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

Crossings consolidate_crossing_events(
    EventBuffer& buffer, std::span<const int32_t> coordinates, std::span<const int64_t> offsets,
    std::span<const uint64_t> source_ids, int workers, const Progress& progress,
    const Parameters& params)
{
    workers = effective_workers(workers);
    validateParameters(params);
    validateTracks(coordinates, offsets);
    const size_t track_count = source_ids.size();
    const size_t point_count = (coordinates.size() / 3);
    if (offsets.size() != track_count + 1
        || static_cast<uint64_t>(offsets[track_count]) != point_count)
        throw std::runtime_error("track offsets do not match coordinates");

    for (const auto& event : buffer.events) {
        if (event.first < 0 || event.second <= event.first
            || static_cast<size_t>(event.second) >= track_count
            || event.first_local < 0 || event.second_local < 0
            || event.first_local >= offsets[event.first+1] - offsets[event.first]
            || event.second_local >= offsets[event.second+1] - offsets[event.second])
            throw std::invalid_argument("crossing event index is outside its tracks");
    }

    auto event_less = [](const Event& left, const Event& right) {
        return std::tie(left.first, left.second,
                        left.first_local, left.second_local)
            < std::tie(right.first, right.second,
                       right.first_local, right.second_local);
    };
    report(progress, "sorting crossing events", 0, buffer.events.size());
    {

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
                                       - buffer.events[cluster_end - 1].first_local) <= params.clusterIndexRadius
                           && std::abs(buffer.events[cluster_end].second_local
                                       - buffer.events[cluster_end - 1].second_local) <= params.clusterIndexRadius)
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
    return {std::move(csr_offsets), std::move(partners), std::move(self_local),
            std::move(partner_local), std::move(positions), std::move(clearances),
            accepted_events, paired_track_count};
}

} // namespace vc::track_crossings
