#include "vc/core/util/TrackCrossings.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
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

using vc::track_crossings::EventBuffer;
struct CrossingIndex {
    std::vector<int64_t> offsets;
    std::vector<int32_t> partners;
    std::vector<int32_t> self_local;
    std::vector<int32_t> partner_local;
    std::vector<int32_t> track_lengths;

    size_t track_count() const { return track_lengths.size(); }
    size_t crossing_count() const { return partners.size(); }
    size_t memory_bytes() const {
        return offsets.capacity() * sizeof(int64_t)
            + (partners.capacity() + self_local.capacity()
               + partner_local.capacity() + track_lengths.capacity())
                * sizeof(int32_t);
    }
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

vc::track_crossings::Progress crossing_progress(const nb::object& progress)
{
    return [progress](const char* phase, uint64_t done, uint64_t total) {
        nb::gil_scoped_acquire acquire;
        report(progress, phase, done, total);
    };
}
nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>> parallel_argsort(
    UInt64Vector packed, int workers, const nb::object& progress)
{
    std::vector<uint32_t> order;
    auto callback = crossing_progress(progress);
    {
        nb::gil_scoped_release release;
        order = vc::track_crossings::parallel_argsort(
            {packed.data(), packed.size()}, workers, callback);
    }
    return own_1d(std::move(order));
}
EventBuffer scan_crossing_events(
    Coordinates coordinates, Int64Vector offsets, Int8Vector family_codes,
    UInt64Vector packed, UInt32Vector order, int workers, const nb::object& progress)
{
    auto callback = crossing_progress(progress);
    nb::gil_scoped_release release;
    return vc::track_crossings::scan_crossing_events(
        {coordinates.data(), coordinates.size()}, {offsets.data(), offsets.size()},
        {family_codes.data(), family_codes.size()}, {packed.data(), packed.size()},
        {order.data(), order.size()}, workers, callback);
}
nb::dict consolidate_crossing_events(
    EventBuffer& buffer, Coordinates coordinates, Int64Vector offsets,
    UInt64Vector source_ids, int workers, const nb::object& progress)
{
    vc::track_crossings::Crossings crossings;
    auto callback = crossing_progress(progress);
    {
        nb::gil_scoped_release release;
        crossings = vc::track_crossings::consolidate_crossing_events(
            buffer, {coordinates.data(), coordinates.size()}, {offsets.data(), offsets.size()},
            {source_ids.data(), source_ids.size()}, workers, callback);
    }
    nb::dict result;
    result["source_ids"] = source_ids;
    result["offsets"] = own_1d(std::move(crossings.offsets));
    result["partners"] = own_1d(std::move(crossings.partners));
    result["self_local"] = own_1d(std::move(crossings.self_local));
    result["partner_local"] = own_1d(std::move(crossings.partner_local));
    result["positions"] = own_1d(std::move(crossings.positions));
    result["clearances"] = own_1d(std::move(crossings.clearances));
    result["accepted_events"] = crossings.accepted_events;
    result["paired_tracks"] = crossings.paired_tracks;
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
    double minimum_spacing, double maximum_spacing, int workers,
    const nb::object& progress, const CrossingIndex* crossing_index)
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
    if (crossing_index != nullptr && crossing_index->track_count() != track_count)
        throw std::runtime_error(
            "crossing index track count does not match resampling offsets");

    const float* coordinate_data = coordinates.data();
    const int64_t* offset_data = offsets.data();

    // Build a compact CSR of every local index that must survive resampling.
    // Entries may repeat, but sorting each row lets lookups remain allocation-free.
    std::vector<uint32_t> anchor_counts(track_count, 0);
    if (crossing_index != nullptr) {
        for (size_t track = 0; track < track_count; ++track) {
            const int64_t count = crossing_index->offsets[track + 1]
                - crossing_index->offsets[track];
            if (count < 0
                || static_cast<uint64_t>(count)
                    > std::numeric_limits<uint32_t>::max())
                throw std::runtime_error(
                    "crossing anchor count exceeds the native row limit");
            anchor_counts[track] = static_cast<uint32_t>(count);
        }
    }
    std::vector<int64_t> anchor_offsets(track_count + 1, 0);
    for (size_t track = 0; track < track_count; ++track)
        anchor_offsets[track + 1] = anchor_offsets[track] + anchor_counts[track];
    std::vector<int32_t> anchors(static_cast<size_t>(anchor_offsets.back()));
    if (crossing_index != nullptr) {
        {
            nb::gil_scoped_release release;
#pragma omp parallel for schedule(dynamic, 4096) num_threads(workers)
            for (int64_t track = 0;
                 track < static_cast<int64_t>(track_count); ++track) {
                const int64_t source_begin = crossing_index->offsets[track];
                const int64_t source_end = crossing_index->offsets[track + 1];
                const int64_t destination = anchor_offsets[track];
                std::copy(
                    crossing_index->self_local.begin() + source_begin,
                    crossing_index->self_local.begin() + source_end,
                    anchors.begin() + destination);
                std::sort(
                    anchors.begin() + destination,
                    anchors.begin() + anchor_offsets[track + 1]);
            }
        }
    }
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

    std::vector<int32_t> crossing_record_sample;
    const auto record_count
        = crossing_index != nullptr ? crossing_index->crossing_count() : 0;
    if (record_count > 0) {
        crossing_record_sample.resize(record_count, -1);
        nb::gil_scoped_release release;
#pragma omp parallel for schedule(dynamic, 4096) num_threads(workers)
        for (int64_t track = 0;
                 track < static_cast<int64_t>(track_count); ++track) {
            const int64_t anchor_begin = anchor_offsets[track];
            const int64_t anchor_end = anchor_offsets[track + 1];
            const int64_t record_begin = crossing_index->offsets[track];
            const int64_t record_end = crossing_index->offsets[track + 1];
            for (int64_t record = record_begin; record < record_end; ++record) {
                const int32_t local = crossing_index->self_local[record];
                const int32_t* found = std::lower_bound(
                    anchors.data() + anchor_begin,
                    anchors.data() + anchor_end, local);
                if (found != anchors.data() + anchor_end && *found == local) {
                    crossing_record_sample[record] = anchor_samples[
                        static_cast<size_t>(found - anchors.data())];
                }
            }
        }
    }

    nb::dict result;
    result["coordinates"] = own_2d(
        std::move(sampled_coordinates), sampled_count, 3);
    result["source_local"] = own_1d(std::move(sampled_source_local));
    result["offsets"] = own_1d(std::move(sampled_offsets));
    result["lengths"] = own_1d(std::move(sampled_lengths));
    if (crossing_index != nullptr)
        result["crossing_record_sample"] = own_1d(std::move(crossing_record_sample));
    result["minimum_observed_spacing"] = minimum_observed;
    result["maximum_observed_spacing"] = maximum_observed;
    result["undersized_anchor_gaps"] = undersized_gaps;
    return result;
}

CrossingIndex* prepare_crossing_index(
    Int64Vector offsets, Int32Vector partners, Int32Vector self_local,
    Int32Vector partner_local, Int32Vector track_lengths)
{
    const size_t tracks = track_lengths.shape(0);
    if (offsets.shape(0) != tracks + 1 || offsets(0) != 0)
        throw std::runtime_error("crossing index offsets have an invalid shape");
    const int64_t records64 = offsets(tracks);
    if (records64 < 0 || partners.shape(0) != static_cast<size_t>(records64)
        || self_local.shape(0) != static_cast<size_t>(records64)
        || partner_local.shape(0) != static_cast<size_t>(records64))
        throw std::runtime_error("crossing index arrays are not parallel");
    const size_t records = static_cast<size_t>(records64);
    if (records > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error(
            "crossing index exceeds the native record id range");

    auto index = std::make_unique<CrossingIndex>();
    index->offsets.assign(offsets.data(), offsets.data() + tracks + 1);
    index->partners.assign(partners.data(), partners.data() + records);
    index->self_local.assign(self_local.data(), self_local.data() + records);
    index->partner_local.assign(
        partner_local.data(), partner_local.data() + records);
    index->track_lengths.assign(
        track_lengths.data(), track_lengths.data() + tracks);
    for (size_t track = 0; track < tracks; ++track) {
        if (index->offsets[track + 1] < index->offsets[track])
            throw std::runtime_error("crossing index offsets must be monotonic");
        for (int64_t record = index->offsets[track];
             record < index->offsets[track + 1]; ++record) {
            const int32_t partner = index->partners[record];
            if (partner < 0 || static_cast<size_t>(partner) >= tracks)
                throw std::runtime_error(
                    "crossing index partner is out of range");
            if (index->self_local[record] < 0
                || index->self_local[record] >= index->track_lengths[track]
                || index->partner_local[record] < 0
                || index->partner_local[record]
                    >= index->track_lengths[static_cast<size_t>(partner)])
                throw std::runtime_error(
                    "crossing index local point is out of range");
        }
    }
    return index.release();
}

CrossingIndex* prepare_cached_crossing_index(
    UInt64Vector cached_source_ids, Int64Vector cached_offsets,
    Int32Vector cached_partners, Int32Vector cached_self_local,
    Int32Vector cached_partner_local, UInt64Vector selected_source_ids,
    Int32Vector track_lengths)
{
    const size_t cached_tracks = cached_source_ids.shape(0);
    const size_t selected_tracks = selected_source_ids.shape(0);
    if (track_lengths.shape(0) != selected_tracks
        || cached_offsets.shape(0) != cached_tracks + 1
        || cached_offsets(0) != 0)
        throw std::runtime_error("cached crossing index arrays have invalid shapes");
    const int64_t cached_records = cached_offsets(cached_tracks);
    if (cached_records < 0
        || cached_partners.shape(0) != static_cast<size_t>(cached_records)
        || cached_self_local.shape(0) != static_cast<size_t>(cached_records)
        || cached_partner_local.shape(0) != static_cast<size_t>(cached_records))
        throw std::runtime_error("cached crossing arrays are not parallel");

    std::vector<int32_t> selected_rows(selected_tracks);
    std::vector<int32_t> global_to_local(cached_tracks, -1);
    for (size_t local = 0; local < selected_tracks; ++local) {
        const uint64_t source = selected_source_ids(local);
        const uint64_t* found = std::lower_bound(
            cached_source_ids.data(), cached_source_ids.data() + cached_tracks,
            source);
        if (found == cached_source_ids.data() + cached_tracks || *found != source)
            throw std::runtime_error(
                "crossing cache does not contain every selected track");
        const size_t row = static_cast<size_t>(
            found - cached_source_ids.data());
        selected_rows[local] = static_cast<int32_t>(row);
        global_to_local[row] = static_cast<int32_t>(local);
    }

    std::vector<int64_t> offsets(selected_tracks + 1, 0);
    std::vector<int32_t> partners, self_local, partner_local;
    for (size_t local = 0; local < selected_tracks; ++local) {
        const int32_t row = selected_rows[local];
        for (int64_t record = cached_offsets(row);
             record < cached_offsets(row + 1); ++record) {
            const int32_t global_partner = cached_partners(record);
            if (global_partner < 0
                || static_cast<size_t>(global_partner) >= cached_tracks)
                continue;
            const int32_t partner = global_to_local[global_partner];
            if (partner < 0)
                continue;
            partners.push_back(partner);
            self_local.push_back(cached_self_local(record));
            partner_local.push_back(cached_partner_local(record));
        }
        offsets[local + 1] = static_cast<int64_t>(partners.size());
    }
    Int64Vector offset_view(offsets.data(), {offsets.size()});
    Int32Vector partner_view(partners.data(), {partners.size()});
    Int32Vector self_view(self_local.data(), {self_local.size()});
    Int32Vector partner_local_view(
        partner_local.data(), {partner_local.size()});
    return prepare_crossing_index(
        offset_view, partner_view, self_view, partner_local_view, track_lengths);
}

nb::dict crossing_index_stats(const CrossingIndex& index)
{
    uint64_t connected_tracks = 0;
    uint64_t maximum_degree = 0;
    for (size_t track = 0; track < index.track_count(); ++track) {
        const uint64_t degree = static_cast<uint64_t>(
            index.offsets[track + 1] - index.offsets[track]);
        connected_tracks += degree > 0;
        maximum_degree = std::max(maximum_degree, degree);
    }
    nb::dict result;
    result["tracks"] = index.track_count();
    result["directed_crossings"] = index.crossing_count();
    result["connected_tracks"] = connected_tracks;
    result["maximum_degree"] = maximum_degree;
    result["memory_bytes"] = index.memory_bytes();
    return result;
}

nb::dict sample_crossing_partners(
    const CrossingIndex& index, Int32Vector primaries, int maximum,
    uint64_t seed)
{
    if (maximum < 0)
        throw std::runtime_error("maximum must be non-negative");
    const size_t rows = primaries.shape(0);
    const size_t width = static_cast<size_t>(maximum);
    if (width != 0 && rows > std::numeric_limits<size_t>::max() / width)
        throw std::runtime_error("sampled crossing table dimensions overflow");
    const size_t output_count = rows * width;
    std::vector<int32_t> partners(output_count, -1);
    std::vector<int32_t> records(output_count, -1);
    std::vector<int32_t> partner_records(output_count, -1);
    std::atomic<uint64_t> selected_slots{0};
    std::atomic<uint64_t> missing_reciprocals{0};

    {
        nb::gil_scoped_release release;
#pragma omp parallel for schedule(dynamic, 1024)
        for (int64_t row = 0; row < static_cast<int64_t>(rows); ++row) {
            const int32_t primary = primaries(static_cast<size_t>(row));
            if (primary < 0
                || static_cast<size_t>(primary) >= index.track_count())
                continue;
            const int64_t begin = index.offsets[static_cast<size_t>(primary)];
            const int64_t end = index.offsets[static_cast<size_t>(primary) + 1];
            const size_t degree = static_cast<size_t>(end - begin);
            const size_t count = std::min(width, degree);
            if (count == 0)
                continue;

            std::vector<int32_t> candidates(degree);
            std::iota(candidates.begin(), candidates.end(),
                      static_cast<int32_t>(begin));
            std::mt19937_64 random(
                seed + 0x9e3779b97f4a7c15ULL
                    * (static_cast<uint64_t>(row) + 1));
            for (size_t slot = 0; slot < count; ++slot) {
                std::uniform_int_distribution<size_t> draw(slot, degree - 1);
                const size_t choice = draw(random);
                std::swap(candidates[slot], candidates[choice]);
                const int32_t record = candidates[slot];
                const int32_t partner = index.partners[record];
                int32_t reciprocal = -1;
                for (int64_t candidate = index.offsets[partner];
                     candidate < index.offsets[static_cast<size_t>(partner) + 1];
                     ++candidate) {
                    if (index.partners[candidate] == primary
                        && index.self_local[candidate]
                            == index.partner_local[record]
                        && index.partner_local[candidate]
                            == index.self_local[record]) {
                        reciprocal = static_cast<int32_t>(candidate);
                        break;
                    }
                }
                const size_t output = static_cast<size_t>(row) * width + slot;
                partners[output] = partner;
                records[output] = record;
                partner_records[output] = reciprocal;
                missing_reciprocals.fetch_add(
                    reciprocal < 0, std::memory_order_relaxed);
            }
            selected_slots.fetch_add(count, std::memory_order_relaxed);
        }
    }
    if (missing_reciprocals.load(std::memory_order_relaxed) != 0)
        throw std::runtime_error(
            "crossing index requires reciprocal directed records");

    nb::dict result;
    result["partners"] = own_2d(std::move(partners), rows, width);
    result["records"] = own_2d(std::move(records), rows, width);
    result["partner_records"] = own_2d(
        std::move(partner_records), rows, width);
    result["selected_slots"] = selected_slots.load(std::memory_order_relaxed);
    return result;
}

} // namespace

NB_MODULE(track_crossings, module)
{
    module.doc() = "Memory-efficient native exact track-crossing construction.";
    nb::class_<EventBuffer>(module, "EventBuffer")
        .def_prop_ro("event_count", &EventBuffer::size)
        .def_prop_ro("memory_bytes", &EventBuffer::memory_bytes);
    nb::class_<CrossingIndex>(module, "CrossingIndex")
        .def_prop_ro("track_count", &CrossingIndex::track_count)
        .def_prop_ro("crossing_count", &CrossingIndex::crossing_count)
        .def_prop_ro("memory_bytes", &CrossingIndex::memory_bytes);
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
        "resample_tracks", &resample_tracks,
        nb::arg("coordinates"), nb::arg("offsets"),
        nb::arg("minimum_spacing"), nb::arg("maximum_spacing"),
        nb::arg("workers") = 1, nb::arg("progress") = nb::none(),
        nb::arg("crossing_index") = nullptr);
    module.def(
        "prepare_crossing_index", &prepare_crossing_index,
        nb::rv_policy::take_ownership,
        nb::arg("offsets"), nb::arg("partners"), nb::arg("self_local"),
        nb::arg("partner_local"), nb::arg("track_lengths"));
    module.def(
        "prepare_cached_crossing_index", &prepare_cached_crossing_index,
        nb::rv_policy::take_ownership,
        nb::arg("cached_source_ids"), nb::arg("cached_offsets"),
        nb::arg("cached_partners"), nb::arg("cached_self_local"),
        nb::arg("cached_partner_local"), nb::arg("selected_source_ids"),
        nb::arg("track_lengths"));
    module.def(
        "crossing_index_stats", &crossing_index_stats, nb::arg("index"));
    module.def(
        "sample_crossing_partners", &sample_crossing_partners,
        nb::arg("index"), nb::arg("primaries"), nb::arg("maximum"),
        nb::arg("seed"));
}
