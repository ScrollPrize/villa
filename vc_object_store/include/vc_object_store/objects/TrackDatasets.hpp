#pragma once
// How registered tracks are prepared for crossing detection: grouped by
// dataset and ordered by source ID. Crossings exist only between tracks of
// one dataset. Shared by the store's track-track detector and by solvers that
// compute the crossings of their selected tracks themselves, so both pair the
// same tracks in the same order.
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace vc::objects {

struct TrackDatasetRow {
    std::string uuid;
    std::string dataset;      // trackDatasetIdentity (ObjectStore::trackDatasets)
    std::string externalKey;  // the track's canonical decimal source ID
    std::string family;       // "horizontal", "vertical" or "" (no crossings)
};

struct TrackDatasetGroup {
    std::string dataset;
    std::vector<size_t> rows;         // indices of the input, by ascending source ID
    std::vector<uint64_t> sourceIds;  // of those rows
    std::vector<int8_t> families;     // of those rows: 0 horizontal, 1 vertical, -1 none
};

// The groups, ordered by dataset. Throws std::runtime_error for a track
// without a dataset, an external key that is no canonical decimal uint64, a
// source ID that repeats within a dataset, and an unknown family.
std::vector<TrackDatasetGroup> groupTrackDatasets(std::span<const TrackDatasetRow> rows);

}
