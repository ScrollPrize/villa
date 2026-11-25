#pragma once
// A local, sealed cache of track crossings: one slim core edge per unordered
// pair of crossing tracks plus one compact measurement row, built with the
// sharded streaming initial importer and read back as numeric columns.
//
// The store holds NO objects. Its edges name tracks of another store by UUID
// (trusted construction does not resolve endpoints), so it is a derived
// result, not a graph to browse: ordinary relationship readers cannot resolve
// its endpoints. `basedOn` records the store position the crossings were
// computed from; it is provenance, not a store_reference.
#include "vc_object_store/objects/InitialImport.hpp"
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace vc::objects {
inline constexpr const char* kTrackCrossingExtension = "track-crossings";
inline constexpr const char* kTrackCrossingTable = "ext_track_crossings";
inline constexpr int kTrackCrossingFormat = 1;

using TrackCrossingUuid = std::array<unsigned char,16>;

// Borrowed input of a write: `rows` crossings of two tracks each. UUIDs are
// contiguous 16-byte binary values (16*rows bytes). A pair may be given in
// either order; the store keeps it with a < b by binary UUID, swapping the
// locals and positions with the endpoints. The arrays must stay alive and
// unchanged until the write returns.
struct TrackCrossingColumnsView {
    size_t rows = 0;
    std::span<const uint8_t> a, b;
    std::span<const int32_t> localA, localB;        // point index of the crossing on a / b
    std::span<const double> positionA, positionB;  // arclength of that point, stored voxels
    std::span<const double> clearance;             // of the pair, stored voxels
};

// Owned result of a read: pairs with a < b, in partition then edge-UUID order.
struct TrackCrossingColumns {
    size_t rows = 0;
    std::vector<uint8_t> a, b; // 16*rows bytes each
    std::vector<int32_t> localA, localB;
    std::vector<double> positionA, positionB, clearance;
};

// What identifies and interprets one cache store. Small by construction: no
// paths, no per-object lists.
struct TrackCrossingStoreInfo {
    std::string algorithm;          // e.g. "vc_spiral.track_crossings/1"
    std::string parameters = "{}";  // small JSON object of the algorithm's parameters
    std::string key;                // the consumer's identity of the track selection
    std::string checksum;           // the consumer's checksum of what it stored
    int64_t tracks = 0;             // tracks of the selection (crossing or not)
    int64_t crossings = 0;          // rows; set by the writer
    std::vector<ExtensionSource> basedOn; // the store position the tracks were read at
    bool operator==(const TrackCrossingStoreInfo&) const = default;
};

struct TrackCrossingWriteOptions {
    size_t workers = 8;         // destination shards and writers (1..32)
    size_t sqlBatchRows = 1;    // core-edge rows per INSERT statement
    // Keep the whole output in memory until publication instead of spilling
    // through the bounded packed cache; `residentBytes` is then a hard budget.
    bool memoryFirst = false;
    size_t residentBytes = size_t(4) << 30;
};

ExtensionTable trackCrossingExtension(const TrackCrossingStoreInfo&);
// The edge of a pair: a name UUID of its two endpoints in canonical order.
std::string trackCrossingEdgeUuid(const TrackCrossingUuid& a, const TrackCrossingUuid& b);

// Builds the sealed, history-off, packed collection at `destination` and
// publishes it atomically. Throws when the destination exists (nothing is
// overwritten), on an invalid row (equal endpoints, a negative local, a
// negative, negative-zero or non-finite position or clearance) and on any
// import failure; nothing is published then.
InitialImportResult writeTrackCrossingStore(const std::filesystem::path& destination,
    const TrackCrossingColumnsView&, TrackCrossingStoreInfo, const TrackCrossingWriteOptions& = {});

// Reads ONLY a store writeTrackCrossingStore made: a sealed, history-off root
// with its own partitions and no dependency or overlay layers. Anything else,
// another format, or rows that are not exactly one measurement per crossing
// edge, is rejected rather than interpreted.
TrackCrossingColumns readTrackCrossingColumns(const ObjectCollection&, const CollectionTxn&,
    size_t workers = 8, const std::function<void(const char*,size_t,size_t)>& progress = {});
TrackCrossingStoreInfo trackCrossingStoreInfo(const ObjectCollection&, const CollectionTxn&);
} // namespace vc::objects
