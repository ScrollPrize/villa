// vc_object_ingest --bulk: ingestTracksBulk builds the same track objects as the per-track ingestGeometrySource path.
// Local data, referenced data (an editable overlay root over the sealed bulk collection), one file and a sharded directory,
// source/destination history policies (history-off sealed base, history-on root), the failure cases, and a write through the
// overlay that leaves the sealed base untouched.
#include "TestSupport.hpp"
#include "vc_object_store/objects/GeometryIngest.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include <nlohmann/json.hpp>
#include <cstring>
#include <fstream>
#include <map>

using namespace vc::objects;
using Json = nlohmann::json;
namespace fs = std::filesystem;

namespace {
template <class T> void writeArray(const fs::path& path, const std::vector<T>& values)
{
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()), std::streamsize(values.size() * sizeof(T)));
}
// A packed track store of `tracks` tracks (VCTRK01 layout, little-endian): track t has 2 + t % 7 points in (z, y, x) order.
fs::path writePackedTracks(const fs::path& directory, int tracks)
{
    fs::create_directories(directory);
    std::vector<int32_t> coordinates; std::vector<int64_t> offsets{0}; std::vector<uint64_t> sourceIds; std::vector<int8_t> families;
    for (int t = 0; t < tracks; ++t) {
        const int length = 2 + t % 7;
        for (int i = 0; i < length; ++i) { coordinates.push_back(100 + 10 * t + i); coordinates.push_back(3 * t + i % 4); coordinates.push_back(7 * t - i); }
        offsets.push_back(offsets.back() + length);
        sourceIds.push_back((uint64_t(t / 4) << 32) | uint64_t(t % 4));
        families.push_back(int8_t(t % 3 - 1));
    }
    writeArray(directory / "coordinates.i32", coordinates); writeArray(directory / "offsets.i64", offsets);
    writeArray(directory / "source_ids.u64", sourceIds); writeArray(directory / "family_codes.i8", families);
    char header[64] = {};
    std::memcpy(header, "VCTRK01\0", 8);
    const uint32_t version = 1, size = 64; const uint64_t count = uint64_t(tracks), points = uint64_t(offsets.back());
    std::memcpy(header + 8, &version, 4); std::memcpy(header + 12, &size, 4); std::memcpy(header + 16, &count, 8); std::memcpy(header + 24, &points, 8);
    std::ofstream(directory / "header.bin", std::ios::binary).write(header, 64);
    std::ofstream(directory / "metadata.json") << Json({{"version", 1}, {"track_count", tracks}, {"point_count", int64_t(points)}}).dump();
    return directory;
}
struct Snapshot { Json meta; TrackDetail track; std::string kind; std::vector<uint8_t> points; Json geometryAttributes; };
// Everything a consumer can observe about each track, keyed by its source ID (the object name); UUIDs are fresh per ingest.
std::map<std::string, Snapshot> snapshot(const fs::path& root)
{
    OpenOptions read; read.readOnly = true;
    auto store = ObjectCollection::open(root, read);
    auto txn = store.read();
    std::map<std::string, Snapshot> result;
    store.scanObjects(txn, [&](const CollectionObject& object) {
        Snapshot s; s.kind = object.record.kind;
        s.meta = Json::parse(object.record.meta);
        require(object.record.track.has_value(), "a track object has track detail");
        s.track = *object.record.track;
        const auto geometry = store.currentGeometry(txn, object.key);
        require(geometry.has_value(), "a track has current geometry");
        s.points = store.readGeometryArray(txn, object.key, "points");
        s.geometryAttributes = Json::parse(geometry->attributes);
        result[object.record.name] = std::move(s);
        return true;
    }, std::string("track"));
    return result;
}
void requireSame(const std::map<std::string, Snapshot>& a, const std::map<std::string, Snapshot>& b, const char* what)
{
    require(a.size() == b.size(), what);
    for (const auto& [name, x] : a) {
        const auto found = b.find(name);
        require(found != b.end(), "same source IDs");
        const auto& y = found->second;
        require(x.kind == y.kind && x.meta == y.meta, "same kind and metadata (coordinate frame, source)");
        require(x.track.datasetRef == y.track.datasetRef && x.track.externalKey == y.track.externalKey && x.track.family == y.track.family
                    && x.track.pointCount == y.track.pointCount && x.track.length == y.track.length,
                "same track detail: identity, family, point count and length");
        require(x.points == y.points, "same points bytes");
        require(x.geometryAttributes == y.geometryAttributes, "same geometry attributes");
    }
}
}

int main()
{
    TemporaryDirectory dir;
    const int tracks = 53;
    const auto packed = writePackedTracks(dir.path / "scroll.vctracks", tracks);
    const std::string frame = geometryCoordinates({{"volume_uuid", "bulk-test-volume"}, {"scale_factor", 1.0}}).dump();
    const std::vector<std::string> tags = {"from-test"};

    // Reference: the per-track path, one transaction per track.
    const auto perTrack = dir.path / "per_track.vcobj";
    {
        OpenOptions options; options.author = "bulk-test";
        auto db = ObjectStore::create(perTrack, options);
        const auto report = ingestGeometrySource(db, "track", packed, {}, frame, tags, {});
        require(report.ingested == tracks && report.problems.empty(), "per-track ingest of every track");
        db.checkpoint();
    }
    const auto expected = snapshot(perTrack);
    require(int(expected.size()) == tracks, "reference store holds every track");

    for (const std::size_t shards : {std::size_t(1), std::size_t(3)}) {
        const auto base = dir.path / ("bulk" + std::to_string(shards) + ".tracks");
        const auto root = dir.path / ("bulk" + std::to_string(shards) + ".vcobj");
        const auto report = ingestTracksBulk(packed, base, "bulk-test", frame, tags, shards);
        require(report.ingested == tracks && report.problems.empty(), "bulk ingest reports every track");
        // One shard is a file; several are a directory whose root locator is collection.vcobj.
        require(shards == 1 ? fs::is_regular_file(base) : fs::is_directory(base), "shard layout: file for one shard, directory for several");
        const auto baseRoot = shards == 1 ? base : base / "collection.vcobj";

        // Placement: the sealed history-off base on its own, and the editable history-on overlay root over it (what a fit opens).
        requireSame(expected, snapshot(baseRoot), "the sealed bulk collection holds the same tracks as the per-track store");
        OpenOptions rootOptions; rootOptions.author = "bulk-test";
        { auto created = ObjectCollection::create(root, {baseRoot}, rootOptions); (void)created; }
        requireSame(expected, snapshot(root), "the overlay root resolves the same tracks through the shared reader");

        // Source/destination history policies and a write through the overlay: the sealed base cannot be opened writable,
        // the overlay can, and its write leaves the base unchanged.
        OpenOptions writable; writable.author = "bulk-test";
        mustFail([&] { auto sealed = ObjectStore::open(baseRoot, writable); });
        {
            auto overlay = ObjectCollection::open(root, writable);
            auto txn = overlay.write();
            std::optional<CollectionObject> first;
            overlay.scanObjects(txn, [&](const CollectionObject& object) { first = object; return false; }, std::string("track"));
            require(first.has_value(), "an inherited track is visible to the overlay writer");
            overlay.setMeta(txn, first->key, "family", R"("H")");
            txn.commit();
            const auto after = snapshot(root);
            require(after.at(first->record.name).meta.value("family", Json()) == Json("H"), "the overlay carries the edited metadata");
            require(after.size() == expected.size(), "the edit does not change the track count");
        }
        requireSame(expected, snapshot(baseRoot), "a write through the overlay leaves the sealed base untouched");
    }

    // Failure cases leave nothing behind: an existing destination, a missing source and an unparseable coordinate frame are refused.
    mustFail([&] { ingestTracksBulk(packed, dir.path / "bulk1.tracks", "bulk-test", frame, tags, 1); });
    mustFail([&] { ingestTracksBulk(dir.path / "no-such.vctracks", dir.path / "missing.tracks", "bulk-test", frame, tags, 1); });
    require(!fs::exists(dir.path / "missing.tracks"), "a failed bulk ingest leaves no destination");
    mustFail([&] { ingestTracksBulk(packed, dir.path / "badframe.tracks", "bulk-test", "not json", tags, 1); });
    require(!fs::exists(dir.path / "badframe.tracks"), "an unparseable coordinate frame is refused and leaves no destination");
    return 0;
}
