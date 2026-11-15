#include "TestSupport.hpp"
#include "vc_object_store/objects/TrackCrossingStore.hpp"
#include "vc_object_store/objects/EdgeKinds.hpp"
#include "internal/Sqlite.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <thread>

using namespace vc::objects;
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
using Uuid = TrackCrossingUuid;

struct Crossing { Uuid a, b; int32_t localA, localB; double positionA, positionB, clearance; };
struct Columns {
    std::vector<uint8_t> a, b;
    std::vector<int32_t> localA, localB;
    std::vector<double> positionA, positionB, clearance;
    void add(const Crossing& c) {
        a.insert(a.end(), c.a.begin(), c.a.end()); b.insert(b.end(), c.b.begin(), c.b.end());
        localA.push_back(c.localA); localB.push_back(c.localB);
        positionA.push_back(c.positionA); positionB.push_back(c.positionB); clearance.push_back(c.clearance);
    }
    TrackCrossingColumnsView view() const {
        return {localA.size(), a, b, localA, localB, positionA, positionB, clearance};
    }
};
Uuid track(uint32_t n)
{
    // Not ordered like n: the bytes are a name UUID of it.
    return detail::parseUuid(util::nameUuid("test-track:" + std::to_string(n)));
}
uint64_t bits(double value) { uint64_t out; std::memcpy(&out, &value, sizeof out); return out; }

TrackCrossingStoreInfo info(int64_t tracks = 7)
{
    TrackCrossingStoreInfo result;
    result.algorithm = "vc_spiral.track_crossings/1";
    result.parameters = R"({"min_angle_degrees":30.0,"tangent_radius_voxels":12.0,"cluster_index_radius":4})";
    result.key = std::string(64, 'k');
    result.checksum = std::string(64, 'c');
    result.tracks = tracks;
    // A store that is neither this collection nor a dependency of it.
    result.basedOn = {{util::newUuid(), util::newUuid(), 41}};
    return result;
}
// Crossings whose values exercise the numeric domain; pairs in mixed order.
std::vector<Crossing> sample(size_t count)
{
    std::vector<Crossing> rows;
    for (uint32_t i = 0; i < count; ++i) {
        const auto a = track(i % 97), b = track(1000 + (i * 7919u) % 1009 + i);
        rows.push_back({a, b, int32_t(i % 5), int32_t(i * 3 % 11), 0.1 * i, 17.0 + i / 3.0, std::sqrt(double(i))});
    }
    if (rows.size() > 3) {
        rows[0] = {track(1), track(2), 0, std::numeric_limits<int32_t>::max(), 0.0, 4.0, 0.0}; // integral and zero REALs
        rows[1] = {track(3), track(4), 7, 8, std::nextafter(1.0, 2.0), 1e300, std::numeric_limits<double>::denorm_min()};
        rows[2] = {track(5), track(6), 1, 2, 9007199254740993.0, 0.1 + 0.2, std::numeric_limits<double>::max()};
    }
    return rows;
}
// Canonical form of what a store must return for these rows.
std::map<std::pair<Uuid,Uuid>, std::array<uint64_t,5>> canonical(const std::vector<Crossing>& rows)
{
    std::map<std::pair<Uuid,Uuid>, std::array<uint64_t,5>> out;
    for (auto c : rows) {
        if (!(c.a < c.b)) { std::swap(c.a, c.b); std::swap(c.localA, c.localB); std::swap(c.positionA, c.positionB); }
        require(out.emplace(std::pair{c.a, c.b}, std::array<uint64_t,5>{uint64_t(c.localA), uint64_t(c.localB),
            bits(c.positionA), bits(c.positionB), bits(c.clearance)}).second, "test pairs are distinct");
    }
    return out;
}
std::map<std::pair<Uuid,Uuid>, std::array<uint64_t,5>> canonical(const TrackCrossingColumns& columns)
{
    std::map<std::pair<Uuid,Uuid>, std::array<uint64_t,5>> out;
    for (size_t i = 0; i < columns.rows; ++i) {
        Uuid a, b;
        std::memcpy(a.data(), columns.a.data() + 16 * i, 16); std::memcpy(b.data(), columns.b.data() + 16 * i, 16);
        require(a < b, "read pairs are in canonical order");
        require(out.emplace(std::pair{a, b}, std::array<uint64_t,5>{uint64_t(columns.localA[i]), uint64_t(columns.localB[i]),
            bits(columns.positionA[i]), bits(columns.positionB[i]), bits(columns.clearance[i])}).second, "read pairs are distinct");
    }
    return out;
}
Columns columnsOf(const std::vector<Crossing>& rows)
{
    Columns columns;
    for (const auto& row : rows) columns.add(row);
    return columns;
}
OpenOptions readOnly() { OpenOptions options; options.readOnly = true; return options; }
TrackCrossingColumns read(const fs::path& path, size_t workers = 4)
{
    auto store = ObjectCollection::open(path, readOnly());
    auto txn = store.read();
    return readTrackCrossingColumns(store, txn, workers);
}
std::string failure(const std::function<void()>& callback)
{
    try { callback(); } catch (const std::exception& error) { return error.what(); }
    throw std::runtime_error("expected the operation to fail");
}
void requireFailure(const std::function<void()>& callback, const char* text)
{
    const auto what = failure(callback);
    if (what.find(text) == std::string::npos)
        throw std::runtime_error(std::string("expected failure mentioning '") + text + "', got: " + what);
}
bool leftovers(const fs::path& directory, const std::string& name)
{
    for (const auto& entry : fs::directory_iterator(directory))
        if (entry.path().filename().string().starts_with(name)) return true;
    return false;
}

// A collection built directly with the importer, to give the reader shapes
// the writer never produces.
struct RawEdge { Uuid a, b; std::string uuid; std::string kind = "crossing"; std::optional<ExtensionRow> row; };
RawEdge rawEdge(uint32_t a, uint32_t b, int64_t localA = 1, int64_t localB = 2)
{
    auto first = track(a), second = track(b);
    if (!(first < second)) std::swap(first, second);
    RawEdge edge{first, second, trackCrossingEdgeUuid(first, second)};
    const auto id = detail::parseUuid(edge.uuid);
    edge.row = ExtensionRow{std::vector<uint8_t>(id.begin(), id.end()), localA, localB, 1.5, 2.5, 0.25};
    return edge;
}
void rawImport(const fs::path& path, size_t shards, bool packed, std::vector<RawEdge> edges,
               std::optional<ExtensionTable> table = trackCrossingExtension(info()), bool slim = true,
               bool everyShard = false)
{
    InitialImportOptions options;
    options.store.author = "test";
    if (slim) options.store.edgeColumns = std::vector<std::string>{};
    options.store.edgeAdjacencyIndex = false;
    options.shards.resize(shards);
    const auto stride = std::numeric_limits<int64_t>::max() / int64_t(shards);
    for (size_t i = 0; i < shards; ++i) options.shards[i].objectIds = {1 + int64_t(i) * stride, int64_t(i + 1) * stride};
    if (table) options.extensions = {*table};
    if (packed) options.packed = PackedStoreOptions{};
    // Every shard holds every edge: only possible without UUID routing.
    options.uuidHashRouting = !everyShard;
    importInitialCollection(path, options, [](size_t, InitialShardImport&) {}, [&](size_t shard, InitialShardImport& writer) {
        bool done = false;
        writer.importEdges([&](std::vector<InitialCollectionEdge>& batch) {
            if (done) return false;
            done = true;
            for (const auto& edge : edges) {
                if (!everyShard && uuidShardIndex(edge.uuid, shards) != shard) continue;
                InitialCollectionEdge out;
                out.record.uuid = edge.uuid; out.record.kind = edge.kind;
                out.record.aUuid = detail::formatUuid(edge.a); out.record.bUuid = detail::formatUuid(edge.b);
                out.a = {"", out.record.aUuid}; out.b = {"", out.record.bUuid};
                if (edge.row && table) out.extensions.push_back({table->name, *edge.row, {}});
                batch.push_back(std::move(out));
            }
            return !batch.empty();
        });
    });
}

void roundTrip(const fs::path& directory)
{
    const auto rows = sample(5000);
    const auto columns = columnsOf(rows);
    const auto expected = canonical(rows);
    for (const size_t workers : {size_t(1), size_t(4), size_t(32)})
        for (const bool memoryFirst : {false, true}) {
            const auto path = directory / ("roundtrip-" + std::to_string(workers) + (memoryFirst ? "m" : "") + ".vcobj");
            auto description = info();
            TrackCrossingWriteOptions options; options.workers = workers; options.memoryFirst = memoryFirst;
            options.sqlBatchRows = workers == 4 ? 64 : 1;
            const auto result = writeTrackCrossingStore(path, columns.view(), description, options);
            require(fs::is_regular_file(path) && result.path == path, "the cache store is one packed file");
            require(!leftovers(directory, path.filename().string() + "."), "publication leaves no staging file");
            const auto back = read(path, workers);
            require(back.rows == rows.size() && canonical(back) == expected, "crossings round-trip bit for bit");
            auto store = ObjectCollection::open(path, readOnly());
            auto txn = store.read();
            description.crossings = int64_t(rows.size());
            auto stored = trackCrossingStoreInfo(store, txn);
            require(Json::parse(stored.parameters) == Json::parse(description.parameters), "parameters are kept");
            stored.parameters = description.parameters;
            require(stored == description, "identity, counts and the foreign input position are kept");
            require(store.supportsTrustedGeometryReads(txn), "the cache store is sealed and history-off");
        }
}

void emptyAndDeterministic(const fs::path& directory)
{
    Columns none;
    const auto path = directory / "empty.vcobj";
    writeTrackCrossingStore(path, none.view(), info(0), {4});
    require(read(path).rows == 0, "a selection without crossings is a valid, empty store");
    {
        auto store = ObjectCollection::open(path, readOnly());
        auto txn = store.read();
        require(trackCrossingStoreInfo(store, txn).crossings == 0, "an empty store declares no crossings");
    }
    // The edge of a pair does not depend on the order it was given in, nor on the run.
    const auto a = track(11), b = track(12);
    require(trackCrossingEdgeUuid(a, b) == trackCrossingEdgeUuid(b, a) && util::isUuid(trackCrossingEdgeUuid(a, b))
                && trackCrossingEdgeUuid(a, b) != trackCrossingEdgeUuid(a, track(13)),
            "the edge UUID identifies the unordered pair");
    // Reversed input order: endpoints, locals and positions swap together.
    Crossing forward{a, b, 3, 9, 1.25, 7.5, 0.5}, backward{b, a, 9, 3, 7.5, 1.25, 0.5};
    Columns one, other;
    one.add(forward); other.add(backward);
    writeTrackCrossingStore(directory / "forward.vcobj", one.view(), info(), {2});
    writeTrackCrossingStore(directory / "backward.vcobj", other.view(), info(), {2});
    require(canonical(read(directory / "forward.vcobj")) == canonical(read(directory / "backward.vcobj"))
                && canonical(read(directory / "forward.vcobj")) == canonical(std::vector<Crossing>{forward}),
            "a pair given in either order is stored once, in canonical order");
}

void refusals(const fs::path& directory)
{
    const auto rows = sample(64);
    const auto columns = columnsOf(rows);
    const auto path = directory / "refused.vcobj";
    const auto absent = [&] { return !fs::exists(path) && !leftovers(directory, "refused.vcobj"); };
    const auto bad = [&](const char* text, const std::function<void(Columns&)>& edit) {
        auto changed = columns;
        edit(changed);
        requireFailure([&] { writeTrackCrossingStore(path, changed.view(), info(), {4}); }, text);
        require(absent(), "a refused write creates nothing");
    };
    bad("negative", [](Columns& c) { c.localA[5] = -1; });
    bad("negative or not finite", [](Columns& c) { c.positionB[6] = -0.0; });
    bad("negative or not finite", [](Columns& c) { c.clearance[7] = std::numeric_limits<double>::quiet_NaN(); });
    bad("negative or not finite", [](Columns& c) { c.positionA[8] = std::numeric_limits<double>::infinity(); });
    bad("negative or not finite", [](Columns& c) { c.clearance[9] = -1.0; });
    bad("one track at both ends", [](Columns& c) { std::copy_n(c.a.begin() + 16, 16, c.b.begin() + 16); });
    bad("one value per row", [](Columns& c) { c.clearance.pop_back(); });
    auto unnamed = info(); unnamed.key.clear();
    requireFailure([&] { writeTrackCrossingStore(path, columns.view(), unnamed, {4}); }, "algorithm and a key");
    requireFailure([&] { writeTrackCrossingStore(path, columns.view(), info(), {0}); }, "workers");
    requireFailure([&] { writeTrackCrossingStore(path, columns.view(), info(), {33}); }, "workers");
    // The same pair twice collides on its edge inside a shard writer: the
    // failed import publishes nothing.
    auto twice = columns;
    twice.add(rows[10]);
    mustFail([&] { writeTrackCrossingStore(path, twice.view(), info(), {4}); });
    require(absent(), "a failed import publishes nothing");

    // Nothing is overwritten.
    writeTrackCrossingStore(path, columns.view(), info(), {4});
    const auto before = canonical(read(path));
    requireFailure([&] { writeTrackCrossingStore(path, columnsOf(sample(8)).view(), info(), {4}); }, "already exists");
    require(canonical(read(path)) == before, "an existing store is kept as it was");
}

void publicationRace(const fs::path& directory)
{
    const auto rows = sample(3000);
    const auto columns = columnsOf(rows);
    const auto path = directory / "race.vcobj";
    std::atomic<int> published{0}, refused{0};
    std::vector<std::thread> writers;
    for (int i = 0; i < 6; ++i)
        writers.emplace_back([&] {
            try { writeTrackCrossingStore(path, columns.view(), info(), {3}); ++published; }
            catch (const std::exception&) { ++refused; }
        });
    for (auto& writer : writers) writer.join();
    require(published.load() == 1 && refused.load() == 5, "exactly one concurrent writer publishes");
    require(canonical(read(path)) == canonical(rows), "the published store is complete");
}

void readerScope(const fs::path& directory)
{
    const std::vector<RawEdge> good{rawEdge(1, 2), rawEdge(3, 4), rawEdge(5, 6), rawEdge(7, 8), rawEdge(9, 10)};
    const auto rejects = [&](const std::string& name, size_t shards, bool packed, std::vector<RawEdge> edges,
                             const char* text, std::optional<ExtensionTable> table = trackCrossingExtension(info()),
                             bool slim = true) {
        const auto path = directory / name;
        auto description = info();
        description.crossings = int64_t(edges.size());
        if (table && table->provisioning == trackCrossingExtension(info()).provisioning)
            table->provisioning = trackCrossingExtension(description).provisioning;
        rawImport(path, shards, packed, std::move(edges), table, slim);
        requireFailure([&] { read(packed || shards == 1 ? path : path / "collection.vcobj"); }, text);
    };
    // The same construction, conforming, is read (packed and as a directory of shards, single and several).
    for (const auto& [name, shards, packed] : {std::tuple{"ok-1p.vcobj", size_t(1), true}, std::tuple{"ok-3p.vcobj", size_t(3), true},
                                               std::tuple{"ok-1.vcobj", size_t(1), false}, std::tuple{"ok-3", size_t(3), false}}) {
        auto description = info(); description.crossings = int64_t(good.size());
        rawImport(directory / name, shards, packed, good, trackCrossingExtension(description));
        require(read(packed || shards == 1 ? directory / name : directory / name / "collection.vcobj").rows == good.size(),
                "a conforming store of any shard layout is read, its manifest root excluded");
    }
    // A full edge layout is fine as long as the edges are active crossings.
    rejects("no-table.vcobj", 2, true, good, "no track crossing table", std::nullopt);
    auto other = trackCrossingExtension(info()); other.formatVersion = 2;
    rejects("format.vcobj", 2, true, good, "unsupported track crossing table format", other);
    auto wider = trackCrossingExtension(info()); wider.columns.push_back({"extra", ExtensionType::Real, true});
    auto wide = good;
    for (auto& edge : wide) edge.row->push_back(std::monostate{});
    rejects("columns.vcobj", 2, true, wide, "unsupported track crossing table format", wider);
    auto missing = good; missing[2].row.reset();
    rejects("missing-row.vcobj", 1, true, missing, "do not correspond");
    auto kind = good; kind[1].kind = "parallel";
    rejects("kind.vcobj", 2, true, kind, "no active crossing");
    auto overflow = good; overflow[0] = rawEdge(1, 2, int64_t(1) << 31, 2);
    rejects("overflow.vcobj", 2, true, overflow, "out of range");
    auto negative = good; negative[0] = rawEdge(1, 2, 1, -4);
    rejects("negative.vcobj", 2, true, negative, "out of range");
    auto renamed = good; renamed[3].uuid = util::newUuid();
    (*renamed[3].row)[0] = extensionUuid(renamed[3].uuid);
    rejects("pair.vcobj", 1, true, renamed, "does not identify its pair");
    auto declared = good;
    {
        auto description = info(); description.crossings = 99;
        rawImport(directory / "count.vcobj", 2, true, declared, trackCrossingExtension(description));
        requireFailure([&] { read(directory / "count.vcobj"); }, "does not hold the rows it declares");
    }
    // NULL in a payload column never gets in: the columns are not nullable.
    auto null = good; (*null[0].row)[4] = std::monostate{};
    mustFail([&] { rawImport(directory / "null.vcobj", 1, true, null); });

    // Shards that disagree on what they hold.
    {
        auto description = info(); description.crossings = int64_t(good.size());
        rawImport(directory / "disagree", 3, false, good, trackCrossingExtension(description));
        sqlite3* db = nullptr;
        require(sqlite3_open((directory / "disagree" / "part-1.vcobj").string().c_str(), &db) == SQLITE_OK, "open one shard");
        auto changed = Json::parse(trackCrossingExtension(description).provisioning); changed["key"] = "another";
        const auto sql = "UPDATE extension_tables SET provisioning='" + changed.dump() + "'";
        const auto rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, nullptr);
        sqlite3_close(db);
        require(rc == SQLITE_OK, "edit one shard's descriptor");
        requireFailure([&] { read(directory / "disagree" / "collection.vcobj"); }, "disagree");
    }
    // Shards that are not routed by edge UUID could each hold the same pair:
    // one edge and its measurement in both shards, declared as two crossings.
    {
        const std::vector<RawEdge> one{rawEdge(1, 2)};
        auto description = info(); description.crossings = 2;
        rawImport(directory / "unrouted.vcobj", 2, true, one, trackCrossingExtension(description), true, true);
        requireFailure([&] { read(directory / "unrouted.vcobj"); }, "routed by edge UUID");
    }
    // The root of several shards is a manifest: it is not taken for a data
    // shard when it carries the table, and a data shard may not lack it.
    {
        auto description = info(); description.crossings = int64_t(good.size());
        const auto raw = [&](const fs::path& file, const std::string& sql) {
            sqlite3* db = nullptr;
            require(sqlite3_open(file.string().c_str(), &db) == SQLITE_OK, "open a member database");
            char* message = nullptr;
            const auto rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &message);
            const std::string what = message ? message : "";
            sqlite3_free(message); sqlite3_close(db);
            if (rc != SQLITE_OK) throw std::runtime_error("test SQL failed: " + what);
        };
        rawImport(directory / "bare-shard", 3, false, good, trackCrossingExtension(description));
        raw(directory / "bare-shard" / "part-2.vcobj",
            std::string("DELETE FROM edges; DROP TABLE ") + kTrackCrossingTable + "; DELETE FROM extension_tables");
        requireFailure([&] { read(directory / "bare-shard" / "collection.vcobj"); }, "every data shard");
        rawImport(directory / "full-root", 3, false, good, trackCrossingExtension(description));
        const auto shard = (directory / "full-root" / "part-0.vcobj").string();
        std::string definition; // the shard's own CREATE TABLE statement
        {
            sqlite3* db = nullptr;
            require(sqlite3_open(shard.c_str(), &db) == SQLITE_OK, "open a shard");
            {
                detail::Statement sql(db, std::string("SELECT sql FROM sqlite_schema WHERE name='") + kTrackCrossingTable + "'");
                require(sql.row(), "the shard has the table");
                definition = sql.text(0);
            }
            sqlite3_close(db);
        }
        raw(directory / "full-root" / "collection.vcobj",
            "ATTACH '" + shard + "' AS shard;"
            "CREATE TABLE IF NOT EXISTS main.extension_tables AS SELECT * FROM shard.extension_tables;"
            "INSERT INTO main.extension_tables SELECT * FROM shard.extension_tables WHERE NOT EXISTS (SELECT 1 FROM main.extension_tables);"
            "INSERT OR IGNORE INTO main.table_directory SELECT * FROM shard.table_directory;"
            + definition + ";");
        requireFailure([&] { read(directory / "full-root" / "collection.vcobj"); }, "manifest-only root");
    }
    // Not a crossing cache at all: an editable, history-on store, and an overlay over a cache.
    {
        OpenOptions options; options.author = "test"; options.history = HistoryMode::On;
        { auto store = ObjectCollection::create(directory / "editable.vcobj", {}, options); }
        requireFailure([&] { read(directory / "editable.vcobj"); }, "sealed and history-off");
        OpenOptions overlay; overlay.author = "test"; overlay.history = HistoryMode::Off; overlay.spatialIndex = false;
        {
            auto store = ObjectCollection::create(directory / "overlay.vcobj", {directory / "ok-1p.vcobj"}, overlay);
            auto txn = store.write(); store.seal(txn); txn.commit();
        }
        mustFail([&] { read(directory / "overlay.vcobj"); });
    }
}
} // namespace

int main()
{
    try {
        TemporaryDirectory temporary;
        roundTrip(temporary.path);
        emptyAndDeterministic(temporary.path);
        refusals(temporary.path);
        publicationRace(temporary.path);
        readerScope(temporary.path);
        std::cout << "track crossing store: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
