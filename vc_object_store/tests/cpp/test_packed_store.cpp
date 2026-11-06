#include "TestSupport.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <sqlite3.h>
#include <array>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <vector>
#ifndef _WIN32
#include <csignal>
#include <sys/resource.h>
#endif

using namespace vc::objects;
namespace {
struct Connection {
    sqlite3* db = nullptr;
    Connection(const std::filesystem::path& path, bool write = true, bool packed = true)
    {
        const auto rc = sqlite3_open_v2(path.string().c_str(), &db,
            (write ? SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE : SQLITE_OPEN_READONLY) | SQLITE_OPEN_FULLMUTEX,
            packed ? packedStoreVfs() : nullptr);
        if (rc != SQLITE_OK) {
            const std::string message = db ? sqlite3_errmsg(db) : "sqlite open failed";
            if (db) sqlite3_close(db); db = nullptr;
            throw std::runtime_error(message);
        }
    }
    ~Connection() { if (db) sqlite3_close(db); }
    void exec(const std::string& sql) { if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, nullptr) != SQLITE_OK) throw std::runtime_error(sql + ": " + sqlite3_errmsg(db) + " (" + std::to_string(sqlite3_extended_errcode(db)) + ") / " + packedStoreLastError()); }
    int scalar(const char* sql)
    {
        sqlite3_stmt* statement = nullptr;
        require(sqlite3_prepare_v2(db, sql, -1, &statement, nullptr) == SQLITE_OK, "prepare scalar");
        const auto rc = sqlite3_step(statement);
        const auto value = rc == SQLITE_ROW ? sqlite3_column_int(statement, 0) : -1;
        sqlite3_finalize(statement);
        require(rc == SQLITE_ROW, "scalar row missing");
        return value;
    }
};
struct RawFile {
    sqlite3_vfs* vfs = sqlite3_vfs_find(packedStoreVfs());
    std::vector<std::max_align_t> storage;
    sqlite3_file* file;
    RawFile(const std::filesystem::path& path, bool write,int kind=SQLITE_OPEN_MAIN_DB)
        : storage((vfs->szOsFile + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t)), file(reinterpret_cast<sqlite3_file*>(storage.data()))
    {
        int actual = 0;
        require(vfs->xOpen(vfs, path.string().c_str(), file, kind | (write ? SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE : SQLITE_OPEN_READONLY), &actual) == SQLITE_OK, "raw open");
    }
    ~RawFile() { file->pMethods->xClose(file); }
};
#ifndef _WIN32
struct ScopedFileLimit {
    rlimit previous{};
    struct sigaction signal{};
    ScopedFileLimit()
    {
        require(getrlimit(RLIMIT_FSIZE, &previous) == 0, "read file limit");
        struct sigaction ignore{}; ignore.sa_handler = SIG_IGN; sigemptyset(&ignore.sa_mask);
        require(sigaction(SIGXFSZ, &ignore, &signal) == 0, "ignore file-limit signal");
        auto limit = previous; limit.rlim_cur = std::min<rlim_t>(previous.rlim_cur, 65536);
        if (setrlimit(RLIMIT_FSIZE, &limit) != 0) { sigaction(SIGXFSZ, &signal, nullptr); throw std::runtime_error("set test file limit"); }
    }
    ~ScopedFileLimit() { setrlimit(RLIMIT_FSIZE, &previous); sigaction(SIGXFSZ, &signal, nullptr); }
};
#endif
void populate(const std::filesystem::path& path, int tag)
{
    Connection db(path);
    db.exec("PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; PRAGMA temp_store=MEMORY; CREATE TABLE rows(id INTEGER PRIMARY KEY, value BLOB, tag INTEGER); BEGIN;");
    for (int i = 0; i < 32; ++i) db.exec("INSERT INTO rows VALUES(" + std::to_string(i) + ",zeroblob(98304)," + std::to_string(tag) + ")");
    db.exec("COMMIT; BEGIN; DELETE FROM rows WHERE id<16; ROLLBACK;");
    require(db.scalar("SELECT count(*) FROM rows") == 32, "rollback lost rows");
    // Repeated rewrites while background persistence is active must retain the
    // newest write, not clear its dirty state when an older write completes.
    for (int i = 0; i < 24; ++i) db.exec("UPDATE rows SET tag=" + std::to_string(i) + " WHERE id=0");
    require(db.scalar("SELECT tag FROM rows WHERE id=0") == 23, "latest rewrite missing");
    Connection other(path);
    db.exec("BEGIN IMMEDIATE;");
    require(sqlite3_exec(other.db, "BEGIN IMMEDIATE", nullptr, nullptr, nullptr) == SQLITE_BUSY, "independent writer escaped SQLite member lock");
    db.exec("ROLLBACK;");
}
void verify(const std::filesystem::path& path)
{
    Connection db(path, false);
    require(db.scalar("SELECT count(*) FROM rows") == 32, "packed row count");
    require(db.scalar("SELECT tag FROM rows WHERE id=0") == 23, "packed rewrite mismatch");
    require(db.scalar("SELECT sum(length(value)) FROM rows") == 32 * 98304, "packed payload truncated");
    require(db.scalar("SELECT count(*) FROM pragma_integrity_check WHERE integrity_check!='ok'") == 0, "SQLite integrity check failed");
    mustFail([&] { db.exec("DELETE FROM rows"); });
}
}
int main()
{
    try {
        TemporaryDirectory temp;
        PackedStoreOptions options;
        options.extentBytes = 1024 * 1024;
        options.residentBytes = 2 * options.extentBytes;
        const auto root = util::newUuid(), second = util::newUuid();
        const auto container = temp.path / "extents.vcobj";
        {
            PackedStoreBuilder builder(container, options);
            const auto a = builder.addDatabase(root), b = builder.addDatabase(second);
            require(!isPackedStore(container), "unpublished output was visible");
            auto first = std::async(std::launch::async, [&] { populate(a, 1); });
            auto other = std::async(std::launch::async, [&] { populate(b, 2); });
            first.get(); other.get();
            mustFail([&] { builder.finalize(util::newUuid()); });
            { Connection open(a); mustFail([&] { builder.finalize(root); }); }
            builder.finalize(root, false);
            mustFail([&] { builder.addDatabase(util::newUuid()); });
        }
        require(isPackedStore(container), "packed magic not detected");
        require(packedDatabases(container).size() == 2, "member directory count");
        verify(packedDatabasePath(container)); verify(packedDatabasePath(container, second));
        mustFail([&] { Connection writable(packedDatabasePath(container)); });
        mustFail([&] { packedDatabasePath(container, util::newUuid()); });
        const auto compact = temp.path / "compact.vcobj";
        compactPackedStore(container, compact, options);
        verify(packedDatabasePath(compact)); verify(packedDatabasePath(compact, second));
        require(std::filesystem::file_size(compact) < std::filesystem::file_size(container), "compact export did not remove extent slack");

        // Partial write, eviction/reload, truncate/reextend, and SQLite's required
        // zero-fill semantics on short reads, independent of SQL page behavior.
        {
            PackedStoreBuilder builder(temp.path / "raw.vcobj", options);
            const auto id = util::newUuid(); const auto path = builder.addDatabase(id);
            {
                RawFile raw(path, true);
                std::vector<char> bytes(4 * options.extentBytes, 'a'), got(bytes.size());
                require(raw.file->pMethods->xWrite(raw.file, bytes.data(), int(bytes.size()), 0) == SQLITE_OK, "large raw write");
                require(raw.file->pMethods->xSync(raw.file, SQLITE_SYNC_FULL) == SQLITE_OK, "initial raw sync");
                for(const auto offset:{std::size_t(4095),std::size_t(8193),options.extentBytes+12289}){
                    const std::array<char,3> update{'x','y','z'};
                    require(raw.file->pMethods->xWrite(raw.file,update.data(),int(update.size()),sqlite3_int64(offset))==SQLITE_OK,"dirty subpage update");
                    std::copy(update.begin(),update.end(),bytes.begin()+std::ptrdiff_t(offset));
                }
                require(raw.file->pMethods->xSync(raw.file, SQLITE_SYNC_FULL) == SQLITE_OK, "dirty subpage sync");
                const std::string patch = "rewritten across boundary";
                const auto pos = sqlite3_int64(options.extentBytes - 4);
                require(raw.file->pMethods->xWrite(raw.file, patch.data(), int(patch.size()), pos) == SQLITE_OK, "partial write");
                std::memcpy(bytes.data() + pos, patch.data(), patch.size());
                require(raw.file->pMethods->xRead(raw.file, got.data(), int(got.size()), 0) == SQLITE_OK && bytes == got, "evict/reload changed bytes");
                require(raw.file->pMethods->xTruncate(raw.file, 100) == SQLITE_OK, "truncate");
                require(raw.file->pMethods->xTruncate(raw.file, 2 * options.extentBytes) == SQLITE_OK, "reextend");
                std::vector<char> zeros(2 * options.extentBytes - 100, 1);
                require(raw.file->pMethods->xRead(raw.file, zeros.data(), int(zeros.size()), 100) == SQLITE_OK, "read extended range");
                require(std::all_of(zeros.begin(), zeros.end(), [](char x) { return x == 0; }), "truncated content reappeared");
                require(raw.file->pMethods->xSync(raw.file, SQLITE_SYNC_FULL) == SQLITE_OK, "raw sync");
                std::array<char, 16> tail{}; tail.fill(1);
                require(raw.file->pMethods->xRead(raw.file, tail.data(), int(tail.size()), 2 * options.extentBytes) == SQLITE_IOERR_SHORT_READ, "short read status");
                require(std::all_of(tail.begin(), tail.end(), [](char x) { return x == 0; }), "short read not zero-filled");
            }
            // Deliberately abort rather than publish raw non-SQLite test bytes.
        }
        require(!std::filesystem::exists(temp.path / "raw.vcobj"), "aborted container published");

        // Closing a delete-on-close journal handle cannot retire bytes still
        // owned by another handle. Deletion happens after the final close.
        {
            PackedStoreBuilder builder(temp.path/"close-raw.vcobj",options);
            const auto path=builder.addDatabase(util::newUuid()).string()+"-journal";
            auto temporary=std::make_unique<RawFile>(path,true,SQLITE_OPEN_TEMP_JOURNAL|SQLITE_OPEN_DELETEONCLOSE);
            auto survivor=std::make_unique<RawFile>(path,true,SQLITE_OPEN_TEMP_JOURNAL);
            const std::array<char,4> expected{'t','e','s','t'};std::array<char,4> got{};
            require(temporary->file->pMethods->xWrite(temporary->file,expected.data(),int(expected.size()),0)==SQLITE_OK,"temporary journal write");
            temporary.reset();
            require(survivor->file->pMethods->xRead(survivor->file,got.data(),int(got.size()),0)==SQLITE_OK&&got==expected,"close retired another handle's bytes");
            survivor.reset();int exists=1;auto* vfs=sqlite3_vfs_find(packedStoreVfs());
            require(vfs->xAccess(vfs,path.c_str(),SQLITE_ACCESS_EXISTS,&exists)==SQLITE_OK&&!exists,"journal not deleted after final close");
        }

        // Independent writers must remain correct after their working sets
        // exceed the shared cache and nonresident page I/O runs concurrently.
        {
            auto parallelOptions=options;parallelOptions.disposableBulk=true;
            PackedStoreBuilder builder(temp.path/"parallel-raw.vcobj",parallelOptions);
            std::vector<std::filesystem::path> paths;
            for(int i=0;i<8;++i)paths.push_back(builder.addDatabase(util::newUuid()));
            std::vector<std::future<void>> workers;
            for(int i=0;i<8;++i)workers.push_back(std::async(std::launch::async,[&,i]{
                RawFile raw(paths[i],true);std::vector<char> expected(3*options.extentBytes,char('a'+i));
                require(raw.file->pMethods->xWrite(raw.file,expected.data(),int(expected.size()),0)==SQLITE_OK,"parallel raw initialization");
                for(int page=0;page<96;++page){
                    const auto offset=std::size_t((page*47)%512)*4096+13;
                    const std::array<char,3> patch{char(i),char(page),char(page+1)};
                    require(raw.file->pMethods->xWrite(raw.file,patch.data(),int(patch.size()),sqlite3_int64(offset))==SQLITE_OK,"parallel evicted write");
                    std::copy(patch.begin(),patch.end(),expected.begin()+std::ptrdiff_t(offset));
                    std::array<char,3> got{};
                    require(raw.file->pMethods->xRead(raw.file,got.data(),int(got.size()),sqlite3_int64(offset))==SQLITE_OK&&got==patch,"parallel evicted read");
                }
                std::vector<char> got(expected.size());
                require(raw.file->pMethods->xRead(raw.file,got.data(),int(got.size()),0)==SQLITE_OK&&got==expected,"parallel persisted bytes changed");
            }));
            for(auto& worker:workers)worker.get();
        }

        // Real ObjectStore operations route automatically, including history-off
        // creation through the writer VFS, sealing, and ordinary root-path open.
        const auto objectContainer = temp.path / "objects.vcobj";
        const auto objectId = util::newUuid();
        {
            PackedStoreBuilder builder(objectContainer, options);
            OpenOptions opts; opts.author = "packed-test"; opts.history = HistoryMode::Off; opts.spatialIndex = false;
            { auto store = ObjectStore::create(builder.addDatabase(objectId), opts); }
            builder.finalize(objectId);
        }
        { OpenOptions opts; opts.readOnly = true; auto store = ObjectStore::open(objectContainer, opts); }

        // Collection repacking changes only physical locations and remains
        // self-contained after both the container and original bases move.
        const auto basePath = temp.path / "linked-base.vcobj", derivedPath = temp.path / "derived.vcobj";
        OpenOptions collectionOptions; collectionOptions.author = "packed-test";
        collectionOptions.history = HistoryMode::Off; collectionOptions.spatialIndex = false;
        ObjectKey key;
        std::vector<uint8_t> originalGeometry;
        {
            auto base = ObjectStore::create(basePath, collectionOptions); auto write = base.write();
            ObjectRecord record; record.kind = "fiber"; record.fiber.emplace();
            key = {base.uuid(write), base.upsertObject(write, record)};
            vc::geometry::Fiber shape;
            for (int i = 0; i < 60000; ++i) shape.linePoints.push_back({double(i), 2, 3});
            const auto revision = base.putGeometry(write, key.objectUuid, vc::geometry::encode(shape));
            base.seal(write); write.commit();
            auto read = base.read(); originalGeometry = base.readGeometryArray(read, revision, "line_points");
        }
        std::int64_t generation = 0;
        {
            auto derived = ObjectCollection::create(derivedPath, {basePath}, collectionOptions);
            auto write = derived.write(); derived.setMeta(write, key, "family", "\"H\""); derived.seal(write); write.commit();
            auto read = derived.read(); generation = derived.provisioning(read).generation;
        }
        packObjectCollection(derivedPath, temp.path / "linked-packed.vcobj", options);
        const auto branchPath = temp.path / "branch.vcobj", dagPath = temp.path / "dag.vcobj";
        auto deformed = originalGeometry;
        double changed = 123;
        std::memcpy(deformed.data(), &changed, sizeof(changed));
        {
            auto branch = ObjectCollection::create(branchPath, {basePath}, collectionOptions);
            std::string revision; { auto read = branch.read(); revision = branch.currentGeometry(read, key)->uuid; }
            GeometryArrayEdit edit{"line_points", 0, std::vector<uint8_t>(sizeof(changed))};
            std::memcpy(edit.bytes.data(), &changed, sizeof(changed));
            auto write = branch.write(); branch.editGeometry(write, key, {edit}, revision); branch.seal(write); write.commit();
        }
        {
            // DFS order is root, derived, common base, branch. The branch's
            // borrowed chunks point backward in that enumeration, but forward
            // along a valid dependency edge; vector order must not reject it.
            auto dag = ObjectCollection::create(dagPath, {derivedPath, branchPath}, collectionOptions);
            { auto read = dag.read(); require(dag.readGeometryArray(read, key, "line_points") == deformed, "shared-base DAG geometry resolution"); }
            auto write = dag.write(); dag.seal(write); write.commit();
        }
        packObjectCollection(dagPath, temp.path / "dag-packed.vcobj", options);
        std::filesystem::rename(basePath, temp.path / "old-linked-base.vcobj");
        std::filesystem::rename(temp.path / "linked-packed.vcobj", temp.path / "relocated-packed.vcobj");
        {
            auto packed = ObjectCollection::open(temp.path / "relocated-packed.vcobj", OpenOptions{.readOnly=true});
            auto read = packed.read();
            require(packed.provisioning(read).generation == generation, "physical repacking changed logical generation");
            require(packed.object(read, key)->record.meta.find("H") != std::string::npos, "packed overlay or dependency lost");
            require(packed.readGeometryArray(read, key, "line_points") == originalGeometry, "packed inherited geometry changed");
        }
        { auto dag = ObjectCollection::open(temp.path / "dag-packed.vcobj", OpenOptions{.readOnly=true}); auto read = dag.read();
          require(dag.readGeometryArray(read, key, "line_points") == deformed, "packed shared-base DAG borrowed payload changed"); }

        // sqlite3_backup must include committed pages still in a WAL.
        const auto wal = temp.path / "wal.sqlite";
        Connection source(wal, true, false);
        source.exec("PRAGMA journal_mode=WAL; CREATE TABLE example(value); INSERT INTO example VALUES(47);");
        {
            PackedStoreBuilder builder(temp.path / "backup.vcobj", options);
            builder.importDatabase(root, wal); builder.finalize(root);
        }
        { Connection db(packedDatabasePath(temp.path / "backup.vcobj"), false); require(db.scalar("SELECT value FROM example") == 47, "committed WAL contents missing"); }

        // Failed publication must preserve the preexisting destination and clean
        // every private staging file when the builder is destroyed.
        const auto failure = temp.path / "conflict.vcobj";
        {
            PackedStoreBuilder builder(failure, options);
            const auto path = builder.addDatabase(root);
            { Connection db(path); db.exec("CREATE TABLE a(value)"); }
            { std::ofstream existing(failure); existing << "leave me alone"; }
            mustFail([&] { builder.finalize(root); });
        }
        { std::ifstream existing(failure); std::string value; std::getline(existing, value); require(value == "leave me alone", "failed publication overwrote destination"); }
        for (const auto& entry : std::filesystem::directory_iterator(temp.path)) {
            const auto name = entry.path().filename().string();
            if (name.find(".building-") != std::string::npos || name.find(".compact-") != std::string::npos)
                throw std::runtime_error("staging file leaked: " + name);
        }
        const auto corrupt = temp.path / "corrupt.vcobj";
        std::filesystem::copy_file(compact, corrupt);
        { std::fstream file(corrupt, std::ios::binary | std::ios::in | std::ios::out); file.seekp(-1, std::ios::end); file.put('!'); }
        mustFail([&] { packedDatabases(corrupt); });
        std::filesystem::resize_file(corrupt, 80);
        mustFail([&] { packedDatabases(corrupt); });
#ifndef _WIN32
        // Force a real asynchronous backing-write failure without test-only
        // branches in production code. The process limit/signal disposition is
        // restored after the builder joins its worker and removes staging.
        {
            ScopedFileLimit limited;
            PackedStoreBuilder builder(temp.path / "io-failure.vcobj", options);
            const auto id = util::newUuid(); const auto path = builder.addDatabase(id);
            {
                RawFile raw(path, true); std::vector<char> bytes(2 * options.extentBytes, 'f');
                const auto status = raw.file->pMethods->xWrite(raw.file, bytes.data(), int(bytes.size()), 0);
                const auto sync = raw.file->pMethods->xSync(raw.file, SQLITE_SYNC_FULL);
                require(status != SQLITE_OK || sync != SQLITE_OK, "backing failure did not propagate");
            }
            mustFail([&] { builder.finalize(id); });
        }
        require(!std::filesystem::exists(temp.path / "io-failure.vcobj"), "failed backing writes published a container");
#endif
        std::cout << "packed VFS tests passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
