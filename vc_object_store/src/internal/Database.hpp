#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"

#include "internal/Sqlite.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <ctime>
#include <mutex>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <nlohmann/json.hpp>

#include "vc_object_store/Uuid.hpp"

namespace vc::objects {
namespace detail { class GeometryReadCache; class SpatialReadCache; }
// Connections are leased to one thread per transaction. Cache idle statements
// on the connection; the thread-local registry only identifies the active lease.
struct StatementPool {
    bool readSnapshot = false;
    bool checkedAction = false, actionConflict = false;
    std::map<std::string, std::vector<int64_t>> components;
    size_t componentNodes = 0;
    std::shared_ptr<detail::GeometryReadCache> geometryCache;
    std::shared_ptr<detail::SpatialReadCache> spatialCache;
    std::unordered_multimap<std::string, sqlite3_stmt*> idle;
    ~StatementPool() { for (auto& [sql, stmt] : idle) sqlite3_finalize(stmt); }
};
inline thread_local std::unordered_map<sqlite3*, StatementPool*> activeStatementPools;
namespace detail {
// Called only after a read connection has pinned its new SQLite snapshot.
void beginReadCaches(sqlite3*, StatementPool&, size_t geometryBudget, size_t spatialBudget);
void beginSpatialRead(StatementPool&, size_t budget, int version);
}
struct ObjectStore::Impl {
    std::filesystem::path path;
    OpenOptions options;
    std::string hostname;
    sqlite3* writer = nullptr;
    std::mutex writerMutex;
    std::vector<sqlite3*> readerPool;
    std::mutex poolMutex;
    std::condition_variable poolCv;
    std::map<sqlite3*, std::unique_ptr<StatementPool>> statementPools;

    ~Impl()
    {
        statementPools.clear();
        for (sqlite3* c : readerPool)
            if (c) sqlite3_close(c);
        if (writer) sqlite3_close(writer);
    }

    sqlite3* checkoutReader()
    {
        std::unique_lock<std::mutex> lock(poolMutex);
        poolCv.wait(lock, [&] { return !readerPool.empty(); });
        sqlite3* conn = readerPool.back();
        readerPool.pop_back();
        return conn;
    }
    void checkinReader(sqlite3* conn)
    {
        {
            std::lock_guard<std::mutex> lock(poolMutex);
            readerPool.push_back(conn);
        }
        poolCv.notify_one();
    }
};

struct ObjectStoreImplAccess {
    static sqlite3* conn(const Txn& txn)
    {
        if (!txn.active())
            throw std::runtime_error("object store: use of finished transaction");
        return static_cast<sqlite3*>(txn.handle_);
    }
};

} // namespace vc::objects
