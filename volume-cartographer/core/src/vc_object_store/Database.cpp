#include "internal/Schema.hpp"

namespace vc::objects {
using namespace detail;
ObjectStore::ObjectStore(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ObjectStore::~ObjectStore() = default;
ObjectStore::ObjectStore(ObjectStore&&) noexcept = default;
ObjectStore& ObjectStore::operator=(ObjectStore&&) noexcept = default;

std::unique_ptr<ObjectStore::Impl> ObjectStore::openImpl(const std::filesystem::path& path, const OpenOptions& options, bool create)
{
    const int kApplicationId = 0x56434f42;
    const int kSchemaVersion = 6;
    if (create && options.readOnly) throw std::invalid_argument("create requires writable options");
    if (sqlite3_threadsafe() == 0)
        throw std::runtime_error("object store: SQLite build is not thread-safe");
    if (!create && !options.readOnly && options.author.empty())
        throw std::runtime_error("object store: read-write open requires an author (session identity)");
    if (create && options.author.empty())
        throw std::runtime_error("object store: create requires an author (session identity)");

    auto impl = std::make_unique<ObjectStore::Impl>();
    impl->path = path;
    impl->options = options;
    impl->options.readerPoolSize = std::max(1, options.readerPoolSize);
    impl->hostname = options.hostnameOverride.value_or(detectHostname());

    if (create) {
        std::error_code ec;
        if (std::filesystem::exists(path) && std::filesystem::file_size(path, ec) > 0)
            throw std::runtime_error("object store: create refuses existing non-empty file: " + path.string());
        impl->writer = openConnection(path, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, options, false);
        exec(impl->writer, "PRAGMA journal_mode=WAL;");
        exec(impl->writer, "PRAGMA synchronous=FULL;");
        exec(impl->writer, ("PRAGMA application_id=" + std::to_string(kApplicationId) + ";").c_str());
        exec(impl->writer, "BEGIN IMMEDIATE;");
        try {
            createSchema(impl->writer);
            exec(impl->writer, ("PRAGMA user_version=" + std::to_string(kSchemaVersion) + ";").c_str());
            exec(impl->writer, "COMMIT;");
        } catch (...) {
            exec(impl->writer, "ROLLBACK;");
            throw;
        }
    } else {
        if (!options.readOnly) {
            impl->writer = openConnection(path, SQLITE_OPEN_READWRITE, options, false);
            if (pragmaInt(impl->writer, "application_id") != kApplicationId)
                throw std::runtime_error("object store: unsupported database format: " + path.string());
            const auto version = pragmaInt(impl->writer, "user_version");
            if (version < 2 || version > kSchemaVersion)
                throw std::runtime_error("object store: unsupported database schema");
            if (version < kSchemaVersion) {
                exec(impl->writer, "BEGIN IMMEDIATE;");
                try {
                    // Recheck after acquiring the writer lock: another handle
                    // may already have migrated this file.
                    if (pragmaInt(impl->writer, "user_version") == 2) {
                        exec(impl->writer, kIncrementalSyncSchema);
                        exec(impl->writer, "PRAGMA user_version=3;");
                    }
                    if (pragmaInt(impl->writer, "user_version") == 3) {
                        detail::createSpatialSchema(impl->writer);
                        detail::backfillSpatialBounds(impl->writer);
                        exec(impl->writer, "PRAGMA user_version=5;");
                    }
                    if (pragmaInt(impl->writer, "user_version") == 4) {
                        detail::migrateSpatialRegions(impl->writer);
                        exec(impl->writer, "PRAGMA user_version=5;");
                    }
                    // v6 gates the new physical layout; retained manifests and
                    // payloads are immutable and need no eager conversion.
                    if (pragmaInt(impl->writer, "user_version") == 5)
                        exec(impl->writer, "PRAGMA user_version=6;");
                    exec(impl->writer, "COMMIT;");
                } catch (...) { exec(impl->writer, "ROLLBACK;"); throw; }
            }
            exec(impl->writer, "PRAGMA journal_mode=WAL;");
            exec(impl->writer, "PRAGMA synchronous=FULL;");
        }
    }

    // WAL readers need write access to the -shm file, so pooled readers
    // are read-write connections pinned to query_only (true READONLY
    // only for a fully read-only open).
    const int readerFlags = options.readOnly ? SQLITE_OPEN_READONLY : SQLITE_OPEN_READWRITE;
    for (int i = 0; i < impl->options.readerPoolSize; ++i) {
        sqlite3* conn = openConnection(path, readerFlags, options, true);
        impl->readerPool.push_back(conn);
        impl->statementPools.emplace(conn, std::make_unique<StatementPool>());
        if (options.readOnly) {
            if (pragmaInt(conn, "application_id") != kApplicationId)
                throw std::runtime_error("object store: unsupported database format: " + path.string());
            const auto version = pragmaInt(conn, "user_version");
            if (version < 2 || version > kSchemaVersion)
                throw std::runtime_error("object store: unsupported database schema: " + path.string());
        }
    }
    if (impl->writer) impl->statementPools.emplace(impl->writer, std::make_unique<StatementPool>());
    return impl;
}

const std::filesystem::path& ObjectStore::path() const { return impl_->path; }
std::string ObjectStore::hostname() const { return impl_->hostname; }
std::string ObjectStore::author() const { return impl_->options.author; }

ObjectStore ObjectStore::create(const std::filesystem::path& p, const OpenOptions& o) { return ObjectStore(openImpl(p,o,true)); }
ObjectStore ObjectStore::open(const std::filesystem::path& p, const OpenOptions& o) { return ObjectStore(openImpl(p,o,false)); }
std::string ObjectStore::uuid(const Txn& txn) const {
    Statement s(static_cast<sqlite3*>(connection(txn)), "SELECT uuid FROM store_identity");
    if (!s.row()) throw std::runtime_error("object store: missing identity");
    return formatUuid(s.blob16(0));
}

} // namespace vc::objects
