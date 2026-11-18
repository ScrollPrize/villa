#pragma once

#include <cstddef>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace vc::objects {

// Container version is independent of the logical vcobj and SQLite schema versions.
inline constexpr unsigned packedStoreFormatVersion = 1;

struct PackedStoreMemoryStats {
    std::atomic<std::size_t> residentBytes{0},peakResidentBytes{0};
};

struct PackedStoreOptions {
    // Optional diagnostics, updated only when an extent is allocated/evicted.
    std::shared_ptr<PackedStoreMemoryStats> memoryStats;
    std::size_t extentBytes = 4 * 1024 * 1024;
    std::size_t residentBytes = 256 * 1024 * 1024;
    // Only for disposable imports: SQLite sync requests do not sync storage.
    // finalize() always performs durable publication, regardless of this setting.
    bool disposableBulk = false;
    // Disposable import only: retain all live extents in RAM, never evict or
    // spill from a producer call. Exhausting residentBytes fails the builder.
    bool memoryFirst = false;
    // Optional asynchronous persistence; false defers it until finalize().
    // Bounded-cache mode may still spill synchronously at its resident limit.
    bool backgroundFlush = true;
    // Independent compact-export writers; one is the conservative default.
    // Actual concurrency is capped by member count, hardware, and RAM budget.
    std::size_t exportWorkers = 1;
    // Set the compact output's logical payload length before copying; this is
    // sparse sizing, not physical preallocation. Final directory follows it.
    bool preSizeExport = false;
};

struct PackedDatabaseInfo {
    std::string uuid;
    std::uint64_t bytes = 0;
};

// Registers the VFS once. Building locators support independent rollback-journal
// databases; WAL is unsupported. Published locators are always read-only.
const char* packedStoreVfs();
std::string packedStoreLastError();
bool isPackedDatabasePath(const std::filesystem::path& path);
bool isBuildingPackedDatabasePath(const std::filesystem::path& path);
bool isPackedStore(const std::filesystem::path& path);
std::filesystem::path packedDatabasePath(const std::filesystem::path& container,
                                         const std::string& databaseUuid = {});
std::filesystem::path packedContainerPath(const std::filesystem::path& locator);
std::string packedDatabaseUuid(const std::filesystem::path& locatorOrContainer);
// Portable manifest reference; resolving a published member substitutes the
// current container's location, never the staging filename or an absolute path.
std::string packedMemberReference(const std::string& databaseUuid);
std::filesystem::path resolvePackedDatabasePath(const std::filesystem::path& context,
                                              const std::string& memberReference);
std::vector<PackedDatabaseInfo> packedDatabases(const std::filesystem::path& container);

// Private sibling staging files are removed on abort. Destination must not
// already exist. Close every SQLite handle before finalize(). Multiple workers
// can open different database locators, but each member still obeys ordinary
// SQLite's single-writer rule. The builder must outlive its connections.
class PackedStoreBuilder {
public:
    explicit PackedStoreBuilder(const std::filesystem::path& destination,
                                const PackedStoreOptions& options = {});
    ~PackedStoreBuilder();
    PackedStoreBuilder(const PackedStoreBuilder&) = delete;
    PackedStoreBuilder& operator=(const PackedStoreBuilder&) = delete;

    std::filesystem::path addDatabase(const std::string& databaseUuid);
    std::filesystem::path databasePath(const std::string& databaseUuid) const;
    // Copies a consistent SQLite backup, including committed WAL contents.
    void importDatabase(const std::string& databaseUuid, const std::filesystem::path& source);
    // compact removes extent slack, not SQLite freelist pages. Any error throws;
    // a failed builder cannot publish a partially finalized database.
    void finalize(const std::string& rootDatabaseUuid, bool compact = true);

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

// Repack a published container without modifying it. Destination must be new.
void compactPackedStore(const std::filesystem::path& source,
                        const std::filesystem::path& destination,
                        const PackedStoreOptions& options = {});

// Package a sealed collection and its complete dependency closure. Rewrites
// physical locators only, preserving store/database/table identities and every
// logical lineage or commit sequence. All store_references become portable internal @packed references.
void packObjectCollection(const std::filesystem::path& source,
                          const std::filesystem::path& destination,
                          const PackedStoreOptions& options = {});

} // namespace vc::objects
