#pragma once

// Shared SQLite transactions and record metadata for objects and graphs.

#include "vc/core/util/Progress.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vc::storage {

struct FiberDetail {
    std::string family;          // "H", "V", or "" (unset)
    bool manualFamily = false;
    int64_t controlPointCount = 0;
    int64_t linePointCount = 0;
    double length = 0.0;       // full polyline arclength in source voxels
    double width = 0.0;
    double widthGapFraction = 0.2;
    std::string optimizationMode;
    std::optional<std::array<double, 6>> bbox;  // minx,miny,minz,maxx,maxy,maxz
};

struct PatchDetail {
    int64_t gridWidth = 0;
    int64_t gridHeight = 0;
    std::array<double, 2> scale{0.0, 0.0};  // column and row scale, as in tifxyz meta.json
    std::optional<std::array<double, 6>> bbox;
};

struct TrackDetail {
    std::string datasetRef;    // identity component: (datasetRef, externalKey) is unique
    std::string externalKey;   // key within the external dataset
    std::string family;
    int64_t pointCount = 0;
    double length = 0.0;       // polyline arclength in source dataset voxels
    std::optional<std::array<double, 6>> bbox;
};

// Fields shared by every record (object or edge). Library-managed
// fields (generation, stamps, approval) are read-only to callers:
// values set on input structs are ignored on write.
struct RecordCommon {
    std::string uuid;  // canonical text; empty on create = mint a new one
    std::string kind;  // required, non-empty
    int64_t generation = 0;
    std::string createdBy, createdAt, createdHost;
    std::string changedBy, changedAt, changedHost;
    std::string approvalState = "pending";  // pending | approved | rejected
    std::optional<std::string> approvedBy, approvedAt, approvedHost;
    std::optional<int64_t> approvedGeneration;
    std::vector<std::string> tags;
    std::string meta = "{}";  // JSON object text
};

struct ObjectRecord : RecordCommon {
    std::string name;
    int64_t formatVersion = 0;
    std::string storageKind;  // e.g. "file", "dataset", "" (db-only)
    std::string storageRef;   // location hint relative to DB directory, or dataset URI; uuid is identity
    int64_t dataGeneration = 0;
    std::optional<std::string> dataChecksum;
    std::optional<FiberDetail> fiber;
    std::optional<PatchDetail> patch;
    std::optional<TrackDetail> track;
};

struct Anchor {
    std::optional<int64_t> index;
    std::optional<std::array<double, 3>> position;
    std::optional<std::array<double, 3>> direction;
};

struct EdgeRecord : RecordCommon {
    std::string aUuid, bUuid;  // endpoint order is normalised by addEdge
    Anchor aAnchor, bAnchor;
};

struct JournalEntry {
    int64_t seq = 0;
    std::string recordUuid;
    int64_t generation = 0;
    std::string author, hostname, timestamp, changeKind;
    // Approval snapshot after this change; absent for deletion events.
    std::optional<std::string> approvalState, approvedBy, approvedAt, approvedHost;
    std::optional<int64_t> approvedGeneration;
};

struct OpenOptions {
    bool readOnly = false;
    int readerPoolSize = 4;
    int busyTimeoutMs = 5000;
    std::string author;  // required for read-write opens
    std::optional<std::string> hostnameOverride;  // tests / reproducible imports only
};

class RecordDb;
struct RecordDbImplAccess;

class Txn {
public:
    Txn(const Txn&) = delete;
    Txn& operator=(const Txn&) = delete;
    Txn(Txn&& other) noexcept;
    Txn& operator=(Txn&&) = delete;
    ~Txn();
    void commit();
    bool active() const { return handle_ != nullptr; }

protected:
    Txn(RecordDb* db, void* handle, bool write) : db_(db), handle_(handle), write_(write) {}
    void release(bool commit);

private:
    friend class RecordDb;
    friend struct RecordDbImplAccess;
    RecordDb* db_ = nullptr;
    void* handle_ = nullptr;  // sqlite3*
    bool write_ = false;
};

class ReadTxn : public Txn {
public:
    ReadTxn(RecordDb* db, void* handle) : Txn(db, handle, false) {}
};

class WriteTxn : public Txn {
public:
    WriteTxn(RecordDb* db, void* handle) : Txn(db, handle, true) {}
};

// Common SQLite record and transaction machinery. Graph records are local
// membership/annotation records; ObjectStore records own object identity.
// One writer and a bounded reader pool per handle. Transactions are thread-confined
// and must finish before their handle is moved/destroyed. Do not nest writes or
// checkpoint while a write is active; decode external files before starting writes.
class RecordDb {
public:
    RecordDb(RecordDb&&) noexcept;
    RecordDb& operator=(RecordDb&&) noexcept;
    RecordDb(const RecordDb&) = delete;
    RecordDb& operator=(const RecordDb&) = delete;
    virtual ~RecordDb();
    ReadTxn read();
    WriteTxn write();
    const std::filesystem::path& path() const;
    std::string hostname() const;
    std::string author() const;
    // Objects. upsertObject creates (minting a UUID when record.uuid is
    // empty) or updates by UUID; updates bump generation, restamp
    // changed_*, reset approval, and journal. Returns the UUID.
    std::string upsertObject(WriteTxn& txn, const ObjectRecord& record);
    // Refresh geometry metadata. Changed values bump generation, restamp
    // changed_*, reset approval and journal; identical values are a no-op.
    // An optional packed-store family fills an unset family; conflicts fail.
    void setTrackLength(WriteTxn& txn, const std::string& uuid, double length,
                        const std::optional<std::string>& family = std::nullopt);
    void setFiberLength(WriteTxn& txn, const std::string& uuid, double length);
    std::optional<ObjectRecord> objectByUuid(const Txn& txn, const std::string& uuid) const;
    std::vector<ObjectRecord> objectsByKind(const Txn& txn, const std::string& kind,
                                            const std::optional<std::string>& approvalState = std::nullopt) const;
    int64_t objectCount(const Txn& txn, const std::string& kind) const;
    std::vector<ObjectRecord> objectsByTag(const Txn& txn, const std::string& tag) const;
    void deleteObject(WriteTxn& txn, const std::string& uuid);  // fails while edges or retained geometry reference it

    // Approval: one generic call for objects and edges. Stamps the
    // current generation without bumping it. State must be
    // "pending", "approved" or "rejected".
    virtual void setApproval(WriteTxn& txn, const std::string& recordUuid, const std::string& state);

    // Open-ended metadata: single-key atomic updates via SQLite JSON1.
    // jsonValue is JSON text (e.g. "\"reviewed\"", "3", "true").
    // Both bump generation and reset approval, like any content change.
    void setMeta(WriteTxn& txn, const std::string& recordUuid, const std::string& key, const std::string& jsonValue);
    void removeMeta(WriteTxn& txn, const std::string& recordUuid, const std::string& key);

    std::vector<JournalEntry> journalSince(const Txn& txn, int64_t afterSeq) const;

    // Resolve a file-backed patch/fiber/point_collection by its source UUID.
    // First verify storageRef against meta.json or the source JSON, then search
    // datasetRoot recursively if the hint is stale. The default root is
    // the DB directory; an explicit root also rebases relative hints.
    // Returns an absolute path, or nullopt for unknown/unsupported/missing
    // objects. Throws on ambiguous fallback matches or traversal errors.
    // Discovery skips hidden directories and Zarr stores, does not follow
    // directory symlinks, and reads JSON only (no geometry). No DB writes
    // or persistent cache: subsequent calls see renamed/replaced files.
    std::optional<std::filesystem::path> resolveStoragePath(
        const Txn& txn, const std::string& objectUuid,
        const std::optional<std::filesystem::path>& datasetRoot = std::nullopt) const;

    // Managed geometry objects are excluded from this external-source audit.
    // Reports objects whose external data reference does not check out
    // (storage file missing, or data_generation ahead of generation).
    // Paths are resolved relative to the DB file's directory.
    std::vector<std::string> sweepExternalData(const Txn& txn) const;

    // Flush the WAL into the main DB file (safe copying). Must not be
    // called while a WriteTxn is live on this handle.
    void checkpoint();

protected:
    enum class Kind { Objects, Graph };
    struct Impl;
    static std::unique_ptr<Impl> openImpl(const std::filesystem::path&, const OpenOptions&, bool create, Kind);
    explicit RecordDb(std::unique_ptr<Impl>);
    void* connection(const Txn&) const;
    std::unique_ptr<Impl> impl_;
private:
    friend class Txn;
};
} // namespace vc::storage
