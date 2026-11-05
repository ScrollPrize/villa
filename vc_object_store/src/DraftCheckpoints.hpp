#pragma once
// Durable session drafts: checkpoints of every draft and every pending
// checked submission in one local directory.
//
// Layout:
//   checkpoint.json              the last completed checkpoint (replaced atomically)
//   draft-<object>-<seq>.bin     a draft's grid as of checkpoint <seq>
//   request-<uuid>.json          a pending submission's canonical request
//   payload-<sha256>.bin         its payloads
// A checkpoint writes only drafts that changed since the previous one, each to
// a new file, and then replaces checkpoint.json; files it no longer names are
// removed afterwards. A crash or a failed write therefore leaves the previous
// checkpoint complete. Requests are written (durably) before they are sent.
// One session uses a directory at a time.

#include "internal/Ownership.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/objects/Transactions.hpp"

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace vc::objects::detail {

struct SurfaceDraft;

// What a checkpoint keeps of a draft besides its grid.
struct DraftMeta {
    std::optional<std::string> storeUuid; // the attachment it belongs to
    std::optional<ObjectVersion> base;
    std::optional<uint64_t> submittedRevision;
    std::optional<std::string> pendingRequest;
    std::string conflict;
};

struct PendingRecord {
    std::string requestUuid, storeUuid;
    // Drafts the submission saves, with the draft revision it captured.
    std::map<std::string, uint64_t> drafts;
};

struct RecoveredDraft {
    std::string objectUuid;
    DraftMeta meta;
    // The stored grid is exactly what the pending submission captured.
    bool submittedAsStored = false;
    CoordinateJson coordinates;
    vc::geometry::Surface surface;
};

class DraftDirectory {
public:
    explicit DraftDirectory(const std::filesystem::path& directory);
    const std::filesystem::path& path() const { return directory_; }

    struct Loaded {
        uint64_t sequence = 0;
        std::vector<RecoveredDraft> drafts;
        std::vector<PendingRecord> pending;
    };
    // The last completed checkpoint (empty when there is none).
    Loaded load();
    // A loaded draft was installed again at `revision`: its file still holds it.
    void adopt(const std::string& objectUuid, uint64_t revision) { if (files_.contains(objectUuid)) written_[objectUuid] = revision; }

    struct Capture {
        std::string objectUuid;
        std::shared_ptr<const SurfaceDraft> draft; // its revision decides whether its file is rewritten
        DraftMeta meta;
    };
    // Writes a checkpoint of these drafts and pending submissions; returns its
    // sequence. Throws (and keeps the previous checkpoint) when it cannot.
    uint64_t write(const std::vector<Capture>& drafts, const std::vector<PendingRecord>& pending);

    void storeRequest(const std::string& requestUuid, const std::string& canonicalJson, const TransactionPayloads&);
    TransactionRequest loadRequest(const std::string& requestUuid) const;

private:
    std::filesystem::path directory_;
    std::unique_ptr<OwnershipGuard> guard_;
    uint64_t sequence_ = 0;
    // Draft object -> (revision, file) of the last checkpoint written here.
    std::map<std::string, uint64_t> written_;
    std::map<std::string, std::string> files_;
};

// A session's durable draft state. `meta`, `pending` and `checkpointed` are
// guarded by the session mutex; `writing` serializes directory writes.
struct DurableDrafts {
    std::unique_ptr<DraftDirectory> directory;
    std::map<std::string, DraftMeta> meta;
    std::map<std::string, PendingRecord> pending;
    std::map<std::string, uint64_t> checkpointed; // draft revisions in the last checkpoint
    std::string checkpointError;
    std::mutex writing;
};

} // namespace vc::objects::detail
