#pragma once

#include "vc_object_store/objects/ObjectStore.hpp"
#include <functional>
#include <mutex>
#include <condition_variable>

namespace vc::objects {

struct ObjectIdentity {
    std::string storeUuid, objectUuid;
    bool operator==(const ObjectIdentity&) const = default;
};

struct AttachmentIdentity {
    std::string storeUuid;
    uint64_t generation = 0;
    bool operator==(const AttachmentIdentity&) const = default;
};

struct PinnedObject {
    AttachmentIdentity attachment;
    ObjectIdentity object;
    std::string objectRevision;
    std::optional<std::string> geometryRevision;
    bool operator==(const PinnedObject&) const = default;
};

struct StoreSessionEvent {
    enum class Kind { Attachment, Committed };
    Kind kind = Kind::Attachment;
    // Empty storeUuid denotes detachment. Generation increases on every switch,
    // including reattachment of the same file/UUID.
    AttachmentIdentity attachment;
    int64_t journalPosition = 0;
    // Generic record UUIDs: objects AND relationships, including deleted records.
    // Consumers invalidate by identity, never by names or storage paths.
    std::vector<JournalEntry> changes;
};

// Application-owned connection and lifetime boundary, independent of object kind
// and UI toolkit. All worker work belongs inside read(); its snapshot and store
// references must not escape the callback. Callbacks must not reenter the session.
// Cancellation is cooperative. A replacement/detachment waits for accepted reads
// and checked writes to finish; the caller must not hold a worker's completion
// hostage to the thread performing that replacement.
class ObjectStoreSession {
public:
    using Read = std::function<void(const ObjectStore&, const ReadTxn&, const std::atomic_bool&)>;
    using Action = std::function<void(ObjectStore&, WriteTxn&)>;

    ObjectStoreSession() = default;
    ~ObjectStoreSession();
    ObjectStoreSession(const ObjectStoreSession&) = delete;
    ObjectStoreSession& operator=(const ObjectStoreSession&) = delete;

    // Existing database only; read-only attachments reject all actions. Validation failure preserves the attachment.
    AttachmentIdentity attach(const std::filesystem::path&, const OpenOptions&);
    void detach();
    std::optional<AttachmentIdentity> attachment() const;
    bool writable() const;
    bool current(const AttachmentIdentity&) const;

    void read(const AttachmentIdentity&, const Read&);
    // All writes are immediate checked atomic actions. The library enforces
    // expectations for every affected object and relationship endpoint.
    RevisionSelection act(const AttachmentIdentity&, const RevisionSelection&, const Action&);
    // Drain accepted writes. No delayed persistence or new geometry publication.
    void finalize();
    // Poll from the application's event loop. Includes external commits and
    // non-fiber records. Events retain attachment identity across replacements.
    std::vector<StoreSessionEvent> takeEvents();

private:
    struct Attachment;
    std::shared_ptr<Attachment> acquire(const AttachmentIdentity&, bool write);
    void release(bool write);
    void switchTo(std::shared_ptr<Attachment>);
    void observe(const std::shared_ptr<Attachment>&);

    // transition_ serializes lifecycle operations; writer_ serializes accepted
    // actions. Neither is held by read callbacks. mutex_ protects admission.
    std::mutex transition_, writer_, journal_;
    mutable std::mutex mutex_;
    std::condition_variable idle_;
    std::shared_ptr<Attachment> attached_;
    uint64_t generation_ = 0;
    size_t readers_ = 0, writers_ = 0;
    bool accepting_ = true;
    std::vector<StoreSessionEvent> events_;
};
}
