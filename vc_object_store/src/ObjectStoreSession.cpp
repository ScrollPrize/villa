#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "hub/HubWorker.hpp"
#include <stdexcept>

namespace vc::objects {
struct ObjectStoreSession::Attachment {
    explicit Attachment(ObjectStore value) : store(std::move(value)) {}
    ObjectStore store;
    AttachmentIdentity identity;
    bool writable = true;
    std::atomic_bool cancelled{false};
    int64_t journalPosition = 0; // guarded by journal_
    // Hosted replicas only; declared after store so it stops first.
    std::unique_ptr<detail::HubWorker> hub;
};

ObjectStoreSession::~ObjectStoreSession()
{
    // switchTo drains before allocating the notification, so even an allocation
    // failure cannot destroy a handle while a worker owns a transaction.
    try { detach(); } catch (...) {}
}

AttachmentIdentity ObjectStoreSession::attach(const std::filesystem::path& path, const OpenOptions& options,
                                              const HubSessionOptions& hubOptions)
{
    std::lock_guard transition(transition_);
    auto next = std::make_shared<Attachment>(ObjectStore::open(path, options));
    // Exercise write access without changing records. Opening a read connection
    // successfully is not sufficient validation for an editing session.
    next->writable = !options.readOnly;
    if (next->writable) { auto writable = next->store.write(RevisionSelection{}); writable.commit(); }
    {
        auto txn = next->store.read();
        next->identity.storeUuid = next->store.uuid(txn);
        next->journalPosition = next->store.journalPosition(txn);
    }
    if (next->writable && vc::objects::hubStatus(next->store))
        next->hub = std::make_unique<detail::HubWorker>(next->store, hubOptions,
            [this, owner = next.get()] { hubChanged(owner); });
    switchTo(next);
    return next->identity;
}

void ObjectStoreSession::switchTo(std::shared_ptr<Attachment> next)
{
    std::unique_lock lock(mutex_);
    accepting_ = false;
    if (attached_) attached_->cancelled.store(true);
    idle_.wait(lock, [&] { return readers_ == 0 && writers_ == 0; });
    const auto previous = attached_;
    lock.unlock();
    try {
        if (previous) observe(previous);
    } catch (...) {
        lock.lock();
        if (previous) previous->cancelled.store(false);
        accepting_ = true;
        throw;
    }
    lock.lock();
    // Accepted writes have finished before announcing the new attachment.
    attached_ = std::move(next);
    const AttachmentIdentity identity{attached_ ? attached_->identity.storeUuid : "", ++generation_};
    if (attached_) attached_->identity = identity;
    accepting_ = true;
    events_.push_back({StoreSessionEvent::Kind::Attachment, identity,
                      attached_ ? attached_->journalPosition : 0, {}});
    // Stop the old replica's sync outside the lock its status reports take.
    lock.unlock();
    if (previous) previous->hub.reset();
}

void ObjectStoreSession::detach()
{
    std::lock_guard transition(transition_);
    switchTo(nullptr);
}

std::optional<AttachmentIdentity> ObjectStoreSession::attachment() const
{
    std::lock_guard lock(mutex_);
    if (!attached_) return std::nullopt;
    return attached_->identity;
}

bool ObjectStoreSession::writable() const
{
    std::lock_guard lock(mutex_);
    return accepting_ && attached_ && attached_->writable;
}

bool ObjectStoreSession::current(const AttachmentIdentity& identity) const
{
    std::lock_guard lock(mutex_);
    return accepting_ && attached_ && attached_->identity == identity;
}

std::shared_ptr<ObjectStoreSession::Attachment> ObjectStoreSession::acquire(
    const AttachmentIdentity& identity, bool write)
{
    std::lock_guard lock(mutex_);
    if (!accepting_ || !attached_ || attached_->identity != identity)
        throw std::runtime_error("object-store attachment is missing or superseded");
    if (write && !attached_->writable) throw std::runtime_error("Object store is attached read-only");
    ++(write ? writers_ : readers_);
    return attached_;
}

void ObjectStoreSession::release(bool write)
{
    std::lock_guard lock(mutex_);
    --(write ? writers_ : readers_);
    idle_.notify_all();
}

void ObjectStoreSession::read(const AttachmentIdentity& identity, const Read& callback)
{
    auto attachment = acquire(identity, false);
    try {
        auto txn = attachment->store.read();
        callback(attachment->store, txn, attachment->cancelled);
    } catch (...) { release(false); throw; }
    release(false);
}

RevisionSelection ObjectStoreSession::act(const AttachmentIdentity& identity,
    const RevisionSelection& expected, const Action& callback)
{
    RevisionSelection committed;
    auto attachment = acquire(identity, true);
    try {
        std::lock_guard writer(writer_);
        auto txn = attachment->store.write(expected);
        callback(attachment->store, txn);
        txn.commit();
        committed = txn.committedRevisions();
        if (attachment->hub) attachment->hub->requestSync(true);
    } catch (...) { release(true); throw; }
    release(true);
    return committed;
}

void ObjectStoreSession::finalize()
{
    std::lock_guard transition(transition_);
    std::unique_lock lock(mutex_);
    accepting_ = false;
    idle_.wait(lock, [&] { return writers_ == 0; });
    accepting_ = true;
}

void ObjectStoreSession::observe(const std::shared_ptr<Attachment>& attachment)
{
    std::lock_guard journal(journal_);
    auto txn = attachment->store.read();
    auto changes = attachment->store.journalSince(txn, attachment->journalPosition);
    if (changes.empty()) return;
    const auto position = changes.back().seq;
    std::lock_guard lock(mutex_);
    events_.push_back({StoreSessionEvent::Kind::Committed, attachment->identity, position, std::move(changes)});
    attachment->journalPosition = position;
}

std::vector<StoreSessionEvent> ObjectStoreSession::takeEvents()
{
    // Observation participates in the worker barrier; no old event can arrive
    // after a replacement notification.
    std::shared_ptr<Attachment> attachment;
    {
        std::lock_guard lock(mutex_);
        if (accepting_ && attached_) { attachment = attached_; ++readers_; }
    }
    if (attachment) {
        try { observe(attachment); } catch (...) { release(false); throw; }
        release(false);
    }
    std::lock_guard lock(mutex_);
    std::vector<StoreSessionEvent> result;
    result.swap(events_);
    return result;
}

LeaseHold ObjectStoreSession::holdLeases(const AttachmentIdentity& identity, const std::vector<std::string>& objects)
{
    auto attachment = acquire(identity, true);
    LeaseHold hold;
    try {
        if (attachment->hub) hold = attachment->hub->hold(objects);
    } catch (...) { release(true); throw; }
    release(true);
    return hold;
}

std::optional<HubSessionStatus> ObjectStoreSession::hubStatus() const
{
    std::lock_guard lock(mutex_);
    if (!attached_ || !attached_->hub) return std::nullopt;
    return attached_->hub->status();
}

void ObjectStoreSession::syncHub()
{
    std::lock_guard lock(mutex_);
    if (attached_ && attached_->hub) attached_->hub->requestSync(false);
}

void ObjectStoreSession::resolveHubConflicts(const std::string& keep)
{
    std::lock_guard lock(mutex_);
    if (attached_ && attached_->hub) attached_->hub->requestResolve(keep);
}

void ObjectStoreSession::hubChanged(const Attachment* owner)
{
    std::lock_guard lock(mutex_);
    if (attached_.get() == owner) events_.push_back({StoreSessionEvent::Kind::Hub, attached_->identity, 0, {}});
}
}
