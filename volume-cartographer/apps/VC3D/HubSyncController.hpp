#pragma once

#include <QObject>
#include <QStringList>
#include <QTimer>

#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace vc::objects { class ObjectStore; }

// Keeps an attached hosted replica in step with its hub (vc_object_store
// Hub.hpp): background pull/push, leases for the fibers being edited, renewal
// and release. Inactive for ordinary stores and read-only attachments.
class HubSyncController : public QObject
{
    Q_OBJECT

public:
    explicit HubSyncController(QObject* parent = nullptr);
    ~HubSyncController() override;

    void attach(const QString& path, const QString& author, bool readOnly);
    void detach();
    bool active() const { return static_cast<bool>(_replica); }

    // Blocks until this replica leases the objects, pulling once if the hub
    // is ahead of it. Throws when another replica holds one of them.
    void ensureLeases(const std::vector<std::string>& objects);
    // Objects whose leases are renewed; other leases are released once pushed.
    void setEditingObjects(std::function<std::set<std::string>()> provider);
    void syncNow();
    // Schedules a push shortly after local commits.
    void localCommitted();
    // keep: "hub" discards local changes, "local" publishes them.
    void resolveConflicts(const QString& keep);
    QString statusText() const { return _status; }

signals:
    void statusChanged(const QString& text);
    void conflictsFound(const QStringList& uuids);

private:
    struct Outcome {
        QString status;
        QStringList conflicts;
    };
    static Outcome syncOnce(vc::objects::ObjectStore& replica, const std::set<std::string>& editing);
    void runJob(std::function<Outcome()> job);
    void runSync();
    void setStatus(const QString& text);

    std::shared_ptr<vc::objects::ObjectStore> _replica;
    // Serializes hub access between the worker and the GUI thread.
    std::shared_ptr<std::mutex> _hubMutex = std::make_shared<std::mutex>();
    QTimer _syncTimer, _pushTimer;
    std::function<std::set<std::string>()> _editing;
    uint64_t _generation = 0;
    bool _busy = false, _again = false, _manual = false;
    QStringList _reportedConflicts;
    QString _status;
};
