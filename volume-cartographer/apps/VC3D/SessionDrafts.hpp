#pragma once
// Crash recovery of the viewer session's drafts (vc_object_store durable
// drafts): one draft directory per VC3D instance, checkpoints on a timer that
// is independent of the editors' save triggers, and durable saves whose
// outcome is unknown (a hosted store that did not answer) resolved when their
// store is attached and on each later checkpoint tick. Recovery restores the
// last completed checkpoint; edits after it may be lost.

#include <QFuture>
#include <QObject>
#include <QString>

#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <vc_object_store/objects/ObjectStoreSession.hpp>

class QuadSurface;

class SessionDrafts : public QObject
{
    Q_OBJECT

public:
    SessionDrafts(std::shared_ptr<vc::objects::ObjectStoreSession> session, QObject* parent = nullptr);
    // Waits for background work, then writes a last checkpoint.
    ~SessionDrafts() override;

    // Use `directory` (call before any draft exists). False, with error(),
    // when it cannot (another VC3D uses it, ...): drafts then stay in memory only.
    // intervalMs <= 0: no timer (checkpoints only before durable saves and at exit).
    bool start(const std::filesystem::path& directory, int intervalMs);
    bool active() const { return _active; }
    QString error() const { return _error; }

    // Drafts recovered from the last session that the user has not restored
    // or discarded yet (and that the session still holds).
    bool recovered(const std::string& objectUuid) const;
    std::vector<std::string> recoveredDrafts() const;
    // The user restored or discarded it.
    void forgetRecovered(const std::string& objectUuid);

    // Apply the recovered grid and validity together before the editor saves it.
    static void restoreSurface(QuadSurface& surface, const vc::geometry::Surface& draft);

    // Resolve, in the background, pending saves of attached stores: those
    // recovered at start, and those still pending one tick after they were
    // first seen (a save in flight resolves itself).
    void resolvePending();
    // Write a checkpoint now (blocking); false when it failed (error in draftCheckpointError()).
    bool checkpointNow();

signals:
    void checkpointFailed(const QString& error);
    void checkpointRestored();
    // A pending save was resolved (committed, or concluded without committing).
    void pendingResolved(const QString& message);

private:
    void tick();
    void reportCheckpoint(const QString& error);

    std::shared_ptr<vc::objects::ObjectStoreSession> _session;
    bool _active = false;
    QString _error;
    QString _checkpointError;
    std::set<std::string> _recovered;
    QFuture<void> _checkpoint;
    QFuture<void> _resolving;
    std::set<std::string> _stalePending; // request UUIDs resolvePending may resend
    std::set<std::string> _seenPending;  // seen at the previous tick
};
