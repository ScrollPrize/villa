#include "HubSyncController.hpp"

#include "vc_object_store/objects/Hub.hpp"

#include <QApplication>
#include <QDateTime>
#include <QFutureWatcher>
#include <QScopeGuard>
#include <QtConcurrent/QtConcurrent>

#include <chrono>

#include <nlohmann/json.hpp>

namespace {
using namespace vc::objects;

constexpr int kSyncIntervalMs = 15000;
constexpr int kPushDelayMs = 2000;
// Renew a lease in use once less than this much of it remains.
constexpr int64_t kRenewMarginUs = 5 * 60 * 1000000LL;
// A lease about to lapse is not trusted for a new edit.
constexpr int64_t kEditMarginUs = 60 * 1000000LL;

int64_t nowUs()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

QStringList conflictUuids(const std::vector<HubConflict>& conflicts)
{
    QStringList result;
    for (const auto& conflict : conflicts) result << QString::fromStdString(conflict.uuid);
    return result;
}

// Renew leases still being edited and release the rest once pushed.
void reconcileLeases(ObjectStore& replica, const std::set<std::string>& editing)
{
    const auto status = hubStatus(replica);
    if (!status) return;
    std::vector<std::string> renew, release;
    for (const auto& lease : status->leases) {
        if (lease.holder != status->replica) continue;
        if (editing.contains(lease.objectUuid)) {
            if (lease.expiresUs - nowUs() < kRenewMarginUs) renew.push_back(lease.objectUuid);
        } else if (status->pending == 0) {
            release.push_back(lease.objectUuid);
        }
    }
    if (!renew.empty()) renewLeases(replica, renew);
    if (!release.empty()) releaseLeases(replica, release);
}
}

HubSyncController::HubSyncController(QObject* parent)
    : QObject(parent)
{
    _syncTimer.setInterval(kSyncIntervalMs);
    connect(&_syncTimer, &QTimer::timeout, this, &HubSyncController::runSync);
    _pushTimer.setSingleShot(true);
    _pushTimer.setInterval(kPushDelayMs);
    connect(&_pushTimer, &QTimer::timeout, this, [this] {
        if (!_replica) return;
        // Pulled changes also commit; only local edits need the hub.
        const auto status = hubStatus(*_replica);
        if (status && status->pending > 0) runSync();
    });
}

HubSyncController::~HubSyncController() = default;

void HubSyncController::attach(const QString& path, const QString& author, bool readOnly)
{
    detach();
    if (readOnly) return;
    OpenOptions options;
    options.author = author.toStdString();
    auto replica = std::make_shared<ObjectStore>(ObjectStore::open(path.toStdString(), options));
    if (!hubStatus(*replica)) return;
    _replica = std::move(replica);
    _syncTimer.start();
    setStatus(tr("Hub: connecting"));
    runSync();
}

void HubSyncController::detach()
{
    ++_generation;
    _syncTimer.stop();
    _pushTimer.stop();
    _replica.reset();
    _again = _manual = false;
    _reportedConflicts.clear();
    setStatus({});
}

void HubSyncController::setEditingObjects(std::function<std::set<std::string>()> provider)
{
    _editing = std::move(provider);
}

void HubSyncController::ensureLeases(const std::vector<std::string>& objects)
{
    if (!_replica || objects.empty()) return;
    const auto status = hubStatus(*_replica);
    std::vector<std::string> missing;
    for (const auto& object : objects) {
        const bool held = std::any_of(status->leases.begin(), status->leases.end(), [&](const auto& lease) {
            return lease.objectUuid == object && lease.holder == status->replica && lease.expiresUs - nowUs() > kEditMarginUs;
        });
        if (!held) missing.push_back(object);
    }
    if (missing.empty()) return;
    QApplication::setOverrideCursor(Qt::WaitCursor);
    const auto restore = qScopeGuard([] { QApplication::restoreOverrideCursor(); });
    std::lock_guard lock(*_hubMutex);
    try {
        acquireLeases(*_replica, missing);
    } catch (const HubLeaseRefused& refused) {
        if (!refused.held.empty() || refused.stale.empty()) throw;
        // The edit must start from the hub's record: bring it in, then retry.
        hubPull(*_replica);
        acquireLeases(*_replica, missing);
    }
}

void HubSyncController::syncNow()
{
    _manual = true;
    runSync();
}

void HubSyncController::localCommitted()
{
    if (_replica) _pushTimer.start();
}

void HubSyncController::resolveConflicts(const QString& keep)
{
    if (!_replica) return;
    const auto side = keep.toStdString();
    const auto editing = _editing ? _editing() : std::set<std::string>{};
    runJob([replica = _replica, mutex = _hubMutex, side, editing]() {
        std::lock_guard lock(*mutex);
        for (const auto& conflict : hubStatus(*replica)->conflicts) {
            const bool object = nlohmann::json::parse(conflict.localRecord).value("record_type", "") == "objects";
            if (side == "local" && object) renewLeases(*replica, {conflict.uuid});
            resolveHubConflict(*replica, conflict.uuid, side);
        }
        return syncOnce(*replica, editing);
    });
}

HubSyncController::Outcome HubSyncController::syncOnce(ObjectStore& replica, const std::set<std::string>& editing)
{
    // Unresolved conflicts block pushing until the user settles them.
    auto report = hubPull(replica);
    const auto status = hubStatus(replica);
    auto conflicts = status->conflicts;
    if (conflicts.empty()) {
        // Edits made while offline outlive their leases; pushing needs them again.
        std::vector<std::string> lapsing;
        if (status->pending > 0)
            for (const auto& lease : status->leases)
                if (lease.holder == status->replica && lease.expiresUs - nowUs() < kRenewMarginUs)
                    lapsing.push_back(lease.objectUuid);
        if (!lapsing.empty()) renewLeases(replica, lapsing);
        report = hubPush(replica);
        conflicts = report.conflicts;
    }
    if (!conflicts.empty())
        return Outcome{tr("Hub: %n conflict(s)", nullptr, int(conflicts.size())), conflictUuids(conflicts)};
    reconcileLeases(replica, editing);
    return Outcome{};
}

void HubSyncController::runSync()
{
    if (!_replica) return;
    const auto editing = _editing ? _editing() : std::set<std::string>{};
    runJob([replica = _replica, mutex = _hubMutex, editing]() {
        std::lock_guard lock(*mutex);
        return syncOnce(*replica, editing);
    });
}

void HubSyncController::runJob(std::function<Outcome()> job)
{
    if (_busy) { _again = true; return; }
    _busy = true;
    const auto generation = _generation;
    auto* watcher = new QFutureWatcher<Outcome>(this);
    connect(watcher, &QFutureWatcher<Outcome>::finished, this, [this, watcher, generation] {
        watcher->deleteLater();
        _busy = false;
        if (generation != _generation) return;
        const auto outcome = watcher->result();
        if (!outcome.status.isEmpty()) {
            setStatus(outcome.status);
        } else if (const auto status = hubStatus(*_replica)) {
            setStatus(status->pending > 0
                ? tr("Hub: %n pending", nullptr, int(status->pending))
                : tr("Hub: synced %1").arg(QDateTime::currentDateTime().toString(QStringLiteral("HH:mm:ss"))));
        }
        // Ask about a set of conflicts once, and again whenever sync is requested.
        const bool manual = std::exchange(_manual, false);
        if (!outcome.conflicts.isEmpty() && (manual || outcome.conflicts != _reportedConflicts))
            emit conflictsFound(outcome.conflicts);
        _reportedConflicts = outcome.conflicts;
        if (std::exchange(_again, false)) runSync();
    });
    watcher->setFuture(QtConcurrent::run([job = std::move(job)]() {
        try { return job(); }
        catch (const std::exception& error) {
            return Outcome{tr("Hub: offline (%1)").arg(QString::fromUtf8(error.what()).section('\n', 0, 0)), {}};
        }
    }));
}

void HubSyncController::setStatus(const QString& text)
{
    if (_status == text) return;
    _status = text;
    emit statusChanged(text);
}
