#include "SessionDrafts.hpp"

#include <QPointer>
#include <QTimer>
#include <QtConcurrent/QtConcurrent>

#include <algorithm>

using namespace vc::objects;

SessionDrafts::SessionDrafts(std::shared_ptr<ObjectStoreSession> session, QObject* parent)
    : QObject(parent), _session(std::move(session))
{
}

SessionDrafts::~SessionDrafts()
{
    _resolving.waitForFinished();
    _checkpoint.waitForFinished();
    if (_active) checkpointNow();
}

bool SessionDrafts::start(const std::filesystem::path& directory, int intervalMs)
{
    if (_active) return true;
    try {
        const auto recovery = _session->useDraftDirectory(directory);
        _recovered.insert(recovery.drafts.begin(), recovery.drafts.end());
        for (const auto& pending : recovery.pending) _stalePending.insert(pending.requestUuid);
    } catch (const std::exception& error) {
        _error = QString::fromUtf8(error.what());
        return false;
    }
    _active = true;
    if (intervalMs > 0) {
        auto* timer = new QTimer(this);
        timer->setInterval(intervalMs);
        connect(timer, &QTimer::timeout, this, &SessionDrafts::tick);
        timer->start();
    }
    return true;
}

bool SessionDrafts::recovered(const std::string& objectUuid) const
{
    return _recovered.contains(objectUuid) && _session->hasDraft(objectUuid);
}

std::vector<std::string> SessionDrafts::recoveredDrafts() const
{
    std::vector<std::string> result;
    for (const auto& uuid : _recovered)
        if (_session->hasDraft(uuid)) result.push_back(uuid);
    return result;
}

void SessionDrafts::forgetRecovered(const std::string& objectUuid)
{
    _recovered.erase(objectUuid);
}

bool SessionDrafts::checkpointNow()
{
    if (!_active) return false;
    try {
        _session->checkpointDrafts();
        reportCheckpoint({});
        return true;
    } catch (const std::exception& error) {
        reportCheckpoint(QString::fromUtf8(error.what()));
        return false;
    }
}

void SessionDrafts::reportCheckpoint(const QString& error)
{
    if (error == _checkpointError) return;
    const bool wasFailing = !_checkpointError.isEmpty();
    _checkpointError = error;
    if (!error.isEmpty()) emit checkpointFailed(error);
    else if (wasFailing) emit checkpointRestored();
}

void SessionDrafts::tick()
{
    // Pending saves seen at the previous tick are not in flight any more.
    std::set<std::string> now;
    for (const auto& pending : _session->pendingSubmissions()) {
        if (_seenPending.contains(pending.requestUuid)) _stalePending.insert(pending.requestUuid);
        now.insert(pending.requestUuid);
    }
    _seenPending = std::move(now);
    resolvePending();

    // Checkpoints write whole changed grids: off the UI thread.
    if (_checkpoint.isRunning()) return;
    QPointer<SessionDrafts> self(this);
    auto session = _session;
    _checkpoint = QtConcurrent::run([self, session] {
        QString failure;
        try { session->checkpointDrafts(); }
        catch (const std::exception& error) { failure = QString::fromUtf8(error.what()); }
        QMetaObject::invokeMethod(qApp, [self, failure] { if (self) self->reportCheckpoint(failure); }, Qt::QueuedConnection);
    });
}

void SessionDrafts::resolvePending()
{
    if (!_active || _resolving.isRunning()) return;
    std::vector<std::pair<AttachmentIdentity, std::string>> work;
    const auto attachments = _session->attachments();
    for (const auto& pending : _session->pendingSubmissions()) {
        if (!_stalePending.contains(pending.requestUuid)) continue;
        const auto attachment = std::find_if(attachments.begin(), attachments.end(),
                                             [&](const auto& a) { return a.storeUuid == pending.storeUuid; });
        if (attachment != attachments.end()) work.emplace_back(*attachment, pending.requestUuid);
    }
    if (work.empty()) return;
    QPointer<SessionDrafts> self(this);
    auto session = _session;
    _resolving = QtConcurrent::run([self, session, work] {
        for (const auto& [attachment, request] : work) {
            QString message;
            try {
                const auto receipt = session->resolvePending(attachment, request);
                message = receipt.commitSequence
                    ? tr("An earlier save was committed (commit %1).").arg(*receipt.commitSequence)
                    : tr("An earlier save was committed (it changed nothing).");
            } catch (const TransactionConflict& conflict) {
                message = tr("An earlier save conflicts with newer changes and was not committed: %1")
                              .arg(QString::fromUtf8(conflict.what()));
            } catch (const std::invalid_argument&) {
                continue; // resolved meanwhile (by the editor that made it), or invalid: concluded
            } catch (const std::exception&) {
                continue; // still unreachable: stays pending for the next tick
            }
            QMetaObject::invokeMethod(qApp, [self, request, message] {
                if (!self) return;
                self->_stalePending.erase(request);
                emit self->pendingResolved(message);
            }, Qt::QueuedConnection);
        }
    });
}
