#include "SpiralServiceManager.hpp"

QString SpiralServiceManager::inputChangesMessage()
{
    return tr("Save edits to the store, then use Update Fit Inputs or Rebuild Fit with current inputs.");
}

QString SpiralServiceManager::StoreBinding::mismatchAdvice() const
{
    if (!advertised() || matches()) return {};
    if (localToService())
        return QObject::tr("Store is on the service host. Attach it to edit or view inputs.");
    if (localUuid.isEmpty())
        return QObject::tr("Attach this store to edit or view inputs.");
    if (serviceUuid == localUuid)
        return QObject::tr("Store history differs. Attach the service's store to edit or view inputs.");
    return QObject::tr("A different store is attached. Attach the service's store to edit or view inputs.");
}

void SpiralServiceManager::claimSessionLease()
{
    // The lease is ownership only: the service snapshots no inputs, since
    // they come from its object store and are frozen for a resident fit.
    postWithRetry(QStringLiteral("/session/lease/claim"),
        {{QStringLiteral("command_id"), commandId()}}, Timeout::Command, 2,
        [this](const QJsonObject& response) {
            const auto lease = response.value(QStringLiteral("lease_id")).toString();
            if (!_sessionLeaseId.isEmpty() && lease != _sessionLeaseId)
                clearSessionLease();
            _sessionLeaseId = lease;
            _leaseOwner = true;
            emit logMessage(tr("Session lease acquired"));
        }, [this](const QString& error) {
            _leaseOwner = false;
            emit errorOccurred(tr("Could not take the Spiral session lease: %1").arg(error));
        });
}

void SpiralServiceManager::releaseSessionLease(std::function<void()> done)
{
    if (!_leaseOwner) { if (done) done(); return; }
    postWithRetry(QStringLiteral("/session/lease/release"),
        {{QStringLiteral("command_id"), commandId()}}, Timeout::LongCommand, 2,
        [this, done](const QJsonObject&) {
            clearSessionLease();
            if (done) done();
        }, [this](const QString& error) { emit errorOccurred(error); });
}

void SpiralServiceManager::clearSessionLease()
{
    _leaseOwner = false;
    _sessionLeaseId.clear();
    emit sessionLeaseReleased();
}
