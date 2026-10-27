#include "SpiralServiceManager.hpp"

QString SpiralServiceManager::inputsFrozenMessage()
{
    return tr("Fit inputs come from the service's object store and are frozen for this "
              "fit. Edit patches, point collections and fibers in the store, then use "
              "Rebuild Fit to read them; editing them from this workspace arrives with "
              "store-backed editing.");
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
