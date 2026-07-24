#include "SeedingBatchTracker.hpp"

void SeedingBatchTracker::reset()
{
    _kind.clear();
    _total = 0;
    _completed = 0;
    _failures = 0;
    _failureMessages.clear();
    _cancelRequested = false;
    _finalized = false;
    _terminalKeys.clear();
    _result = Result{};
}

void SeedingBatchTracker::begin(const QString& kind, int total)
{
    reset();
    _kind = kind;
    _total = total;
}

bool SeedingBatchTracker::recordTerminal(int key, bool failedToStart,
                                         bool crashed, int exitCode,
                                         const QString& tail, const QString& label)
{
    // Each child contributes to completion exactly once.
    if (_terminalKeys.contains(key)) {
        return false;
    }
    _terminalKeys.insert(key);

    const bool failed = failedToStart || crashed || exitCode != 0;
    if (failed) {
        _failures++;
        QString reason;
        if (failedToStart) {
            reason = QStringLiteral("failed to start (missing executable or priority wrapper)");
        } else if (crashed) {
            reason = QStringLiteral("crashed");
        } else {
            reason = QStringLiteral("exit code %1").arg(exitCode);
        }
        QString diag = label.isEmpty() ? reason
                                       : QStringLiteral("%1: %2").arg(label, reason);
        const QString trimmed = tail.trimmed();
        if (!trimmed.isEmpty()) {
            diag += QStringLiteral(" [%1]").arg(trimmed.right(200));
        }
        // Bound the retained failure tail.
        _failureMessages.append(diag);
        while (_failureMessages.size() > 10) {
            _failureMessages.removeFirst();
        }
    }

    _completed++;
    return true;
}

void SeedingBatchTracker::requestCancel()
{
    _cancelRequested = true;
}

SeedingBatchTracker::Result SeedingBatchTracker::finalize()
{
    // Idempotent: cancel + the last child completion can both reach here.
    if (_finalized) {
        return _result;
    }
    _finalized = true;

    Result r;
    r.completed = _completed;
    r.total = _total;
    r.canceled = _cancelRequested;
    r.success = !_cancelRequested && _failures == 0;

    if (r.success) {
        r.message = QStringLiteral("Seeding %1 finished: %2/%3")
                        .arg(_kind).arg(r.completed).arg(r.total);
    } else if (r.canceled) {
        r.message = QStringLiteral("Seeding %1 canceled after %2/%3")
                        .arg(_kind).arg(r.completed).arg(r.total);
    } else {
        const QString diag = _failureMessages.join(QStringLiteral("; "));
        r.message = QStringLiteral("Seeding %1 failed: %2 of %3 child processes failed; %4")
                        .arg(_kind).arg(_failures).arg(r.total).arg(diag);
    }

    _result = r;
    // Clear the live kind; the cached result keeps repeat finalization idempotent.
    _kind.clear();
    return r;
}
