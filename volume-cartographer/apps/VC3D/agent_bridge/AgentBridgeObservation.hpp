#pragma once

// Revision bookkeeping behind state.get's "revision" and the state.changed
// notification. Pure QtCore so the contract test can drive it directly.

#include <algorithm>

#include <QByteArray>
#include <QHash>
#include <QSet>
#include <QString>
#include <QStringList>

// The fields state.changed can name, one fingerprint each.
inline QStringList agentBridgeObservedFields()
{
    return {
        QStringLiteral("project"),
        QStringLiteral("volume"),
        QStringLiteral("activeSegment"),
        QStringLiteral("editing"),
        QStringLiteral("viewers"),
        QStringLiteral("workspace"),
        QStringLiteral("overlay"),
        QStringLiteral("jobs"),
    };
}

class AgentBridgeStateTracker
{
public:
    // Records the latest fingerprint of each field. The first call only
    // primes the tracker. Later calls bump the revision once when any field's
    // fingerprint differs (a field that disappears counts as changed) and add
    // the changed fields to the pending set. Returns whether anything changed.
    bool update(const QHash<QString, QByteArray>& fingerprints)
    {
        if (!_primed) {
            _fingerprints = fingerprints;
            _primed = true;
            return false;
        }
        bool changed = false;
        for (auto it = fingerprints.cbegin(); it != fingerprints.cend(); ++it) {
            const auto previous = _fingerprints.constFind(it.key());
            if (previous == _fingerprints.cend() || previous.value() != it.value()) {
                _pending.insert(it.key());
                changed = true;
            }
        }
        for (auto it = _fingerprints.cbegin(); it != _fingerprints.cend(); ++it) {
            if (!fingerprints.contains(it.key())) {
                _pending.insert(it.key());
                changed = true;
            }
        }
        _fingerprints = fingerprints;
        if (changed)
            ++_revision;
        return changed;
    }

    qint64 revision() const { return _revision; }

    // Fields changed since the previous call, sorted; clears them.
    QStringList takePending()
    {
        QStringList fields(_pending.cbegin(), _pending.cend());
        std::sort(fields.begin(), fields.end());
        _pending.clear();
        return fields;
    }

private:
    QHash<QString, QByteArray> _fingerprints;
    QSet<QString> _pending;
    qint64 _revision = 0;
    bool _primed = false;
};
