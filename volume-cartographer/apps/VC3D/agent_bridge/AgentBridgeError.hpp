#pragma once

// Standalone JSON-RPC error payload. Split out of AgentBridgeServer.hpp so the
// strict wire-parsing helpers (AgentBridgeInternal.hpp) and their tests can
// throw/catch it without pulling in the full Q_OBJECT server class.

#include <QJsonObject>
#include <QString>

// Thrown by handlers to produce a JSON-RPC error response. Codes follow SPEC §2.5.
struct AgentBridgeError {
    int code;
    QString message;
    QJsonObject data;
};
