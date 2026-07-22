#pragma once

// Standalone JSON-RPC error payload for the Agent Bridge.
//
// Split out of AgentBridgeServer.hpp so the strict wire-parsing helpers in
// AgentBridgeInternal.hpp (and their unit tests) can throw/catch this type
// without pulling in the full Q_OBJECT server class and its heavy app/Qt-Widgets
// dependencies (CWindow, MenuActionController, ...).

#include <QJsonObject>
#include <QString>

// Thrown by handlers to produce a JSON-RPC error response. Codes follow SPEC §2.5.
struct AgentBridgeError {
    int code;
    QString message;
    QJsonObject data;
};
