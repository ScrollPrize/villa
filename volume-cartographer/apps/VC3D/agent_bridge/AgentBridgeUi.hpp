#pragma once

// Pure helpers for the bridge's workspace methods. They depend only on QtCore
// so the contract test can exercise them without a running application.

#include <QString>
#include <QStringList>

// Workspace ids in tab order, as stored in each tab page's "workspaceId"
// property by CWindow (the setProperty calls next to the addTab calls in
// CWindow's constructor, plus the Spiral and Fiber Map pages it creates
// later). workspace.switch's descriptor enum is built from this list, and the
// offscreen smoke test checks it against the live tab bar, so a tab added to
// CWindow must be added here.
inline QStringList agentBridgeWorkspaceIds()
{
    return {
        QStringLiteral("main"),
        QStringLiteral("lasagna"),
        QStringLiteral("atlas"),
        QStringLiteral("fiber-slice"),
        QStringLiteral("intersections"),
        QStringLiteral("spiral"),
        QStringLiteral("fiber-map"),
    };
}

// workspace.switch keeps accepting the protocol's original "fiber_slice"
// spelling; every other accepted name is already a workspace id.
inline QString canonicalWorkspaceId(const QString& name)
{
    if (name == QLatin1String("fiber_slice"))
        return QStringLiteral("fiber-slice");
    return name;
}
