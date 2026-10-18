#pragma once

// ui.inspect's tree, read from Qt's accessibility interfaces (the model a
// screen reader uses): roles, names from form labels and buddies, values,
// ranges, and states are Qt's own, so the bridge keeps no widget-specific
// label rules. Kept free of CWindow so the offscreen widget test can drive
// it.

#include <QHash>
#include <QJsonObject>
#include <QString>

class QObject;
class QWidget;

struct AgentBridgeInspectOptions {
    int maxDepth = 8;
    bool includeHidden = false;
    int maxNodes = 4000;
    // Viewer panes, reported as leaves carrying their bridge viewer id.
    QHash<const QObject*, QString> viewerIds;
};

struct AgentBridgeInspection {
    QJsonObject tree;
    int nodeCount = 0;
    bool truncated = false;
};

// Reads the tree below `root`. Other top-level windows, such as dialogs, are
// never entered (Qt's accessibility tree leaves them out). A text entry that
// holds a credential (agentBridgeWidgetShowsSecret) reports no value, and any
// name, value, description or item that contains credential-shaped text is
// left out; both set "redacted": true on the node.
AgentBridgeInspection agentBridgeInspect(QWidget* root, const AgentBridgeInspectOptions& options);

// True for a text entry whose echo mode hides input or whose name, row label,
// placeholder, tooltip or object name names a credential, and for any widget
// whose text contains credential-shaped content. ui.inspect withholds what
// such a widget shows; screenshot.capture paints over it.
bool agentBridgeWidgetShowsSecret(QWidget* widget);
