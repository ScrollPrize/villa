#include "agent_bridge/AgentBridgeServer.hpp"
#include "agent_bridge/AgentBridgeAccessible.hpp"
#include "agent_bridge/AgentBridgeInternal.hpp"
#include "agent_bridge/AgentBridgeUi.hpp"

#include <QApplication>
#include <QDockWidget>
#include <QJsonArray>
#include <QWidget>

#include "CWindow.hpp"
#include "StatusDockPanelHost.hpp"

// ---------------------------------------------------------------------------
// ui.inspect: a read-only view of the main window through Qt's accessibility
// interfaces (AgentBridgeAccessible.cpp). Other top-level windows, the
// settings dialog among them, are never read.
// ---------------------------------------------------------------------------

namespace {

// True when `widget` belongs to the main window: its parent chain reaches the
// window without passing through another top-level window other than a
// floating dock.
bool inMainWindow(const QWidget* widget, const QWidget* window)
{
    for (const QWidget* w = widget; w; w = w->parentWidget()) {
        if (w == window)
            return true;
        if (w->isWindow() && !qobject_cast<const QDockWidget*>(w))
            return false;
    }
    return false;
}

// A dock title as a person reads it: no mnemonic markers.
QString dockTitle(const QDockWidget* dock)
{
    QString title = dock->windowTitle();
    title.replace(QStringLiteral("&&"), QStringLiteral("\x01"));
    title.remove(QLatin1Char('&'));
    title.replace(QStringLiteral("\x01"), QStringLiteral("&"));
    return title.trimmed();
}

}  // namespace


QDockWidget* AgentBridgeServer::findDock(const QString& objectName) const
{
    if (!_window)
        return nullptr;
    QDockWidget* found = nullptr;
    for (QDockWidget* dock : _window->findChildren<QDockWidget*>(objectName)) {
        if (!inMainWindow(dock, _window))
            continue;
        if (!found || (!found->isVisible() && dock->isVisible()))
            found = dock;
    }
    return found;
}


QWidget* AgentBridgeServer::dockBarPage(const QDockWidget* dock) const
{
    if (!_window || !_window->_statusDockPanelHost || !dock)
        return nullptr;
    return _window->_statusDockPanelHost->panelPage(dock);
}


QJsonArray AgentBridgeServer::docksJson() const
{
    QJsonArray docks;
    if (!_window)
        return docks;
    for (QDockWidget* dock : _window->findChildren<QDockWidget*>()) {
        if (dock->objectName().isEmpty() || !inMainWindow(dock, _window))
            continue;
        QJsonObject entry;
        entry["objectName"] = dock->objectName();
        entry["title"] = dockTitle(dock);
        // A dock in the bottom dock bar shows its content on a panel page.
        QWidget* page = dockBarPage(dock);
        entry["visible"] = page ? page->isVisible() : dock->isVisible();
        entry["floating"] = dock->isFloating();
        entry["dockBar"] = page != nullptr;
        const QString workspace = workspaceIdFor(dock);
        entry["workspace"] = workspace.isEmpty() ? QJsonValue(QJsonValue::Null)
                                                 : QJsonValue(workspace);
        docks.append(entry);
    }
    return docks;
}


QJsonObject AgentBridgeServer::handleUiInspect(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    const QString root = p.value("root").toString(QStringLiteral("window"));

    QWidget* rootWidget = nullptr;
    QString kind;
    QString name;
    if (root == QLatin1String("window")) {
        rootWidget = _window;
    } else if (root == QLatin1String("focused")) {
        QWidget* focus = QApplication::focusWidget();
        if (!focus && _window)
            focus = _window->focusWidget();
        if (!focus || !inMainWindow(focus, _window)) {
            QJsonObject data;
            data["kind"] = "widget";
            data["detail"] = "no widget of the main window has keyboard focus";
            throw AgentBridgeError{-32007, "No focused widget", data};
        }
        rootWidget = focus;
    } else if (splitAgentBridgeTarget(root, &kind, &name)) {
        if (kind == QLatin1String("dock")) {
            QDockWidget* dock = findDock(name);
            QWidget* page = dockBarPage(dock);
            rootWidget = page ? page : static_cast<QWidget*>(dock);
        } else {
            rootWidget = workspacePage(name);
        }
        if (!rootWidget) {
            QJsonObject data;
            data["kind"] = kind;
            data["id"] = name;
            throw AgentBridgeError{-32007,
                kind == QLatin1String("dock") ? QStringLiteral("Dock not found")
                                              : QStringLiteral("Workspace not available"),
                data};
        }
    } else {
        throwParamError("root", QStringLiteral(
            "must be \"window\", \"focused\", \"dock:<objectName>\" or \"workspace:<id>\""));
    }
    if (!rootWidget)
        throw AgentBridgeError{-32010, "Main window unavailable", {}};

    AgentBridgeInspectOptions options;
    options.maxDepth = p.value("depth").toInt(8);
    options.includeHidden = p.value("includeHidden").toBool(false);
    for (const ViewerEntry& e : _viewers) {
        if (e.object)
            options.viewerIds.insert(e.object.data(), e.id);
    }
    const AgentBridgeInspection inspection = agentBridgeInspect(rootWidget, options);

    QJsonObject result;
    result["root"] = root;
    result["tree"] = inspection.tree;
    result["nodeCount"] = inspection.nodeCount;
    result["truncated"] = inspection.truncated;
    result["docks"] = docksJson();
    result["workspace"] = workspacesJson();
    return result;
}
