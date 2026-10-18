#include "agent_bridge/AgentBridgeServer.hpp"

#include <QJsonArray>
#include <QTabWidget>
#include <QVariant>
#include <QWidget>

#include "CWindow.hpp"

// ---------------------------------------------------------------------------
// Workspace tabs. The GUI identifies each workspace by the "workspaceId"
// property on its tab page (the same key its window-state restore uses), so the
// bridge reads that property instead of keeping its own table.
// ---------------------------------------------------------------------------

std::vector<AgentBridgeServer::WorkspaceTab> AgentBridgeServer::workspaceTabs() const
{
    std::vector<WorkspaceTab> tabs;
    QTabWidget* tabWidget = _window ? _window->_workspaceTabs : nullptr;
    if (!tabWidget)
        return tabs;
    for (int i = 0; i < tabWidget->count(); ++i) {
        QWidget* page = tabWidget->widget(i);
        const QString id = page ? page->property("workspaceId").toString() : QString();
        if (id.isEmpty())
            continue;
        tabs.push_back({id, tabWidget->tabText(i), i, page});
    }
    return tabs;
}


QWidget* AgentBridgeServer::workspacePage(const QString& id) const
{
    for (const WorkspaceTab& tab : workspaceTabs()) {
        if (tab.id == id)
            return tab.page;
    }
    return nullptr;
}


QString AgentBridgeServer::currentWorkspaceId() const
{
    QTabWidget* tabWidget = _window ? _window->_workspaceTabs : nullptr;
    QWidget* page = tabWidget ? tabWidget->currentWidget() : nullptr;
    return page ? page->property("workspaceId").toString() : QString();
}


QString AgentBridgeServer::workspaceIdFor(const QWidget* widget) const
{
    QTabWidget* tabWidget = _window ? _window->_workspaceTabs : nullptr;
    if (!tabWidget)
        return QString();
    // parentWidget() crosses floating dock windows, which stay children of
    // their workspace's QMainWindow.
    for (const QWidget* w = widget; w; w = w->parentWidget()) {
        if (tabWidget->indexOf(const_cast<QWidget*>(w)) >= 0)
            return w->property("workspaceId").toString();
    }
    return QString();
}


bool AgentBridgeServer::setCurrentWorkspace(const QString& id)
{
    QWidget* page = workspacePage(id);
    if (!page)
        return false;
    // Reuse the GUI's own switch paths where they exist; the other tabs are
    // switched exactly as a tab click does.
    if (id == QLatin1String("main"))
        _window->switchToMainWorkspace();
    else if (id == QLatin1String("lasagna"))
        _window->switchToLasagnaWorkspace();
    else if (id == QLatin1String("fiber-slice"))
        _window->switchToFiberSliceWorkspace();
    else
        _window->_workspaceTabs->setCurrentWidget(page);
    return true;
}


QJsonObject AgentBridgeServer::workspacesJson() const
{
    QJsonArray available;
    for (const WorkspaceTab& tab : workspaceTabs()) {
        QJsonObject entry;
        entry["id"] = tab.id;
        entry["title"] = tab.title;
        available.append(entry);
    }
    const QString current = currentWorkspaceId();
    QJsonObject result;
    result["current"] = current.isEmpty() ? QJsonValue(QJsonValue::Null) : QJsonValue(current);
    result["available"] = available;
    return result;
}
