#include "CommandCatalog.hpp"

#include "Keybinds.hpp"

#include <QAction>
#include <QDockWidget>
#include <QWidget>

#include <algorithm>

CommandCatalog::CommandCatalog(QObject* parent)
    : QObject(parent)
{
}

bool CommandCatalog::add(const QString& id, QAction* action, const QString& category)
{
    if (id.isEmpty() || !action) {
        return false;
    }
    Entry* existing = find(id);
    if (existing && existing->action) {
        if (existing->action == action) {
            return true;
        }
        qWarning("CommandCatalog: command id %s is already registered", qPrintable(id));
        return false;
    }
    action->setObjectName(id);
    if (existing) {
        existing->action = action;
        existing->category = category;
    } else {
        _entries.push_back({id, action, category});
    }
    return true;
}

QAction* CommandCatalog::addShortcut(const vc3d::keybinds::ShortcutDef& def, QWidget* owner)
{
    auto* action = new QAction(QString::fromUtf8(def.description), owner);
    action->setShortcut(vc3d::keybinds::sequenceFor(def));
    action->setShortcutContext(Qt::ApplicationShortcut);
    if (owner) {
        owner->addAction(action);
    }
    add(QString::fromUtf8(def.id), action, QString::fromUtf8(def.section));
    return action;
}

bool CommandCatalog::addPanel(QDockWidget* dock, QAction* openAction)
{
    if (!dock || dock->objectName().isEmpty()) {
        return false;
    }
    return add(panelId(dock->objectName()), openAction, tr("Panels"));
}

std::optional<CommandInfo> CommandCatalog::command(const QString& id) const
{
    const Entry* entry = find(id);
    if (!entry || !entry->action) {
        return std::nullopt;
    }
    return info(*entry);
}

std::vector<CommandInfo> CommandCatalog::commands() const
{
    std::vector<CommandInfo> result;
    result.reserve(_entries.size());
    for (const Entry& entry : _entries) {
        if (entry.action) {
            result.push_back(info(entry));
        }
    }
    return result;
}

bool CommandCatalog::invoke(const QString& id)
{
    const Entry* entry = find(id);
    QAction* target = entry ? entry->action.data() : nullptr;
    if (!target || !target->isEnabled()) {
        return false;
    }
    target->trigger();
    return true;
}

QString CommandCatalog::displayText(const QString& menuText)
{
    QString result;
    result.reserve(menuText.size());
    for (qsizetype i = 0; i < menuText.size(); ++i) {
        const QChar character = menuText.at(i);
        if (character == QLatin1Char('&')) {
            if (i + 1 < menuText.size() && menuText.at(i + 1) == QLatin1Char('&')) {
                result.append(character);
                ++i;
            }
            continue;
        }
        result.append(character);
    }
    return result;
}

QString CommandCatalog::panelId(const QString& dockObjectName)
{
    QString name = dockObjectName;
    if (name.startsWith(QStringLiteral("dockWidget")) && name.size() > 10) {
        name.remove(0, 10);
    }
    if (name.endsWith(QStringLiteral("Dock")) && name.size() > 4) {
        name.chop(4);
    }
    QString id = QStringLiteral("panel.");
    for (qsizetype i = 0; i < name.size(); ++i) {
        const QChar character = name.at(i);
        if (character.isUpper()) {
            if (i > 0 && !id.endsWith(QLatin1Char('_')) && !id.endsWith(QLatin1Char('.'))) {
                id.append(QLatin1Char('_'));
            }
            id.append(character.toLower());
        } else if (character.isLetterOrNumber()) {
            id.append(character);
        } else if (!id.endsWith(QLatin1Char('_')) && !id.endsWith(QLatin1Char('.'))) {
            id.append(QLatin1Char('_'));
        }
    }
    return id;
}

CommandCatalog::Entry* CommandCatalog::find(const QString& id)
{
    const auto it = std::find_if(_entries.begin(), _entries.end(), [&id](const Entry& entry) {
        return entry.id == id;
    });
    return it == _entries.end() ? nullptr : &*it;
}

const CommandCatalog::Entry* CommandCatalog::find(const QString& id) const
{
    const auto it = std::find_if(_entries.begin(), _entries.end(), [&id](const Entry& entry) {
        return entry.id == id;
    });
    return it == _entries.end() ? nullptr : &*it;
}

CommandInfo CommandCatalog::info(const Entry& entry)
{
    const QAction* action = entry.action.data();
    CommandInfo result;
    result.id = entry.id;
    result.title = displayText(action->text());
    result.category = entry.category;
    result.shortcut = action->shortcut();
    result.checkable = action->isCheckable();
    result.checked = action->isChecked();
    result.enabled = action->isEnabled();
    return result;
}
