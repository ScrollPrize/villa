#pragma once

#include <QKeySequence>
#include <QObject>
#include <QPointer>
#include <QString>

#include <optional>
#include <vector>

class QAction;
class QDockWidget;
class QWidget;

namespace vc3d::keybinds {
struct ShortcutDef;
}

// Snapshot of one catalog command as the command palette sees it.
struct CommandInfo {
    QString id;
    QString title;
    QString category;
    QKeySequence shortcut;
    bool checkable{false};
    bool checked{false};
    bool enabled{true};
};

// The window's user-invocable commands. Every command is a QAction the GUI
// already owns (a menu entry, a keyboard shortcut or a panel's opener); the
// catalog adds a stable id and a category, so every list of commands shares
// one source.
class CommandCatalog : public QObject
{
    Q_OBJECT

public:
    explicit CommandCatalog(QObject* parent = nullptr);

    // Registers action under id and sets the action's objectName to id. Ids are
    // unique while their action lives: a second live action under a used id is
    // rejected, and registering the same action under the same id again is a
    // no-op that succeeds. An id whose action was destroyed can be registered
    // again; the new action takes the old one's place.
    bool add(const QString& id, QAction* action, const QString& category);

    // Creates an application-wide shortcut action for def, owned by and added
    // to owner, and registers it under def.id in def.section.
    QAction* addShortcut(const vc3d::keybinds::ShortcutDef& def, QWidget* owner);

    // Registers openAction, which opens dock's status bar panel, under
    // panelId(dock->objectName()) in the Panels category.
    bool addPanel(QDockWidget* dock, QAction* openAction);

    std::optional<CommandInfo> command(const QString& id) const;

    // Live commands in registration order.
    std::vector<CommandInfo> commands() const;

    // Triggers the command, as a menu click or its key would. Returns false
    // when the id is unknown or the command is disabled.
    bool invoke(const QString& id);

    // Menu text without mnemonic markers: "&Open Project..." -> "Open Project...".
    static QString displayText(const QString& menuText);

    // "panel." and the dock's object name in snake_case without a
    // "dockWidget" prefix or a "Dock" suffix: "dockWidgetViewerControls" ->
    // "panel.viewer_controls", "wrapAnnotationDock" -> "panel.wrap_annotation".
    // The object name also keys the dock's saved layout, so it is stable.
    static QString panelId(const QString& dockObjectName);

private:
    struct Entry {
        QString id;
        QPointer<QAction> action;
        QString category;
    };

    Entry* find(const QString& id);
    const Entry* find(const QString& id) const;
    static CommandInfo info(const Entry& entry);

    std::vector<Entry> _entries;
};
