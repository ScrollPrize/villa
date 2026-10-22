#pragma once

#include "CommandCatalog.hpp"

#include <QFrame>
#include <QString>
#include <QStringList>
#include <QStringView>

#include <vector>

class QLineEdit;
class QTreeWidget;

namespace vc3d::commands {

// How word occurs in text, ignoring case: 2 when a word of text starts with
// it, 1 when it occurs only inside a word, 0 when it does not occur.
int wordMatch(QStringView word, QStringView text);

// Orders commands for the palette. Every whitespace-separated word of query
// must occur in the title, the category or the id. Commands whose title
// holds the words, best at word starts, come first; recently used commands
// win ties, then catalog order. With an empty query, recent commands come
// first in recency order, then the rest in catalog order.
std::vector<CommandInfo> rankCommands(const std::vector<CommandInfo>& commands,
                                      const QString& query,
                                      const QStringList& recentIds);

} // namespace vc3d::commands

// Keyboard-first popup that searches the command catalog and runs the chosen
// command: type to filter, Up/Down/PageUp/PageDown to move, Enter to run,
// Escape or the palette's own key to close. Commands run from the palette are
// remembered as recent.
class CommandPalette : public QFrame
{
    Q_OBJECT

public:
    static constexpr int kMaxRecent = 8;

    // settingsPath names the INI file that keeps the recent commands; an empty
    // path keeps them for this session only.
    CommandPalette(CommandCatalog* catalog, QString settingsPath, QWidget* parent);

    // Shows the palette that belongs to window, creating it on first use, or
    // hides it when it is already showing, so its shortcut toggles it.
    static CommandPalette* open(CommandCatalog* catalog, const QString& settingsPath, QWidget* window);

    // Shows the palette with an empty query, centered near the top of the
    // active window.
    void popup();
    QStringList recentIds() const { return _recentIds; }

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    void refresh();
    void moveSelection(int delta);
    void invokeCurrent();
    void rememberRecent(const QString& id);

    CommandCatalog* _catalog{nullptr};
    QString _settingsPath;
    QStringList _recentIds;
    QLineEdit* _search{nullptr};
    QTreeWidget* _list{nullptr};
};
