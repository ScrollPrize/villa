#pragma once

#include "Keybinds.hpp"

#include <QDialog>
#include <QString>

#include <vector>

class CommandCatalog;
class QLabel;
class QLineEdit;
class QTableWidget;

namespace vc3d::keybinds {

// The keybinds help with the command catalog merged in. A help row for a
// catalog command shows that command's current shortcut; a catalog command
// with a shortcut the help does not list is appended under its category.
std::vector<HelpRow> sheetRows(const CommandCatalog* catalog);

// True when every whitespace-separated word of filter occurs, ignoring case,
// in the row's section, keys, description or id. Keys also match in portable
// form, so "ctrl+k" finds Cmd+K on macOS.
bool rowMatches(const HelpRow& row, const QString& filter);

// The rows as a Markdown table with Section, Keys and Action columns.
QString rowsToMarkdown(const std::vector<HelpRow>& rows);

} // namespace vc3d::keybinds

// Help > Keybinds: every key binding in a table that a filter box narrows,
// with a button that copies the visible rows as Markdown. While the filter
// holds text, Enter moves to the first visible row and Escape clears it.
class KeybindsSheet : public QDialog
{
    Q_OBJECT

public:
    explicit KeybindsSheet(const CommandCatalog* catalog, QWidget* parent = nullptr);

    std::vector<vc3d::keybinds::HelpRow> visibleRows() const;

private:
    void applyFilter(const QString& text);
    void focusFirstVisibleRow();
    void copyVisibleRowsAsMarkdown();

    std::vector<vc3d::keybinds::HelpRow> _rows;
    QLineEdit* _filter{nullptr};
    QTableWidget* _table{nullptr};
    QLabel* _status{nullptr};
};
