#include "KeybindsSheet.hpp"

#include "CommandCatalog.hpp"
#include "SettingsFilter.hpp"

#include <QClipboard>
#include <QDialogButtonBox>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSet>
#include <QTableWidget>
#include <QVBoxLayout>

namespace vc3d::keybinds {

namespace {

QString markdownCell(QString text)
{
    text.replace(QLatin1Char('|'), QStringLiteral("\\|"));
    return text;
}

} // namespace

std::vector<HelpRow> sheetRows(const CommandCatalog* catalog)
{
    std::vector<HelpRow> rows = helpRows();
    if (!catalog) {
        return rows;
    }

    QSet<QString> listed;
    for (HelpRow& row : rows) {
        if (row.id.isEmpty()) {
            continue;
        }
        const auto command = catalog->command(row.id);
        if (!command) {
            continue;
        }
        listed.insert(row.id);
        if (!command->shortcut.isEmpty()) {
            row.sequence = command->shortcut;
            row.keys = command->shortcut.toString(QKeySequence::NativeText);
        }
    }
    for (const CommandInfo& command : catalog->commands()) {
        if (command.shortcut.isEmpty() || listed.contains(command.id)) {
            continue;
        }
        rows.push_back({command.category,
                        command.title,
                        command.shortcut.toString(QKeySequence::NativeText),
                        command.id,
                        command.shortcut});
    }
    return rows;
}

bool rowMatches(const HelpRow& row, const QString& filter)
{
    const QStringList words = filter.split(QLatin1Char(' '), Qt::SkipEmptyParts);
    if (words.isEmpty()) {
        return true;
    }
    const QString text = QStringList{row.section,
                                     row.keys,
                                     row.sequence.toString(QKeySequence::PortableText),
                                     row.description,
                                     row.id}
                             .join(QLatin1Char('\n'));
    for (const QString& word : words) {
        if (!text.contains(word, Qt::CaseInsensitive)) {
            return false;
        }
    }
    return true;
}

QString rowsToMarkdown(const std::vector<HelpRow>& rows)
{
    QString result = QStringLiteral("| Section | Keys | Action |\n| --- | --- | --- |\n");
    for (const HelpRow& row : rows) {
        result += QStringLiteral("| %1 | %2 | %3 |\n")
                      .arg(markdownCell(row.section), markdownCell(row.keys), markdownCell(row.description));
    }
    return result;
}

} // namespace vc3d::keybinds

KeybindsSheet::KeybindsSheet(const CommandCatalog* catalog, QWidget* parent)
    : QDialog(parent)
    , _rows(vc3d::keybinds::sheetRows(catalog))
{
    setObjectName(QStringLiteral("keybindsSheet"));
    setWindowTitle(tr("Keybindings for Volume Cartographer"));

    auto* layout = new QVBoxLayout(this);

    _filter = new QLineEdit(this);
    _filter->setObjectName(QStringLiteral("keybindsFilter"));
    _filter->setPlaceholderText(tr("Filter by key, action or section"));
    _filter->setClearButtonEnabled(true);
    layout->addWidget(_filter);

    _table = new QTableWidget(static_cast<int>(_rows.size()), 3, this);
    _table->setObjectName(QStringLiteral("keybindsTable"));
    _table->setHorizontalHeaderLabels({tr("Section"), tr("Keys"), tr("Action")});
    _table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _table->setSelectionBehavior(QAbstractItemView::SelectRows);
    _table->setWordWrap(false);
    _table->verticalHeader()->setVisible(false);
    _table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    _table->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    _table->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    for (int row = 0; row < static_cast<int>(_rows.size()); ++row) {
        const auto& help = _rows[static_cast<std::size_t>(row)];
        _table->setItem(row, 0, new QTableWidgetItem(help.section));
        _table->setItem(row, 1, new QTableWidgetItem(help.keys));
        auto* action = new QTableWidgetItem(help.description);
        if (!help.id.isEmpty()) {
            action->setToolTip(help.id);
        }
        _table->setItem(row, 2, action);
    }
    layout->addWidget(_table);

    auto* footer = new QHBoxLayout;
    _status = new QLabel(this);
    _status->setObjectName(QStringLiteral("keybindsStatus"));
    footer->addWidget(_status, 1);
    auto* copy = new QPushButton(tr("Copy as Markdown"), this);
    copy->setObjectName(QStringLiteral("keybindsCopyMarkdown"));
    copy->setAutoDefault(false);
    footer->addWidget(copy);
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Close, this);
    footer->addWidget(buttons);
    layout->addLayout(footer);

    connect(_filter, &QLineEdit::textChanged, this, &KeybindsSheet::applyFilter);
    vc3d::settings_filter::installFilterKeys(_filter, [this]() { focusFirstVisibleRow(); });
    connect(copy, &QPushButton::clicked, this, &KeybindsSheet::copyVisibleRowsAsMarkdown);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

    applyFilter(QString());
    resize(760, 560);
    setMinimumHeight(360);
    _filter->setFocus();
}

std::vector<vc3d::keybinds::HelpRow> KeybindsSheet::visibleRows() const
{
    std::vector<vc3d::keybinds::HelpRow> visible;
    for (int row = 0; row < static_cast<int>(_rows.size()); ++row) {
        if (!_table->isRowHidden(row)) {
            visible.push_back(_rows[static_cast<std::size_t>(row)]);
        }
    }
    return visible;
}

void KeybindsSheet::applyFilter(const QString& text)
{
    int shown = 0;
    for (int row = 0; row < static_cast<int>(_rows.size()); ++row) {
        const bool match = vc3d::keybinds::rowMatches(_rows[static_cast<std::size_t>(row)], text);
        _table->setRowHidden(row, !match);
        shown += match ? 1 : 0;
    }
    _status->setText(tr("%1 of %2 bindings").arg(shown).arg(_rows.size()));
}

void KeybindsSheet::focusFirstVisibleRow()
{
    for (int row = 0; row < _table->rowCount(); ++row) {
        if (!_table->isRowHidden(row)) {
            _table->setCurrentCell(row, 0);
            _table->setFocus(Qt::TabFocusReason);
            return;
        }
    }
}

void KeybindsSheet::copyVisibleRowsAsMarkdown()
{
    const auto rows = visibleRows();
    QGuiApplication::clipboard()->setText(vc3d::keybinds::rowsToMarkdown(rows));
    _status->setText(tr("Copied %1 bindings as Markdown").arg(rows.size()));
}
