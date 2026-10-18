#include "CommandPalette.hpp"

#include "Keybinds.hpp"
#include "VCSettings.hpp"

#include <QApplication>
#include <QEvent>
#include <QHeaderView>
#include <QKeyEvent>
#include <QLineEdit>
#include <QScreen>
#include <QSettings>
#include <QTreeWidget>
#include <QVBoxLayout>

#include <algorithm>
#include <cstdlib>

namespace vc3d::commands {

namespace {

// A query word scores this much more in the title than in the category or
// id, so title matches lead.
constexpr int kTitleWeight = 2;

bool startsWord(QStringView text, qsizetype index)
{
    return index == 0 || !text.at(index - 1).isLetterOrNumber();
}

// Sum of the per-word scores, or 0 when a word occurs nowhere.
int matchScore(const QStringList& words, const CommandInfo& command)
{
    int total = 0;
    for (const QString& word : words) {
        if (const int title = wordMatch(word, command.title)) {
            total += kTitleWeight + title;
        } else if (wordMatch(word, command.category) || wordMatch(word, command.id)) {
            total += 1;
        } else {
            return 0;
        }
    }
    return total;
}

} // namespace

int wordMatch(QStringView word, QStringView text)
{
    if (word.isEmpty()) {
        return 0;
    }
    int best = 0;
    for (qsizetype from = 0;;) {
        const qsizetype index = text.indexOf(word, from, Qt::CaseInsensitive);
        if (index < 0) {
            return best;
        }
        if (startsWord(text, index)) {
            return 2;
        }
        best = 1;
        from = index + 1;
    }
}

std::vector<CommandInfo> rankCommands(const std::vector<CommandInfo>& commands,
                                      const QString& query,
                                      const QStringList& recentIds)
{
    const QStringList words = query.split(QLatin1Char(' '), Qt::SkipEmptyParts);
    struct Ranked {
        int score;
        qsizetype recent;
        std::size_t order;
    };
    std::vector<Ranked> ranked;
    ranked.reserve(commands.size());
    for (std::size_t index = 0; index < commands.size(); ++index) {
        const int score = words.isEmpty() ? 0 : matchScore(words, commands[index]);
        if (!words.isEmpty() && score == 0) {
            continue;
        }
        const qsizetype recent = recentIds.indexOf(commands[index].id);
        ranked.push_back({score, recent < 0 ? recentIds.size() : recent, index});
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const Ranked& a, const Ranked& b) {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        return a.recent < b.recent;
    });

    std::vector<CommandInfo> result;
    result.reserve(ranked.size());
    for (const Ranked& entry : ranked) {
        result.push_back(commands[entry.order]);
    }
    return result;
}

} // namespace vc3d::commands

namespace {
constexpr int kIdRole = Qt::UserRole;
}

CommandPalette::CommandPalette(CommandCatalog* catalog, QString settingsPath, QWidget* parent)
    : QFrame(parent, Qt::Popup)
    , _catalog(catalog)
    , _settingsPath(std::move(settingsPath))
{
    setObjectName(QStringLiteral("commandPalette"));
    setFrameShape(QFrame::StyledPanel);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(6, 6, 6, 6);
    layout->setSpacing(4);

    _search = new QLineEdit(this);
    _search->setObjectName(QStringLiteral("commandPaletteSearch"));
    _search->setPlaceholderText(tr("Search commands"));
    _search->installEventFilter(this);
    layout->addWidget(_search);

    _list = new QTreeWidget(this);
    _list->setObjectName(QStringLiteral("commandPaletteList"));
    _list->setColumnCount(3);
    _list->setHeaderHidden(true);
    _list->setRootIsDecorated(false);
    _list->setUniformRowHeights(true);
    _list->setAllColumnsShowFocus(true);
    _list->setFocusPolicy(Qt::NoFocus);
    _list->setSelectionMode(QAbstractItemView::SingleSelection);
    _list->header()->setStretchLastSection(false);
    _list->header()->setSectionResizeMode(0, QHeaderView::Stretch);
    _list->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    _list->header()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    layout->addWidget(_list);

    connect(_search, &QLineEdit::textChanged, this, &CommandPalette::refresh);
    connect(_list, &QTreeWidget::itemClicked, this, [this](QTreeWidgetItem* item) {
        _list->setCurrentItem(item);
        invokeCurrent();
    });

    if (!_settingsPath.isEmpty()) {
        const QSettings settings(_settingsPath, QSettings::IniFormat);
        _recentIds = settings.value(vc3d::settings::command_palette::RECENT).toStringList();
    }
}

CommandPalette* CommandPalette::open(CommandCatalog* catalog, const QString& settingsPath, QWidget* window)
{
    if (!catalog || !window) {
        return nullptr;
    }
    auto* palette = window->findChild<CommandPalette*>(QString(), Qt::FindDirectChildrenOnly);
    if (!palette) {
        palette = new CommandPalette(catalog, settingsPath, window);
    } else if (palette->isVisible()) {
        palette->hide();
        return palette;
    }
    palette->popup();
    return palette;
}

void CommandPalette::popup()
{
    _search->clear();
    refresh();

    // The shortcut is application-wide, so center on whichever window it was
    // pressed in: a floating dock or a workspace window as well as the main one.
    QWidget* host = QApplication::activeWindow();
    if (!host || host == this) {
        host = parentWidget() ? parentWidget()->window() : nullptr;
    }
    QRect area;
    if (host) {
        area = QRect(host->mapToGlobal(QPoint(0, 0)), host->size());
    } else if (const QScreen* currentScreen = screen()) {
        area = currentScreen->availableGeometry();
    }
    const int width = std::clamp(area.width() / 2, std::min(420, area.width()), 720);
    const int height = std::clamp(area.height() - 120, std::min(200, area.height()), 420);
    resize(width, height);
    move(area.x() + (area.width() - width) / 2, area.y() + std::min(80, area.height() / 8));

    show();
    raise();
    activateWindow();
    _search->setFocus(Qt::PopupFocusReason);
}

bool CommandPalette::eventFilter(QObject* watched, QEvent* event)
{
    if (watched == _search && event->type() == QEvent::KeyPress) {
        const auto* keyEvent = static_cast<QKeyEvent*>(event);
        // The palette's own key closes it. Text fields that bind the key
        // themselves (Ctrl+K deletes to the end of the line on Linux) keep it
        // from reaching the shortcut, so the field handles it here.
        if (QKeySequence(keyEvent->keyCombination()) ==
            vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::CommandPalette)) {
            hide();
            return true;
        }
        const int rowHeight = std::max(1, _list->sizeHintForRow(0));
        const int page = std::max(1, _list->viewport()->height() / rowHeight - 1);
        switch (keyEvent->key()) {
        case Qt::Key_Down:
            moveSelection(1);
            return true;
        case Qt::Key_Up:
            moveSelection(-1);
            return true;
        case Qt::Key_PageDown:
            moveSelection(page);
            return true;
        case Qt::Key_PageUp:
            moveSelection(-page);
            return true;
        case Qt::Key_Return:
        case Qt::Key_Enter:
            invokeCurrent();
            return true;
        case Qt::Key_Escape:
            hide();
            return true;
        default:
            break;
        }
    }
    return QFrame::eventFilter(watched, event);
}

void CommandPalette::refresh()
{
    _list->clear();
    if (!_catalog) {
        return;
    }

    // The palette does not list the command that opens it.
    auto commands = _catalog->commands();
    const QString paletteId = QString::fromUtf8(vc3d::keybinds::shortcuts::CommandPalette.id);
    std::erase_if(commands, [&paletteId](const CommandInfo& command) { return command.id == paletteId; });

    const auto ranked = vc3d::commands::rankCommands(commands, _search->text(), _recentIds);
    QTreeWidgetItem* first = nullptr;
    for (const CommandInfo& command : ranked) {
        auto* item = new QTreeWidgetItem(_list);
        item->setText(0, command.title);
        item->setText(1, command.category);
        item->setText(2, command.shortcut.toString(QKeySequence::NativeText));
        item->setTextAlignment(2, Qt::AlignRight | Qt::AlignVCenter);
        item->setData(0, kIdRole, command.id);
        item->setToolTip(0, command.id);
        item->setForeground(1, palette().brush(QPalette::Disabled, QPalette::Text));
        if (command.checkable) {
            item->setCheckState(0, command.checked ? Qt::Checked : Qt::Unchecked);
        }
        item->setFlags(command.enabled ? (Qt::ItemIsEnabled | Qt::ItemIsSelectable) : Qt::NoItemFlags);
        if (!first && command.enabled) {
            first = item;
        }
    }

    if (ranked.empty()) {
        auto* item = new QTreeWidgetItem(_list);
        item->setText(0, tr("No matching commands"));
        item->setFlags(Qt::NoItemFlags);
    }
    if (first) {
        _list->setCurrentItem(first);
    }
}

void CommandPalette::moveSelection(int delta)
{
    const int count = _list->topLevelItemCount();
    if (count == 0 || delta == 0) {
        return;
    }
    const int step = delta > 0 ? 1 : -1;
    int row = _list->indexOfTopLevelItem(_list->currentItem());
    int target = row;
    int remaining = std::abs(delta);
    for (int candidate = row + step; candidate >= 0 && candidate < count && remaining > 0; candidate += step) {
        if (_list->topLevelItem(candidate)->flags() & Qt::ItemIsEnabled) {
            target = candidate;
            --remaining;
        }
    }
    if (target != row && target >= 0) {
        _list->setCurrentItem(_list->topLevelItem(target));
        _list->scrollToItem(_list->currentItem());
    }
}

void CommandPalette::invokeCurrent()
{
    QTreeWidgetItem* item = _list->currentItem();
    if (!item || !(item->flags() & Qt::ItemIsEnabled) || !_catalog) {
        return;
    }
    const QString id = item->data(0, kIdRole).toString();
    if (id.isEmpty()) {
        return;
    }
    // Hide first so a dialog opened by the command is not covered by the popup.
    hide();
    rememberRecent(id);
    _catalog->invoke(id);
}

void CommandPalette::rememberRecent(const QString& id)
{
    _recentIds.removeAll(id);
    _recentIds.prepend(id);
    while (_recentIds.size() > kMaxRecent) {
        _recentIds.removeLast();
    }
    if (!_settingsPath.isEmpty()) {
        QSettings settings(_settingsPath, QSettings::IniFormat);
        settings.setValue(vc3d::settings::command_palette::RECENT, _recentIds);
    }
}
