#include "StatusMessageHistory.hpp"

#include <QAction>
#include <QClipboard>
#include <QGuiApplication>
#include <QMenu>
#include <QRegularExpression>
#include <QStyle>

#include <algorithm>

namespace {
constexpr int kMaxMenuTextLength = 120;

// The text with each number (digits, with any decimal part) replaced by '#'.
QString withoutNumbers(const QString& text)
{
    static const QRegularExpression number(QStringLiteral("\\d+(?:[.,]\\d+)*"));
    return QString(text).replace(number, QStringLiteral("#"));
}
}

StatusMessageHistory::StatusMessageHistory(std::size_t capacity)
    : _capacity(std::max<std::size_t>(capacity, 1))
{
}

void StatusMessageHistory::add(const QString& text, const QDateTime& time)
{
    const QString trimmed = text.trimmed();
    if (trimmed.isEmpty()) {
        return;
    }
    if (!_entries.empty() && countsOn(_entries.back(), trimmed, time)) {
        _entries.back().time = time;
        _entries.back().text = trimmed;
        ++_entries.back().repeats;
        return;
    }
    _entries.push_back({time, trimmed, 1});
    while (_entries.size() > _capacity) {
        _entries.pop_front();
    }
}

bool StatusMessageHistory::repeats(const QString& a, const QString& b)
{
    return a == b || withoutNumbers(a) == withoutNumbers(b);
}

bool StatusMessageHistory::countsOn(const Entry& last, const QString& text, const QDateTime& time)
{
    if (last.text == text) {
        return true;
    }
    const qint64 gap = last.time.msecsTo(time);
    return gap >= 0 && gap <= kNumberRepeatWindowMs && repeats(last.text, text);
}

QString StatusMessageHistory::formatEntry(const Entry& entry)
{
    QString line = entry.time.toString(QStringLiteral("HH:mm:ss")) + QStringLiteral("  ") + entry.text;
    if (entry.repeats > 1) {
        line += QStringLiteral(" (x%1)").arg(entry.repeats);
    }
    return line;
}

QString StatusMessageHistory::toPlainText() const
{
    QStringList lines;
    for (auto it = _entries.rbegin(); it != _entries.rend(); ++it) {
        lines << formatEntry(*it);
    }
    return lines.join(QLatin1Char('\n'));
}

StatusMessageHistoryButton::StatusMessageHistoryButton(QWidget* parent)
    : QToolButton(parent)
    , _menu(new QMenu(this))
{
    setObjectName(QStringLiteral("statusMessageHistoryButton"));
    // Icon only; the text stays for accessibility and for styles without
    // the icon.
    setText(tr("History"));
    setIcon(style()->standardIcon(QStyle::SP_FileDialogDetailedView));
    setToolButtonStyle(icon().isNull() ? Qt::ToolButtonTextOnly : Qt::ToolButtonIconOnly);
    setAccessibleName(tr("Status message history"));
    setAutoRaise(true);
    setPopupMode(QToolButton::InstantPopup);
    _menu->setObjectName(QStringLiteral("statusMessageHistoryMenu"));
    _menu->setToolTipsVisible(true);
    setMenu(_menu);
    connect(_menu, &QMenu::aboutToShow, this, &StatusMessageHistoryButton::rebuildMenu);
    updateToolTip();
}

void StatusMessageHistoryButton::record(const QString& text)
{
    _history.add(text);
    updateToolTip();
}

void StatusMessageHistoryButton::updateToolTip()
{
    const auto count = static_cast<int>(_history.entries().size());
    setToolTip(count == 0 ? tr("Status message history (none yet)")
                          : tr("Status message history (%1)").arg(count));
}

void StatusMessageHistoryButton::rebuildMenu()
{
    _menu->clear();
    const auto& entries = _history.entries();
    if (entries.empty()) {
        _menu->addAction(tr("No messages yet"))->setEnabled(false);
        return;
    }

    int shown = 0;
    for (auto it = entries.rbegin(); it != entries.rend() && shown < kMenuEntries; ++it, ++shown) {
        QString label = StatusMessageHistory::formatEntry(*it);
        // One line; a tab would push the rest into the shortcut column.
        for (const QChar breaking : {QLatin1Char('\t'), QLatin1Char('\n'), QLatin1Char('\r')}) {
            label.replace(breaking, QLatin1Char(' '));
        }
        if (label.size() > kMaxMenuTextLength) {
            label = label.left(kMaxMenuTextLength - 3) + QStringLiteral("...");
        }
        // A literal & would otherwise be read as a mnemonic.
        label.replace(QLatin1Char('&'), QStringLiteral("&&"));
        QAction* action = _menu->addAction(label);
        const QString text = it->text;
        action->setToolTip(tr("%1\nClick to copy.").arg(text));
        connect(action, &QAction::triggered, this, [text]() {
            QGuiApplication::clipboard()->setText(text);
        });
    }

    _menu->addSeparator();
    QAction* copyAll = _menu->addAction(tr("Copy All"));
    copyAll->setObjectName(QStringLiteral("statusMessageHistoryCopyAll"));
    connect(copyAll, &QAction::triggered, this, [this]() {
        QGuiApplication::clipboard()->setText(_history.toPlainText());
    });
    QAction* clear = _menu->addAction(tr("Clear"));
    clear->setObjectName(QStringLiteral("statusMessageHistoryClear"));
    connect(clear, &QAction::triggered, this, [this]() {
        _history.clear();
        updateToolTip();
    });
}
