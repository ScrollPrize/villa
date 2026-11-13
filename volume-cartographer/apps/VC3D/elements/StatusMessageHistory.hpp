#pragma once

#include <QDateTime>
#include <QString>
#include <QToolButton>

#include <cstddef>
#include <deque>

class QMenu;

// Recent status bar messages, oldest first. A message repeated back to back
// is counted on that entry instead of being stored again. So is one that
// differs from the one before only in its numbers and comes within
// kNumberRepeatWindowMs of it (a radius readout while the mouse wheel turns);
// the entry then keeps the newest text. Messages further apart stay separate,
// so two segment ids selected a few seconds apart are two entries.
class StatusMessageHistory
{
public:
    struct Entry
    {
        QDateTime time;  // When the message last appeared
        QString text;
        int repeats{1};
    };

    static constexpr qint64 kNumberRepeatWindowMs = 2000;

    explicit StatusMessageHistory(std::size_t capacity = 200);

    void add(const QString& text, const QDateTime& time = QDateTime::currentDateTime());
    void clear() { _entries.clear(); }
    [[nodiscard]] const std::deque<Entry>& entries() const { return _entries; }

    // "14:02:31  Saved segment (x3)"
    static QString formatEntry(const Entry& entry);
    // Whether b repeats a: the same text, or the same apart from its numbers.
    static bool repeats(const QString& a, const QString& b);
    // Whether text, arriving at time, is counted on the entry last: the same
    // text at any time, or the same apart from its numbers within
    // kNumberRepeatWindowMs of the entry's last appearance.
    static bool countsOn(const Entry& last, const QString& text, const QDateTime& time);
    // One formatted line per message, newest first.
    [[nodiscard]] QString toPlainText() const;

private:
    std::size_t _capacity;
    std::deque<Entry> _entries;
};

// Status bar button listing recent messages with their times, newest first.
// Choosing a message copies it; the menu can also copy or clear the list. It
// shows only an icon, so the status bar keeps its look; the tooltip gives the
// count.
class StatusMessageHistoryButton : public QToolButton
{
    Q_OBJECT

public:
    static constexpr int kMenuEntries = 30;

    explicit StatusMessageHistoryButton(QWidget* parent = nullptr);

    void record(const QString& text);
    [[nodiscard]] const StatusMessageHistory& history() const { return _history; }

private:
    void rebuildMenu();
    void updateToolTip();

    StatusMessageHistory _history;
    QMenu* _menu{nullptr};
};
