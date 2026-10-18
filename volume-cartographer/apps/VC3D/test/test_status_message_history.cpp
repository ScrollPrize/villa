#include "elements/StatusMessageHistory.hpp"

#include <QAction>
#include <QClipboard>
#include <QGuiApplication>
#include <QMenu>
#include <QtTest/QtTest>

class StatusMessageHistoryTest : public QObject
{
    Q_OBJECT

private slots:
    void keepsRecentMessagesWithTimes()
    {
        StatusMessageHistory history(3);
        const QDateTime start(QDate(2026, 10, 18), QTime(14, 2, 31));
        history.add(QStringLiteral("Applied segmentation edits."), start);
        history.add(QString(), start);
        history.add(QStringLiteral("   "), start);
        history.add(QStringLiteral("Undid last segmentation change."), start.addSecs(5));
        QCOMPARE(history.entries().size(), std::size_t{2});
        QCOMPARE(history.entries().back().time, start.addSecs(5));
        QCOMPARE(StatusMessageHistory::formatEntry(history.entries().front()),
                 QStringLiteral("14:02:31  Applied segmentation edits."));
    }

    void collapsesRepeatsAndDropsTheOldest()
    {
        StatusMessageHistory history(2);
        const QDateTime start(QDate(2026, 10, 18), QTime(9, 0, 0));
        history.add(QStringLiteral("a"), start);
        history.add(QStringLiteral("b"), start.addSecs(1));
        history.add(QStringLiteral("b"), start.addSecs(2));
        history.add(QStringLiteral("b"), start.addSecs(3));
        QCOMPARE(history.entries().size(), std::size_t{2});
        QCOMPARE(history.entries().back().repeats, 3);
        QCOMPARE(history.toPlainText(), QStringLiteral("09:00:03  b (x3)\n09:00:00  a"));

        history.add(QStringLiteral("c"), start.addSecs(4));
        QCOMPARE(history.entries().size(), std::size_t{2});
        QCOMPARE(history.entries().front().text, QStringLiteral("b"));
    }

    void menuListsNewestFirstAndCopies()
    {
        StatusMessageHistoryButton button;
        auto* menu = button.findChild<QMenu*>(QStringLiteral("statusMessageHistoryMenu"));
        QVERIFY(menu);
        QCOMPARE(button.menu(), menu);

        emit menu->aboutToShow();
        QCOMPARE(menu->actions().size(), 1);
        QVERIFY(!menu->actions().front()->isEnabled());

        button.record(QStringLiteral("Saved 3 points & 1 collection"));
        button.record(QStringLiteral("Undid last segmentation change."));
        QVERIFY(button.toolTip().contains(QStringLiteral("2")));
        emit menu->aboutToShow();
        const QList<QAction*> actions = menu->actions();
        QVERIFY(actions.size() >= 4);
        QVERIFY(actions.at(0)->text().endsWith(QStringLiteral("Undid last segmentation change.")));
        QVERIFY(actions.at(1)->text().contains(QStringLiteral("&&")));

        actions.at(1)->trigger();
        QCOMPARE(QGuiApplication::clipboard()->text(), QStringLiteral("Saved 3 points & 1 collection"));

        auto* copyAll = menu->findChild<QAction*>(QStringLiteral("statusMessageHistoryCopyAll"));
        QVERIFY(copyAll);
        copyAll->trigger();
        QCOMPARE(QGuiApplication::clipboard()->text(), button.history().toPlainText());

        auto* clear = menu->findChild<QAction*>(QStringLiteral("statusMessageHistoryClear"));
        QVERIFY(clear);
        clear->trigger();
        QVERIFY(button.history().entries().empty());
    }

    void menuShowsOnlyTheNewestEntries()
    {
        StatusMessageHistoryButton button;
        for (int i = 0; i < StatusMessageHistoryButton::kMenuEntries + 10; ++i) {
            button.record(QStringLiteral("message %1").arg(i));
        }
        auto* menu = button.menu();
        emit menu->aboutToShow();
        // Entries, a separator, Copy All and Clear.
        QCOMPARE(menu->actions().size(), StatusMessageHistoryButton::kMenuEntries + 3);
        QVERIFY(menu->actions().front()->text().endsWith(
            QStringLiteral("message %1").arg(StatusMessageHistoryButton::kMenuEntries + 9)));
    }

    void menuLabelsStayOnOneLine()
    {
        StatusMessageHistoryButton button;
        const QString message = QStringLiteral("Lasagna failed:\tconfig.json\nmissing key");
        button.record(message);
        auto* menu = button.menu();
        emit menu->aboutToShow();
        const QString label = menu->actions().front()->text();
        QVERIFY(!label.contains(QLatin1Char('\t')));
        QVERIFY(!label.contains(QLatin1Char('\n')));
        QVERIFY(label.endsWith(QStringLiteral("Lasagna failed: config.json missing key")));

        // Copying gives the message as it was.
        menu->actions().front()->trigger();
        QCOMPARE(QGuiApplication::clipboard()->text(), message);
    }
};

QTEST_MAIN(StatusMessageHistoryTest)
#include "test_status_message_history.moc"
