// Covers the keybinds help rows and the Help > Keybinds sheet built from them,
// including a check that every ShortcutDef and KeyPressDef in Keybinds.cpp is
// listed in the help.

#include "CommandCatalog.hpp"
#include "Keybinds.hpp"
#include "KeybindsSheet.hpp"

#include <QAction>
#include <QApplication>
#include <QClipboard>
#include <QFile>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QRegularExpression>
#include <QSet>
#include <QSignalSpy>
#include <QTableWidget>
#include <QtTest/QtTest>

#include <algorithm>
#include <vector>

#ifndef VC3D_KEYBINDS_SOURCE
#error "VC3D_KEYBINDS_SOURCE must name apps/VC3D/Keybinds.cpp"
#endif

namespace {

using vc3d::keybinds::HelpRow;

// Ids of the ShortcutDef and KeyPressDef definitions in Keybinds.cpp.
QStringList definedIds()
{
    QFile file(QStringLiteral(VC3D_KEYBINDS_SOURCE));
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return {};
    }
    static const QRegularExpression definition(
        QStringLiteral("const\\s+(?:ShortcutDef|KeyPressDef)\\s+\\w+\\s*\\{\\s*\"([^\"]+)\""));
    QStringList ids;
    auto matches = definition.globalMatch(QString::fromUtf8(file.readAll()));
    while (matches.hasNext()) {
        ids.append(matches.next().captured(1));
    }
    return ids;
}

const HelpRow* findRow(const std::vector<HelpRow>& rows, const QString& id)
{
    const auto it = std::find_if(rows.begin(), rows.end(), [&id](const HelpRow& row) { return row.id == id; });
    return it == rows.end() ? nullptr : &*it;
}

} // namespace

class KeybindsSheetTest : public QObject
{
    Q_OBJECT

private slots:
    void everyKeybindDefinitionIsListed()
    {
        const QStringList defined = definedIds();
        // Guards the scan itself: it must see both kinds of definition.
        QVERIFY(defined.contains(QStringLiteral("command_palette")));
        QVERIFY(defined.contains(QStringLiteral("range_slider_page_up")));
        QVERIFY(defined.size() > 40);

        QSet<QString> listed;
        for (const HelpRow& row : vc3d::keybinds::helpRows()) {
            listed.insert(row.id);
        }
        QStringList missing;
        for (const QString& id : defined) {
            if (!listed.contains(id)) {
                missing.append(id);
            }
        }
        QVERIFY2(missing.isEmpty(),
                 qPrintable(QStringLiteral("not listed in helpRows(): ") + missing.join(QStringLiteral(", "))));
    }

    void helpRowsAreWellFormed()
    {
        for (const HelpRow& row : vc3d::keybinds::helpRows()) {
            QVERIFY(!row.section.isEmpty());
            QVERIFY(!row.description.isEmpty());
            QVERIFY(!row.keys.isEmpty());
            if (row.id.isEmpty()) {
                QVERIFY(row.sequence.isEmpty());
            } else {
                QVERIFY2(!row.sequence.isEmpty(), qPrintable(row.id));
                QCOMPARE(row.keys, row.sequence.toString(QKeySequence::NativeText));
            }
        }
    }

    void menuSectionsAreNamedAfterTheirMenus()
    {
        const auto rows = vc3d::keybinds::helpRows();
        QCOMPARE(findRow(rows, QStringLiteral("open_volpkg"))->section, QStringLiteral("File"));
        QCOMPARE(findRow(rows, QStringLiteral("surface_normals"))->section, QStringLiteral("View"));
        QCOMPARE(findRow(rows, QStringLiteral("command_palette"))->section, QStringLiteral("Help"));
    }

    void sheetRowsFollowTheCatalog()
    {
        CommandCatalog catalog;
        QAction zoomIn(QStringLiteral("Zoom in (active viewer)"));
        zoomIn.setShortcut(QKeySequence(QStringLiteral("Ctrl+Alt+Z")));
        QAction extra(QStringLiteral("Extra command"));
        extra.setShortcut(QKeySequence(Qt::Key_F9));
        QAction quiet(QStringLiteral("Quiet command"));
        catalog.add(QStringLiteral("zoom_in"), &zoomIn, QStringLiteral("Viewer Controls"));
        catalog.add(QStringLiteral("extra_command"), &extra, QStringLiteral("Tools"));
        catalog.add(QStringLiteral("quiet_command"), &quiet, QStringLiteral("Tools"));

        const auto plain = vc3d::keybinds::helpRows();
        const auto rows = vc3d::keybinds::sheetRows(&catalog);
        QCOMPARE(rows.size(), plain.size() + 1);

        const HelpRow* zoomRow = findRow(rows, QStringLiteral("zoom_in"));
        QVERIFY(zoomRow);
        QCOMPARE(zoomRow->keys, QKeySequence(QStringLiteral("Ctrl+Alt+Z")).toString(QKeySequence::NativeText));
        QCOMPARE(zoomRow->sequence, QKeySequence(QStringLiteral("Ctrl+Alt+Z")));

        const HelpRow& appended = rows.back();
        QCOMPARE(appended.id, QStringLiteral("extra_command"));
        QCOMPARE(appended.section, QStringLiteral("Tools"));
        QCOMPARE(appended.description, QStringLiteral("Extra command"));
        QCOMPARE(appended.sequence, QKeySequence(Qt::Key_F9));

        QCOMPARE(vc3d::keybinds::sheetRows(nullptr).size(), plain.size());
    }

    void filterMatchesEveryWordAcrossColumns()
    {
        const QKeySequence key(QStringLiteral("Ctrl+K"));
        const HelpRow row{QStringLiteral("Help"),
                          QStringLiteral("Search and run commands (command palette)"),
                          key.toString(QKeySequence::NativeText),
                          QStringLiteral("command_palette"),
                          key};
        using vc3d::keybinds::rowMatches;
        QVERIFY(rowMatches(row, QString()));
        QVERIFY(rowMatches(row, QStringLiteral("palette")));
        // Portable text matches too, so "ctrl+k" finds Cmd+K on macOS.
        QVERIFY(rowMatches(row, QStringLiteral("ctrl+k")));
        QVERIFY(rowMatches(row, QStringLiteral("HELP search")));
        QVERIFY(rowMatches(row, QStringLiteral("command_palette")));
        QVERIFY(!rowMatches(row, QStringLiteral("palette zoom")));

        const HelpRow literal{QStringLiteral("Line Annotation Workspace"),
                              QStringLiteral("Toggle the volume overlay"),
                              QStringLiteral("Ctrl+Space"),
                              {},
                              {}};
        QVERIFY(rowMatches(literal, QStringLiteral("ctrl+space")));
    }

    void markdownEscapesPipes()
    {
        const std::vector<HelpRow> rows{
            {QStringLiteral("Push/Pull"), QStringLiteral("Push | pull"), QStringLiteral("A"), {}, {}},
        };
        QCOMPARE(vc3d::keybinds::rowsToMarkdown(rows),
                 QStringLiteral("| Section | Keys | Action |\n| --- | --- | --- |\n| Push/Pull | A | Push \\| pull |\n"));
    }

    void sheetFiltersRowsAndCopiesMarkdown()
    {
        KeybindsSheet sheet(nullptr);
        sheet.show();
        auto* filter = sheet.findChild<QLineEdit*>(QStringLiteral("keybindsFilter"));
        auto* copy = sheet.findChild<QPushButton*>(QStringLiteral("keybindsCopyMarkdown"));
        auto* status = sheet.findChild<QLabel*>(QStringLiteral("keybindsStatus"));
        QVERIFY(filter && copy && status);

        const std::size_t all = sheet.visibleRows().size();
        QCOMPARE(all, vc3d::keybinds::helpRows().size());

        QTest::keyClicks(filter, QStringLiteral("lasagna"));
        const auto visible = sheet.visibleRows();
        QCOMPARE(visible.size(), std::size_t{3});
        for (const HelpRow& row : visible) {
            QCOMPARE(row.section, QStringLiteral("Lasagna Workspace"));
        }
        QCOMPARE(status->text(), QStringLiteral("3 of %1 bindings").arg(all));

        QTest::mouseClick(copy, Qt::LeftButton);
        const QString copied = QApplication::clipboard()->text();
        QVERIFY(copied.startsWith(QStringLiteral("| Section | Keys | Action |")));
        QVERIFY(copied.contains(QKeySequence(QStringLiteral("Ctrl+Shift+L")).toString(QKeySequence::NativeText)));
        QCOMPARE(copied.count(QLatin1Char('\n')), 5);

        filter->clear();
        QCOMPARE(sheet.visibleRows().size(), all);
    }

    void filterKeysKeepTheSheetOpen()
    {
        KeybindsSheet sheet(nullptr);
        sheet.show();
        QSignalSpy rejected(&sheet, &QDialog::rejected);
        auto* filter = sheet.findChild<QLineEdit*>(QStringLiteral("keybindsFilter"));
        auto* table = sheet.findChild<QTableWidget*>(QStringLiteral("keybindsTable"));
        QVERIFY(filter && table);

        // Enter moves to the first matching row instead of closing.
        QTest::keyClicks(filter, QStringLiteral("zoom in"));
        QTest::keyClick(filter, Qt::Key_Return);
        QVERIFY(sheet.isVisible());
        QCOMPARE(sheet.focusWidget(), table);
        QVERIFY(!table->isRowHidden(table->currentRow()));
        QCOMPARE(sheet.visibleRows().front().description, table->item(table->currentRow(), 2)->text());

        // Escape clears the filter first, then closes the sheet.
        filter->setFocus();
        QTest::keyClick(filter, Qt::Key_Escape);
        QVERIFY(filter->text().isEmpty());
        QVERIFY(sheet.isVisible());
        QCOMPARE(rejected.count(), 0);
        QTest::keyClick(filter, Qt::Key_Escape);
        QCOMPARE(rejected.count(), 1);
    }
};

QTEST_MAIN(KeybindsSheetTest)
#include "test_keybinds_sheet.moc"
