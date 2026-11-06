#include "CommandPalette.hpp"
#include "CommandCatalog.hpp"
#include "Keybinds.hpp"
#include "VCSettings.hpp"

#include <QAction>
#include <QLineEdit>
#include <QPushButton>
#include <QSettings>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTreeWidget>
#include <QtTest/QtTest>

#include <memory>

using vc3d::commands::rankCommands;
using vc3d::commands::wordMatch;

namespace {

CommandInfo command(const QString& id, const QString& title, const QString& category)
{
    CommandInfo info;
    info.id = id;
    info.title = title;
    info.category = category;
    return info;
}

QStringList ids(const std::vector<CommandInfo>& commands)
{
    QStringList result;
    for (const auto& info : commands) {
        result.append(info.id);
    }
    return result;
}

std::vector<CommandInfo> sampleCommands()
{
    return {
        command(QStringLiteral("open_volpkg"), QStringLiteral("Open Project..."), QStringLiteral("File")),
        command(QStringLiteral("zoom_in"), QStringLiteral("Zoom in (active viewer)"), QStringLiteral("Viewer Controls")),
        command(QStringLiteral("zoom_out"), QStringLiteral("Zoom out (active viewer)"), QStringLiteral("Viewer Controls")),
        command(QStringLiteral("surface_normals"), QStringLiteral("Toggle surface normals visualization"), QStringLiteral("View")),
        command(QStringLiteral("panel.volumes"), QStringLiteral("Volume Package"), QStringLiteral("Panels")),
        command(QStringLiteral("reset_view"), QStringLiteral("Reset view (fit surface and reset Z offset)"), QStringLiteral("Viewer Controls")),
    };
}

// A shown host window with a catalog of real actions for the palette tests.
struct PaletteFixture {
    QWidget window;
    CommandCatalog catalog;
    QAction zoomIn{QStringLiteral("Zoom in (active viewer)")};
    QAction zoomOut{QStringLiteral("Zoom out (active viewer)")};
    QAction normals{QStringLiteral("Toggle surface normals visualization")};
    QAction disabled{QStringLiteral("Zoom to fit segment")};
    QAction bbox{QStringLiteral("Draw &BBox")};
    QAction paletteOpener{QStringLiteral("Command &Palette...")};

    PaletteFixture()
    {
        zoomIn.setShortcut(QKeySequence(QStringLiteral("Shift+=")));
        disabled.setEnabled(false);
        bbox.setCheckable(true);
        bbox.setChecked(true);
        catalog.add(QStringLiteral("zoom_in"), &zoomIn, QStringLiteral("Viewer Controls"));
        catalog.add(QStringLiteral("zoom_out"), &zoomOut, QStringLiteral("Viewer Controls"));
        catalog.add(QStringLiteral("surface_normals"), &normals, QStringLiteral("View"));
        catalog.add(QStringLiteral("zoom_to_fit"), &disabled, QStringLiteral("Viewer Controls"));
        catalog.add(QStringLiteral("draw_bbox"), &bbox, QStringLiteral("Actions"));
        catalog.add(QString::fromUtf8(vc3d::keybinds::shortcuts::CommandPalette.id), &paletteOpener,
                    QStringLiteral("Help"));
        window.resize(900, 600);
        window.show();
    }
};

QLineEdit* searchField(CommandPalette* palette)
{
    return palette->findChild<QLineEdit*>(QStringLiteral("commandPaletteSearch"));
}

QTreeWidget* resultList(CommandPalette* palette)
{
    return palette->findChild<QTreeWidget*>(QStringLiteral("commandPaletteList"));
}

QString currentId(CommandPalette* palette)
{
    QTreeWidgetItem* item = resultList(palette)->currentItem();
    return item ? item->data(0, Qt::UserRole).toString() : QString();
}

QStringList listedIds(CommandPalette* palette)
{
    QStringList result;
    QTreeWidget* list = resultList(palette);
    for (int row = 0; row < list->topLevelItemCount(); ++row) {
        result.append(list->topLevelItem(row)->data(0, Qt::UserRole).toString());
    }
    return result;
}

const QKeySequence& paletteKey()
{
    static const QKeySequence key = vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::CommandPalette);
    return key;
}

void pressPaletteKey(QWidget* target)
{
    const QKeyCombination combination = paletteKey()[0];
    QTest::keyClick(target, combination.key(), combination.keyboardModifiers());
}

} // namespace

class CommandPaletteTest : public QObject
{
    Q_OBJECT

private slots:
    void wordMatchPrefersWordStarts()
    {
        QCOMPARE(wordMatch(u"zoom", u"Zoom in"), 2);
        QCOMPARE(wordMatch(u"ZOOM", u"zoom in"), 2);
        QCOMPARE(wordMatch(u"in", u"Zoom in"), 2);
        QCOMPARE(wordMatch(u"oom", u"Zoom in"), 1);
        QCOMPARE(wordMatch(u"vol", u"open_volpkg"), 2);
        // A later occurrence at a word start counts.
        QCOMPARE(wordMatch(u"out", u"Reset layout, show output"), 2);
        QCOMPARE(wordMatch(u"niz", u"Zoom in"), 0);
        QCOMPARE(wordMatch(u"", u"anything"), 0);
    }

    void rankingPrefersTitleWordStarts()
    {
        const std::vector<CommandInfo> commands{
            command(QStringLiteral("reset_layout"), QStringLiteral("Reset layout"), QStringLiteral("View")),
            command(QStringLiteral("sync_cursor"), QStringLiteral("Sync cursor"), QStringLiteral("View")),
            command(QStringLiteral("show_output"), QStringLiteral("Show output"), QStringLiteral("View")),
        };
        // A title word that starts with the query beats one that contains it.
        QCOMPARE(ids(rankCommands(commands, QStringLiteral("out"), {})),
                 (QStringList{QStringLiteral("show_output"), QStringLiteral("reset_layout")}));
        // Any title match beats a match in the category alone.
        QCOMPARE(ids(rankCommands(commands, QStringLiteral("view lay"), {})), QStringList{QStringLiteral("reset_layout")});
        QCOMPARE(rankCommands(commands, QStringLiteral("view"), {}).size(), std::size_t{3});
    }

    void rankingPutsTheBestTitleMatchFirst()
    {
        const auto ranked = rankCommands(sampleCommands(), QStringLiteral("normals"), {});
        QCOMPARE(ids(ranked), QStringList{QStringLiteral("surface_normals")});

        const auto zoom = rankCommands(sampleCommands(), QStringLiteral("zoom in"), {});
        QVERIFY(!zoom.empty());
        QCOMPARE(zoom.front().id, QStringLiteral("zoom_in"));

        QVERIFY(rankCommands(sampleCommands(), QStringLiteral("qqq"), {}).empty());
    }

    void rankingMatchesCategoryWordsAndIds()
    {
        // "panel vol" spans the category and the title.
        const auto panels = rankCommands(sampleCommands(), QStringLiteral("panel vol"), {});
        QCOMPARE(ids(panels), QStringList{QStringLiteral("panel.volumes")});

        // Ids work too, for people who know them.
        const auto byId = rankCommands(sampleCommands(), QStringLiteral("open_volpkg"), {});
        QVERIFY(!byId.empty());
        QCOMPARE(byId.front().id, QStringLiteral("open_volpkg"));
    }

    void emptyQueryListsRecentCommandsFirst()
    {
        const QStringList recent{QStringLiteral("reset_view"), QStringLiteral("missing"), QStringLiteral("zoom_in")};
        const auto ranked = rankCommands(sampleCommands(), QString(), recent);
        QCOMPARE(ids(ranked),
                 (QStringList{QStringLiteral("reset_view"), QStringLiteral("zoom_in"), QStringLiteral("open_volpkg"),
                              QStringLiteral("zoom_out"), QStringLiteral("surface_normals"),
                              QStringLiteral("panel.volumes")}));
        QCOMPARE(ranked.size(), sampleCommands().size());
    }

    void recentCommandsWinCloseCalls()
    {
        const auto plain = rankCommands(sampleCommands(), QStringLiteral("zoom"), {});
        QCOMPARE(plain.front().id, QStringLiteral("zoom_in"));
        const auto recent = rankCommands(sampleCommands(), QStringLiteral("zoom"), {QStringLiteral("zoom_out")});
        QCOMPARE(recent.front().id, QStringLiteral("zoom_out"));
    }

    void paletteRunsTheSelectedCommandFromTheKeyboard()
    {
        PaletteFixture fixture;
        QSignalSpy zoomInSpy(&fixture.zoomIn, &QAction::triggered);
        CommandPalette* palette = CommandPalette::open(&fixture.catalog, QString(), &fixture.window);
        QVERIFY(palette);
        QVERIFY(palette->isVisible());
        QVERIFY(searchField(palette)->hasFocus() || QApplication::focusWidget() == searchField(palette));

        QTest::keyClicks(searchField(palette), QStringLiteral("zoom in"));
        QCOMPARE(currentId(palette), QStringLiteral("zoom_in"));
        QTreeWidgetItem* item = resultList(palette)->currentItem();
        QCOMPARE(item->text(2), QKeySequence(QStringLiteral("Shift+=")).toString(QKeySequence::NativeText));

        QTest::keyClick(searchField(palette), Qt::Key_Return);
        QCOMPARE(zoomInSpy.count(), 1);
        QVERIFY(!palette->isVisible());
        QCOMPARE(palette->recentIds(), QStringList{QStringLiteral("zoom_in")});

        // Reopening reuses the palette, clears the query and lists the recent command first.
        QCOMPARE(CommandPalette::open(&fixture.catalog, QString(), &fixture.window), palette);
        QVERIFY(searchField(palette)->text().isEmpty());
        QCOMPARE(currentId(palette), QStringLiteral("zoom_in"));
        palette->hide();
    }

    void paletteMovesSelectionAndSkipsDisabledCommands()
    {
        PaletteFixture fixture;
        QSignalSpy disabledSpy(&fixture.disabled, &QAction::triggered);
        CommandPalette* palette = CommandPalette::open(&fixture.catalog, QString(), &fixture.window);

        QTest::keyClicks(searchField(palette), QStringLiteral("zoom"));
        QTreeWidget* list = resultList(palette);
        QCOMPARE(list->topLevelItemCount(), 3);
        const QString first = currentId(palette);
        QVERIFY(first == QStringLiteral("zoom_in") || first == QStringLiteral("zoom_out"));

        QTest::keyClick(searchField(palette), Qt::Key_Down);
        const QString second = currentId(palette);
        QVERIFY(second != first);
        QVERIFY(second != QStringLiteral("zoom_to_fit"));
        QTest::keyClick(searchField(palette), Qt::Key_Down);
        QCOMPARE(currentId(palette), second);
        QTest::keyClick(searchField(palette), Qt::Key_Up);
        QCOMPARE(currentId(palette), first);

        // The disabled command is listed but cannot be selected or run.
        bool listedDisabled = false;
        for (int row = 0; row < list->topLevelItemCount(); ++row) {
            QTreeWidgetItem* item = list->topLevelItem(row);
            if (item->data(0, Qt::UserRole).toString() == QStringLiteral("zoom_to_fit")) {
                listedDisabled = true;
                QVERIFY(!(item->flags() & Qt::ItemIsEnabled));
            }
        }
        QVERIFY(listedDisabled);
        QCOMPARE(disabledSpy.count(), 0);
        palette->hide();
    }

    void openingAnOpenPaletteClosesIt()
    {
        PaletteFixture fixture;
        CommandPalette* palette = CommandPalette::open(&fixture.catalog, QString(), &fixture.window);
        QTest::keyClicks(searchField(palette), QStringLiteral("normals"));
        QCOMPARE(CommandPalette::open(&fixture.catalog, QString(), &fixture.window), palette);
        QVERIFY(!palette->isVisible());
        QCOMPARE(CommandPalette::open(&fixture.catalog, QString(), &fixture.window), palette);
        QVERIFY(palette->isVisible());
        QVERIFY(searchField(palette)->text().isEmpty());
        palette->hide();
    }

    void escapeClosesWithoutRunning()
    {
        PaletteFixture fixture;
        QSignalSpy normalsSpy(&fixture.normals, &QAction::triggered);
        CommandPalette* palette = CommandPalette::open(&fixture.catalog, QString(), &fixture.window);
        QTest::keyClicks(searchField(palette), QStringLiteral("normals"));
        QCOMPARE(currentId(palette), QStringLiteral("surface_normals"));
        QTest::keyClick(searchField(palette), Qt::Key_Escape);
        QVERIFY(!palette->isVisible());
        QCOMPARE(normalsSpy.count(), 0);
        QVERIFY(palette->recentIds().isEmpty());
    }

    void noMatchShowsAPlaceholder()
    {
        PaletteFixture fixture;
        CommandPalette* palette = CommandPalette::open(&fixture.catalog, QString(), &fixture.window);
        QTest::keyClicks(searchField(palette), QStringLiteral("qqqq"));
        QTreeWidget* list = resultList(palette);
        QCOMPARE(list->topLevelItemCount(), 1);
        QVERIFY(!(list->topLevelItem(0)->flags() & Qt::ItemIsEnabled));
        QTest::keyClick(searchField(palette), Qt::Key_Return);
        QVERIFY(palette->isVisible());
        palette->hide();
    }

    void checkableCommandsShowTheirState()
    {
        PaletteFixture fixture;
        CommandPalette* palette = CommandPalette::open(&fixture.catalog, QString(), &fixture.window);
        QTest::keyClicks(searchField(palette), QStringLiteral("draw bbox"));
        QTreeWidgetItem* item = resultList(palette)->currentItem();
        QVERIFY(item);
        QCOMPARE(item->text(0), QStringLiteral("Draw BBox"));
        QCOMPARE(item->checkState(0), Qt::Checked);
        QVERIFY(!(item->flags() & Qt::ItemIsUserCheckable));

        QTest::keyClick(searchField(palette), Qt::Key_Return);
        QVERIFY(!fixture.bbox.isChecked());
    }

    void paletteDoesNotListItself()
    {
        PaletteFixture fixture;
        CommandPalette* palette = CommandPalette::open(&fixture.catalog, QString(), &fixture.window);
        QVERIFY(!listedIds(palette).contains(QStringLiteral("command_palette")));
        QTest::keyClicks(searchField(palette), QStringLiteral("palette"));
        QVERIFY(!listedIds(palette).contains(QStringLiteral("command_palette")));
        palette->hide();
    }

    void paletteKeyClosesItFromTheSearchField()
    {
        // No shortcut is bound here, as when the search field keeps the key
        // from the shortcut map: the field itself closes the palette.
        PaletteFixture fixture;
        QSignalSpy normalsSpy(&fixture.normals, &QAction::triggered);
        CommandPalette* palette = CommandPalette::open(&fixture.catalog, QString(), &fixture.window);
        QTest::keyClicks(searchField(palette), QStringLiteral("normals"));
        pressPaletteKey(searchField(palette));
        QVERIFY(!palette->isVisible());
        QCOMPARE(normalsSpy.count(), 0);
        QCOMPARE(searchField(palette)->text(), QStringLiteral("normals"));
    }

    void paletteShortcutOpensAndClosesIt()
    {
        // The Help menu's action: the palette key, application-wide.
        PaletteFixture fixture;
        auto* button = new QPushButton(QStringLiteral("Focus"), &fixture.window);
        button->show();
        QAction opener;
        opener.setShortcut(paletteKey());
        opener.setShortcutContext(Qt::ApplicationShortcut);
        fixture.window.addAction(&opener);
        QObject::connect(&opener, &QAction::triggered, &fixture.window, [&fixture]() {
            CommandPalette::open(&fixture.catalog, QString(), &fixture.window);
        });
        fixture.window.activateWindow();
        QVERIFY(QTest::qWaitForWindowActive(&fixture.window));
        button->setFocus();

        pressPaletteKey(button);
        auto* palette = fixture.window.findChild<CommandPalette*>();
        QVERIFY(palette);
        QVERIFY(palette->isVisible());
        pressPaletteKey(searchField(palette));
        QVERIFY(!palette->isVisible());
        pressPaletteKey(button);
        QVERIFY(palette->isVisible());
        palette->hide();
    }

    void paletteOpensOverTheActiveWindow()
    {
        PaletteFixture fixture;
        fixture.window.move(0, 0);
        QWidget other;
        other.setGeometry(1400, 300, 600, 400);
        other.show();
        other.activateWindow();
        QVERIFY(QTest::qWaitForWindowActive(&other));

        CommandPalette* palette = CommandPalette::open(&fixture.catalog, QString(), &fixture.window);
        const QRect otherArea(other.mapToGlobal(QPoint(0, 0)), other.size());
        QVERIFY2(otherArea.contains(palette->geometry().center()),
                 qPrintable(QStringLiteral("palette at %1,%2").arg(palette->geometry().center().x())
                                .arg(palette->geometry().center().y())));
        palette->hide();
    }

    void recentCommandsPersistInSettings()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString settingsPath = dir.filePath(QStringLiteral("VC3D.ini"));

        PaletteFixture fixture;
        {
            CommandPalette palette(&fixture.catalog, settingsPath, &fixture.window);
            palette.popup();
            QTest::keyClicks(searchField(&palette), QStringLiteral("normals"));
            QTest::keyClick(searchField(&palette), Qt::Key_Return);
            palette.popup();
            QTest::keyClicks(searchField(&palette), QStringLiteral("zoom out"));
            QTest::keyClick(searchField(&palette), Qt::Key_Return);
        }

        const QSettings settings(settingsPath, QSettings::IniFormat);
        QCOMPARE(settings.value(vc3d::settings::command_palette::RECENT).toStringList(),
                 (QStringList{QStringLiteral("zoom_out"), QStringLiteral("surface_normals")}));

        CommandPalette reopened(&fixture.catalog, settingsPath, &fixture.window);
        QCOMPARE(reopened.recentIds(), (QStringList{QStringLiteral("zoom_out"), QStringLiteral("surface_normals")}));
    }

    void recentListIsCapped()
    {
        PaletteFixture fixture;
        std::vector<std::unique_ptr<QAction>> extra;
        for (int i = 0; i < CommandPalette::kMaxRecent + 3; ++i) {
            extra.push_back(std::make_unique<QAction>(QStringLiteral("Extra command %1").arg(i)));
            fixture.catalog.add(QStringLiteral("extra_%1").arg(i), extra.back().get(), QStringLiteral("Test"));
        }
        CommandPalette palette(&fixture.catalog, QString(), &fixture.window);
        for (int i = 0; i < CommandPalette::kMaxRecent + 3; ++i) {
            palette.popup();
            QTest::keyClicks(searchField(&palette), QStringLiteral("extra_%1").arg(i));
            QTest::keyClick(searchField(&palette), Qt::Key_Return);
        }
        QCOMPARE(palette.recentIds().size(), CommandPalette::kMaxRecent);
        QCOMPARE(palette.recentIds().front(), QStringLiteral("extra_%1").arg(CommandPalette::kMaxRecent + 2));
    }
};

QTEST_MAIN(CommandPaletteTest)
#include "test_command_palette.moc"
