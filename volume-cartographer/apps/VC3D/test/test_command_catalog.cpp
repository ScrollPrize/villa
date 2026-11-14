#include "CommandCatalog.hpp"
#include "Keybinds.hpp"
#include "StatusDockPanelHost.hpp"

#include <QAction>
#include <QDockWidget>
#include <QFrame>
#include <QLabel>
#include <QMainWindow>
#include <QPointer>
#include <QPushButton>
#include <QSignalSpy>
#include <QStatusBar>
#include <QTabWidget>
#include <QTemporaryDir>
#include <QtTest/QtTest>

#include <memory>

namespace {

QStringList ids(const std::vector<CommandInfo>& commands)
{
    QStringList result;
    for (const auto& command : commands) {
        result.append(command.id);
    }
    return result;
}

} // namespace

class CommandCatalogTest : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase()
    {
        // StatusDockPanelHost keeps panel state in the VC3D settings file.
        QVERIFY(_config.isValid());
        qputenv("VC3D_CONFIG_DIR", _config.path().toUtf8());
    }

    void addSetsObjectNameAndRejectsDuplicateIds()
    {
        CommandCatalog catalog;
        QAction first(QStringLiteral("&Open Project..."));
        QAction second(QStringLiteral("Other"));

        QVERIFY(catalog.add(QStringLiteral("open_volpkg"), &first, QStringLiteral("File")));
        QCOMPARE(first.objectName(), QStringLiteral("open_volpkg"));
        QVERIFY(catalog.add(QStringLiteral("open_volpkg"), &first, QStringLiteral("File")));
        QTest::ignoreMessage(QtWarningMsg, "CommandCatalog: command id open_volpkg is already registered");
        QVERIFY(!catalog.add(QStringLiteral("open_volpkg"), &second, QStringLiteral("File")));
        QVERIFY(!catalog.add(QString(), &second, QStringLiteral("File")));
        QVERIFY(!catalog.add(QStringLiteral("missing"), nullptr, QStringLiteral("File")));

        QCOMPARE(ids(catalog.commands()), QStringList{QStringLiteral("open_volpkg")});
        QCOMPARE(catalog.command(QStringLiteral("open_volpkg"))->title, QStringLiteral("Open Project..."));
        QVERIFY(!catalog.command(QStringLiteral("missing")).has_value());
    }

    void commandInfoReflectsTheLiveAction()
    {
        CommandCatalog catalog;
        QAction action(QStringLiteral("Draw &BBox"));
        action.setCheckable(true);
        action.setShortcut(QKeySequence(QStringLiteral("Ctrl+B")));
        catalog.add(QStringLiteral("draw_bbox"), &action, QStringLiteral("Actions"));

        auto info = catalog.command(QStringLiteral("draw_bbox"));
        QVERIFY(info.has_value());
        QCOMPARE(info->title, QStringLiteral("Draw BBox"));
        QCOMPARE(info->category, QStringLiteral("Actions"));
        QCOMPARE(info->shortcut, QKeySequence(QStringLiteral("Ctrl+B")));
        QVERIFY(info->checkable);
        QVERIFY(!info->checked);
        QVERIFY(info->enabled);

        action.setChecked(true);
        action.setEnabled(false);
        action.setText(QStringLiteral("Renamed"));
        info = catalog.command(QStringLiteral("draw_bbox"));
        QVERIFY(info->checked);
        QVERIFY(!info->enabled);
        QCOMPARE(info->title, QStringLiteral("Renamed"));
    }

    void deletedActionsLeaveTheCatalog()
    {
        CommandCatalog catalog;
        auto* action = new QAction(QStringLiteral("Transient"));
        catalog.add(QStringLiteral("transient"), action, QStringLiteral("View"));
        QCOMPARE(catalog.commands().size(), std::size_t{1});
        delete action;
        QVERIFY(catalog.commands().empty());
        QVERIFY(!catalog.command(QStringLiteral("transient")).has_value());
        QVERIFY(!catalog.invoke(QStringLiteral("transient")));
    }

    void recreatedActionTakesTheIdOfADestroyedOne()
    {
        CommandCatalog catalog;
        QAction before(QStringLiteral("Before"));
        auto* panel = new QAction(QStringLiteral("Fiber"));
        QAction after(QStringLiteral("After"));
        catalog.add(QStringLiteral("before"), &before, QStringLiteral("View"));
        catalog.add(QStringLiteral("panel.fiber"), panel, QStringLiteral("Panels"));
        catalog.add(QStringLiteral("after"), &after, QStringLiteral("View"));
        delete panel;

        QAction recreated(QStringLiteral("Fiber again"));
        QSignalSpy spy(&recreated, &QAction::triggered);
        QVERIFY(catalog.add(QStringLiteral("panel.fiber"), &recreated, QStringLiteral("Panels")));
        QCOMPARE(recreated.objectName(), QStringLiteral("panel.fiber"));
        QCOMPARE(catalog.command(QStringLiteral("panel.fiber"))->title, QStringLiteral("Fiber again"));
        // The id keeps its place in the list.
        QCOMPARE(ids(catalog.commands()),
                 (QStringList{QStringLiteral("before"), QStringLiteral("panel.fiber"), QStringLiteral("after")}));
        QVERIFY(catalog.invoke(QStringLiteral("panel.fiber")));
        QCOMPARE(spy.count(), 1);
    }

    void commandsKeepRegistrationOrder()
    {
        CommandCatalog catalog;
        QAction a(QStringLiteral("A"));
        QAction b(QStringLiteral("B"));
        QAction c(QStringLiteral("C"));
        catalog.add(QStringLiteral("zoom_out"), &a, QStringLiteral("Viewer"));
        catalog.add(QStringLiteral("panel.volumes"), &b, QStringLiteral("Panels"));
        catalog.add(QStringLiteral("zoom_in"), &c, QStringLiteral("Viewer"));

        QCOMPARE(ids(catalog.commands()),
                 (QStringList{QStringLiteral("zoom_out"), QStringLiteral("panel.volumes"), QStringLiteral("zoom_in")}));
    }

    void invokeTriggersEnabledCommands()
    {
        CommandCatalog catalog;
        QAction plain(QStringLiteral("Plain"));
        QAction toggle(QStringLiteral("Toggle"));
        toggle.setCheckable(true);
        catalog.add(QStringLiteral("plain"), &plain, QStringLiteral("View"));
        catalog.add(QStringLiteral("toggle"), &toggle, QStringLiteral("View"));
        QSignalSpy plainSpy(&plain, &QAction::triggered);

        QVERIFY(catalog.invoke(QStringLiteral("plain")));
        QCOMPARE(plainSpy.count(), 1);
        QVERIFY(!catalog.invoke(QStringLiteral("missing")));

        // A checkable command toggles, as a menu click does.
        QVERIFY(catalog.invoke(QStringLiteral("toggle")));
        QVERIFY(toggle.isChecked());
        QVERIFY(catalog.command(QStringLiteral("toggle"))->checked);
        QVERIFY(catalog.invoke(QStringLiteral("toggle")));
        QVERIFY(!toggle.isChecked());

        plain.setEnabled(false);
        QVERIFY(!catalog.invoke(QStringLiteral("plain")));
        QCOMPARE(plainSpy.count(), 1);
    }

    void addShortcutCreatesAnApplicationWideAction()
    {
        QWidget window;
        CommandCatalog catalog;
        const auto& def = vc3d::keybinds::shortcuts::OpenLasagnaWorkspace;
        QAction* action = catalog.addShortcut(def, &window);

        QVERIFY(action);
        QCOMPARE(action->parent(), &window);
        QVERIFY(window.actions().contains(action));
        QCOMPARE(action->shortcut(), QKeySequence(QStringLiteral("Ctrl+L")));
        QCOMPARE(action->shortcutContext(), Qt::ApplicationShortcut);

        const auto info = catalog.command(QStringLiteral("open_lasagna_workspace"));
        QVERIFY(info.has_value());
        QCOMPARE(info->title, QString::fromUtf8(def.description));
        QCOMPARE(info->category, QString::fromUtf8(def.section));
    }

    void shortcutActionFiresFromTheKeyboard()
    {
        QWidget window;
        window.resize(200, 120);
        CommandCatalog catalog;
        QAction* action = catalog.addShortcut(vc3d::keybinds::shortcuts::RepeatLasagnaAction, &window);
        QSignalSpy spy(action, &QAction::triggered);

        window.show();
        window.activateWindow();
        QVERIFY(QTest::qWaitForWindowActive(&window));
        QTest::keyClick(&window, Qt::Key_L, Qt::ControlModifier | Qt::ShiftModifier);
        QCOMPARE(spy.count(), 1);
    }

    void singleKeyAndPunctuationShortcutsFireFromAFocusedChild()
    {
        QMainWindow window;
        window.resize(300, 200);
        auto* child = new QWidget(&window);
        child->setFocusPolicy(Qt::StrongFocus);
        window.setCentralWidget(child);
        CommandCatalog catalog;
        QAction* composite = catalog.addShortcut(vc3d::keybinds::shortcuts::CompositeView, &window);
        QAction* offsetUp = catalog.addShortcut(vc3d::keybinds::shortcuts::WorldOffsetZPos, &window);
        QAction* offsetDown = catalog.addShortcut(vc3d::keybinds::shortcuts::WorldOffsetZNeg, &window);
        QSignalSpy compositeSpy(composite, &QAction::triggered);
        QSignalSpy upSpy(offsetUp, &QAction::triggered);
        QSignalSpy downSpy(offsetDown, &QAction::triggered);

        window.show();
        window.activateWindow();
        QVERIFY(QTest::qWaitForWindowActive(&window));
        child->setFocus();
        QTRY_VERIFY(child->hasFocus());

        QTest::keyClick(child, Qt::Key_C);
        QTest::keyClick(child, Qt::Key_Period, Qt::ControlModifier);
        QTest::keyClick(child, Qt::Key_Comma, Qt::ControlModifier);
        QCOMPARE(compositeSpy.count(), 1);
        QCOMPARE(upSpy.count(), 1);
        QCOMPARE(downSpy.count(), 1);
    }

    void menuShortcutsShareTheirMenuCategory()
    {
        // The palette groups commands by category, so a keybind that belongs to a menu uses
        // the menu's title, the same name MenuActionController gives that menu's actions.
        QCOMPARE(QString::fromUtf8(vc3d::keybinds::shortcuts::OpenVolpkg.section), QStringLiteral("File"));
        QCOMPARE(QString::fromUtf8(vc3d::keybinds::shortcuts::SurfaceNormals.section), QStringLiteral("View"));
        QCOMPARE(QString::fromUtf8(vc3d::keybinds::shortcuts::CommandPalette.section), QStringLiteral("Help"));
    }

    void panelIdsComeFromDockObjectNames()
    {
        QCOMPARE(CommandCatalog::panelId(QStringLiteral("dockWidgetVolumes")), QStringLiteral("panel.volumes"));
        QCOMPARE(CommandCatalog::panelId(QStringLiteral("dockWidgetViewerControls")),
                 QStringLiteral("panel.viewer_controls"));
        QCOMPARE(CommandCatalog::panelId(QStringLiteral("dockWidgetAtlasOverview")),
                 QStringLiteral("panel.atlas_overview"));
        QCOMPARE(CommandCatalog::panelId(QStringLiteral("wrapAnnotationDock")), QStringLiteral("panel.wrap_annotation"));
        QCOMPARE(CommandCatalog::panelId(QStringLiteral("fiberDock")), QStringLiteral("panel.fiber"));
        QCOMPARE(CommandCatalog::panelId(QStringLiteral("las-ctl-diff")), QStringLiteral("panel.las_ctl_diff"));
        QCOMPARE(CommandCatalog::panelId(QStringLiteral("Dock")), QStringLiteral("panel.dock"));
    }

    void panelsOpenTheirStatusBarPanel()
    {
        QMainWindow window;
        window.resize(900, 600);
        auto* host = new StatusDockPanelHost(new QWidget, &window);
        window.setCentralWidget(host);
        auto* dock = new QDockWidget(QStringLiteral("Volume Package"), &window);
        dock->setObjectName(QStringLiteral("dockWidgetVolumes"));
        auto* content = new QLabel(QStringLiteral("volumes"));
        dock->setWidget(content);
        window.addDockWidget(Qt::LeftDockWidgetArea, dock);
        auto* unnamed = new QDockWidget(QStringLiteral("Scratch"), &window);
        unnamed->setWidget(new QLabel(QStringLiteral("scratch")));
        window.addDockWidget(Qt::RightDockWidgetArea, unnamed);
        host->addDock(dock);
        host->addDock(unnamed);
        window.show();
        QVERIFY(QTest::qWaitForWindowExposed(&window));

        QVERIFY(!host->openAction(nullptr));
        CommandCatalog catalog;
        QVERIFY(catalog.addPanel(dock, host->openAction(dock)));
        QVERIFY(!catalog.addPanel(unnamed, host->openAction(unnamed)));
        QVERIFY(!catalog.addPanel(nullptr, nullptr));
        QCOMPARE(ids(catalog.commands()), QStringList{QStringLiteral("panel.volumes")});

        const auto info = catalog.command(QStringLiteral("panel.volumes"));
        QCOMPARE(info->title, QStringLiteral("Volume Package"));
        QCOMPARE(info->category, QStringLiteral("Panels"));
        QVERIFY(!info->checkable);

        auto* frame = window.findChild<QFrame*>(QStringLiteral("statusDockPanelFrame"));
        QVERIFY(frame);
        QVERIFY(!frame->isVisible());
        QVERIFY(catalog.invoke(QStringLiteral("panel.volumes")));
        QVERIFY(frame->isVisible());
        QVERIFY(content->isVisible());
        // Opening an open panel keeps it open.
        catalog.invoke(QStringLiteral("panel.volumes"));
        QVERIFY(frame->isVisible());

        // The View menu's toggle closes the panel and removes its button;
        // opening the panel puts the button back.
        dock->toggleViewAction()->setChecked(false);
        QVERIFY(!frame->isVisible());
        QPushButton* button = nullptr;
        for (auto* candidate : host->findChildren<QPushButton*>()) {
            if (candidate->property("statusDockPanelButton").toBool()) {
                if (candidate->text().contains(QStringLiteral("Volume Package"))) {
                    button = candidate;
                }
            }
        }
        QVERIFY(button);
        QVERIFY(button->isHidden());
        catalog.invoke(QStringLiteral("panel.volumes"));
        QVERIFY(!button->isHidden());
        QVERIFY(dock->toggleViewAction()->isChecked());
        QVERIFY(frame->isVisible());
    }

    void panelsOpenOverAnotherWorkspaceTab()
    {
        // As in CWindow: the panel host is the central widget of the "main"
        // workspace tab and its button bar sits in the window's status bar,
        // so a panel opens over whichever workspace tab is showing.
        QMainWindow window;
        window.resize(900, 600);
        auto* tabs = new QTabWidget(&window);
        window.setCentralWidget(tabs);
        auto* workspace = new QMainWindow(&window);
        auto* host = new StatusDockPanelHost(new QWidget, workspace);
        workspace->setCentralWidget(host);
        auto* dock = new QDockWidget(QStringLiteral("Segmentation"), workspace);
        dock->setObjectName(QStringLiteral("dockWidgetSegmentation"));
        auto* content = new QLabel(QStringLiteral("segmentation"));
        dock->setWidget(content);
        workspace->addDockWidget(Qt::RightDockWidgetArea, dock);
        tabs->addTab(workspace, QStringLiteral("main"));
        auto* other = new QWidget;
        tabs->addTab(other, QStringLiteral("Lasagna"));
        window.statusBar()->insertWidget(0, host->takeBarWidget(), 0);
        host->addDock(dock);
        window.show();
        QVERIFY(QTest::qWaitForWindowExposed(&window));
        tabs->setCurrentWidget(other);
        QVERIFY(!host->isVisible());

        CommandCatalog catalog;
        QVERIFY(catalog.addPanel(dock, host->openAction(dock)));
        QVERIFY(catalog.invoke(QStringLiteral("panel.segmentation")));
        auto* frame = window.findChild<QFrame*>(QStringLiteral("statusDockPanelFrame"));
        QVERIFY(frame);
        QCOMPARE(frame->window(), &window);
        QVERIFY(frame->isVisible());
        QVERIFY(content->isVisible());
        // Like a click on its status bar button, it leaves the tab alone.
        QCOMPARE(tabs->currentWidget(), other);
    }

    void displayTextDropsMnemonicMarkers()
    {
        QCOMPARE(CommandCatalog::displayText(QStringLiteral("&Open Project...")), QStringLiteral("Open Project..."));
        QCOMPARE(CommandCatalog::displayText(QStringLiteral("Save Project &As...")), QStringLiteral("Save Project As..."));
        QCOMPARE(CommandCatalog::displayText(QStringLiteral("Fish && Chips")), QStringLiteral("Fish & Chips"));
        QCOMPARE(CommandCatalog::displayText(QStringLiteral("Plain")), QStringLiteral("Plain"));
    }

private:
    QTemporaryDir _config;
};

QTEST_MAIN(CommandCatalogTest)
#include "test_command_catalog.moc"
