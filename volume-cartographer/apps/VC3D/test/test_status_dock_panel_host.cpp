// Offscreen test for StatusDockPanelHost's panel API, which the agent bridge
// uses to read and capture docks kept in the bottom dock bar: the panel page
// that holds a dock's content, the open panel, and opening and closing a
// panel the way a bar button does. Settings go to a temporary folder.

#include <QAction>
#include <QDockWidget>
#include <QMainWindow>
#include <QSpinBox>
#include <QStatusBar>
#include <QTemporaryDir>
#include <QVBoxLayout>
#include <QtTest>

#include "StatusDockPanelHost.hpp"

namespace {

QDockWidget* makeDock(QMainWindow* window, const QString& objectName, const QString& title)
{
    auto* dock = new QDockWidget(title, window);
    dock->setObjectName(objectName);
    auto* content = new QWidget(dock);
    content->setObjectName(objectName + QStringLiteral("Content"));
    auto* layout = new QVBoxLayout(content);
    auto* spin = new QSpinBox(content);
    spin->setObjectName(objectName + QStringLiteral("Spin"));
    layout->addWidget(spin);
    dock->setWidget(content);
    window->addDockWidget(Qt::RightDockWidgetArea, dock);
    return dock;
}

}  // namespace

class TestStatusDockPanelHost : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void panelApi();

private:
    QTemporaryDir _config;
};

void TestStatusDockPanelHost::initTestCase()
{
    QVERIFY(_config.isValid());
    qputenv("VC3D_CONFIG_DIR", _config.path().toLocal8Bit());
}

void TestStatusDockPanelHost::panelApi()
{
    QMainWindow window;
    window.resize(1200, 800);
    auto* central = new QWidget;
    auto* host = new StatusDockPanelHost(central, &window);
    window.setCentralWidget(host);
    QDockWidget* first = makeDock(&window, QStringLiteral("dockFirst"), QStringLiteral("First"));
    QDockWidget* second = makeDock(&window, QStringLiteral("dockSecond"), QStringLiteral("Second"));
    host->addDock(first);
    host->addDock(second);
    window.statusBar()->insertWidget(0, host->takeBarWidget(), 0);
    window.show();
    QVERIFY(QTest::qWaitForWindowExposed(&window));

    // The dock is an empty hidden shell; its content sits on a panel page.
    QWidget* firstPage = host->panelPage(first);
    QWidget* secondPage = host->panelPage(second);
    QVERIFY(firstPage);
    QVERIFY(secondPage);
    QVERIFY(firstPage != secondPage);
    QCOMPARE(first->widget(), static_cast<QWidget*>(nullptr));
    QVERIFY(!first->isVisible());
    QVERIFY(firstPage->findChild<QSpinBox*>(QStringLiteral("dockFirstSpin")));
    QCOMPARE(host->panelPage(nullptr), static_cast<QWidget*>(nullptr));
    QDockWidget stranger(QStringLiteral("Stranger"));
    QCOMPARE(host->panelPage(&stranger), static_cast<QWidget*>(nullptr));
    QCOMPARE(host->expandDock(&stranger), false);

    // Closed at start; opening one panel shows its page only.
    QCOMPARE(host->expandedDock(), static_cast<QDockWidget*>(nullptr));
    QVERIFY(!firstPage->isVisible());
    QVERIFY(host->expandDock(first));
    QCOMPARE(host->expandedDock(), first);
    QVERIFY(firstPage->isVisible());
    QVERIFY(!secondPage->isVisible());
    QVERIFY(firstPage->width() > 0 && firstPage->height() > 0);

    // Opening another switches the panel; closing shuts it.
    QVERIFY(host->expandDock(second));
    QCOMPARE(host->expandedDock(), second);
    QVERIFY(secondPage->isVisible());
    QVERIFY(!firstPage->isVisible());
    host->collapsePanel();
    QCOMPARE(host->expandedDock(), static_cast<QDockWidget*>(nullptr));
    QVERIFY(!firstPage->isVisible());
    QVERIFY(!secondPage->isVisible());

    // Opening and closing again works the same way and leaves the docks in
    // the bar (their toggle actions stay checked) and their shells hidden.
    for (int round = 0; round < 2; ++round) {
        QVERIFY(host->expandDock(first));
        QVERIFY(firstPage->isVisible());
        host->collapsePanel();
        QVERIFY(!firstPage->isVisible());
    }
    QVERIFY(first->toggleViewAction()->isChecked());
    QVERIFY(!first->isVisible());

    // A dock closed from the bar keeps its page but cannot be opened until
    // it is turned back on.
    first->toggleViewAction()->setChecked(false);
    QVERIFY(host->panelPage(first));
    QCOMPARE(host->expandDock(first), false);
    QCOMPARE(host->expandedDock(), static_cast<QDockWidget*>(nullptr));
    first->toggleViewAction()->setChecked(true);
    QVERIFY(host->expandDock(first));
    QCOMPARE(host->expandedDock(), first);
    host->collapsePanel();
}

QTEST_MAIN(TestStatusDockPanelHost)
#include "test_status_dock_panel_host.moc"
