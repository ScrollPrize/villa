// Offscreen widget test for the bridge's widget-level code: keyboard
// synthesis (shortcuts and widgets receive synthesized keys, the cached
// modifier state is reset, the dialogs and menus a key opens are answered as
// Escape would answer them, keyboard focus returns), ui.inspect's
// accessibility tree with its redaction, and screenshot masking. It needs no
// VC3D window.

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDockWidget>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMainWindow>
#include <QMenu>
#include <QMessageBox>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QProgressDialog>
#include <QPushButton>
#include <QShortcut>
#include <QSpinBox>
#include <QTabWidget>
#include <QVBoxLayout>
#include <QtTest>

#include <functional>

#include "agent_bridge/AgentBridgeAccessible.hpp"
#include "agent_bridge/AgentBridgeInput.hpp"
#include "agent_bridge/AgentBridgeKeyboard.hpp"

namespace {

AgentBridgeKeyStroke stroke(const char* key)
{
    const AgentBridgeKeySpec spec =
        parseAgentBridgeKey(QString::fromLatin1(key), Qt::NoModifier);
    return {spec.key, spec.modifiers, spec.text};
}

// A window with a line edit and a focusable target, plus a window shortcut
// that runs `onShortcut`.
struct Fixture {
    QWidget window;
    QLineEdit* edit = nullptr;
    QLineEdit* target = nullptr;
    QShortcut* shortcut = nullptr;
    int fired = 0;
    std::function<void()> onShortcut;

    explicit Fixture(const char* sequence)
    {
        auto* layout = new QVBoxLayout(&window);
        edit = new QLineEdit(&window);
        target = new QLineEdit(&window);
        layout->addWidget(edit);
        layout->addWidget(target);
        shortcut = new QShortcut(QKeySequence(QString::fromLatin1(sequence)), &window);
        QObject::connect(shortcut, &QShortcut::activated, &window, [this]() {
            ++fired;
            if (onShortcut)
                onShortcut();
        });
        window.resize(240, 120);
        window.show();
        window.activateWindow();
    }
};

}  // namespace


class AgentBridgeWidgetsTest : public QObject
{
    Q_OBJECT

private slots:
    void keyReachesShortcutsAndWidgets();
    void modifierStateIsReset();
    void modalDialogIsRejectedAndActivationRestored();
    void messageBoxGetsItsEscapeAnswer();
    void escapeAnswerFollowsQMessageBoxRule();
    void progressAndNonModalDialogsAreLeftAlone();
    void menuIsClosed();
    void nestedScopesAnswerOnce();
    void blockingWindowIsReported();
    void scopedFocusReturnsFocus();
    void inspectReadsLabelsValuesAndItems();
    void inspectRedactsCredentials();
    void inspectRedactionDoesNotDependOnRoot();
    void inspectBoundsDepthNodesAndViewers();
    void screenshotMaskCoversSecrets();
};


void AgentBridgeWidgetsTest::keyReachesShortcutsAndWidgets()
{
    Fixture f("Ctrl+K");
    QVERIFY(QTest::qWaitForWindowActive(&f.window));
    f.target->setFocus();

    QVERIFY(agentBridgeSendKeyPress(f.target, stroke("Ctrl+K"), false));
    agentBridgeSendKeyRelease(f.target, stroke("Ctrl+K"));
    QCOMPARE(f.fired, 1);

    agentBridgeSendKeyPress(f.target, stroke("x"), false);
    agentBridgeSendKeyRelease(f.target, stroke("x"));
    QCOMPARE(f.target->text(), QStringLiteral("x"));

    // A target deleted by a handler is not touched again.
    QPointer<QLineEdit> doomed(new QLineEdit(&f.window));
    f.onShortcut = [&]() { delete doomed.data(); };
    agentBridgeSendKeyPress(doomed, stroke("Ctrl+K"), false);
    QVERIFY(doomed.isNull());
    QVERIFY(!agentBridgeSendKeyRelease(doomed, stroke("Ctrl+K")));
}


void AgentBridgeWidgetsTest::modifierStateIsReset()
{
    Fixture f("Ctrl+K");
    QVERIFY(QTest::qWaitForWindowActive(&f.window));
    QWidget sink;

    agentBridgeSendKeyPress(f.target, stroke("Ctrl+Z"), false);
    agentBridgeSendKeyRelease(f.target, stroke("Ctrl+Z"));
    agentBridgeResetModifiers(&sink, Qt::NoModifier);
    QCOMPARE(QGuiApplication::keyboardModifiers(), Qt::KeyboardModifiers(Qt::NoModifier));

    // A key still held keeps its modifiers; the reset fires no shortcut.
    agentBridgeResetModifiers(&sink, Qt::ControlModifier);
    QCOMPARE(QGuiApplication::keyboardModifiers(), Qt::KeyboardModifiers(Qt::ControlModifier));
    QCOMPARE(f.fired, 0);
    agentBridgeResetModifiers(&sink, Qt::NoModifier);
    QCOMPARE(QGuiApplication::keyboardModifiers(), Qt::KeyboardModifiers(Qt::NoModifier));
    QCOMPARE(f.edit->text(), QString());
    QCOMPARE(f.target->text(), QString());
}


void AgentBridgeWidgetsTest::modalDialogIsRejectedAndActivationRestored()
{
    Fixture f("Ctrl+O");
    QVERIFY(QTest::qWaitForWindowActive(&f.window));
    int result = -1;
    f.onShortcut = [&]() {
        QDialog dialog(&f.window);
        dialog.setWindowTitle(QStringLiteral("Open Project"));
        result = dialog.exec();
    };
    QJsonArray dismissed;
    {
        AgentBridgeWindowDismissal dismissal;
        QVERIFY(QCoreApplication::testAttribute(Qt::AA_DontUseNativeDialogs));
        agentBridgeSendKeyPress(f.target, stroke("Ctrl+O"), false);
        agentBridgeSendKeyRelease(f.target, stroke("Ctrl+O"));
        dismissed = dismissal.dismissed();
    }
    QCOMPARE(result, int(QDialog::Rejected));
    QCOMPARE(dismissed.size(), 1);
    QCOMPARE(dismissed[0].toObject().value("class").toString(), QStringLiteral("QDialog"));
    QCOMPARE(dismissed[0].toObject().value("title").toString(), QStringLiteral("Open Project"));
    QCOMPARE(dismissed[0].toObject().value("answer").toString(), QStringLiteral("rejected"));
    QVERIFY(!QCoreApplication::testAttribute(Qt::AA_DontUseNativeDialogs));

    // The main window is active again, so a window shortcut still fires.
    QTRY_COMPARE(QApplication::activeWindow(), &f.window);
    f.onShortcut = nullptr;
    agentBridgeSendKeyPress(f.target, stroke("Ctrl+O"), false);
    agentBridgeSendKeyRelease(f.target, stroke("Ctrl+O"));
    QCOMPARE(f.fired, 2);
}


void AgentBridgeWidgetsTest::messageBoxGetsItsEscapeAnswer()
{
    Fixture f("F6");
    QVERIFY(QTest::qWaitForWindowActive(&f.window));
    QMessageBox::StandardButton answer = QMessageBox::NoButton;
    std::function<QMessageBox::StandardButton()> prompt;
    f.onShortcut = [&]() { answer = prompt(); };
    // Each prompt is a separate request, so the event loop runs in between
    // and the main window is active again.
    auto ask = [&]() {
        QJsonArray dismissed;
        {
            AgentBridgeWindowDismissal dismissal;
            agentBridgeSendKeyPress(f.target, stroke("F6"), false);
            agentBridgeSendKeyRelease(f.target, stroke("F6"));
            dismissed = dismissal.dismissed();
        }
        (void)QTest::qWaitFor([&]() { return QApplication::activeWindow() == &f.window; });
        return dismissed;
    };

    // Yes|No answers No, never the destructive Yes.
    prompt = [&]() { return QMessageBox::question(&f.window, "Overwrite?", "Replace it?"); };
    QJsonArray dismissed = ask();
    QCOMPARE(answer, QMessageBox::No);
    QCOMPARE(dismissed.size(), 1);
    // The answer is the button's text without its mnemonic marker.
    QCOMPARE(dismissed[0].toObject().value("answer").toString(), QStringLiteral("No"));

    prompt = [&]() {
        return QMessageBox::warning(&f.window, "Unsaved", "Save changes?",
                                    QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);
    };
    ask();
    QCOMPARE(answer, QMessageBox::Cancel);

    prompt = [&]() { return QMessageBox::information(&f.window, "Done", "Finished."); };
    ask();
    QCOMPARE(answer, QMessageBox::Ok);

    // No escape answer: the box is closed without a button.
    prompt = [&]() {
        return QMessageBox::warning(&f.window, "Unsaved", "Save changes?",
                                    QMessageBox::Save | QMessageBox::Discard);
    };
    dismissed = ask();
    QVERIFY(answer != QMessageBox::Save && answer != QMessageBox::Discard);
    QCOMPARE(dismissed[0].toObject().value("answer").toString(), QStringLiteral("rejected"));
}


void AgentBridgeWidgetsTest::escapeAnswerFollowsQMessageBoxRule()
{
    QMessageBox yesNo(QMessageBox::Question, "", "", QMessageBox::Yes | QMessageBox::No);
    QCOMPARE(agentBridgeEscapeAnswer(&yesNo), yesNo.button(QMessageBox::No));

    QMessageBox saveCancel(QMessageBox::Warning, "", "",
                           QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);
    QCOMPARE(agentBridgeEscapeAnswer(&saveCancel), saveCancel.button(QMessageBox::Cancel));

    QMessageBox single(QMessageBox::Information, "", "", QMessageBox::Ok);
    QCOMPARE(agentBridgeEscapeAnswer(&single), single.button(QMessageBox::Ok));

    QMessageBox noEscape(QMessageBox::Warning, "", "", QMessageBox::Save | QMessageBox::Discard);
    QCOMPARE(agentBridgeEscapeAnswer(&noEscape), static_cast<QAbstractButton*>(nullptr));

    QMessageBox custom(QMessageBox::Question, "", "");
    QPushButton* keep = custom.addButton(QStringLiteral("Keep"), QMessageBox::AcceptRole);
    custom.addButton(QStringLiteral("Delete"), QMessageBox::DestructiveRole);
    custom.setEscapeButton(keep);
    QCOMPARE(agentBridgeEscapeAnswer(&custom), static_cast<QAbstractButton*>(keep));
    QCOMPARE(agentBridgeEscapeAnswer(nullptr), static_cast<QAbstractButton*>(nullptr));
}


void AgentBridgeWidgetsTest::progressAndNonModalDialogsAreLeftAlone()
{
    Fixture f("F2");
    QVERIFY(QTest::qWaitForWindowActive(&f.window));
    QProgressDialog progress(QStringLiteral("Working"), QStringLiteral("Cancel"), 0, 10,
                             &f.window);
    progress.setWindowModality(Qt::WindowModal);
    progress.setMinimumDuration(0);
    QDialog tool(&f.window);
    tool.setModal(false);
    f.onShortcut = [&]() {
        progress.show();
        tool.show();
    };
    QJsonArray dismissed;
    {
        AgentBridgeWindowDismissal dismissal;
        agentBridgeSendKeyPress(f.target, stroke("F2"), false);
        agentBridgeSendKeyRelease(f.target, stroke("F2"));
        QCoreApplication::processEvents();
        dismissed = dismissal.dismissed();
    }
    QCoreApplication::processEvents();
    QVERIFY(dismissed.isEmpty());
    QVERIFY(!progress.wasCanceled());
    QVERIFY(progress.isVisible());
    QVERIFY(tool.isVisible());
}


void AgentBridgeWidgetsTest::menuIsClosed()
{
    Fixture f("Menu");
    QVERIFY(QTest::qWaitForWindowActive(&f.window));
    bool returned = false;
    QAction* chosen = reinterpret_cast<QAction*>(1);
    f.onShortcut = [&]() {
        QMenu menu(&f.window);
        menu.addAction(QStringLiteral("Delete segment"));
        chosen = menu.exec(f.window.mapToGlobal(QPoint(10, 10)));
        returned = true;
    };
    QJsonArray dismissed;
    {
        AgentBridgeWindowDismissal dismissal;
        agentBridgeSendKeyPress(f.target, stroke("Menu"), false);
        agentBridgeSendKeyRelease(f.target, stroke("Menu"));
        dismissed = dismissal.dismissed();
    }
    QVERIFY(returned);
    QCOMPARE(chosen, static_cast<QAction*>(nullptr));
    QCOMPARE(dismissed.size(), 1);
    QCOMPARE(dismissed[0].toObject().value("answer").toString(), QStringLiteral("closed"));
}


void AgentBridgeWidgetsTest::nestedScopesAnswerOnce()
{
    Fixture f("F3");
    QVERIFY(QTest::qWaitForWindowActive(&f.window));
    int finished = 0;
    f.onShortcut = [&]() {
        QDialog dialog(&f.window);
        QObject::connect(&dialog, &QDialog::finished, &dialog, [&]() { ++finished; });
        dialog.exec();
    };
    AgentBridgeWindowDismissal outer;
    {
        AgentBridgeWindowDismissal inner;
        agentBridgeSendKeyPress(f.target, stroke("F3"), false);
        agentBridgeSendKeyRelease(f.target, stroke("F3"));
        QCOMPARE(outer.dismissed().size() + inner.dismissed().size(), qsizetype(1));
    }
    QCoreApplication::processEvents();
    QCOMPARE(finished, 1);
}


void AgentBridgeWidgetsTest::blockingWindowIsReported()
{
    Fixture f("F4");
    QVERIFY(QTest::qWaitForWindowActive(&f.window));
    QCOMPARE(agentBridgeBlockingWindow(), static_cast<QWidget*>(nullptr));
    QDialog dialog(&f.window);
    dialog.open();
    QCOMPARE(agentBridgeBlockingWindow(), static_cast<QWidget*>(&dialog));
    dialog.reject();
    QCOMPARE(agentBridgeBlockingWindow(), static_cast<QWidget*>(nullptr));
}


void AgentBridgeWidgetsTest::scopedFocusReturnsFocus()
{
    Fixture f("F5");
    QVERIFY(QTest::qWaitForWindowActive(&f.window));
    f.edit->setFocus();
    QCOMPARE(f.window.focusWidget(), static_cast<QWidget*>(f.edit));
    {
        AgentBridgeScopedFocus focus(f.target);
        QCOMPARE(f.window.focusWidget(), static_cast<QWidget*>(f.target));
    }
    QCOMPARE(f.window.focusWidget(), static_cast<QWidget*>(f.edit));

    // A key that moves focus keeps its move.
    auto* other = new QLineEdit(&f.window);
    other->show();
    {
        AgentBridgeScopedFocus focus(f.target);
        other->setFocus();
    }
    QCOMPARE(f.window.focusWidget(), static_cast<QWidget*>(other));
}


namespace {

// A main window with the kinds of controls VC3D's panels use, including
// fields that hold credentials.
struct Panel {
    QMainWindow window;
    QLineEdit* apiKey = nullptr;
    QLineEdit* password = nullptr;
    QLineEdit* endpoint = nullptr;
    QLineEdit* token = nullptr;
    QLineEdit* draws = nullptr;
    QLabel* signedUrl = nullptr;
    QWidget* viewer = nullptr;
    QDialog* settings = nullptr;

    Panel()
    {
        auto* central = new QWidget;
        window.setCentralWidget(central);
        auto* form = new QFormLayout(central);
        auto* steps = new QSpinBox;
        steps->setRange(1, 99);
        steps->setValue(7);
        form->addRow(QStringLiteral("&Steps:"), steps);
        auto* method = new QComboBox;
        method->addItems({QStringLiteral("tracer"), QStringLiteral("corrections")});
        form->addRow(QStringLiteral("Method:"), method);
        auto* editing = new QCheckBox(QStringLiteral("Enable editing"));
        editing->setChecked(true);
        form->addRow(editing);
        apiKey = new QLineEdit(QStringLiteral("abc123"));
        form->addRow(QStringLiteral("API key:"), apiKey);
        password = new QLineEdit(QStringLiteral("hunter2"));
        password->setEchoMode(QLineEdit::Password);
        form->addRow(QStringLiteral("SSH:"), password);
        // A form row whose field is a container around the entry.
        auto* row = new QWidget;
        auto* rowLayout = new QHBoxLayout(row);
        endpoint = new QLineEdit(QStringLiteral("http://user:pw@gpu-host:8765"));
        endpoint->setPlaceholderText(QStringLiteral("http://gpu-host:8765"));
        rowLayout->addWidget(endpoint);
        rowLayout->addWidget(new QPushButton(QStringLiteral("Test")));
        form->addRow(QStringLiteral("Endpoint:"), row);
        auto* tokenRow = new QWidget;
        auto* tokenLayout = new QHBoxLayout(tokenRow);
        token = new QLineEdit(QStringLiteral("t0k3n"));
        tokenLayout->addWidget(token);
        form->addRow(QStringLiteral("Token:"), tokenRow);
        draws = new QLineEdit(QStringLiteral("12"));
        form->addRow(QStringLiteral("Draws:"), draws);
        auto* checkpoint = new QPlainTextEdit(QStringLiteral("{\"checkpoint_path\": \"/models/a.pt\"}"));
        form->addRow(QStringLiteral("Custom params:"), checkpoint);
        signedUrl = new QLabel(QStringLiteral(
            "https://bucket.s3.amazonaws.com/a.zarr?X-Amz-Signature=abc&X-Amz-Credential=def"));
        form->addRow(signedUrl);
        auto* segments = new QListWidget;
        for (int i = 0; i < 5; ++i)
            segments->addItem(QStringLiteral("seg %1").arg(i));
        segments->setCurrentRow(2);
        form->addRow(QStringLiteral("Segments:"), segments);
        auto* tabs = new QTabWidget;
        tabs->addTab(new QLabel(QStringLiteral("first page")), QStringLiteral("One"));
        tabs->addTab(new QLabel(QStringLiteral("second page")), QStringLiteral("Two"));
        form->addRow(tabs);
        viewer = new QWidget;
        viewer->setObjectName(QStringLiteral("viewerPane"));
        auto* viewerLayout = new QVBoxLayout(viewer);
        viewerLayout->addWidget(new QLabel(QStringLiteral("inside viewer")));
        form->addRow(viewer);

        auto* dock = new QDockWidget(QStringLiteral("Growth"));
        dock->setObjectName(QStringLiteral("dockGrowth"));
        auto* group = new QGroupBox(QStringLiteral("Radius"));
        auto* groupForm = new QFormLayout(group);
        groupForm->addRow(QStringLiteral("Brush:"), new QSpinBox);
        dock->setWidget(group);
        window.addDockWidget(Qt::RightDockWidgetArea, dock);

        settings = new QDialog(&window);
        settings->setWindowTitle(QStringLiteral("Settings"));
        auto* settingsLayout = new QVBoxLayout(settings);
        settingsLayout->addWidget(new QLineEdit(QStringLiteral("AKIAABCDEFGHIJKLMNOP")));

        window.resize(400, 900);
        window.show();
        settings->show();
    }
};

void walk(const QJsonObject& node, const std::function<void(const QJsonObject&)>& visit)
{
    visit(node);
    for (const QJsonValue& child : node.value("children").toArray())
        walk(child.toObject(), visit);
}

QJsonObject findNode(const QJsonObject& tree,
                     const std::function<bool(const QJsonObject&)>& match)
{
    QJsonObject found;
    walk(tree, [&](const QJsonObject& node) {
        if (found.isEmpty() && match(node))
            found = node;
    });
    return found;
}

QJsonObject named(const QJsonObject& tree, const QString& role, const QString& name)
{
    return findNode(tree, [&](const QJsonObject& node) {
        return node.value("role").toString() == role && node.value("name").toString() == name;
    });
}

AgentBridgeInspectOptions inspectOptions(int depth, bool includeHidden = false, int maxNodes = 4000)
{
    AgentBridgeInspectOptions options;
    options.maxDepth = depth;
    options.includeHidden = includeHidden;
    options.maxNodes = maxNodes;
    return options;
}

QJsonObject entry(const QJsonObject& tree, const QString& name)
{
    return named(tree, QStringLiteral("EditableText"), name);
}

}  // namespace


void AgentBridgeWidgetsTest::inspectReadsLabelsValuesAndItems()
{
    Panel panel;
    QVERIFY(QTest::qWaitForWindowExposed(&panel.window));
    const AgentBridgeInspection result = agentBridgeInspect(&panel.window, inspectOptions(64));
    QVERIFY(!result.truncated);
    QCOMPARE(result.tree.value("class").toString(), QStringLiteral("QMainWindow"));

    const QJsonObject steps = named(result.tree, QStringLiteral("SpinBox"), QStringLiteral("Steps"));
    QCOMPARE(steps.value("value").toInt(), 7);
    QCOMPARE(steps.value("minimum").toInt(), 1);
    QCOMPARE(steps.value("maximum").toInt(), 99);

    const QJsonObject method = findNode(result.tree, [](const QJsonObject& n) {
        return n.value("role").toString() == QLatin1String("ComboBox");
    });
    QCOMPARE(method.value("value").toString(), QStringLiteral("tracer"));
    QCOMPARE(method.value("items").toArray(),
             QJsonArray({QStringLiteral("tracer"), QStringLiteral("corrections")}));

    const QJsonObject editing =
        named(result.tree, QStringLiteral("CheckBox"), QStringLiteral("Enable editing"));
    QCOMPARE(editing.value("checked").toBool(), true);

    const QJsonObject segments = named(result.tree, QStringLiteral("List"), QStringLiteral("Segments"));
    QCOMPARE(segments.value("rows").toInt(), 5);
    QCOMPARE(segments.value("items").toArray().at(4).toString(), QStringLiteral("seg 4"));
    QCOMPARE(segments.value("selectedRows").toArray(), QJsonArray({2}));

    // Not a credential: values in fields named like programs or checkpoints are read.
    QCOMPARE(entry(result.tree, QStringLiteral("Draws")).value("value").toString(),
             QStringLiteral("12"));
    QVERIFY(entry(result.tree, QStringLiteral("Custom params"))
                .value("value").toString().contains(QStringLiteral("checkpoint_path")));

    const QJsonObject dock = findNode(result.tree, [](const QJsonObject& n) {
        return n.value("objectName").toString() == QLatin1String("dockGrowth");
    });
    QCOMPARE(dock.value("name").toString(), QStringLiteral("Growth"));
    QVERIFY(!named(dock, QStringLiteral("SpinBox"), QStringLiteral("Brush")).isEmpty());

    // Only the current tab page is listed unless hidden widgets are requested.
    QVERIFY(!named(result.tree, QStringLiteral("StaticText"), QStringLiteral("first page")).isEmpty());
    QVERIFY(named(result.tree, QStringLiteral("StaticText"), QStringLiteral("second page")).isEmpty());
    const AgentBridgeInspection hidden =
        agentBridgeInspect(&panel.window, inspectOptions(64, true));
    QVERIFY(!named(hidden.tree, QStringLiteral("StaticText"), QStringLiteral("second page")).isEmpty());
}


void AgentBridgeWidgetsTest::inspectRedactsCredentials()
{
    Panel panel;
    QVERIFY(QTest::qWaitForWindowExposed(&panel.window));
    const AgentBridgeInspection result = agentBridgeInspect(&panel.window, inspectOptions(64));

    for (const char* name : {"API key", "SSH", "Token"}) {
        const QJsonObject node = entry(result.tree, QString::fromLatin1(name));
        QVERIFY2(!node.isEmpty(), name);
        QVERIFY2(node.value("redacted").toBool(), name);
        QVERIFY2(!node.contains("value"), name);
    }
    // A URL with a password in a field whose label names no credential.
    const QJsonObject endpoint = entry(result.tree, QStringLiteral("Endpoint"));
    QVERIFY(endpoint.value("redacted").toBool());
    QVERIFY(!endpoint.contains("value"));
    QCOMPARE(endpoint.value("description").toString(), QStringLiteral("http://gpu-host:8765"));

    // A signed URL shown in a label.
    const QJsonObject label = findNode(result.tree, [](const QJsonObject& n) {
        return n.value("role").toString() == QLatin1String("StaticText") &&
               n.value("redacted").toBool();
    });
    QVERIFY(!label.isEmpty());
    QVERIFY(!label.contains("name"));

    // Other windows, such as a settings dialog, are never entered.
    bool sawSettings = false;
    walk(result.tree, [&](const QJsonObject& n) {
        sawSettings = sawSettings || n.value("class").toString() == QLatin1String("QDialog") ||
                      n.value("value").toString().startsWith(QLatin1String("AKIA"));
    });
    QVERIFY(!sawSettings);

    // Nothing redacted ever carries a value.
    walk(result.tree, [](const QJsonObject& n) {
        if (n.value("redacted").toBool() && n.value("role").toString() == QLatin1String("EditableText"))
            QVERIFY(!n.contains("value"));
    });

    QVERIFY(agentBridgeWidgetShowsSecret(panel.apiKey));
    QVERIFY(agentBridgeWidgetShowsSecret(panel.password));
    QVERIFY(agentBridgeWidgetShowsSecret(panel.endpoint));
    QVERIFY(agentBridgeWidgetShowsSecret(panel.signedUrl));
    QVERIFY(!agentBridgeWidgetShowsSecret(panel.draws));
    panel.endpoint->setText(QStringLiteral("http://gpu-host:8765"));
    QVERIFY(!agentBridgeWidgetShowsSecret(panel.endpoint));
}


void AgentBridgeWidgetsTest::inspectRedactionDoesNotDependOnRoot()
{
    Panel panel;
    QVERIFY(QTest::qWaitForWindowExposed(&panel.window));
    // The token entry has no label of its own; its row container does.
    const AgentBridgeInspection fromEntry = agentBridgeInspect(panel.token, inspectOptions(8));
    QCOMPARE(fromEntry.tree.value("name").toString(), QStringLiteral("Token"));
    QVERIFY(fromEntry.tree.value("redacted").toBool());
    QVERIFY(!fromEntry.tree.contains("value"));
    const AgentBridgeInspection fromRow = agentBridgeInspect(panel.token->parentWidget(), inspectOptions(8));
    QVERIFY(entry(fromRow.tree, QStringLiteral("Token")).value("redacted").toBool());
}


void AgentBridgeWidgetsTest::inspectBoundsDepthNodesAndViewers()
{
    Panel panel;
    QVERIFY(QTest::qWaitForWindowExposed(&panel.window));
    const AgentBridgeInspection shallow = agentBridgeInspect(&panel.window, inspectOptions(0));
    QVERIFY(shallow.tree.value("childrenOmitted").toBool());
    QVERIFY(!shallow.tree.contains("children"));
    QCOMPARE(shallow.nodeCount, 1);

    const AgentBridgeInspection capped =
        agentBridgeInspect(&panel.window, inspectOptions(64, false, 5));
    QVERIFY(capped.truncated);
    QCOMPARE(capped.nodeCount, 5);

    AgentBridgeInspectOptions options = inspectOptions(64);
    options.viewerIds.insert(panel.viewer, QStringLiteral("v1"));
    const AgentBridgeInspection withViewer = agentBridgeInspect(&panel.window, options);
    const QJsonObject pane = findNode(withViewer.tree, [](const QJsonObject& n) {
        return n.value("viewerId").toString() == QLatin1String("v1");
    });
    QCOMPARE(pane.value("objectName").toString(), QStringLiteral("viewerPane"));
    QVERIFY(!pane.contains("children"));
    QVERIFY(named(withViewer.tree, QStringLiteral("StaticText"), QStringLiteral("inside viewer")).isEmpty());
}


void AgentBridgeWidgetsTest::screenshotMaskCoversSecrets()
{
    Panel panel;
    QVERIFY(QTest::qWaitForWindowExposed(&panel.window));
    QWidget* central = panel.window.centralWidget();
    const QRect area = central->rect();
    QPixmap pixmap = central->grab(area);
    // The API key, SSH password, endpoint with a password, token, and the
    // signed URL label.
    QCOMPARE(agentBridgeMaskSecrets(pixmap, central, area), 5);
    const QImage image = pixmap.toImage();
    auto colorAt = [&](QWidget* widget) {
        const QPoint center = widget->mapTo(central, widget->rect().center());
        const qreal ratio = pixmap.devicePixelRatio();
        return image.pixelColor(QPoint(int(center.x() * ratio), int(center.y() * ratio)));
    };
    QCOMPARE(colorAt(panel.apiKey), QColor(Qt::black));
    QCOMPARE(colorAt(panel.signedUrl), QColor(Qt::black));
    QVERIFY(colorAt(panel.draws) != QColor(Qt::black));

    // A region is masked in its own coordinates; widgets outside it are not
    // counted.
    const QRect apiKeyRect(panel.apiKey->mapTo(central, QPoint(0, 0)), panel.apiKey->size());
    QPixmap part = central->grab(apiKeyRect);
    QCOMPARE(agentBridgeMaskSecrets(part, central, apiKeyRect), 1);
    QCOMPARE(part.toImage().pixelColor(2, 2), QColor(Qt::black));
}


QTEST_MAIN(AgentBridgeWidgetsTest)
#include "test_agent_bridge_widgets.moc"
