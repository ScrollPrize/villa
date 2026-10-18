// Offscreen widget test for the bridge's keyboard synthesis: shortcuts and
// widgets receive synthesized keys, the cached modifier state is reset, the
// dialogs and menus a key opens are answered as Escape would answer them,
// and keyboard focus returns to where it was. It needs no VC3D window.

#include <QApplication>
#include <QDialog>
#include <QJsonArray>
#include <QJsonObject>
#include <QLineEdit>
#include <QMenu>
#include <QMessageBox>
#include <QProgressDialog>
#include <QPushButton>
#include <QShortcut>
#include <QVBoxLayout>
#include <QtTest>

#include <functional>

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


QTEST_MAIN(AgentBridgeWidgetsTest)
#include "test_agent_bridge_widgets.moc"
