#pragma once

// Widget-level keyboard synthesis behind canvas.key, kept free of CWindow so
// the offscreen widget test can drive it: key strokes sent as real QKeyEvent
// objects, keyboard focus for the length of one call, the application's
// cached modifier state, and the scope that answers the dialogs and menus a
// key opens.

#include <QJsonArray>
#include <QObject>
#include <QPointer>
#include <QString>
#include <Qt>

class QAbstractButton;
class QEvent;
class QMessageBox;
class QWidget;

// One key, the modifiers held with it, and the text a keyboard reports.
struct AgentBridgeKeyStroke {
    int key = 0;
    Qt::KeyboardModifiers modifiers = Qt::NoModifier;
    QString text;
};

// Presses the stroke's modifier keys (Shift, Control, Alt, Meta), each
// carrying the modifiers already down, and then the key. QApplication::notify
// offers a non-spontaneous press to the shortcut map first and propagates an
// ignored event to the parent widgets, as for keyboard input. A key handler
// may delete the target, so each send checks it is alive. Returns whether a
// widget or shortcut accepted the key press. An auto-repeat sends the key
// press alone.
bool agentBridgeSendKeyPress(QWidget* target, const AgentBridgeKeyStroke& stroke,
                             bool autoRepeat);

// Releases the key and then its modifier keys in reverse order, each still
// reported held. Returns whether the key release was accepted.
bool agentBridgeSendKeyRelease(QWidget* target, const AgentBridgeKeyStroke& stroke);

// The modal dialog or popup that blocks a person's input to the other
// windows, or null. Synthesized events bypass that blocking, so canvas input
// checks it first.
QWidget* agentBridgeBlockingWindow();

// The button Escape clicks in a message box: its escape button, else Cancel,
// else its only button, else its one reject-role button, else its one
// no-role button (QMessageBox's own rule). Null when none qualifies.
QAbstractButton* agentBridgeEscapeAnswer(QMessageBox* box);

// QGuiApplication::keyboardModifiers() keeps the modifiers of the latest key
// press, and synthesized releases never clear it, so item views and other
// readers would keep seeing Ctrl or Shift after a synthesized Ctrl+Z. This
// sends one key press with no key to `sink`, a widget without a parent that
// is never shown, which sets the cached modifiers to `held` without reaching
// a visible widget or a shortcut.
void agentBridgeResetModifiers(QWidget* sink, Qt::KeyboardModifiers held);

// Moves keyboard focus to `target` for one call, as a click into it would, so
// shortcuts scoped to the viewer match. When the call ends, focus goes back
// to the widget that had it (or nowhere, if nothing had it), unless the key
// moved focus itself.
class AgentBridgeScopedFocus
{
public:
    explicit AgentBridgeScopedFocus(QWidget* target);
    ~AgentBridgeScopedFocus();
    AgentBridgeScopedFocus(const AgentBridgeScopedFocus&) = delete;
    AgentBridgeScopedFocus& operator=(const AgentBridgeScopedFocus&) = delete;

private:
    QPointer<QWidget> _target;
    QPointer<QWidget> _previous;
    bool _moved = false;
};

// While alive, answers every modal dialog and closes every menu that is
// shown, as Escape would: a message box gets agentBridgeEscapeAnswer() (or is
// rejected when it has none), any other modal dialog is rejected, and a menu
// is closed. The answer is queued, so the dialog's exec() returns as soon as
// its event loop starts. Non-modal windows and progress dialogs are left
// alone: rejecting a progress dialog would cancel its operation. Native
// dialogs are switched off for the scope, so a file picker is a Qt dialog
// that can be answered. When the scope ends, the window that was active
// before is activated again if no window is active.
class AgentBridgeWindowDismissal : public QObject
{
public:
    AgentBridgeWindowDismissal();
    ~AgentBridgeWindowDismissal() override;
    AgentBridgeWindowDismissal(const AgentBridgeWindowDismissal&) = delete;
    AgentBridgeWindowDismissal& operator=(const AgentBridgeWindowDismissal&) = delete;

    // {class, title, answer} per window, in the order they were shown.
    const QJsonArray& dismissed() const { return _dismissed; }

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    void note(const QWidget* window, const QString& answer);

    QPointer<QWidget> _previousActive;
    bool _nativeDialogsWereOff = false;
    QJsonArray _dismissed;
};
