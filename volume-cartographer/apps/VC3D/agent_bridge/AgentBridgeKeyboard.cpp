#include "agent_bridge/AgentBridgeKeyboard.hpp"

#include <QAbstractButton>
#include <QApplication>
#include <QCoreApplication>
#include <QDialog>
#include <QJsonObject>
#include <QKeyEvent>
#include <QMenu>
#include <QMessageBox>
#include <QMetaObject>
#include <QProgressDialog>
#include <QVariant>
#include <QWidget>

#include "agent_bridge/AgentBridgeInput.hpp"

namespace {

// Marks a window one dismissal scope already answered, so a nested scope
// does not answer it twice.
constexpr char kDismissedProperty[] = "agentBridgeDismissed";

bool sendKeyEvent(const QPointer<QWidget>& target, QEvent::Type type, int key,
                  Qt::KeyboardModifiers modifiers, const QString& text, bool autoRepeat)
{
    if (!target)
        return false;
    QKeyEvent event(type, key, modifiers, text, autoRepeat);
    QCoreApplication::sendEvent(target.data(), &event);
    return event.isAccepted();
}

QString buttonText(const QAbstractButton* button)
{
    QString text = button->text();
    text.replace(QStringLiteral("&&"), QStringLiteral("\x01"));
    text.remove(QLatin1Char('&'));
    text.replace(QStringLiteral("\x01"), QStringLiteral("&"));
    return text;
}

}  // namespace


bool agentBridgeSendKeyPress(QWidget* target, const AgentBridgeKeyStroke& stroke,
                             bool autoRepeat)
{
    const QPointer<QWidget> alive(target);
    if (!autoRepeat) {
        Qt::KeyboardModifiers held = Qt::NoModifier;
        for (const auto& [modifierKey, flag] :
             agentBridgeModifierKeys(stroke.modifiers, stroke.key)) {
            sendKeyEvent(alive, QEvent::KeyPress, modifierKey, held, QString(), false);
            held |= flag;
        }
    }
    return sendKeyEvent(alive, QEvent::KeyPress, stroke.key, stroke.modifiers, stroke.text,
                        autoRepeat);
}


bool agentBridgeSendKeyRelease(QWidget* target, const AgentBridgeKeyStroke& stroke)
{
    const QPointer<QWidget> alive(target);
    const bool accepted = sendKeyEvent(alive, QEvent::KeyRelease, stroke.key,
                                       stroke.modifiers, stroke.text, false);
    const auto modifiers = agentBridgeModifierKeys(stroke.modifiers, stroke.key);
    Qt::KeyboardModifiers held = stroke.modifiers;
    for (auto it = modifiers.rbegin(); it != modifiers.rend(); ++it) {
        sendKeyEvent(alive, QEvent::KeyRelease, it->first, held, QString(), false);
        held &= ~Qt::KeyboardModifiers(it->second);
    }
    return accepted;
}


QWidget* agentBridgeBlockingWindow()
{
    if (QWidget* modal = QApplication::activeModalWidget())
        return modal;
    return QApplication::activePopupWidget();
}


QAbstractButton* agentBridgeEscapeAnswer(QMessageBox* box)
{
    if (!box)
        return nullptr;
    if (QAbstractButton* button = box->escapeButton())
        return button;
    if (QAbstractButton* cancel = box->button(QMessageBox::Cancel))
        return cancel;
    const QList<QAbstractButton*> buttons = box->buttons();
    if (buttons.size() == 1)
        return buttons.front();
    for (const QMessageBox::ButtonRole role : {QMessageBox::RejectRole, QMessageBox::NoRole}) {
        QAbstractButton* match = nullptr;
        int count = 0;
        for (QAbstractButton* button : buttons) {
            if (box->buttonRole(button) == role) {
                match = button;
                ++count;
            }
        }
        if (count == 1)
            return match;
        if (count > 1)
            return nullptr;
    }
    return nullptr;
}


void agentBridgeResetModifiers(QWidget* sink, Qt::KeyboardModifiers held)
{
    if (!sink)
        return;
    QKeyEvent event(QEvent::KeyPress, 0, held);
    QCoreApplication::sendEvent(sink, &event);
}


AgentBridgeScopedFocus::AgentBridgeScopedFocus(QWidget* target)
    : _target(target)
{
    QWidget* window = target ? target->window() : nullptr;
    if (!window)
        return;
    _previous = window->focusWidget();
    _moved = _previous.data() != target;
    if (_moved)
        target->setFocus(Qt::OtherFocusReason);
}

AgentBridgeScopedFocus::~AgentBridgeScopedFocus()
{
    if (!_moved || !_target)
        return;
    QWidget* window = _target->window();
    if (!window || window->focusWidget() != _target.data())
        return;  // the key moved focus itself
    if (_previous)
        _previous->setFocus(Qt::OtherFocusReason);
    else
        _target->clearFocus();
}


AgentBridgeWindowDismissal::AgentBridgeWindowDismissal()
    : _previousActive(QApplication::activeWindow()),
      _nativeDialogsWereOff(QCoreApplication::testAttribute(Qt::AA_DontUseNativeDialogs))
{
    QCoreApplication::setAttribute(Qt::AA_DontUseNativeDialogs, true);
    qApp->installEventFilter(this);
}

AgentBridgeWindowDismissal::~AgentBridgeWindowDismissal()
{
    qApp->removeEventFilter(this);
    QCoreApplication::setAttribute(Qt::AA_DontUseNativeDialogs, _nativeDialogsWereOff);
    if (_previousActive && !QApplication::activeWindow())
        _previousActive->activateWindow();
}

bool AgentBridgeWindowDismissal::eventFilter(QObject* watched, QEvent* event)
{
    if (event->type() != QEvent::Show || !watched->isWidgetType())
        return false;
    auto* window = static_cast<QWidget*>(watched);
    if (!window->isWindow() || window->property(kDismissedProperty).toBool())
        return false;

    if (auto* menu = qobject_cast<QMenu*>(window)) {
        menu->setProperty(kDismissedProperty, true);
        note(menu, QStringLiteral("closed"));
        QMetaObject::invokeMethod(menu, [menu]() { menu->close(); }, Qt::QueuedConnection);
        return false;
    }

    auto* dialog = qobject_cast<QDialog*>(window);
    if (!dialog || !dialog->isModal() || qobject_cast<QProgressDialog*>(dialog))
        return false;
    dialog->setProperty(kDismissedProperty, true);
    if (auto* box = qobject_cast<QMessageBox*>(dialog)) {
        const QPointer<QAbstractButton> answer(agentBridgeEscapeAnswer(box));
        note(box, answer ? buttonText(answer) : QStringLiteral("rejected"));
        QMetaObject::invokeMethod(
            box,
            [box, answer]() {
                if (answer)
                    answer->click();
                else
                    box->reject();
            },
            Qt::QueuedConnection);
    } else {
        note(dialog, QStringLiteral("rejected"));
        QMetaObject::invokeMethod(dialog, [dialog]() { dialog->reject(); },
                                  Qt::QueuedConnection);
    }
    return false;
}

void AgentBridgeWindowDismissal::note(const QWidget* window, const QString& answer)
{
    QJsonObject entry;
    entry["class"] = QString::fromLatin1(window->metaObject()->className());
    entry["title"] = window->windowTitle();
    entry["answer"] = answer;
    _dismissed.append(entry);
}
