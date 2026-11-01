#pragma once

// Pure helpers for synthesized keyboard and wheel input: modifier lists, key
// spelling, and the order in which modifier keys go down and up. Kept free of
// widgets so the contract test can check them without an application.

#include <utility>
#include <vector>

#include <QChar>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QKeyCombination>
#include <QKeySequence>
#include <QString>
#include <Qt>

#include "agent_bridge/AgentBridgeError.hpp"

// Parses the descriptor-validated "modifiers" array (shift, ctrl, alt, meta,
// keypad). Absent and null mean no modifiers.
inline Qt::KeyboardModifiers jsonToModifiers(const QJsonValue& value)
{
    Qt::KeyboardModifiers mods = Qt::NoModifier;
    if (value.isUndefined() || value.isNull())
        return mods;
    for (const QJsonValue& mv : value.toArray()) {
        const QString s = mv.toString();
        if (s == QLatin1String("shift"))
            mods |= Qt::ShiftModifier;
        else if (s == QLatin1String("ctrl"))
            mods |= Qt::ControlModifier;
        else if (s == QLatin1String("alt"))
            mods |= Qt::AltModifier;
        else if (s == QLatin1String("meta"))
            mods |= Qt::MetaModifier;
        else if (s == QLatin1String("keypad"))
            mods |= Qt::KeypadModifier;
    }
    return mods;
}

inline QJsonArray modifiersToJson(Qt::KeyboardModifiers mods)
{
    QJsonArray arr;
    if (mods & Qt::ShiftModifier)   arr.append(QStringLiteral("shift"));
    if (mods & Qt::ControlModifier) arr.append(QStringLiteral("ctrl"));
    if (mods & Qt::AltModifier)     arr.append(QStringLiteral("alt"));
    if (mods & Qt::MetaModifier)    arr.append(QStringLiteral("meta"));
    if (mods & Qt::KeypadModifier)  arr.append(QStringLiteral("keypad"));
    return arr;
}

// One key to synthesize. `modifiers` merges the modifiers spelled in the key
// string with the request's modifiers array.
struct AgentBridgeKeySpec {
    int key = 0;
    Qt::KeyboardModifiers modifiers = Qt::NoModifier;
    QString text;  // QKeyEvent::text() a keyboard would report
    QString name;  // portable name of the key alone, e.g. "E", "Esc"
};

// QKeyEvent text for a key: the character for printable keys without Ctrl,
// Alt or Meta, lower case unless Shift is held; empty otherwise.
inline QString agentBridgeKeyText(int key, Qt::KeyboardModifiers mods)
{
    if (mods & (Qt::ControlModifier | Qt::AltModifier | Qt::MetaModifier))
        return QString();
    switch (key) {
    case Qt::Key_Space:  return QStringLiteral(" ");
    case Qt::Key_Return:
    case Qt::Key_Enter:  return QStringLiteral("\r");
    case Qt::Key_Tab:    return QStringLiteral("\t");
    default: break;
    }
    if (key >= Qt::Key_A && key <= Qt::Key_Z) {
        const QChar c(static_cast<char16_t>(key));
        return (mods & Qt::ShiftModifier) ? QString(c) : QString(c.toLower());
    }
    if (key > Qt::Key_Space && key <= Qt::Key_AsciiTilde)
        return QString(QChar(static_cast<char16_t>(key)));
    return QString();
}

// Parses one key or key combination in QKeySequence portable text: "E", "e",
// "Esc", "Delete", "F5", "Space", "Ctrl+Z", "Shift+G". Bare modifier keys are
// spelled "Shift", "Ctrl" (or "Control"), "Alt", "Meta". Throws -32602 with
// data.param "key" for anything else, including sequences of several keys.
inline AgentBridgeKeySpec parseAgentBridgeKey(const QString& text,
                                              Qt::KeyboardModifiers extraModifiers)
{
    // QKeySequence spells the bare Control key "Control" and rejects "Ctrl"
    // on its own; accept both, matching the "ctrl" modifier name.
    const QString spelled =
        text.compare(QLatin1String("ctrl"), Qt::CaseInsensitive) == 0
            ? QStringLiteral("Control")
            : text;
    const QKeySequence sequence =
        QKeySequence::fromString(spelled, QKeySequence::PortableText);
    if (sequence.count() != 1 || sequence[0].key() == Qt::Key_unknown ||
        sequence[0].key() == 0) {
        QJsonObject data;
        data["param"] = QStringLiteral("key");
        data["detail"] = QStringLiteral(
            "expected one key in Qt portable text, e.g. \"E\", \"Esc\", "
            "\"Ctrl+Z\"");
        throw AgentBridgeError{-32602, QStringLiteral("key is not a single key"), data};
    }
    AgentBridgeKeySpec spec;
    spec.key = sequence[0].key();
    spec.modifiers = sequence[0].keyboardModifiers() | extraModifiers;
    spec.text = agentBridgeKeyText(spec.key, spec.modifiers);
    spec.name = QKeySequence(spec.key).toString(QKeySequence::PortableText);
    return spec;
}

// The modifiers a held key keeps down: its own, plus its flag when the key
// is itself a modifier key (a bare "Shift" press).
inline Qt::KeyboardModifiers agentBridgeHeldModifiers(int key, Qt::KeyboardModifiers mods)
{
    switch (key) {
    case Qt::Key_Shift:   return mods | Qt::ShiftModifier;
    case Qt::Key_Control: return mods | Qt::ControlModifier;
    case Qt::Key_Alt:     return mods | Qt::AltModifier;
    case Qt::Key_Meta:    return mods | Qt::MetaModifier;
    default:              return mods;
    }
}

// Modifier keys pressed before `mainKey`, in the order a keyboard user and
// QTest press them (Shift, Control, Alt, Meta), each paired with its flag.
// Keypad has no key of its own. A modifier that is the main key itself is
// left out so it is not pressed twice.
inline std::vector<std::pair<int, Qt::KeyboardModifier>> agentBridgeModifierKeys(
    Qt::KeyboardModifiers mods, int mainKey)
{
    static const std::pair<int, Qt::KeyboardModifier> kOrder[] = {
        {Qt::Key_Shift, Qt::ShiftModifier},
        {Qt::Key_Control, Qt::ControlModifier},
        {Qt::Key_Alt, Qt::AltModifier},
        {Qt::Key_Meta, Qt::MetaModifier},
    };
    std::vector<std::pair<int, Qt::KeyboardModifier>> keys;
    for (const auto& entry : kOrder) {
        if ((mods & entry.second) && entry.first != mainKey)
            keys.push_back(entry);
    }
    return keys;
}
