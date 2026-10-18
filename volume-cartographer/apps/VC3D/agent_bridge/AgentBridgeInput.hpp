#pragma once

// Pure helpers for synthesized canvas input. Kept free of widgets so the
// contract test can check them without an application.

#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <Qt>

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
