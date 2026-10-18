#include "agent_bridge/AgentBridgeServer.hpp"
#include "agent_bridge/AgentBridgeInput.hpp"
#include "agent_bridge/AgentBridgeInternal.hpp"
#include "agent_bridge/AgentBridgeKeyboard.hpp"

#include <QCoreApplication>
#include <QJsonArray>
#include <QKeySequence>
#include <QLocalSocket>
#include <QTimer>
#include <QWheelEvent>
#include <QWidget>

#include <algorithm>

#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"

namespace {

// Wheel deltas use Qt's angle units: 120 per notch of a standard mouse wheel.
constexpr int kWheelNotch = 120;
// A press without holdMs is released automatically after this long.
constexpr int kDefaultPressLimitMs = 10000;
// A hold's deferred response outlives the hold by this margin.
constexpr int kHoldResponseMarginMs = 5000;

QString keyName(int key)
{
    return QKeySequence(key).toString(QKeySequence::PortableText);
}

}  // namespace


void AgentBridgeServer::requireNoBlockingWindow() const
{
    QWidget* blocking = agentBridgeBlockingWindow();
    if (!blocking)
        return;
    QJsonObject window;
    window["class"] = QString::fromLatin1(blocking->metaObject()->className());
    window["title"] = blocking->windowTitle();
    QJsonObject data;
    data["detail"] = "a modal dialog or menu is open, so input to the viewers is blocked "
                     "until it closes";
    data["window"] = window;
    throw AgentBridgeError{-32009, "A modal window is open", data};
}


AgentBridgeServer::HeldKey& AgentBridgeServer::holdKey(QLocalSocket* socket, QWidget* target,
                                                        const QString& viewerId,
                                                        const AgentBridgeKeyStroke& stroke)
{
    HeldKey key;
    key.serial = _nextHeldKeySerial++;
    key.socket = socket;
    key.target = target;
    key.viewerId = viewerId;
    key.stroke = stroke;
    _heldKeys.push_back(key);
    return _heldKeys.back();
}


void AgentBridgeServer::resetKeyboardModifiers()
{
    if (!_keyboardModifierSink)
        _keyboardModifierSink = new QWidget;
    Qt::KeyboardModifiers held = Qt::NoModifier;
    for (const HeldKey& key : _heldKeys)
        held |= agentBridgeHeldModifiers(key.stroke.key, key.stroke.modifiers);
    agentBridgeResetModifiers(_keyboardModifierSink, held);
}


void AgentBridgeServer::scheduleHeldKeyRelease(int serial, int delayMs, int token,
                                               const QJsonObject& result,
                                               QLocalSocket* socket)
{
    const QPointer<QLocalSocket> owner(socket);
    QTimer::singleShot(delayMs, this, [this, serial, token, result, owner]() {
        runWhenIdle([this, serial, token, result, owner]() {
            finishHeldKey(serial, token, result, owner);
        });
    });
}


void AgentBridgeServer::finishHeldKey(int serial, int token, QJsonObject result,
                                      const QPointer<QLocalSocket>& socket)
{
    // The serial matches only the press this timer belongs to, so a key
    // released early and pressed again is not released by the old timer.
    const auto it = std::find_if(_heldKeys.begin(), _heldKeys.end(),
                                 [serial](const HeldKey& h) { return h.serial == serial; });
    const bool stillHeld = it != _heldKeys.end();
    if (stillHeld) {
        const HeldKey key = *it;
        _heldKeys.erase(it);
        AgentBridgeWindowDismissal dismissal;
        agentBridgeSendKeyRelease(key.target, key.stroke);
        resetKeyboardModifiers();
        QJsonArray dismissed = result["dismissed"].toArray();
        for (const QJsonValue& entry : dismissal.dismissed())
            dismissed.append(entry);
        result["dismissed"] = dismissed;
    }
    if (token == 0)
        return;
    result["releasedEarly"] = !stillHeld;
    result["heldKeys"] = heldKeyNames(socket.data());
    completeDeferredResult(token, result);
}


void AgentBridgeServer::releaseHeldKeys(QLocalSocket* socket)
{
    // Called on disconnect, possibly while a request is being handled; the
    // releases run once the bridge is idle.
    for (HeldKey& key : _heldKeys) {
        if (key.socket.data() == socket)
            key.orphaned = true;
    }
    runWhenIdle([this]() { releaseOrphanedKeys(); });
}


void AgentBridgeServer::releaseOrphanedKeys()
{
    std::vector<HeldKey> released;
    for (auto it = _heldKeys.begin(); it != _heldKeys.end();) {
        if (it->orphaned || it->socket.isNull()) {
            released.push_back(*it);
            it = _heldKeys.erase(it);
        } else {
            ++it;
        }
    }
    if (released.empty())
        return;
    AgentBridgeWindowDismissal dismissal;
    for (auto it = released.rbegin(); it != released.rend(); ++it)
        agentBridgeSendKeyRelease(it->target, it->stroke);
    resetKeyboardModifiers();
}


QJsonArray AgentBridgeServer::heldKeyNames(QLocalSocket* socket) const
{
    QJsonArray names;
    for (const HeldKey& held : _heldKeys) {
        if (held.socket.data() == socket && !held.orphaned)
            names.append(keyName(held.stroke.key));
    }
    return names;
}


QJsonArray AgentBridgeServer::heldKeysJson() const
{
    QJsonArray keys;
    for (const HeldKey& held : _heldKeys) {
        if (held.orphaned)
            continue;
        QJsonObject entry;
        entry["key"] = keyName(held.stroke.key);
        entry["modifiers"] = modifiersToJson(held.stroke.modifiers);
        entry["viewerId"] = held.viewerId;
        keys.append(entry);
    }
    return keys;
}


QJsonObject AgentBridgeServer::handleCanvasKey(const QJsonValue& params)
{
    // The caller, captured once: answering a dialog runs a nested event loop.
    QLocalSocket* const socket = _currentSocket;
    const QJsonObject p = params.toObject();
    const QString action = p.value("action").toString(QStringLiteral("tap"));
    const bool tap = action == QLatin1String("tap");
    const bool press = action == QLatin1String("press");
    const bool hold = p.contains("holdMs");
    if (hold && !tap && !press)
        throwParamError("holdMs", QStringLiteral("does not apply to action \"release\""));
    const int holdMs = p.value("holdMs").toInt(press ? kDefaultPressLimitMs : 0);

    const AgentBridgeKeySpec spec =
        parseAgentBridgeKey(p.value("key").toString(), jsonToModifiers(p.value("modifiers")));
    const AgentBridgeKeyStroke stroke{spec.key, spec.modifiers, spec.text};

    VolumeViewerBase* viewer = resolveViewer(p.value("viewer"));
    CChunkedVolumeViewer* chunked = requireChunkedViewer(viewer);
    QWidget* view = chunked->graphicsView();
    const QString viewerId = viewerIdFor(viewer);
    // A release is always delivered, so a held key can be let go behind a
    // dialog or after its workspace was switched away.
    if (action != QLatin1String("release")) {
        requireNoBlockingWindow();
        if (!view->isVisible()) {
            // Shortcuts are matched against the focused widget, which cannot
            // be a hidden viewer: the key would reach another widget's
            // shortcut.
            QJsonObject data;
            data["viewerId"] = viewerId;
            data["detail"] = "the viewer is not visible; switch to its workspace first";
            throw AgentBridgeError{-32009, "Viewer not visible", data};
        }
    }

    auto heldIt = std::find_if(_heldKeys.begin(), _heldKeys.end(), [&](const HeldKey& h) {
        return h.socket.data() == socket && !h.orphaned && h.stroke.key == spec.key;
    });
    const bool alreadyHeld = heldIt != _heldKeys.end();
    if (tap && alreadyHeld) {
        QJsonObject data;
        data["key"] = spec.name;
        data["detail"] = "a press on this connection holds the key; release it first";
        throw AgentBridgeError{-32009, "Key is already held", data};
    }
    // A hold answers after the release; claim the deferred slot before any
    // input is sent so a refused request changes nothing.
    const int token = tap && hold
                          ? beginDeferred(holdMs + kHoldResponseMarginMs,
                                          QStringLiteral("key release after %1 ms").arg(holdMs))
                          : 0;

    QJsonObject result;
    result["viewerId"] = viewerId;
    result["key"] = spec.name;
    result["modifiers"] = modifiersToJson(spec.modifiers);
    result["action"] = action;

    AgentBridgeScopedFocus focus(view);
    AgentBridgeWindowDismissal dismissal;
    if (press) {
        // A second press of a held key is the keyboard's auto-repeat.
        if (alreadyHeld) {
            const HeldKey held = *heldIt;
            result["accepted"] = agentBridgeSendKeyPress(held.target, held.stroke, true);
        } else {
            const int serial = holdKey(socket, view, viewerId, stroke).serial;
            result["accepted"] = agentBridgeSendKeyPress(view, stroke, false);
            result["holdMs"] = holdMs;
            scheduleHeldKeyRelease(serial, holdMs, 0, QJsonObject(), socket);
        }
        result["autoRepeat"] = alreadyHeld;
    } else if (action == QLatin1String("release")) {
        // A held key is released with the modifiers it was pressed with.
        HeldKey releasing;
        releasing.target = view;
        releasing.stroke = stroke;
        if (alreadyHeld) {
            releasing = *heldIt;
            _heldKeys.erase(heldIt);
        }
        result["accepted"] = agentBridgeSendKeyRelease(releasing.target, releasing.stroke);
        result["wasHeld"] = alreadyHeld;
        resetKeyboardModifiers();
    } else if (!hold) {
        result["accepted"] = agentBridgeSendKeyPress(view, stroke, false);
        agentBridgeSendKeyRelease(view, stroke);
        resetKeyboardModifiers();
    } else {
        // Hold: press now, release after holdMs, then answer. A disconnect
        // during the hold releases the key at once, and a release on this
        // connection ends the hold early.
        const int serial = holdKey(socket, view, viewerId, stroke).serial;
        result["accepted"] = agentBridgeSendKeyPress(view, stroke, false);
        result["holdMs"] = holdMs;
        result["dismissed"] = dismissal.dismissed();
        scheduleHeldKeyRelease(serial, holdMs, token, result, socket);
        throw AgentBridgeDeferred{};
    }
    result["heldKeys"] = heldKeyNames(socket);
    result["dismissed"] = dismissal.dismissed();
    return result;
}


QJsonObject AgentBridgeServer::handleCanvasWheel(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    const int deltaY = p.value("deltaY").toInt();
    if (deltaY == 0)
        throwParamError("deltaY", QStringLiteral("must be non-zero"));

    VolumeViewerBase* viewer = resolveViewer(p.value("viewer"));
    CChunkedVolumeViewer* chunked = requireChunkedViewer(viewer);
    if (!chunked->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};
    requireNoBlockingWindow();

    const Qt::KeyboardModifiers modifiers = jsonToModifiers(p.value("modifiers"));
    const QPointF scenePos = canvasScenePoint(
        chunked, p.value("position"),
        p.value("space").toString(QStringLiteral("volume")));

    // A wheel event belongs to the viewport under the pointer; the view maps
    // the global position back into viewport coordinates.
    CVolumeViewerView* view = chunked->graphicsView();
    QWidget* viewport = view->viewport();
    const QPointF local(view->mapFromScene(scenePos));
    QWheelEvent event(local, viewport->mapToGlobal(local), QPoint(), QPoint(0, deltaY),
                      Qt::NoButton, modifiers, Qt::NoScrollPhase, false,
                      Qt::MouseEventSynthesizedByApplication);
    const QPointer<CChunkedVolumeViewer> alive(chunked);
    const QString viewerId = viewerIdFor(viewer);
    QCoreApplication::sendEvent(viewport, &event);

    QJsonValue volumePointJson = QJsonValue::Null;
    if (alive) {
        if (const auto sample = chunked->sampleSceneVolume(scenePos))
            volumePointJson = vec3ToJson(sample->position);
    }

    QJsonObject scene;
    scene["x"] = scenePos.x();
    scene["y"] = scenePos.y();

    QJsonObject result;
    result["viewerId"] = viewerId;
    result["scene"] = scene;
    result["volumePoint"] = volumePointJson;
    result["deltaY"] = deltaY;
    result["notches"] = static_cast<double>(deltaY) / kWheelNotch;
    result["modifiers"] = modifiersToJson(modifiers);
    result["accepted"] = event.isAccepted();
    return result;
}
