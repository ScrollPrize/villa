#pragma once

#include <QEvent>
#include <QKeyEvent>

#include <utility>

// Internal state machine for Spiral's persistent same-winding point placement.
// It owns only the mode lifetime; the brush controller owns collection data and
// responds to interaction transitions by updating the active collection.
class SpiralPointPlacementMode
{
public:
    enum class Transition {
        None,
        Activated,
        ClearInteraction,
        ClearInteractionPreserveDraft,
        ReverseActive,
        DeleteActive,
    };

    struct EventResult {
        bool handled = false;
        Transition transition = Transition::None;
    };

    EventResult handleEvent(const QEvent& event, bool hasActivePcl = false)
    {
        if (event.type() != QEvent::KeyPress && event.type() != QEvent::KeyRelease)
            return {};
        const auto& key = static_cast<const QKeyEvent&>(event);

        if (key.key() == Qt::Key_Q) {
            // Q belongs to this mode even when it does not cause a transition,
            // so releases and autorepeat presses cannot leak elsewhere.
            if (event.type() == QEvent::KeyRelease || key.isAutoRepeat())
                return {true, Transition::None};
            if (!_active) {
                _active = true;
                return {true, Transition::Activated};
            }
            return {true, Transition::None};
        }

        if (key.key() == Qt::Key_Escape)
            return handleActiveKey(event, hasActivePcl || _active, _escapeDown,
                                   Transition::ClearInteraction, true);
        if (key.key() == Qt::Key_F)
            return handleActiveKey(event, hasActivePcl, _reverseDown,
                                   Transition::ReverseActive, false);
        if (key.key() == Qt::Key_Delete)
            return handleActiveKey(event, hasActivePcl, _deleteDown,
                                   Transition::DeleteActive, false);
        return {};
    }

    Transition surfaceChanged(bool hasActivePcl = false)
    {
        if (!_active && !hasActivePcl) return Transition::None;
        _active = false;
        return Transition::ClearInteractionPreserveDraft;
    }

    bool deactivate()
    {
        return std::exchange(_active, false);
    }

    bool active() const { return _active; }

private:
    EventResult handleActiveKey(const QEvent& event, bool relevant,
                                bool& keyDown, Transition transition,
                                bool deactivateOnPress)
    {
        if (event.type() == QEvent::KeyRelease) {
            if (!std::exchange(keyDown, false)) return {};
            return {true, Transition::None};
        }
        if (!relevant && !keyDown) return {};
        if (keyDown || static_cast<const QKeyEvent&>(event).isAutoRepeat())
            return {true, Transition::None};
        keyDown = true;
        if (deactivateOnPress) _active = false;
        return {true, transition};
    }

    bool _active = false;
    bool _escapeDown = false;
    bool _reverseDown = false;
    bool _deleteDown = false;
};
