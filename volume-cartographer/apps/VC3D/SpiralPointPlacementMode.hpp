#pragma once

#include "SpiralPclRole.hpp"

#include <QEvent>
#include <QKeyEvent>

#include <optional>
#include <utility>

// Internal state machine for Spiral's persistent point placement. One role is
// active at a time: Q places same-winding points, E places relative-winding
// points. It owns only the mode lifetime; the brush controller owns collection
// data and responds to interaction transitions by updating the active
// collection.
class SpiralPointPlacementMode
{
public:
    enum class Transition {
        None,
        Activated,
        // The other role's key was pressed while a role was active: the
        // controller closes the current collection and continues in the new
        // role.
        SwitchRole,
        ClearInteraction,
        ClearInteractionPreserveDraft,
        ReverseActive,
        DeleteActive,
    };

    struct EventResult {
        bool handled = false;
        Transition transition = Transition::None;
    };

    static std::optional<vc3d::spiral::PclRole> roleForKey(int key)
    {
        for (const vc3d::spiral::PclRole role : vc3d::spiral::kEditablePclRoles) {
            if (vc3d::spiral::pclRoleToggleKey(role) == key) return role;
        }
        return std::nullopt;
    }

    EventResult handleEvent(const QEvent& event, bool hasActivePcl = false)
    {
        if (event.type() != QEvent::KeyPress && event.type() != QEvent::KeyRelease)
            return {};
        const auto& key = static_cast<const QKeyEvent&>(event);

        if (const auto role = roleForKey(key.key())) {
            // A toggle key belongs to this mode even when it does not cause a
            // transition, so releases and autorepeat presses cannot leak
            // elsewhere.
            if (event.type() == QEvent::KeyRelease || key.isAutoRepeat())
                return {true, Transition::None};
            if (!_activeRole) {
                _activeRole = role;
                return {true, Transition::Activated};
            }
            if (*_activeRole != *role) {
                _activeRole = role;
                return {true, Transition::SwitchRole};
            }
            return {true, Transition::None};
        }

        if (key.key() == Qt::Key_Escape)
            return handleActiveKey(event, hasActivePcl || active(), _escapeDown,
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
        if (!_activeRole && !hasActivePcl) return Transition::None;
        _activeRole.reset();
        return Transition::ClearInteractionPreserveDraft;
    }

    bool deactivate()
    {
        return std::exchange(_activeRole, std::nullopt).has_value();
    }

    bool active() const { return _activeRole.has_value(); }
    std::optional<vc3d::spiral::PclRole> activeRole() const { return _activeRole; }

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
        if (deactivateOnPress) _activeRole.reset();
        return {true, transition};
    }

    std::optional<vc3d::spiral::PclRole> _activeRole;
    bool _escapeDown = false;
    bool _reverseDown = false;
    bool _deleteDown = false;
};
