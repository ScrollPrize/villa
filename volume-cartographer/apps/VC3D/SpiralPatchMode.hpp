#pragma once

#include <QEvent>
#include <QKeyEvent>

// A modifier becomes a toggle only when released without having been used.
// Observe chords before dispatching them to the viewer or another tool.
class SpiralPatchMode
{
public:
    bool active() const { return _active; }
    void deactivate() { _active = false; _tap = false; }

    bool observe(const QEvent& event)
    {
        if (event.type() == QEvent::WindowDeactivate
            || event.type() == QEvent::FocusOut || event.type() == QEvent::Leave) {
            _tap = false;
            _pointerGesture = false;
        } else if (event.type() == QEvent::MouseButtonRelease) {
            _pointerGesture = false;
            _tap = false;
        } else if (event.type() == QEvent::MouseButtonPress
                   || event.type() == QEvent::MouseButtonDblClick
                   || event.type() == QEvent::Wheel) {
            _tap = false;
            if (event.type() != QEvent::Wheel) _pointerGesture = true;
        } else if (event.type() == QEvent::KeyPress
                   || event.type() == QEvent::KeyRelease) {
            const auto& key = static_cast<const QKeyEvent&>(event);
            if (key.key() != Qt::Key_Control) {
                _tap = false;
                if (key.key() == Qt::Key_Escape && _active) {
                    deactivate();
                    return true;
                }
            } else if (!key.isAutoRepeat()) {
                if (event.type() == QEvent::KeyPress) {
                    _tap = !_pointerGesture
                        && (key.modifiers() & ~Qt::ControlModifier) == Qt::NoModifier;
                } else {
                    const bool toggle = _tap;
                    _tap = false;
                    if (toggle) {
                        _active = !_active;
                        return true;
                    }
                }
            }
        }
        return false;
    }

private:
    bool _active = false;
    bool _tap = false;
    bool _pointerGesture = false;
};
